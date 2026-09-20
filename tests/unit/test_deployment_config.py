"""Static checks on the container/DAG wiring.

These cannot replace building the images, but they pin down specific mistakes that
previously left the stack unable to start or the retrain loop unable to run.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]
DAG_SRC = (ROOT / "src/retraining_pipeline/dags/retrain_dag.py").read_text()


def _pins(text: str) -> dict[str, str]:
    pins = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([\w.]+)", line.strip())
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def _dockerfile_for(service: dict) -> Path:
    build = service["build"]
    context = ROOT / (build if isinstance(build, str) else build.get("context", "."))
    dockerfile = "Dockerfile" if isinstance(build, str) else build.get("dockerfile", "Dockerfile")
    return context / dockerfile


def _mounts(service: dict) -> list[str]:
    return [v.split(":")[1] for v in service.get("volumes", []) if ":" in v]


def test_compose_depends_on_only_references_defined_services():
    for name, svc in SERVICES.items():
        for dep in svc.get("depends_on", {}):
            assert dep in SERVICES, f"{name} depends on undefined service {dep}"


def test_every_built_service_has_its_dockerfile():
    for name, svc in SERVICES.items():
        if "build" in svc:
            assert _dockerfile_for(svc).exists(), f"{name}: missing {_dockerfile_for(svc)}"


def test_curl_healthchecks_are_backed_by_curl_in_built_images():
    """python:*-slim has no curl; a curl healthcheck then never passes and every service
    that waits on it (service_healthy) never starts."""
    for name, svc in SERVICES.items():
        health = " ".join(map(str, svc.get("healthcheck", {}).get("test", [])))
        if "curl" not in health:
            continue
        if "build" in svc:
            assert "curl" in _dockerfile_for(svc).read_text(), f"{name}: healthcheck uses curl"
        else:
            image = svc.get("image", "")
            assert not image.startswith("python:"), f"{name}: {image} has no curl"


def test_services_do_not_use_an_unshared_local_mlflow_artifact_path():
    """A local artifact root (/mlruns) is not shared between containers and is not writable
    by the non-root API user, so models logged elsewhere were invisible or unwritable."""
    for name, svc in SERVICES.items():
        assert "/mlruns" not in _mounts(svc), f"{name} mounts /mlruns"


def test_mlflow_proxies_artifacts_over_http():
    cmd = " ".join(SERVICES["mlflow"]["command"])
    assert "--serve-artifacts" in cmd
    assert "mlflow-artifacts:/" in cmd

    k8s = (ROOT / "k8s/mlflow/deployment.yaml").read_text()
    assert "--serve-artifacts" in k8s and "mlflow-artifacts:/" in k8s


def test_mlflow_server_version_matches_the_clients():
    client = _pins((ROOT / "requirements.txt").read_text())["mlflow"]
    assert f"mlflow=={client}" in (ROOT / "mlflow/Dockerfile").read_text()
    assert f"mlflow=={client}" in (ROOT / "k8s/mlflow/deployment.yaml").read_text()


class TestAirflowService:
    airflow = SERVICES["airflow"]

    def test_rest_api_accepts_basic_auth_for_the_drift_detector(self):
        assert "basic_auth" in self.airflow["environment"]["AIRFLOW__API__AUTH_BACKENDS"]

    def test_dags_are_not_paused_on_creation(self):
        paused = self.airflow["environment"]["AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION"]
        assert str(paused).lower() == "false"

    def test_app_code_config_and_data_are_mounted_where_the_dag_runs(self):
        mounts = _mounts(self.airflow)
        for path in ("/app/src", "/app/configs", "/app/data"):
            assert path in mounts

    def test_fernet_key_default_is_not_an_invalid_placeholder(self):
        default = str(self.airflow["environment"]["AIRFLOW__CORE__FERNET_KEY"])
        assert "change_me" not in default
        assert "change_me" not in (ROOT / ".env.example").read_text().split("FERNET_KEY=")[1].splitlines()[0]

    def test_image_installs_what_the_entrypoint_needs(self):
        dockerfile = (ROOT / "airflow/Dockerfile").read_text()
        entrypoint = (ROOT / "airflow/entrypoint.sh").read_text()

        assert entrypoint.startswith("#!"), "no shebang: Docker fails with 'exec format error'"
        if "pg_isready" in entrypoint:
            assert "postgresql-client" in dockerfile
        # COPY as the airflow user creates root-owned files, so the later chmod would fail.
        copy_line = next(line for line in dockerfile.splitlines() if "entrypoint.sh" in line and "COPY" in line)
        assert "--chown=" in copy_line

    def test_ml_dependencies_are_isolated_from_airflows_own_environment(self):
        dockerfile = (ROOT / "airflow/Dockerfile").read_text()
        assert "venv" in dockerfile and "requirements-ml.txt" in dockerfile

    def test_ml_requirements_match_the_repo_pins(self):
        repo = _pins((ROOT / "requirements.txt").read_text())
        ml = _pins((ROOT / "airflow/requirements-ml.txt").read_text())
        assert ml, "no pins parsed"
        for pkg, version in ml.items():
            assert repo.get(pkg) == version, f"{pkg}: ml={version} repo={repo.get(pkg)}"


class TestRetrainDag:
    def test_config_files_the_dag_uses_exist(self):
        configs = re.findall(r"configs/[\w.\-]+\.yaml", DAG_SRC)
        assert configs, "DAG references no config files"
        for rel in configs:
            assert (ROOT / rel).exists(), rel

    def test_tasks_run_from_the_app_root_that_compose_mounts(self):
        app_dir = re.search(r'APP_DIR\s*=\s*"([^"]+)"', DAG_SRC).group(1)
        assert app_dir in {m.rsplit("/", 1)[0] for m in _mounts(SERVICES["airflow"]) if m.startswith("/app/")}
        assert DAG_SRC.count("cwd=APP_DIR") == 2
        assert "/opt/airflow/src" not in DAG_SRC

    def test_tasks_use_the_ml_virtualenv_not_airflows_python(self):
        assert "ML_PYTHON" in DAG_SRC
        assert 'bash_command="python ' not in DAG_SRC

    def test_train_runs_before_the_gate(self):
        assert "train_model >> validate_and_deploy" in DAG_SRC


class TestTrainerService:
    trainer = SERVICES["trainer"]

    def test_is_opt_in_and_can_see_the_dataset_directory(self):
        assert "tools" in self.trainer["profiles"]
        assert "/app/data" in _mounts(self.trainer)

    def test_training_config_paths_live_under_the_mounted_data_dir(self):
        cfg = yaml.safe_load((ROOT / "configs/training.yaml").read_text())["data"]
        for key in ("raw_path", "reference_path", "holdout_path"):
            assert cfg[key].startswith("data/"), key


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "airflow").glob("*.sh")))
def test_shell_scripts_have_a_shebang_and_lf_endings(script):
    raw = (ROOT / "airflow" / script).read_bytes()
    assert raw.startswith(b"#!")
    assert b"\r" not in raw


class TestDriftDetectorCronJob:
    cron = yaml.safe_load((ROOT / "k8s/drift-detector/cronjob.yaml").read_text())
    pod = cron["spec"]["jobTemplate"]["spec"]["template"]["spec"]
    container = pod["containers"][0]

    def test_runs_a_single_pass_instead_of_the_daemon_loop(self):
        """The detector's default mode never returns, so under activeDeadlineSeconds the job
        could only ever end by being killed."""
        assert "--once" in self.container["command"]

    def test_state_that_must_survive_between_runs_is_on_a_persistent_volume(self):
        """Each CronJob run is a new pod. The retrain cooldown and latest metrics live in
        drift_reports, so an emptyDir there would silently reset the cooldown every run."""
        volumes = {v["name"]: v for v in self.pod["volumes"]}
        mounts = {m["mountPath"]: m for m in self.container["volumeMounts"]}

        reports = mounts["/app/data/drift_reports"]
        assert "persistentVolumeClaim" in volumes[reports["name"]]
        assert not reports.get("readOnly", False)
        assert not any("emptyDir" in v for v in volumes.values())

    def test_reference_data_is_read_only_but_the_claim_is_writable(self):
        volumes = {v["name"]: v for v in self.pod["volumes"]}
        mounts = {m["mountPath"]: m for m in self.container["volumeMounts"]}

        assert mounts["/app/data/reference"]["readOnly"] is True
        claim = volumes[mounts["/app/data/reference"]["name"]]["persistentVolumeClaim"]
        assert not claim.get("readOnly", False), "a read-only claim makes every mount read-only"

    def test_non_root_user_can_write_to_the_volume(self):
        ctx = self.pod["securityContext"]
        assert ctx["runAsUser"] == 1001
        assert ctx["fsGroup"] == 1001
        assert "USER appuser" in (ROOT / "src/drift_detector/Dockerfile").read_text()

    def test_interval_leaves_room_for_the_run_to_finish(self):
        assert self.cron["spec"]["concurrencyPolicy"] == "Forbid"
        assert self.cron["spec"]["jobTemplate"]["spec"]["activeDeadlineSeconds"] < 300


def test_filled_in_k8s_secrets_are_git_ignored():
    """k8s/README promises this; without the rule `git add .` would commit real credentials."""
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert "k8s/secret.yaml" in ignored
