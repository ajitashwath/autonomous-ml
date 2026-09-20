from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ModelLoadError, ModelNotFoundError
from src.inference_api import model_loader as ml
from src.inference_api.metrics import MODEL_INFO
from src.inference_api.model_loader import ModelLoader, ModelSnapshot
from src.model_registry.registry import ModelInfo


def _info(version: str, stage: str = "Production", run_id: str | None = None) -> ModelInfo:
    return ModelInfo(
        name="m", version=version, stage=stage, run_id=run_id or f"run-{version}",
        run_link="", metrics={"roc_auc": 0.8}, params={},
    )


def _snapshot(version: str) -> ModelSnapshot:
    # The model/preprocessor are tagged with the version so tests can detect a mixed pair.
    model, preprocessor = MagicMock(tag=version), MagicMock(tag=version)
    return ModelSnapshot(model=model, preprocessor=preprocessor, info=_info(version))


@pytest.fixture
def loader():
    registry = MagicMock()
    registry.get_version_uri.side_effect = lambda v: f"models:/m/{v}"
    settings = MagicMock(inference_model_stage="Production")
    with patch.object(ml, "ModelRegistry", return_value=registry), \
         patch.object(ml, "get_settings", return_value=settings):
        instance = ModelLoader()
    instance.registry = registry  # convenience handle for tests
    return instance


def _live_versions() -> set[str]:
    return {
        s.labels["model_version"]
        for metric in MODEL_INFO.collect()
        for s in metric.samples
        if s.value == 1
    }


class TestSnapshot:
    def test_threshold_comes_from_the_run_params(self):
        info = _info("1")
        info.params["threshold"] = "0.35"
        assert ModelSnapshot(MagicMock(), MagicMock(), info).threshold == 0.35

    @pytest.mark.parametrize("params", [{}, {"threshold": "not-a-number"}])
    def test_threshold_defaults_to_half(self, params):
        info = _info("1")
        info.params.update(params)
        assert ModelSnapshot(MagicMock(), MagicMock(), info).threshold == 0.5

    def test_snapshot_is_immutable(self):
        snap = _snapshot("1")
        with pytest.raises(Exception):
            snap.model = MagicMock()


class TestLoad:
    def test_not_loaded_until_a_model_is_published(self, loader):
        assert not loader.is_loaded()
        assert loader.current() is None
        assert loader.get_info() is None
        with pytest.raises(ModelLoadError):
            loader.get_snapshot()

    def test_load_publishes_a_snapshot(self, loader):
        loader.registry.get_model_info.return_value = _info("4")
        with patch.object(loader, "_build_snapshot", return_value=_snapshot("4")):
            loader.load()

        assert loader.is_loaded()
        assert loader.get_snapshot().info.version == "4"

    def test_load_with_no_production_model_raises(self, loader):
        loader.registry.get_model_info.side_effect = ModelNotFoundError("none")
        with pytest.raises(ModelLoadError):
            loader.load()
        assert not loader.is_loaded()

    def test_build_loads_the_exact_version_not_the_stage(self, loader):
        """Regression: loading by stage after reading the version separately let a promotion
        in between leave `info` describing a different model than the one loaded."""
        with patch.object(ml.mlflow.xgboost, "load_model", return_value=MagicMock()) as load_model, \
             patch.object(loader, "_load_preprocessor", return_value=MagicMock()):
            loader._build_snapshot(_info("7"))

        load_model.assert_called_once_with("models:/m/7")

    def test_build_refuses_to_serve_without_the_preprocessor(self, loader):
        """Regression: a missing preprocessor used to degrade to raw features and wrong output."""
        with patch.object(ml.mlflow.xgboost, "load_model", return_value=MagicMock()), \
             patch.object(ml.mlflow.artifacts, "download_artifacts", side_effect=OSError("no artifact")):
            with pytest.raises(ModelLoadError) as exc:
                loader._build_snapshot(_info("7"))

        assert "Preprocessor" in exc.value.message
        assert exc.value.details["version"] == "7"


class TestHotSwap:
    def _running(self, loader, version="1"):
        loader._publish(_snapshot(version))

    def test_new_version_replaces_the_whole_snapshot(self, loader):
        self._running(loader, "1")
        loader.registry.get_model_info.return_value = _info("2")
        with patch.object(loader, "_build_snapshot", return_value=_snapshot("2")):
            loader._try_hot_swap()

        snap = loader.get_snapshot()
        assert snap.info.version == snap.model.tag == snap.preprocessor.tag == "2"

    def test_same_version_is_not_rebuilt(self, loader):
        self._running(loader, "1")
        loader.registry.get_model_info.return_value = _info("1")
        with patch.object(loader, "_build_snapshot") as build:
            loader._try_hot_swap()

        build.assert_not_called()

    def test_failed_load_keeps_the_serving_model(self, loader):
        self._running(loader, "1")
        loader.registry.get_model_info.return_value = _info("2")
        with patch.object(loader, "_build_snapshot", side_effect=ModelLoadError("corrupt")):
            loader._try_hot_swap()

        assert loader.get_snapshot().info.version == "1"

    def test_registry_outage_keeps_the_serving_model(self, loader):
        self._running(loader, "1")
        loader.registry.get_model_info.side_effect = RuntimeError("mlflow down")

        loader._try_hot_swap()

        assert loader.get_snapshot().info.version == "1"

    def test_first_model_is_picked_up_after_a_failed_startup(self, loader):
        loader.registry.get_model_info.return_value = _info("1")
        with patch.object(loader, "_build_snapshot", return_value=_snapshot("1")):
            loader._try_hot_swap()

        assert loader.get_snapshot().info.version == "1"

    def test_only_the_live_version_is_reported_in_metrics(self, loader):
        """Regression: old versions stayed at 1, so dashboards showed several 'active' models."""
        self._running(loader, "1")
        loader.registry.get_model_info.return_value = _info("2")
        with patch.object(loader, "_build_snapshot", return_value=_snapshot("2")):
            loader._try_hot_swap()

        assert _live_versions() == {"2"}


class TestAtomicity:
    def test_readers_never_see_a_mixed_model_and_preprocessor(self, loader):
        """Readers hammer get_snapshot() while versions are swapped continuously."""
        snapshots = {v: _snapshot(v) for v in ("a", "b")}
        loader._publish(snapshots["a"])
        stop = threading.Event()
        mismatches: list[tuple[str, str, str]] = []

        def reader():
            while not stop.is_set():
                s = loader.get_snapshot()
                if not (s.model.tag == s.preprocessor.tag == s.info.version):
                    mismatches.append((s.model.tag, s.preprocessor.tag, s.info.version))

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for i in range(2000):
            loader._publish(snapshots["ab"[i % 2]])
        stop.set()
        for t in threads:
            t.join()

        assert mismatches == []


class TestPolling:
    def test_poller_survives_an_unexpected_error(self, loader, monkeypatch):
        """An unhandled exception used to kill the thread and silently end hot-swapping."""
        monkeypatch.setattr(ml, "POLL_INTERVAL_SECONDS", 0.01)
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            if len(calls) >= 3:
                loader._stop_event.set()

        with patch.object(loader, "_try_hot_swap", side_effect=flaky):
            loader.start_polling()
            loader._thread.join(timeout=5)

        assert len(calls) >= 3

    def test_start_polling_twice_does_not_spawn_a_second_thread(self, loader, monkeypatch):
        monkeypatch.setattr(ml, "POLL_INTERVAL_SECONDS", 60)
        loader.start_polling()
        first = loader._thread
        loader.start_polling()
        try:
            assert loader._thread is first
        finally:
            loader.stop_polling()
