"""Drift detector against a real (in-memory SQLite) prediction_logs table and a real
Evidently run. Only Airflow/Slack are mocked; report output goes to tmp_path."""
from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.data_logger.models import Base, PredictionLog
from src.drift_detector import detector

THRESHOLD = 0.3
MIN_SAMPLES = 50
N_REF = 500


def _ref_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "tenure": rng.integers(1, 72, size=N_REF),
        "MonthlyCharges": rng.uniform(20, 100, size=N_REF),
        "TotalCharges": rng.uniform(100, 8000, size=N_REF),
    })


@pytest.fixture()
def session_factory(monkeypatch):
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def fake_db_session():
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(detector, "db_session", fake_db_session)
    return factory


@pytest.fixture()
def env(tmp_path, session_factory):
    ref = _ref_df()
    ref_path = tmp_path / "reference.parquet"
    ref.to_parquet(ref_path, index=False)
    out_dir = tmp_path / "reports"

    config = {
        "detection": {
            "window_size": 1000,
            "drift_share_threshold": THRESHOLD,
            "pvalue_threshold": 0.05,
            "min_samples": MIN_SAMPLES,
            "retrain_cooldown_seconds": 3600,
        },
        "reference": {"data_path": str(ref_path)},
        "evidently": {"report_output_dir": str(out_dir)},
        "alerting": {"trigger_airflow": True},
    }

    class Env:
        pass

    e = Env()
    e.ref, e.out_dir, e.config, e.session_factory = ref, out_dir, config, session_factory
    e.run = lambda: _run(config)
    return e


def _run(config):
    with patch.object(detector, "load_drift_config", return_value=config):
        return detector.run_drift_detection(config_path="ignored.yaml")


def _seed(factory, rows: list[dict]) -> None:
    with factory() as session:
        for features in rows:
            session.add(PredictionLog(
                id=uuid.uuid4(), request_id=str(uuid.uuid4()), features=features,
                prediction=0, probability=0.3, model_version="1", drift_analyzed=False,
            ))
        session.commit()


def _matching_rows(ref: pd.DataFrame, n: int) -> list[dict]:
    return ref.sample(n, random_state=7).to_dict(orient="records")


def _drifted_rows(n: int) -> list[dict]:
    return [
        {"tenure": 1, "MonthlyCharges": 9999.0 + i, "TotalCharges": 99999.0 + i}
        for i in range(n)
    ]


def _count(factory, analyzed: bool) -> int:
    with factory() as session:
        return session.query(PredictionLog).filter(
            PredictionLog.drift_analyzed.is_(analyzed)
        ).count()


@pytest.fixture()
def trigger():
    with patch.object(detector, "trigger_retraining_dag", return_value="run-1") as mock, \
         patch.object(detector, "send_slack_alert"):
        yield mock


class TestNoDrift:
    def test_matching_data_does_not_trigger_and_reports_observed_share(self, env, trigger):
        """Regression: `drift_share` from Evidently is its threshold (0.5), which always
        exceeded 0.3 — so retraining fired on every run even with zero drifted columns."""
        _seed(env.session_factory, _matching_rows(env.ref, 100))

        env.run()

        trigger.assert_not_called()
        metrics = json.loads((env.out_dir / "latest_drift_metrics.json").read_text())
        assert metrics["drifted_features"] == 0
        assert metrics["drift_share"] == 0.0
        assert metrics["drift_share"] < THRESHOLD

    def test_analyzed_rows_are_marked_after_a_successful_run(self, env, trigger):
        _seed(env.session_factory, _matching_rows(env.ref, 100))

        env.run()

        assert _count(env.session_factory, analyzed=True) == 100
        assert _count(env.session_factory, analyzed=False) == 0

    def test_no_logs_is_a_noop(self, env, trigger):
        env.run()

        trigger.assert_not_called()
        assert not (env.out_dir / "latest_drift_metrics.json").exists()


class TestMinimumSampleSize:
    def test_small_window_is_not_analyzed_and_not_consumed(self, env, trigger):
        _seed(env.session_factory, _drifted_rows(MIN_SAMPLES - 1))

        env.run()

        trigger.assert_not_called()
        assert not (env.out_dir / "latest_drift_metrics.json").exists()
        assert _count(env.session_factory, analyzed=False) == MIN_SAMPLES - 1

    def test_rows_accumulate_until_the_window_is_large_enough(self, env, trigger):
        _seed(env.session_factory, _drifted_rows(MIN_SAMPLES - 1))
        env.run()
        trigger.assert_not_called()

        _seed(env.session_factory, _drifted_rows(1))
        env.run()

        trigger.assert_called_once()
        assert _count(env.session_factory, analyzed=True) == MIN_SAMPLES


class TestDriftDetected:
    def test_severe_drift_triggers_retraining_once_with_observed_metrics(self, env, trigger):
        _seed(env.session_factory, _drifted_rows(100))

        env.run()

        trigger.assert_called_once()
        payload = trigger.call_args.args[0]
        assert payload["drift_share"] == pytest.approx(1.0)
        assert payload["drift_share"] >= payload["drift_share_threshold"]
        assert (env.out_dir / "latest_drift_report.html").exists()
        assert _count(env.session_factory, analyzed=True) == 100

    def test_trigger_disabled_never_calls_airflow(self, env, trigger):
        env.config["alerting"]["trigger_airflow"] = False
        _seed(env.session_factory, _drifted_rows(100))

        env.run()

        trigger.assert_not_called()


class TestCooldown:
    def test_persistent_drift_does_not_retrigger_within_cooldown(self, env, trigger):
        with patch.object(detector.time, "time", return_value=1_000_000.0):
            _seed(env.session_factory, _drifted_rows(100))
            env.run()
        with patch.object(detector.time, "time", return_value=1_000_000.0 + 300):
            _seed(env.session_factory, _drifted_rows(100))
            env.run()

        assert trigger.call_count == 1

    def test_retriggers_after_cooldown_expires(self, env, trigger):
        with patch.object(detector.time, "time", return_value=1_000_000.0):
            _seed(env.session_factory, _drifted_rows(100))
            env.run()
        with patch.object(detector.time, "time", return_value=1_000_000.0 + 3601):
            _seed(env.session_factory, _drifted_rows(100))
            env.run()

        assert trigger.call_count == 2

    def test_zero_cooldown_disables_suppression(self, env, trigger):
        env.config["detection"]["retrain_cooldown_seconds"] = 0
        for _ in range(2):
            _seed(env.session_factory, _drifted_rows(100))
            env.run()

        assert trigger.call_count == 2


class TestFailureHandling:
    def test_trigger_failure_is_contained_and_does_not_start_cooldown(self, env):
        with patch.object(detector, "send_slack_alert"), \
             patch.object(detector, "trigger_retraining_dag", side_effect=RuntimeError("airflow down")):
            _seed(env.session_factory, _drifted_rows(100))
            env.run()  # must not raise

        assert not (env.out_dir / detector.TRIGGER_STATE_FILE).exists()

        with patch.object(detector, "send_slack_alert"), \
             patch.object(detector, "trigger_retraining_dag", return_value="run-2") as retry:
            _seed(env.session_factory, _drifted_rows(100))
            env.run()

        retry.assert_called_once()

    def test_metric_extraction_failure_leaves_rows_unanalyzed(self, env, trigger):
        _seed(env.session_factory, _matching_rows(env.ref, 100))

        with patch.object(detector, "extract_drift_summary", side_effect=KeyError("boom")):
            env.run()

        assert _count(env.session_factory, analyzed=False) == 100
        trigger.assert_not_called()

    def test_evidently_crash_does_not_consume_the_window(self, env, trigger):
        """Regression: rows used to be marked analyzed before analysis ran."""
        _seed(env.session_factory, _matching_rows(env.ref, 100))

        with patch.object(detector.Report, "run", side_effect=RuntimeError("evidently crashed")):
            with pytest.raises(RuntimeError):
                env.run()

        assert _count(env.session_factory, analyzed=False) == 100
        assert _count(env.session_factory, analyzed=True) == 0


class TestOutcomes:
    """run_drift_detection reports what it did, so a one-shot run can exit non-zero when the
    detector cannot do its job (a scheduler would otherwise show it green forever)."""

    def test_no_logs(self, env, trigger):
        assert env.run() == detector.Outcome.NO_NEW_LOGS

    def test_insufficient_samples(self, env, trigger):
        _seed(env.session_factory, _drifted_rows(MIN_SAMPLES - 1))
        assert env.run() == detector.Outcome.INSUFFICIENT_SAMPLES

    def test_no_drift(self, env, trigger):
        _seed(env.session_factory, _matching_rows(env.ref, 100))
        assert env.run() == detector.Outcome.NO_DRIFT

    def test_drift_detected(self, env, trigger):
        _seed(env.session_factory, _drifted_rows(100))
        assert env.run() == detector.Outcome.DRIFT_DETECTED

    def test_drift_still_reported_while_the_trigger_is_in_cooldown(self, env, trigger):
        with patch.object(detector.time, "time", return_value=1_000_000.0):
            _seed(env.session_factory, _drifted_rows(100))
            env.run()
            _seed(env.session_factory, _drifted_rows(100))
            assert env.run() == detector.Outcome.DRIFT_DETECTED

    def test_missing_reference_data_is_a_failure_outcome(self, env, trigger):
        env.config["reference"]["data_path"] = str(env.out_dir / "missing.parquet")
        outcome = env.run()
        assert outcome == detector.Outcome.NO_REFERENCE
        assert outcome in detector.FAILURE_OUTCOMES

    def test_unreachable_airflow_is_a_failure_outcome(self, env):
        with patch.object(detector, "send_slack_alert"),              patch.object(detector, "trigger_retraining_dag", side_effect=RuntimeError("airflow down")):
            _seed(env.session_factory, _drifted_rows(100))
            outcome = env.run()
        assert outcome == detector.Outcome.TRIGGER_FAILED
        assert outcome in detector.FAILURE_OUTCOMES
