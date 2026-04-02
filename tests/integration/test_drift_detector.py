
from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from src.data_logger.models import Base, PredictionLog
import src.drift_detector.detector



REFERENCE_FEATURES = {
    "tenure": 24,
    "MonthlyCharges": 50.0,
    "TotalCharges": 1200.0,
}

DRIFT_WINDOW_SIZE = 10


def _make_ref_df(n=50, shift: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "tenure": rng.integers(1, 72, size=n).tolist(),
        "MonthlyCharges": (rng.uniform(20, 100, size=n) * shift).tolist(),
        "TotalCharges": (rng.uniform(100, 8000, size=n) * shift).tolist(),
    })


def _seed_logs(session: Session, n: int, features: dict | None = None) -> list[str]:
    ids = []
    base_features = features or {
        "tenure": 24,
        "MonthlyCharges": 50.0,
        "TotalCharges": 1200.0,
    }
    for _ in range(n):
        log = PredictionLog(
            id=uuid.uuid4(),
            request_id=str(uuid.uuid4()),
            features=base_features,
            prediction=0,
            probability=0.3,
            model_version="1",
            drift_analyzed=False,
        )
        session.add(log)
        ids.append(str(log.id))
    session.commit()
    return ids


@pytest.fixture()
def in_memory_engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine



def _count_analyzed(session: Session) -> int:
    return session.query(PredictionLog).filter(
        PredictionLog.drift_analyzed == True
    ).count()


def _count_unanalyzed(session: Session) -> int:
    return session.query(PredictionLog).filter(
        PredictionLog.drift_analyzed == False
    ).count()



class TestDriftDetectorNoData:

    def test_no_logs_does_not_trigger_airflow(self, in_memory_engine):
        ref_df = _make_ref_df(50)

        with patch("src.drift_detector.detector.db_session") as mock_db, \
             patch("src.drift_detector.detector.pd.read_parquet", return_value=ref_df), \
             patch("src.drift_detector.detector.Path.exists", return_value=True), \
             patch("src.drift_detector.detector.trigger_retraining_dag") as mock_trigger, \
             patch("src.drift_detector.detector.load_drift_config") as mock_cfg:

            mock_cfg.return_value = {
                "detection": {
                    "window_size": DRIFT_WINDOW_SIZE,
                    "drift_share_threshold": 0.3,
                },
                "reference": {"data_path": "fake/path.parquet"},
                "evidently": {"report_output_dir": "data/drift_reports"},
                "alerting": {"trigger_airflow": True},
            }

            mock_session = MagicMock()
            mock_session.scalars.return_value.all.return_value = []
            mock_db.return_value.__enter__ = lambda s, *a: mock_session
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from src.drift_detector.detector import run_drift_detection
            run_drift_detection(config_path="fake/drift.yaml")

            mock_trigger.assert_not_called()


class TestDriftDetectorNoDrift:

    def test_matching_data_does_not_alert(self, in_memory_engine):
        ref_df = _make_ref_df(50, shift=1.0)
        curr_logs = [
            MagicMock(
                drift_analyzed=False,
                features={"tenure": int(ref_df["tenure"].iloc[i]),
                          "MonthlyCharges": float(ref_df["MonthlyCharges"].iloc[i]),
                          "TotalCharges": float(ref_df["TotalCharges"].iloc[i])},
                id=uuid.uuid4(),
                created_at=None,
            )
            for i in range(min(DRIFT_WINDOW_SIZE, len(ref_df)))
        ]

        with patch("src.drift_detector.detector.db_session") as mock_db, \
             patch("src.drift_detector.detector.pd.read_parquet", return_value=ref_df), \
             patch("src.drift_detector.detector.Path.exists", return_value=True), \
             patch("src.drift_detector.detector.trigger_retraining_dag") as mock_trigger, \
             patch("src.drift_detector.detector.load_drift_config") as mock_cfg:

            mock_cfg.return_value = {
                "detection": {
                    "window_size": DRIFT_WINDOW_SIZE,
                    "drift_share_threshold": 0.3,
                },
                "reference": {"data_path": "fake/path.parquet"},
                "evidently": {"report_output_dir": "data/drift_reports"},
                "alerting": {"trigger_airflow": True},
            }

            mock_session = MagicMock()
            mock_session.scalars.return_value.all.return_value = curr_logs
            mock_db.return_value.__enter__ = lambda s, *a: mock_session
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from src.drift_detector.detector import run_drift_detection
            run_drift_detection(config_path="fake/drift.yaml")

            mock_trigger.assert_not_called()

    def test_all_logs_marked_analyzed_after_run(self, in_memory_engine):
        ref_df = _make_ref_df(50)
        curr_logs = [
            MagicMock(
                drift_analyzed=False,
                features={"tenure": 24, "MonthlyCharges": 50.0, "TotalCharges": 1200.0},
                id=uuid.uuid4(),
                created_at=None,
            )
            for _ in range(DRIFT_WINDOW_SIZE)
        ]
        analyzed_flags = [False] * len(curr_logs)

        for i, log in enumerate(curr_logs):
            idx = i

            def make_setter(j):
                def setter(val):
                    analyzed_flags[j] = val
                return setter

            type(log).__setattr__ = lambda self, name, val, _i=i: (
                analyzed_flags.__setitem__(_i, val) if name == "drift_analyzed" else None
            )

        with patch("src.drift_detector.detector.db_session") as mock_db, \
             patch("src.drift_detector.detector.pd.read_parquet", return_value=ref_df), \
             patch("src.drift_detector.detector.Path.exists", return_value=True), \
             patch("src.drift_detector.detector.trigger_retraining_dag"), \
             patch("src.drift_detector.detector.load_drift_config") as mock_cfg:

            mock_cfg.return_value = {
                "detection": {
                    "window_size": DRIFT_WINDOW_SIZE,
                    "drift_share_threshold": 0.3,
                },
                "reference": {"data_path": "fake/path.parquet"},
                "evidently": {"report_output_dir": "data/drift_reports"},
                "alerting": {"trigger_airflow": True},
            }

            mock_session = MagicMock()
            mock_session.scalars.return_value.all.return_value = curr_logs
            mock_db.return_value.__enter__ = lambda s, *a: mock_session
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from src.drift_detector import detector as det_module
            import importlib
            importlib.reload(det_module)
            det_module.run_drift_detection(config_path="fake/drift.yaml")

            for log in curr_logs:
                assert log.drift_analyzed is True


class TestDriftDetectorSevereDrift:

    def test_severe_drift_triggers_airflow(self, in_memory_engine):
        rng = np.random.default_rng(0)
        ref_df = pd.DataFrame({
            "tenure": rng.integers(1, 72, size=50).tolist(),
            "MonthlyCharges": rng.uniform(20, 100, size=50).tolist(),
            "TotalCharges": rng.uniform(100, 8000, size=50).tolist(),
        })

        drifted_features = {
            "tenure": 1,
            "MonthlyCharges": 9999.0,
            "TotalCharges": 99999.0,
        }
        curr_logs = [
            MagicMock(
                drift_analyzed=False,
                features=drifted_features,
                id=uuid.uuid4(),
                created_at=None,
            )
            for _ in range(DRIFT_WINDOW_SIZE)
        ]

        captured_payload = {}

        def capture_trigger(payload):
            captured_payload.update(payload)
            return "dag_run_test_001"

        with patch("src.drift_detector.detector.db_session") as mock_db, \
             patch("src.drift_detector.detector.pd.read_parquet", return_value=ref_df), \
             patch("src.drift_detector.detector.Path.exists", return_value=True), \
             patch("src.drift_detector.detector.trigger_retraining_dag",
                   side_effect=capture_trigger) as mock_trigger, \
             patch("src.drift_detector.detector.load_drift_config") as mock_cfg, \
             patch("evidently.report.Report.save_html"):

            mock_cfg.return_value = {
                "detection": {
                    "window_size": DRIFT_WINDOW_SIZE,
                    "drift_share_threshold": 0.01,
                },
                "reference": {"data_path": "fake/path.parquet"},
                "evidently": {"report_output_dir": "/tmp/drift_reports"},
                "alerting": {"trigger_airflow": True},
            }

            mock_session = MagicMock()
            mock_session.scalars.return_value.all.return_value = curr_logs
            mock_db.return_value.__enter__ = lambda s, *a: mock_session
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from src.drift_detector.detector import run_drift_detection
            run_drift_detection(config_path="fake/drift.yaml")

            mock_trigger.assert_called_once()
            assert "drift_share" in captured_payload
            assert captured_payload["drift_share"] > 0.0

    def test_airflow_trigger_disabled_does_not_call(self):
        rng = np.random.default_rng(1)
        ref_df = pd.DataFrame({
            "tenure": rng.integers(1, 72, size=50).tolist(),
            "MonthlyCharges": rng.uniform(20, 100, size=50).tolist(),
            "TotalCharges": rng.uniform(100, 8000, size=50).tolist(),
        })

        curr_logs = [
            MagicMock(
                drift_analyzed=False,
                features={"tenure": 1, "MonthlyCharges": 9999.0, "TotalCharges": 99999.0},
                id=uuid.uuid4(),
                created_at=None,
            )
            for _ in range(DRIFT_WINDOW_SIZE)
        ]

        with patch("src.drift_detector.detector.db_session") as mock_db, \
             patch("src.drift_detector.detector.pd.read_parquet", return_value=ref_df), \
             patch("src.drift_detector.detector.Path.exists", return_value=True), \
             patch("src.drift_detector.detector.trigger_retraining_dag") as mock_trigger, \
             patch("src.drift_detector.detector.load_drift_config") as mock_cfg, \
             patch("evidently.report.Report.save_html"):

            mock_cfg.return_value = {
                "detection": {
                    "window_size": DRIFT_WINDOW_SIZE,
                    "drift_share_threshold": 0.01,
                },
                "reference": {"data_path": "fake/path.parquet"},
                "evidently": {"report_output_dir": "/tmp/drift_reports"},
                "alerting": {"trigger_airflow": False},
            }

            mock_session = MagicMock()
            mock_session.scalars.return_value.all.return_value = curr_logs
            mock_db.return_value.__enter__ = lambda s, *a: mock_session
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            from src.drift_detector.detector import run_drift_detection
            run_drift_detection(config_path="fake/drift.yaml")

            mock_trigger.assert_not_called()