from pathlib import Path

import pytest
from tenacity import wait_none

from src.model_registry.registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]

# Two payloads the synthetic model below separates cleanly.
CHURN_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 75.35, "TotalCharges": 904.20,
}
NO_CHURN_PAYLOAD = {**CHURN_PAYLOAD, "tenure": 60, "Contract": "Two year", "TotalCharges": 4500.0}


@pytest.fixture(scope="session")
def churn_model_bundle():
    """A real fitted preprocessor + XGBoost model over the real request schema.

    Churn is a deterministic function (month-to-month contract and tenure < 24), so
    CHURN_PAYLOAD -> 1 and NO_CHURN_PAYLOAD -> 0 without flakiness.
    """
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import yaml

    from src.inference_api.schemas import ChurnFeatures
    from src.training_service.preprocessor import build_preprocessor

    features = yaml.safe_load((ROOT / "configs/training.yaml").read_text())["features"]
    rng = np.random.default_rng(0)
    n = 600
    data = {
        name: rng.choice([e.value for e in ChurnFeatures.model_fields[name].annotation], size=n)
        for name in features["categorical"]
    }
    data["tenure"] = rng.integers(0, 73, size=n)
    data["MonthlyCharges"] = rng.uniform(20, 120, size=n)
    data["TotalCharges"] = data["tenure"] * data["MonthlyCharges"]
    df = pd.DataFrame(data)
    y = ((df["Contract"] == "Month-to-month") & (df["tenure"] < 24)).astype(int).to_numpy()

    preprocessor = build_preprocessor(features["categorical"], features["numerical"])
    model = xgb.XGBClassifier(n_estimators=60, max_depth=3, verbosity=0)
    model.fit(preprocessor.fit_transform(df), y)
    return preprocessor, model


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch):
    """Registry methods retry with exponential backoff; don't actually sleep in tests."""
    for name in (
        "_latest_versions",
        "register_new_version",
        "transition_to_staging",
        "promote_to_production",
    ):
        monkeypatch.setattr(getattr(ModelRegistry, name).retry, "wait", wait_none())


class SqliteDb:
    """In-memory prediction_logs database plus a drop-in replacement for `db_session`."""

    def __init__(self):
        from contextlib import contextmanager

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        from src.data_logger.models import Base

        self.engine = create_engine(
            "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)

        @contextmanager
        def session():
            s = self.factory()
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise
            finally:
                s.close()

        self.session = session

    def add_prediction(self, request_id: str, features: dict, prediction: int = 0,
                       actual_label: int | None = None, created_at=None, version: str = "1"):
        import uuid

        from src.data_logger.models import PredictionLog

        with self.factory() as s:
            row = PredictionLog(
                id=uuid.uuid4(), request_id=request_id, features=features,
                prediction=prediction, probability=0.5, model_version=version,
                actual_label=actual_label,
            )
            if created_at is not None:
                row.created_at = created_at
            s.add(row)
            s.commit()

    def rows(self):
        from src.data_logger.models import PredictionLog

        with self.factory() as s:
            return {r.request_id: r for r in s.query(PredictionLog).all()}


@pytest.fixture()
def sqlite_db():
    return SqliteDb()
