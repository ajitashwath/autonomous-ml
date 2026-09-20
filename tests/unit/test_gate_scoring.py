"""The gate's real scoring path: pinned model + its own preprocessor, on *raw* features.

Previously the gate fed raw features to the bare model, the resulting error was
swallowed, and the significance test was silently skipped.
"""
from __future__ import annotations

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.core.exceptions import ValidationGateError
from src.model_registry.registry import ModelInfo, ModelRegistry
from src.training_service.preprocessor import build_preprocessor, save_preprocessor
from src.validation_gate.gate import score_model

MODEL = "gate_scoring_model"


@pytest.fixture()
def logged_model(tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    try:
        exp_id = mlflow.create_experiment("gate", artifact_location=(tmp_path / "art").as_uri())

        rng = np.random.default_rng(0)
        X = pd.DataFrame({
            "tenure": rng.integers(1, 72, size=300),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], size=300),
        })
        y = ((X["tenure"] < 24) & (X["Contract"] == "Month-to-month")).astype(int).to_numpy()

        preprocessor = build_preprocessor(["Contract"], ["tenure"])
        X_proc = preprocessor.fit_transform(X)
        model = xgb.XGBClassifier(n_estimators=25, verbosity=0).fit(X_proc, y)

        pkl = tmp_path / "preprocessor.pkl"
        save_preprocessor(preprocessor, str(pkl))
        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.log_artifact(str(pkl), artifact_path="preprocessor")
            mlflow.xgboost.log_model(model, "model", registered_model_name=MODEL)

        registry = ModelRegistry.__new__(ModelRegistry)
        registry._client = mlflow.MlflowClient()
        registry._model_name = MODEL
        info = ModelInfo(
            name=MODEL, version="1", stage="None", run_id=run.info.run_id,
            run_link="", metrics={}, params={},
        )
        yield registry, info, X, model.predict_proba(X_proc)[:, 1]
    finally:
        mlflow.set_tracking_uri(previous_uri)
        try:
            store = mlflow.MlflowClient(tracking_uri=uri)._tracking_client.store
            store.engine.dispose()
        except Exception:
            pass


def test_score_model_applies_the_models_own_preprocessor_to_raw_features(logged_model):
    registry, info, X_raw, expected = logged_model

    proba = score_model(registry, info, X_raw)

    np.testing.assert_allclose(proba, expected, rtol=1e-5)


def test_score_model_raises_instead_of_degrading_when_preprocessor_is_missing(logged_model):
    registry, info, X_raw, _ = logged_model
    broken = ModelInfo(**{**info.__dict__, "run_id": "does-not-exist"})

    with pytest.raises(ValidationGateError):
        score_model(registry, broken, X_raw)


def test_score_model_raises_when_features_do_not_match_the_preprocessor(logged_model):
    registry, info, X_raw, _ = logged_model

    with pytest.raises(ValidationGateError):
        score_model(registry, info, X_raw.drop(columns=["tenure"]))
