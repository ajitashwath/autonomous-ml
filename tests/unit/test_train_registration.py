"""Training must leave the new model in Staging — the stage the validation gate reads.

Previously log_model() registered a version in stage "None" and nothing ever moved it to
Staging, so the retrain -> gate hand-off in the Airflow DAG could never work.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlflow
import mlflow.xgboost
import numpy as np
import pytest
import xgboost as xgb

from src.core.config import get_settings
from src.model_registry.registry import (
    STAGE_ARCHIVED,
    STAGE_PRODUCTION,
    STAGE_STAGING,
    ModelRegistry,
)
from src.training_service import train
from src.training_service.train import register_and_stage

MODEL = "train_registration_model"


@pytest.fixture()
def tracking(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_MODEL_NAME", MODEL)
    get_settings.cache_clear()
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    exp_id = mlflow.create_experiment("train", artifact_location=(tmp_path / "art").as_uri())

    rng = np.random.default_rng(0)
    X = rng.random((100, 3))
    model = xgb.XGBClassifier(n_estimators=5, verbosity=0).fit(X, (X[:, 0] > 0.5).astype(int))

    def log_run() -> str:
        with mlflow.start_run(experiment_id=exp_id) as run:
            mlflow.xgboost.log_model(model, "model")
        return run.info.run_id

    try:
        yield log_run
    finally:
        mlflow.set_tracking_uri(previous_uri)
        get_settings.cache_clear()
        try:
            mlflow.MlflowClient(tracking_uri=uri)._tracking_client.store.engine.dispose()
        except Exception:
            pass


def _stages() -> dict[str, str]:
    client = mlflow.MlflowClient()
    return {str(v.version): v.current_stage for v in client.search_model_versions(f"name='{MODEL}'")}


def test_new_model_lands_in_staging_where_the_gate_looks(tracking):
    run_id = tracking()

    staged = register_and_stage(run_id)

    assert staged.version == "1"
    assert _stages() == {"1": STAGE_STAGING}
    assert ModelRegistry().get_model_info(stage=STAGE_STAGING).run_id == run_id


def test_log_run_logs_the_model_without_registering_it(tmp_path, monkeypatch):
    """Registration belongs to register_and_stage(); log_model(registered_model_name=...)
    would create a stage-"None" version that nothing ever promotes."""
    monkeypatch.chdir(tmp_path)
    config = {
        "model": {"hyperparameters": {"n_estimators": 5}},
        "data": {"test_size": 0.2, "random_state": 1},
        "evaluation": {"threshold": 0.5},
    }
    eval_result = MagicMock()
    eval_result.to_dict.return_value = {"roc_auc": 0.8}

    with patch.object(train, "mlflow") as mock_mlflow, patch.object(train, "save_preprocessor"):
        mock_mlflow.active_run.return_value.info.run_id = "run-1"
        run_id = train.log_run(
            config=config, settings=MagicMock(), eval_result=eval_result,
            feature_names=["a"], preprocessor=MagicMock(), model=MagicMock(),
            elapsed_seconds=1.0,
        )

    assert run_id == "run-1"
    mock_mlflow.xgboost.log_model.assert_called_once()
    assert "registered_model_name" not in mock_mlflow.xgboost.log_model.call_args.kwargs


def test_newer_candidate_supersedes_an_unpromoted_one(tracking):
    register_and_stage(tracking())

    register_and_stage(tracking())

    assert _stages() == {"1": STAGE_ARCHIVED, "2": STAGE_STAGING}


def test_staging_a_new_candidate_never_touches_production(tracking):
    registry_stage = register_and_stage(tracking())
    ModelRegistry().promote_to_production(registry_stage.version)

    register_and_stage(tracking())

    assert _stages() == {"1": STAGE_PRODUCTION, "2": STAGE_STAGING}
