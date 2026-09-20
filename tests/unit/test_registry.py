

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ModelNotFoundError, ModelPromotionError, ModelRollbackError
from src.model_registry.registry import (
    STAGE_PRODUCTION,
    STAGE_STAGING,
    ModelInfo,
    ModelRegistry,
)


def _make_model_version(version: str, stage: str, run_id: str = "run-abc") -> MagicMock:
    mv = MagicMock()
    mv.version = version
    mv.current_stage = stage
    mv.run_id = run_id
    mv.name = "churn_classifier"
    return mv


def _make_registry() -> tuple[ModelRegistry, MagicMock]:
    with patch("src.model_registry.registry.MlflowClient") as MockClient:
        mock_client = MockClient.return_value
        registry = ModelRegistry.__new__(ModelRegistry)
        registry._client = mock_client
        registry._model_name = "churn_classifier"
        return registry, mock_client



class TestGetModelInfo:
    def test_returns_model_info(self):
        registry, client = _make_registry()
        mv = _make_model_version("3", STAGE_PRODUCTION)
        client.get_latest_versions.return_value = [mv]
        client.get_run.return_value.data.metrics = {"roc_auc": 0.85}
        client.get_run.return_value.data.params = {"n_estimators": "300"}

        with patch("src.model_registry.registry.get_settings") as mock_settings:
            mock_settings.return_value.mlflow_tracking_uri = "http://mlflow:5000"
            info = registry.get_model_info(stage=STAGE_PRODUCTION)

        assert isinstance(info, ModelInfo)
        assert info.version == "3"
        assert info.stage == STAGE_PRODUCTION

    def test_raises_when_no_version(self):
        registry, client = _make_registry()
        client.get_latest_versions.return_value = []

        with pytest.raises(ModelNotFoundError):
            registry.get_model_info(stage=STAGE_PRODUCTION)

    def test_raises_on_client_error(self):
        registry, client = _make_registry()
        client.get_latest_versions.side_effect = Exception("MLflow unreachable")

        with pytest.raises(ModelNotFoundError):
            registry.get_model_info(stage=STAGE_PRODUCTION)



class TestPromoteToProduction:
    def test_calls_transition(self):
        registry, client = _make_registry()
        mv = _make_model_version("5", STAGE_PRODUCTION)
        # 1st lookup: no current Production model. 2nd lookup: the promoted version.
        client.get_latest_versions.side_effect = [[], [mv]]
        client.get_run.return_value.data.metrics = {"roc_auc": 0.87}
        client.get_run.return_value.data.params = {}

        with patch("src.model_registry.registry.get_settings") as mock_settings:
            mock_settings.return_value.mlflow_tracking_uri = "http://mlflow:5000"
            mock_settings.return_value.mlflow_model_name = "churn_classifier"
            registry.promote_to_production(version="5")

        client.transition_model_version_stage.assert_called_once_with(
            name="churn_classifier",
            version="5",
            stage=STAGE_PRODUCTION,
            archive_existing_versions=True,
        )

    def test_raises_on_transition_failure(self):
        registry, client = _make_registry()
        client.get_latest_versions.return_value = []
        client.transition_model_version_stage.side_effect = Exception("MLflow error")

        with pytest.raises(ModelPromotionError):
            registry.promote_to_production(version="5")

    def test_empty_stage_is_not_retried(self):
        registry, client = _make_registry()
        client.get_latest_versions.return_value = []

        with pytest.raises(ModelNotFoundError):
            registry.get_model_info(stage=STAGE_PRODUCTION)

        assert client.get_latest_versions.call_count == 1

    def test_transport_errors_are_retried(self):
        registry, client = _make_registry()
        client.get_latest_versions.side_effect = Exception("connection reset")

        with pytest.raises(ModelNotFoundError):
            registry.get_model_info(stage=STAGE_PRODUCTION)

        assert client.get_latest_versions.call_count == 3



class TestRollback:
    def test_raises_when_no_earlier_version(self):
        registry, client = _make_registry()
        mv_prod = _make_model_version("6", STAGE_PRODUCTION)
        client.get_latest_versions.return_value = [mv_prod]
        client.search_model_versions.return_value = [mv_prod]

        with pytest.raises(ModelRollbackError):
            registry.rollback()

        # Nothing may be demoted when there is nothing to restore.
        client.transition_model_version_stage.assert_not_called()



class TestGetModelUri:
    def test_uri_format(self):
        registry, _ = _make_registry()
        uri = registry.get_model_uri(stage=STAGE_PRODUCTION)
        assert uri == "models:/churn_classifier/Production"

    def test_staging_uri(self):
        registry, _ = _make_registry()
        uri = registry.get_model_uri(stage=STAGE_STAGING)
        assert uri == "models:/churn_classifier/Staging"
