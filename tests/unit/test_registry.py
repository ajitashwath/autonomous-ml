"""Unit tests for model_registry.registry (fully mocked — no MLflow server needed)"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ModelNotFoundError, ModelPromotionError, ModelRollbackError
from src.model_registry.registry import (
    ModelInfo,
    ModelRegistry,
    STAGE_ARCHIVED,
    STAGE_PRODUCTION,
    STAGE_STAGING,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_model_version(version: str, stage: str, run_id: str = "run-abc") -> MagicMock:
    mv = MagicMock()
    mv.version = version
    mv.current_stage = stage
    mv.run_id = run_id
    mv.name = "churn_classifier"
    return mv


def _make_registry() -> tuple[ModelRegistry, MagicMock]:
    """Return a ModelRegistry with a mocked MlflowClient."""
    with patch("src.model_registry.registry.MlflowClient") as MockClient:
        mock_client = MockClient.return_value
        registry = ModelRegistry.__new__(ModelRegistry)
        registry._client = mock_client
        registry._model_name = "churn_classifier"
        return registry, mock_client


# ── Tests: get_model_info ──────────────────────────────────────────────────────

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

        from tenacity import RetryError
        with pytest.raises(RetryError) as exc_info:
            registry.get_model_info(stage=STAGE_PRODUCTION)
            
        assert isinstance(exc_info.value.last_attempt.exception(), ModelNotFoundError)

    def test_raises_on_client_error(self):
        registry, client = _make_registry()
        client.get_latest_versions.side_effect = Exception("MLflow unreachable")

        from tenacity import RetryError
        with pytest.raises(RetryError) as exc_info:
            registry.get_model_info(stage=STAGE_PRODUCTION)
            
        assert isinstance(exc_info.value.last_attempt.exception(), ModelNotFoundError)


# ── Tests: promote_to_production ───────────────────────────────────────────────

class TestPromoteToProduction:
    def test_calls_transition(self):
        registry, client = _make_registry()
        mv = _make_model_version("5", STAGE_PRODUCTION)
        # We need two responses because get_model_info is called twice
        # 1st call: prev_info (returns nothing, so ModelNotFoundError is caught)
        # 2nd call: info after transition (returns the promoted version mv)
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
        # Mock finding a staging version, but transition fails
        mv = _make_model_version("5", STAGE_STAGING)
        client.get_latest_versions.side_effect = [[], [mv]]
        client.transition_model_version_stage.side_effect = Exception("MLflow error")

        from tenacity import RetryError
        with pytest.raises(RetryError) as exc_info:
            registry.promote_to_production(version="5")
            
        assert isinstance(exc_info.value.last_attempt.exception(), ModelPromotionError)


# ── Tests: rollback ────────────────────────────────────────────────────────────

class TestRollback:
    def test_raises_when_no_archived_versions(self):
        registry, client = _make_registry()
        # Current prod exists
        mv_prod = _make_model_version("6", STAGE_PRODUCTION)
        # No archived versions
        client.get_latest_versions.side_effect = [
            [mv_prod],   # get Production (to archive it)
            [],          # getting Archived versions
            [], [], [],  # other stages
        ]

        from tenacity import RetryError
        with pytest.raises(RetryError) as exc_info:
            registry.rollback()
            
        assert isinstance(exc_info.value.last_attempt.exception(), ModelRollbackError)


# ── Tests: get_model_uri ───────────────────────────────────────────────────────

class TestGetModelUri:
    def test_uri_format(self):
        registry, _ = _make_registry()
        uri = registry.get_model_uri(stage=STAGE_PRODUCTION)
        assert uri == "models:/churn_classifier/Production"

    def test_staging_uri(self):
        registry, _ = _make_registry()
        uri = registry.get_model_uri(stage=STAGE_STAGING)
        assert uri == "models:/churn_classifier/Staging"
