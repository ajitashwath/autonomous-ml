

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ModelNotFoundError
from src.model_registry.registry import STAGE_PRODUCTION, STAGE_STAGING
from src.validation_gate.gate import validate_and_deploy


@pytest.fixture
def mock_registry():
    with patch("src.validation_gate.gate.ModelRegistry") as mock:
        yield mock.return_value


@pytest.fixture
def mock_deploy_model():
    with patch("src.validation_gate.gate.promote_model") as mock:
        yield mock


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_promotes_on_first_deployment(mock_load_config, mock_registry, mock_deploy_model):
    mock_load_config.return_value = {"gate": {"auc_delta_threshold": 0.01}}
    staging_info = MagicMock()
    staging_info.version = "1"
    staging_info.metrics = {"roc_auc": 0.85}
    def side_effect(stage):
        if stage == STAGE_STAGING:
            return staging_info
        elif stage == STAGE_PRODUCTION:
            raise ModelNotFoundError(f"Model not found {STAGE_PRODUCTION}")
    mock_registry.get_model_info.side_effect = side_effect
    validate_and_deploy(config_path="dummy.yaml")
    mock_deploy_model.assert_called_once_with(
        mock_registry, "1", "First deployment (auto-promoted)"
    )


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_promotes_on_improvement(mock_load_config, mock_registry, mock_deploy_model):
    mock_load_config.return_value = {"gate": {"auc_delta_threshold": 0.02}}
    staging_info = MagicMock(version="2", metrics={"roc_auc": 0.90})
    prod_info = MagicMock(version="1", metrics={"roc_auc": 0.85})
    def side_effect(stage):
        if stage == STAGE_STAGING:
            return staging_info
        elif stage == STAGE_PRODUCTION:
            return prod_info
    mock_registry.get_model_info.side_effect = side_effect
    validate_and_deploy(config_path="dummy.yaml")
    mock_deploy_model.assert_called_once()
    args, _ = mock_deploy_model.call_args
    assert args[1] == "2"
    assert "AUC improved by 0.05" in args[2]


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_rejects_on_degradation(mock_load_config, mock_registry, mock_deploy_model):
    mock_load_config.return_value = {"gate": {"auc_delta_threshold": 0.01}}
    staging_info = MagicMock(version="2", metrics={"roc_auc": 0.82})
    prod_info = MagicMock(version="1", metrics={"roc_auc": 0.85})
    def side_effect(stage):
        if stage == STAGE_STAGING:
            return staging_info
        elif stage == STAGE_PRODUCTION:
            return prod_info
    mock_registry.get_model_info.side_effect = side_effect
    validate_and_deploy(config_path="dummy.yaml")
    mock_deploy_model.assert_not_called()


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_exits_if_no_staging_model(mock_load_config, mock_registry):
    mock_registry.get_model_info.side_effect = ModelNotFoundError(f"No {STAGE_STAGING}")
    with pytest.raises(SystemExit) as exc:
        validate_and_deploy(config_path="dummy.yaml")
    assert exc.value.code == 1



class TestMcNemarGate:

    @patch("src.validation_gate.gate.load_validation_config")
    @patch("src.validation_gate.gate.get_predictions_on_reference")
    def test_gate_rejects_on_mcnemar_failure(
        self, mock_preds, mock_load_config, mock_registry, mock_deploy_model
    ):
        mock_load_config.return_value = {
            "gate": {
                "auc_delta_threshold": 0.01,
                "require_significance": True,
                "significance_level": 0.05,
            },
            "reference": {"data_path": "data/reference/reference.parquet"},
        }

        staging_info = MagicMock(version="2", metrics={"roc_auc": 0.90})
        prod_info    = MagicMock(version="1", metrics={"roc_auc": 0.85})

        def side_effect(stage):
            if stage == STAGE_STAGING:
                return staging_info
            return prod_info

        mock_registry.get_model_info.side_effect = side_effect

        import numpy as np
        n = 100
        staging_preds = np.array([1, 0] * (n // 2))
        prod_preds    = np.array([0, 1] * (n // 2))

        def preds_side_effect(registry, stage, ref_path):
            if stage == STAGE_STAGING:
                return staging_preds
            return prod_preds

        mock_preds.side_effect = preds_side_effect

        with patch("src.validation_gate.gate.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            validate_and_deploy(config_path="dummy.yaml")

        mock_deploy_model.assert_not_called()


    @patch("src.validation_gate.gate.load_validation_config")
    @patch("src.validation_gate.gate.get_predictions_on_reference")
    def test_gate_promotes_when_both_stages_pass(
        self, mock_preds, mock_load_config, mock_registry, mock_deploy_model
    ):
        mock_load_config.return_value = {
            "gate": {
                "auc_delta_threshold": 0.01,
                "require_significance": True,
                "significance_level": 0.05,
            },
            "reference": {"data_path": "data/reference/reference.parquet"},
        }

        staging_info = MagicMock(version="2", metrics={"roc_auc": 0.90})
        prod_info    = MagicMock(version="1", metrics={"roc_auc": 0.85})

        def side_effect(stage):
            if stage == STAGE_STAGING:
                return staging_info
            return prod_info

        mock_registry.get_model_info.side_effect = side_effect

        import numpy as np
        n = 500
        rng = np.random.default_rng(42)
        staging_preds = np.ones(n, dtype=int)
        prod_preds    = rng.integers(0, 2, size=n)

        def preds_side_effect(registry, stage, ref_path):
            if stage == STAGE_STAGING:
                return staging_preds
            return prod_preds

        mock_preds.side_effect = preds_side_effect

        with patch("src.validation_gate.gate.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            validate_and_deploy(config_path="dummy.yaml")

        mock_deploy_model.assert_called_once()

    @patch("src.validation_gate.gate.load_validation_config")
    def test_gate_skips_mcnemar_when_disabled(
        self, mock_load_config, mock_registry, mock_deploy_model
    ):
        mock_load_config.return_value = {
            "gate": {
                "auc_delta_threshold": 0.01,
                "require_significance": False,
            },
        }

        staging_info = MagicMock(version="2", metrics={"roc_auc": 0.90})
        prod_info    = MagicMock(version="1", metrics={"roc_auc": 0.85})

        def side_effect(stage):
            if stage == STAGE_STAGING:
                return staging_info
            return prod_info

        mock_registry.get_model_info.side_effect = side_effect

        with patch("src.validation_gate.gate.get_predictions_on_reference") as mock_preds:
            validate_and_deploy(config_path="dummy.yaml")
            mock_preds.assert_not_called()

        mock_deploy_model.assert_called_once()
