"""Unit tests for the Validation Gate."""

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
    """Test that Staging is auto-promoted when Production doesn't exist."""
    mock_load_config.return_value = {"gate": {"auc_delta_threshold": 0.01}}
    
    # Staging exists
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
    """Test promotion when Staging AUC > Production AUC + threshold."""
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
    """Test rejection when Staging is worse than Production."""
    mock_load_config.return_value = {"gate": {"auc_delta_threshold": 0.01}}
    
    staging_info = MagicMock(version="2", metrics={"roc_auc": 0.82})
    prod_info = MagicMock(version="1", metrics={"roc_auc": 0.85})
    
    def side_effect(stage):
        if stage == STAGE_STAGING:
            return staging_info
        elif stage == STAGE_PRODUCTION:
            return prod_info
            
    mock_registry.get_model_info.side_effect = side_effect
    
    # Execution should finish cleanly (exit code 0 via return)
    validate_and_deploy(config_path="dummy.yaml")
    
    # Deployer should NOT have been called
    mock_deploy_model.assert_not_called()


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_exits_if_no_staging_model(mock_load_config, mock_registry):
    """Test script crashes (exit 1) if Airflow somehow runs this without a Staging model."""
    mock_registry.get_model_info.side_effect = ModelNotFoundError(f"No {STAGE_STAGING}")
    
    with pytest.raises(SystemExit) as exc:
        validate_and_deploy(config_path="dummy.yaml")
    
    assert exc.value.code == 1


# ── McNemar stage tests ────────────────────────────────────────────────────────

class TestMcNemarGate:
    """Tests for the two-stage McNemar statistical gate."""

    @patch("src.validation_gate.gate.load_validation_config")
    @patch("src.validation_gate.gate.get_predictions_on_reference")
    def test_gate_rejects_on_mcnemar_failure(
        self, mock_preds, mock_load_config, mock_registry, mock_deploy_model
    ):
        """
        AUC delta passes, but McNemar test fails (p >= alpha meaning the
        improvement is NOT statistically significant).
        → model must NOT be promoted.

        To create a high p-value we need b ≈ c: symmetric disagreement
        (staging correct on as many rows as production, i.e., no clear winner).
        """
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

        # Create symmetric disagreement: b == c, so chi2_stat ≈ 0 and p_value ≈ 1.0
        # staging_preds: [1,0,1,0,...], prod_preds: [0,1,0,1,...]
        # n_discordant rows where staging != prod, but b == c → no winner
        import numpy as np
        n = 100
        staging_preds = np.array([1, 0] * (n // 2))  # alternating
        prod_preds    = np.array([0, 1] * (n // 2))  # perfectly symmetric → b == c

        def preds_side_effect(registry, stage, ref_path):
            if stage == STAGE_STAGING:
                return staging_preds
            return prod_preds

        mock_preds.side_effect = preds_side_effect

        with patch("src.validation_gate.gate.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            validate_and_deploy(config_path="dummy.yaml")

        # Should NOT promote because McNemar p_value >= alpha (b == c, symmetric)
        mock_deploy_model.assert_not_called()


    @patch("src.validation_gate.gate.load_validation_config")
    @patch("src.validation_gate.gate.get_predictions_on_reference")
    def test_gate_promotes_when_both_stages_pass(
        self, mock_preds, mock_load_config, mock_registry, mock_deploy_model
    ):
        """
        Stage 1 (AUC delta) passes AND Stage 2 (McNemar) passes
        → promotion should proceed.
        """
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

        # Staging is clearly better: many rows where staging=1 (correct) and prod=0 (wrong)
        import numpy as np
        n = 500
        rng = np.random.default_rng(42)
        staging_preds = np.ones(n, dtype=int)  # staging always correct
        prod_preds    = rng.integers(0, 2, size=n)  # production is random

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
        """When require_significance=False, only AUC delta is checked."""
        mock_load_config.return_value = {
            "gate": {
                "auc_delta_threshold": 0.01,
                "require_significance": False,  # McNemar OFF
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

