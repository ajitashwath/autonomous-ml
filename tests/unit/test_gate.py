from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import ModelNotFoundError, ValidationGateError
from src.model_registry.registry import STAGE_PRODUCTION, STAGE_STAGING, ModelInfo
from src.validation_gate.gate import (
    check_hard_floors,
    run_mcnemar_test,
    validate_and_deploy,
)


def _info(version: str, auc: float, **extra_metrics: float) -> ModelInfo:
    return ModelInfo(
        name="churn_classifier",
        version=version,
        stage="-",
        run_id=f"run-{version}",
        run_link="",
        metrics={"roc_auc": auc, **extra_metrics},
        params={"threshold": "0.5"},
    )


@pytest.fixture
def mock_registry():
    with patch("src.validation_gate.gate.ModelRegistry") as mock:
        yield mock.return_value


@pytest.fixture
def mock_deploy_model():
    with patch("src.validation_gate.gate.promote_model") as mock:
        yield mock


def _serve(registry, staging: ModelInfo, prod: ModelInfo | None) -> None:
    def side_effect(stage):
        if stage == STAGE_STAGING:
            return staging
        if prod is None:
            raise ModelNotFoundError(f"Model not found {STAGE_PRODUCTION}")
        return prod

    registry.get_model_info.side_effect = side_effect


# --------------------------------------------------------------------------- #
# Gate flow without a holdout (require_significance disabled)
# --------------------------------------------------------------------------- #

@patch("src.validation_gate.gate.load_validation_config")
def test_gate_promotes_on_first_deployment(mock_load_config, mock_registry, mock_deploy_model):
    mock_load_config.return_value = {"gate": {"auc_delta_threshold": 0.01}}
    _serve(mock_registry, _info("1", 0.85), prod=None)

    validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_called_once_with(
        mock_registry, "1", "First deployment (auto-promoted)"
    )


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_promotes_on_improvement(mock_load_config, mock_registry, mock_deploy_model):
    mock_load_config.return_value = {
        "gate": {"auc_delta_threshold": 0.02, "require_significance": False}
    }
    _serve(mock_registry, _info("2", 0.90), _info("1", 0.85))

    validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_called_once()
    args, _ = mock_deploy_model.call_args
    assert args[1] == "2"
    assert "AUC improved by 0.05" in args[2]


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_rejects_on_degradation(mock_load_config, mock_registry, mock_deploy_model):
    mock_load_config.return_value = {
        "gate": {"auc_delta_threshold": 0.01, "require_significance": False}
    }
    _serve(mock_registry, _info("2", 0.82), _info("1", 0.85))

    validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_not_called()


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_exits_if_no_staging_model(mock_load_config, mock_registry):
    mock_registry.get_model_info.side_effect = ModelNotFoundError(f"No {STAGE_STAGING}")
    with pytest.raises(SystemExit) as exc:
        validate_and_deploy(config_path="dummy.yaml")
    assert exc.value.code == 1


# --------------------------------------------------------------------------- #
# Hard floors
# --------------------------------------------------------------------------- #

def test_check_hard_floors_reports_each_violation_and_missing_metrics():
    violations = check_hard_floors(
        {"roc_auc": 0.65, "f1": 0.60}, {"roc_auc": 0.70, "f1": 0.55, "precision": 0.50}
    )
    assert len(violations) == 2
    assert any("roc_auc" in v for v in violations)
    assert any("precision missing" in v for v in violations)


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_rejects_model_below_hard_floor_even_on_first_deployment(
    mock_load_config, mock_registry, mock_deploy_model
):
    mock_load_config.return_value = {"gate": {"hard_floors": {"roc_auc": 0.70}}}
    _serve(mock_registry, _info("1", 0.60), prod=None)

    validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_not_called()


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_rejects_improvement_that_is_below_hard_floor(
    mock_load_config, mock_registry, mock_deploy_model
):
    mock_load_config.return_value = {
        "gate": {
            "auc_delta_threshold": 0.01,
            "require_significance": False,
            "hard_floors": {"f1": 0.55},
        }
    }
    _serve(mock_registry, _info("2", 0.90, f1=0.40), _info("1", 0.85))

    validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_not_called()


# --------------------------------------------------------------------------- #
# Gate flow with a labelled holdout + McNemar
# --------------------------------------------------------------------------- #

N_HOLDOUT = 400


@pytest.fixture
def holdout(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=N_HOLDOUT)
    df = pd.DataFrame({"tenure": rng.integers(1, 72, size=N_HOLDOUT), "Churn": y})
    path = tmp_path / "holdout.parquet"
    df.to_parquet(path, index=False)
    return path, y


def _noisy_proba(y: np.ndarray, error_rate: float, seed: int) -> np.ndarray:
    """Probabilities that are on the correct side of 0.5 except for `error_rate` of samples."""
    rng = np.random.default_rng(seed)
    wrong = rng.random(len(y)) < error_rate
    label = np.where(wrong, 1 - y, y)
    return np.where(label == 1, 0.8, 0.2) + rng.normal(0, 0.02, len(y))


def _gate_config(path, **gate) -> dict:
    return {
        "gate": {"auc_delta_threshold": 0.0, "require_significance": True, **gate},
        "evaluation": {"holdout_path": str(path), "target_column": "Churn"},
    }


def _score_by_version(probas: dict[str, np.ndarray]):
    def score(registry, info, X):
        return probas[info.version]

    return score


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_promotes_when_staging_is_significantly_more_correct(
    mock_load_config, mock_registry, mock_deploy_model, holdout
):
    path, y = holdout
    mock_load_config.return_value = _gate_config(path)
    _serve(mock_registry, _info("2", 0.0), _info("1", 0.0))
    probas = {"2": _noisy_proba(y, 0.05, 1), "1": _noisy_proba(y, 0.30, 2)}

    with patch("src.validation_gate.gate.score_model", _score_by_version(probas)):
        validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_called_once()
    assert mock_deploy_model.call_args.args[1] == "2"


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_rejects_when_staging_is_significantly_less_correct(
    mock_load_config, mock_registry, mock_deploy_model, holdout
):
    """Regression: the old test compared raw predictions, so 'staging predicts churn more
    often' looked like an improvement regardless of who was actually right."""
    path, y = holdout
    # AUC delta is disabled so this exercises the McNemar stage on its own.
    mock_load_config.return_value = _gate_config(path, auc_delta_threshold=-1.0)
    _serve(mock_registry, _info("2", 0.0), _info("1", 0.0))
    probas = {"2": _noisy_proba(y, 0.30, 1), "1": _noisy_proba(y, 0.05, 2)}

    with patch("src.validation_gate.gate.score_model", _score_by_version(probas)):
        validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_not_called()


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_rejects_when_improvement_is_not_significant(
    mock_load_config, mock_registry, mock_deploy_model, holdout
):
    path, y = holdout
    mock_load_config.return_value = _gate_config(path, auc_delta_threshold=-1.0)
    _serve(mock_registry, _info("2", 0.0), _info("1", 0.0))
    # Identical error rates with different noise seeds: any difference is chance.
    probas = {"2": _noisy_proba(y, 0.20, 1), "1": _noisy_proba(y, 0.20, 2)}

    with patch("src.validation_gate.gate.score_model", _score_by_version(probas)):
        validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_not_called()


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_fails_closed_when_holdout_missing_and_significance_required(
    mock_load_config, mock_registry, mock_deploy_model, tmp_path
):
    mock_load_config.return_value = _gate_config(tmp_path / "does_not_exist.parquet")
    _serve(mock_registry, _info("2", 0.90), _info("1", 0.85))

    with pytest.raises(ValidationGateError):
        validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_not_called()


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_fails_closed_when_a_model_cannot_be_scored(
    mock_load_config, mock_registry, mock_deploy_model, holdout
):
    """Regression: a scoring failure used to be swallowed, silently skipping the test
    and promoting on AUC alone."""
    path, _ = holdout
    mock_load_config.return_value = _gate_config(path)
    _serve(mock_registry, _info("2", 0.90), _info("1", 0.85))

    with patch(
        "src.validation_gate.gate.score_model",
        side_effect=ValidationGateError("cannot load model"),
    ):
        with pytest.raises(ValidationGateError):
            validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_not_called()


@patch("src.validation_gate.gate.load_validation_config")
def test_gate_scores_on_shared_holdout_not_on_logged_metrics(
    mock_load_config, mock_registry, mock_deploy_model, holdout
):
    """Logged AUCs claim staging is far better, but on the shared holdout it is worse."""
    path, y = holdout
    mock_load_config.return_value = _gate_config(path, require_significance=False)
    _serve(mock_registry, _info("2", 0.99), _info("1", 0.60))
    probas = {"2": _noisy_proba(y, 0.35, 1), "1": _noisy_proba(y, 0.05, 2)}

    with patch("src.validation_gate.gate.score_model", _score_by_version(probas)):
        validate_and_deploy(config_path="dummy.yaml")

    mock_deploy_model.assert_not_called()


# --------------------------------------------------------------------------- #
# McNemar's test directly
# --------------------------------------------------------------------------- #

class TestMcNemar:
    def test_uses_correctness_not_raw_predictions(self):
        """Staging predicts 1 far more often, but is wrong every time it disagrees."""
        y_true = np.zeros(200, dtype=int)
        staging = np.ones(200, dtype=int)   # always wrong
        prod = np.zeros(200, dtype=int)     # always right

        _, passed, reason = run_mcnemar_test(y_true, staging, prod, 0.05)

        assert not passed
        assert "not directionally better" in reason

    def test_passes_when_staging_fixes_prods_mistakes(self):
        y_true = np.ones(200, dtype=int)
        staging = np.ones(200, dtype=int)   # always right
        prod = np.zeros(200, dtype=int)     # always wrong

        p, passed, _ = run_mcnemar_test(y_true, staging, prod, 0.05)

        assert passed
        assert p < 0.05

    def test_equal_errors_do_not_pass(self):
        n = 100
        y_true = np.ones(n, dtype=int)
        staging = np.array([1, 0] * (n // 2))
        prod = np.array([0, 1] * (n // 2))   # b == c == 50

        _, passed, _ = run_mcnemar_test(y_true, staging, prod, 0.05)

        assert not passed

    def test_identical_correctness_is_not_evidence_of_improvement(self):
        y_true = np.array([0, 1, 0, 1])
        preds = np.array([0, 1, 1, 1])

        p, passed, reason = run_mcnemar_test(y_true, preds, preds.copy(), 0.05)

        assert not passed
        assert p == 1.0
        assert "no discordant pairs" in reason

    def test_small_samples_use_the_exact_test(self):
        # 8 discordant pairs, all favouring staging: exact two-sided p = 2 * 0.5**8 ≈ 0.0078.
        y_true = np.ones(8, dtype=int)
        staging = np.ones(8, dtype=int)
        prod = np.zeros(8, dtype=int)

        p, passed, _ = run_mcnemar_test(y_true, staging, prod, 0.05)

        assert p == pytest.approx(2 * 0.5**8)
        assert passed

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValidationGateError):
            run_mcnemar_test(np.zeros(3), np.zeros(3), np.zeros(4), 0.05)
