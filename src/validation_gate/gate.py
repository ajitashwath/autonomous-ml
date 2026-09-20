from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, chi2
from sklearn.metrics import roc_auc_score

from src.core.exceptions import ModelNotFoundError, ValidationGateError
from src.core.logging import get_logger
from src.model_registry.registry import (
    STAGE_PRODUCTION,
    STAGE_STAGING,
    ModelInfo,
    ModelRegistry,
)

logger = get_logger(__name__)

# Below this many discordant pairs the chi-square approximation is unreliable,
# so the exact (binomial) form of McNemar's test is used instead.
EXACT_TEST_MAX_DISCORDANT = 25

DEFAULT_MIN_PRODUCTION_ROWS = 50

PREPROCESSOR_ARTIFACT = "preprocessor/preprocessor.pkl"


def load_validation_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_mcnemar_test(
    y_true: np.ndarray,
    y_pred_staging: np.ndarray,
    y_pred_prod: np.ndarray,
    significance_level: float,
) -> tuple[float, bool, str]:
    """McNemar's test on the *correctness* of two classifiers over the same labelled samples.

    b = production right, staging wrong
    c = staging right, production wrong
    Staging passes only if the difference is significant AND c > b.
    """
    y_true = np.asarray(y_true)
    y_pred_staging = np.asarray(y_pred_staging)
    y_pred_prod = np.asarray(y_pred_prod)
    if not (len(y_true) == len(y_pred_staging) == len(y_pred_prod)):
        raise ValidationGateError(
            "McNemar inputs differ in length.",
            details={
                "y_true": len(y_true),
                "staging": len(y_pred_staging),
                "production": len(y_pred_prod),
            },
        )

    staging_ok = y_pred_staging == y_true
    prod_ok = y_pred_prod == y_true
    b = int(np.sum(prod_ok & ~staging_ok))
    c = int(np.sum(staging_ok & ~prod_ok))
    n_discordant = b + c

    if n_discordant == 0:
        # Identical correctness everywhere: there is no evidence that staging is better.
        logger.warning("mcnemar_no_discordant_pairs", b=b, c=c)
        return 1.0, False, "no discordant pairs — no evidence staging is better than production"

    if n_discordant < EXACT_TEST_MAX_DISCORDANT:
        p_value = float(binomtest(c, n_discordant, 0.5, alternative="two-sided").pvalue)
        method = "exact"
    else:
        chi2_stat = (abs(b - c) - 1.0) ** 2 / n_discordant
        p_value = float(chi2.sf(chi2_stat, df=1))
        method = "chi2"

    staging_is_better = c > b
    test_passed = p_value < significance_level and staging_is_better
    if p_value >= significance_level:
        reason = (
            f"p={p_value:.4f} ≥ α={significance_level} → improvement not statistically significant"
        )
    elif not staging_is_better:
        reason = (
            f"p={p_value:.4f} < α={significance_level} but staging is not directionally better "
            f"(b={b} prod-correct-staging-wrong, c={c} staging-correct-prod-wrong; need c > b)"
        )
    else:
        reason = (
            f"p={p_value:.4f} < α={significance_level} and c={c} > b={b} → significant improvement"
        )
    logger.info(
        "mcnemar_test_complete",
        method=method,
        b=b,
        c=c,
        p_value=round(p_value, 4),
        significance_level=significance_level,
        staging_is_better=staging_is_better,
        test_passed=test_passed,
    )
    return p_value, test_passed, reason


def check_hard_floors(metrics: dict[str, float], floors: dict[str, float]) -> list[str]:
    """Return a human-readable violation for every floor the metrics do not meet."""
    violations = []
    for name, floor in floors.items():
        value = metrics.get(name)
        if value is None:
            violations.append(f"{name} missing (floor {floor})")
        elif value < floor:
            violations.append(f"{name}={value:.4f} < floor {floor}")
    return violations


def _decision_threshold(info: ModelInfo) -> float:
    try:
        return float(info.params.get("threshold", 0.5))
    except (TypeError, ValueError):
        return 0.5


def load_holdout(path: str, target_column: str) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Labelled evaluation data (raw features + target), or None if the file is absent."""
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if target_column not in df.columns:
        raise ValidationGateError(
            f"Holdout data has no target column '{target_column}'.",
            details={"path": str(p), "columns": list(df.columns)},
        )
    y = df[target_column].astype(int).to_numpy()
    return df.drop(columns=[target_column]), y


def load_evaluation_set(
    eval_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, str] | None:
    """The labelled data candidates are scored on, plus a name for logging.

    Real, recent production outcomes are preferred over the static holdout carved out of the
    original CSV: they reflect the distribution the model is actually serving. They are only
    used when there are enough rows and both classes are present, since AUC and McNemar's
    test are meaningless otherwise.
    """
    target_column = eval_cfg.get("target_column", "Churn")

    production_path = eval_cfg.get("production_holdout_path")
    if production_path:
        min_rows = eval_cfg.get("min_production_holdout_rows", DEFAULT_MIN_PRODUCTION_ROWS)
        production = load_holdout(production_path, target_column)
        if production is not None:
            X, y = production
            if len(y) >= min_rows and len(set(y.tolist())) == 2:
                return X, y, "production_holdout"
            logger.warning(
                "production_holdout_unusable_falling_back",
                rows=len(y),
                required=min_rows,
                classes=sorted(set(y.tolist())),
            )

    static = load_holdout(
        eval_cfg.get("holdout_path", "data/reference/holdout.parquet"), target_column
    )
    if static is None:
        return None
    return static[0], static[1], "static_holdout"


def score_model(registry: ModelRegistry, info: ModelInfo, X: pd.DataFrame) -> np.ndarray:
    """Churn probabilities for raw features, using the model *and* its own preprocessor.

    Both are loaded by pinned version/run, so what is scored is exactly what `info` describes.
    Any failure raises — a gate that cannot score a model must not approve it.
    """
    import mlflow.artifacts
    import mlflow.xgboost

    from src.training_service.preprocessor import load_preprocessor

    try:
        model = mlflow.xgboost.load_model(registry.get_version_uri(info.version))
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                run_id=info.run_id, artifact_path=PREPROCESSOR_ARTIFACT, dst_path=tmp
            )
            preprocessor = load_preprocessor(local)
        return np.asarray(model.predict_proba(preprocessor.transform(X))[:, 1])
    except Exception as exc:
        raise ValidationGateError(
            f"Could not score model version {info.version}: {exc}",
            details={"version": info.version, "run_id": info.run_id},
        ) from exc


def validate_and_deploy(config_path: str = "configs/validation.yaml") -> None:
    logger.info("validation_gate_started")
    config = load_validation_config(config_path)
    gate_cfg = config.get("gate", {})
    auc_threshold = gate_cfg.get("auc_delta_threshold", 0.005)
    require_significance = gate_cfg.get("require_significance", True)
    significance_level = gate_cfg.get("significance_level", 0.05)
    hard_floors = gate_cfg.get("hard_floors", {}) or {}
    eval_cfg = config.get("evaluation", {})
    holdout_path = eval_cfg.get("holdout_path", "data/reference/holdout.parquet")

    registry = ModelRegistry()

    try:
        staging_info = registry.get_model_info(stage=STAGE_STAGING)
    except ModelNotFoundError:
        logger.error("validation_failed_no_staging_model")
        sys.exit(1)

    new_auc = staging_info.metrics.get("roc_auc")
    if new_auc is None:
        logger.error("validation_failed_staging_model_missing_auc_metric")
        sys.exit(1)

    violations = check_hard_floors(staging_info.metrics, hard_floors)
    if violations:
        logger.warning(
            "validation_failed_hard_floors",
            violations=violations,
            staging_version=staging_info.version,
            action="keeping_current_production",
        )
        return

    try:
        prod_info = registry.get_model_info(stage=STAGE_PRODUCTION)
    except ModelNotFoundError:
        logger.warning("no_production_model_found_first_deployment")
        promote_model(registry, staging_info.version, "First deployment (auto-promoted)")
        return

    evaluation = load_evaluation_set(eval_cfg)
    y_pred_staging = y_pred_prod = y_true = None
    evaluated_on = "logged_metrics"

    if evaluation is None:
        if require_significance:
            raise ValidationGateError(
                "Holdout data is required for the significance test but was not found; "
                "refusing to promote without it.",
                details={"holdout_path": holdout_path},
            )
        logger.warning("holdout_missing_falling_back_to_logged_auc", path=holdout_path)
        prod_auc = prod_info.metrics.get("roc_auc")
        if prod_auc is None:
            logger.warning("production_model_missing_auc_metric_forcing_promotion")
            promote_model(registry, staging_info.version, "Forced promotion (Prod AUC missing)")
            return
    else:
        X_holdout, y_true, evaluated_on = evaluation
        proba_staging = score_model(registry, staging_info, X_holdout)
        proba_prod = score_model(registry, prod_info, X_holdout)
        # Both models are compared on the same labelled samples, not on whatever
        # split each one happened to log during its own training run.
        new_auc = float(roc_auc_score(y_true, proba_staging))
        prod_auc = float(roc_auc_score(y_true, proba_prod))
        y_pred_staging = (proba_staging >= _decision_threshold(staging_info)).astype(int)
        y_pred_prod = (proba_prod >= _decision_threshold(prod_info)).astype(int)

    logger.info(
        "comparing_models",
        production_version=prod_info.version,
        production_auc=round(prod_auc, 4),
        staging_version=staging_info.version,
        staging_auc=round(new_auc, 4),
        required_improvement=auc_threshold,
        evaluated_on=evaluated_on,
        evaluation_rows=None if y_true is None else len(y_true),
    )

    delta = new_auc - prod_auc
    if delta < auc_threshold:
        logger.warning(
            "validation_failed_stage1_auc_delta",
            delta=round(delta, 4),
            reason=f"Improvement {delta:.4f} < threshold {auc_threshold}",
            action="keeping_current_production",
        )
        return

    logger.info("stage1_auc_delta_passed", delta=round(delta, 4))

    if require_significance:
        p_value, sig_passed, reason = run_mcnemar_test(
            y_true, y_pred_staging, y_pred_prod, significance_level
        )
        if not sig_passed:
            logger.warning(
                "validation_failed_stage2_mcnemar",
                p_value=round(p_value, 4),
                significance_level=significance_level,
                reason=reason,
                action="keeping_current_production",
            )
            return
        logger.info("stage2_mcnemar_passed", p_value=round(p_value, 4), reason=reason)

    logger.info(
        "validation_passed",
        delta=round(delta, 4),
        staging_version=staging_info.version,
    )
    promote_model(
        registry,
        staging_info.version,
        f"Validation passed. AUC improved by {delta:.4f}",
    )


def promote_model(registry: ModelRegistry, version: str, annotation: str) -> None:
    from src.auto_deployer.deployer import deploy_model

    deploy_model(registry, version, annotation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/validation.yaml")
    args = parser.parse_args()

    try:
        validate_and_deploy(config_path=args.config)
    except Exception as exc:
        logger.error("validation_gate_crashed", error=str(exc), exc_info=True)
        sys.exit(1)
