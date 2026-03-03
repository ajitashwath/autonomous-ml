"""
validation_gate/gate.py

Two-stage statistical gate for promoting Staging → Production.

Stage 1 — AUC Delta Check:
    The Staging model's roc_auc must exceed the Production model's roc_auc
    by at least `auc_delta_threshold` (config: gate.auc_delta_threshold).

Stage 2 — McNemar's Test (optional, config: gate.require_significance):
    McNemar's test measures whether the *disagrement pattern* between the two
    models' predictions on a shared reference dataset is statistically
    significant. A model that is merely "luckily better" on averaged metrics
    will fail this test.

    NULL HYPOTHESIS: The two models have the same error rate.
    We PROMOTE only if we REJECT the null (p_value < significance_level).

Usage (Airflow DAG):
    python -m src.validation_gate.gate --config configs/validation.yaml

Production notes:
    - The McNemar test requires at least one disagreement row. If both models
      agree on every row, the test is skipped and only the AUC delta is used.
    - The reference.parquet saved during training is used as the shared test
      set. This is the same data the drift detector uses — no separate holdout
      needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.stats import chi2

from src.core.exceptions import ModelNotFoundError
from src.core.logging import get_logger
from src.model_registry.registry import ModelRegistry, STAGE_PRODUCTION, STAGE_STAGING

logger = get_logger(__name__)


def load_validation_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── McNemar's test ─────────────────────────────────────────────────────────────

def _run_mcnemar_test(
    y_pred_staging: np.ndarray,
    y_pred_prod: np.ndarray,
    significance_level: float,
) -> tuple[float, bool, str]:
    """
    Run McNemar's test on two sets of binary predictions.

    Returns:
        (p_value, test_passed, reason_string)
        test_passed=True means the improvement is statistically significant
        (i.e., reject H0 that both models have the same error rate).
    """
    # Build the 2×2 contingency table
    # b = staging wrong, prod correct (b counts cases unique to prod's advantage)
    # c = staging correct, prod wrong (c counts cases unique to staging's advantage)
    b = int(np.sum((y_pred_staging != y_pred_prod) & (y_pred_prod == 1)))
    c = int(np.sum((y_pred_staging != y_pred_prod) & (y_pred_staging == 1)))
    n_discordant = b + c

    if n_discordant == 0:
        logger.warning(
            "mcnemar_skipped_no_discordant_pairs",
            reason="Both models agree on every sample — test has no discriminating power.",
        )
        return 1.0, True, "skipped (models agree on all samples)"

    # McNemar statistic with continuity correction (Yates):
    chi2_stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)

    test_passed = p_value < significance_level
    reason = (
        f"p={p_value:.4f} < α={significance_level} → significant improvement"
        if test_passed
        else f"p={p_value:.4f} ≥ α={significance_level} → improvement not statistically significant"
    )

    logger.info(
        "mcnemar_test_complete",
        b=b,
        c=c,
        chi2_stat=round(chi2_stat, 4),
        p_value=round(p_value, 4),
        significance_level=significance_level,
        test_passed=test_passed,
    )
    return p_value, test_passed, reason


def _get_predictions_on_reference(
    registry: ModelRegistry,
    stage: str,
    reference_path: str,
    target_col: str = "Churn",
) -> Optional[np.ndarray]:
    """
    Load a model from MLflow and run it on the reference parquet dataset.
    Returns binary predictions (0/1) or None if loading fails.
    """
    try:
        import mlflow.xgboost
        model_uri = registry.get_model_uri(stage=stage)
        model = mlflow.xgboost.load_model(model_uri)

        ref_df = pd.read_parquet(reference_path)
        # Drop target col + any non-feature columns
        feature_cols = [c for c in ref_df.columns if c != target_col]
        X = ref_df[feature_cols]

        proba = model.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)
    except Exception as exc:
        logger.warning("mcnemar_model_load_failed", stage=stage, error=str(exc))
        return None


# ── Main gate logic ────────────────────────────────────────────────────────────

def validate_and_deploy(config_path: str = "configs/validation.yaml") -> None:
    logger.info("validation_gate_started")
    config = load_validation_config(config_path)
    gate_cfg = config.get("gate", {})
    auc_threshold = gate_cfg.get("auc_delta_threshold", 0.005)
    require_significance = gate_cfg.get("require_significance", True)
    significance_level = gate_cfg.get("significance_level", 0.05)

    registry = ModelRegistry()

    # ── Fetch Staging model ────────────────────────────────────────────────────
    try:
        staging_info = registry.get_model_info(stage=STAGE_STAGING)
    except ModelNotFoundError:
        logger.error("validation_failed_no_staging_model")
        sys.exit(1)

    new_auc = staging_info.metrics.get("roc_auc")
    if new_auc is None:
        logger.error("validation_failed_staging_model_missing_auc_metric")
        sys.exit(1)

    # ── First deployment fast-path ─────────────────────────────────────────────
    try:
        prod_info = registry.get_model_info(stage=STAGE_PRODUCTION)
    except ModelNotFoundError:
        logger.warning("no_production_model_found_first_deployment")
        promote_model(registry, staging_info.version, "First deployment (auto-promoted)")
        return

    prod_auc = prod_info.metrics.get("roc_auc")
    if prod_auc is None:
        logger.warning("production_model_missing_auc_metric_forcing_promotion")
        promote_model(registry, staging_info.version, "Forced promotion (Prod AUC missing)")
        return

    logger.info(
        "comparing_models",
        production_version=prod_info.version,
        production_auc=round(prod_auc, 4),
        staging_version=staging_info.version,
        staging_auc=round(new_auc, 4),
        required_improvement=auc_threshold,
    )

    # ── Stage 1: AUC delta check ───────────────────────────────────────────────
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

    # ── Stage 2: McNemar statistical significance test ─────────────────────────
    if require_significance:
        ref_path = config.get("reference", {}).get(
            "data_path", "data/reference/reference.parquet"
        )
        if not Path(ref_path).exists():
            logger.warning(
                "mcnemar_skipped_no_reference_data",
                path=ref_path,
                reason="Proceeding with AUC gate only.",
            )
        else:
            y_staging = _get_predictions_on_reference(registry, STAGE_STAGING, ref_path)
            y_prod = _get_predictions_on_reference(registry, STAGE_PRODUCTION, ref_path)

            if y_staging is not None and y_prod is not None:
                p_value, sig_passed, reason = _run_mcnemar_test(
                    y_staging, y_prod, significance_level
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

    # ── Both stages passed → promote ───────────────────────────────────────────
    logger.info(
        "validation_passed_both_stages",
        delta=round(delta, 4),
        staging_version=staging_info.version,
    )
    promote_model(
        registry,
        staging_info.version,
        f"Validation passed. AUC improved by {delta:.4f}",
    )


def promote_model(registry: ModelRegistry, version: str, annotation: str) -> None:
    try:
        from src.auto_deployer.deployer import deploy_model
        deploy_model(registry, version, annotation)
    except ImportError:
        registry.promote_to_production(version=version)
        registry.annotate_version(version=version, description=annotation)
        logger.info("model_promoted_successfully", version=version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/validation.yaml")
    args = parser.parse_args()

    try:
        validate_and_deploy(config_path=args.config)
    except Exception as exc:
        logger.error("validation_gate_crashed", error=str(exc), exc_info=True)
        sys.exit(1)
