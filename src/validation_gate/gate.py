from __future__ import annotations

import argparse
import sys
from typing import Any

import yaml

from src.core.exceptions import ModelNotFoundError
from src.core.logging import get_logger
from src.model_registry.registry import ModelRegistry, STAGE_PRODUCTION, STAGE_STAGING

logger = get_logger(__name__)


def load_validation_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_and_deploy(config_path: str = "configs/validation.yaml") -> None:
    logger.info("validation_gate_started")
    config = load_validation_config(config_path)
    auc_threshold = config.get("gate", {}).get("auc_delta_threshold", 0.005)

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

    delta = new_auc - prod_auc
    if delta >= auc_threshold:
        logger.info(
            "validation_passed",
            delta=round(delta, 4),
            reason=f"Improvement >= {auc_threshold}",
        )
        promote_model(
            registry, 
            staging_info.version, 
            f"Validation passed. AUC improved by {delta:.4f}"
        )
    else:
        logger.warning(
            "validation_failed",
            delta=round(delta, 4),
            reason=f"Improvement < {auc_threshold}",
            action="keeping_current_production",
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
