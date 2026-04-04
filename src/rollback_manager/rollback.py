

from __future__ import annotations

import argparse
import sys
from typing import Optional

from src.core.exceptions import ModelRollbackError
from src.core.logging import get_logger
from src.model_registry.registry import ModelRegistry

logger = get_logger(__name__)


def execute_rollback(reason: str, target_version: Optional[str] = None) -> None:
    logger.warning("emergency_rollback_initiated", reason=reason, target=target_version or "latest_archived")
    registry = ModelRegistry()
    try:
        # Capture the current production version before rolling back — this is
        # what will be demoted/archived by registry.rollback().
        from src.core.exceptions import ModelNotFoundError
        try:
            current_prod = registry.get_model_info()
            demoted_version: Optional[str] = current_prod.version
        except ModelNotFoundError:
            demoted_version = None

        # rollback() returns a single ModelInfo for the newly-restored model.
        restored_info = registry.rollback(target_version=target_version)
        restored_version = restored_info.version

        if demoted_version:
            registry.annotate_version(
                version=demoted_version,
                description=f"[ROLLED-BACK] Reason: {reason}"
            )
        registry.annotate_version(
            version=restored_version,
            description=f"[RESTORED-TO-PROD] Emergency rollback replacement."
        )
        logger.info(
            "rollback_successful",
            demoted_version=demoted_version,
            restored_version=restored_version,
            action="API will hot-swap to the restored model within 60 seconds."
        )
    except ModelRollbackError as e:
        logger.critical("rollback_failed_catastrophically", error=str(e))
        sys.exit(1)
    except Exception as e:
        logger.critical("rollback_unexpected_error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency Model Rollback Tool")
    parser.add_argument(
        "--reason", 
        type=str, 
        required=True, 
        help="Mandatory reason for the rollback (added to MLflow audit logs)"
    )
    parser.add_argument(
        "--target-version", 
        type=str, 
        default=None, 
        help="Optional specific version string to restore. Defaults to the latest archived version."
    )
    args = parser.parse_args()
    execute_rollback(reason=args.reason, target_version=args.target_version)