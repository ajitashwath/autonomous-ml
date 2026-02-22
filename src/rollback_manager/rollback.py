"""
rollback_manager/rollback.py

CLI utility to instantly demote the current Production model and restore the
previous known-stable version from the 'Archived' stage.

Usage:
  python -m src.rollback_manager.rollback --reason "Severe P99 latency degradation"

Design:
  This script heavily relies on the `ModelRegistry.rollback()` method we
  built in Phase 3. It simply provides a user-friendly CLI and standardizes
  the annotation structure for audit trailing.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from src.core.exceptions import ModelRollbackError
from src.core.logging import get_logger
from src.model_registry.registry import ModelRegistry

logger = get_logger(__name__)


def execute_rollback(reason: str, target_version: Optional[str] = None) -> None:
    """
    Triggers an emergency rollback in MLflow.
    
    Args:
        reason: Mandatory justification for the audit log.
        target_version: If provided, rolls back to this exact version.
                        Otherwise, rolls back to the most recent 'Archived' model.
    """
    logger.warning("emergency_rollback_initiated", reason=reason, target=target_version or "latest_archived")
    
    registry = ModelRegistry()
    
    try:
        # The registry handles finding the current Prod model, archiving it,
        # retrieving the old Archived model, and promoting it back to Prod.
        restored_version, demoted_version = registry.rollback(target_version=target_version)
        
        # Annotate the demoted model explaining WHY it was killed
        registry.annotate_version(
            version=demoted_version,
            description=f"[ROLLED-BACK] Reason: {reason}"
        )
        
        # Annotate the restored model 
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
