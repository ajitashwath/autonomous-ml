"""
model_registry/registry.py

MLflow Model Registry wrapper — the single interface for all registry operations
across the AutoMLOps platform.

Every service (auto_deployer, rollback_manager, inference_api, validation_gate)
imports from this module. No service touches the MLflow client directly.

MLflow Stage Lifecycle:
    None → Staging → Production → Archived

    training_service.train  → registers → None
    auto_deployer           → promotes  → Staging → Production
    rollback_manager        → demotes   → Production → Archived → (prev) → Production

Production notes:
    - MLflow's Python client is thread-safe for reads. For concurrent writes
      (e.g., two DAGs racing to promote), add a Redis distributed lock around
      promote_to_production() and rollback().
    - In MLflow 2.x, model stages are deprecated in favour of model aliases.
      We use stages here for compatibility with MLflow 2.x OSS deployments.
      Migrate to aliases when upgrading to MLflow 3.x.
    - Use `tenacity` retry on all client calls — MLflow server can be slow to
      respond during artifact uploads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.exceptions import (
    ModelNotFoundError,
    ModelPromotionError,
    ModelRegistrationError,
    ModelRollbackError,
)
from src.core.logging import get_logger

logger = get_logger(__name__)

# ── Stage constants ────────────────────────────────────────────────────────────
STAGE_NONE       = "None"
STAGE_STAGING    = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED   = "Archived"


@dataclass
class ModelInfo:
    """Lightweight summary of a registered model version."""
    name: str
    version: str
    stage: str
    run_id: str
    run_link: str
    metrics: dict[str, float]
    params: dict[str, str]

    def __str__(self) -> str:
        return (
            f"ModelInfo(name={self.name}, version={self.version}, "
            f"stage={self.stage}, auc={self.metrics.get('roc_auc', 'N/A'):.4f})"
        )


class ModelRegistry:
    """
    High-level wrapper around MLflow's MlflowClient.

    Usage:
        registry = ModelRegistry()
        info = registry.get_model_info(stage="Production")
        registry.promote_to_production(version="5")
    """

    def __init__(self) -> None:
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self._client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        self._model_name = settings.mlflow_model_name
        logger.info(
            "model_registry_initialised",
            tracking_uri=settings.mlflow_tracking_uri,
            model_name=self._model_name,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_run_data(self, run_id: str) -> tuple[dict[str, float], dict[str, str]]:
        """Fetch metrics and params from an MLflow run."""
        try:
            run = self._client.get_run(run_id)
            metrics = {k: float(v) for k, v in run.data.metrics.items()}
            params  = {k: str(v) for k, v in run.data.params.items()}
            return metrics, params
        except Exception as exc:
            logger.warning("failed_to_fetch_run_data", run_id=run_id, error=str(exc))
            return {}, {}

    def _version_to_info(self, mv: ModelVersion) -> ModelInfo:
        """Convert an MLflow ModelVersion to our typed ModelInfo."""
        metrics, params = self._get_run_data(mv.run_id)
        tracking_uri = get_settings().mlflow_tracking_uri
        return ModelInfo(
            name=mv.name,
            version=mv.version,
            stage=mv.current_stage,
            run_id=mv.run_id,
            run_link=f"{tracking_uri}/#/experiments/1/runs/{mv.run_id}",
            metrics=metrics,
            params=params,
        )

    # ── Read operations ────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def get_model_info(self, stage: str = STAGE_PRODUCTION) -> ModelInfo:
        """
        Fetch the latest model version in the given stage.

        Args:
            stage: One of "Production", "Staging", "Archived", "None".

        Returns:
            ModelInfo for the latest version in that stage.

        Raises:
            ModelNotFoundError: if no version exists in the given stage.
        """
        try:
            versions = self._client.get_latest_versions(
                name=self._model_name, stages=[stage]
            )
        except Exception as exc:
            raise ModelNotFoundError(
                f"Could not query MLflow for model '{self._model_name}' in stage '{stage}': {exc}"
            ) from exc

        if not versions:
            raise ModelNotFoundError(
                f"No model version found for '{self._model_name}' in stage '{stage}'",
                details={"model_name": self._model_name, "stage": stage},
            )

        # Return the highest version number (latest)
        latest = max(versions, key=lambda v: int(v.version))
        info = self._version_to_info(latest)
        logger.info("model_info_fetched", stage=stage, version=info.version, auc=info.metrics.get("roc_auc"))
        return info

    def get_all_versions(self, stage: Optional[str] = None) -> list[ModelInfo]:
        """
        List all registered versions, optionally filtered by stage.

        Returns:
            List of ModelInfo sorted by version descending.
        """
        stages = [stage] if stage else [STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED]
        all_versions: list[ModelVersion] = []
        for s in stages:
            try:
                all_versions.extend(
                    self._client.get_latest_versions(name=self._model_name, stages=[s])
                )
            except Exception:
                pass  # stage may not exist yet
        return sorted(
            [self._version_to_info(v) for v in all_versions],
            key=lambda m: int(m.version),
            reverse=True,
        )

    def model_exists_in_stage(self, stage: str) -> bool:
        """Return True if at least one version exists in the given stage."""
        try:
            self.get_model_info(stage=stage)
            return True
        except ModelNotFoundError:
            return False

    # ── Transition operations ──────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def register_new_version(self, run_id: str) -> ModelInfo:
        """
        Register a completed training run as a new model version (stage=None).
        Called automatically by train.py via mlflow.xgboost.log_model(registered_model_name=...).
        This method is for explicit re-registration from a run_id.

        Returns:
            ModelInfo of the newly registered version.
        """
        logger.info("registering_new_model_version", run_id=run_id)
        try:
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name=self._model_name,
            )
        except Exception as exc:
            raise ModelRegistrationError(
                f"Failed to register model for run '{run_id}': {exc}",
                details={"run_id": run_id},
            ) from exc

        info = self._version_to_info(result)
        logger.info("model_version_registered", version=info.version, run_id=run_id)
        return info

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def transition_to_staging(self, version: str) -> ModelInfo:
        """
        Move a model version from None → Staging.
        Called by auto_deployer after validation gate passes preliminary checks.

        Args:
            version: String version number (e.g. "7").

        Returns:
            Updated ModelInfo.
        """
        logger.info("transitioning_to_staging", version=version)
        try:
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=version,
                stage=STAGE_STAGING,
                archive_existing_versions=False,
            )
        except Exception as exc:
            raise ModelPromotionError(
                f"Failed to transition version '{version}' to Staging: {exc}",
                details={"version": version},
            ) from exc

        info = self.get_model_info(stage=STAGE_STAGING)
        logger.info("model_transitioned_to_staging", version=version)
        return info

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def promote_to_production(self, version: str, archive_existing: bool = True) -> ModelInfo:
        """
        Promote a Staging model version to Production.
        Called by auto_deployer after the validation gate passes.

        Args:
            version:          String version number to promote.
            archive_existing: If True, the current Production model is archived.

        Returns:
            Updated ModelInfo for the newly promoted Production model.

        Raises:
            ModelPromotionError: if the MLflow transition call fails.
        """
        logger.info("promoting_to_production", version=version, archive_existing=archive_existing)

        # Record the current production version before overwriting (for audit trail)
        prev_production: Optional[str] = None
        try:
            prev_info = self.get_model_info(stage=STAGE_PRODUCTION)
            prev_production = prev_info.version
        except ModelNotFoundError:
            pass  # First-ever deployment — no existing production

        try:
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=version,
                stage=STAGE_PRODUCTION,
                archive_existing_versions=archive_existing,
            )
        except Exception as exc:
            raise ModelPromotionError(
                f"Failed to promote version '{version}' to Production: {exc}",
                details={"version": version, "previous_production": prev_production},
            ) from exc

        info = self.get_model_info(stage=STAGE_PRODUCTION)
        logger.info(
            "model_promoted_to_production",
            new_version=version,
            previous_version=prev_production,
            auc=info.metrics.get("roc_auc"),
        )
        return info

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def rollback(self, target_version: Optional[str] = None) -> ModelInfo:
        """
        Roll back Production to a specific version (or the most recent Archived version).

        This is the SAFE PATH called by rollback_manager when the validation gate rejects
        a new model. It:
          1. Archives the current Production version.
          2. Promotes the target version (or latest Archived) back to Production.

        Args:
            target_version: Explicit version string to restore. If None, restores
                            the most recently archived version.

        Returns:
            ModelInfo of the restored Production model.

        Raises:
            ModelRollbackError: if no suitable rollback target exists.
        """
        logger.info("rollback_initiated", target_version=target_version)

        # ── 1. Archive current Production ──────────────────────────────────────
        try:
            current_prod = self.get_model_info(stage=STAGE_PRODUCTION)
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=current_prod.version,
                stage=STAGE_ARCHIVED,
                archive_existing_versions=False,
            )
            logger.info("current_production_archived", version=current_prod.version)
        except ModelNotFoundError:
            logger.warning("no_production_model_found_during_rollback")

        # ── 2. Determine rollback target ───────────────────────────────────────
        if target_version is None:
            archived_versions = self.get_all_versions(stage=STAGE_ARCHIVED)
            if not archived_versions:
                raise ModelRollbackError(
                    "No archived model versions available for rollback.",
                    details={"model_name": self._model_name},
                )
            target_version = archived_versions[0].version  # most recent
            logger.info("rollback_target_auto_selected", version=target_version)

        # ── 3. Restore to Production ───────────────────────────────────────────
        try:
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=target_version,
                stage=STAGE_PRODUCTION,
                archive_existing_versions=False,
            )
        except Exception as exc:
            raise ModelRollbackError(
                f"Failed to restore version '{target_version}' to Production: {exc}",
                details={"target_version": target_version},
            ) from exc

        info = self.get_model_info(stage=STAGE_PRODUCTION)
        logger.info(
            "rollback_complete",
            restored_version=target_version,
            auc=info.metrics.get("roc_auc"),
        )
        return info

    def annotate_version(self, version: str, description: str) -> None:
        """
        Add a human-readable description to a model version.
        Used by auto_deployer and rollback_manager to leave an audit trail.
        """
        try:
            self._client.update_model_version(
                name=self._model_name,
                version=version,
                description=description,
            )
            logger.info("model_version_annotated", version=version, description=description)
        except Exception as exc:
            # Non-fatal — just log
            logger.warning("failed_to_annotate_version", version=version, error=str(exc))

    def get_model_uri(self, stage: str = STAGE_PRODUCTION) -> str:
        """
        Return the MLflow model URI for loading.
        Used by inference_api.model_loader to fetch the artifact.

        Example return value:
            "models:/churn_classifier/Production"
        """
        return f"models:/{self._model_name}/{stage}"
