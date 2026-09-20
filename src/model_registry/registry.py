

from __future__ import annotations

from dataclasses import dataclass

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

STAGE_NONE       = "None"
STAGE_STAGING    = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED   = "Archived"

TAG_PREVIOUS_PRODUCTION = "previous_production"
TAG_ROLLED_BACK         = "rolled_back"


@dataclass
class ModelInfo:
    name: str
    version: str
    stage: str
    run_id: str
    run_link: str
    metrics: dict[str, float]
    params: dict[str, str]

    def __str__(self) -> str:
        auc = self.metrics.get("roc_auc")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        return (
            f"ModelInfo(name={self.name}, version={self.version}, "
            f"stage={self.stage}, auc={auc_str})"
        )


class ModelRegistry:

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


    def _get_run_data(self, run_id: str) -> tuple[dict[str, float], dict[str, str]]:
        try:
            run = self._client.get_run(run_id)
            metrics = {k: float(v) for k, v in run.data.metrics.items()}
            params  = {k: str(v) for k, v in run.data.params.items()}
            return metrics, params
        except Exception as exc:
            logger.warning("failed_to_fetch_run_data", run_id=run_id, error=str(exc))
            return {}, {}

    def _version_to_info(self, mv: ModelVersion) -> ModelInfo:
        metrics, params = self._get_run_data(mv.run_id)
        tracking_uri = get_settings().mlflow_tracking_uri
        return ModelInfo(
            name=mv.name,
            version=str(mv.version),
            stage=mv.current_stage,
            run_id=mv.run_id,
            run_link=f"{tracking_uri}/#/experiments/1/runs/{mv.run_id}",
            metrics=metrics,
            params=params,
        )


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _latest_versions(self, stage: str) -> list[ModelVersion]:
        # Only transport errors are retried here; an empty stage is a valid answer.
        return self._client.get_latest_versions(name=self._model_name, stages=[stage])

    def _all_versions(self) -> list[ModelVersion]:
        # get_latest_versions() returns a single version per stage, so it cannot be
        # used to enumerate history.
        return list(self._client.search_model_versions(f"name='{self._model_name}'"))

    def _set_tag(self, version: str, key: str, value: str) -> None:
        try:
            self._client.set_model_version_tag(self._model_name, version, key, value)
        except Exception as exc:
            logger.warning("failed_to_set_version_tag", version=version, key=key, error=str(exc))

    def _clear_tag(self, version: str, key: str) -> None:
        try:
            self._client.delete_model_version_tag(self._model_name, version, key)
        except Exception:
            pass

    def get_model_info(self, stage: str = STAGE_PRODUCTION) -> ModelInfo:
        try:
            versions = self._latest_versions(stage)
        except Exception as exc:
            raise ModelNotFoundError(
                f"Could not query MLflow for model '{self._model_name}' in stage '{stage}': {exc}"
            ) from exc

        if not versions:
            raise ModelNotFoundError(
                f"No model version found for '{self._model_name}' in stage '{stage}'",
                details={"model_name": self._model_name, "stage": stage},
            )

        latest = max(versions, key=lambda v: int(v.version))
        info = self._version_to_info(latest)
        logger.info("model_info_fetched", stage=stage, version=info.version, auc=info.metrics.get("roc_auc"))
        return info

    def get_all_versions(self, stage: str | None = None) -> list[ModelInfo]:
        try:
            versions = self._all_versions()
        except Exception as exc:
            logger.warning("failed_to_list_model_versions", error=str(exc))
            return []
        if stage:
            versions = [v for v in versions if v.current_stage == stage]
        return sorted(
            [self._version_to_info(v) for v in versions],
            key=lambda m: int(m.version),
            reverse=True,
        )

    def model_exists_in_stage(self, stage: str) -> bool:
        try:
            self.get_model_info(stage=stage)
            return True
        except ModelNotFoundError:
            return False


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def register_new_version(self, run_id: str) -> ModelInfo:
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
        logger.info("transitioning_to_staging", version=version)
        try:
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=version,
                stage=STAGE_STAGING,
                # Staging holds the single candidate awaiting validation; a newer
                # candidate supersedes (archives) any earlier one that was never promoted.
                archive_existing_versions=True,
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
        logger.info("promoting_to_production", version=version, archive_existing=archive_existing)

        prev_production: str | None = None
        try:
            prev_info = self.get_model_info(stage=STAGE_PRODUCTION)
            prev_production = prev_info.version
        except ModelNotFoundError:
            pass

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

        # Record lineage so a later rollback knows what to restore, and clear any
        # earlier "rolled_back" mark now that this version is deliberately live again.
        if prev_production and prev_production != str(version):
            self._set_tag(str(version), TAG_PREVIOUS_PRODUCTION, prev_production)
        self._clear_tag(str(version), TAG_ROLLED_BACK)

        info = self.get_model_info(stage=STAGE_PRODUCTION)
        logger.info(
            "model_promoted_to_production",
            new_version=version,
            previous_version=prev_production,
            auc=info.metrics.get("roc_auc"),
        )
        return info

    @staticmethod
    def _tags(mv: ModelVersion) -> dict[str, str]:
        tags = getattr(mv, "tags", None)
        return tags if isinstance(tags, dict) else {}

    def _resolve_rollback_target(
        self, current: ModelInfo | None, target_version: str | None
    ) -> str:
        """Pick the version to restore *before* touching any stage."""
        current_version = current.version if current else None
        versions = {str(v.version): v for v in self._all_versions()}

        if target_version is not None:
            target = str(target_version)
            if target == current_version:
                raise ModelRollbackError(
                    f"Version '{target}' is already in Production.",
                    details={"target_version": target},
                )
            if target not in versions:
                raise ModelRollbackError(
                    f"Version '{target}' does not exist for model '{self._model_name}'.",
                    details={"target_version": target},
                )
            return target

        def usable(version: str) -> bool:
            mv = versions.get(version)
            return (
                mv is not None
                and version != current_version
                and self._tags(mv).get(TAG_ROLLED_BACK) != "true"
            )

        # 1. The version that was live before the current one was promoted.
        if current_version is not None:
            previous = self._tags(versions[current_version]).get(TAG_PREVIOUS_PRODUCTION)
            if previous and usable(previous):
                return previous

        # 2. Newest archived version older than the current Production model.
        archived = [
            int(v)
            for v, mv in versions.items()
            if mv.current_stage == STAGE_ARCHIVED
            and usable(v)
            and (current_version is None or int(v) < int(current_version))
        ]
        if not archived:
            raise ModelRollbackError(
                "No earlier model version is available for rollback.",
                details={"model_name": self._model_name, "current_production": current_version},
            )
        return str(max(archived))

    def rollback(self, target_version: str | None = None) -> ModelInfo:
        # Deliberately not wrapped in @retry: a partial failure that is re-run must
        # not re-evaluate the registry after it has already been modified.
        logger.info("rollback_initiated", target_version=target_version)

        try:
            current = self.get_model_info(stage=STAGE_PRODUCTION)
        except ModelNotFoundError:
            current = None
            logger.warning("no_production_model_found_during_rollback")

        target = self._resolve_rollback_target(current, target_version)
        logger.info("rollback_target_selected", version=target)

        try:
            # One call restores the target and archives the demoted model, so
            # Production is never empty and the demoted model is never re-selected.
            self._client.transition_model_version_stage(
                name=self._model_name,
                version=target,
                stage=STAGE_PRODUCTION,
                archive_existing_versions=True,
            )
        except Exception as exc:
            raise ModelRollbackError(
                f"Failed to restore version '{target}' to Production: {exc}",
                details={"target_version": target},
            ) from exc

        if current is not None:
            self._set_tag(current.version, TAG_ROLLED_BACK, "true")
            logger.info("current_production_archived", version=current.version)
        self._clear_tag(target, TAG_ROLLED_BACK)

        info = self.get_model_info(stage=STAGE_PRODUCTION)
        logger.info(
            "rollback_complete",
            restored_version=target,
            auc=info.metrics.get("roc_auc"),
        )
        return info

    def annotate_version(self, version: str, description: str) -> None:
        try:
            self._client.update_model_version(
                name=self._model_name,
                version=version,
                description=description,
            )
            logger.info("model_version_annotated", version=version, description=description)
        except Exception as exc:
            logger.warning("failed_to_annotate_version", version=version, error=str(exc))

    def get_model_uri(self, stage: str = STAGE_PRODUCTION) -> str:
        return f"models:/{self._model_name}/{stage}"

    def get_version_uri(self, version: str) -> str:
        # Pinned to a version so a stage change between lookup and load can't swap the model.
        return f"models:/{self._model_name}/{version}"
