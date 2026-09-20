from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass

import mlflow
import mlflow.artifacts
import mlflow.xgboost
import xgboost as xgb
from sklearn.compose import ColumnTransformer

from src.core.config import get_settings
from src.core.exceptions import ModelLoadError, ModelNotFoundError
from src.core.logging import get_logger
from src.inference_api.metrics import MODEL_INFO, MODEL_LOAD_TOTAL
from src.model_registry.registry import ModelInfo, ModelRegistry
from src.training_service.preprocessor import load_preprocessor

logger = get_logger(__name__)

POLL_INTERVAL_SECONDS = 60
PREPROCESSOR_ARTIFACT = "preprocessor/preprocessor.pkl"


@dataclass(frozen=True)
class ModelSnapshot:
    """A model, the preprocessor it was trained with, and its registry metadata.

    Always replaced as a whole, never mutated. A request that takes one snapshot therefore
    sees a consistent trio even if a hot-swap happens while it is being served.
    """

    model: xgb.XGBClassifier
    preprocessor: ColumnTransformer
    info: ModelInfo

    @property
    def threshold(self) -> float:
        try:
            return float(self.info.params.get("threshold", 0.5))
        except (TypeError, ValueError):
            return 0.5


class ModelLoader:

    def __init__(self) -> None:
        self._snapshot: ModelSnapshot | None = None
        self._publish_lock = threading.Lock()
        self._registry = ModelRegistry()
        self._settings = get_settings()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ reads

    def current(self) -> ModelSnapshot | None:
        # A single attribute read is atomic, so readers need no lock.
        return self._snapshot

    def get_snapshot(self) -> ModelSnapshot:
        snapshot = self._snapshot
        if snapshot is None:
            raise ModelLoadError("Model is not loaded. Check startup logs.")
        return snapshot

    def get_info(self) -> ModelInfo | None:
        snapshot = self._snapshot
        return snapshot.info if snapshot else None

    def is_loaded(self) -> bool:
        return self._snapshot is not None

    # ---------------------------------------------------------------- loading

    def _build_snapshot(self, info: ModelInfo) -> ModelSnapshot:
        """Load exactly the version described by `info`; raise rather than degrade."""
        # Pinned to the version so a promotion between "which version?" and "load it"
        # cannot leave `info` describing a different model than the one loaded.
        model_uri = self._registry.get_version_uri(info.version)
        try:
            model = mlflow.xgboost.load_model(model_uri)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from URI '{model_uri}': {exc}",
                details={"model_uri": model_uri, "version": info.version},
            ) from exc

        preprocessor = self._load_preprocessor(info.run_id, info.version)
        return ModelSnapshot(model=model, preprocessor=preprocessor, info=info)

    def _load_preprocessor(self, run_id: str, version: str) -> ColumnTransformer:
        # Serving without the fitted preprocessor would feed raw features to a model trained
        # on transformed ones. That produces wrong predictions or errors, so it is refused.
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=PREPROCESSOR_ARTIFACT, dst_path=tmp_dir
                )
                preprocessor = load_preprocessor(local_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Preprocessor artifact for model version {version} could not be loaded: {exc}",
                details={"run_id": run_id, "version": version, "artifact": PREPROCESSOR_ARTIFACT},
            ) from exc
        return preprocessor

    def _publish(self, snapshot: ModelSnapshot) -> None:
        with self._publish_lock:
            self._snapshot = snapshot
        # Only the live version is reported; older labels must not linger at 1.
        MODEL_INFO.clear()
        MODEL_INFO.labels(model_version=snapshot.info.version, model_stage=snapshot.info.stage).set(1)
        MODEL_LOAD_TOTAL.labels(status="success").inc()

    def load(self) -> None:
        stage = self._settings.inference_model_stage
        logger.info("model_load_started", stage=stage)

        try:
            info = self._registry.get_model_info(stage=stage)
        except ModelNotFoundError as exc:
            MODEL_LOAD_TOTAL.labels(status="failure").inc()
            raise ModelLoadError(
                f"No '{stage}' model found in MLflow registry. "
                "Run the training pipeline first.",
                details={"stage": stage},
            ) from exc

        try:
            snapshot = self._build_snapshot(info)
        except ModelLoadError:
            MODEL_LOAD_TOTAL.labels(status="failure").inc()
            raise

        self._publish(snapshot)
        logger.info(
            "model_loaded",
            version=info.version,
            stage=info.stage,
            auc=info.metrics.get("roc_auc"),
        )

    def _try_hot_swap(self) -> None:
        try:
            latest = self._registry.get_model_info(stage=self._settings.inference_model_stage)
        except Exception as exc:
            logger.warning("hot_swap_registry_check_failed", error=str(exc))
            return

        current = self._snapshot
        current_version = current.info.version if current else None
        if latest.version == current_version:
            return

        logger.info(
            "hot_swap_triggered",
            current_version=current_version,
            new_version=latest.version,
        )
        try:
            snapshot = self._build_snapshot(latest)
        except ModelLoadError as exc:
            # The running model keeps serving; the next poll retries.
            logger.error("hot_swap_load_failed", error=exc.message, details=exc.details)
            MODEL_LOAD_TOTAL.labels(status="failure").inc()
            return

        self._publish(snapshot)
        logger.info("hot_swap_complete", new_version=latest.version)

    # ---------------------------------------------------------------- polling

    def start_polling(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="model-hot-swap-poller",
            daemon=True,
        )
        self._thread.start()
        logger.info("model_polling_started", interval_seconds=POLL_INTERVAL_SECONDS)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=POLL_INTERVAL_SECONDS)
            if not self._stop_event.is_set():
                try:
                    self._try_hot_swap()
                except Exception as exc:
                    # An unexpected error must not kill the poller and freeze hot-swap forever.
                    logger.error("hot_swap_unexpected_error", error=str(exc), exc_info=True)

    def stop_polling(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("model_polling_stopped")


_loader: ModelLoader | None = None


def get_model_loader() -> ModelLoader:
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
