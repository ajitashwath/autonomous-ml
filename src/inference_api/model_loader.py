"""
inference_api/model_loader.py

Loads and caches the Production model from MLflow at startup.

Design decisions:
  - Model is loaded ONCE at startup (FastAPI lifespan) and held in memory.
    Re-loading on every request would add 200–500ms latency — unacceptable.
  - A background thread polls MLflow every POLL_INTERVAL_SECONDS to detect
    a new Production version and hot-swap without restarting the container.
  - Thread lock ensures consistent reads during the hot-swap window.

Production notes:
  - For high-traffic APIs, use a read-write lock so predictions don't block
    during model swap (concurrent reads are safe; only swap blocks writes).
  - In Kubernetes, the preferred pattern is a rolling deployment to new pods
    rather than in-process hot-swap. Use hot-swap only in single-container setups.
  - If the preprocessor (sklearn pipeline) is stored separately from the model,
    load it here too — serve skew is one of the most common production bugs.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import mlflow
import mlflow.xgboost
import xgboost as xgb

from src.core.config import get_settings
from src.core.exceptions import ModelLoadError, ModelNotFoundError
from src.core.logging import get_logger
from src.inference_api.metrics import MODEL_INFO, MODEL_LOAD_TOTAL
from src.model_registry.registry import ModelInfo, ModelRegistry

logger = get_logger(__name__)

# Polling interval for hot-swap (seconds)
POLL_INTERVAL_SECONDS = 60


class ModelLoader:
    """
    Thread-safe model loader with hot-swap support.

    Usage (FastAPI lifespan):
        loader = ModelLoader()
        loader.load()           # blocking, called at startup
        loader.start_polling()  # async hot-swap loop
        model = loader.get_model()
        info  = loader.get_info()
    """

    def __init__(self) -> None:
        self._model: Optional[xgb.XGBClassifier] = None
        self._info:  Optional[ModelInfo] = None
        self._lock   = threading.RLock()
        self._registry = ModelRegistry()
        self._settings = get_settings()
        self._stop_event = threading.Event()

    # ── Core load ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Blocking model load — called once at startup.
        Raises ModelLoadError if the Production model cannot be fetched.
        """
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

        model_uri = self._registry.get_model_uri(stage=stage)
        try:
            model = mlflow.xgboost.load_model(model_uri)
        except Exception as exc:
            MODEL_LOAD_TOTAL.labels(status="failure").inc()
            raise ModelLoadError(
                f"Failed to load model from URI '{model_uri}': {exc}",
                details={"model_uri": model_uri, "version": info.version},
            ) from exc

        with self._lock:
            self._model = model
            self._info  = info

        # Update Prometheus gauge
        MODEL_INFO.labels(
            model_version=info.version,
            model_stage=info.stage,
        ).set(1)
        MODEL_LOAD_TOTAL.labels(status="success").inc()

        logger.info(
            "model_loaded",
            version=info.version,
            stage=info.stage,
            auc=info.metrics.get("roc_auc"),
        )

    def _try_hot_swap(self) -> None:
        """
        Check if a newer Production version exists; swap without restart if so.
        Called by background polling thread — never raises, only logs on error.
        """
        try:
            latest_info = self._registry.get_model_info(
                stage=self._settings.inference_model_stage
            )
        except Exception as exc:
            logger.warning("hot_swap_registry_check_failed", error=str(exc))
            return

        with self._lock:
            current_version = self._info.version if self._info else None

        if latest_info.version == current_version:
            return  # No change — most common case, very cheap

        logger.info(
            "hot_swap_triggered",
            current_version=current_version,
            new_version=latest_info.version,
        )

        model_uri = self._registry.get_model_uri(
            stage=self._settings.inference_model_stage
        )
        try:
            new_model = mlflow.xgboost.load_model(model_uri)
        except Exception as exc:
            logger.error("hot_swap_load_failed", error=str(exc))
            MODEL_LOAD_TOTAL.labels(status="failure").inc()
            return

        # Atomic swap under lock
        with self._lock:
            self._model = new_model
            self._info  = latest_info

        MODEL_INFO.labels(
            model_version=latest_info.version,
            model_stage=latest_info.stage,
        ).set(1)
        MODEL_LOAD_TOTAL.labels(status="success").inc()
        logger.info("hot_swap_complete", new_version=latest_info.version)

    # ── Background polling ─────────────────────────────────────────────────────

    def start_polling(self) -> None:
        """Start the background hot-swap polling thread."""
        thread = threading.Thread(
            target=self._poll_loop,
            name="model-hot-swap-poller",
            daemon=True,   # dies with the main process
        )
        thread.start()
        logger.info("model_polling_started", interval_seconds=POLL_INTERVAL_SECONDS)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=POLL_INTERVAL_SECONDS)
            if not self._stop_event.is_set():
                self._try_hot_swap()

    def stop_polling(self) -> None:
        """Signal the polling thread to stop (called on app shutdown)."""
        self._stop_event.set()
        logger.info("model_polling_stopped")

    # ── Accessors (thread-safe) ────────────────────────────────────────────────

    def get_model(self) -> xgb.XGBClassifier:
        """
        Return the currently loaded model.
        Raises ModelLoadError if no model is loaded (should not happen post-startup).
        """
        with self._lock:
            if self._model is None:
                raise ModelLoadError("Model is not loaded. Check startup logs.")
            return self._model

    def get_info(self) -> Optional[ModelInfo]:
        """Return the ModelInfo of the currently loaded model."""
        with self._lock:
            return self._info

    def is_loaded(self) -> bool:
        """True if a model is currently held in memory."""
        with self._lock:
            return self._model is not None


# ── Module-level singleton ─────────────────────────────────────────────────────
# Shared across all FastAPI request handlers via app.state
_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Return the singleton ModelLoader (instantiated in FastAPI lifespan)."""
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader
