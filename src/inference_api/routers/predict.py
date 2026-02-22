"""
inference_api/routers/predict.py

POST /predict — the core prediction endpoint.

Request lifecycle:
    1. Pydantic validates input (400 on bad input — before model is touched)
    2. Convert features to DataFrame (preserving column order from training)
    3. Run model.predict_proba()
    4. Apply threshold → binary label
    5. Log request + result to PostgreSQL (async, via data_logger)
    6. Update Prometheus metrics
    7. Return PredictionResponse

Failure modes handled:
    - Bad input:           400 (Pydantic catches this before we see it)
    - Model not loaded:    503 (ModelLoadError → caught in exception handler)
    - Unexpected error:    500 (caught in exception handler, logged with trace)
    - DB write failure:    Warning only — never block a prediction for a log write
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.core.config import get_settings
from src.core.exceptions import ModelLoadError
from src.core.logging import get_logger
from src.inference_api.metrics import (
    ACTIVE_REQUESTS,
    ERRORS_TOTAL,
    PREDICTION_PROBABILITY,
    PREDICTIONS_TOTAL,
    REQUEST_LATENCY,
)
from src.inference_api.model_loader import ModelLoader
from src.inference_api.schemas import ChurnFeatures, PredictionResponse

logger = get_logger(__name__)
router = APIRouter(tags=["Predictions"])


def _get_loader(request: Request) -> ModelLoader:
    return request.app.state.model_loader


def _features_to_dataframe(features: ChurnFeatures) -> pd.DataFrame:
    """
    Convert the validated Pydantic model to a single-row DataFrame.
    Column order MUST match the column order seen during training.
    We rely on dict ordering (Python 3.7+) which matches model_fields order.
    """
    row = {
        field: getattr(features, field)
        for field in features.model_fields
    }
    # Convert enum values to their string representation
    row = {k: v.value if hasattr(v, "value") else v for k, v in row.items()}
    return pd.DataFrame([row])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict customer churn",
    description="Submit customer features and receive a churn prediction with probability.",
    responses={
        400: {"description": "Invalid input features"},
        503: {"description": "Model not loaded"},
        500: {"description": "Internal prediction error"},
    },
)
async def predict(
    features: ChurnFeatures,
    loader: Annotated[ModelLoader, Depends(_get_loader)],
    request: Request,
) -> PredictionResponse:
    """
    Core prediction endpoint.
    All metrics are updated synchronously; DB logging is fire-and-forget.
    """
    request_id = str(uuid.uuid4())
    settings = get_settings()
    threshold = float(settings.inference_model_stage and 0.5)  # default 0.5
    # Read threshold from config (we store it as a param in MLflow)
    info = loader.get_info()
    if info and "threshold" in info.params:
        threshold = float(info.params["threshold"])

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()

    try:
        # ── 1. Get model ───────────────────────────────────────────────────────
        try:
            model = loader.get_model()
        except ModelLoadError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc.message),
            )

        # ── 2. Convert features ────────────────────────────────────────────────
        X = _features_to_dataframe(features)

        # ── 3. Predict ─────────────────────────────────────────────────────────
        try:
            proba = float(model.predict_proba(X)[0, 1])
        except Exception as exc:
            ERRORS_TOTAL.labels(error_type="prediction_error").inc()
            logger.error(
                "prediction_failed",
                request_id=request_id,
                error=str(exc),
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed due to an internal error.",
            )

        prediction = int(proba >= threshold)
        model_version = info.version if info else "unknown"

        # ── 4. Metrics ─────────────────────────────────────────────────────────
        PREDICTIONS_TOTAL.labels(
            prediction_label=str(prediction),
            model_version=model_version,
        ).inc()
        PREDICTION_PROBABILITY.labels(model_version=model_version).observe(proba)

        latency = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

        logger.info(
            "prediction_served",
            request_id=request_id,
            prediction=prediction,
            probability=round(proba, 4),
            model_version=model_version,
            latency_ms=round(latency * 1000, 2),
        )

        # ── 5. Async DB log (fire-and-forget) ──────────────────────────────────
        # data_logger is added in Phase 5 — import guarded to avoid early failure
        try:
            from src.data_logger.logger import log_prediction_async
            await log_prediction_async(
                request_id=request_id,
                features=features.model_dump(),
                prediction=prediction,
                probability=proba,
                model_version=model_version,
            )
        except Exception as log_exc:
            # NEVER block a prediction because the logging failed
            logger.warning(
                "prediction_logging_failed",
                request_id=request_id,
                error=str(log_exc),
            )

        # ── 6. Return response ─────────────────────────────────────────────────
        return PredictionResponse(
            prediction=prediction,
            probability=round(proba, 6),
            model_version=model_version,
            model_stage=info.stage if info else "unknown",
            threshold=threshold,
        )

    finally:
        ACTIVE_REQUESTS.dec()
