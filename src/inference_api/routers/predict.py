from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from src.core.exceptions import ModelLoadError
from src.core.logging import get_logger
from src.data_logger.logger import log_predictions_safe
from src.inference_api.metrics import (
    ACTIVE_REQUESTS,
    ERRORS_TOTAL,
    PREDICTION_PROBABILITY,
    PREDICTIONS_TOTAL,
    REQUEST_LATENCY,
)
from src.inference_api.model_loader import ModelLoader, ModelSnapshot
from src.inference_api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnFeatures,
    PredictionResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["Predictions"])

_BATCH_MAX_SIZE = 500

# These handlers are plain `def`, not `async def`: preprocessing and XGBoost inference are
# CPU-bound and synchronous, so FastAPI runs them in its threadpool instead of blocking the
# event loop (and with it /health, /metrics and every other in-flight request).


def _get_loader(request: Request) -> ModelLoader:
    return request.app.state.model_loader


def _get_snapshot(loader: ModelLoader) -> ModelSnapshot:
    # One snapshot per request: model, preprocessor and version metadata always belong together
    # even if a hot-swap lands mid-request.
    try:
        return loader.get_snapshot()
    except ModelLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc.message),
        )


def _to_dataframe(items: Sequence[ChurnFeatures]) -> pd.DataFrame:
    return pd.DataFrame([item.model_dump(mode="json") for item in items])


def _score(snapshot: ModelSnapshot, X: pd.DataFrame, *, kind: str = "") -> np.ndarray:
    """Churn probabilities for raw features. `kind` prefixes the error metric ("batch_")."""
    try:
        X_processed = snapshot.preprocessor.transform(X)
    except Exception as exc:
        ERRORS_TOTAL.labels(error_type=f"{kind}preprocessing_error").inc()
        logger.error(f"{kind}preprocessing_failed", n=len(X), error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Batch feature preprocessing failed." if kind else "Feature preprocessing failed."
            ),
        )

    try:
        return np.asarray(snapshot.model.predict_proba(X_processed)[:, 1], dtype=float)
    except Exception as exc:
        ERRORS_TOTAL.labels(error_type=f"{kind}prediction_error").inc()
        logger.error(f"{kind}prediction_failed", n=len(X), error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Batch prediction failed due to an internal error."
                if kind
                else "Prediction failed due to an internal error."
            ),
        )


def _record(request_id: str, features: ChurnFeatures, prediction: int, proba: float, version: str) -> dict:
    return {
        "request_id": request_id,
        "features": features.model_dump(mode="json"),
        "prediction": prediction,
        "probability": proba,
        "model_version": version,
    }


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
def predict(
    features: ChurnFeatures,
    background_tasks: BackgroundTasks,
    loader: Annotated[ModelLoader, Depends(_get_loader)],
) -> PredictionResponse:
    request_id = str(uuid.uuid4())

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()

    try:
        snapshot = _get_snapshot(loader)
        info = snapshot.info
        threshold = snapshot.threshold

        proba = float(_score(snapshot, _to_dataframe([features]))[0])
        prediction = int(proba >= threshold)

        PREDICTIONS_TOTAL.labels(
            prediction_label=str(prediction),
            model_version=info.version,
        ).inc()
        PREDICTION_PROBABILITY.labels(model_version=info.version).observe(proba)

        latency = time.perf_counter() - start
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)

        logger.info(
            "prediction_served",
            request_id=request_id,
            prediction=prediction,
            probability=round(proba, 4),
            model_version=info.version,
            latency_ms=round(latency * 1000, 2),
        )

        # Runs after the response is sent: a slow or unavailable database adds no latency.
        background_tasks.add_task(
            log_predictions_safe,
            [_record(request_id, features, prediction, proba, info.version)],
        )

        return PredictionResponse(
            prediction=prediction,
            probability=round(proba, 6),
            model_version=info.version,
            model_stage=info.stage,
            threshold=threshold,
        )

    finally:
        ACTIVE_REQUESTS.dec()


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch churn prediction",
    description=(
        "Submit up to 500 customer records in a single call. "
        "Predictions run in one vectorised forward pass — much faster than "
        "calling /predict in a loop."
    ),
    responses={
        400: {"description": "Invalid input"},
        413: {"description": "Batch exceeds maximum size"},
        503: {"description": "Model not loaded"},
        500: {"description": "Internal prediction error"},
    },
)
def predict_batch(
    body: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    loader: Annotated[ModelLoader, Depends(_get_loader)],
) -> BatchPredictionResponse:
    n = len(body.requests)

    if n > _BATCH_MAX_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size {n} exceeds maximum of {_BATCH_MAX_SIZE}.",
        )

    ACTIVE_REQUESTS.inc()
    batch_start = time.perf_counter()

    try:
        snapshot = _get_snapshot(loader)
        info = snapshot.info
        threshold = snapshot.threshold

        probas = _score(snapshot, _to_dataframe(body.requests), kind="batch_")
        if len(probas) != n:
            # zip() would otherwise silently truncate the response.
            ERRORS_TOTAL.labels(error_type="batch_prediction_error").inc()
            logger.error("batch_prediction_count_mismatch", expected=n, got=len(probas))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch prediction failed due to an internal error.",
            )

        results: list[PredictionResponse] = []
        records: list[dict] = []
        for feat, proba in zip(body.requests, probas):
            proba_f = float(proba)
            prediction = int(proba_f >= threshold)

            PREDICTIONS_TOTAL.labels(
                prediction_label=str(prediction),
                model_version=info.version,
            ).inc()
            PREDICTION_PROBABILITY.labels(model_version=info.version).observe(proba_f)

            records.append(_record(str(uuid.uuid4()), feat, prediction, proba_f, info.version))
            results.append(PredictionResponse(
                prediction=prediction,
                probability=round(proba_f, 6),
                model_version=info.version,
                model_stage=info.stage,
                threshold=threshold,
            ))

        # One INSERT for the whole batch, after the response is sent.
        background_tasks.add_task(log_predictions_safe, records)

        batch_latency = time.perf_counter() - batch_start
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(batch_latency)

        logger.info(
            "batch_prediction_served",
            n=n,
            model_version=info.version,
            latency_ms=round(batch_latency * 1000, 2),
        )

        return BatchPredictionResponse(
            predictions=results,
            count=n,
            model_version=info.version,
        )

    finally:
        ACTIVE_REQUESTS.dec()
