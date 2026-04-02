

from __future__ import annotations

import time
import uuid
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, status

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
from src.inference_api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ChurnFeatures,
    PredictionResponse,
)

logger = get_logger(__name__)
router = APIRouter(tags=["Predictions"])


def _get_loader(request: Request) -> ModelLoader:
    return request.app.state.model_loader


def _features_to_dataframe(features: ChurnFeatures) -> pd.DataFrame:
    row = {
        field: getattr(features, field)
        for field in features.model_fields
    }
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
    request_id = str(uuid.uuid4())
    threshold = 0.5
    info = loader.get_info()
    if info and "threshold" in info.params:
        threshold = float(info.params["threshold"])

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()

    try:
        try:
            model = loader.get_model()
        except ModelLoadError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc.message),
            )

        X = _features_to_dataframe(features)

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
            logger.warning(
                "prediction_logging_failed",
                request_id=request_id,
                error=str(log_exc),
            )

        return PredictionResponse(
            prediction=prediction,
            probability=round(proba, 6),
            model_version=model_version,
            model_stage=info.stage if info else "unknown",
            threshold=threshold,
        )

    finally:
        ACTIVE_REQUESTS.dec()



_BATCH_MAX_SIZE = 500


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
async def predict_batch(
    body: BatchPredictionRequest,
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
        try:
            model = loader.get_model()
        except ModelLoadError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc.message),
            )

        info = loader.get_info()
        threshold = 0.5
        if info and "threshold" in info.params:
            threshold = float(info.params["threshold"])

        model_version = info.version if info else "unknown"
        model_stage   = info.stage   if info else "unknown"

        rows = []
        for feat in body.requests:
            row = {
                field: getattr(feat, field)
                for field in feat.model_fields
            }
            row = {k: v.value if hasattr(v, "value") else v for k, v in row.items()}
            rows.append(row)

        X = pd.DataFrame(rows)

        try:
            probas = model.predict_proba(X)[:, 1]
        except Exception as exc:
            ERRORS_TOTAL.labels(error_type="batch_prediction_error").inc()
            logger.error("batch_prediction_failed", n=n, error=str(exc), exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch prediction failed due to an internal error.",
            )

        results: list[PredictionResponse] = []
        for i, (feat, proba) in enumerate(zip(body.requests, probas)):
            proba_f = float(proba)
            prediction = int(proba_f >= threshold)
            request_id = str(uuid.uuid4())

            PREDICTIONS_TOTAL.labels(
                prediction_label=str(prediction),
                model_version=model_version,
            ).inc()
            PREDICTION_PROBABILITY.labels(model_version=model_version).observe(proba_f)

            try:
                from src.data_logger.logger import log_prediction_async
                await log_prediction_async(
                    request_id=request_id,
                    features=feat.model_dump(),
                    prediction=prediction,
                    probability=proba_f,
                    model_version=model_version,
                )
            except Exception as log_exc:
                logger.warning(
                    "batch_item_logging_failed",
                    item_index=i,
                    request_id=request_id,
                    error=str(log_exc),
                )

            results.append(PredictionResponse(
                prediction=prediction,
                probability=round(proba_f, 6),
                model_version=model_version,
                model_stage=model_stage,
                threshold=threshold,
            ))

        batch_latency = time.perf_counter() - batch_start
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(batch_latency)

        logger.info(
            "batch_prediction_served",
            n=n,
            model_version=model_version,
            latency_ms=round(batch_latency * 1000, 2),
        )

        return BatchPredictionResponse(
            predictions=results,
            count=n,
            model_version=model_version,
        )

    finally:
        ACTIVE_REQUESTS.dec()
