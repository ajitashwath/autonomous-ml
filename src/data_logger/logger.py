import asyncio
from typing import Any

from src.core.db import db_session
from src.core.logging import get_logger
from src.data_logger.models import PredictionLog

logger = get_logger(__name__)

def log_prediction_sync(request_id: str, features: dict[str, Any], prediction: int, probability: float, model_version: str) -> None:
    try:
        with db_session() as session:
            log_entry = PredictionLog(
                request_id=request_id,
                features=features,
                prediction=prediction,
                probability=probability,
                model_version=model_version,
            )
            session.add(log_entry)
            session.commit()
    except Exception as exc:
        raise RuntimeError(f"Database insert failed: {exc}") from exc


async def log_prediction_async(request_id: str, features: dict[str, Any], prediction: int, probability: float, model_version: str) -> None:
    try:
        await asyncio.to_thread(
            log_prediction_sync,
            request_id=request_id,
            features=features,
            prediction=prediction,
            probability=probability,
            model_version=model_version,
        )
    except Exception as exc:
        logger.warning(
            "async_prediction_log_failed",
            request_id=request_id,
            error=str(exc),
        )