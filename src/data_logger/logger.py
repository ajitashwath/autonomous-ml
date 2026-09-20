from typing import Any

from src.core.db import db_session
from src.core.logging import get_logger
from src.data_logger.models import PredictionLog

logger = get_logger(__name__)


def log_predictions_safe(records: list[dict[str, Any]]) -> None:
    """Insert prediction logs for drift analysis in a single transaction.

    Meant to run as a background task after the response has been sent, so it never raises:
    a database outage must not surface as a failed request or an unhandled task error.
    Each record needs: request_id, features, prediction, probability, model_version.
    """
    if not records:
        return
    try:
        with db_session() as session:
            session.add_all([PredictionLog(**record) for record in records])
    except Exception as exc:
        logger.warning("prediction_logging_failed", count=len(records), error=str(exc))
