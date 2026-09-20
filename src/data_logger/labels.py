from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy import select

from src.core.db import db_session
from src.core.logging import get_logger
from src.data_logger.models import PredictionLog

logger = get_logger(__name__)


@dataclass
class LabelResult:
    updated: int = 0
    correct: int = 0
    incorrect: int = 0
    unknown_request_ids: list[str] = field(default_factory=list)


def record_labels(labels: Sequence[tuple[str, int]]) -> LabelResult:
    """Attach ground-truth outcomes to previously logged predictions.

    `labels` is (request_id, actual_label) pairs. A label sent again for the same request
    overwrites the earlier one (corrections are allowed). Request ids that were never logged
    are reported back rather than raising, so one bad id does not reject the whole batch.
    """
    result = LabelResult()
    if not labels:
        return result

    now = datetime.datetime.now(datetime.timezone.utc)
    request_ids = {request_id for request_id, _ in labels}

    with db_session() as session:
        rows = session.scalars(
            select(PredictionLog).where(PredictionLog.request_id.in_(request_ids))
        ).all()
        by_request_id = {row.request_id: row for row in rows}

        for request_id, actual in labels:
            row = by_request_id.get(request_id)
            if row is None:
                result.unknown_request_ids.append(request_id)
                continue
            row.actual_label = actual
            row.label_received_at = now
            result.updated += 1
            if row.prediction == actual:
                result.correct += 1
            else:
                result.incorrect += 1

    logger.info(
        "labels_recorded",
        updated=result.updated,
        correct=result.correct,
        incorrect=result.incorrect,
        unknown=len(result.unknown_request_ids),
    )
    return result
