from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from src.core.exceptions import DatabaseError
from src.core.logging import get_logger
from src.data_logger.labels import record_labels
from src.inference_api.metrics import LABELS_TOTAL
from src.inference_api.schemas import LabelRequest, LabelResponse

logger = get_logger(__name__)
router = APIRouter(tags=["Feedback"])


@router.post(
    "/labels",
    response_model=LabelResponse,
    status_code=status.HTTP_200_OK,
    summary="Report what actually happened",
    description=(
        "Attach real outcomes to earlier predictions using the `request_id` each prediction "
        "returned. Labelled predictions become training data for the next retrain and the "
        "evaluation set the validation gate scores candidate models on. Sending a label again "
        "for the same request corrects it."
    ),
    responses={503: {"description": "Database unavailable"}},
)
def submit_labels(body: LabelRequest) -> LabelResponse:
    # Plain `def`: this does blocking database I/O, so it runs in the threadpool.
    try:
        result = record_labels([(item.request_id, item.actual_label) for item in body.labels])
    except DatabaseError as exc:
        logger.error("label_recording_failed", error=exc.message)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Labels could not be stored; please retry.",
        )

    LABELS_TOTAL.labels(outcome="correct").inc(result.correct)
    LABELS_TOTAL.labels(outcome="incorrect").inc(result.incorrect)

    return LabelResponse(
        received=len(body.labels),
        updated=result.updated,
        unknown_request_ids=result.unknown_request_ids,
    )
