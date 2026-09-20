

from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from src.core.db import check_db_connection
from src.core.logging import get_logger
from src.inference_api.model_loader import ModelLoader
from src.inference_api.schemas import HealthResponse

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])

_start_time: float = time.time()


def _get_loader(request: Request) -> ModelLoader:
    return request.app.state.model_loader


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns 200 if the model is loaded and DB is reachable. Returns 503 otherwise.",
)
def health_check(
    loader: Annotated[ModelLoader, Depends(_get_loader)],
) -> JSONResponse:
    # Plain `def`: check_db_connection() is a blocking call, so this runs in the threadpool
    # rather than stalling the event loop whenever the database is slow.
    snapshot = loader.current()
    model_loaded = snapshot is not None
    info = snapshot.info if snapshot else None
    db_ok = check_db_connection()

    status_str = "ok" if (model_loaded and db_ok) else "degraded"
    http_status = status.HTTP_200_OK if (model_loaded and db_ok) else status.HTTP_503_SERVICE_UNAVAILABLE

    body = HealthResponse(
        status=status_str,
        model_loaded=model_loaded,
        model_version=info.version if info else None,
        model_stage=info.stage if info else None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )

    if http_status != status.HTTP_200_OK:
        logger.warning(
            "health_check_degraded",
            model_loaded=model_loaded,
            db_ok=db_ok,
        )

    return JSONResponse(content=body.model_dump(), status_code=http_status)
