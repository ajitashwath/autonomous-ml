"""
inference_api/main.py

FastAPI application factory with lifespan management.

Lifespan (startup/shutdown):
    STARTUP:
      1. Configure logging
      2. Create DB tables (PredictionLog, etc.)
      3. Load Production model from MLflow (blocking — fail fast)
      4. Start background hot-swap polling thread

    SHUTDOWN:
      1. Stop polling thread gracefully
      2. SQLAlchemy connection pool closes automatically

Endpoints:
    GET  /health    — liveness + readiness probe
    POST /predict   — churn prediction
    GET  /metrics   — Prometheus metrics (scraped by prometheus container)

Production notes:
    - gunicorn + uvicorn workers in prod (not bare uvicorn).
      Command: gunicorn main:app -k uvicorn.workers.UvicornWorker -w 4
    - Add rate limiting middleware (slowapi) before going public.
    - CORS is intentionally not added — inference APIs are internal services.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from src.core.config import get_settings
from src.core.db import create_all_tables
from src.core.exceptions import AutoMLOpsError, ModelLoadError
from src.core.logging import configure_logging, get_logger
from src.inference_api.model_loader import ModelLoader, get_model_loader
from src.inference_api.routers import health, predict

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

# ── App startup time (used by /health) ─────────────────────────────────────────
_start_time = time.time()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    Everything before `yield` runs on startup; everything after on shutdown.
    """
    logger.info("inference_api_starting", env=settings.env)

    # 1. Create DB tables (idempotent — safe to run on every startup)
    try:
        create_all_tables()
    except Exception as exc:
        logger.warning("db_table_creation_failed", error=str(exc))

    # 2. Load model (BLOCKING — container stays unhealthy until this succeeds)
    loader: ModelLoader = get_model_loader()
    try:
        loader.load()
    except ModelLoadError as exc:
        logger.error(
            "startup_model_load_failed",
            error=exc.message,
            details=exc.details,
        )
        # In development, continue with a degraded API (health returns 503).
        # In production, you may want to raise here to force a container restart.
        if settings.env == "production":
            raise

    # 3. Store loader on app state (accessed by route dependencies)
    app.state.model_loader = loader
    app.state.start_time   = _start_time

    # 4. Start background hot-swap thread
    loader.start_polling()
    logger.info("inference_api_ready")

    yield  # ← API is live here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("inference_api_shutting_down")
    loader.stop_polling()
    logger.info("inference_api_shutdown_complete")


# ── App factory ────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="AutoMLOps Inference API",
        description=(
            "Production churn prediction API. "
            "Automatically loads the current Production model from MLflow. "
            "Supports hot-swap on new model promotions."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(predict.router, prefix="/api/v1")

    # ── Prometheus metrics endpoint ────────────────────────────────────────────
    # Mounts the prometheus_client ASGI app at /metrics
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Global exception handlers ──────────────────────────────────────────────
    @app.exception_handler(AutoMLOpsError)
    async def automlops_error_handler(request: Request, exc: AutoMLOpsError) -> JSONResponse:
        logger.error(
            "automlops_error",
            error=exc.message,
            details=exc.details,
            path=str(request.url),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": exc.message, "detail": str(exc.details), "code": 500},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=str(request.url),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal server error", "code": 500},
        )

    # ── Middleware: request logging ─────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=round(latency * 1000, 2),
        )
        return response

    return app


app = create_app()


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.inference_api.main:app",
        host=settings.inference_api_host,
        port=settings.inference_api_port,
        reload=(settings.env == "development"),
        log_level=settings.log_level.lower(),
    )
