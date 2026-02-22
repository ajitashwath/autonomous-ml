# ============================================================
# inference_api/Dockerfile
# Production-grade image for the FastAPI inference service.
# ============================================================

FROM python:3.10-slim

LABEL maintainer="AutoMLOps Team"
LABEL service="inference_api"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (layer-cached)
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn[standard]==0.29.0 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1 \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    mlflow==2.13.0 \
    sqlalchemy==2.0.30 \
    psycopg2-binary==2.9.9 \
    prometheus-client==0.20.0 \
    structlog==24.1.0 \
    PyYAML==6.0.1 \
    python-dotenv==1.0.1 \
    tenacity==8.3.0 \
    httpx==0.27.0 \
    pyarrow==16.0.0

# Copy project source
COPY src/ /app/src/
COPY configs/ /app/configs/

ENV PYTHONPATH=/app

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check (Docker uses this to mark container as healthy)
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use uvicorn directly in dev; in prod use gunicorn with uvicorn workers
CMD ["uvicorn", "src.inference_api.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]
