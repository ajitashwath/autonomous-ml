# ============================================================
# training_service/Dockerfile
# Lightweight image for running the training pipeline.
# Built separately from inference_api — only needs ML deps.
# ============================================================

FROM python:3.10-slim

LABEL maintainer="AutoMLOps Team"
LABEL service="training_service"

# System deps for XGBoost + psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    mlflow==2.13.0 \
    sqlalchemy==2.0.30 \
    psycopg2-binary==2.9.9 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1 \
    structlog==24.1.0 \
    PyYAML==6.0.1 \
    pyarrow==16.0.0 \
    scipy==1.13.0 \
    python-dotenv==1.0.1 \
    tenacity==8.3.0

# Copy project source
COPY src/ /app/src/
COPY configs/ /app/configs/

# Data is mounted at runtime (volume in docker-compose)
# so we only create the directory structure here
RUN mkdir -p /app/data/raw /app/data/processed /app/data/reference /app/artifacts

# Set PYTHONPATH so `src.*` imports resolve
ENV PYTHONPATH=/app

# Entrypoint: run training pipeline
# Override --config at `docker run` time if needed
ENTRYPOINT ["python", "-m", "src.training_service.train"]
CMD ["--config", "configs/training.yaml"]
