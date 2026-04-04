FROM python:3.10-slim

LABEL maintainer="AutoMLOps Team"
LABEL service="drift_detector"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    evidently==0.4.30 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.4.2 \
    sqlalchemy==2.0.30 \
    psycopg2-binary==2.9.9 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1 \
    structlog==24.1.0 \
    PyYAML==6.0.1 \
    httpx==0.27.0 \
    tenacity==8.3.0 \
    pyarrow==16.0.0

COPY src/ /app/src/
COPY configs/ /app/configs/

RUN mkdir -p /app/data/reference /app/data/drift_reports

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "src.drift_detector.detector"]
