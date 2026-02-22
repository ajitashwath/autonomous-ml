FROM apache/airflow:2.9.1-python3.10

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    mlflow==2.13.0 \
    evidently==0.4.30 \
    sqlalchemy==2.0.30 \
    psycopg2-binary==2.9.9 \
    pydantic==2.7.1 \
    pydantic-settings==2.2.1 \
    structlog==24.1.0 \
    PyYAML==6.0.1 \
    httpx==0.27.0 \
    tenacity==8.3.0 \
    pyarrow==16.0.0 \
    scipy==1.13.0 \
    python-dotenv==1.0.1

COPY src/ /opt/airflow/src/

COPY airflow/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
