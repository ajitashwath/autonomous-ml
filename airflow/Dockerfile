FROM apache/airflow:2.9.1-python3.10

USER root

# postgresql-client provides pg_isready, which entrypoint.sh uses to wait for the database.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data && chown -R airflow:root /app

USER airflow

# The ML code runs in its own virtualenv so its dependencies (notably SQLAlchemy 2.x)
# cannot conflict with Airflow's own pinned environment.
COPY airflow/requirements-ml.txt /tmp/requirements-ml.txt
RUN python -m venv /home/airflow/ml-venv \
    && /home/airflow/ml-venv/bin/pip install --no-cache-dir -r /tmp/requirements-ml.txt

ENV ML_PYTHON=/home/airflow/ml-venv/bin/python

# Same layout the rest of the platform uses (/app/src, /app/configs, /app/data); docker-compose
# bind-mounts over these for development.
COPY --chown=airflow:root src/ /app/src/
COPY --chown=airflow:root configs/ /app/configs/

COPY --chown=airflow:root airflow/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
