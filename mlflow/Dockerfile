FROM python:3.10-slim

LABEL service="mlflow"

# curl is required by the docker-compose healthcheck (python:slim does not ship it).
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pinned to the same version the clients use; an unpinned install at container start
# could drift from them and break the client/server protocol.
RUN pip install --no-cache-dir mlflow==2.13.0 psycopg2-binary==2.9.9

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
