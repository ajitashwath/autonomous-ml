set -euo pipefail

echo "[entrypoint] Waiting for PostgreSQL..."
until pg_isready -h "${POSTGRES_HOST}" -U "${POSTGRES_USER}" -d "airflow"; do
    sleep 2
done
echo "[entrypoint] PostgreSQL is ready."

echo "[entrypoint] Running airflow db migrate..."
airflow db migrate

echo "[entrypoint] Creating admin user (idempotent)..."
airflow users create \
    --username "${AIRFLOW_USERNAME:-admin}" \
    --password "${AIRFLOW_PASSWORD:-admin}" \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@automlops.local \
    || true

echo "[entrypoint] Starting Airflow scheduler in background..."
airflow scheduler &

echo "[entrypoint] Starting Airflow webserver..."
exec airflow webserver --port 8080
