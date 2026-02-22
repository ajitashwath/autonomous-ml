"""
drift_detector/alerter.py

Handles triggering the Airflow retraining DAG when drift is detected.

Design decisions:
  - We use Airflow's REST API (/api/v1/dags/{dag_id}/dagRuns) to trigger
    the DAG, passing the drift metrics as `conf` payload.
  - Alerter only fires if `trigger_airflow: true` in drift.yaml.
  - Failures to reach Airflow are logged but don't crash the detector.

Event-Driven Architecture:
  Instead of Airflow polling the DB on a cron schedule, the detector pushes
  an event to Airflow the exact second drift crosses the threshold.
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def trigger_retraining_dag(drift_metrics: dict[str, Any]) -> str | None:
    """
    Trigger the Airflow retraining DAG via REST API.

    Args:
        drift_metrics: JSON-serializable dict of drift stats (e.g., drift_share).

    Returns:
        The Airflow dag_run_id if successful, None if skipped/failed.
    """
    settings = get_settings()

    url = f"{settings.airflow_api_url}/api/v1/dags/{settings.airflow_retrain_dag_id}/dagRuns"
    auth = (settings.airflow_api_user, settings.airflow_api_password)

    # Pass the drift metrics into the DAG run configuration
    # The DAG can read this using {{ dag_run.conf }}
    payload = {
        "conf": {
            "trigger_reason": "data_drift_detected",
            "drift_metrics": drift_metrics,
        }
    }

    logger.info(
        "triggering_airflow_dag",
        dag_id=settings.airflow_retrain_dag_id,
        url=url,
    )

    try:
        response = httpx.post(url, json=payload, auth=auth, timeout=10.0)
        response.raise_for_status()
        run_id = response.json().get("dag_run_id")
        logger.info("airflow_dag_triggered_successfully", run_id=run_id)
        return run_id
    except httpx.HTTPStatusError as exc:
        logger.error(
            "airflow_trigger_failed",
            status_code=exc.response.status_code,
            response=exc.response.text,
        )
        raise
    except Exception as exc:
        logger.error("airflow_trigger_network_error", error=str(exc))
        raise
