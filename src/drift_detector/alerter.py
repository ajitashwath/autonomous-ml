from __future__ import annotations

import json
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def trigger_retraining_dag(drift_metrics: dict[str, Any]) -> str | None:
    settings = get_settings()

    url = f"{settings.airflow_host}/api/v1/dags/{settings.airflow_retrain_dag_id}/dagRuns"
    auth = (settings.airflow_username, settings.airflow_password)

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


def send_slack_alert(drift_metrics: dict[str, Any], config: dict[str, Any]) -> None:
    alerting_cfg = config.get("alerting", {})

    if not alerting_cfg.get("slack_enabled", False):
        return

    webhook_url: Optional[str] = alerting_cfg.get("slack_webhook_url", "").strip()
    if not webhook_url:
        logger.debug("slack_alert_skipped_no_webhook_url_configured")
        return

    drift_share = drift_metrics.get("drift_share", "N/A")
    drifted_features = drift_metrics.get("drifted_features", "N/A")
    analyzed_rows = drift_metrics.get("analyzed_rows", "N/A")
    report_path = drift_metrics.get("report_path", "N/A")
    threshold = config.get("detection", {}).get("drift_share_threshold", "N/A")

    message = {
        "text": ":rotating_light: *AutoMLOps — Data Drift Detected*",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Data Drift Threshold Breached",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Drift Share:*\n`{drift_share:.4f}`" if isinstance(drift_share, float) else f"*Drift Share:*\n`{drift_share}`"},
                    {"type": "mrkdwn", "text": f"*Threshold:*\n`{threshold}`"},
                    {"type": "mrkdwn", "text": f"*Drifted Features:*\n`{drifted_features}`"},
                    {"type": "mrkdwn", "text": f"*Analyzed Rows:*\n`{analyzed_rows}`"},
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Evidently Report:* `{report_path}`\n_Airflow retraining DAG triggered automatically._",
                },
            },
        ],
    }
    logger.info("sending_slack_drift_alert", drift_share=drift_share)
    try:
        response = httpx.post(
            webhook_url,
            content=json.dumps(message),
            headers={"Content-Type": "application/json"},
            timeout=5.0,
        )
        response.raise_for_status()
        logger.info("slack_alert_sent_successfully")
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "slack_alert_http_error",
            status_code=exc.response.status_code,
            response=exc.response.text[:200],
        )
    except Exception as exc:
        logger.warning("slack_alert_failed", error=str(exc))