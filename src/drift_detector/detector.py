from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import select

from src.core.db import db_session
from src.core.logging import get_logger
from src.data_logger.models import PredictionLog
from src.drift_detector.alerter import send_slack_alert, trigger_retraining_dag

logger = get_logger(__name__)


def load_drift_config(config_path: str = "configs/drift.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)

def fetch_unprocessed_logs(limit: int) -> tuple[pd.DataFrame, list[str]]:
    with db_session() as session:
        stmt = (
            select(PredictionLog)
            .where(PredictionLog.drift_analyzed == False)
            .order_by(PredictionLog.created_at.asc())
            .limit(limit)
        )
        logs = session.scalars(stmt).all()

        if not logs:
            return pd.DataFrame(), []

        features_list = [log.features for log in logs]
        log_ids = [str(log.id) for log in logs]

        df = pd.DataFrame(features_list)
        for log in logs:
            log.drift_analyzed = True
        session.commit()
        return df, log_ids


def run_drift_detection(config_path: str = "configs/drift.yaml") -> None:
    logger.info("drift_detection_started")
    config = load_drift_config(config_path)
    det_cfg = config["detection"]
    ref_path = config["reference"]["data_path"]

    if not Path(ref_path).exists():
        logger.warning(
            "reference_data_not_found",
            path=ref_path,
            reason="Training pipeline must run first to generate reference data.",
        )
        return

    ref_df = pd.read_parquet(ref_path)

    curr_df, log_ids = fetch_unprocessed_logs(limit=det_cfg["window_size"])
    if curr_df.empty:
        logger.info("no_new_logs_for_drift_analysis")
        return

    curr_features = curr_df.columns.tolist()
    ref_features = [c for c in ref_df.columns if c in curr_features]
    ref_df_clean = ref_df[ref_features]
    curr_df_clean = curr_df[ref_features]

    if curr_df_clean.empty or ref_df_clean.empty:
        logger.warning("dataframe_column_mismatch_cannot_run_drift")
        return

    logger.info(
        "running_evidently",
        reference_rows=len(ref_df_clean),
        current_rows=len(curr_df_clean),
    )

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_df_clean, current_data=curr_df_clean)

    report_dict = drift_report.as_dict()
    try:
        metrics = report_dict["metrics"][0]["result"]
        drift_share = metrics["drift_share"]
        dataset_drift = metrics["dataset_drift"]
        drifted_features = metrics["number_of_drifted_columns"]
    except (KeyError, IndexError) as e:
        logger.error("evidently_metric_extraction_failed", error=str(e))
        return

    logger.info(
        "drift_analysis_complete",
        drift_share=round(drift_share, 4),
        dataset_drift=dataset_drift,
        drifted_features=drifted_features,
    )

    # Always persist the latest metrics so the status CLI can read them.
    out_dir = Path(config["evidently"]["report_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "drift_share": drift_share,
        "dataset_drift": dataset_drift,
        "drifted_features": drifted_features,
        "analyzed_rows": len(curr_df_clean),
        "report_path": str(out_dir / "latest_drift_report.html"),
    }
    metrics_json_path = out_dir / "latest_drift_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("drift_metrics_json_saved", path=str(metrics_json_path))

    if drift_share >= det_cfg["drift_share_threshold"]:
        logger.warning(
            "data_drift_detected_threshold_breached",
            drift_share=drift_share,
            threshold=det_cfg["drift_share_threshold"],
        )
        report_path = out_dir / "latest_drift_report.html"
        drift_report.save_html(str(report_path))
        logger.info("drift_html_report_saved", path=str(report_path))
        # Update the report_path in the persisted JSON now that the HTML exists.
        metrics_payload["report_path"] = str(report_path)
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)

        if config["alerting"]["trigger_airflow"]:
            trigger_retraining_dag(metrics_payload)

        send_slack_alert(
            drift_metrics=metrics_payload,
            config=config,
        )


if __name__ == "__main__":
    import time
    from src.core.logging import configure_logging
    configure_logging()

    config = load_drift_config()
    interval = config["detection"]["check_interval_seconds"]

    logger.info("starting_drift_detector_daemon", interval=interval)
    while True:
        try:
            run_drift_detection()
        except Exception as e:
            logger.error("drift_detector_crash", error=str(e), exc_info=True)
        time.sleep(interval)