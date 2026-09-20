from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sqlalchemy import select, update

from src.core.db import db_session
from src.core.logging import get_logger
from src.data_logger.models import PredictionLog
from src.drift_detector.alerter import send_slack_alert, trigger_retraining_dag

logger = get_logger(__name__)

DEFAULT_MIN_SAMPLES = 100
DEFAULT_COOLDOWN_SECONDS = 3600
TRIGGER_STATE_FILE = "retrain_trigger_state.json"


class Outcome:
    """What a detection pass did. Lets a one-shot run (k8s CronJob) tell "nothing to do"
    apart from "cannot do its job", which would otherwise both look like success."""

    NO_REFERENCE = "no_reference"
    NO_NEW_LOGS = "no_new_logs"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    COLUMN_MISMATCH = "column_mismatch"
    ANALYSIS_FAILED = "analysis_failed"
    NO_DRIFT = "no_drift"
    DRIFT_DETECTED = "drift_detected"
    TRIGGER_FAILED = "trigger_failed"


# Outcomes meaning the detector is unable to work, as opposed to having nothing to report.
FAILURE_OUTCOMES = {
    Outcome.NO_REFERENCE,
    Outcome.COLUMN_MISMATCH,
    Outcome.ANALYSIS_FAILED,
    Outcome.TRIGGER_FAILED,
}


def load_drift_config(config_path: str = "configs/drift.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_unprocessed_logs(limit: int) -> tuple[pd.DataFrame, list[Any]]:
    """Oldest un-analyzed prediction logs. Read-only: rows are marked separately, once
    they have actually been analyzed, so a crash cannot silently discard a window."""
    with db_session() as session:
        stmt = (
            select(PredictionLog)
            .where(PredictionLog.drift_analyzed.is_(False))
            .order_by(PredictionLog.created_at.asc())
            .limit(limit)
        )
        logs = session.scalars(stmt).all()
        if not logs:
            return pd.DataFrame(), []
        return pd.DataFrame([log.features for log in logs]), [log.id for log in logs]


def mark_logs_analyzed(log_ids: list[Any]) -> None:
    if not log_ids:
        return
    with db_session() as session:
        session.execute(
            update(PredictionLog)
            .where(PredictionLog.id.in_(log_ids))
            .values(drift_analyzed=True)
        )


def extract_drift_summary(report_dict: dict[str, Any]) -> dict[str, Any]:
    """Observed drift from an Evidently DataDriftPreset report.

    Evidently's `drift_share` field is the *threshold* the preset was configured with
    (0.5 by default), not the measured value. The measured share of drifted columns is
    `share_of_drifted_columns`.
    """
    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})
        if "share_of_drifted_columns" in result:
            return {
                "share_of_drifted_columns": float(result["share_of_drifted_columns"]),
                "number_of_drifted_columns": int(result["number_of_drifted_columns"]),
                "number_of_columns": int(result["number_of_columns"]),
                "dataset_drift": bool(result["dataset_drift"]),
            }
    raise KeyError("share_of_drifted_columns not found in Evidently report")


def _last_trigger_time(state_path: Path) -> float | None:
    try:
        return float(json.loads(state_path.read_text())["last_triggered_at"])
    except (FileNotFoundError, KeyError, ValueError, TypeError):
        return None


def _in_cooldown(state_path: Path, cooldown_seconds: float, now: float) -> bool:
    if cooldown_seconds <= 0:
        return False
    last = _last_trigger_time(state_path)
    return last is not None and (now - last) < cooldown_seconds


def _record_trigger(state_path: Path, now: float) -> None:
    state_path.write_text(json.dumps({"last_triggered_at": now}))


def run_drift_detection(config_path: str = "configs/drift.yaml") -> str:
    """Run one detection pass and return an `Outcome`."""
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
        return Outcome.NO_REFERENCE

    ref_df = pd.read_parquet(ref_path)

    window_size = det_cfg["window_size"]
    curr_df, log_ids = fetch_unprocessed_logs(limit=window_size)
    if curr_df.empty:
        logger.info("no_new_logs_for_drift_analysis")
        return Outcome.NO_NEW_LOGS

    min_samples = min(det_cfg.get("min_samples", DEFAULT_MIN_SAMPLES), window_size)
    if len(curr_df) < min_samples:
        # Leave the rows un-analyzed so they accumulate into a statistically useful window.
        logger.info(
            "insufficient_samples_for_drift_analysis",
            available=len(curr_df),
            required=min_samples,
        )
        return Outcome.INSUFFICIENT_SAMPLES

    ref_features = [c for c in ref_df.columns if c in curr_df.columns]
    if not ref_features:
        logger.error("dataframe_column_mismatch_cannot_run_drift")
        # These rows can never be analyzed against this reference; don't let them block the queue.
        mark_logs_analyzed(log_ids)
        return Outcome.COLUMN_MISMATCH
    ref_df_clean = ref_df[ref_features]
    curr_df_clean = curr_df[ref_features]

    threshold = det_cfg["drift_share_threshold"]
    logger.info(
        "running_evidently",
        reference_rows=len(ref_df_clean),
        current_rows=len(curr_df_clean),
    )

    preset_kwargs: dict[str, Any] = {"drift_share": threshold}
    if det_cfg.get("pvalue_threshold") is not None:
        preset_kwargs["stattest_threshold"] = det_cfg["pvalue_threshold"]
    drift_report = Report(metrics=[DataDriftPreset(**preset_kwargs)])
    drift_report.run(reference_data=ref_df_clean, current_data=curr_df_clean)

    try:
        summary = extract_drift_summary(drift_report.as_dict())
    except (KeyError, IndexError, TypeError, ValueError) as e:
        # Rows stay un-analyzed: this is a detector fault, not a property of the data.
        logger.error("evidently_metric_extraction_failed", error=str(e))
        return Outcome.ANALYSIS_FAILED

    drift_share = summary["share_of_drifted_columns"]
    breached = drift_share >= threshold
    logger.info(
        "drift_analysis_complete",
        drift_share=round(drift_share, 4),
        threshold=threshold,
        dataset_drift=summary["dataset_drift"],
        drifted_features=summary["number_of_drifted_columns"],
        breached=breached,
    )

    out_dir = Path(config["evidently"]["report_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "latest_drift_report.html"
    if breached:
        drift_report.save_html(str(report_path))

    # Persisted every run so the status CLI can read it.
    metrics_payload = {
        "drift_share": drift_share,
        "drift_share_threshold": threshold,
        "dataset_drift": summary["dataset_drift"],
        "drifted_features": summary["number_of_drifted_columns"],
        "number_of_columns": summary["number_of_columns"],
        "analyzed_rows": len(curr_df_clean),
        "report_path": str(report_path) if breached else None,
        "checked_at": time.time(),
    }
    metrics_json_path = out_dir / "latest_drift_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("drift_metrics_json_saved", path=str(metrics_json_path))

    mark_logs_analyzed(log_ids)

    if not breached:
        return Outcome.NO_DRIFT

    logger.warning(
        "data_drift_detected_threshold_breached",
        drift_share=drift_share,
        threshold=threshold,
    )

    state_path = out_dir / TRIGGER_STATE_FILE
    cooldown = det_cfg.get("retrain_cooldown_seconds", DEFAULT_COOLDOWN_SECONDS)
    now = time.time()
    if _in_cooldown(state_path, cooldown, now):
        logger.info(
            "retrain_trigger_suppressed_cooldown",
            cooldown_seconds=cooldown,
            seconds_since_last=round(now - (_last_trigger_time(state_path) or now), 1),
        )
        return Outcome.DRIFT_DETECTED

    if config.get("alerting", {}).get("trigger_airflow", False):
        try:
            trigger_retraining_dag(metrics_payload)
        except Exception as exc:
            # No cooldown is recorded, so the next window that still shows drift retries.
            logger.error("retrain_trigger_failed", error=str(exc))
            return Outcome.TRIGGER_FAILED
        _record_trigger(state_path, now)

    send_slack_alert(drift_metrics=metrics_payload, config=config)
    return Outcome.DRIFT_DETECTED


def main(argv: list[str] | None = None) -> int:
    """`--once`: one pass then exit (for a k8s CronJob / cron). Default: run as a daemon.

    Exit codes for --once: 0 = ran (including "nothing to analyze"), 1 = crashed,
    2 = the detector could not do its job (no reference data, cannot analyze, or cannot
    reach Airflow) -- so a scheduler shows the run as failed instead of silently green.
    """
    parser = argparse.ArgumentParser(description="Data drift detector")
    parser.add_argument("--once", action="store_true", help="run a single pass and exit")
    parser.add_argument("--config", default="configs/drift.yaml")
    args = parser.parse_args(argv)

    from src.core.logging import configure_logging
    configure_logging()

    if args.once:
        try:
            outcome = run_drift_detection(args.config)
        except Exception as exc:
            logger.error("drift_detector_crash", error=str(exc), exc_info=True)
            return 1
        logger.info("drift_detection_finished", outcome=outcome)
        return 2 if outcome in FAILURE_OUTCOMES else 0

    interval = load_drift_config(args.config)["detection"]["check_interval_seconds"]
    logger.info("starting_drift_detector_daemon", interval=interval)
    while True:
        try:
            run_drift_detection(args.config)
        except Exception as exc:
            logger.error("drift_detector_crash", error=str(exc), exc_info=True)
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
