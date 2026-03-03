"""
cli/status.py

System-wide status dashboard for the AutoMLOps platform.

Usage:
    python -m src.cli.status

Prints a rich terminal snapshot of:
  - MLflow model registry (Production + Staging versions)
  - Inference API health (HTTP GET /health)
  - Recent prediction logs (last 5 rows from Postgres)
  - Latest drift metrics (from data/drift_reports/latest_drift_metrics.json)

All sources are fetched independently — partial failures are shown as
"UNAVAILABLE" rather than crashing the dashboard.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# ── Colour helpers (ANSI) ──────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"


def _col(text: str, colour: str) -> str:
    return f"{colour}{text}{_RESET}"


def _header(title: str) -> None:
    width = 60
    bar = "─" * width
    print(f"\n{_BOLD}{_CYAN}{bar}{_RESET}")
    print(f"{_BOLD}{_CYAN}  {title}{_RESET}")
    print(f"{_BOLD}{_CYAN}{bar}{_RESET}")


def _row(label: str, value: str, value_colour: str = _RESET) -> None:
    print(f"  {_DIM}{label:<24}{_RESET} {value_colour}{value}{_RESET}")


# ── Data fetchers ──────────────────────────────────────────────────────────────

def _fetch_model_registry() -> dict[str, Any]:
    """Pull Production and Staging model info from MLflow."""
    result: dict[str, Any] = {}
    try:
        import mlflow
        from src.model_registry.registry import (
            ModelRegistry,
            STAGE_PRODUCTION,
            STAGE_STAGING,
            ModelNotFoundError,
        )
        from src.core.exceptions import ModelNotFoundError

        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        registry = ModelRegistry()

        for stage in [STAGE_PRODUCTION, STAGE_STAGING]:
            try:
                info = registry.get_model_info(stage=stage)
                result[stage] = {
                    "version": info.version,
                    "auc": info.metrics.get("roc_auc", "N/A"),
                    "run_id": info.run_id[:8] + "…",
                }
            except Exception:
                result[stage] = None
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _fetch_api_health(base_url: str) -> dict[str, Any]:
    """GET /health from the Inference API."""
    try:
        response = httpx.get(f"{base_url}/health", timeout=3.0)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc)}


def _fetch_recent_logs(n: int = 5) -> list[dict] | str:
    """Fetch the last N prediction logs from Postgres."""
    try:
        from sqlalchemy import select, desc
        from src.core.db import db_session
        from src.data_logger.models import PredictionLog

        with db_session() as session:
            stmt = (
                select(PredictionLog)
                .order_by(desc(PredictionLog.created_at))
                .limit(n)
            )
            logs = session.scalars(stmt).all()
            return [
                {
                    "request_id": l.request_id[:8] + "…",
                    "prediction": l.prediction,
                    "probability": round(l.probability, 3),
                    "model_version": l.model_version,
                    "created_at": str(l.created_at)[:19] if l.created_at else "N/A",
                }
                for l in logs
            ]
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def _fetch_drift_metrics(report_dir: str = "data/drift_reports") -> dict | str:
    """Read the latest drift metrics JSON written by the detector."""
    metrics_path = Path(report_dir) / "latest_drift_metrics.json"
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return "No drift analysis has run yet."
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


# ── Display functions ──────────────────────────────────────────────────────────

def _show_model_registry(data: dict) -> None:
    _header("🤖  Model Registry")
    if "error" in data:
        _row("Error", data["error"], _RED)
        return

    for stage in ["Production", "Staging"]:
        info = data.get(stage)
        if info:
            _row(f"{stage} Version", info["version"], _GREEN)
            auc = info["auc"]
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
            _row(f"{stage} AUC", auc_str)
            _row(f"{stage} Run ID", info["run_id"], _DIM)
        else:
            _row(stage, "None registered", _YELLOW)


def _show_api_health(data: dict) -> None:
    _header("🌐  Inference API")
    if "error" in data:
        _row("Status", "UNREACHABLE", _RED)
        _row("Error", data["error"][:60], _RED)
        return

    status_colour = _GREEN if data.get("status") == "ok" else _YELLOW
    _row("Status", data.get("status", "unknown").upper(), status_colour)
    _row("Model Loaded", str(data.get("model_loaded", "N/A")))
    _row("Model Version", str(data.get("model_version", "N/A")))
    _row("Uptime", f"{data.get('uptime_seconds', 0):.0f}s")


def _show_recent_logs(data: list | str) -> None:
    _header("📝  Last 5 Predictions")
    if isinstance(data, str):
        _row("Status", data, _YELLOW)
        return

    if not data:
        _row("Status", "No predictions logged yet", _YELLOW)
        return

    print(f"  {'Request ID':<12} {'Pred':>4} {'Prob':>6} {'Version':>8}  {'Time'}")
    print(f"  {'─'*11:<12} {'─'*4:>4} {'─'*6:>6} {'─'*8:>8}  {'─'*19}")
    for log in data:
        pred_colour = _RED if log["prediction"] == 1 else _GREEN
        print(
            f"  {log['request_id']:<12} "
            f"{_col(str(log['prediction']), pred_colour):>4} "
            f"{log['probability']:>6.3f} "
            f"{log['model_version']:>8}  "
            f"{_DIM}{log['created_at']}{_RESET}"
        )


def _show_drift(data: dict | str) -> None:
    _header("🌊  Latest Drift Analysis")
    if isinstance(data, str):
        _row("Status", data, _YELLOW)
        return

    drift_share = data.get("drift_share", "N/A")
    if isinstance(drift_share, float):
        drift_colour = _RED if drift_share >= 0.3 else _GREEN
        ds_str = f"{drift_share:.4f}"
    else:
        drift_colour = _YELLOW
        ds_str = str(drift_share)

    _row("Drift Share", ds_str, drift_colour)
    _row("Drifted Features", str(data.get("drifted_features", "N/A")))
    _row("Analyzed Rows", str(data.get("analyzed_rows", "N/A")))
    _row("Report", str(data.get("report_path", "N/A")), _DIM)


# ── Main ───────────────────────────────────────────────────────────────────────

def print_status() -> None:
    settings = get_settings()
    api_base = f"http://{settings.inference_api_host}:{settings.inference_api_port}"

    print(f"\n{_BOLD}AutoMLOps — Platform Status{_RESET}  {_DIM}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{_RESET}")

    registry_data = _fetch_model_registry()
    api_data      = _fetch_api_health(api_base)
    log_data      = _fetch_recent_logs(5)
    drift_data    = _fetch_drift_metrics()

    _show_model_registry(registry_data)
    _show_api_health(api_data)
    _show_recent_logs(log_data)
    _show_drift(drift_data)
    print()


if __name__ == "__main__":
    try:
        print_status()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)
