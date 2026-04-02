
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestStatusHandlesFailures:

    def test_handles_unreachable_api_gracefully(self, capsys):
        with patch("src.cli.status.fetch_api_health") as mock_api, \
             patch("src.cli.status.fetch_model_registry") as mock_reg, \
             patch("src.cli.status.fetch_recent_logs") as mock_logs, \
             patch("src.cli.status.fetch_drift_metrics") as mock_drift:

            mock_api.return_value = {"error": "Connection refused"}
            mock_reg.return_value = {}
            mock_logs.return_value = []
            mock_drift.return_value = "No drift analysis has run yet."

            from src.cli.status import print_status
            print_status()

            out = capsys.readouterr().out
            assert "UNREACHABLE" in out

    def test_handles_no_production_model(self, capsys):
        with patch("src.cli.status.fetch_api_health") as mock_api, \
             patch("src.cli.status.fetch_model_registry") as mock_reg, \
             patch("src.cli.status.fetch_recent_logs") as mock_logs, \
             patch("src.cli.status.fetch_drift_metrics") as mock_drift:

            mock_api.return_value = {"status": "ok", "model_loaded": False,
                                      "model_version": None, "uptime_seconds": 5.0}
            mock_reg.return_value = {"Production": None, "Staging": None}
            mock_logs.return_value = []
            mock_drift.return_value = "No drift analysis has run yet."

            from src.cli.status import print_status
            print_status()

            out = capsys.readouterr().out
            assert "None registered" in out

    def test_handles_drift_data_present(self, capsys):
        with patch("src.cli.status.fetch_api_health") as mock_api, \
             patch("src.cli.status.fetch_model_registry") as mock_reg, \
             patch("src.cli.status.fetch_recent_logs") as mock_logs, \
             patch("src.cli.status.fetch_drift_metrics") as mock_drift:

            mock_api.return_value = {"status": "ok", "model_loaded": True,
                                      "model_version": "3", "uptime_seconds": 120.0}
            mock_reg.return_value = {
                "Production": {"version": "3", "auc": 0.85, "run_id": "abc12345…"},
                "Staging": None,
            }
            mock_logs.return_value = [
                {"request_id": "abc12345…", "prediction": 1, "probability": 0.76,
                 "model_version": "3", "created_at": "2026-03-03 14:00:00"},
            ]
            mock_drift.return_value = {
                "drift_share": 0.42,
                "drifted_features": 3,
                "analyzed_rows": 500,
                "report_path": "/data/drift_reports/latest.html",
            }

            from src.cli.status import print_status
            print_status()

            out = capsys.readouterr().out
            assert "0.4200" in out
            assert "Production" in out

    def test_handles_db_error_gracefully(self, capsys):
        with patch("src.cli.status.fetch_api_health") as mock_api, \
             patch("src.cli.status.fetch_model_registry") as mock_reg, \
             patch("src.cli.status.fetch_recent_logs") as mock_logs, \
             patch("src.cli.status.fetch_drift_metrics") as mock_drift:

            mock_api.return_value = {"status": "ok", "model_loaded": True,
                                      "model_version": "1", "uptime_seconds": 0.0}
            mock_reg.return_value = {}
            mock_logs.return_value = "UNAVAILABLE: could not connect"
            mock_drift.return_value = {}

            from src.cli.status import print_status
            print_status()

            out = capsys.readouterr().out
            assert "UNAVAILABLE" in out