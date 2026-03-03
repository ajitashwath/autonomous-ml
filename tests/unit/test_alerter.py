"""Unit tests for drift_detector.alerter"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.drift_detector.alerter import trigger_retraining_dag


def test_trigger_retraining_dag_success():
    payload = {"drift_share": 0.45}
    
    with patch("src.drift_detector.alerter.httpx.post") as mock_post, \
         patch("src.drift_detector.alerter.get_settings") as mock_settings:
        
        mock_settings.return_value.airflow_host = "http://airflow:8080"
        mock_settings.return_value.airflow_retrain_dag_id = "retrain_churn_model"
        mock_settings.return_value.airflow_username = "admin"
        mock_settings.return_value.airflow_password = "admin"
        
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"dag_run_id": "manual__2024"}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp
        
        run_id = trigger_retraining_dag(payload)
        
        assert run_id == "manual__2024"
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["auth"] == ("admin", "admin")
        assert kwargs["json"]["conf"]["drift_metrics"]["drift_share"] == 0.45


def test_trigger_retraining_dag_failure_raises():
    payload = {"drift_share": 0.45}
    
    with patch("src.drift_detector.alerter.httpx.post") as mock_post, \
         patch("src.drift_detector.alerter.get_settings"):
        
        mock_post.side_effect = httpx.RequestError("Network unreachable")
        
        with pytest.raises(httpx.RequestError):
            trigger_retraining_dag(payload)


# ── Slack alert tests ──────────────────────────────────────────────────────────

class TestSlackAlert:
    """Tests for send_slack_alert()."""

    from src.drift_detector.alerter import send_slack_alert

    _DRIFT_METRICS = {
        "drift_share": 0.52,
        "drifted_features": 3,
        "analyzed_rows": 1000,
        "report_path": "/data/drift_reports/latest.html",
    }

    _CONFIG_ENABLED = {
        "detection": {"drift_share_threshold": 0.3},
        "alerting": {
            "slack_enabled": True,
            "slack_webhook_url": "https://hooks.slack.com/services/fake/webhook",
        },
    }

    _CONFIG_DISABLED = {
        "detection": {"drift_share_threshold": 0.3},
        "alerting": {
            "slack_enabled": False,
            "slack_webhook_url": "",
        },
    }

    def test_slack_alert_sends_when_configured(self):
        from src.drift_detector.alerter import send_slack_alert
        with patch("src.drift_detector.alerter.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp

            send_slack_alert(self._DRIFT_METRICS, self._CONFIG_ENABLED)

            mock_post.assert_called_once()
            call_args = mock_post.call_args
            # httpx.post(url, content=...) → URL is first positional arg
            posted_url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
            assert "hooks.slack.com" in posted_url
            # Confirm the message body contains the drift share
            body_bytes = call_args.kwargs.get("content", b"{}")
            body_str = body_bytes if isinstance(body_bytes, str) else body_bytes.decode()
            assert "0.52" in body_str or "drift" in body_str.lower()


    def test_slack_alert_skipped_when_disabled(self):
        from src.drift_detector.alerter import send_slack_alert
        with patch("src.drift_detector.alerter.httpx.post") as mock_post:
            send_slack_alert(self._DRIFT_METRICS, self._CONFIG_DISABLED)
            mock_post.assert_not_called()

    def test_slack_alert_skipped_when_no_url(self):
        from src.drift_detector.alerter import send_slack_alert
        config = {
            "detection": {"drift_share_threshold": 0.3},
            "alerting": {
                "slack_enabled": True,
                "slack_webhook_url": "",  # empty URL → skip
            },
        }
        with patch("src.drift_detector.alerter.httpx.post") as mock_post:
            send_slack_alert(self._DRIFT_METRICS, config)
            mock_post.assert_not_called()

    def test_slack_alert_failure_does_not_raise(self):
        """HTTP errors from Slack must NOT propagate — detector must keep running."""
        from src.drift_detector.alerter import send_slack_alert
        with patch("src.drift_detector.alerter.httpx.post") as mock_post:
            mock_post.side_effect = Exception("Connection failed")
            # Should not raise
            send_slack_alert(self._DRIFT_METRICS, self._CONFIG_ENABLED)

