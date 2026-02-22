"""Unit tests for drift_detector.alerter"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.drift_detector.alerter import trigger_retraining_dag


def test_trigger_retraining_dag_success():
    payload = {"drift_share": 0.45}
    
    with patch("src.drift_detector.alerter.httpx.post") as mock_post, \
         patch("src.drift_detector.alerter.get_settings") as mock_settings:
        
        mock_settings.return_value.airflow_api_url = "http://airflow:8080"
        mock_settings.return_value.airflow_retrain_dag_id = "retrain_churn_model"
        mock_settings.return_value.airflow_api_user = "admin"
        mock_settings.return_value.airflow_api_password = "admin"
        
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
        
        from tenacity import RetryError
        with pytest.raises(RetryError) as exc_info:
            trigger_retraining_dag(payload)
            
        # Verify the actual error that caused tenacity to fail
        assert isinstance(exc_info.value.last_attempt.exception(), httpx.RequestError)
