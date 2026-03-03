"""
Integration tests for inference_api.
Uses FastAPI TestClient with a mocked ModelLoader — no MLflow server needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.inference_api.main import create_app
from src.model_registry.registry import ModelInfo

# ── Sample valid request payload ───────────────────────────────────────────────
VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.35,
    "TotalCharges": 904.20,
}


def _make_mock_loader(proba: float = 0.75) -> MagicMock:
    """Return a mock ModelLoader that returns a fixed probability."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[1 - proba, proba]])

    mock_info = ModelInfo(
        name="churn_classifier",
        version="3",
        stage="Production",
        run_id="run-abc",
        run_link="http://mlflow:5000",
        metrics={"roc_auc": 0.85},
        params={"threshold": "0.5"},
    )

    loader = MagicMock()
    loader.is_loaded.return_value = True
    loader.get_model.return_value = mock_model
    loader.get_info.return_value = mock_info
    loader.start_polling.return_value = None
    loader.stop_polling.return_value = None
    return loader


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """TestClient with a fully mocked model loader."""
    loader = _make_mock_loader(proba=0.75)
    with patch("src.inference_api.main.get_model_loader", return_value=loader), \
         patch("src.inference_api.main.create_all_tables"), \
         patch("src.core.db.check_db_connection", return_value=True):
        app = create_app()
        app.state.model_loader = loader
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health endpoint ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_200_when_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "3"

    def test_health_response_schema(self, client):
        response = client.get("/health")
        body = response.json()
        for key in ["status", "model_loaded", "model_version", "model_stage", "uptime_seconds"]:
            assert key in body, f"Missing key: {key}"


# ── Predict endpoint ───────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client):
        response = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_prediction_response_schema(self, client):
        response = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        body = response.json()
        assert "prediction" in body
        assert "probability" in body
        assert "model_version" in body
        assert body["prediction"] in [0, 1]
        assert 0.0 <= body["probability"] <= 1.0

    def test_high_probability_predicts_churn(self, client):
        # proba=0.75 > threshold=0.5 → prediction=1
        response = client.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert response.json()["prediction"] == 1

    def test_missing_required_field_returns_422(self, client):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
        response = client.post("/api/v1/predict", json=bad_payload)
        assert response.status_code == 422

    def test_invalid_enum_value_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "Contract": "Weekly"}  # not a valid enum
        response = client.post("/api/v1/predict", json=bad_payload)
        assert response.status_code == 422

    def test_negative_tenure_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "tenure": -5}
        response = client.post("/api/v1/predict", json=bad_payload)
        assert response.status_code == 422

    def test_metrics_endpoint_accessible(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"automlops_predictions_total" in response.content


# ── Batch predict endpoint ─────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        payload = {"requests": [VALID_PAYLOAD, VALID_PAYLOAD]}
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 200

    def test_batch_predict_response_count_matches_input(self, client):
        batch_size = 3
        payload = {"requests": [VALID_PAYLOAD] * batch_size}
        response = client.post("/api/v1/predict/batch", json=payload)
        body = response.json()
        assert body["count"] == batch_size
        assert len(body["predictions"]) == batch_size

    def test_batch_predict_response_schema(self, client):
        payload = {"requests": [VALID_PAYLOAD]}
        response = client.post("/api/v1/predict/batch", json=payload)
        body = response.json()
        assert "predictions" in body
        assert "count" in body
        assert "model_version" in body
        # Each prediction item should match the single-predict schema
        item = body["predictions"][0]
        assert "prediction" in item
        assert "probability" in item
        assert item["prediction"] in [0, 1]
        assert 0.0 <= item["probability"] <= 1.0

    def test_batch_predict_rejects_empty_list(self, client):
        payload = {"requests": []}
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 422  # Pydantic min_length=1

    def test_batch_predict_rejects_oversized_batch(self, client):
        # BATCH_MAX_SIZE is 500; send 501
        payload = {"requests": [VALID_PAYLOAD] * 501}
        response = client.post("/api/v1/predict/batch", json=payload)
        assert response.status_code == 413

    def test_batch_predict_all_churners_when_high_proba(self, client):
        # The fixture uses proba=0.75 > threshold=0.5 → all predict 1
        payload = {"requests": [VALID_PAYLOAD, VALID_PAYLOAD, VALID_PAYLOAD]}
        response = client.post("/api/v1/predict/batch", json=payload)
        body = response.json()
        for pred in body["predictions"]:
            assert pred["prediction"] == 1

