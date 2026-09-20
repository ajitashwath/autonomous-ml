"""Inference API driven through HTTP with a real fitted preprocessor + XGBoost model.

Only the model *registry/loader* and the database are stubbed; preprocessing, prediction,
schemas, metrics and background logging all run for real.
"""
from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core.exceptions import ModelLoadError
from src.inference_api.main import create_app
from src.inference_api.model_loader import ModelSnapshot
from src.inference_api.routers import predict as predict_module
from src.model_registry.registry import ModelInfo
from tests.conftest import CHURN_PAYLOAD, NO_CHURN_PAYLOAD

VALID_PAYLOAD = CHURN_PAYLOAD


def _info(version: str = "3") -> ModelInfo:
    return ModelInfo(
        name="churn_classifier", version=version, stage="Production", run_id="run-abc",
        run_link="http://mlflow:5000", metrics={"roc_auc": 0.85}, params={"threshold": "0.5"},
    )


def _loader_for(snapshot: ModelSnapshot | None) -> MagicMock:
    loader = MagicMock()
    if snapshot is None:
        loader.get_snapshot.side_effect = ModelLoadError("Model is not loaded. Check startup logs.")
    else:
        loader.get_snapshot.return_value = snapshot
    loader.current.return_value = snapshot
    return loader


@pytest.fixture
def snapshot(churn_model_bundle) -> ModelSnapshot:
    preprocessor, model = churn_model_bundle
    return ModelSnapshot(model=model, preprocessor=preprocessor, info=_info())


@pytest.fixture
def logged():
    """Captures what the background logger receives instead of touching a database."""
    with patch.object(predict_module, "log_predictions_safe") as mock:
        yield mock


def _make_app(loader: MagicMock):
    """App for ASGI-transport tests (which do not run the lifespan)."""
    app = create_app()
    app.state.model_loader = loader
    return app


@contextmanager
def _serving(loader: MagicMock, *, db_ok: bool = True):
    """TestClient with the lifespan's external dependencies stubbed for as long as it runs."""
    with patch("src.inference_api.main.get_model_loader", return_value=loader), \
         patch("src.inference_api.main.create_all_tables"), \
         patch("src.inference_api.routers.health.check_db_connection", return_value=db_ok):
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def client(snapshot, logged):
    with _serving(_loader_for(snapshot)) as c:
        yield c


class TestHealthEndpoint:
    def test_returns_200_when_model_loaded(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "3"

    def test_health_response_schema(self, client):
        body = client.get("/health").json()
        for key in ["status", "model_loaded", "model_version", "model_stage", "uptime_seconds"]:
            assert key in body, f"Missing key: {key}"

    def test_degraded_when_database_is_down(self, snapshot, logged):
        with _serving(_loader_for(snapshot), db_ok=False) as c:
            response = c.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "degraded"

    def test_degraded_when_no_model_is_loaded(self, logged):
        with _serving(_loader_for(None)) as c:
            response = c.get("/health")
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client):
        assert client.post("/api/v1/predict", json=VALID_PAYLOAD).status_code == 200

    def test_prediction_response_schema(self, client):
        body = client.post("/api/v1/predict", json=VALID_PAYLOAD).json()
        assert body["prediction"] in [0, 1]
        assert 0.0 <= body["probability"] <= 1.0
        assert body["model_version"] == "3"
        assert body["model_stage"] == "Production"
        assert body["threshold"] == 0.5

    def test_model_separates_churners_from_non_churners_through_the_real_pipeline(self, client):
        assert client.post("/api/v1/predict", json=CHURN_PAYLOAD).json()["prediction"] == 1
        assert client.post("/api/v1/predict", json=NO_CHURN_PAYLOAD).json()["prediction"] == 0

    def test_missing_required_field_returns_422(self, client):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
        assert client.post("/api/v1/predict", json=bad_payload).status_code == 422

    def test_invalid_enum_value_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "Contract": "Weekly"}
        assert client.post("/api/v1/predict", json=bad_payload).status_code == 422

    def test_negative_tenure_returns_422(self, client):
        bad_payload = {**VALID_PAYLOAD, "tenure": -5}
        assert client.post("/api/v1/predict", json=bad_payload).status_code == 422

    def test_metrics_endpoint_accessible(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert b"automlops_predictions_total" in response.content

    def test_returns_503_when_no_model_is_loaded(self, logged):
        with _serving(_loader_for(None)) as c:
            response = c.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503
        logged.assert_not_called()

    def test_preprocessing_failure_returns_500_and_logs_nothing(self, snapshot, logged):
        broken = MagicMock()
        broken.transform.side_effect = ValueError("bad column")
        with _serving(_loader_for(ModelSnapshot(snapshot.model, broken, snapshot.info))) as c:
            response = c.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert response.status_code == 500
        assert "preprocessing failed" in response.json()["detail"]
        logged.assert_not_called()


class TestPredictionLogging:
    def test_one_record_is_logged_with_json_safe_features(self, client, logged):
        client.post("/api/v1/predict", json=VALID_PAYLOAD)

        logged.assert_called_once()
        (records,), _ = logged.call_args
        assert len(records) == 1
        record = records[0]
        assert record["model_version"] == "3"
        assert record["prediction"] == 1
        # Enums must be stored as plain strings, matching the drift reference data.
        assert record["features"]["Contract"] == "Month-to-month"
        assert type(record["features"]["Contract"]) is str
        assert set(record["features"]) == set(VALID_PAYLOAD)

    def test_request_succeeds_when_the_database_is_unavailable(self, snapshot):
        """The real logger runs (as a background task) against a failing database."""
        with patch("src.data_logger.logger.db_session", side_effect=RuntimeError("db down")), \
             _serving(_loader_for(snapshot)) as c:
            response = c.post("/api/v1/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_batch_is_logged_as_a_single_call_with_every_record(self, client, logged):
        client.post("/api/v1/predict/batch", json={"requests": [CHURN_PAYLOAD] * 4})

        logged.assert_called_once()
        (records,), _ = logged.call_args
        assert len(records) == 4
        assert len({r["request_id"] for r in records}) == 4


class TestBatchPredictEndpoint:
    def test_batch_predict_returns_200(self, client):
        payload = {"requests": [VALID_PAYLOAD, VALID_PAYLOAD]}
        assert client.post("/api/v1/predict/batch", json=payload).status_code == 200

    def test_batch_predict_response_count_matches_input(self, client):
        batch_size = 3
        body = client.post("/api/v1/predict/batch", json={"requests": [VALID_PAYLOAD] * batch_size}).json()
        assert body["count"] == batch_size
        assert len(body["predictions"]) == batch_size

    def test_batch_predictions_keep_input_order(self, client):
        payload = {"requests": [CHURN_PAYLOAD, NO_CHURN_PAYLOAD, CHURN_PAYLOAD]}
        body = client.post("/api/v1/predict/batch", json=payload).json()
        assert [p["prediction"] for p in body["predictions"]] == [1, 0, 1]

    def test_batch_predict_response_schema(self, client):
        body = client.post("/api/v1/predict/batch", json={"requests": [VALID_PAYLOAD]}).json()
        assert "predictions" in body and "count" in body and "model_version" in body
        item = body["predictions"][0]
        assert item["prediction"] in [0, 1]
        assert 0.0 <= item["probability"] <= 1.0

    def test_batch_predict_rejects_empty_list(self, client):
        assert client.post("/api/v1/predict/batch", json={"requests": []}).status_code == 422

    def test_batch_predict_rejects_oversized_batch(self, client):
        payload = {"requests": [VALID_PAYLOAD] * 501}
        assert client.post("/api/v1/predict/batch", json=payload).status_code == 413

    def test_batch_with_wrong_number_of_model_outputs_is_an_error_not_a_truncated_reply(
        self, snapshot, logged
    ):
        """Regression: zip() silently truncated the response when the model returned fewer rows."""
        short_model = MagicMock()
        short_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 1 row for 3 inputs
        loader = _loader_for(ModelSnapshot(short_model, snapshot.preprocessor, snapshot.info))
        with _serving(loader) as c:
            response = c.post("/api/v1/predict/batch", json={"requests": [VALID_PAYLOAD] * 3})
        assert response.status_code == 500
        logged.assert_not_called()


class _SlowModel:
    """Delegates to a real model after a blocking delay, like a heavy synchronous predict."""

    def __init__(self, model, delay: float):
        self._model, self._delay = model, delay

    def predict_proba(self, X):
        time.sleep(self._delay)
        return self._model.predict_proba(X)


class TestEventLoopIsNotBlocked:
    """Regression: `async def` handlers ran CPU-bound inference on the event loop, so
    requests serialised and /health stalled behind them."""

    DELAY = 0.3

    def _app(self, snapshot):
        slow = ModelSnapshot(_SlowModel(snapshot.model, self.DELAY), snapshot.preprocessor, snapshot.info)
        return _make_app(_loader_for(slow))

    @pytest.mark.asyncio
    async def test_concurrent_predictions_overlap(self, snapshot, logged):
        app = self._app(snapshot)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
            start = time.perf_counter()
            responses = await asyncio.gather(
                *[c.post("/api/v1/predict", json=VALID_PAYLOAD) for _ in range(4)]
            )
            elapsed = time.perf_counter() - start

        assert all(r.status_code == 200 for r in responses)
        # Serialised on the event loop this would take >= 4 * DELAY (1.2s).
        assert elapsed < 3 * self.DELAY

    @pytest.mark.asyncio
    async def test_health_answers_while_predictions_are_running(self, snapshot, logged):
        app = self._app(snapshot)
        with patch("src.inference_api.routers.health.check_db_connection", return_value=True):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                # The clock starts before we yield to the loop: if inference blocks the loop, even
                # sleep() and the health request cannot proceed until it finishes.
                start = time.perf_counter()
                predictions = [asyncio.create_task(c.post("/api/v1/predict", json=VALID_PAYLOAD)) for _ in range(3)]
                await asyncio.sleep(0.05)  # let the predictions start
                health = await c.get("/health")
                health_answered_after = time.perf_counter() - start
                await asyncio.gather(*predictions)

        assert health.status_code == 200
        # Blocked on the event loop this cannot happen before all three predictions (~0.9s) finish.
        assert health_answered_after < self.DELAY
