

from prometheus_client import Counter, Gauge, Histogram


PREDICTIONS_TOTAL = Counter(
    name="automlops_predictions_total",
    documentation="Total number of predictions served",
    labelnames=["prediction_label", "model_version"],
)

MODEL_LOAD_TOTAL = Counter(
    name="automlops_model_load_total",
    documentation="Number of model load attempts",
    labelnames=["status"],
)

ERRORS_TOTAL = Counter(
    name="automlops_errors_total",
    documentation="Total prediction errors by error type",
    labelnames=["error_type"],
)


REQUEST_LATENCY = Histogram(
    name="automlops_request_latency_seconds",
    documentation="End-to-end prediction request latency",
    labelnames=["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

PREDICTION_PROBABILITY = Histogram(
    name="automlops_prediction_probability",
    documentation="Distribution of predicted churn probabilities",
    labelnames=["model_version"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


ACTIVE_REQUESTS = Gauge(
    name="automlops_active_requests",
    documentation="Number of prediction requests currently being processed",
)

MODEL_INFO = Gauge(
    name="automlops_model_info",
    documentation="Currently loaded model metadata (always 1 if model is loaded)",
    labelnames=["model_version", "model_stage"],
)