# Automated ML Monitoring Pipeline
It autonomously manages the full ML lifecycle: from training and deployment to monitoring, drift detection, statistical validation, and emergency rollbacks.

## Architecture Highlights
- **FastAPI Inference** with zero-downtime background model hot-swapping. Each request is served by one consistent snapshot of model + preprocessor + version, so a swap never mixes them.
- **Evidently AI Drift Detection** pulling traffic directly from PostgreSQL, with a minimum sample size and a retrain cooldown so noise doesn't trigger retraining.
- **Event-Driven Airflow DAGs** triggered via REST API upon drift.
- **Production feedback loop**: report real outcomes with `POST /api/v1/labels`; labelled predictions become training data and the evaluation set new models must beat Production on.
- **Automated Statistical Gating**: a candidate must beat Production on AUC **and** pass McNemar's test on per-sample correctness (exact test for small samples), and meet hard floors, before promotion. The gate fails closed if it cannot evaluate.
- **Safe rollback** that restores the *previous* Production version, never the one being demoted.

## Quick Start (Docker Compose)

The entire platform (PostgreSQL, Redis, MLflow, Airflow, Inference API, Prom/Grafana, Drift Detector) is orchestrated via Docker Compose.

```bash
# 1. Clone the repo
git clone https://github.com/ajitashwath/autonomous-ml.git
cd autonomous-ml

# 2. Setup environment variables
cp .env.example .env

# 3. Put the dataset where the trainer expects it
#    (Telco Customer Churn CSV -> data/raw/telco_churn.csv)

# 4. Start the entire platform
docker compose up -d --build
```

### Accessing the UI's
Once the containers are healthy, you can access the localized services:

- **Inference API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow Registry:** [http://localhost:5000](http://localhost:5000)
- **Airflow UI:** [http://localhost:8080](http://localhost:8080) (admin / admin)
- **Grafana Dashboards:** [http://localhost:3000](http://localhost:3000) (admin / admin)
- **Prometheus Targets:** [http://localhost:9090](http://localhost:9090)

## The "Day in the Life" Flow

1. **Initial Training:**
   Run the trainer to ingest `telco_churn.csv`, log metrics to MLflow, register the model in **Staging**, and save the reference and holdout Parquet files under `data/reference/`.
   `docker compose run --rm trainer`
2. **First Deployment:**
   Run the validation gate. With no Production model it auto-promotes the Staging model (subject to the hard floors in `configs/validation.yaml`). The Inference API polls the registry and loads it within a minute.
   `docker compose run --rm --entrypoint python trainer -m src.validation_gate.gate`
3. **Live Traffic:**
   Clients call `POST /api/v1/predict`. The response includes a `request_id`. Features and the prediction are logged to Postgres *after* the response is sent, so logging adds no latency and a database outage never fails a request.
4. **Feedback:**
   When the real outcome is known (did the customer actually churn?), send it back with that `request_id` (see below). This is optional but it is what lets retraining learn from production.
5. **Drift Detection:**
   The `drift_detector` compares logged traffic to the training baseline. It waits for at least `min_samples` predictions, and only acts when the *observed* share of drifted columns reaches the threshold.
6. **Self-Healing:**
   On drift the detector triggers the Airflow `retrain_pipeline` DAG (at most once per `retrain_cooldown_seconds`). The DAG runs `train_model`, which trains on the CSV **plus older labelled production rows** and stages the new model, then `validate_and_deploy`: the candidate is scored against Production on the newest labelled production rows (falling back to the static holdout) and must pass the AUC delta, McNemar's test and the hard floors. If it does, the Deployer swaps them and the Inference API hot-swaps to it automatically. The drift baseline is rebuilt from the data the new model trained on.

## Reporting Outcomes (Feedback Loop)

Every prediction returns a `request_id`:

```bash
curl -s -X POST localhost:8000/api/v1/predict -H 'Content-Type: application/json' -d @customer.json
# {"request_id": "6f1c…", "prediction": 1, "probability": 0.83, "model_version": "4", ...}
```

Once you know what actually happened, report it (up to 1000 labels per call; `actual_label` is `0` = stayed, `1` = churned):

```bash
curl -s -X POST localhost:8000/api/v1/labels -H 'Content-Type: application/json' \
  -d '{"labels": [{"request_id": "6f1c…", "actual_label": 1}]}'
# {"received": 1, "updated": 1, "unknown_request_ids": []}
```

Re-sending a label for the same `request_id` corrects it. Unknown ids are reported, not fatal.

How labels are used:

| Where | What happens |
|---|---|
| **Metrics** | `automlops_labels_total{outcome="correct"\|"incorrect"}`: live accuracy is `correct / (correct + incorrect)`, e.g. `sum(increase(automlops_labels_total{outcome="correct"}[7d])) / sum(increase(automlops_labels_total[7d]))`. |
| **Retraining** | Once at least `feedback.min_labelled_rows` (default 300) labelled rows exist, the older 70% are added to the training set. Below that, training uses the CSV alone. |
| **Validation gate** | The newest 30% are held out (never trained on) and saved as `production_holdout.parquet`. The gate scores Production and the candidate on it. It needs at least `min_production_holdout_rows` (50) rows containing both classes, otherwise it falls back to the static holdout. |
| **Drift baseline** | `reference.parquet` is rebuilt from the model's training data, so after a retrain the same drift is not detected again. |

## Emergency Rollback

If a model crashes live traffic despite passing offline gates, instantly demote it and restore the previous version:

```bash
docker exec -it automlops_inference_api python -m src.rollback_manager.rollback --reason "Severe P99 latency degradation"
```

The target is the version that was in Production before the current one (recorded as a `previous_production` tag when it was promoted), or else the newest older archived version. The demoted version is tagged `rolled_back` and is never auto-selected again. Pass `--target-version N` to choose explicitly. Production is swapped in a single MLflow call, so it is never empty.

## Configuration

| File | Controls |
|---|---|
| `configs/training.yaml` | Data paths, features, XGBoost hyperparameters, the `feedback` block. |
| `configs/drift.yaml` | `drift_share_threshold`, `pvalue_threshold`, `min_samples`, `retrain_cooldown_seconds`, window size, Slack/Airflow alerting. |
| `configs/validation.yaml` | `auc_delta_threshold`, `significance_level`, `require_significance`, `hard_floors`, evaluation-set paths. |

The drift detector and gate read these YAML files. The `DRIFT_*` / `VALIDATION_*` variables in `.env` are **not** currently used by them.

## Kubernetes

See [k8s/README.md](k8s/README.md). The drift detector runs as a CronJob (`--once`); it exits non-zero when it cannot do its job (no reference data, analysis failure, Airflow unreachable), so a failing Job means detection is not working.

## Development

```bash
pip install -r requirements.txt
ruff check src/ tests/          # lint (must pass; CI runs it)
pytest tests/ -v                # unit + integration; hermetic (SQLite, real models, no services needed)
```

CI (`.github/workflows/ci.yml`) runs lint, the full test suite, and builds every Docker image, and validates the Compose file.

## Directory Structure
- `src/core/` - Shared DB (including the idempotent column migration), Config, Logging, Exceptions.
- `src/training_service/` - Pipeline, XGBoost Model, Datasets, production-feedback loading.
- `src/inference_api/` - FastAPI endpoints (`/predict`, `/predict/batch`, `/labels`, `/health`), model hot-loader.
- `src/data_logger/` - Prediction logging and label storage.
- `src/drift_detector/` - Evidently AI detector (daemon or `--once`).
- `src/model_registry/` - MLflow registry wrapper: stages, promotion lineage, rollback.
- `src/retraining_pipeline/` - Airflow DAG definitions.
- `src/validation_gate/` - Model comparison, McNemar test, hard floors.
- `airflow/`, `mlflow/` - Dockerfiles for those services (Airflow runs the ML tasks in an isolated virtualenv).
- `configs/` - YAML definitions driving the whole platform.
- `k8s/`, `monitoring/` - Kubernetes manifests, Prometheus and Grafana config.

## Known Limitations
- **No authentication** on the inference API (including `/labels`); put it behind a gateway before exposing it.
- **Label quality is on you.** Wrong or biased labels feed straight into retraining. Labels are usually delayed, so the newest predictions are the least likely to be labelled yet.
- **Baseline files are overwritten on every training run**, even if the candidate is later rejected by the gate, so `reference.parquet` can move ahead of the model actually in Production.
- **Holdout independence is approximate.** The production slice is the newest rows, but a model retrained earlier may already have trained on some of them, which biases the comparison toward the incumbent (the conservative direction).
- **Drift and validation thresholds come from YAML only** (see Configuration).
- **Kubernetes:** manifests do not include Airflow or PostgreSQL, and have not been applied to a real cluster. The Docker images are built in CI but the full Compose stack has not been exercised end to end.
- The API still uses MLflow model *stages*, which MLflow has deprecated in favour of aliases.
