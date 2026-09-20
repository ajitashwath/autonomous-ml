# Automated ML Monitoring Pipeline
It autonomously manages the full ML lifecycle: from training and deployment to monitoring, drift detection, statistical validation, and emergency rollbacks.

## Architecture Highlights
- **FastAPI Inference** with zero-downtime background model hot-swapping.
- **Evidently AI Drift Detection** daemon pulling traffic directly from PostgreSQL.
- **Event-Driven Airflow DAGs** triggered instantly via REST API upon drift.
- **Automated Statistical Gating** using McNemar's test to ensure new models mathematically beat Production before promotion.

## Quick Start (Docker Compose)

The entire platform (PostgreSQL, Redis, MLflow, Airflow, Inference API, Prom/Grafana, Drift Detector) is orchestrated via Docker Compose.

```bash
# 1. Clone the repo
git clone https://github.com/ajitashwath/self-healing.git
cd self-healing

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
   Clients hit `POST /predict`. The API logs features to Postgres asynchronously so I/O never blocks the event loop.
4. **Drift Detection:** 
   The `drift_detector` container continually compares Postgres logs to the Parquet baseline. 
5. **Self-Healing:** 
   If the share of drifted columns reaches `0.3` (`configs/drift.yaml`) on a window of at least `min_samples` predictions, the detector triggers the Airflow Retraining DAG via REST API (at most once per `retrain_cooldown_seconds`). The DAG runs `train_model`, which stages a new model, then `validate_and_deploy`: the candidate must beat Production by the configured AUC delta on the labelled holdout **and** pass McNemar's test on per-sample correctness. If it does, the Deployer swaps them and the Inference API pulls the new model automatically. 

## Emergency Rollback

If a model crashes live traffic despite passing offline gates, instantly demote it and restore the previous version:

```bash
docker exec -it automlops_inference_api python -m src.rollback_manager.rollback --reason "Severe P99 latency degradation"
```

## Directory Structure
- `src/core/` - Shared DB, Config, Logging, Exceptions.
- `src/training_service/` - Pipeline, XGBoost Model, Datasets.
- `src/inference_api/` - FastAPI Endpoints, Background Model Hot-Loader.
- `src/drift_detector/` - Evidently AI Daemon.
- `src/retraining_pipeline/` - Airflow DAG Definitions.
- `src/validation_gate/` - Model Comparison & Statistical Tests.
- `configs/` - YAML definitions driving the whole platform. 
