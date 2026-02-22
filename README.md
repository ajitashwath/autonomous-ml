# AutoMLOps — Self-Healing ML Platform

AutoMLOps is a production-grade, self-healing Machine Learning platform. It autonomously manages the full ML lifecycle: from training and deployment to monitoring, drift detection, statistical validation, and emergency rollbacks.

## Architecture Highlights

- **FastAPI Inference** with zero-downtime background model hot-swapping.
- **Evidently AI Drift Detection** daemon pulling traffic directly from PostgreSQL.
- **Event-Driven Airflow DAGs** triggered instantly via REST API upon drift.
- **Automated Statistical Gating** using McNemar's test to ensure new models mathematically beat Production before promotion.

For a full deep-dive, read the `walkthrough.md` or `implementation_plan.md`.

## Quick Start (Docker Compose)

The entire platform (PostgreSQL, Redis, MLflow, Airflow, Inference API, Prom/Grafana, Drift Detector) is orchestrated via Docker Compose.

```bash
# 1. Clone the repo
git clone https://github.com/your-org/automlops.git
cd automlops

# 2. Setup environment variables
cp .env.example .env

# 3. Start the entire platform
docker-compose up -d --build
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
   Run the CLI trainer to ingest `telco_churn.csv`, log metrics to MLflow, and save the reference Parquet.
   `docker exec -it automlops_inference_api python -m src.training_service.train --config configs/training.yaml`
2. **First Deployment:** 
   The validation gate sees no Production model exists, and auto-promotes the new Staging model. The Inference API background thread loads it into memory.
3. **Live Traffic:** 
   Clients hit `POST /predict`. The API logs features to Postgres asynchronously so I/O never blocks the event loop.
4. **Drift Detection:** 
   The `drift_detector` container continually compares Postgres logs to the Parquet baseline. 
5. **Self-Healing:** 
   If drift exceeds `0.3` (configurable), the detector triggers the Airflow Retraining DAG via REST API. Airflow trains a new model, and if it beats the current one by `0.01` AUC, the Deployer swaps them. The Inference API pulls the new model automatically. 

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
