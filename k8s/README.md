# AutoMLOps — Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying the AutoMLOps platform to a production cluster.

## Directory Structure

```
k8s/
├── namespace.yaml                  # automlops namespace
├── configmap.yaml                  # Non-secret environment config
├── secret.yaml.template            # Secret template (fill in, do NOT commit real values)
├── inference-api/
│   └── deployment.yaml             # Deployment (3 replicas) + ClusterIP Service + HPA
├── drift-detector/
│   └── cronjob.yaml                # CronJob (every 5 min, Forbid concurrency)
└── mlflow/
    └── deployment.yaml             # MLflow Deployment + Service + PVCs
```

## Prerequisites

- Kubernetes 1.25+
- `kubectl` configured for your target cluster
- A running PostgreSQL instance accessible from the cluster (or add `k8s/postgres/` separately)
- Container images built and pushed to your registry

## Quick Start

### 1. Build and Push Images

```bash
# Build the images (run from project root)
docker build -t your-registry/automlops/inference-api:latest   -f src/inference_api/Dockerfile .
docker build -t your-registry/automlops/drift-detector:latest  -f src/drift_detector/Dockerfile .

# Push to your registry
docker push your-registry/automlops/inference-api:latest
docker push your-registry/automlops/drift-detector:latest
```

Update the `image:` fields in the deployment manifests to point to your registry.

### 2. Create Secrets

```bash
# Copy the template
cp k8s/secret.yaml.template k8s/secret.yaml

# Fill in base64-encoded values, e.g.:
echo -n "my_postgres_password" | base64

# Apply secrets
kubectl apply -f k8s/secret.yaml
```

> ⚠️ **Never commit `secret.yaml`** — it contains real credentials. It is listed in `.gitignore`.

### 3. Deploy Everything

```bash
# Apply in order (namespace first)
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml          # your filled-in copy
kubectl apply -f k8s/mlflow/
kubectl apply -f k8s/inference-api/
kubectl apply -f k8s/drift-detector/
```

### 4. Verify

```bash
# Check that pods come up healthy
kubectl get pods -n automlops

# Check inference API readiness
kubectl port-forward svc/inference-api-service 8000:8000 -n automlops
curl http://localhost:8000/health

# Check drift detector CronJob
kubectl get cronjobs -n automlops
kubectl get jobs -n automlops
```

## Service Architecture

```
                    ┌─────────────────────┐
Internet/Ingress ──►│   inference-api     │ :8000 (ClusterIP)
                    │   (3→10 replicas)   │◄── HPA (CPU/Memory)
                    └──────────┬──────────┘
                               │ logs predictions
                               ▼
                    ┌─────────────────────┐   ┌──────────────────┐
                    │     PostgreSQL       │◄──│  drift-detector  │
                    │   prediction_logs   │   │  (CronJob 5min)  │
                    └─────────────────────┘   └────────┬─────────┘
                                                        │ triggers
                    ┌─────────────────────┐             ▼
                    │       MLflow        │   ┌──────────────────┐
                    │  Model Registry     │◄──│  Airflow DAG     │
                    │  :5000 (ClusterIP)  │   │  (retraining)    │
                    └─────────────────────┘   └──────────────────┘
```

## Scaling Notes

- **Inference API**: The HPA scales from 3→10 pods at 70% CPU. For GPU workloads, replace with a custom metric on `automlops_active_requests`.
- **Drift Detector**: Runs as a CronJob with `concurrencyPolicy: Forbid` — at most one detection job runs at a time.
- **MLflow**: Single replica is sufficient for model registry operations. Add a ReadWriteMany PV if you need HA.

## Health Checks

All workloads include liveness and readiness probes. The inference API `/health` endpoint is the probe target — it returns 200 only when a model is loaded.

## Monitoring

Prometheus scraping is enabled via pod annotations on the inference API:
```yaml
prometheus.io/scrape: "true"
prometheus.io/port:   "8000"
prometheus.io/path:   "/metrics"
```

The Grafana dashboard (`monitoring/grafana/dashboards/automlops.json`) auto-provisions with 8 panels covering request rate, latency, error rate, prediction distribution, active model version, active requests, and model load events.
