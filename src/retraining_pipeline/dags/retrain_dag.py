"""
retraining_pipeline/dags/retrain_dag.py

Airflow DAG that orchestrates the retraining and validation pipeline.

Triggers:
  1. REST API (sent by drift_detector/alerter.py when drift is detected)
  2. Manual UI trigger

Process:
  1. Train new model (`BashOperator` running `src.training_service.train`)
  2. Validate model (`BashOperator` running `src.validation_gate.gate`)
  3. Promote/Rollback (Handled natively inside the validation gate logic)

Production mechanics:
  - This DAG relies on the fact that the Airflow worker container has access
    to the same volume mounts (`/app/data`, `/app/configs`) and the same
    source code as the rest of the AutoMLOps platform.
  - The Airflow Dockerfile (built in Phase 1) installs all ML dependencies,
    allowing us to run the training script directly inside the worker.
  - In a massive-scale system, you would use `KubernetesPodOperator` instead
    of `BashOperator` to spin up a dedicated isolated training container with GPUs.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# Default arguments applied to all tasks
default_args = {
    "owner": "automlops_team",
    "depends_on_past": False,
    # Start date in the past allows manual triggers immediately
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_churn_model",
    default_args=default_args,
    description="Triggered by data drift to retrain, validate, and promote the churn model.",
    schedule_interval=None,  # No cron schedule -> Event Driven via REST API only
    catchup=False,
    tags=["automlops", "retraining", "self-healing"],
    max_active_runs=1,       # Prevent parallel training runs from corrupting MLflow
) as dag:

    # ── Task 1: Train New Model ────────────────────────────────────────────────
    # Runs the exact same CLI entrypoint we built in Phase 2
    # The PYTHONPATH env var is set in the Airflow Dockerfile
    train_model = BashOperator(
        task_id="train_model",
        bash_command="python -m src.training_service.train --config /app/configs/training.yaml",
        # We can extract the trigger reason (drift metrics) from the DAG run conf
        # and echo it into the logs for debugging purposes
        env={
            "PYTHONPATH": "/app",
            "DRIFT_METRICS": "{{ dag_run.conf.get('drift_metrics', '{}') }}",
        },
        append_env=True,
    )

    # ── Task 2: Validation Gate + Auto Deploy ──────────────────────────────────
    # This module will be built in Phase 8
    # It fetches the new Staging model, compares to Production, and promotes/rolls back
    validate_and_deploy = BashOperator(
        task_id="validate_and_deploy",
        bash_command="python -m src.validation_gate.gate --config /app/configs/validation.yaml",
        env={"PYTHONPATH": "/app"},
        append_env=True,
    )

    # ── DAG Topology ───────────────────────────────────────────────────────────
    train_model >> validate_and_deploy
