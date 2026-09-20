import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# The ML code runs in a dedicated virtualenv (see airflow/Dockerfile), not Airflow's own Python.
ML_PYTHON = os.environ.get("ML_PYTHON", "/home/airflow/ml-venv/bin/python")
APP_DIR = "/app"

default_args = {
    "owner": "automlops_team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retrain_pipeline",
    default_args=default_args,
    description="Triggered by data drift to retrain, validate, and promote the churn model.",
    schedule=None,
    catchup=False,
    tags=["automlops", "retraining", "self-healing"],
    max_active_runs=1,
) as dag:

    # cwd matters: configs reference data/ paths relative to the app root.
    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"{ML_PYTHON} -m src.training_service.train --config configs/training.yaml",
        cwd=APP_DIR,
        env={"PYTHONPATH": APP_DIR},
        append_env=True,
    )

    validate_and_deploy = BashOperator(
        task_id="validate_and_deploy",
        bash_command=f"{ML_PYTHON} -m src.validation_gate.gate --config configs/validation.yaml",
        cwd=APP_DIR,
        env={"PYTHONPATH": APP_DIR},
        append_env=True,
    )

    train_model >> validate_and_deploy
