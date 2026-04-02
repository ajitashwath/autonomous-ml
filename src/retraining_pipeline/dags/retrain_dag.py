

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

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
    dag_id="retrain_churn_model",
    default_args=default_args,
    description="Triggered by data drift to retrain, validate, and promote the churn model.",
    schedule_interval=None,
    catchup=False,
    tags=["automlops", "retraining", "self-healing"],
    max_active_runs=1,
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python -m src.training_service.train --config /app/configs/training.yaml",
        env={
            "PYTHONPATH": "/app",
            "DRIFT_METRICS": "{{ dag_run.conf.get('drift_metrics', '{}') }}",
        },
        append_env=True,
    )

    validate_and_deploy = BashOperator(
        task_id="validate_and_deploy",
        bash_command="python -m src.validation_gate.gate --config /app/configs/validation.yaml",
        env={"PYTHONPATH": "/app"},
        append_env=True,
    )

    train_model >> validate_and_deploy