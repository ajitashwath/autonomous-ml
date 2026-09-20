from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    project_name: str = Field(default="automlops", description="Project/service name")
    env: Literal["development", "staging", "production"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    database_url: str = Field(
        default="postgresql+psycopg2://automlops:changeme@postgres:5432/automlops",
        description="SQLAlchemy connection string",
    )
    db_pool_size: int = Field(default=10)
    db_max_overflow: int = Field(default=20)
    db_pool_timeout: int = Field(default=30)

    mlflow_tracking_uri: str = Field(default="http://mlflow:5000")
    mlflow_experiment_name: str = Field(default="automlops_churn")
    mlflow_model_name: str = Field(default="churn_classifier")

    inference_api_host: str = Field(default="0.0.0.0")
    inference_api_port: int = Field(default=8000)
    inference_model_stage: str = Field(default="Production")
    model_load_timeout_seconds: int = Field(default=30)

    airflow_host: str = Field(default="http://airflow:8080")
    airflow_username: str = Field(default="admin")
    airflow_password: str = Field(default="admin")
    airflow_retrain_dag_id: str = Field(default="retrain_pipeline")

    drift_reference_data_path: str = Field(
        default="/app/data/reference/reference.parquet"
    )
    drift_window_size: int = Field(default=1000)
    drift_check_interval_seconds: int = Field(default=300)
    drift_share_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    drift_pvalue_threshold: float = Field(default=0.05, ge=0.0, le=1.0)

    validation_auc_delta_threshold: float = Field(default=0.005)
    validation_significance_level: float = Field(default=0.05)

    redis_url: str = Field(default="redis://redis:6379/0")
    prometheus_port: int = Field(default=9090)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
