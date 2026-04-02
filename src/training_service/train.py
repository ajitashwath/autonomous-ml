

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import yaml

from src.core.config import get_settings
from src.core.exceptions import AutoMLOpsError
from src.core.logging import configure_logging, get_logger
from src.training_service.data_loader import load_and_split
from src.training_service.evaluator import evaluate
from src.training_service.model import build_model, predict_proba, train_model
from src.training_service.preprocessor import (
    build_preprocessor,
    fit_transform,
    save_preprocessor,
)

configure_logging()
logger = get_logger(__name__)



def load_training_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info("training_config_loaded", path=str(path))
    return config



def setup_mlflow(settings: Any, config: dict[str, Any]) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info(
        "mlflow_configured",
        tracking_uri=settings.mlflow_tracking_uri,
        experiment=settings.mlflow_experiment_name,
    )


def log_run(
    config: dict[str, Any],
    settings: Any,
    eval_result: Any,
    feature_names: list[str],
    preprocessor: Any,
    model: Any,
    elapsed_seconds: float,
) -> str:
    run = mlflow.active_run()
    run_id = run.info.run_id

    mlflow.set_tags({
        **config.get("mlflow", {}).get("tags", {}),
        "run_id": run_id,
    })

    mlflow.log_params(config["model"]["hyperparameters"])
    mlflow.log_param("test_size",      config["data"]["test_size"])
    mlflow.log_param("random_state",   config["data"]["random_state"])
    mlflow.log_param("n_features",     len(feature_names))
    mlflow.log_param("threshold",      config["evaluation"]["threshold"])
    mlflow.log_param("training_time_s", round(elapsed_seconds, 2))

    mlflow.log_metrics(eval_result.to_dict())

    preprocessor_path = "artifacts/preprocessor.pkl"
    save_preprocessor(preprocessor, preprocessor_path)
    mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

    features_path = "artifacts/feature_names.txt"
    Path("artifacts").mkdir(exist_ok=True)
    Path(features_path).write_text("\n".join(feature_names))
    mlflow.log_artifact(features_path, artifact_path="preprocessor")

    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        registered_model_name=settings.mlflow_model_name,
    )

    logger.info("mlflow_run_logged", run_id=run_id, **eval_result.to_dict())
    return run_id



def run_training_pipeline(config_path: str = "configs/training.yaml") -> None:
    settings = get_settings()
    config = load_training_config(config_path)

    setup_mlflow(settings, config)

    data_cfg = config["data"]
    feat_cfg = config["features"]
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]

    with mlflow.start_run() as run:
        logger.info("mlflow_run_started", run_id=run.info.run_id)
        start_time = time.time()

        data_split = load_and_split(
            raw_path=data_cfg["raw_path"],
            target_column=data_cfg["target_column"],
            drop_columns=data_cfg.get("drop_columns", []),
            test_size=data_cfg["test_size"],
            random_state=data_cfg["random_state"],
            save_reference=data_cfg.get("save_reference", True),
            reference_path=data_cfg.get("reference_path"),
        )

        preprocessor = build_preprocessor(
            categorical_features=feat_cfg["categorical"],
            numerical_features=feat_cfg["numerical"],
        )
        X_train_proc, X_test_proc, feature_names = fit_transform(
            preprocessor, data_split.X_train, data_split.X_test
        )

        model = build_model(model_cfg["hyperparameters"])
        model = train_model(
            model=model,
            X_train=X_train_proc,
            y_train=data_split.y_train.values,
            X_val=X_test_proc,
            y_val=data_split.y_test.values,
            early_stopping_rounds=model_cfg["hyperparameters"].get(
                "early_stopping_rounds", 20
            ),
        )

        y_proba_test = predict_proba(model, X_test_proc)
        eval_result = evaluate(
            y_true=data_split.y_test.values,
            y_proba=y_proba_test,
            threshold=eval_cfg["threshold"],
        )

        elapsed = time.time() - start_time
        logger.info("pipeline_complete", elapsed_seconds=round(elapsed, 2))
        logger.info("evaluation_summary", summary=eval_result.summary())

        run_id = log_run(
            config=config,
            settings=settings,
            eval_result=eval_result,
            feature_names=feature_names,
            preprocessor=preprocessor,
            model=model,
            elapsed_seconds=elapsed,
        )

        logger.info(
            "training_pipeline_finished",
            run_id=run_id,
            model_name=settings.mlflow_model_name,
            auc=round(eval_result.roc_auc, 4),
        )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoMLOps Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training config YAML (default: configs/training.yaml)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_training_pipeline(config_path=args.config)
        sys.exit(0)
    except AutoMLOpsError as exc:
        logger.error("training_pipeline_failed", error=exc.message, details=exc.details)
        sys.exit(1)
    except Exception as exc:
        logger.error("training_pipeline_unexpected_error", error=str(exc), exc_info=True)
        sys.exit(1)