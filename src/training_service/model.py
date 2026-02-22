"""
training_service/model.py

XGBoost model builder — reads hyperparameters from configs/training.yaml.

Design decisions:
  - We wrap XGBClassifier in a thin builder function so hyperparams come
    from config, not hardcoded in train.py.
  - scale_pos_weight handles class imbalance without oversampling (faster,
    no data leakage risk vs SMOTE at training time).
  - eval_set + early_stopping_rounds prevent overfitting during training.

Production notes:
  - To swap to LightGBM or sklearn RandomForest, change model.type in
    training.yaml and add a corresponding branch in build_model().
  - Never tune hyperparameters manually in this file — use Optuna or Ray Tune
    wired to MLflow for hyperparameter search runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline

from src.core.exceptions import TrainingError
from src.core.logging import get_logger

logger = get_logger(__name__)


def build_model(hyperparameters: dict[str, Any]) -> xgb.XGBClassifier:
    """
    Instantiate an XGBClassifier from the hyperparameter dict in training.yaml.

    Args:
        hyperparameters: Dict matching XGBClassifier constructor arguments.

    Returns:
        Unfitted XGBClassifier ready for .fit().
    """
    # Remove non-XGB keys that we may have added for our own bookkeeping
    xgb_params = {
        k: v for k, v in hyperparameters.items()
        if k != "early_stopping_rounds"     # passed at fit-time, not init-time
    }

    model = xgb.XGBClassifier(
        **xgb_params,
        use_label_encoder=False,
        verbosity=0,
    )
    logger.info("model_built", model_type="xgboost", params=xgb_params)
    return model


def train_model(
    model: xgb.XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int = 20,
) -> xgb.XGBClassifier:
    """
    Fit the model with an evaluation set for early stopping.

    Args:
        model:                  Unfitted XGBClassifier.
        X_train/y_train:        Training data.
        X_val/y_val:            Validation data for early stopping.
        early_stopping_rounds:  Stop if no improvement after N rounds.

    Returns:
        Fitted XGBClassifier.

    Raises:
        TrainingError: if .fit() raises an unexpected exception.
    """
    logger.info(
        "training_started",
        n_train=len(X_train),
        n_val=len(X_val),
        early_stopping_rounds=early_stopping_rounds,
    )
    try:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
    except Exception as exc:
        raise TrainingError(f"XGBoost training failed: {exc}") from exc

    best_round = getattr(model, "best_iteration", "N/A")
    logger.info("training_complete", best_iteration=best_round)
    return model


def predict_proba(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """
    Return the positive-class probability for each row.

    Returns:
        1D numpy array of probabilities in [0, 1].
    """
    return model.predict_proba(X)[:, 1]
