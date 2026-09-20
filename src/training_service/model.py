

from __future__ import annotations

from typing import Any

import numpy as np
import xgboost as xgb

from src.core.exceptions import TrainingError
from src.core.logging import get_logger

logger = get_logger(__name__)


def build_model(hyperparameters: dict[str, Any]) -> xgb.XGBClassifier:
    xgb_params = {
        k: v for k, v in hyperparameters.items()
        if k != "early_stopping_rounds"
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
    return model.predict_proba(X)[:, 1]
