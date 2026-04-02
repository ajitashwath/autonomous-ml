

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.exceptions import DataValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


def build_preprocessor(
    categorical_features: list[str],
    numerical_features: list[str],
) -> ColumnTransformer:
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="first",
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "preprocessor_built",
        n_numerical=len(numerical_features),
        n_categorical=len(categorical_features),
    )
    return preprocessor


def fit_transform(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    logger.info("fitting_preprocessor", n_train=len(X_train))
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
    except Exception as exc:
        raise DataValidationError(
            f"Preprocessing failed on training data: {exc}",
            details={"error": str(exc)},
        ) from exc

    X_test_processed = preprocessor.transform(X_test)

    try:
        feature_names: list[str] = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    logger.info(
        "preprocessing_complete",
        input_features=len(X_train.columns),
        output_features=len(feature_names),
        train_shape=X_train_processed.shape,
        test_shape=X_test_processed.shape,
    )
    return X_train_processed, X_test_processed, feature_names


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("preprocessor_saved", path=str(out_path))


def load_preprocessor(path: str) -> ColumnTransformer:
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at: {path}")
    with open(in_path, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("preprocessor_loaded", path=str(in_path))
    return preprocessor