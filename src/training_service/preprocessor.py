"""
training_service/preprocessor.py

Builds a reproducible sklearn preprocessing Pipeline.

Design decisions:
  - All transformers are wrapped in a single Pipeline so the fitted object
    can be serialised as one artifact — no risk of train/serve skew.
  - ColumnTransformer applies different transforms to categorical vs numerical
    columns declaratively (driven by configs/training.yaml).
  - We use Pipeline.fit_transform(X_train) then Pipeline.transform(X_test)
    so test statistics NEVER leak into the fit step.

Production note:
  - In production this pipeline is saved alongside the model in MLflow so that
    the inference_api applies the IDENTICAL transformations seen at training time.
  - Adding new features is as simple as adding them to training.yaml — the
    pipeline handles them automatically.
"""

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
    """
    Build a ColumnTransformer that:
      - Numerical columns: median-impute → standard-scale
      - Categorical columns: constant-impute ("missing") → one-hot encode

    Returns:
        An unfitted ColumnTransformer (call .fit_transform on X_train).
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",    # gracefully handles unseen categories at serve time
                sparse_output=False,        # return dense array for compatibility
                drop="first",               # avoid multicollinearity
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",       # drop any column not explicitly listed
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
    """
    Fit the preprocessor on X_train, then transform both splits.
    Returns dense numpy arrays + the final feature names list.

    Raises:
        DataValidationError: if X_train is missing columns the preprocessor expects.
    """
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
    """Persist the fitted preprocessor to disk (used by inference_api)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("preprocessor_saved", path=str(out_path))


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a previously saved preprocessor from disk."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at: {path}")
    with open(in_path, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("preprocessor_loaded", path=str(in_path))
    return preprocessor
