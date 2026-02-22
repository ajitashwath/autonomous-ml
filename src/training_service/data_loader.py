"""
training_service/data_loader.py

Loads, validates, and splits the raw churn dataset.

Design decisions:
  - Returns a DataLoadResult named-tuple so callers never deal with raw tuples.
  - Validates schema + types before any processing — fail fast at data ingest.
  - TotalCharges is stored as string in the raw CSV (Telco quirk); we coerce it.
  - Saves the training split as a Parquet reference file for Evidently drift checks.

Production notes:
  - Swap `from_csv` with a `from_postgres` or `from_feature_store` method without
    changing the rest of the pipeline — only this file needs to change.
  - Add Great Expectations or Pandera schemas here for data contract enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.config import get_settings
from src.core.exceptions import DataLoadError, DataValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)


# ── Data contract ──────────────────────────────────────────────────────────────

REQUIRED_COLUMNS: set[str] = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}


@dataclass
class DataSplit:
    """Container for train/test feature matrices and target arrays."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    n_train: int
    n_test: int
    class_balance: dict[str, float]


def _validate_schema(df: pd.DataFrame) -> None:
    """
    Assert that all required columns are present in the raw DataFrame.

    Raises:
        DataValidationError: if any required column is missing.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Raw dataset is missing required columns: {missing}",
            details={"missing_columns": list(missing)},
        )


def _clean(df: pd.DataFrame, target_column: str, drop_columns: list[str]) -> pd.DataFrame:
    """
    Clean raw DataFrame:
      - Drop irrelevant columns (e.g. customerID).
      - Coerce TotalCharges to numeric (Telco CSV quirk — spaces in empty rows).
      - Drop rows where target is null.
      - Convert binary target to int (Yes/No → 1/0).

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    df = df.copy()

    # Telco-specific: TotalCharges contains whitespace strings for new customers
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["TotalCharges"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("dropped_rows_with_null_total_charges", count=dropped)

    # Drop unwanted columns (customerID etc.)
    existing_drops = [c for c in drop_columns if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Encode binary target: "Yes" → 1, "No" → 0
    if df[target_column].dtype == object:
        df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

    # Drop rows where target is still null after mapping
    before = len(df)
    df = df.dropna(subset=[target_column])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning("dropped_rows_with_null_target", count=dropped)

    return df


def load_and_split(
    raw_path: str,
    target_column: str,
    drop_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    save_reference: bool = True,
    reference_path: Optional[str] = None,
) -> DataSplit:
    """
    Load the raw CSV, validate schema, clean, and produce a train/test split.

    Args:
        raw_path:         Path to the raw CSV file.
        target_column:    Name of the binary target column.
        drop_columns:     List of column names to drop before splitting.
        test_size:        Fraction of data for the test set.
        random_state:     Seed for reproducibility.
        save_reference:   If True, save X_train as Parquet for drift detection.
        reference_path:   Where to save the Parquet reference file.

    Returns:
        DataSplit with X_train, X_test, y_train, y_test, metadata.

    Raises:
        DataLoadError:       if the CSV cannot be read.
        DataValidationError: if the schema contract is violated.
    """
    path = Path(raw_path)
    if not path.exists():
        raise DataLoadError(
            f"Raw data file not found: {raw_path}",
            details={"path": str(path.resolve())},
        )

    logger.info("loading_raw_data", path=str(path))
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DataLoadError(f"Failed to read CSV: {exc}") from exc

    logger.info("raw_data_loaded", rows=len(df), columns=len(df.columns))

    # ── Validate ───────────────────────────────────────────────────────────────
    _validate_schema(df)

    # ── Clean ──────────────────────────────────────────────────────────────────
    df = _clean(df, target_column=target_column, drop_columns=drop_columns)
    logger.info("data_cleaned", rows=len(df))

    # ── Split ──────────────────────────────────────────────────────────────────
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    class_balance = {
        "churn_rate_train": float(y_train.mean()),
        "churn_rate_test": float(y_test.mean()),
    }

    logger.info(
        "data_split_complete",
        n_train=len(X_train),
        n_test=len(X_test),
        **class_balance,
    )

    # ── Save reference ─────────────────────────────────────────────────────────
    if save_reference and reference_path:
        ref_path = Path(reference_path)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        # Save X_train + y_train together as the reference distribution
        ref_df = X_train.copy()
        ref_df[target_column] = y_train.values
        ref_df.to_parquet(ref_path, index=False)
        logger.info("reference_data_saved", path=str(ref_path), rows=len(ref_df))

    return DataSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X_train.columns),
        n_train=len(X_train),
        n_test=len(X_test),
        class_balance=class_balance,
    )
