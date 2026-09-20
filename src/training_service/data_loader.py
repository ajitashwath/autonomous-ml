

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.exceptions import DataLoadError, DataValidationError
from src.core.logging import get_logger

logger = get_logger(__name__)



REQUIRED_COLUMNS: set[str] = {
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
}


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    n_train: int
    n_test: int
    class_balance: dict[str, float]


def _validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Raw dataset is missing required columns: {missing}",
            details={"missing_columns": list(missing)},
        )


def _clean(df: pd.DataFrame, target_column: str, drop_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["TotalCharges"])
        dropped = before - len(df)
        if dropped > 0:
            logger.warning("dropped_rows_with_null_total_charges", count=dropped)

    existing_drops = [c for c in drop_columns if c in df.columns]
    df = df.drop(columns=existing_drops)

    if df[target_column].dtype == object:
        df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

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
    reference_path: str | None = None,
    holdout_path: str | None = None,
) -> DataSplit:
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

    _validate_schema(df)

    df = _clean(df, target_column=target_column, drop_columns=drop_columns)
    logger.info("data_cleaned", rows=len(df))

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

    if save_reference and reference_path:
        ref_path = Path(reference_path)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        ref_df = X_train.copy()
        ref_df[target_column] = y_train.values
        ref_df.to_parquet(ref_path, index=False)
        logger.info("reference_data_saved", path=str(ref_path), rows=len(ref_df))

    if holdout_path:
        # Labelled, never-trained-on rows used by the validation gate to compare models.
        hold_path = Path(holdout_path)
        hold_path.parent.mkdir(parents=True, exist_ok=True)
        hold_df = X_test.copy()
        hold_df[target_column] = y_test.values
        hold_df.to_parquet(hold_path, index=False)
        logger.info("holdout_data_saved", path=str(hold_path), rows=len(hold_df))

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
