import numpy as np
import pandas as pd

from src.training_service.data_loader import REQUIRED_COLUMNS, load_and_split


def _write_csv(path, n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: ["x"] * n for c in REQUIRED_COLUMNS})
    df["customerID"] = [f"id{i}" for i in range(n)]
    df["tenure"] = rng.integers(1, 72, size=n)
    df["TotalCharges"] = rng.uniform(20, 5000, size=n).round(2)
    df["Churn"] = ["Yes" if i % 4 == 0 else "No" for i in range(n)]
    df.to_csv(path, index=False)


def test_holdout_is_saved_labelled_and_disjoint_from_reference(tmp_path):
    csv = tmp_path / "raw.csv"
    _write_csv(csv)
    ref, hold = tmp_path / "ref.parquet", tmp_path / "holdout.parquet"

    split = load_and_split(
        raw_path=str(csv),
        target_column="Churn",
        drop_columns=["customerID"],
        reference_path=str(ref),
        holdout_path=str(hold),
    )

    hold_df = pd.read_parquet(hold)
    ref_df = pd.read_parquet(ref)
    assert len(hold_df) == split.n_test
    assert "Churn" in hold_df.columns
    assert set(hold_df["Churn"]) <= {0, 1}
    assert len(ref_df) == split.n_train
    assert len(hold_df) + len(ref_df) == 200


def test_holdout_not_written_unless_requested(tmp_path):
    csv = tmp_path / "raw.csv"
    _write_csv(csv)

    load_and_split(
        raw_path=str(csv),
        target_column="Churn",
        drop_columns=["customerID"],
        reference_path=str(tmp_path / "ref.parquet"),
    )

    assert not (tmp_path / "holdout.parquet").exists()
