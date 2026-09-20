import datetime as dt
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DataValidationError
from src.training_service import feedback as feedback_module
from src.training_service.data_loader import REQUIRED_COLUMNS, load_and_split
from src.training_service.feedback import (
    FeedbackData,
    fetch_labelled_rows,
    load_feedback,
    prepare_feedback,
)

T0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _rows(n: int, start_tenure: int = 1) -> pd.DataFrame:
    """Labelled rows as fetch_labelled_rows() returns them, oldest first."""
    return pd.DataFrame({
        "tenure": range(start_tenure, start_tenure + n),
        "MonthlyCharges": np.linspace(20, 100, n),
        "Churn": [i % 2 for i in range(n)],
        "_created_at": [T0 + dt.timedelta(minutes=i) for i in range(n)],
    })


class TestPrepareFeedback:
    def test_too_few_rows_means_no_feedback(self):
        assert prepare_feedback(_rows(299), "Churn", min_rows=300) is None

    def test_newest_rows_are_held_out_for_evaluation(self):
        data = prepare_feedback(_rows(100), "Churn", eval_fraction=0.3, min_rows=10)

        assert len(data.train) == 70 and len(data.evaluation) == 30
        # Chronological: everything trained on is older than everything evaluated on.
        assert data.train["tenure"].max() < data.evaluation["tenure"].min()

    def test_order_is_by_time_not_by_input_position(self):
        shuffled = _rows(100).sample(frac=1, random_state=0)

        data = prepare_feedback(shuffled, "Churn", eval_fraction=0.3, min_rows=10)

        assert data.train["tenure"].max() < data.evaluation["tenure"].min()

    def test_helper_column_is_not_leaked_into_the_data(self):
        data = prepare_feedback(_rows(50), "Churn", min_rows=10)

        assert "_created_at" not in data.train.columns
        assert "_created_at" not in data.evaluation.columns

    def test_always_keeps_at_least_one_evaluation_row(self):
        data = prepare_feedback(_rows(20), "Churn", eval_fraction=0.0, min_rows=10)

        assert len(data.evaluation) == 1


class TestFetchLabelledRows:
    @pytest.fixture()
    def db(self, sqlite_db):
        with patch("src.core.db.db_session", sqlite_db.session):
            yield sqlite_db

    def test_returns_only_labelled_rows_oldest_first_with_the_target(self, db):
        db.add_prediction("late", {"tenure": 9}, actual_label=1, created_at=T0 + dt.timedelta(days=2))
        db.add_prediction("early", {"tenure": 2}, actual_label=0, created_at=T0)
        db.add_prediction("unlabelled", {"tenure": 5})

        df = fetch_labelled_rows("Churn")

        assert list(df["tenure"]) == [2, 9]
        assert list(df["Churn"]) == [0, 1]
        assert len(df) == 2

    def test_no_labels_gives_an_empty_frame(self, db):
        db.add_prediction("unlabelled", {"tenure": 5})

        assert fetch_labelled_rows("Churn").empty


class TestLoadFeedback:
    CFG = {"enabled": True, "min_labelled_rows": 10, "eval_fraction": 0.3}

    def test_disabled(self):
        with patch.object(feedback_module, "fetch_labelled_rows") as fetch:
            assert load_feedback({"enabled": False}, "Churn") is None
            assert load_feedback(None, "Churn") is None
        fetch.assert_not_called()

    def test_loads_when_enough_labels_exist(self):
        with patch.object(feedback_module, "fetch_labelled_rows", return_value=_rows(100)):
            data = load_feedback(self.CFG, "Churn")

        assert len(data.train) == 70 and len(data.evaluation) == 30

    def test_insufficient_labels_falls_back_to_csv_only(self):
        with patch.object(feedback_module, "fetch_labelled_rows", return_value=_rows(5)):
            assert load_feedback(self.CFG, "Churn") is None

    def test_database_outage_does_not_stop_retraining(self):
        with patch.object(feedback_module, "fetch_labelled_rows", side_effect=RuntimeError("db down")):
            assert load_feedback(self.CFG, "Churn") is None


# ---------------------------------------------------------------------------
# load_and_split with feedback
# ---------------------------------------------------------------------------

def _write_csv(path, n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: ["x"] * n for c in REQUIRED_COLUMNS})
    df["customerID"] = [f"id{i}" for i in range(n)]
    df["tenure"] = rng.integers(1, 72, size=n)
    df["TotalCharges"] = rng.uniform(20, 5000, size=n).round(2)
    df["Churn"] = ["Yes" if i % 4 == 0 else "No" for i in range(n)]
    df.to_csv(path, index=False)


FEATURE_COLUMNS = [c for c in sorted(REQUIRED_COLUMNS) if c not in ("customerID", "Churn")]


def _feedback(columns: list[str], n_train=40, n_eval=15) -> FeedbackData:
    def frame(n, tenure):
        df = pd.DataFrame({c: ["x"] * n for c in columns})
        df["tenure"] = tenure
        df["TotalCharges"] = 999.0
        df["Churn"] = [1] * n
        return df

    return FeedbackData(train=frame(n_train, 500), evaluation=frame(n_eval, 900))


@pytest.fixture()
def split_kwargs(tmp_path):
    csv = tmp_path / "raw.csv"
    _write_csv(csv)
    return dict(
        raw_path=str(csv), target_column="Churn", drop_columns=["customerID"],
        reference_path=str(tmp_path / "ref.parquet"),
        holdout_path=str(tmp_path / "holdout.parquet"),
        production_holdout_path=str(tmp_path / "prod_holdout.parquet"),
    )


class TestLoadAndSplitWithFeedback:
    def test_older_labelled_rows_join_training_only(self, split_kwargs):
        baseline = load_and_split(**{**split_kwargs, "production_holdout_path": None})

        split = load_and_split(**split_kwargs, feedback=_feedback(FEATURE_COLUMNS))

        assert split.n_train == baseline.n_train + 40
        assert split.n_test == baseline.n_test          # test split untouched
        assert (split.X_train["tenure"] == 500).sum() == 40
        assert (split.X_test["tenure"] >= 500).sum() == 0
        assert len(split.X_train) == len(split.y_train)
        assert (split.n_feedback_train, split.n_feedback_eval) == (40, 15)

    def test_newest_rows_are_saved_as_a_production_holdout_and_never_trained_on(
        self, split_kwargs, tmp_path
    ):
        split = load_and_split(**split_kwargs, feedback=_feedback(FEATURE_COLUMNS))

        held_out = pd.read_parquet(tmp_path / "prod_holdout.parquet")
        assert len(held_out) == 15
        assert "Churn" in held_out.columns
        assert set(held_out["tenure"]) == {900}
        assert (split.X_train["tenure"] == 900).sum() == 0

    def test_drift_reference_reflects_what_the_model_trained_on(self, split_kwargs, tmp_path):
        """After a retrain the baseline must include the new production data, otherwise the
        same drift keeps being detected against the old distribution."""
        split = load_and_split(**split_kwargs, feedback=_feedback(FEATURE_COLUMNS))

        reference = pd.read_parquet(tmp_path / "ref.parquet")
        assert len(reference) == split.n_train
        assert (reference["tenure"] == 500).sum() == 40

    def test_without_feedback_a_stale_production_holdout_is_removed(self, split_kwargs, tmp_path):
        stale = tmp_path / "prod_holdout.parquet"
        pd.DataFrame({"tenure": [1], "Churn": [0]}).to_parquet(stale)

        load_and_split(**split_kwargs)

        assert not stale.exists()

    def test_feedback_missing_a_feature_column_is_rejected(self, split_kwargs):
        fb = _feedback([c for c in FEATURE_COLUMNS if c != "Contract"])

        with pytest.raises(DataValidationError) as exc:
            load_and_split(**split_kwargs, feedback=fb)

        assert "Contract" in exc.value.details["missing_columns"]
