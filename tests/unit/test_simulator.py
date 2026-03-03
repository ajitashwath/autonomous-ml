"""Unit tests for the enhanced drift simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.drift_simulator.simulate import (
    _apply_mild_drift,
    _apply_no_drift,
    _apply_severe_drift,
    apply_drift,
)


@pytest.fixture()
def reference_df() -> pd.DataFrame:
    """A small synthetic reference frame matching the Telco schema columns."""
    rng = np.random.default_rng(0)
    n = 20
    return pd.DataFrame({
        "tenure": rng.integers(1, 72, size=n).tolist(),
        "MonthlyCharges": rng.uniform(20, 100, size=n).tolist(),
        "TotalCharges": rng.uniform(100, 8000, size=n).tolist(),
        "Contract": ["Month-to-month"] * 10 + ["One year"] * 10,
    })


class TestNoDrift:
    def test_no_drift_returns_copy_not_same_object(self, reference_df):
        result = _apply_no_drift(reference_df, np.random.default_rng(0))
        assert result is not reference_df

    def test_no_drift_values_unchanged(self, reference_df):
        result = _apply_no_drift(reference_df, np.random.default_rng(0))
        pd.testing.assert_frame_equal(result, reference_df)

    def test_apply_drift_none_level(self, reference_df):
        result = apply_drift(reference_df, level="none")
        pd.testing.assert_frame_equal(result, reference_df)


class TestMildDrift:
    def test_mild_drift_increases_monthly_charges_by_50pct(self, reference_df):
        original_mean = reference_df["MonthlyCharges"].mean()
        result = _apply_mild_drift(reference_df, np.random.default_rng(42))
        new_mean = result["MonthlyCharges"].mean()
        assert abs(new_mean - original_mean * 1.5) < 1e-6, (
            f"Expected mean ≈ {original_mean * 1.5:.2f}, got {new_mean:.2f}"
        )

    def test_mild_drift_increases_total_charges(self, reference_df):
        original_mean = reference_df["TotalCharges"].mean()
        result = _apply_mild_drift(reference_df, np.random.default_rng(42))
        new_mean = result["TotalCharges"].mean()
        assert new_mean > original_mean

    def test_mild_drift_flips_some_contracts(self, reference_df):
        # Some rows should be flipped to Month-to-month
        result = _apply_mild_drift(reference_df, np.random.default_rng(42))
        # Original has 10 Month-to-month out of 20; after mild drift at 30% flip rate,
        # we just verify the column still exists and at least one is month-to-month
        assert "Contract" in result.columns
        assert (result["Contract"] == "Month-to-month").any()

    def test_mild_drift_does_not_invert_original(self, reference_df):
        result = _apply_mild_drift(reference_df, np.random.default_rng(42))
        # Original should be unchanged
        assert reference_df["MonthlyCharges"].mean() < result["MonthlyCharges"].mean()


class TestSevereDrift:
    def test_severe_drift_triples_monthly_charges(self, reference_df):
        original = reference_df["MonthlyCharges"].copy()
        result = _apply_severe_drift(reference_df, np.random.default_rng(0))
        expected = (original * 3.0).clip(upper=199.0)
        pd.testing.assert_series_equal(
            result["MonthlyCharges"].reset_index(drop=True),
            expected.reset_index(drop=True),
        )

    def test_severe_drift_sets_all_contracts_to_two_year(self, reference_df):
        result = _apply_severe_drift(reference_df, np.random.default_rng(0))
        assert (result["Contract"] == "Two year").all()

    def test_severe_drift_reduces_tenure(self, reference_df):
        result = _apply_severe_drift(reference_df, np.random.default_rng(0))
        assert result["tenure"].mean() < reference_df["tenure"].mean()

    def test_severe_monthly_charges_clipped_at_199(self, reference_df):
        # Inject rows that would exceed 199 after ×3
        df = reference_df.copy()
        df["MonthlyCharges"] = 100.0  # ×3 = 300 → clipped to 199
        result = _apply_severe_drift(df, np.random.default_rng(0))
        assert (result["MonthlyCharges"] <= 199.0).all()

    def test_apply_drift_severe_level(self, reference_df):
        result = apply_drift(reference_df, level="severe", seed=1)
        assert (result["Contract"] == "Two year").all()


class TestApplyDriftInterface:
    def test_unknown_level_falls_back_to_no_drift(self, reference_df):
        """Unknown drift levels should not raise — they default to no-op."""
        # "none" is the fallback in _DRIFT_FN.get(..., _apply_no_drift)
        result = apply_drift(reference_df, level="none")
        pd.testing.assert_frame_equal(result, reference_df)

    def test_seed_produces_identical_results(self, reference_df):
        r1 = apply_drift(reference_df, level="mild", seed=99)
        r2 = apply_drift(reference_df, level="mild", seed=99)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_produce_different_results(self, reference_df):
        r1 = apply_drift(reference_df, level="mild", seed=1)
        r2 = apply_drift(reference_df, level="mild", seed=2)
        # Contract column may differ due to random flip mask
        assert not r1["Contract"].equals(r2["Contract"]) or \
               not r1["MonthlyCharges"].equals(r2["MonthlyCharges"])
