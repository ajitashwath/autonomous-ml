

import numpy as np
import pandas as pd
import pytest

from src.training_service.preprocessor import build_preprocessor, fit_transform


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame({
        "tenure":         rng.integers(1, 72, size=n).astype(float),
        "MonthlyCharges": rng.uniform(20, 120, size=n),
        "TotalCharges":   rng.uniform(100, 8000, size=n),
        "Contract":       rng.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaymentMethod":  rng.choice(["Credit card", "Bank transfer", "Mailed check"], size=n),
        "InternetService":rng.choice(["DSL", "Fiber optic", "No"], size=n),
    })
    return df


NUMERICAL = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "PaymentMethod", "InternetService"]


def test_preprocessor_output_shape(sample_data):
    preprocessor = build_preprocessor(CATEGORICAL, NUMERICAL)
    X_train, X_test = sample_data[:80], sample_data[80:]
    X_tr, X_te, features = fit_transform(preprocessor, X_train, X_test)
    assert X_tr.shape[0] == 80
    assert X_te.shape[0] == 20
    assert X_tr.shape[1] == X_te.shape[1]


def test_feature_names_not_empty(sample_data):
    preprocessor = build_preprocessor(CATEGORICAL, NUMERICAL)
    X_train, X_test = sample_data[:80], sample_data[80:]
    _, _, features = fit_transform(preprocessor, X_train, X_test)
    assert len(features) > 0
    assert all(isinstance(f, str) for f in features)


def test_no_nan_in_output(sample_data):
    sample_data.at[0, "MonthlyCharges"] = np.nan
    preprocessor = build_preprocessor(CATEGORICAL, NUMERICAL)
    X_train, X_test = sample_data[:80], sample_data[80:]
    X_tr, X_te, _ = fit_transform(preprocessor, X_train, X_test)
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_te).any()
