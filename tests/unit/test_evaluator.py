

import numpy as np
import pytest

from src.training_service.evaluator import EvaluationResult, evaluate


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.1, 0.9, 0.95, 0.85])
    return y_true, y_proba


@pytest.fixture
def random_predictions():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_proba = rng.uniform(0, 1, size=200)
    return y_true, y_proba


def test_evaluate_returns_result(perfect_predictions):
    y_true, y_proba = perfect_predictions
    result = evaluate(y_true, y_proba)
    assert isinstance(result, EvaluationResult)


def test_perfect_predictions_high_auc(perfect_predictions):
    y_true, y_proba = perfect_predictions
    result = evaluate(y_true, y_proba)
    assert result.roc_auc > 0.95


def test_random_predictions_near_chance(random_predictions):
    y_true, y_proba = random_predictions
    result = evaluate(y_true, y_proba)
    assert 0.35 < result.roc_auc < 0.65


def test_to_dict_keys(perfect_predictions):
    y_true, y_proba = perfect_predictions
    result = evaluate(y_true, y_proba)
    d = result.to_dict()
    for key in ["roc_auc", "f1", "precision", "recall", "log_loss"]:
        assert key in d, f"Missing key: {key}"
        assert isinstance(d[key], float)


def test_confusion_matrix_sums(perfect_predictions):
    y_true, y_proba = perfect_predictions
    result = evaluate(y_true, y_proba)
    total = (
        result.true_negatives + result.false_positives
        + result.false_negatives + result.true_positives
    )
    assert total == len(y_true)


def test_threshold_affects_predictions():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.6, 0.6, 0.6, 0.6])
    result_low = evaluate(y_true, y_proba, threshold=0.5)
    result_high = evaluate(y_true, y_proba, threshold=0.9)
    assert result_low.recall == 1.0
    assert result_high.recall == 0.0
