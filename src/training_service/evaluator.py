"""
training_service/evaluator.py

Computes all evaluation metrics for a trained model.

Design decisions:
  - Returns a typed EvaluationResult dataclass — downstream code (MLflow logging,
    validation_gate) gets named fields, not positional tuple indices.
  - All metrics are computed with the SAME threshold so comparisons are fair.
  - Confusion matrix is included because precision/recall alone hide class imbalance.

Production note:
  - For production models, add calibration plots (Brier score, reliability diagram).
  - For regulated industries (credit, insurance), add fairness metrics here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Typed container for all model evaluation metrics."""
    roc_auc: float
    f1: float
    precision: float
    recall: float
    log_loss: float
    threshold: float
    # Confusion matrix quadrants (useful for imbalanced datasets)
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, float]:
        """Flat dict suitable for MLflow log_metrics()."""
        return {k: float(v) for k, v in asdict(self).items()}

    def summary(self) -> str:
        return (
            f"AUC={self.roc_auc:.4f} | F1={self.f1:.4f} | "
            f"P={self.precision:.4f} | R={self.recall:.4f} | "
            f"LogLoss={self.log_loss:.4f}"
        )


def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    """
    Compute all evaluation metrics from true labels and predicted probabilities.

    Args:
        y_true:    Ground truth binary labels (0 / 1).
        y_proba:   Predicted probabilities for the positive class.
        threshold: Decision boundary for converting probabilities to labels.

    Returns:
        EvaluationResult with all metrics populated.
    """
    y_pred = (y_proba >= threshold).astype(int)

    auc   = roc_auc_score(y_true, y_proba)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    ll    = log_loss(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    result = EvaluationResult(
        roc_auc=auc,
        f1=f1,
        precision=prec,
        recall=rec,
        log_loss=ll,
        threshold=threshold,
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )

    logger.info("model_evaluated", **{k: round(v, 4) for k, v in result.to_dict().items()})
    return result
