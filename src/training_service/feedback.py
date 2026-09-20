from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.core.logging import get_logger

logger = get_logger(__name__)

_CREATED_AT = "_created_at"
DEFAULT_MIN_LABELLED_ROWS = 300
DEFAULT_EVAL_FRACTION = 0.3


@dataclass
class FeedbackData:
    """Labelled production rows (features + target), split by time.

    `train` is the older part and is added to the training set. `evaluation` is the newest
    part; it is never trained on and is what the validation gate scores candidates against.
    """

    train: pd.DataFrame
    evaluation: pd.DataFrame


def fetch_labelled_rows(target_column: str) -> pd.DataFrame:
    """All logged predictions that have a ground-truth label, oldest first."""
    # Imported here so the rest of the training code has no hard dependency on the database.
    from sqlalchemy import select

    from src.core.db import db_session
    from src.data_logger.models import PredictionLog

    with db_session() as session:
        rows = session.scalars(
            select(PredictionLog)
            .where(PredictionLog.actual_label.is_not(None))
            .order_by(PredictionLog.created_at.asc())
        ).all()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([row.features for row in rows])
        df[target_column] = [int(row.actual_label) for row in rows]
        df[_CREATED_AT] = [row.created_at for row in rows]
        return df


def prepare_feedback(
    rows: pd.DataFrame,
    target_column: str,
    eval_fraction: float = DEFAULT_EVAL_FRACTION,
    min_rows: int = DEFAULT_MIN_LABELLED_ROWS,
) -> FeedbackData | None:
    """Split labelled rows chronologically; None when there are too few to be worth using."""
    if len(rows) < min_rows:
        return None

    ordered = rows.sort_values(_CREATED_AT, kind="stable").drop(columns=[_CREATED_AT])
    # Hold out the *most recent* rows. Random splitting would let the model train on
    # near-duplicates of what it is scored on; chronological is also how the model is used.
    n_eval = max(1, int(round(len(ordered) * eval_fraction)))
    train = ordered.iloc[:-n_eval].reset_index(drop=True)
    evaluation = ordered.iloc[-n_eval:].reset_index(drop=True)
    return FeedbackData(train=train, evaluation=evaluation)


def load_feedback(config: dict[str, Any] | None, target_column: str) -> FeedbackData | None:
    """Labelled production data for this training run, or None to train on the CSV alone."""
    config = config or {}
    if not config.get("enabled", False):
        logger.info("feedback_disabled")
        return None

    min_rows = config.get("min_labelled_rows", DEFAULT_MIN_LABELLED_ROWS)
    try:
        rows = fetch_labelled_rows(target_column)
    except Exception as exc:
        # Feedback is an enhancement: an unreachable database must not stop retraining.
        logger.warning("feedback_unavailable_training_on_csv_only", error=str(exc))
        return None

    feedback = prepare_feedback(
        rows,
        target_column,
        eval_fraction=config.get("eval_fraction", DEFAULT_EVAL_FRACTION),
        min_rows=min_rows,
    )
    if feedback is None:
        logger.info(
            "feedback_insufficient_training_on_csv_only",
            labelled_rows=len(rows),
            required=min_rows,
        )
        return None

    logger.info(
        "feedback_loaded",
        train_rows=len(feedback.train),
        evaluation_rows=len(feedback.evaluation),
    )
    return feedback
