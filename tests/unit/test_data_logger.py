import uuid
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.data_logger import logger as data_logger
from src.data_logger.models import Base, PredictionLog


@pytest.fixture()
def factory(monkeypatch):
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    @contextmanager
    def fake_db_session():
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr(data_logger, "db_session", fake_db_session)
    return session_factory


def _record(i: int = 0) -> dict:
    return {
        "request_id": str(uuid.uuid4()),
        "features": {"tenure": i, "Contract": "Two year"},
        "prediction": i % 2,
        "probability": 0.1 * (i % 10),
        "model_version": "3",
    }


def _rows(factory) -> list[PredictionLog]:
    with factory() as session:
        return session.query(PredictionLog).all()


def test_all_records_are_inserted(factory):
    data_logger.log_predictions_safe([_record(i) for i in range(50)])

    rows = _rows(factory)
    assert len(rows) == 50
    assert all(r.drift_analyzed is False for r in rows)
    assert rows[0].features["Contract"] == "Two year"


def test_batch_uses_a_single_session_and_transaction(factory):
    with patch.object(data_logger, "db_session", wraps=data_logger.db_session) as spy:
        data_logger.log_predictions_safe([_record(i) for i in range(25)])

    assert spy.call_count == 1


def test_is_atomic_a_bad_record_stores_nothing(factory):
    bad = {**_record(1), "unknown_column": 1}

    data_logger.log_predictions_safe([_record(0), bad])  # must not raise

    assert _rows(factory) == []


def test_empty_batch_is_a_noop(factory):
    with patch.object(data_logger, "db_session") as session:
        data_logger.log_predictions_safe([])

    session.assert_not_called()


def test_database_outage_is_swallowed():
    with patch.object(data_logger, "db_session", side_effect=RuntimeError("db down")):
        data_logger.log_predictions_safe([_record()])  # must not raise
