import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

from src.core.db import add_missing_columns
from src.data_logger.models import Base, PredictionLog

# The prediction_logs table as it existed before ground-truth labels were introduced.
LEGACY_DDL = """
CREATE TABLE prediction_logs (
    id CHAR(32) NOT NULL PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    features JSON NOT NULL,
    prediction INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    drift_analyzed BOOLEAN NOT NULL,
    created_at DATETIME NOT NULL
)
"""


@pytest.fixture()
def legacy_engine():
    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text(LEGACY_DDL))
        conn.execute(text(
            "INSERT INTO prediction_logs VALUES "
            "('00000000000000000000000000000001', 'req-1', '{}', 1, 0.9, '3', 0, "
            "'2026-01-01 00:00:00')"
        ))
    return engine


def _columns(engine) -> set[str]:
    return {c["name"] for c in inspect(engine).get_columns("prediction_logs")}


def test_create_all_alone_does_not_upgrade_an_existing_table(legacy_engine):
    """The reason the migration exists: this is the failure it prevents."""
    Base.metadata.create_all(legacy_engine)

    assert "actual_label" not in _columns(legacy_engine)


def test_adds_the_label_columns_to_an_existing_table(legacy_engine):
    added = add_missing_columns(legacy_engine, Base.metadata)

    assert sorted(added) == ["prediction_logs.actual_label", "prediction_logs.label_received_at"]
    assert {"actual_label", "label_received_at"} <= _columns(legacy_engine)


def test_existing_rows_survive_with_null_labels(legacy_engine):
    add_missing_columns(legacy_engine, Base.metadata)

    with legacy_engine.connect() as conn:
        row = conn.execute(text(
            "SELECT request_id, prediction, actual_label, label_received_at FROM prediction_logs"
        )).one()
    assert tuple(row) == ("req-1", 1, None, None)


def test_creates_the_index_for_indexed_columns(legacy_engine):
    add_missing_columns(legacy_engine, Base.metadata)

    index_names = {i["name"] for i in inspect(legacy_engine).get_indexes("prediction_logs")}
    assert "ix_prediction_logs_actual_label" in index_names


def test_is_idempotent(legacy_engine):
    add_missing_columns(legacy_engine, Base.metadata)

    assert add_missing_columns(legacy_engine, Base.metadata) == []


def test_upgraded_table_works_with_the_orm(legacy_engine):
    from sqlalchemy.orm import Session

    add_missing_columns(legacy_engine, Base.metadata)

    with Session(legacy_engine) as session:
        row = session.query(PredictionLog).one()
        row.actual_label = 1
        session.commit()
        assert session.query(PredictionLog).filter(PredictionLog.actual_label == 1).count() == 1


def test_fresh_database_needs_no_migration():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)

    assert add_missing_columns(engine, Base.metadata) == []


def test_missing_table_is_left_to_create_all():
    engine = create_engine("sqlite://", poolclass=StaticPool)

    assert add_missing_columns(engine, Base.metadata) == []
