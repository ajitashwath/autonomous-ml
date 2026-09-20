from unittest.mock import patch

import pytest

from src.data_logger import labels as labels_module
from src.data_logger.labels import record_labels


@pytest.fixture()
def db(sqlite_db):
    with patch.object(labels_module, "db_session", sqlite_db.session):
        yield sqlite_db


def test_label_is_attached_to_the_matching_prediction(db):
    db.add_prediction("req-1", {"tenure": 3}, prediction=1)

    result = record_labels([("req-1", 1)])

    row = db.rows()["req-1"]
    assert row.actual_label == 1
    assert row.label_received_at is not None
    assert result.updated == 1


def test_reports_whether_each_served_prediction_was_right(db):
    # Deliberately unequal (2 right, 1 wrong) so swapped counters would be detected.
    db.add_prediction("right-1", {}, prediction=1)
    db.add_prediction("right-2", {}, prediction=0)
    db.add_prediction("wrong", {}, prediction=1)

    result = record_labels([("right-1", 1), ("right-2", 0), ("wrong", 0)])

    assert (result.correct, result.incorrect) == (2, 1)


def test_unknown_request_ids_are_reported_not_fatal(db):
    db.add_prediction("known", {}, prediction=0)

    result = record_labels([("known", 0), ("never-logged", 1)])

    assert result.updated == 1
    assert result.unknown_request_ids == ["never-logged"]
    assert db.rows()["known"].actual_label == 0


def test_a_later_label_corrects_an_earlier_one(db):
    db.add_prediction("req-1", {}, prediction=1)
    record_labels([("req-1", 0)])

    record_labels([("req-1", 1)])

    assert db.rows()["req-1"].actual_label == 1


def test_only_the_named_rows_are_touched(db):
    db.add_prediction("a", {}, prediction=0)
    db.add_prediction("b", {}, prediction=0)

    record_labels([("a", 1)])

    assert db.rows()["b"].actual_label is None


def test_empty_input_does_not_touch_the_database():
    with patch.object(labels_module, "db_session") as session:
        result = record_labels([])

    session.assert_not_called()
    assert result.updated == 0
