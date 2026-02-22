"""Unit tests for src.core.exceptions"""

import pytest

from src.core.exceptions import (
    AutoMLOpsError,
    DataValidationError,
    ModelNotFoundError,
    PipelineTriggerError,
)


def test_base_exception_message():
    exc = AutoMLOpsError("something broke", details={"code": 42})
    assert exc.message == "something broke"
    assert exc.details == {"code": 42}


def test_typed_exception_is_subclass_of_base():
    assert issubclass(ModelNotFoundError, AutoMLOpsError)
    assert issubclass(DataValidationError, AutoMLOpsError)
    assert issubclass(PipelineTriggerError, AutoMLOpsError)


def test_exception_repr():
    exc = ModelNotFoundError("no model found", details={"stage": "Production"})
    r = repr(exc)
    assert "ModelNotFoundError" in r
    assert "no model found" in r
