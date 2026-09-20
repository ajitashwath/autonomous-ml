

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ModelNotFoundError, ModelRollbackError
from src.rollback_manager.rollback import execute_rollback


def _make_info(version: str) -> MagicMock:
    info = MagicMock()
    info.version = version
    return info


@pytest.fixture
def mock_registry():
    with patch("src.rollback_manager.rollback.ModelRegistry") as mock:
        yield mock.return_value


def test_execute_rollback_success(mock_registry):
    # get_model_info() returns the current prod (to be demoted)
    mock_registry.get_model_info.return_value = _make_info("2")
    # rollback() returns the restored ModelInfo
    mock_registry.rollback.return_value = _make_info("1")

    execute_rollback(reason="Model crashed")

    mock_registry.rollback.assert_called_once_with(target_version=None)
    assert mock_registry.annotate_version.call_count == 2

    demoted_call = mock_registry.annotate_version.call_args_list[0]
    assert demoted_call.kwargs["version"] == "2"
    assert "Model crashed" in demoted_call.kwargs["description"]

    restored_call = mock_registry.annotate_version.call_args_list[1]
    assert restored_call.kwargs["version"] == "1"
    assert "Emergency rollback replacement" in restored_call.kwargs["description"]


def test_execute_rollback_with_target_version(mock_registry):
    mock_registry.get_model_info.return_value = _make_info("6")
    mock_registry.rollback.return_value = _make_info("5")

    execute_rollback(reason="Rolling back to stable V5", target_version="5")
    mock_registry.rollback.assert_called_once_with(target_version="5")


def test_execute_rollback_no_current_prod(mock_registry):
    """When there is no current production model, rollback should still work
    and only annotate the restored version (no demotion annotation)."""
    mock_registry.get_model_info.side_effect = ModelNotFoundError("no prod model")
    mock_registry.rollback.return_value = _make_info("3")

    execute_rollback(reason="Restoring to v3")

    mock_registry.rollback.assert_called_once_with(target_version=None)
    # Only one annotation (the restored version), no demotion call
    assert mock_registry.annotate_version.call_count == 1
    only_call = mock_registry.annotate_version.call_args_list[0]
    assert only_call.kwargs["version"] == "3"


def test_execute_rollback_failure_crashes(mock_registry):
    mock_registry.get_model_info.side_effect = ModelNotFoundError("no prod")
    mock_registry.rollback.side_effect = ModelRollbackError("No archived models found")
    with pytest.raises(SystemExit) as exc:
        execute_rollback(reason="Emergency")
    assert exc.value.code == 1
    mock_registry.annotate_version.assert_not_called()