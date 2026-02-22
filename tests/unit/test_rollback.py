"""Unit tests for the Rollback Manager."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import ModelRollbackError
from src.rollback_manager.rollback import execute_rollback


@pytest.fixture
def mock_registry():
    with patch("src.rollback_manager.rollback.ModelRegistry") as mock:
        yield mock.return_value


def test_execute_rollback_success(mock_registry):
    """Test successful demotion and restoration tracking."""
    # Registry mock returns (restored_version, demoted_version)
    mock_registry.rollback.return_value = ("1", "2")
    
    execute_rollback(reason="Model crashed")
    
    mock_registry.rollback.assert_called_once_with(target_version=None)
    
    # 2 annotations should be created (one for demoted, one for restored)
    assert mock_registry.annotate_version.call_count == 2
    
    # Check annotation on demoted model
    demoted_call = mock_registry.annotate_version.call_args_list[0]
    assert demoted_call.kwargs["version"] == "2"
    assert "Model crashed" in demoted_call.kwargs["description"]
    
    # Check annotation on restored model
    restored_call = mock_registry.annotate_version.call_args_list[1]
    assert restored_call.kwargs["version"] == "1"
    assert "Emergency rollback replacement" in restored_call.kwargs["description"]


def test_execute_rollback_with_target_version(mock_registry):
    """Test rollback to a specific version instead of latest archived."""
    mock_registry.rollback.return_value = ("5", "6")
    
    execute_rollback(reason="Rolling back to stable V5", target_version="5")
    
    mock_registry.rollback.assert_called_once_with(target_version="5")


def test_execute_rollback_failure_crashes(mock_registry):
    """Test that a failure in the registry causes the CLI to crash (SysExit 1)."""
    mock_registry.rollback.side_effect = ModelRollbackError("No archived models found")
    
    with pytest.raises(SystemExit) as exc:
        execute_rollback(reason="Emergency")
        
    assert exc.value.code == 1
    mock_registry.annotate_version.assert_not_called()
