"""Unit tests for src.core.config"""

import os

import pytest

from src.core.config import Settings, get_settings


def test_default_settings():
    """Settings should load with sane defaults when no env vars are set."""
    s = Settings()
    assert s.project_name == "automlops"
    assert s.env == "development"
    assert s.log_level == "INFO"
    assert s.drift_window_size == 1000
    assert 0.0 <= s.drift_share_threshold <= 1.0


def test_settings_env_override(monkeypatch):
    """Environment variables should override defaults."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DRIFT_WINDOW_SIZE", "500")
    s = Settings()
    assert s.log_level == "DEBUG"
    assert s.drift_window_size == 500


def test_get_settings_is_cached():
    """get_settings() should return the same object on repeated calls."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
