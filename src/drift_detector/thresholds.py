"""
drift_detector/thresholds.py

Centralized module for defining and loading drift thresholds.
Can be expanded to include custom statistical tests not supported by Evidently.
"""

import yaml


def get_drift_thresholds(config_path: str = "configs/drift.yaml") -> dict:
    """Reads the static drift thresholds from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["detection"]


def get_critical_features(config_path: str = "configs/drift.yaml") -> list[str]:
    """Reads the list of critical features that always require alerts if drifted."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["alerting"].get("critical_features", [])
