import yaml


def get_drift_thresholds(config_path: str = "configs/drift.yaml") -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["detection"]

def get_critical_features(config_path: str = "configs/drift.yaml") -> list[str]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["alerting"].get("critical_features", [])
