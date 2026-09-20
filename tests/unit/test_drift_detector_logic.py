import pytest

from src.drift_detector.detector import (
    _in_cooldown,
    _record_trigger,
    extract_drift_summary,
)


def _report(**result):
    return {"metrics": [{"metric": "DatasetDriftMetric", "result": result}]}


def test_extract_uses_observed_share_not_the_configured_threshold():
    """Evidently reports its *threshold* as `drift_share` and the measurement as
    `share_of_drifted_columns`. Reading the wrong one made every window look drifted."""
    summary = extract_drift_summary(_report(
        drift_share=0.5,
        share_of_drifted_columns=0.25,
        number_of_columns=4,
        number_of_drifted_columns=1,
        dataset_drift=False,
    ))

    assert summary["share_of_drifted_columns"] == 0.25
    assert summary["number_of_drifted_columns"] == 1
    assert summary["number_of_columns"] == 4
    assert summary["dataset_drift"] is False


def test_extract_finds_the_metric_regardless_of_position():
    report = {"metrics": [
        {"metric": "DataDriftTable", "result": {"something": "else"}},
        {"metric": "DatasetDriftMetric", "result": {
            "share_of_drifted_columns": 1.0, "number_of_columns": 3,
            "number_of_drifted_columns": 3, "dataset_drift": True,
        }},
    ]}

    assert extract_drift_summary(report)["share_of_drifted_columns"] == 1.0


def test_extract_raises_when_the_metric_is_absent():
    with pytest.raises(KeyError):
        extract_drift_summary({"metrics": [{"result": {"drift_share": 0.5}}]})


class TestCooldown:
    def test_no_state_file_means_no_cooldown(self, tmp_path):
        assert not _in_cooldown(tmp_path / "state.json", 3600, now=1000.0)

    def test_within_and_after_cooldown(self, tmp_path):
        state = tmp_path / "state.json"
        _record_trigger(state, now=1000.0)

        assert _in_cooldown(state, 3600, now=1000.0 + 3599)
        assert not _in_cooldown(state, 3600, now=1000.0 + 3600)

    def test_zero_cooldown_never_suppresses(self, tmp_path):
        state = tmp_path / "state.json"
        _record_trigger(state, now=1000.0)

        assert not _in_cooldown(state, 0, now=1000.0)

    def test_corrupt_state_file_is_treated_as_no_cooldown(self, tmp_path):
        state = tmp_path / "state.json"
        state.write_text("not json")

        assert not _in_cooldown(state, 3600, now=1000.0)
