from unittest.mock import patch

import pytest

from src.drift_detector import detector
from src.drift_detector.detector import FAILURE_OUTCOMES, Outcome, main


@pytest.fixture(autouse=True)
def _no_logging_setup():
    with patch("src.core.logging.configure_logging"):
        yield


class TestOnce:
    @pytest.mark.parametrize(
        "outcome",
        [Outcome.NO_DRIFT, Outcome.DRIFT_DETECTED, Outcome.NO_NEW_LOGS, Outcome.INSUFFICIENT_SAMPLES],
    )
    def test_healthy_outcomes_exit_zero(self, outcome):
        with patch.object(detector, "run_drift_detection", return_value=outcome):
            assert main(["--once"]) == 0

    @pytest.mark.parametrize("outcome", sorted(FAILURE_OUTCOMES))
    def test_outcomes_that_mean_the_detector_cannot_work_exit_two(self, outcome):
        """A CronJob whose reference data is missing must not stay green forever."""
        with patch.object(detector, "run_drift_detection", return_value=outcome):
            assert main(["--once"]) == 2

    def test_a_crash_exits_one(self):
        with patch.object(detector, "run_drift_detection", side_effect=RuntimeError("boom")):
            assert main(["--once"]) == 1

    def test_runs_exactly_once_and_never_sleeps(self):
        with patch.object(detector, "run_drift_detection", return_value=Outcome.NO_DRIFT) as run, \
             patch.object(detector.time, "sleep") as sleep:
            main(["--once"])

        run.assert_called_once()
        sleep.assert_not_called()

    def test_passes_the_config_path_through(self):
        with patch.object(detector, "run_drift_detection", return_value=Outcome.NO_DRIFT) as run:
            main(["--once", "--config", "custom.yaml"])

        run.assert_called_once_with("custom.yaml")


class TestDaemon:
    def test_loops_forever_surviving_crashes(self):
        class StopLoopError(Exception):
            pass

        calls = []

        def run(_config):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("transient")

        def sleep(_seconds):
            if len(calls) >= 3:
                raise StopLoopError

        with patch.object(detector, "run_drift_detection", side_effect=run), \
             patch.object(detector, "load_drift_config", return_value={"detection": {"check_interval_seconds": 1}}), \
             patch.object(detector.time, "sleep", side_effect=sleep):
            with pytest.raises(StopLoopError):
                main([])

        assert len(calls) == 3
