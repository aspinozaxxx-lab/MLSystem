from __future__ import annotations

import unittest
from unittest.mock import patch

from mlsystem.src.pipeline_runner.runner import _pid_running


class PipelineRunnerStatusTests(unittest.TestCase):
    def test_pid_running_treats_linux_zombie_as_exited(self) -> None:
        with (
            patch("mlsystem.src.pipeline_runner.runner._proc_state_for_pid", return_value="Z"),
            patch("mlsystem.src.pipeline_runner.runner.os.kill") as kill,
        ):
            self.assertFalse(_pid_running(123))
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
