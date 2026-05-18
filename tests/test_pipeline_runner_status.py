from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from mlsystem.src.pipeline_runner.api import PipelineRunConfig, PipelineRunStore
from mlsystem.src.pipeline_runner.runner import _pid_running, _worker_exit_diagnostic


class PipelineRunnerStatusTests(unittest.TestCase):
    def test_pid_running_treats_linux_zombie_as_exited(self) -> None:
        with (
            patch("mlsystem.src.pipeline_runner.runner._proc_state_for_pid", return_value="Z"),
            patch("mlsystem.src.pipeline_runner.runner.os.kill") as kill,
        ):
            self.assertFalse(_pid_running(123))
            kill.assert_not_called()

    def test_worker_exit_diagnostic_includes_log_tails(self) -> None:
        with TemporaryDirectory() as tmp:
            store = PipelineRunStore(tmp)
            config = PipelineRunConfig.model_validate(
                {"run_id": "diag_run", "experiment_id": "diag", "pipeline": {"stages": ["train_model"]}}
            )
            store.create_run(config)
            bound = store.bind("diag_run")
            (bound.log_dir / "worker_stderr.log").write_text("fatal worker stderr\n", encoding="utf-8")
            (bound.log_dir / "train_model.log").write_text("train stage log\n", encoding="utf-8")
            diagnostic = _worker_exit_diagnostic(
                store,
                "diag_run",
                {"pid": 123, "current_stage": "train_model"},
            )
            self.assertEqual(diagnostic["last_stage"], "train_model")
            self.assertFalse(diagnostic["exit_code_available"])
            self.assertIn("fatal worker stderr", diagnostic["worker_stderr_tail"])
            self.assertIn("train stage log", diagnostic["stage_log_tail"])


if __name__ == "__main__":
    unittest.main()
