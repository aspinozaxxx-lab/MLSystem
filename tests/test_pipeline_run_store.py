from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlsystem.src.pipeline_runner.config import PipelineRunConfig
from mlsystem.src.pipeline_runner.run_store import PipelineRunStore


class PipelineRunStoreTests(unittest.TestCase):
    def test_create_update_log_and_stage_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PipelineRunStore(Path(tmp))
            run = store.create_run(PipelineRunConfig.model_validate({"run_id": "run1", "experiment_id": "unit", "pipeline": {"stages": ["inventory"]}}))
            self.assertEqual(run["state"], "queued")
            self.assertTrue((Path(tmp) / "run1" / "trace.json").exists())

            store.update_run("run1", state="running", progress_percent=10)
            self.assertEqual(store.read_run("run1")["progress_percent"], 10)

            store.append_log("run1", "hello\n")
            self.assertEqual(store.tail_log("run1"), "hello\n")

            report = store.write_stage_report("run1", "inventory_scenes", {"status": "success", "summary": "ok", "artifacts": {"a": "/tmp/a"}})
            self.assertEqual(report["status"], "success")
            self.assertTrue((Path(tmp) / "run1" / "stages" / "inventory_scenes.json").exists())
            self.assertIn("inventory_scenes", store.read_summary("run1")["stages"])

            store.mark_cancel_requested("run1")
            self.assertTrue(store.cancel_requested("run1"))


if __name__ == "__main__":
    unittest.main()
