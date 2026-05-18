from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlsystem.src.pipeline_runner.api import PipelineRunConfig, PipelineRunStore
from mlsystem.src.pipeline_runner.worker import run_worker


class PipelineRunnerWorkerTests(unittest.TestCase):
    def test_fake_stages_run_sequentially_and_reach_100(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PipelineRunStore(Path(tmp))
            store.create_run(PipelineRunConfig.model_validate({"run_id": "run1", "experiment_id": "unit", "pipeline": {"stages": ["inventory_scenes", "prepare_dataset"]}}))
            calls: list[str] = []

            def fake_run_stage(stage, _config, stage_store, run_id):
                calls.append(stage)
                return stage_store.write_stage_report(run_id, stage, {"status": "success", "summary": stage})

            with patch("mlsystem.src.pipeline_runner.worker.stages.run_stage", side_effect=fake_run_stage):
                rc = run_worker("run1", Path(tmp))
            self.assertEqual(rc, 0)
            self.assertEqual(calls, ["inventory_scenes", "prepare_dataset"])
            status = store.read_run("run1")
            self.assertEqual(status["state"], "succeeded")
            self.assertEqual(status["progress_percent"], 100)

    def test_worker_imports_raw_stage_payload_into_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PipelineRunStore(Path(tmp))
            store.create_run(PipelineRunConfig.model_validate({"run_id": "raw_payload", "experiment_id": "unit", "pipeline": {"stages": ["train_model"]}}))

            def fake_run_stage(_stage, _config, _stage_store, _run_id):
                return {
                    "status": "success",
                    "summary": "raw dispatcher payload",
                    "mlflow": {"run_id": "mlflow-run-1", "experiment_id": "40"},
                    "artifacts": {"training_result.json": "training_result.json"},
                }

            with patch("mlsystem.src.pipeline_runner.worker.stages.run_stage", side_effect=fake_run_stage):
                rc = run_worker("raw_payload", Path(tmp))

            self.assertEqual(rc, 0)
            status = store.read_run("raw_payload")
            self.assertEqual(status["state"], "succeeded")
            self.assertEqual(status["stages"][0]["state"], "succeeded")
            self.assertEqual(status["stages"][0]["status"], "success")
            self.assertEqual(status["mlflow"]["run_id"], "mlflow-run-1")
            self.assertEqual(status["artifacts"]["training_result.json"], "training_result.json")

    def test_failure_stops_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PipelineRunStore(Path(tmp))
            store.create_run(PipelineRunConfig.model_validate({"run_id": "run2", "experiment_id": "unit", "pipeline": {"stages": ["inventory_scenes", "prepare_dataset"]}}))

            def fake_run_stage(stage, _config, stage_store, run_id):
                if stage == "prepare_dataset":
                    raise RuntimeError("boom")
                return stage_store.write_stage_report(run_id, stage, {"status": "success", "summary": stage})

            with patch("mlsystem.src.pipeline_runner.worker.stages.run_stage", side_effect=fake_run_stage):
                rc = run_worker("run2", Path(tmp))
            self.assertEqual(rc, 1)
            status = store.read_run("run2")
            self.assertEqual(status["state"], "failed")
            self.assertIn("boom", str(status["error"]))

    def test_cancel_marker_stops_before_next_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PipelineRunStore(Path(tmp))
            store.create_run(PipelineRunConfig.model_validate({"run_id": "run3", "experiment_id": "unit", "pipeline": {"stages": ["inventory_scenes", "prepare_dataset"]}}))

            def fake_run_stage(stage, _config, stage_store, run_id):
                report = stage_store.write_stage_report(run_id, stage, {"status": "success", "summary": stage})
                stage_store.mark_cancel_requested(run_id)
                return report

            with patch("mlsystem.src.pipeline_runner.worker.stages.run_stage", side_effect=fake_run_stage):
                rc = run_worker("run3", Path(tmp))
            self.assertEqual(rc, 2)
            self.assertEqual(store.read_run("run3")["state"], "cancelled")


if __name__ == "__main__":
    unittest.main()
