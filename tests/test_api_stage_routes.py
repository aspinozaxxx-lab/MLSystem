from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlsystem.src.api.job_runner import JobRunner
from mlsystem.src.api.job_store import JobStore
from mlsystem.src.api.models import StageStartRequest
from mlsystem.src.api.stage_routes import debug_run_stage_sync, stages_payload, start_stage, validate_stage_name


class ApiStageRoutesTests(unittest.TestCase):
    def test_stages_payload_contains_main_stages(self) -> None:
        payload = stages_payload()
        self.assertIn("inventory_scenes", payload["main_dag_stages"])
        self.assertIn("run_pseudolabel_inference", payload["main_dag_stages"])
        self.assertEqual(payload["aliases"]["stitch_probability_maps"], "validate_probability_maps")

    def test_unknown_stage_fails_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown stage"):
            validate_stage_name("unknown_stage")

    def test_start_stage_dry_run_creates_succeeded_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = JobRunner(JobStore(Path(tmp)))
            request = StageStartRequest(experiment_config={"experiment_id": "unit"}, dry_run=True)
            response = start_stage("run1", "inventory_scenes", request, runner)
            self.assertEqual(response.state, "succeeded")
            status = runner.store.read_job(response.job_id)
            self.assertEqual(status.state, "succeeded")

    def test_debug_sync_failed_stage_returns_failed_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            request = StageStartRequest(
                experiment_config={
                    "experiment_id": "unit_api_failure",
                    "pseudolabel": {"run_on": "explicit_scene_list", "scene_list": ["missing.tif"]},
                },
                status_root=str(Path(tmp) / "status"),
            )
            status = debug_run_stage_sync("unit_api_failure", "prepare_inference_scenes", request, store)
            self.assertEqual(status.state, "failed")
            self.assertIn("missing", status.error.message.lower())


if __name__ == "__main__":
    unittest.main()
