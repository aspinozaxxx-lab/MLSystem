from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from mlsystem.src.api.app import app
from mlsystem.src.pipeline_runner.api import PipelineRun, PipelineRunConfig, PipelineRunStore


class PipelineApiTests(unittest.TestCase):
    def test_post_pipeline_run_with_json_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"MLSYSTEM_RUN_ROOT": tmp, "MLSYSTEM_API_TOKEN": "token"}, clear=False):
            run = PipelineRun.model_validate(
                {
                    "run_id": "api_run",
                    "state": "queued",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            )
            with patch("mlsystem.src.api.app.PipelineRunner") as runner_cls:
                runner_cls.return_value.start_run.return_value = run
                response = TestClient(app).post(
                    "/api/v1/pipeline-runs",
                    headers={"Authorization": "Bearer token"},
                    json={"trace": {"experiment_id": "api_unit", "pipeline": {"dry_run": True}}, "dry_run": True},
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["run_id"], "api_run")
            self.assertEqual(response.json()["status_url"], "/api/v1/pipeline-runs/api_run")

    def test_status_log_stages_and_cancel_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"MLSYSTEM_RUN_ROOT": tmp, "MLSYSTEM_API_TOKEN": "token"}, clear=False):
            store = PipelineRunStore(Path(tmp))
            store.create_run(PipelineRunConfig.model_validate({"run_id": "api_status", "experiment_id": "api_unit", "pipeline": {"stages": ["inventory_scenes"], "dry_run": True}}))
            store.append_log("api_status", "line1\n")
            store.write_stage_report("api_status", "inventory_scenes", {"status": "success", "summary": "ok"})
            client = TestClient(app)
            headers = {"Authorization": "Bearer token"}
            self.assertEqual(client.get("/api/v1/pipeline-runs/api_status", headers=headers).status_code, 200)
            self.assertIn("line1", client.get("/api/v1/pipeline-runs/api_status/log", headers=headers).json()["log_tail"])
            self.assertEqual(client.get("/api/v1/pipeline-runs/api_status/stages", headers=headers).json()["reports"][0]["stage"], "inventory_scenes")
            cancelled = client.post("/api/v1/pipeline-runs/api_status/cancel", headers=headers)
            self.assertEqual(cancelled.status_code, 200)
            self.assertTrue((Path(tmp) / "api_status" / "cancel.requested").exists())


if __name__ == "__main__":
    unittest.main()
