from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlsystem.src.inference_pipeline.api import InferencePipelineRunner, PseudolabelRunRequest
from mlsystem.src.inference_pipeline._run_store import PseudolabelRunStore


class FakeInferenceEngineClient:
    def create_job(self, payload: dict) -> dict:
        self.payload = payload
        return {"job_id": "job-unit"}

    def wait(self, job_id: str, *, poll_sec: float = 10.0, timeout_sec: float = 24 * 3600) -> dict:
        return {"job_id": job_id, "status": "success", "metrics": {"tiles_done": 2}}

    def get_artifacts(self, job_id: str) -> dict:
        return {"job_id": job_id, "artifacts": {"accepted_geojson": "s3://bucket/accepted.geojson"}}


class InferencePipelineApiTests(unittest.TestCase):
    def test_start_run_submits_inference_engine_job_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch("mlsystem.src.inference_pipeline._worker._log_mlflow_metadata", return_value={"ok": True}):
            runner = InferencePipelineRunner(
                PseudolabelRunStore(Path(tmp)),
                client_factory=FakeInferenceEngineClient,
                run_worker_inline=True,
            )
            run = runner.start_run(
                PseudolabelRunRequest(
                    run_id="pseudo_unit",
                    experiment_id="exp",
                    model_ref="models:/segformer/latest",
                    images_uri="s3://bucket/images/",
                    scenes=["scene_a.tif"],
                )
            )

            self.assertEqual(run.state, "succeeded")
            self.assertEqual(run.inference_engine_job_id, "job-unit")
            self.assertEqual(run.progress_percent, 100)
            payload = json.loads((Path(tmp) / "pseudo_unit" / "inference_engine_payload.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "inference_pipeline")
            self.assertEqual(payload["scenes"][0]["name"], "scene_a.tif")
            self.assertIn("accepted_geojson", run.artifacts["artifacts"])
            self.assertIn("worker finished successfully", runner.tail_log("pseudo_unit"))


if __name__ == "__main__":
    unittest.main()
