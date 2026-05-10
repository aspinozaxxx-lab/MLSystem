from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from InferenceEngine.src.inference_engine.api import app as app_module
from InferenceEngine.src.inference_engine.config.settings import InferenceEngineSettings
from InferenceEngine.src.inference_engine.storage.job_store import JobStore


class ApiTests(unittest.TestCase):
    def test_start_status_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            app_module.settings = InferenceEngineSettings(job_root=root / "jobs", spool_root=root / "spool", artifact_root=root / "artifacts", logs_root=root / "logs")
            app_module.settings.ensure_dirs()
            app_module.store = JobStore(app_module.settings.job_root)
            client = TestClient(app_module.app)
            payload = {
                "experiment_id": "api_unit",
                "scenes": [
                    {
                        "scene_id": "scene_api",
                        "name": "scene_api",
                        "width": 16,
                        "height": 16,
                        "transform": [1, 0, 0, 0, -1, 16],
                        "probability_rects": [[2, 2, 14, 14, 1.0]],
                    }
                ],
                "preprocess": {"patch_size": 16, "stride": 16},
                "pseudolabel": {"threshold": 0.5, "core_size_px": 16, "halo_px": 2, "final_min_area": 0},
            }
            created = client.post("/api/v1/jobs", json=payload)
            self.assertEqual(created.status_code, 200)
            job_id = created.json()["job_id"]
            status = client.get(f"/api/v1/jobs/{job_id}")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["status"], "success")
            artifacts = client.get(f"/api/v1/jobs/{job_id}/artifacts").json()["artifacts"]
            self.assertIn("accepted_geojson", artifacts)


if __name__ == "__main__":
    unittest.main()
