from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlsystem.src.train_pipeline.api import StageJobStore as JobStore
from mlsystem.src.train_pipeline.api import StageStartRequest


class ApiJobStoreTests(unittest.TestCase):
    def test_create_job_persists_masked_request_and_job_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp))
            request = StageStartRequest(experiment_config={"experiment_id": "unit", "token": "secret"})
            job = store.create_job("run1", "inventory_scenes", request)
            self.assertTrue((store.job_dir(job.job_id) / "job.json").exists())
            self.assertTrue((store.job_dir(job.job_id) / "request.json").exists())
            self.assertIn('"token": "***"', (store.job_dir(job.job_id) / "request.json").read_text(encoding="utf-8"))
            loaded = store.read_job(job.job_id)
            self.assertEqual(loaded.state, "queued")


if __name__ == "__main__":
    unittest.main()
