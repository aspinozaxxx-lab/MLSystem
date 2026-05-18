from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from mlsystem.src.api.app import app
from mlsystem.src.inference_pipeline.api import PseudolabelRun


def _run(state: str = "queued") -> PseudolabelRun:
    return PseudolabelRun(
        run_id="pseudo_api",
        experiment_id="exp",
        state=state,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )


class InferencePipelineFastApiTests(unittest.TestCase):
    def test_pseudolabel_run_endpoints_delegate_to_public_api(self) -> None:
        headers = {"Authorization": "Bearer token"}
        with patch.dict(os.environ, {"MLSYSTEM_API_TOKEN": "token"}, clear=False), \
            patch("mlsystem.src.api.app.start_pseudolabel_run", return_value=_run()) as start, \
            patch("mlsystem.src.api.app.get_pseudolabel_run", return_value=_run("running")) as get_run, \
            patch("mlsystem.src.api.app.tail_pseudolabel_log", return_value="log tail") as tail, \
            patch("mlsystem.src.api.app.cancel_pseudolabel_run", return_value=_run("cancelled")) as cancel:
            client = TestClient(app)
            created = client.post(
                "/api/v1/pseudolabel-runs",
                headers=headers,
                json={"experiment_id": "exp", "model_ref": "model", "images_uri": "s3://bucket/images/"},
            )
            status = client.get("/api/v1/pseudolabel-runs/pseudo_api", headers=headers)
            log = client.get("/api/v1/pseudolabel-runs/pseudo_api/log", headers=headers)
            cancelled = client.post("/api/v1/pseudolabel-runs/pseudo_api/cancel", headers=headers)

        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["status_url"], "/api/v1/pseudolabel-runs/pseudo_api")
        self.assertEqual(status.json()["state"], "running")
        self.assertEqual(log.json()["log_tail"], "log tail")
        self.assertEqual(cancelled.json()["state"], "cancelled")
        start.assert_called_once()
        self.assertGreaterEqual(get_run.call_count, 2)
        tail.assert_called_once()
        cancel.assert_called_once_with("pseudo_api")


if __name__ == "__main__":
    unittest.main()
