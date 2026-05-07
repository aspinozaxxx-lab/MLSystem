from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from frontend.app.config import FrontendConfig
from frontend.app.main import create_app
from frontend.app.upload_store import StoredUploads, UploadValidationError, parse_scene_names, sanitize_filename


class InlineThread:
    def __init__(self, target, args=(), kwargs=None, daemon=None):  # noqa: ANN001
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self) -> None:
        self.target(*self.args, **self.kwargs)


class FakeApiClient:
    calls: list[str] = []

    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url
        self.token = token

    def start_stage(self, run_id: str, stage: str, payload: dict):
        self.calls.append(stage)
        return {"job_id": f"{stage}_job", "state": "queued"}

    def wait_for_job(self, job_id: str):
        stage = job_id.replace("_job", "")
        return {"job_id": job_id, "stage": stage, "state": "succeeded", "report": {"status": "success"}}


def cfg(tmp: Path) -> FrontendConfig:
    return FrontendConfig(
        username="mluser",
        password="qazwsxedc",
        session_secret="unit-secret",
        upload_root=tmp / "uploads",
        status_root=tmp / "status",
        api_base_url="http://mlsystem-api:8088",
        api_token="server-token",
    )


class AnnotationCheckFlowTests(unittest.TestCase):
    def test_flow_calls_only_mlsystem_api_stages(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = cfg(root)
            client = TestClient(create_app(config))
            client.post("/login", data={"username": "mluser", "password": "qazwsxedc"})
            FakeApiClient.calls = []

            async def fake_store(**kwargs):  # noqa: ANN003
                run_id = kwargs["run_id"]
                run_dir = config.upload_root / run_id
                run_dir.mkdir(parents=True)
                status_dir = config.status_root / run_id
                (status_dir / "stages").mkdir(parents=True)
                (status_dir / "inventory_scenes.json").write_text(json.dumps({"matched": [{"entry": "a.tif"}], "matched_count": 1}), encoding="utf-8")
                (status_dir / "dataset_manifest.json").write_text(json.dumps({"train_scenes": [{"entry": "a.tif"}], "val_scenes": [], "scene_object_counts": [{"scene_name": "a.tif", "object_count": 1}]}), encoding="utf-8")
                (status_dir / "stages" / "inventory_scenes.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
                (status_dir / "stages" / "prepare_dataset.json").write_text(json.dumps({"status": "success", "counters": {"total_objects": 1}}), encoding="utf-8")
                return StoredUploads(
                    run_dir,
                    run_dir / "a.geojson",
                    run_dir / "s.txt",
                    "s3://b/frontend/run/",
                    "a.geojson",
                    "s.txt",
                    42,
                    6,
                    1,
                    ["a.tif"],
                    "frontend/run/a.geojson",
                    "frontend/run/s.txt",
                )

            with patch("frontend.app.main.store_uploads", side_effect=fake_store), \
                patch("frontend.app.main.MLSystemApiClient", FakeApiClient), \
                patch("frontend.app.main.threading.Thread", InlineThread):
                response = client.post(
                    "/api/annotation-check",
                    files={
                        "annotation_file": ("a.geojson", b'{"type":"FeatureCollection","features":[]}', "application/json"),
                        "scenes_file": ("s.txt", b"a.tif\n", "text/plain"),
                    },
                )
            self.assertEqual(response.status_code, 200)
            run_id = response.json()["run_id"]
            status = client.get(f"/api/annotation-check/{run_id}")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(FakeApiClient.calls, ["inventory_scenes", "prepare_dataset"])
            self.assertNotIn("server-token", status.text)
            self.assertNotIn("airflow", "".join(FakeApiClient.calls).lower())

    def test_upload_validation_helpers(self) -> None:
        self.assertEqual(sanitize_filename("../bad name.geojson", "x.geojson"), "bad_name.geojson")
        self.assertIsInstance(UploadValidationError("bad"), ValueError)
        self.assertEqual(parse_scene_names("\ufeff a.tif\r\n\n# comment\r\nb.TIFF\r\nfolder\\scene.tif extra\r\n"), ["a.tif", "b.TIFF", "folder\\scene.tif extra"])

    def test_api_start_error_is_visible_and_not_zeroed(self) -> None:
        class FailingApiClient(FakeApiClient):
            def start_stage(self, run_id: str, stage: str, payload: dict):  # noqa: ARG002
                raise RuntimeError("MLSystem API request failed after 3 attempts: connection refused")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = cfg(root)
            client = TestClient(create_app(config))
            client.post("/login", data={"username": "mluser", "password": "qazwsxedc"})

            async def fake_store(**kwargs):  # noqa: ANN003
                run_id = kwargs["run_id"]
                run_dir = config.upload_root / run_id
                run_dir.mkdir(parents=True)
                return StoredUploads(
                    run_dir,
                    run_dir / "a.geojson",
                    run_dir / "s.txt",
                    "s3://b/frontend/run/",
                    "a.geojson",
                    "s.txt",
                    42,
                    12,
                    2,
                    ["a.tif", "b.tif"],
                    "frontend/run/a.geojson",
                    "frontend/run/s.txt",
                )

            with patch("frontend.app.main.store_uploads", side_effect=fake_store), \
                patch("frontend.app.main.MLSystemApiClient", FailingApiClient), \
                patch("frontend.app.main.threading.Thread", InlineThread):
                response = client.post(
                    "/api/annotation-check",
                    files={
                        "annotation_file": ("a.geojson", b'{"type":"FeatureCollection","features":[]}', "application/json"),
                        "scenes_file": ("s.txt", b"a.tif\nb.tif\n", "text/plain"),
                    },
                )
            run_id = response.json()["run_id"]
            status = client.get(f"/api/annotation-check/{run_id}").json()
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["summary"]["total_scenes_requested"], 2)
            self.assertIsNone(status["summary"]["matched_scenes"])
            self.assertIn("connection refused", status["error"])
            self.assertEqual(status["stages"][0]["name"], "inventory_scenes")
            self.assertEqual(status["stages"][0]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
