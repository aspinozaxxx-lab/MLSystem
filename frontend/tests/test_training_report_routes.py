from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from frontend.app.config import FrontendConfig
from frontend.app.main import create_app


def _config(tmp: Path) -> FrontendConfig:
    return FrontendConfig(
        username="mluser",
        password="qazwsxedc",
        session_secret="unit-secret",
        upload_root=tmp / "uploads",
        status_root=tmp / "status",
        training_report_root=tmp / "training_report",
        training_report_background_enabled=False,
        mlmarkup_path=tmp / "MLMarkup",
    )


def _sample_report() -> dict[str, object]:
    return {
        "status": "ok",
        "updated_at": "2026-05-11T10:00:00Z",
        "source": {"mlflow_tracking_uri": "http://mlflow:5000/mlflow", "mlmarkup_path": "/data/MLMarkup", "cache_path": "/tmp/index.json"},
        "classes": [
            {
                "class_name": "Озера",
                "class_slug": "lakes",
                "best_pixel_f1": 0.5,
                "dataset_date": "2026-05-10",
                "dataset_objects": 202,
                "dataset_scenes": 124,
                "best_run_id": "run-lakes",
                "best_run_url": "/mlflow/#/experiments/38/runs/run-lakes",
                "best_train_date": "2026-05-11",
                "validation_kind": "scene_level",
                "warning": None,
                "top_runs": [
                    {
                        "rank": 1,
                        "run_id": "run-lakes",
                        "run_url": "/mlflow/#/experiments/38/runs/run-lakes",
                        "pixel_f1": 0.5,
                        "dataset_date": "2026-05-10",
                        "dataset_objects": 202,
                        "dataset_scenes": 124,
                        "train_date": "2026-05-11",
                        "class_name": "Озера",
                        "class_slug": "lakes",
                        "split_strategy": "scene_level",
                        "model_name": "segformer_b2",
                        "params_summary": "segformer_b2, lr=0.0002",
                    }
                ],
            },
            {
                "class_name": "Абразия",
                "class_slug": "abrasion",
                "best_pixel_f1": None,
                "dataset_date": "2026-05-10",
                "dataset_objects": 11,
                "dataset_scenes": 3,
                "best_run_id": None,
                "best_run_url": None,
                "best_train_date": None,
                "validation_kind": "unknown",
                "warning": "no runs",
                "top_runs": [],
            },
        ],
    }


class TrainingReportRouteTests(unittest.TestCase):
    def test_training_report_requires_login_and_contains_template_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = create_app(_config(Path(td)))
            client = TestClient(app)
            denied = client.get("/training-report", follow_redirects=False)
            self.assertEqual(denied.status_code, 303)
            client.post("/login", data={"username": "mluser", "password": "qazwsxedc"})
            response = client.get("/training-report")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Отчет об обучении", response.text)
            self.assertIn("Озера", response.text)
            self.assertIn("Абразия", response.text)
            self.assertIn("training-report-expandable", response.text)

    def test_training_report_api_and_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            app = create_app(_config(Path(td)))
            service = app.state.training_report_service
            service.cache.write_index(_sample_report())
            service.refresh_now = lambda reason="manual": {"status": "ok", "updated_at": "2026-05-11T10:01:00Z"}
            client = TestClient(app)
            client.post("/login", data={"username": "mluser", "password": "qazwsxedc"})
            response = client.get("/api/training-report")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["classes"][0]["class_name"], "Озера")
            self.assertEqual(payload["classes"][0]["best_run_url"], "/mlflow/#/experiments/38/runs/run-lakes")
            refresh = client.post("/api/training-report/refresh")
            self.assertEqual(refresh.status_code, 200)
            self.assertEqual(refresh.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()

