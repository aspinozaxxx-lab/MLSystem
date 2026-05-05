from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from mlsystem.src.orchestration.airflow_api_client import AirflowApiStageError, MLSystemApiClient, run_stage_via_api


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class AirflowApiClientTests(unittest.TestCase):
    def test_client_start_stage_posts_json(self) -> None:
        with patch("urllib.request.urlopen", return_value=_Response({"job_id": "job1"})) as opened:
            client = MLSystemApiClient("http://api", token="token")
            response = client.start_stage("run1", "inventory_scenes", {"secret": "value"})
        self.assertEqual(response["job_id"], "job1")
        request = opened.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer token")
        self.assertIn(b'"secret": "***"', request.data)

    def test_client_raises_clear_error_on_connection_failure(self) -> None:
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            client = MLSystemApiClient("http://api")
            with self.assertRaises(AirflowApiStageError):
                client.job_status("job1")

    def test_run_stage_via_api_returns_compact_xcom_summary(self) -> None:
        full_report = {
            "stage": "prepare_dataset",
            "status": "success",
            "summary": "prepare_dataset completed",
            "stage_json_path": "/opt/airflow/mlsystem_runs/unit/stages/prepare_dataset.json",
            "report_path": "/opt/airflow/mlsystem_runs/unit/stages/prepare_dataset.report.md",
            "counters": {"total_scenes": 3, "train_objects": 2, "val_objects": 1},
            "warnings": ["zero scene"],
            "resources": {"before": {"gpu": [{"name": "GPU"}]}},
            "stage_report": {"details": {"large": ["x"] * 100}},
            "artifacts": {"dataset_manifest.json": "/opt/airflow/mlsystem_runs/unit/dataset_manifest.json"},
        }
        responses = [
            _Response({"job_id": "job1", "state": "queued"}),
            _Response({"job_id": "job1", "run_id": "run1", "stage": "prepare_dataset", "state": "succeeded", "duration_sec": 1.2, "report": full_report}),
        ]
        with patch("urllib.request.urlopen", side_effect=responses), patch.dict("os.environ", {"MLSYSTEM_API_URL": "http://api", "MLSYSTEM_AIRFLOW_API_POLL_SEC": "0"}):
            summary = run_stage_via_api("prepare_dataset", {"experiment_id": "unit"}, "run1", "/opt/airflow/mlsystem_runs")
        self.assertEqual(summary["stage"], "prepare_dataset")
        self.assertEqual(summary["job_id"], "job1")
        self.assertEqual(summary["status"], "success")
        self.assertIn("report_path", summary)
        self.assertIn("total_scenes", summary["key_counters"])
        self.assertNotIn("resources", summary)
        self.assertNotIn("stage_report", summary)
        self.assertNotIn("artifacts", summary)
        self.assertLess(len(json.dumps(summary, ensure_ascii=False).encode("utf-8")), 10_000)

    def test_failed_api_stage_exception_contains_report_path_and_job_id(self) -> None:
        failed_report = {
            "stage": "inventory_scenes",
            "status": "failed",
            "summary": "inventory_scenes failed",
            "report_path": "/opt/airflow/mlsystem_runs/unit/stages/inventory_scenes.report.md",
            "stage_json_path": "/opt/airflow/mlsystem_runs/unit/stages/inventory_scenes.json",
            "errors": ["missing_scene.tif"],
        }
        responses = [
            _Response({"job_id": "job_failed", "state": "queued"}),
            _Response({"job_id": "job_failed", "run_id": "run1", "stage": "inventory_scenes", "state": "failed", "duration_sec": 1.0, "report": failed_report, "error": {"message": "missing_scene.tif"}}),
        ]
        with patch("urllib.request.urlopen", side_effect=responses), patch.dict("os.environ", {"MLSYSTEM_API_URL": "http://api", "MLSYSTEM_AIRFLOW_API_POLL_SEC": "0"}):
            with self.assertRaises(AirflowApiStageError) as raised:
                run_stage_via_api("inventory_scenes", {"experiment_id": "unit"}, "run1", "/opt/airflow/mlsystem_runs")
        text = str(raised.exception)
        self.assertIn("job_failed", text)
        self.assertIn("inventory_scenes.report.md", text)


if __name__ == "__main__":
    unittest.main()
