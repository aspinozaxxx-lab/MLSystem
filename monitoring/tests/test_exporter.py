from __future__ import annotations

import unittest
from unittest.mock import patch

from monitoring.exporter import exporter


class MonitoringExporterTests(unittest.TestCase):
    def test_renders_service_and_airflow_metrics(self) -> None:
        def fake_json(url: str, *, timeout: int = 5, headers=None):  # noqa: ANN001, ARG001
            if url.endswith("/api/v1/health"):
                return 200, {"scheduler": {"status": "healthy"}}
            if "dagRuns?limit=5" in url:
                return 200, {"dag_runs": [{"dag_run_id": "run1", "state": "success"}]}
            if url.endswith("/taskInstances"):
                return 200, {"task_instances": [{"task_id": "inference_engine_pipeline", "state": "success", "duration": 12.5}]}
            if url.endswith("/metrics"):
                return 200, {"aggregate": {"tiles_done": 7, "spool_bytes": 3}, "jobs": {"job1": {"status": "success", "metrics": {}}}}
            return 200, {"status": "ok"}

        with patch("monitoring.exporter.exporter.http_json", side_effect=fake_json):
            text = exporter.render_metrics()
        self.assertIn('mlsystem_service_up{service="frontend"} 1', text)
        self.assertIn("airflow_scheduler_healthy 1", text)
        self.assertIn('airflow_task_duration_seconds{dag_id="mlsystem_experiment_pipeline",task_id="inference_engine_pipeline"} 12.5', text)
        self.assertIn("mlsystem_inference_engine_tiles_done 7.0", text)


if __name__ == "__main__":
    unittest.main()
