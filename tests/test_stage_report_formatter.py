from __future__ import annotations

import json
import unittest

from mlsystem.src.pipeline.stage_report_formatter import compact_xcom_summary, format_stage_report, path_views


class StageReportFormatterTests(unittest.TestCase):
    def test_path_views_maps_container_status_path_to_host_path(self) -> None:
        views = path_views(
            "/opt/airflow/mlsystem_runs/run1/stages/inventory_scenes.json",
            container_status_root="/opt/airflow/mlsystem_runs",
            host_status_root="/data/mlsystem/airflow/status",
        )
        self.assertEqual(views["container_path"], "/opt/airflow/mlsystem_runs/run1/stages/inventory_scenes.json")
        self.assertEqual(views["host_path"], "/data/mlsystem/airflow/status/run1/stages/inventory_scenes.json")

    def test_format_stage_report_contains_human_sections_and_diagnostics(self) -> None:
        report = {
            "status": "failed",
            "checks": [{"name": "files found", "status": "failed", "message": "missing=1"}],
            "counters": {"missing_scenes": 1},
            "errors": ["missing_scene.tif"],
            "artifacts": {"missing_scenes.txt": "/opt/airflow/mlsystem_runs/run1/missing_scenes.txt"},
            "resources": {"sample_count": 2},
        }
        text = format_stage_report(
            report,
            stage="inventory_scenes",
            run_id="run1",
            job_id="job1",
            stage_json_path="/opt/airflow/mlsystem_runs/run1/stages/inventory_scenes.json",
            report_path="/opt/airflow/mlsystem_runs/run1/stages/inventory_scenes.report.md",
        )
        self.assertIn("MLSystem stage report", text)
        self.assertIn("missing_scene.tif", text)
        self.assertIn("host_path: `/data/mlsystem/airflow/status/run1/missing_scenes.txt`", text)
        self.assertIn("docker logs --tail 300 mlsystem-gpu-api", text)

    def test_format_stage_report_contains_input_lineage(self) -> None:
        report = {
            "status": "success",
            "details": {
                "input_lineage": {
                    "source": "inventory_scenes",
                    "inventory_matched_count": 24,
                    "selected_count": 2,
                    "dataset_input_limit": 2,
                    "limit_source": "dag_run.conf.preprocess.max_dataset_scenes",
                    "limit_reason": "unit mini dataset",
                    "invariant_status": "OK with explicit limit",
                    "excluded_count": 22,
                }
            },
        }
        text = format_stage_report(report, stage="prepare_dataset", run_id="run1")
        self.assertIn("## Input lineage", text)
        self.assertIn("inventory matched scenes: `24`", text)
        self.assertIn("selected scenes for dataset: `2`", text)
        self.assertIn("limit source: `dag_run.conf.preprocess.max_dataset_scenes`", text)
        self.assertIn("excluded scenes: `22`", text)

    def test_compact_xcom_summary_excludes_full_payload(self) -> None:
        report = {
            "status": "success",
            "summary": "done",
            "resources": {"huge": ["x"] * 100},
            "stage_report": {"details": {"huge": ["x"] * 100}},
            "artifacts": {"a": "b"},
            "counters": {"total_scenes": 2},
        }
        summary = compact_xcom_summary(report, stage="prepare_dataset", run_id="run1", job_id="job1", report_path="/r.md", stage_json_path="/s.json")
        self.assertEqual(summary["job_id"], "job1")
        self.assertEqual(summary["report_path"], "/r.md")
        self.assertEqual(summary["status"], "success")
        self.assertNotIn("resources", summary)
        self.assertNotIn("stage_report", summary)
        self.assertNotIn("artifacts", summary)
        self.assertLess(len(json.dumps(summary, ensure_ascii=False).encode("utf-8")), 10_000)


if __name__ == "__main__":
    unittest.main()
