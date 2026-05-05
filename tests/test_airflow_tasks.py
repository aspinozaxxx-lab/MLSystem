from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlsystem.src.pipeline.airflow_tasks import (
    LEGACY_FALLBACK_STAGES,
    MAIN_DAG_STAGES,
    STAGE_POOLS,
    push_stage_xcom,
    run_airflow_stage,
    run_stage,
    stage_return_message,
)
from mlsystem.src.pipeline.stages.registry import get_stage_entrypoint


SMOKE_CONF = {
    "experiment_id": "unit_airflow_smoke",
    "class_name": "deforest",
    "task": "smoke",
    "smoke": True,
    "train": {"enabled": False},
    "pseudolabel": {"enabled": True},
    "postprocess": {"enabled": True},
    "mlflow": {"experiment": "mlsystem-airflow-smoke"},
}


class AirflowTasksTests(unittest.TestCase):
    def test_main_dag_has_required_stage_order(self) -> None:
        self.assertEqual(MAIN_DAG_STAGES[0], "inventory_scenes")
        self.assertEqual(MAIN_DAG_STAGES[1], "prepare_dataset")
        self.assertEqual(MAIN_DAG_STAGES[-1], "finalize_mlflow_run")
        self.assertIn("prepare_inference_scenes", MAIN_DAG_STAGES)
        self.assertIn("run_pseudolabel_inference", MAIN_DAG_STAGES)
        self.assertIn("validate_probability_maps", MAIN_DAG_STAGES)

    def test_main_dag_stages_have_pool_and_entrypoint_or_fallback(self) -> None:
        for stage in MAIN_DAG_STAGES:
            self.assertIn(stage, STAGE_POOLS)
            if stage in LEGACY_FALLBACK_STAGES:
                continue
            self.assertTrue(callable(get_stage_entrypoint(stage)))

    def test_unknown_stage_error_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown Airflow MLSystem stage"):
                run_stage("unknown_stage", SMOKE_CONF, "manual__unit", Path(tmp))

    def test_validate_config_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stage("validate_experiment_config", SMOKE_CONF, "manual__unit", Path(tmp))
            self.assertEqual(result["status"], "success")
            summary = Path(tmp) / "unit_airflow_smoke" / "summary.json"
            self.assertTrue(summary.exists())

    def test_s3_stage_skipped_for_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stage("check_s3_layout", SMOKE_CONF, "manual__unit", Path(tmp))
            self.assertEqual(result["status"], "skipped")

    def test_run_airflow_stage_returns_compact_xcom_summary_in_local_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"MLSYSTEM_AIRFLOW_EXECUTION_MODE": "local"}):
                result = run_airflow_stage("check_s3_layout", SMOKE_CONF, "manual__unit", Path(tmp))
            self.assertEqual(result["stage"], "check_s3_layout")
            self.assertEqual(result["status"], "skipped")
            self.assertIn("report_path", result)
            self.assertIn("stage_json_path", result)
            self.assertNotIn("resources", result)
            self.assertNotIn("stage_report", result)
            self.assertLess(len(str(result).encode("utf-8")), 10_000)

    def test_push_stage_xcom_uses_readable_key_value_pairs(self) -> None:
        class FakeTaskInstance:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def xcom_push(self, *, key: str, value: object) -> None:
                self.values[key] = value

        summary = {
            "stage": "prepare_dataset",
            "status": "success",
            "run_id": "manual__unit",
            "job_id": "job1",
            "summary": "prepare_dataset completed",
            "report_path": "/opt/airflow/mlsystem_runs/unit/stages/prepare_dataset.report.md",
            "stage_json_path": "/opt/airflow/mlsystem_runs/unit/stages/prepare_dataset.json",
            "warnings_count": 1,
            "errors_count": 0,
            "duration_sec": 1.2,
            "key_counters": {
                "total_scenes": 24,
                "total_objects": 301,
                "train_scenes": 19,
                "val_scenes": 5,
                "split_strategy": "object_balanced",
                "upstream_inventory_matched_scenes": 24,
            },
        }
        ti = FakeTaskInstance()
        push_stage_xcom(summary, ti)
        self.assertEqual(ti.values["stage"], "prepare_dataset")
        self.assertEqual(ti.values["status"], "success")
        self.assertEqual(ti.values["job_id"], "job1")
        self.assertEqual(ti.values["report_path"], summary["report_path"])
        self.assertEqual(ti.values["counter_total_scenes"], 24)
        self.assertEqual(ti.values["counter_split_strategy"], "object_balanced")
        self.assertNotIn("resources", ti.values)
        self.assertNotIn("stage_report", ti.values)
        self.assertLess(len(str(ti.values).encode("utf-8")), 10_000)
        self.assertLess(len(stage_return_message(summary).encode("utf-8")), 512)

    def test_stage_failure_writes_stage_report_before_raising(self) -> None:
        conf = {
            "experiment_id": "unit_stage_failure",
            "pseudolabel": {"run_on": "explicit_scene_list", "scene_list": ["missing.tif"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_stage("prepare_inference_scenes", conf, "manual__unit", Path(tmp))
            stage_path = Path(tmp) / "unit_stage_failure" / "stages" / "prepare_inference_scenes.json"
            report_path = Path(tmp) / "unit_stage_failure" / "stages" / "prepare_inference_scenes.report.md"
            self.assertTrue(stage_path.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("stage_report", stage_path.read_text(encoding="utf-8"))
            self.assertIn("missing.tif", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
