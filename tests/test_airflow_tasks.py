from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlsystem.src.pipeline.airflow_tasks import LEGACY_FALLBACK_STAGES, MAIN_DAG_STAGES, STAGE_POOLS, run_stage
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

    def test_stage_failure_writes_stage_report_before_raising(self) -> None:
        conf = {
            "experiment_id": "unit_stage_failure",
            "pseudolabel": {"run_on": "explicit_scene_list", "scene_list": ["missing.tif"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_stage("prepare_inference_scenes", conf, "manual__unit", Path(tmp))
            stage_path = Path(tmp) / "unit_stage_failure" / "stages" / "prepare_inference_scenes.json"
            self.assertTrue(stage_path.exists())
            self.assertIn("stage_report", stage_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
