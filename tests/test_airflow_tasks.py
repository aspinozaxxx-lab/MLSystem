from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlsystem.src.pipeline.airflow_tasks import MAIN_DAG_STAGES, run_stage


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
        self.assertEqual(MAIN_DAG_STAGES[0], "validate_experiment_config")
        self.assertEqual(MAIN_DAG_STAGES[-1], "finalize_mlflow_run")
        self.assertIn("stitch_probability_maps", MAIN_DAG_STAGES)

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


if __name__ == "__main__":
    unittest.main()
