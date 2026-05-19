from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from mlsystem.src.train_pipeline.api import DEFAULT_PIPELINE_STAGES, load_trace_file, parse_pipeline_run_config
from mlsystem.src.train_pipeline.experiment_stages import _select_short_mlflow_run_name, _validate_explicit_training_epochs
from mlsystem.src.train_pipeline.runner import worker_module_name
from mlsystem.src.train_pipeline.stage_jobs import stage_job_worker_module_name


class TrainPipelineRunnerConfigTests(unittest.TestCase):
    def test_parse_json_trace_defaults_and_aliases(self) -> None:
        config = parse_pipeline_run_config({"experiment_id": "unit", "pipeline": {"stages": ["inventory", "prepare-dataset", "compute-f1"]}})
        self.assertEqual(config.pipeline.stages, ["inventory_scenes", "prepare_dataset", "compute_f1"])
        self.assertEqual(config.task, "train_predict_pseudolabel")

    def test_parse_yaml_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.yaml"
            path.write_text("experiment_id: yaml_unit\npipeline:\n  dry_run: true\n", encoding="utf-8")
            config = load_trace_file(path)
        self.assertEqual(config.experiment_id, "yaml_unit")
        self.assertTrue(config.pipeline.dry_run)
        self.assertEqual(config.pipeline.stages, DEFAULT_PIPELINE_STAGES)

    def test_parse_json_trace_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            path.write_text(json.dumps({"experiment_id": "json_unit"}), encoding="utf-8")
            config = load_trace_file(path)
        self.assertEqual(config.experiment_id, "json_unit")

    def test_rejects_bad_run_id(self) -> None:
        with self.assertRaises(ValueError):
            parse_pipeline_run_config({"experiment_id": "unit", "run_id": "../bad"})

    def test_short_mlflow_run_name_sequence(self) -> None:
        started_on = date(2026, 5, 19)
        self.assertEqual(_select_short_mlflow_run_name("lakes", started_on, []), "lakes_0519")
        self.assertEqual(_select_short_mlflow_run_name("lakes", started_on, ["lakes_0519"]), "lakes_0519_2")
        self.assertEqual(_select_short_mlflow_run_name("lakes", started_on, ["lakes_0519", "lakes_0519_2"]), "lakes_0519_3")

    def test_non_smoke_training_requires_explicit_epochs(self) -> None:
        config = parse_pipeline_run_config({"experiment_id": "unit", "task": "train", "train": {"enabled": True}})
        with self.assertRaisesRegex(ValueError, "train.epochs"):
            _validate_explicit_training_epochs(config)
        _validate_explicit_training_epochs(parse_pipeline_run_config({"experiment_id": "unit", "task": "train", "smoke": True}))

    def test_legacy_worker_module_env_is_normalized(self) -> None:
        with patch.dict("os.environ", {"MLSYSTEM_API_WORKER_MODULE": "src.api.stage_job_worker"}):
            self.assertTrue(stage_job_worker_module_name().endswith(".train_pipeline.stage_job_worker"))
        with patch.dict("os.environ", {"MLSYSTEM_PIPELINE_WORKER_MODULE": "src.pipeline_runner.worker"}):
            self.assertTrue(worker_module_name().endswith(".train_pipeline.worker"))


if __name__ == "__main__":
    unittest.main()
