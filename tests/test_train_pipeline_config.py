from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlsystem.src.train_pipeline.api import DEFAULT_PIPELINE_STAGES, load_trace_file, parse_pipeline_run_config


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


if __name__ == "__main__":
    unittest.main()
