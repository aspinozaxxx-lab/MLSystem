from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlsystem.src.pipeline.experiment_stages import run_stage
from mlsystem.src.pipeline_runner.api import PipelineRunStore, parse_pipeline_run_config


class ExperimentStagesTests(unittest.TestCase):
    def test_train_model_stage_exposes_mlflow_from_training_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            training_result = {
                "mode": "checkpoint_reeval",
                "epochs_completed": 1,
                "best_val_iou": 0.4,
                "checkpoint_path": str(Path(tmp) / "model.pt"),
                "mlflow": {"run_id": "mlflow-run-1", "experiment_id": "40"},
            }
            conf = parse_pipeline_run_config({"experiment_id": "unit", "train": {"enabled": True}})
            store = PipelineRunStore(Path(tmp), "unit_run", conf)

            with patch("mlsystem.src.pipeline.experiment_stages._run_training_pipeline", return_value=training_result):
                payload = run_stage("train_model", conf, store)

            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["mlflow"]["run_id"], "mlflow-run-1")


if __name__ == "__main__":
    unittest.main()
