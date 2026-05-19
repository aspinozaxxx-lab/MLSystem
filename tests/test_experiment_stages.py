from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlsystem.src.train_pipeline.experiment_stages import run_stage
from mlsystem.src.train_pipeline.api import TrainPipelineRunStore, parse_pipeline_run_config


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
            store = TrainPipelineRunStore(Path(tmp), "unit_run", conf)

            with patch("mlsystem.src.train_pipeline.experiment_stages._run_training_pipeline", return_value=training_result):
                payload = run_stage("train_model", conf, store)

            self.assertEqual(payload["status"], "success")
            self.assertEqual(payload["mlflow"]["run_id"], "mlflow-run-1")

    def test_train_pipeline_delegates_training_result_logging_to_mlflow_adapter(self) -> None:
        source = Path("mlsystem/src/train_pipeline/experiment_stages.py").read_text(encoding="utf-8")
        self.assertIn("mlflow_log_training_result_to_run(", source)
        self.assertNotIn("_allowed_training_metric_payload", source)


if __name__ == "__main__":
    unittest.main()
