from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from mlsystem.src.train.api import train_model
from mlsystem.src.train.contracts import TrainConfig, TrainRequest


class TrainLoopSmokeTests(unittest.TestCase):
    def test_train_model_runs_one_epoch_and_returns_raw_training_result(self) -> None:
        x = torch.rand((4, 4, 16, 16), dtype=torch.float32)
        y = (torch.rand((4, 1, 16, 16)) > 0.5).float()
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            result = train_model(
                TrainRequest(
                    config=TrainConfig(model_name="tiny_unet_4ch", base_channels=2, epochs=1, require_gpu=False),
                    train_dataloader=loader,
                    val_dataloader=loader,
                    output_dir=Path(tmp),
                )
            )
            self.assertTrue((Path(tmp) / "training_diagnostics.json").exists())
        self.assertEqual(result.status, "done")
        self.assertEqual(result.epochs_completed, 1)
        self.assertIsNotNone(result.checkpoint)
        self.assertGreaterEqual(result.training_time_sec, 0.0)
        self.assertFalse(hasattr(result, "mlflow_metrics"))

    def test_threshold_sweep_is_logged_separately_from_configured_f1(self) -> None:
        x = torch.rand((4, 4, 16, 16), dtype=torch.float32)
        y = (torch.rand((4, 1, 16, 16)) > 0.5).float()
        loader = DataLoader(TensorDataset(x, y), batch_size=2)
        with tempfile.TemporaryDirectory() as tmp:
            result = train_model(
                TrainRequest(
                    config=TrainConfig(
                        model_name="tiny_unet_4ch",
                        base_channels=2,
                        epochs=1,
                        require_gpu=False,
                        metric_threshold=0.8,
                        metric_thresholds=[0.5, 0.8],
                    ),
                    train_dataloader=loader,
                    val_dataloader=loader,
                    output_dir=Path(tmp),
                )
            )
            threshold_summary_exists = (Path(tmp) / "threshold_sweep_summary.json").exists()
        last_metrics = result.history[-1].metrics
        self.assertIn("val/pixel_f1", last_metrics)
        self.assertIn("val/pixel_f1_at_threshold_0_5", last_metrics)
        self.assertIn("val/pixel_f1_at_threshold_0_8", last_metrics)
        self.assertIn("val/pixel_f1_best_threshold", last_metrics)
        self.assertTrue(threshold_summary_exists)
        self.assertFalse(hasattr(result, "mlflow_metrics"))


if __name__ == "__main__":
    unittest.main()
