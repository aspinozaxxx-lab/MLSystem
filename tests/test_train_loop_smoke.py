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
    def test_train_model_runs_one_epoch_and_returns_mlflow_payload(self) -> None:
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
        self.assertEqual(result.status, "done")
        self.assertEqual(result.epochs_completed, 1)
        self.assertIsNotNone(result.checkpoint)
        self.assertIn("f1_pixel", result.mlflow_metrics)
        self.assertIn("epochs_total", result.mlflow_metrics)
        self.assertFalse(any(key.startswith("diagnostics/") for key in result.mlflow_metrics))
        self.assertTrue((Path(tmp) / "training_diagnostics.json").exists())


if __name__ == "__main__":
    unittest.main()
