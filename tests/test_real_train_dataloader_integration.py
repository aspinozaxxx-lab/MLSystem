from __future__ import annotations

import unittest
from pathlib import Path


class RealTrainDataloaderIntegrationTests(unittest.TestCase):
    def test_real_train_uses_facade_dataloaders(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertIn("TilePreparationFacade.train_dataloader", text)
        self.assertIn("TilePreparationFacade.val_dataloader", text)
        self.assertNotIn("from .tile_preparation.dataset import iter_dataset_batches", text)

    def test_real_train_uses_non_blocking_device_transfer(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertIn(".to(device, non_blocking=True)", text)

    def test_legacy_sync_batches_remain_outside_training_loop(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertNotIn("iter_dataset_batches(", text)

    def test_real_train_logs_explicit_batch_wait_metrics(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        for key in (
            "data/batch_wait_total_sec",
            "data/batch_wait_mean_sec",
            "data/batch_wait_median_sec",
            "data/batch_wait_p95_sec",
            "data/batch_wait_max_sec",
            "data/batch_prepare_or_wait_samples",
            "data/batches_per_sec",
            "data/samples_per_sec",
        ):
            self.assertIn(key, text)
        self.assertIn('payload["data/batch_wait_sec"] = payload["data/batch_wait_total_sec"]', text)


if __name__ == "__main__":
    unittest.main()
