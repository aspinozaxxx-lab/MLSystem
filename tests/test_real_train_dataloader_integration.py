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

    def test_real_train_disables_recreated_loader_persistent_workers(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertIn("persistent_workers_requested", text)
        self.assertIn("persistent_workers_effective = False", text)
        self.assertIn("persistent_workers=persistent_workers_effective", text)

    def test_real_train_uses_internal_tile_dataloader_defaults(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertIn("resolve_worker_count(None)", text)
        self.assertIn("resolve_prefetch_factor(dataloader_workers, None)", text)
        self.assertIn("dataloader_config_source", text)
        self.assertIn("deprecated_dataloader_trace_keys_ignored", text)
        self.assertNotIn('job.train.get("dataloader_workers"', text)
        self.assertNotIn("workers=dataloader_workers", text)
        self.assertNotIn("prefetch_factor=dataloader_prefetch_factor", text)

    def test_real_train_has_opt_in_step_timing_profile_metrics(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertIn("profile_step_timing", text)
        for key in (
            "model/cpu_to_gpu",
            "model/forward",
            "model/loss_timing",
            "model/backward",
            "model/optimizer_step",
        ):
            self.assertIn(key, text)

    def test_real_train_can_collect_tile_profile_metadata_with_workers(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertIn("tile_collate_with_metadata_fn", text)
        self.assertIn("_accumulate_tile_profile_metadata", text)
        self.assertIn("collect_tile_profile_metadata", text)


if __name__ == "__main__":
    unittest.main()
