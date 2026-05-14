from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from mlsystem.src.io_utils import write_json
from mlsystem.src.real_train import (
    SceneMatch,
    TinyUNet,
    _build_optimizer,
    _build_scheduler,
    _configure_dropout,
    _load_initial_checkpoint,
    _load_prepared_dataset_split,
    _resolve_batch_limit,
    _resolve_augmentation_level,
    _resolve_metric_thresholds,
    _threshold_metric_suffix,
)


class RealTrainPreparedSplitTests(unittest.TestCase):
    def test_load_prepared_dataset_split_matches_manifest_to_current_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            write_json(
                manifest,
                {
                    "split_strategy": "object_balanced",
                    "train_scenes": [{"entry": "scene_a.tif", "name": "scene_a.tif", "key": "images/scene_a.tif"}],
                    "val_scenes": [{"entry": "scene_b.tif", "name": "scene_b.tif", "key": "images/scene_b.tif"}],
                    "split_summary": {"total_files": 2},
                },
            )
            matches = [
                SceneMatch(entry="scene_a.tif", name="scene_a.tif", key="images/scene_a.tif", score=1.0),
                SceneMatch(entry="scene_b.tif", name="scene_b.tif", key="images/scene_b.tif", score=1.0),
            ]
            job = SimpleNamespace(preprocess={"prepared_dataset_manifest": str(manifest)})
            split = _load_prepared_dataset_split(root, matches, job)
            self.assertIsNotNone(split)
            train, val, metadata = split
            self.assertEqual([item.name for item in train], ["scene_a.tif"])
            self.assertEqual([item.name for item in val], ["scene_b.tif"])
            self.assertEqual(metadata["split_strategy"], "object_balanced")

    def test_load_prepared_dataset_split_fails_on_lost_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            write_json(
                manifest,
                {
                    "train_scenes": [{"entry": "scene_a.tif", "name": "scene_a.tif", "key": "images/scene_a.tif"}],
                    "val_scenes": [{"entry": "scene_b.tif", "name": "scene_b.tif", "key": "images/scene_b.tif"}],
                },
            )
            matches = [
                SceneMatch(entry="scene_a.tif", name="scene_a.tif", key="images/scene_a.tif", score=1.0),
                SceneMatch(entry="scene_b.tif", name="scene_b.tif", key="images/scene_b.tif", score=1.0),
                SceneMatch(entry="scene_c.tif", name="scene_c.tif", key="images/scene_c.tif", score=1.0),
            ]
            job = SimpleNamespace(preprocess={"prepared_dataset_manifest": str(manifest)})
            with self.assertRaisesRegex(RuntimeError, "lost 1 matched scenes"):
                _load_prepared_dataset_split(root, matches, job)

    def test_load_prepared_dataset_split_allows_explicit_input_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            write_json(
                manifest,
                {
                    "train_scenes": [{"entry": "scene_a.tif", "name": "scene_a.tif", "key": "images/scene_a.tif"}],
                    "val_scenes": [{"entry": "scene_b.tif", "name": "scene_b.tif", "key": "images/scene_b.tif"}],
                    "input_lineage": {"limit_applied": True, "dataset_input_limit": 2},
                    "limits": {"dataset_input_limit": 2},
                },
            )
            matches = [
                SceneMatch(entry="scene_a.tif", name="scene_a.tif", key="images/scene_a.tif", score=1.0),
                SceneMatch(entry="scene_b.tif", name="scene_b.tif", key="images/scene_b.tif", score=1.0),
                SceneMatch(entry="scene_c.tif", name="scene_c.tif", key="images/scene_c.tif", score=1.0),
            ]
            job = SimpleNamespace(preprocess={"prepared_dataset_manifest": str(manifest)})
            split = _load_prepared_dataset_split(root, matches, job)
            self.assertIsNotNone(split)
            train, val, metadata = split
            self.assertEqual([item.name for item in train], ["scene_a.tif"])
            self.assertEqual([item.name for item in val], ["scene_b.tif"])
            self.assertEqual(metadata["excluded_scene_count"], 1)
            self.assertTrue(metadata["input_limit_applied"])

    def test_load_initial_checkpoint_restores_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            source = TinyUNet(in_channels=4, out_channels=1, base=4)
            with torch.no_grad():
                for param in source.parameters():
                    param.fill_(0.25)
            torch.save({"model_state_dict": source.state_dict()}, path)

            target = TinyUNet(in_channels=4, out_channels=1, base=4)
            info = _load_initial_checkpoint(target, path, device=torch.device("cpu"), strict=True)

            self.assertEqual(info["path"], str(path))
            self.assertEqual(info["missing_keys"], [])
            self.assertEqual(info["unexpected_keys"], [])
            for param in target.parameters():
                self.assertTrue(torch.allclose(param, torch.full_like(param, 0.25)))

    def test_training_knobs_build_optimizer_scheduler_and_dropout(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Dropout2d(p=0.5))
        updated = _configure_dropout(model, 0.15)
        self.assertEqual(updated, 1)
        self.assertAlmostEqual(model[1].p, 0.15)

        optimizer = _build_optimizer(model, {"optimizer": "adam", "learning_rate": 1e-4, "weight_decay": 0.01})
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(optimizer.param_groups[0]["weight_decay"], 0.01)
        zero_lr_optimizer = _build_optimizer(model, {"optimizer": "adamw", "learning_rate": 0.0, "weight_decay": 0.0})
        self.assertAlmostEqual(zero_lr_optimizer.param_groups[0]["lr"], 0.0)

        scheduler = _build_scheduler(optimizer, {"scheduler": {"name": "cosine", "t_max": 3, "eta_min": 1e-6}}, epochs=5)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_metric_threshold_sweep_keeps_base_and_deduplicates(self) -> None:
        job = SimpleNamespace(train={"metric_thresholds": [0.7, "0.80", 0.8]}, evaluate={}, params={})
        self.assertEqual(_resolve_metric_thresholds(job, 0.75), [0.7, 0.75, 0.8])
        self.assertEqual(_threshold_metric_suffix(0.8), "0_8")

    def test_train_augmentation_level_is_honored_when_preprocess_is_unset(self) -> None:
        job = SimpleNamespace(preprocess={"train_sampling": {"enabled": True}}, train={"augmentation_level": 1})
        self.assertEqual(_resolve_augmentation_level(job, train_sampling_enabled=True), 1)

        override = SimpleNamespace(preprocess={"augmentation_level": 3, "train_sampling": {"enabled": True}}, train={"augmentation_level": 1})
        self.assertEqual(_resolve_augmentation_level(override, train_sampling_enabled=True), 3)

    def test_batch_limit_resolves_null_and_positive_values(self) -> None:
        self.assertIsNone(_resolve_batch_limit(None))
        self.assertIsNone(_resolve_batch_limit("null"))
        self.assertIsNone(_resolve_batch_limit(0))
        self.assertEqual(_resolve_batch_limit("3"), 3)


if __name__ == "__main__":
    unittest.main()
