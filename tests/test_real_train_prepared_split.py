from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mlsystem.src.io_utils import write_json
from mlsystem.src.real_train import (
    SceneMatch,
    TinyUNet,
    _apply_train_augmentations,
    _build_model,
    _build_optimizer,
    _build_scheduler,
    _configure_dropout,
    _load_initial_checkpoint,
    _load_prepared_dataset_split,
    _resolve_metric_thresholds,
    _save_training_checkpoint,
    _scene_tile_limit,
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

    def test_save_training_checkpoint_writes_best_epoch_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.best.pt"
            state = {"weight": torch.ones(1)}

            _save_training_checkpoint(
                path,
                state_dict=state,
                job_id="run_train_model_abcd",
                model_name="segformer_b2",
                best_epoch=17,
                best_objective_metric="val/pixel_f1_best_threshold",
                best_objective_value=0.514,
                final_epoch=17,
            )

            payload = torch.load(path, map_location="cpu")
            self.assertEqual(payload["job_id"], "run_train_model_abcd")
            self.assertEqual(payload["model_name"], "segformer_b2")
            self.assertEqual(payload["best_epoch"], 17)
            self.assertEqual(payload["best_objective_metric"], "val/pixel_f1_best_threshold")
            self.assertAlmostEqual(payload["best_objective_value"], 0.514)
            self.assertTrue(torch.equal(payload["model_state_dict"]["weight"], torch.ones(1)))

    def test_training_knobs_build_optimizer_scheduler_and_dropout(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Dropout2d(p=0.5))
        updated = _configure_dropout(model, 0.15)
        self.assertEqual(updated, 1)
        self.assertAlmostEqual(model[1].p, 0.15)

        optimizer = _build_optimizer(model, {"optimizer": "adam", "learning_rate": 1e-4, "weight_decay": 0.01})
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(optimizer.param_groups[0]["weight_decay"], 0.01)

        scheduler = _build_scheduler(optimizer, {"scheduler": {"name": "cosine", "t_max": 3, "eta_min": 1e-6}}, epochs=5)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_model_passes_encoder_weights_to_smp_models(self) -> None:
        captured: dict[str, object] = {}

        class FakeSegformer(torch.nn.Module):
            def __init__(self, **kwargs: object) -> None:
                super().__init__()
                captured.update(kwargs)

        fake_smp = SimpleNamespace(Segformer=FakeSegformer)
        with patch.dict(sys.modules, {"segmentation_models_pytorch": fake_smp}):
            _build_model("segformer_b2", in_channels=4, out_channels=1, base_channels=8, encoder_weights="imagenet")

        self.assertEqual(captured["encoder_name"], "mit_b2")
        self.assertEqual(captured["encoder_weights"], "imagenet")
        self.assertEqual(captured["in_channels"], 4)

    def test_train_augmentations_support_hflip_and_vflip_keys(self) -> None:
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16.0
        y = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        random_masks = [torch.tensor([0.0]), torch.tensor([0.0])]

        def fake_rand(*args: object, **kwargs: object) -> torch.Tensor:
            return random_masks.pop(0)

        with patch("torch.rand", side_effect=fake_rand):
            augmented_x, augmented_y = _apply_train_augmentations(x, y, {"hflip": True, "vflip": True})

        expected_y = torch.flip(torch.flip(y, dims=(-1,)), dims=(-2,))
        self.assertTrue(torch.equal(augmented_y, expected_y))
        self.assertTrue(torch.allclose(augmented_x, torch.flip(torch.flip(x, dims=(-1,)), dims=(-2,))))

    def test_metric_threshold_sweep_keeps_base_and_deduplicates(self) -> None:
        job = SimpleNamespace(train={"metric_thresholds": [0.7, "0.80", 0.8]}, evaluate={}, params={})
        self.assertEqual(_resolve_metric_thresholds(job, 0.75), [0.7, 0.75, 0.8])
        self.assertEqual(_threshold_metric_suffix(0.8), "0_8")

    def test_scene_tile_limit_allows_balancing_negative_scenes(self) -> None:
        self.assertEqual(_scene_tile_limit(32, positive_scene=True), 32)
        self.assertEqual(_scene_tile_limit(32, positive_scene=False), 32)
        self.assertEqual(
            _scene_tile_limit(
                32,
                positive_scene=True,
                max_tiles_per_positive_scene=24,
                max_tiles_per_negative_scene=4,
            ),
            24,
        )
        self.assertEqual(
            _scene_tile_limit(
                32,
                positive_scene=False,
                max_tiles_per_positive_scene=24,
                max_tiles_per_negative_scene=4,
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
