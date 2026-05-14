from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from mlsystem.src.io_utils import write_json
from scripts.tune_cuttings_forever import (
    build_checkpoint_reeval_trace,
    build_finetune_trace,
    run_dir_checkpoint_rows,
)


class CuttingsTuningOperatorTests(unittest.TestCase):
    def test_run_dir_checkpoint_rows_discovers_cuttings_training_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run-a"
            run_dir.mkdir()
            checkpoint = run_dir / "unet_resnet34.pt"
            torch.save({"model_state_dict": {}}, checkpoint)
            write_json(
                run_dir / "trace.json",
                {
                    "experiment_id": "old_cuttings_run",
                    "class_name": "cuttings",
                    "model": {"name": "unet_resnet34", "input_bands": [1, 2, 3, 4]},
                    "preprocess": {"tile_size": 512, "stride": 512},
                    "train": {"loss": {"name": "bce_dice"}, "learning_rate": 0.0005},
                    "mlflow": {"experiment": "mlsystem-deforest"},
                },
            )
            write_json(
                run_dir / "training_result.json",
                {
                    "model_name": "unet_resnet34",
                    "checkpoint_path": str(checkpoint),
                    "best_val_pixel_f1": 0.42,
                    "best_epoch": 17,
                    "mlflow": {"run_id": "old-run-id"},
                    "last_epoch_metrics": {"val/pixel_iou": 0.27, "val/object_f1": 0.31},
                },
            )

            rows = run_dir_checkpoint_rows(Path(tmp))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mlflow_run_id"], "old-run-id")
        self.assertEqual(rows[0]["model_name"], "unet_resnet34")
        self.assertEqual(rows[0]["old_val_pixel_f1"], 0.42)
        self.assertTrue(rows[0]["artifact_exists"])

    def test_checkpoint_reeval_trace_is_eval_only_and_logs_parent(self) -> None:
        snapshot = {
            "repo_path": "/data/mlsystem/MLMarkup",
            "class_dir": "cuttings",
            "scenes_file": "deforestation.txt",
            "annotation_file": "deforestation.geojson",
            "commit": "abc123",
            "branch": "main",
            "dirty": False,
        }
        checkpoint = {
            "checkpoint_id": "old1",
            "checkpoint_path_or_uri": "/data/mlsystem/runs/old/model.pt",
            "mlflow_run_id": "mlflow-old",
            "model_name": "tiny_unet_4ch",
            "tile_size": 512,
            "stride": 512,
            "old_val_pixel_f1": 0.5,
            "checkpoint_loadable": True,
        }

        trace = build_checkpoint_reeval_trace(checkpoint, snapshot, "mlsystem-cuttings-tuning")

        self.assertEqual(trace["train"]["mode"], "eval_only")
        self.assertEqual(trace["train"]["initial_checkpoint_path"], checkpoint["checkpoint_path_or_uri"])
        self.assertEqual(trace["params"]["checkpoint.parent_run_id"], "mlflow-old")
        self.assertFalse(trace["pseudolabel"]["enabled"])

    def test_finetune_trace_keeps_parent_checkpoint_and_uses_full_training(self) -> None:
        snapshot = {
            "repo_path": "/data/mlsystem/MLMarkup",
            "class_dir": "cuttings",
            "scenes_file": "deforestation.txt",
            "annotation_file": "deforestation.geojson",
            "commit": "abc123",
            "branch": "main",
            "dirty": False,
        }
        parent = {
            "checkpoint_id": "old1",
            "checkpoint_path_or_uri": "/data/mlsystem/runs/old/model.pt",
            "mlflow_run_id": "mlflow-old",
            "model_name": "tiny_unet_4ch",
            "new_val_pixel_f1": 0.45,
        }
        candidate = {
            "phase": "short_finetune",
            "tile_size": 768,
            "stride": 512,
            "augmentation_level": 1,
            "loss": "bce_dice",
            "lr": 2e-5,
            "weight_decay": 1e-5,
            "scheduler": "cosine",
            "max_epochs": 15,
            "batch_size": 2,
            "hypothesis": "short ft",
        }

        trace = build_finetune_trace(parent, snapshot, "mlsystem-cuttings-tuning", candidate)

        self.assertEqual(trace["train"]["mode"], "train")
        self.assertEqual(trace["train"]["initial_checkpoint_path"], parent["checkpoint_path_or_uri"])
        self.assertTrue(trace["train"]["checkpoint_finetune"])
        self.assertIsNone(trace["train"]["max_train_batches"])
        self.assertEqual(trace["params"]["tuning.parent_run_id"], "mlflow-old")


if __name__ == "__main__":
    unittest.main()
