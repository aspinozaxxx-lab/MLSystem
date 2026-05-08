from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mlsystem.src.metrics.debug_dump import recompute_global_metrics, write_epoch_debug, write_metrics_debug_report
from mlsystem.src.metrics.segmentation import (
    PixelCounts,
    WeightedLossAccumulator,
    binary_segmentation_metrics_from_logits,
    metrics_from_counts,
    pixel_counts_from_masks,
)
from mlsystem.src.training.losses import segmentation_loss_components


class SegmentationMetricsTests(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        gt = np.zeros((8, 8), dtype=np.uint8)
        gt[2:6, 2:6] = 1
        metrics = metrics_from_counts(pixel_counts_from_masks(gt, gt))
        self.assertEqual(metrics["pixel_f1"], 1.0)
        self.assertEqual(metrics["pixel_iou"], 1.0)
        self.assertEqual(metrics["pixel_tp"], 16)

    def test_empty_pred_non_empty_gt(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[:2, :2] = 1
        pred = np.zeros_like(gt)
        metrics = metrics_from_counts(pixel_counts_from_masks(pred, gt))
        self.assertEqual(metrics["pixel_f1"], 0.0)
        self.assertEqual(metrics["pixel_iou"], 0.0)
        self.assertEqual(metrics["pixel_fn"], 4)

    def test_non_empty_pred_empty_gt(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros_like(gt)
        pred[:2, :2] = 1
        metrics = metrics_from_counts(pixel_counts_from_masks(pred, gt))
        self.assertEqual(metrics["pixel_f1"], 0.0)
        self.assertEqual(metrics["pixel_iou"], 0.0)
        self.assertEqual(metrics["pixel_fp"], 4)

    def test_both_empty_is_defined_as_perfect_empty_match(self) -> None:
        metrics = metrics_from_counts(pixel_counts_from_masks(np.zeros((4, 4)), np.zeros((4, 4))))
        self.assertEqual(metrics["pixel_precision"], 1.0)
        self.assertEqual(metrics["pixel_recall"], 1.0)
        self.assertEqual(metrics["pixel_f1"], 1.0)
        self.assertEqual(metrics["pixel_iou"], 1.0)

    def test_half_overlap_counts(self) -> None:
        gt = np.array([[1, 1, 0, 0]], dtype=np.uint8)
        pred = np.array([[0, 1, 1, 0]], dtype=np.uint8)
        metrics = metrics_from_counts(pixel_counts_from_masks(pred, gt))
        self.assertEqual(metrics["pixel_tp"], 1)
        self.assertEqual(metrics["pixel_fp"], 1)
        self.assertEqual(metrics["pixel_fn"], 1)
        self.assertAlmostEqual(metrics["pixel_f1"], 0.5)
        self.assertAlmostEqual(metrics["pixel_iou"], 1 / 3)

    def test_ignore_index_pixels_are_excluded(self) -> None:
        gt = np.array([[1, 255, 0, 0]], dtype=np.uint8)
        pred = np.array([[1, 1, 1, 0]], dtype=np.uint8)
        metrics = metrics_from_counts(pixel_counts_from_masks(pred, gt, ignore_index=255))
        self.assertEqual(metrics["pixel_tp"], 1)
        self.assertEqual(metrics["pixel_fp"], 1)
        self.assertEqual(metrics["pixel_fn"], 0)
        self.assertEqual(metrics["pixel_tn"], 1)

    def test_threshold_from_logits(self) -> None:
        probs = torch.tensor([[[[0.49, 0.50, 0.51]]]], dtype=torch.float32)
        logits = torch.logit(probs.clamp(1e-5, 1 - 1e-5))
        target = torch.tensor([[[[0.0, 1.0, 1.0]]]], dtype=torch.float32)
        metrics = binary_segmentation_metrics_from_logits(logits, target, threshold=0.5)
        self.assertEqual(metrics["pixel_tp"], 2)
        self.assertEqual(metrics["pixel_fp"], 0)

    def test_shape_mismatch_fails_instead_of_silent_resize(self) -> None:
        with self.assertRaises(ValueError):
            pixel_counts_from_masks(np.zeros((4, 4)), np.zeros((8, 8)))

    def test_weighted_loss_average_is_not_last_batch(self) -> None:
        acc = WeightedLossAccumulator()
        acc.update({"loss_total": 1.0, "loss_bce": 0.25}, weight=1)
        acc.update({"loss_total": 3.0, "loss_bce": 0.75}, weight=3)
        averages = acc.averages("train")
        self.assertAlmostEqual(averages["train/loss_total"], 2.5)
        self.assertAlmostEqual(averages["train/loss_bce"], 0.625)

    def test_loss_config_supports_weighted_focal_dice(self) -> None:
        logits = torch.tensor([[[[0.0, 1.0], [-1.0, 0.5]]]], dtype=torch.float32)
        target = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32)
        components = segmentation_loss_components(
            logits,
            target,
            config={"name": "focal_dice", "focal_weight": 0.25, "dice_weight": 0.75, "focal_gamma": 2.0},
        )
        expected = components["loss_focal"] * 0.25 + components["loss_dice"] * 0.75
        self.assertTrue(torch.isfinite(components["loss_total"]))
        self.assertTrue(torch.allclose(components["loss_total"], expected))

    def test_loss_config_keeps_bce_dice_default(self) -> None:
        logits = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        target = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        components = segmentation_loss_components(logits, target)
        self.assertTrue(torch.allclose(components["loss_total"], components["loss_bce"] + components["loss_dice"]))

    def test_debug_recompute_matches_logged_metrics(self) -> None:
        rows = [
            {"sample_id": "a", "tp": 2, "fp": 1, "fn": 0, "tn": 5},
            {"sample_id": "b", "tp": 1, "fp": 0, "fn": 2, "tn": 4},
        ]
        row = {
            "val/pixel_tp": 3,
            "val/pixel_fp": 1,
            "val/pixel_fn": 2,
            "val/pixel_tn": 9,
            "val/precision": 0.75,
            "val/recall": 0.6,
            "val/pixel_f1": 2 * 3 / (2 * 3 + 1 + 2),
            "val/pixel_iou": 3 / (3 + 1 + 2),
        }
        result = recompute_global_metrics(rows, row)
        self.assertTrue(result["ok"])

    def test_debug_dump_writes_epoch_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = write_epoch_debug(
                root_dir=Path(tmp),
                run_id="r1",
                epoch=1,
                mlflow_run_id="m1",
                model_checkpoint_path=None,
                val_manifest_path=None,
                val_manifest=[{"sample_id": "s1"}],
                class_name="deforest",
                class_id=1,
                threshold=0.5,
                metric_row={
                    "val/pixel_tp": 1,
                    "val/pixel_fp": 0,
                    "val/pixel_fn": 0,
                    "val/pixel_tn": 3,
                    "val/precision": 1.0,
                    "val/recall": 1.0,
                    "val/pixel_f1": 1.0,
                    "val/pixel_iou": 1.0,
                },
                train_loss={"train/loss_total": 0.2},
                val_loss={"val/loss_total": 0.1},
                per_sample_metrics=[{"sample_id": "s1", "tp": 1, "fp": 0, "fn": 0, "tn": 3}],
                sample_payloads=[],
                logged_metrics={"val/pixel_f1": 1.0},
            )
            self.assertTrue(Path(result["epoch_summary"]).exists())
            self.assertTrue(Path(result["production_metrics_snapshot"]).exists())
            self.assertTrue(Path(result["per_sample_metrics"]).exists())
            self.assertTrue(result["recompute"]["ok"])

    def test_debug_report_folder_structure_excludes_raw_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "debug"
            report_root = Path(tmp) / "reports"
            metric_row = {
                "val/pixel_tp": 1,
                "val/pixel_fp": 0,
                "val/pixel_fn": 0,
                "val/pixel_tn": 3,
                "val/precision": 1.0,
                "val/recall": 1.0,
                "val/pixel_f1": 1.0,
                "val/pixel_iou": 1.0,
                "val/object_tp": 1,
                "val/object_fp": 0,
                "val/object_fn": 0,
                "val/object_f1": 1.0,
            }
            payload = {
                "metadata": {"sample_id": "s1", "scene_id": "scene.tif"},
                "image": np.ones((3, 4, 4), dtype=np.float32),
                "gt_mask": np.ones((4, 4), dtype=np.uint8),
                "pred_prob": np.ones((4, 4), dtype=np.float32),
                "pred_mask": np.ones((4, 4), dtype=np.uint8),
                "objects_gt": [],
                "objects_pred": [],
            }
            write_epoch_debug(
                root_dir=root,
                run_id="r1",
                epoch=1,
                mlflow_run_id="m1",
                model_checkpoint_path=None,
                val_manifest_path="manifest.json",
                val_manifest=[{"sample_id": "s1"}],
                class_name="вырубки",
                class_id=1,
                threshold=0.5,
                metric_row=metric_row,
                train_loss={"train/loss_total": 0.2},
                val_loss={"val/loss_total": 0.1},
                per_sample_metrics=[{"sample_id": "s1", "tp": 1, "fp": 0, "fn": 0, "tn": 3}],
                sample_payloads=[payload],
                logged_metrics=metric_row,
            )
            result = write_metrics_debug_report(
                debug_root=root / "r1",
                report_root=report_root,
                report_name="report",
                run_metadata={"airflow": {"dag_id": "mlsystem_experiment_pipeline", "run_id": "airflow_run"}},
                dataset_check={"class_name": "вырубки", "full_dataset": True, "synthetic": False},
            )
            report_dir = Path(result["report_dir"])
            self.assertTrue((report_dir / "summary.md").exists())
            self.assertTrue((report_dir / "epochs" / "epoch_0001" / "production_metrics_snapshot.json").exists())
            self.assertFalse((report_dir / "epochs" / "epoch_0001" / "samples" / "s1" / "image.png").exists())
            self.assertFalse((report_dir / "epochs" / "epoch_0001" / "samples" / "s1" / "pred_prob.npz").exists())
            self.assertTrue((report_dir / "epochs" / "epoch_0001" / "samples" / "s1" / "gt_mask.png").exists())
            source_check = json.loads((report_dir / "checks" / "source_of_truth_check.json").read_text(encoding="utf-8"))
            self.assertTrue(source_check["production_snapshot_reused"])


if __name__ == "__main__":
    unittest.main()
