from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlsystem.src.metrics.api import compute_pixel_metrics_from_logits
from mlsystem.src.metrics.segmentation import metrics_from_counts, pixel_counts_from_masks
from mlsystem.src.train._trainer import _WeightedLossAccumulator
from mlsystem.src.train._losses import segmentation_loss_components


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
        metrics = compute_pixel_metrics_from_logits(logits, target, threshold=0.5)
        self.assertEqual(metrics["pixel_tp"], 2)
        self.assertEqual(metrics["pixel_fp"], 0)

    def test_shape_mismatch_fails_instead_of_silent_resize(self) -> None:
        with self.assertRaises(ValueError):
            pixel_counts_from_masks(np.zeros((4, 4)), np.zeros((8, 8)))

    def test_weighted_loss_average_is_not_last_batch(self) -> None:
        acc = _WeightedLossAccumulator()
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


if __name__ == "__main__":
    unittest.main()
