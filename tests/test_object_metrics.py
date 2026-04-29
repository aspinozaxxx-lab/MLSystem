from __future__ import annotations

import unittest

from shapely.geometry import box

from mlsystem.src.object_metrics import compute_object_f1


class ObjectMetricsTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        metrics = compute_object_f1([box(0, 0, 10, 10)], [box(0, 0, 10, 10)])
        self.assertEqual(metrics["object_tp"], 1)
        self.assertEqual(metrics["object_fp"], 0)
        self.assertEqual(metrics["object_fn"], 0)
        self.assertEqual(metrics["object_f1"], 1.0)

    def test_no_intersection(self) -> None:
        metrics = compute_object_f1([box(0, 0, 10, 10)], [box(20, 20, 30, 30)])
        self.assertEqual(metrics["object_tp"], 0)
        self.assertEqual(metrics["object_fp"], 1)
        self.assertEqual(metrics["object_fn"], 1)
        self.assertEqual(metrics["object_f1"], 0.0)

    def test_iou_above_threshold_is_tp(self) -> None:
        metrics = compute_object_f1([box(0, 0, 10, 10)], [box(0, 0, 8, 10)])
        self.assertEqual(metrics["object_tp"], 1)
        self.assertGreater(metrics["matches"][0]["iou"], 0.5)

    def test_iou_equal_or_below_threshold_is_not_tp(self) -> None:
        metrics = compute_object_f1([box(0, 0, 10, 10)], [box(5, 0, 15, 10)])
        self.assertEqual(metrics["object_tp"], 0)
        self.assertEqual(metrics["object_fp"], 1)
        self.assertEqual(metrics["object_fn"], 1)

    def test_two_predictions_for_one_gt_count_once(self) -> None:
        metrics = compute_object_f1(
            [box(0, 0, 10, 10), box(0, 0, 9, 10)],
            [box(0, 0, 10, 10)],
        )
        self.assertEqual(metrics["object_tp"], 1)
        self.assertEqual(metrics["object_fp"], 1)
        self.assertEqual(metrics["object_fn"], 0)

    def test_one_prediction_for_two_gt_count_once(self) -> None:
        metrics = compute_object_f1(
            [box(0, 0, 10, 10)],
            [box(0, 0, 10, 10), box(0, 0, 9, 10)],
        )
        self.assertEqual(metrics["object_tp"], 1)
        self.assertEqual(metrics["object_fp"], 0)
        self.assertEqual(metrics["object_fn"], 1)


if __name__ == "__main__":
    unittest.main()
