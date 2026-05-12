from __future__ import annotations

import unittest

from frontend.app.training_report.collector import TrainingReportCollector
from frontend.app.training_report.mlflow_reader import normalize_pixel_f1, normalize_run


class TrainingReportMLflowReaderTests(unittest.TestCase):
    def test_normalize_pixel_f1_uses_priority_metric(self) -> None:
        payload = normalize_pixel_f1({"object_f1": 0.99, "val_pixel_f1": 0.42, "dice": 0.4})
        self.assertEqual(payload["pixel_f1"], 0.42)
        self.assertEqual(payload["metric_name_source"], "val_pixel_f1")

    def test_threshold_sweep_uses_best_pixel_f1_threshold(self) -> None:
        payload = normalize_pixel_f1({"val_pixel_f1_threshold_0.3": 0.31, "val_pixel_f1_threshold_0.5": 0.44})
        self.assertEqual(payload["pixel_f1"], 0.44)
        self.assertEqual(payload["best_threshold"], 0.5)

    def test_normalize_run_detects_class_gateway_url_and_training_metadata(self) -> None:
        run = normalize_run(
            {
                "info": {
                    "run_id": "abc",
                    "experiment_id": "38",
                    "start_time": 1_714_000_000_000,
                    "end_time": 1_714_000_123_000,
                    "status": "FINISHED",
                },
                "data": {
                    "metrics": [{"key": "best_val_pixel_f1", "value": 0.55, "step": 11}],
                    "params": [{"key": "model_name", "value": "segformer_b2"}, {"key": "train.epochs", "value": 12}],
                    "tags": [{"key": "mlsystem.class_name", "value": "lakes"}],
                },
            }
        )
        self.assertEqual(run["class_slug"], "lakes")
        self.assertEqual(run["run_url"], "/mlflow/#/experiments/38/runs/abc")
        self.assertEqual(run["pixel_f1"], 0.55)
        self.assertEqual(run["run_status"], "FINISHED")
        self.assertEqual(run["training_duration_sec"], 123.0)
        self.assertEqual(run["best_epoch"], 11.0)
        self.assertEqual(run["epochs_planned"], 12.0)

    def test_top_runs_sorted_desc(self) -> None:
        collector = object.__new__(TrainingReportCollector)
        rows = collector._build_class_rows(  # pylint: disable=protected-access
            {"lakes": {"objects_count": 2, "scenes_count": 3, "dataset_date": "2026-05-10"}},
            [
                {
                    "run_id": "low",
                    "class_slug": "lakes",
                    "pixel_f1": 0.1,
                    "run_url": "/mlflow/#/experiments/1/runs/low",
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                },
                {
                    "run_id": "high",
                    "class_slug": "lakes",
                    "pixel_f1": 0.9,
                    "run_url": "/mlflow/#/experiments/1/runs/high",
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                },
            ],
        )
        lakes = next(item for item in rows if item["class_slug"] == "lakes")
        self.assertEqual(lakes["top_runs"][0]["run_id"], "high")
        self.assertEqual(lakes["best_pixel_f1"], 0.9)

    def test_perfect_pixel_f1_runs_are_excluded(self) -> None:
        collector = object.__new__(TrainingReportCollector)
        rows = collector._build_class_rows(  # pylint: disable=protected-access
            {"lakes": {"objects_count": 2, "scenes_count": 3, "dataset_date": "2026-05-10"}},
            [
                {
                    "run_id": "perfect",
                    "class_slug": "lakes",
                    "pixel_f1": 1.0,
                    "run_url": "/mlflow/#/experiments/1/runs/perfect",
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                },
                {
                    "run_id": "real",
                    "class_slug": "lakes",
                    "pixel_f1": 0.42,
                    "run_url": "/mlflow/#/experiments/1/runs/real",
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                },
            ],
        )
        lakes = next(item for item in rows if item["class_slug"] == "lakes")
        self.assertEqual([run["run_id"] for run in lakes["top_runs"]], ["real"])
        self.assertEqual(lakes["best_pixel_f1"], 0.42)

    def test_runs_before_cutoff_or_before_epoch_10_are_excluded(self) -> None:
        collector = object.__new__(TrainingReportCollector)
        rows = collector._build_class_rows(  # pylint: disable=protected-access
            {"wind_erosion": {"objects_count": 21, "scenes_count": 2, "dataset_date": "2026-05-10"}},
            [
                {
                    "run_id": "old",
                    "class_slug": "wind_erosion",
                    "pixel_f1": 0.31,
                    "run_url": "/mlflow/#/experiments/1/runs/old",
                    "train_date": "2026-05-05",
                    "best_epoch": 12,
                    "run_status": "FINISHED",
                },
                {
                    "run_id": "early_best",
                    "class_slug": "wind_erosion",
                    "pixel_f1": 0.28,
                    "run_url": "/mlflow/#/experiments/1/runs/early_best",
                    "train_date": "2026-05-10",
                    "best_epoch": 9,
                    "run_status": "FINISHED",
                },
                {
                    "run_id": "trusted",
                    "class_slug": "wind_erosion",
                    "pixel_f1": 0.18,
                    "run_url": "/mlflow/#/experiments/1/runs/trusted",
                    "train_date": "2026-05-10",
                    "best_epoch": 10,
                    "run_status": "FINISHED",
                },
            ],
        )
        wind = next(item for item in rows if item["class_slug"] == "wind_erosion")
        self.assertEqual([run["run_id"] for run in wind["top_runs"]], ["trusted"])
        self.assertEqual(wind["quality_filter"]["excluded_runs"], 2)


if __name__ == "__main__":
    unittest.main()
