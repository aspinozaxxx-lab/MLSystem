from __future__ import annotations

import unittest

from frontend.app.training_report.collector import TrainingReportCollector, _overall_best
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
                    "metrics": [
                        {"key": "f1_pixel", "value": 0.55, "step": 11},
                        {"key": "epochs_total", "value": 12, "step": 11},
                        {"key": "training_time_sec", "value": 123.0, "step": 11},
                    ],
                    "params": [
                        {"key": "model_name", "value": "segformer_b2"},
                        {"key": "train.epochs", "value": 12},
                        {"key": "best_epoch", "value": 11},
                        {"key": "dataset.version", "value": "abc123"},
                        {"key": "dataset.version_source", "value": "mlmarkup_git_commit"},
                        {"key": "dataset.fingerprint", "value": "fp1"},
                        {"key": "dataset.objects", "value": "42"},
                        {"key": "dataset.scenes", "value": "7"},
                        {"key": "dataset.train_scenes", "value": "5"},
                        {"key": "dataset.val_scenes", "value": "2"},
                        {"key": "dataset.git_commit_date", "value": "2026-05-11T10:00:00+00:00"},
                    ],
                    "tags": [{"key": "dataset.class_slug", "value": "lakes"}],
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
        self.assertEqual(run["dataset_version"], "abc123")
        self.assertEqual(run["dataset_version_source"], "mlmarkup_git_commit")
        self.assertEqual(run["dataset_fingerprint"], "fp1")
        self.assertEqual(run["dataset_objects"], 42)
        self.assertEqual(run["dataset_scenes"], 7)
        self.assertEqual(run["dataset_train_scenes"], 5)
        self.assertEqual(run["dataset_val_scenes"], 2)
        self.assertEqual(run["dataset_date"], "2026-05-11")

    def test_best_run_per_dataset_version_selected_by_pixel_f1(self) -> None:
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
                    "epochs_completed": 12,
                    "dataset_version": "v1",
                    "dataset_objects": 2,
                    "dataset_scenes": 3,
                },
                {
                    "run_id": "high",
                    "class_slug": "lakes",
                    "pixel_f1": 0.9,
                    "run_url": "/mlflow/#/experiments/1/runs/high",
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                    "dataset_version": "v1",
                    "dataset_objects": 2,
                    "dataset_scenes": 3,
                },
                {
                    "run_id": "other_version",
                    "class_slug": "lakes",
                    "pixel_f1": 0.3,
                    "run_url": "/mlflow/#/experiments/1/runs/other_version",
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                    "dataset_version": "v2",
                    "dataset_objects": 2,
                    "dataset_scenes": 3,
                },
            ],
        )
        lakes = next(item for item in rows if item["class_slug"] == "lakes")
        self.assertEqual([run["run_id"] for run in lakes["dataset_versions"]], ["high", "other_version"])
        self.assertEqual(lakes["top_runs"][0]["run_id"], "high")
        self.assertEqual(lakes["best_pixel_f1"], 0.9)
        self.assertEqual(lakes["dataset_version"], "v1")

    def test_overall_best_selected_from_best_per_dataset_version_rows(self) -> None:
        collector = object.__new__(TrainingReportCollector)
        rows = collector._build_class_rows(  # pylint: disable=protected-access
            {"lakes": {"objects_count": 2, "scenes_count": 3}, "deforest": {"objects_count": 3, "scenes_count": 4}},
            [
                {"run_id": "lakes", "class_slug": "lakes", "pixel_f1": 0.4, "train_date": "2026-05-10", "best_epoch": 2, "epochs_completed": 10, "dataset_version": "v1", "dataset_objects": 2, "dataset_scenes": 3},
                {"run_id": "deforest", "class_slug": "deforest", "pixel_f1": 0.7, "train_date": "2026-05-10", "best_epoch": 10, "epochs_completed": 10, "dataset_version": "v2", "dataset_objects": 3, "dataset_scenes": 4},
            ],
        )
        overall = _overall_best(rows)  # pylint: disable=protected-access
        self.assertEqual(overall["run_id"], "deforest")

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
                    "epochs_completed": 12,
                    "dataset_version": "v1",
                    "dataset_objects": 2,
                    "dataset_scenes": 3,
                },
                {
                    "run_id": "real",
                    "class_slug": "lakes",
                    "pixel_f1": 0.42,
                    "run_url": "/mlflow/#/experiments/1/runs/real",
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                    "dataset_version": "v1",
                    "dataset_objects": 2,
                    "dataset_scenes": 3,
                },
            ],
        )
        lakes = next(item for item in rows if item["class_slug"] == "lakes")
        self.assertEqual([run["run_id"] for run in lakes["top_runs"]], ["real"])
        self.assertEqual(lakes["best_pixel_f1"], 0.42)

    def test_old_run_without_dataset_version_is_not_assigned_current_inventory(self) -> None:
        collector = object.__new__(TrainingReportCollector)
        rows = collector._build_class_rows(  # pylint: disable=protected-access
            {"lakes": {"objects_count": 2, "scenes_count": 3, "dataset_date": "2026-05-10", "dataset_fingerprint": "inventory-fp"}},
            [
                {
                    "run_id": "run",
                    "class_slug": "lakes",
                    "pixel_f1": 0.42,
                    "run_url": "/mlflow/#/experiments/1/runs/run",
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                },
            ],
        )
        lakes = next(item for item in rows if item["class_slug"] == "lakes")
        self.assertEqual(lakes["dataset_versions"], [])
        self.assertEqual(lakes["quality_filter"]["reasons"]["missing_dataset_version"], 1)

    def test_run_before_dataset_publication_date_is_excluded(self) -> None:
        collector = object.__new__(TrainingReportCollector)
        rows = collector._build_class_rows(  # pylint: disable=protected-access
            {"lakes": {"objects_count": 2, "scenes_count": 3}},
            [
                {
                    "run_id": "too_old_for_version",
                    "class_slug": "lakes",
                    "pixel_f1": 0.42,
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                    "dataset_version": "v2",
                    "dataset_date": "2026-05-18",
                    "dataset_objects": 2,
                    "dataset_scenes": 3,
                },
            ],
        )
        lakes = next(item for item in rows if item["class_slug"] == "lakes")
        self.assertEqual(lakes["dataset_versions"], [])
        self.assertEqual(lakes["quality_filter"]["reasons"]["train_date_before_dataset_date"], 1)

    def test_run_after_next_dataset_version_date_is_excluded(self) -> None:
        collector = object.__new__(TrainingReportCollector)
        rows = collector._build_class_rows(  # pylint: disable=protected-access
            {"lakes": {"objects_count": 2, "scenes_count": 3}},
            [
                {
                    "run_id": "stale_v1",
                    "class_slug": "lakes",
                    "pixel_f1": 0.42,
                    "train_date": "2026-05-19",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                    "dataset_version": "v1",
                    "dataset_date": "2026-05-10",
                    "dataset_objects": 2,
                    "dataset_scenes": 3,
                },
                {
                    "run_id": "v2",
                    "class_slug": "lakes",
                    "pixel_f1": 0.5,
                    "train_date": "2026-05-18",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                    "dataset_version": "v2",
                    "dataset_date": "2026-05-18",
                    "dataset_objects": 2,
                    "dataset_scenes": 3,
                },
            ],
        )
        lakes = next(item for item in rows if item["class_slug"] == "lakes")
        self.assertEqual([run["run_id"] for run in lakes["dataset_versions"]], ["v2"])
        self.assertEqual(lakes["quality_filter"]["reasons"]["train_date_after_next_dataset_version"], 1)

    def test_run_with_zero_scenes_is_excluded_unless_exact_inventory_version_matches(self) -> None:
        collector = object.__new__(TrainingReportCollector)
        rows = collector._build_class_rows(  # pylint: disable=protected-access
            {"lakes": {"objects_count": 2, "scenes_count": 3, "dataset_date": "2026-05-10", "dataset_fingerprint": "inventory-fp"}},
            [
                {
                    "run_id": "bad",
                    "class_slug": "lakes",
                    "pixel_f1": 0.42,
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                    "dataset_version": "other-fp",
                    "dataset_scenes": 0,
                    "dataset_objects": 2,
                },
                {
                    "run_id": "fallback_exact",
                    "class_slug": "lakes",
                    "pixel_f1": 0.5,
                    "train_date": "2026-05-10",
                    "best_epoch": 12,
                    "epochs_completed": 12,
                    "dataset_version": "inventory-fp",
                    "dataset_fingerprint": "inventory-fp",
                    "dataset_scenes": 0,
                    "dataset_objects": 0,
                },
            ],
        )
        lakes = next(item for item in rows if item["class_slug"] == "lakes")
        self.assertEqual([run["run_id"] for run in lakes["dataset_versions"]], ["fallback_exact"])
        self.assertEqual(lakes["dataset_versions"][0]["dataset_scenes"], 3)
        self.assertEqual(lakes["quality_filter"]["reasons"]["missing_dataset_scene_count"], 1)

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
                    "epochs_completed": 12,
                    "run_status": "FINISHED",
                    "dataset_version": "v1",
                    "dataset_objects": 21,
                    "dataset_scenes": 2,
                },
                {
                    "run_id": "early_best",
                    "class_slug": "wind_erosion",
                    "pixel_f1": 0.28,
                    "run_url": "/mlflow/#/experiments/1/runs/early_best",
                    "train_date": "2026-05-10",
                    "best_epoch": 9,
                    "epochs_completed": 9,
                    "run_status": "FINISHED",
                    "dataset_version": "v1",
                    "dataset_objects": 21,
                    "dataset_scenes": 2,
                },
                {
                    "run_id": "trusted",
                    "class_slug": "wind_erosion",
                    "pixel_f1": 0.18,
                    "run_url": "/mlflow/#/experiments/1/runs/trusted",
                    "train_date": "2026-05-10",
                    "best_epoch": 1,
                    "epochs_completed": 10,
                    "run_status": "FINISHED",
                    "dataset_version": "v1",
                    "dataset_objects": 21,
                    "dataset_scenes": 2,
                },
            ],
        )
        wind = next(item for item in rows if item["class_slug"] == "wind_erosion")
        self.assertEqual([run["run_id"] for run in wind["top_runs"]], ["trusted"])
        self.assertEqual(wind["quality_filter"]["excluded_runs"], 2)


if __name__ == "__main__":
    unittest.main()
