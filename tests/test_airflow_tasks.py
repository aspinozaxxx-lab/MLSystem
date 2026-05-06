from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mlsystem.src.pipeline.airflow_tasks import (
    LEGACY_FALLBACK_STAGES,
    MAIN_DAG_STAGES,
    STAGE_POOLS,
    _extract_pixel_metrics,
    push_stage_xcom,
    run_airflow_stage,
    run_stage,
    safe_xcom_push,
    stage_return_message,
    xcom_safe_value,
)
from mlsystem.src.pipeline.stages.registry import get_stage_entrypoint


SMOKE_CONF = {
    "experiment_id": "unit_airflow_smoke",
    "class_name": "deforest",
    "task": "smoke",
    "smoke": True,
    "train": {"enabled": False},
    "pseudolabel": {"enabled": True},
    "postprocess": {"enabled": True},
    "mlflow": {"experiment": "mlsystem-airflow-smoke"},
}


class AirflowTasksTests(unittest.TestCase):
    def test_main_dag_has_required_stage_order(self) -> None:
        self.assertEqual(MAIN_DAG_STAGES[0], "inventory_scenes")
        self.assertEqual(MAIN_DAG_STAGES[1], "prepare_dataset")
        self.assertEqual(MAIN_DAG_STAGES[-1], "finalize_mlflow_run")
        self.assertIn("prepare_inference_scenes", MAIN_DAG_STAGES)
        self.assertIn("run_pseudolabel_inference", MAIN_DAG_STAGES)
        self.assertIn("validate_probability_maps", MAIN_DAG_STAGES)
        self.assertIn("compute_f1", MAIN_DAG_STAGES)
        self.assertNotIn("compute_object_f1", MAIN_DAG_STAGES)

    def test_main_dag_stages_have_pool_and_entrypoint_or_fallback(self) -> None:
        for stage in MAIN_DAG_STAGES:
            self.assertIn(stage, STAGE_POOLS)
            if stage in LEGACY_FALLBACK_STAGES:
                continue
            self.assertTrue(callable(get_stage_entrypoint(stage)))

    def test_gpu_stages_use_expected_pools(self) -> None:
        self.assertEqual(STAGE_POOLS["train_model"][0], "gpu_training")
        self.assertEqual(STAGE_POOLS["predict_validation_scenes"][0], "gpu_inference")
        self.assertEqual(STAGE_POOLS["run_pseudolabel_inference"][0], "gpu_inference")

    def test_unknown_stage_error_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown Airflow MLSystem stage"):
                run_stage("unknown_stage", SMOKE_CONF, "manual__unit", Path(tmp))

    def test_validate_config_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stage("validate_experiment_config", SMOKE_CONF, "manual__unit", Path(tmp))
            self.assertEqual(result["status"], "skipped")
            summary = Path(tmp) / "unit_airflow_smoke" / "summary.json"
            self.assertTrue(summary.exists())

    def test_s3_stage_skipped_for_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stage("check_s3_layout", SMOKE_CONF, "manual__unit", Path(tmp))
            self.assertEqual(result["status"], "skipped")
            self.assertTrue(result["is_smoke_synthetic"])
            self.assertIn("skip_reason", result)
            self.assertNotIn("resources", result)
            self.assertNotIn("counters", result)

    def test_run_airflow_stage_returns_compact_xcom_summary_in_local_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"MLSYSTEM_AIRFLOW_EXECUTION_MODE": "local"}):
                result = run_airflow_stage("check_s3_layout", SMOKE_CONF, "manual__unit", Path(tmp))
            self.assertEqual(result["stage"], "check_s3_layout")
            self.assertEqual(result["status"], "skipped")
            self.assertIn("report_path", result)
            self.assertIn("stage_json_path", result)
            self.assertNotIn("resources", result)
            self.assertNotIn("stage_report", result)
            self.assertLess(len(str(result).encode("utf-8")), 10_000)

    def test_push_stage_xcom_uses_readable_key_value_pairs(self) -> None:
        class FakeTaskInstance:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def xcom_push(self, *, key: str, value: object) -> None:
                self.values[key] = value

        summary = {
            "stage": "prepare_dataset",
            "status": "success",
            "run_id": "manual__unit",
            "job_id": "job1",
            "summary": "prepare_dataset completed",
            "report_path": "/opt/airflow/mlsystem_runs/unit/stages/prepare_dataset.report.md",
            "stage_json_path": "/opt/airflow/mlsystem_runs/unit/stages/prepare_dataset.json",
            "warnings_count": 1,
            "errors_count": 0,
            "duration_sec": 1.2,
            "key_counters": {
                "total_scenes": 24,
                "total_objects": 301,
                "train_scenes": 19,
                "val_scenes": 5,
                "split_strategy": "object_balanced",
                "upstream_inventory_matched_scenes": 24,
                "accepted_objects": np.int64(7),
            },
            "key_metrics": {"pixel_f1": np.float32(0.75), "pixel_iou": float("nan")},
        }
        ti = FakeTaskInstance()
        push_stage_xcom(summary, ti)
        self.assertEqual(ti.values["stage"], "prepare_dataset")
        self.assertEqual(ti.values["status"], "success")
        self.assertEqual(ti.values["job_id"], "job1")
        self.assertEqual(ti.values["report_path"], summary["report_path"])
        self.assertEqual(ti.values["counter_total_scenes"], 24)
        self.assertEqual(ti.values["split_strategy"], "object_balanced")
        self.assertNotIn("counter_accepted_objects", ti.values)
        self.assertNotIn("metric_pixel_f1", ti.values)
        self.assertNotIn("metric_pixel_iou", ti.values)
        self.assertEqual(ti.values["requested_pool"], "cpu_heavy")
        self.assertEqual(ti.values["effective_pool"], "cpu_heavy")
        self.assertNotIn("return_value", ti.values)
        self.assertNotIn("resources", ti.values)
        self.assertNotIn("stage_report", ti.values)
        self.assertLess(len(str(ti.values).encode("utf-8")), 10_000)
        self.assertLess(len(stage_return_message(summary).encode("utf-8")), 512)

    def test_synthetic_smoke_xcom_is_minimal(self) -> None:
        class FakeTaskInstance:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def xcom_push(self, *, key: str, value: object) -> None:
                self.values[key] = value

        summary = {
            "stage": "inventory_scenes",
            "status": "skipped",
            "run_id": "manual__unit",
            "job_id": "job1",
            "summary": "Synthetic smoke stage skipped: orchestration-only run.",
            "report_path": "/r.md",
            "stage_json_path": "/s.json",
            "duration_sec": 0.1,
            "warnings_count": 1,
            "errors_count": 0,
            "skip_reason": "synthetic smoke validates orchestration only",
            "is_smoke_synthetic": True,
            "key_counters": {"matched_scenes": 24, "cuda_available": True, "gpu_name": "GPU"},
            "key_metrics": {"pixel_f1": 1.0},
        }
        ti = FakeTaskInstance()
        push_stage_xcom(summary, ti)
        self.assertEqual(ti.values["status"], "skipped")
        self.assertTrue(ti.values["is_smoke_synthetic"])
        self.assertIn("skip_reason", ti.values)
        self.assertNotIn("counter_matched_scenes", ti.values)
        self.assertNotIn("cuda_available", ti.values)
        self.assertNotIn("gpu_name", ti.values)
        self.assertNotIn("metric_pixel_f1", ti.values)

    def test_cpu_stage_does_not_push_gpu_xcom(self) -> None:
        class FakeTaskInstance:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def xcom_push(self, *, key: str, value: object) -> None:
                self.values[key] = value

        summary = {
            "stage": "inventory_scenes",
            "status": "success",
            "key_counters": {
                "matched_scenes": 24,
                "missing_scenes": 0,
                "cuda_available": True,
                "gpu_name": "GPU",
            },
        }
        ti = FakeTaskInstance()
        push_stage_xcom(summary, ti)
        self.assertEqual(ti.values["counter_matched_scenes"], 24)
        self.assertNotIn("cuda_available", ti.values)
        self.assertNotIn("gpu_name", ti.values)

    def test_gpu_stage_pushes_gpu_xcom(self) -> None:
        class FakeTaskInstance:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def xcom_push(self, *, key: str, value: object) -> None:
                self.values[key] = value

        summary = {
            "stage": "run_pseudolabel_inference",
            "status": "success",
            "key_counters": {
                "pseudolabel_scenes_requested": 5,
                "pseudolabel_scenes_processed": 1,
                "cuda_available": True,
                "gpu_name": "GPU",
                "device": "cuda",
            },
        }
        ti = FakeTaskInstance()
        push_stage_xcom(summary, ti)
        self.assertTrue(ti.values["cuda_available"])
        self.assertEqual(ti.values["gpu_name"], "GPU")
        self.assertEqual(ti.values["device"], "cuda")
        self.assertEqual(ti.values["counter_pseudolabel_scenes_processed"], 1)

    def test_evaluate_pixel_metrics_xcom_allowlist(self) -> None:
        class FakeTaskInstance:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def xcom_push(self, *, key: str, value: object) -> None:
                self.values[key] = value

        summary = {
            "stage": "evaluate_pixel_metrics",
            "status": "success",
            "key_counters": {
                "scenes_evaluated": 5,
                "pixel_tp": 1,
                "pixel_fp": 2,
                "cuda_available": True,
                "gpu_name": "GPU",
                "threshold": 0.5,
            },
            "key_metrics": {
                "pixel_precision": 0.1,
                "pixel_recall": 0.2,
                "pixel_f1": 0.133333,
                "object_f1": 0.9,
            },
        }
        ti = FakeTaskInstance()
        push_stage_xcom(summary, ti)
        self.assertEqual(ti.values["counter_scenes_evaluated"], 5)
        self.assertEqual(ti.values["counter_pixel_tp"], 1)
        self.assertEqual(ti.values["threshold"], 0.5)
        self.assertEqual(ti.values["metric_pixel_precision"], 0.1)
        self.assertNotIn("metric_object_f1", ti.values)
        self.assertNotIn("cuda_available", ti.values)

    def test_pixel_f1_inconsistency_is_not_published_as_original_value(self) -> None:
        metrics, _counters, aliases, warnings = _extract_pixel_metrics(
            {"last_epoch_metrics": {"val/precision": 1e-8, "val/recall": 0.75, "val/pixel_f1": 0.25, "val/iou": 1e-8}}
        )
        expected_f1 = 2 * 1e-8 * 0.75 / (1e-8 + 0.75)
        self.assertAlmostEqual(metrics["pixel_f1"], expected_f1)
        self.assertTrue(any("inconsistent" in warning for warning in warnings))
        self.assertTrue(any(row["normalized_metric_name"] == "pixel_f1_source_value_not_used" for row in aliases))

    def test_xcom_safe_value_normalizes_unsafe_types(self) -> None:
        self.assertEqual(xcom_safe_value(np.int64(5)), 5)
        self.assertEqual(type(xcom_safe_value(np.int64(5))), int)
        self.assertAlmostEqual(xcom_safe_value(np.float32(1.25)), 1.25, places=5)
        self.assertIsInstance(xcom_safe_value(np.float64(0.25)), float)
        self.assertEqual(xcom_safe_value(Decimal("0.25")), 0.25)
        self.assertEqual(xcom_safe_value(0), 0)
        self.assertEqual(xcom_safe_value(0.0), 0.0)
        self.assertIsNone(xcom_safe_value(float("nan")))
        self.assertIsNone(xcom_safe_value(float("inf")))
        self.assertIsNone(xcom_safe_value(float("-inf")))
        self.assertEqual(xcom_safe_value(Path("/tmp/report.txt")), str(Path("/tmp/report.txt")))
        self.assertEqual(xcom_safe_value(datetime(2026, 5, 5, 12, 0)), "2026-05-05T12:00:00")
        self.assertIsNone(xcom_safe_value({"a": 1}))
        self.assertIsNone(xcom_safe_value(["a", "b"]))
        self.assertIsNone(xcom_safe_value({"a", "b"}))
        self.assertIsNone(xcom_safe_value(("a", "b")))
        self.assertIsNone(xcom_safe_value(b"bytes"))
        self.assertIsNone(xcom_safe_value(object()))

    def test_safe_xcom_push_keeps_values_scalar(self) -> None:
        class FakeTaskInstance:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def xcom_push(self, *, key: str, value: object) -> None:
                self.values[key] = value

        ti = FakeTaskInstance()
        safe_xcom_push(ti, "counter_accepted_objects", np.int64(9))
        safe_xcom_push(ti, "metric_pixel_f1", np.float64(0.25))
        safe_xcom_push(ti, "metric_pixel_iou", Decimal("0.125"))
        safe_xcom_push(ti, "counter_zero", 0)
        safe_xcom_push(ti, "metric_zero", 0.0)
        safe_xcom_push(ti, "metric_none", None)
        safe_xcom_push(ti, "counter_none", None)
        safe_xcom_push(ti, "metric_nan", float("nan"))
        safe_xcom_push(ti, "metric_inf", float("inf"))
        safe_xcom_push(ti, "counter_unknown", "unknown")
        safe_xcom_push(ti, "metric_not_available", "not_available")
        safe_xcom_push(ti, "counter_bool", True)
        safe_xcom_push(ti, "counter_path", Path("/tmp/report.txt"))
        safe_xcom_push(ti, "summary_path", Path("/tmp/report.txt"))
        safe_xcom_push(ti, "summary", {"too": "nested"})
        safe_xcom_push(ti, "unsupported_blob", {"skip": True})
        self.assertEqual(ti.values["counter_accepted_objects"], 9)
        self.assertEqual(ti.values["metric_pixel_f1"], 0.25)
        self.assertEqual(ti.values["metric_pixel_iou"], 0.125)
        self.assertEqual(ti.values["counter_zero"], 0)
        self.assertEqual(ti.values["metric_zero"], 0.0)
        self.assertEqual(ti.values["summary_path"], str(Path("/tmp/report.txt")))
        self.assertNotIn("summary", ti.values)
        self.assertNotIn("unsupported_blob", ti.values)
        self.assertNotIn("metric_none", ti.values)
        self.assertNotIn("counter_none", ti.values)
        self.assertNotIn("metric_nan", ti.values)
        self.assertNotIn("metric_inf", ti.values)
        self.assertNotIn("counter_unknown", ti.values)
        self.assertNotIn("metric_not_available", ti.values)
        self.assertNotIn("counter_bool", ti.values)
        self.assertNotIn("counter_path", ti.values)
        for value in ti.values.values():
            self.assertIsInstance(value, (str, int, float, bool))
            json.dumps(value, allow_nan=False)

    def test_compute_f1_xcom_skips_unavailable_object_metrics(self) -> None:
        class FakeTaskInstance:
            def __init__(self) -> None:
                self.values: dict[str, object] = {}

            def xcom_push(self, *, key: str, value: object) -> None:
                self.values[key] = value

        summary = {
            "stage": "compute_f1",
            "status": "success_with_warning",
            "summary": "Pixel metrics summarized; object metrics are not available because validation vectorization is not implemented as a distinct stage yet.",
            "key_counters": {
                "reference_objects": None,
                "predicted_objects": None,
                "tp_objects": None,
                "fp_objects": None,
                "fn_objects": None,
                "reference_scenes": None,
                "prediction_scenes": None,
            },
            "key_metrics": {
                "object_f1": None,
                "object_precision": "not_available",
                "object_recall": None,
                "pixel_precision": np.float64(0.1),
                "pixel_recall": np.float64(0.2),
                "pixel_f1": np.float64(0.13333333333333333),
                "pixel_iou": np.float64(0.07142857142857142),
                "pixel_accuracy": float("nan"),
            },
        }
        ti = FakeTaskInstance()
        push_stage_xcom(summary, ti)
        self.assertEqual(ti.values["status"], "success_with_warning")
        self.assertFalse(ti.values["object_metrics_available"])
        self.assertEqual(ti.values["object_metrics_reason"], "validation vectorization is not implemented as a distinct stage yet")
        self.assertTrue(ti.values["pixel_metrics_available"])
        self.assertNotIn("metric_object_f1", ti.values)
        self.assertNotIn("metric_object_precision", ti.values)
        self.assertNotIn("metric_object_recall", ti.values)
        self.assertNotIn("counter_reference_objects", ti.values)
        self.assertNotIn("counter_predicted_objects", ti.values)
        self.assertNotIn("counter_tp_objects", ti.values)
        self.assertNotIn("counter_reference_scenes", ti.values)
        self.assertEqual(ti.values["metric_pixel_precision"], 0.1)
        self.assertEqual(ti.values["metric_pixel_recall"], 0.2)
        self.assertAlmostEqual(ti.values["metric_pixel_f1"], 0.13333333333333333)
        self.assertEqual(ti.values["metric_pixel_iou"], 0.07142857142857142)
        self.assertNotIn("metric_pixel_accuracy", ti.values)
        for value in ti.values.values():
            self.assertIsInstance(value, (str, int, float, bool))
            json.dumps(value, allow_nan=False)

    def test_stage_failure_writes_stage_report_before_raising(self) -> None:
        conf = {
            "experiment_id": "unit_stage_failure",
            "pseudolabel": {"run_on": "explicit_scene_list", "scene_list": ["missing.tif"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                run_stage("prepare_inference_scenes", conf, "manual__unit", Path(tmp))
            stage_path = Path(tmp) / "unit_stage_failure" / "stages" / "prepare_inference_scenes.json"
            report_path = Path(tmp) / "unit_stage_failure" / "stages" / "prepare_inference_scenes.report.md"
            self.assertTrue(stage_path.exists())
            self.assertTrue(report_path.exists())
            self.assertIn("stage_report", stage_path.read_text(encoding="utf-8"))
            self.assertIn("missing.tif", report_path.read_text(encoding="utf-8"))

    def test_validation_prediction_placeholder_is_not_reported_as_success(self) -> None:
        conf = {"experiment_id": "unit_real_placeholder", "train": {"enabled": True}}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "unit_real_placeholder"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "model.pt"
            checkpoint.write_bytes(b"checkpoint")
            (run_dir / "dataset_manifest.json").write_text(
                '{"val_scene_count": 2, "val_scenes": [{"name": "a.tif"}, {"name": "b.tif"}]}',
                encoding="utf-8",
            )
            (run_dir / "training_result.json").write_text(
                '{"checkpoint_path": "%s", "device": "cuda", "cuda_available": true, "gpu_name": "GPU"}' % str(checkpoint).replace("\\", "\\\\"),
                encoding="utf-8",
            )
            result = run_stage("predict_validation_scenes", conf, "manual__unit", Path(tmp))
            self.assertEqual(result["status"], "skipped")
            self.assertIn("Compatibility placeholder", result["summary"])
            self.assertEqual(result["counters"]["validation_scenes_processed"], 0)


if __name__ == "__main__":
    unittest.main()
