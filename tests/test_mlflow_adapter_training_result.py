from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from mlsystem.src.mlflow_adapter import _client
from mlsystem.src.train.contracts import CheckpointArtifact, EpochMetrics, TrainResult


class MLflowAdapterTrainingResultTests(unittest.TestCase):
    def test_log_training_result_to_run_logs_only_allowed_ui_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "model.pt"
            checkpoint_path.write_text("checkpoint", encoding="utf-8")
            (Path(tmp) / "history.json").write_text("[]\n", encoding="utf-8")
            (Path(tmp) / "threshold_sweep_summary.json").write_text("{}\n", encoding="utf-8")
            result = TrainResult(
                status="done",
                model_name="tiny_unet_4ch",
                device="cpu",
                cuda_available=False,
                epochs_completed=2,
                best_epoch=1,
                best_val_pixel_f1=0.72,
                best_val_iou=0.58,
                last_val_pixel_f1=0.64,
                training_time_sec=12.0,
                checkpoint=CheckpointArtifact(
                    path=str(checkpoint_path),
                    model_name="tiny_unet_4ch",
                    epoch=1,
                    metric_name="val/pixel_f1",
                    metric_value=0.72,
                ),
                history=[
                    EpochMetrics(
                        epoch=1,
                        metrics={
                            "val/pixel_f1": 0.72,
                            "val/pixel_f1_at_threshold_0_8": 0.81,
                            "val/pixel_f1_best_threshold": 0.83,
                            "epoch_duration_sec": 5.0,
                        },
                    ),
                    EpochMetrics(epoch=2, metrics={"val/pixel_f1": 0.64, "epoch_duration_sec": 7.0}),
                ],
            )

            with (
                patch.object(_client, "log_dataset_input_to_run") as dataset_input,
                patch.object(_client, "log_metrics_to_run") as metrics,
                patch.object(_client, "log_params_to_run") as params,
                patch.object(_client, "log_artifacts_to_run", return_value=[]) as artifacts,
            ):
                errors = _client.log_training_result_to_run(
                    types.SimpleNamespace(),
                    "mlflow-run-id",
                    result,
                    dataset_identity=types.SimpleNamespace(name="dataset"),
                )

            self.assertEqual(errors, [])
            dataset_input.assert_called_once()
            params.assert_called_once()
            artifacts.assert_called_once()
            metrics.assert_called_once()
            logged_metrics = metrics.call_args.args[2]
            self.assertEqual(set(logged_metrics), {"f1_pixel", "epochs_total", "epoch_time_sec", "training_time_sec"})
            self.assertEqual(logged_metrics["f1_pixel"], 0.72)
            self.assertNotIn("val/pixel_f1_at_threshold_0_8", logged_metrics)
            self.assertNotIn("val/pixel_f1_best_threshold", logged_metrics)

    def test_log_metrics_to_run_filters_training_excluded_prefixes(self) -> None:
        calls: list[tuple[dict[str, float], int | None]] = []

        class RunContext:
            def __enter__(self) -> object:
                return self

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                return None

        def start_run(run_id: str, **kwargs: object) -> RunContext:
            return RunContext()

        fake_mlflow = types.SimpleNamespace(
            set_tracking_uri=lambda uri: None,
            start_run=start_run,
            log_metrics=lambda payload, step=None: calls.append((dict(payload), step)),
        )
        metrics = {
            "f1_pixel": 0.7,
            "epochs_total": 3,
            "val/pixel_f1": 0.8,
            "train/loss_total": 0.1,
            "diagnostics/foo": 1.0,
            "system/cpu": 2.0,
            "resource/ram": 3.0,
            "resources/gpu": 4.0,
            "val/pixel_f1_at_threshold_0_8": 0.9,
            "val/pixel_f1_best_threshold": 0.91,
            "pixel_tp": 10,
            "debug/microtiming": 5.0,
        }
        config = types.SimpleNamespace(mlflow_tracking_uri_internal="http://mlflow")

        with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
            _client.log_metrics_to_run(config, "run-id", metrics, step=3)

        self.assertEqual(calls, [({"f1_pixel": 0.7, "epochs_total": 3.0}, 3)])


if __name__ == "__main__":
    unittest.main()
