from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

from ..io_utils import write_json
from ..job_schema import JobSpec
from ..mlflow_adapter.api import MLflowJobRun


def write_history(experiment_dir: Path, history: list[dict[str, float]]) -> list[Path]:
    json_path = experiment_dir / "history.json"
    csv_path = experiment_dir / "history.csv"
    json_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if history:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    return [json_path, csv_path]


def write_epoch_table(experiment_dir: Path, history: list[dict[str, float]]) -> Path:
    table_path = experiment_dir / "epoch_metrics_table.json"
    rows: list[dict[str, float]] = []
    key_map = {
        "epoch": "epoch",
        "train/loss": "train_loss",
        "train/dice": "train_dice",
        "train/iou": "train_iou",
        "val/loss": "val_loss",
        "val/dice": "val_dice",
        "val/iou": "val_iou",
        "val/precision": "val_precision",
        "val/recall": "val_recall",
        "val/pixel_f1": "val_pixel_f1",
        "learning_rate": "learning_rate",
        "epoch_duration_sec": "epoch_duration_sec",
    }
    for item in history:
        rows.append({out_key: item.get(in_key) for in_key, out_key in key_map.items()})
    write_json(table_path, {"schema_version": 1, "rows": rows})
    return table_path


def run_smoke_train(job: JobSpec, experiment_dir: Path, mlflow_run: MLflowJobRun, job_log: Path, log_fn: Any) -> dict[str, Any]:
    epochs = int(job.train.get("epochs") or job.params.get("epochs") or 3)
    epochs = max(1, min(epochs, 3))
    sleep_sec = float(job.train.get("epoch_sleep_sec") or job.params.get("epoch_sleep_sec") or 0.2)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        time.sleep(max(0.0, min(sleep_sec, 2.0)))
        progress = epoch / epochs
        metrics = {
            "train/loss": round(0.9 - 0.22 * progress, 6),
            "train/dice": round(0.45 + 0.18 * progress, 6),
            "train/iou": round(0.32 + 0.16 * progress, 6),
            "val/loss": round(1.0 - 0.18 * progress, 6),
            "val/dice": round(0.4 + 0.15 * progress, 6),
            "val/iou": round(0.28 + 0.13 * progress, 6),
            "val/precision": round(0.5 + 0.12 * progress, 6),
            "val/recall": round(0.46 + 0.10 * progress, 6),
            "val/pixel_f1": round(0.48 + 0.11 * progress, 6),
            "learning_rate": round(0.001 * (1.0 - 0.2 * (epoch - 1)), 8),
            "epoch_duration_sec": round(time.time() - started, 6),
        }
        mlflow_run.log_metrics(metrics, step=epoch)
        history.append({"epoch": float(epoch), **metrics})
        log_fn(job_log, f"smoke_train epoch={epoch} metrics_logged=true")
    artifacts = write_history(experiment_dir, history)
    epoch_table_path = write_epoch_table(experiment_dir, history)
    mlflow_run.log_table({"rows": history}, "epoch_metrics_table_mlflow.json")
    artifacts.append(epoch_table_path)
    mlflow_run.log_artifacts(artifacts)
    return {
        "status": "done",
        "mode": "smoke_train",
        "epochs": epochs,
        "history_path": str(experiment_dir / "history.json"),
        "history_csv_path": str(experiment_dir / "history.csv"),
        "epoch_table_path": str(epoch_table_path),
        "last_epoch_metrics": history[-1] if history else {},
    }
