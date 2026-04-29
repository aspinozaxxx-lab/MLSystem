from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..debug.smoke_train import write_epoch_table
from ..job_schema import JobSpec
from ..mlflow_adapter import MLFLOW_NOTE_TAG, MLflowJobRun, build_run_note
from ..storage.local_io import write_json


def log_epoch_table_if_present(
    *,
    result: dict[str, Any],
    experiment_dir: Path,
    mlflow_run: MLflowJobRun,
    log_fn: Any,
    job_log: Path,
) -> None:
    history_path_value = result.get("history_path")
    history_path = Path(str(history_path_value)) if history_path_value else None
    if not history_path or not history_path.exists():
        return
    try:
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(history_payload, list):
            epoch_table_path = write_epoch_table(experiment_dir, history_payload)
            mlflow_run.log_table({"rows": history_payload}, "epoch_metrics_table_mlflow.json")
            mlflow_run.log_artifacts([epoch_table_path])
            result.setdefault("mlflow_artifacts", {})["epoch_metrics_table"] = "epoch_metrics_table.json"
    except Exception as exc:
        log_fn(job_log, f"epoch_table_log_failed={type(exc).__name__}: {exc}")


def write_pipeline_timeline(
    *,
    experiment_dir: Path,
    job: JobSpec,
    claim_id: str,
    queued_at: str | None,
    started_at: str | None,
    finished_at: str,
    duration_sec: float,
    result: dict[str, Any],
) -> Path:
    timeline_path = experiment_dir / "pipeline_timeline.json"
    write_json(
        timeline_path,
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "claim_id": claim_id,
            "queued_at": queued_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "task": job.task,
            "mode": result.get("mode"),
            "timing": result.get("timing") or {},
            "events": [
                {"name": "queue", "status": "done"},
                {"name": "prepare", "status": "done" if result.get("mode") == "real_train" else "skipped"},
                {"name": "train", "status": result.get("status")},
                {"name": "pseudolabel", "status": (result.get("pseudolabel") or {}).get("status")},
                {"name": "postprocess", "status": "done" if result.get("postprocess_metrics") else "skipped"},
            ],
        },
    )
    return timeline_path


def write_eval_threshold_table(experiment_dir: Path, result: dict[str, Any]) -> Path:
    eval_table_path = experiment_dir / "eval_threshold_table.json"
    post = result.get("postprocess_metrics") or {}
    write_json(
        eval_table_path,
        {
            "schema_version": 1,
            "rows": [
                {
                    "threshold": post.get("threshold_used"),
                    "min_object_area_m2": post.get("min_object_area_m2_used"),
                    "simplify_tolerance_m": post.get("simplify_tolerance_m_used"),
                    "accepted_objects": post.get("accepted_objects"),
                    "accepted_geojson_mb": post.get("accepted_geojson_mb"),
                    "total_area_m2": post.get("total_area_m2"),
                    "total_vertices": post.get("total_vertices"),
                }
            ]
            if post
            else [],
        },
    )
    return eval_table_path


def build_final_metrics(result: dict[str, Any]) -> dict[str, Any]:
    final_metrics: dict[str, Any] = {}
    final_metrics.update(result.get("last_epoch_metrics") or {})
    final_metrics.update(result.get("postprocess_metrics") or {})
    if result.get("best_val_iou") is not None:
        final_metrics["best_val_iou"] = result.get("best_val_iou")
    return final_metrics


def update_final_run_note(
    *,
    mlflow_run: MLflowJobRun,
    job: JobSpec,
    run_name: str,
    model_name: str | None,
    tile_size: Any,
    stride: Any,
    result: dict[str, Any],
) -> None:
    final_note = build_run_note(
        job_id=job.job_id,
        run_label=run_name,
        model_name=result.get("model_name") or model_name,
        tile_size=tile_size,
        stride=stride,
        data_uri=job.data.get("images_uri"),
        layout_uri=job.data.get("layout_uri"),
        metrics=build_final_metrics(result),
        counts={
            "train_scene_count": result.get("train_scene_count"),
            "val_scene_count": result.get("val_scene_count"),
            "test_scene_count": result.get("test_scene_count"),
            "positive_scene_count": result.get("positive_scene_count"),
            "negative_scene_count": result.get("negative_scene_count"),
        },
        artifacts={
            "history": "history.csv / history.json / epoch_metrics_table.json",
            "timeline": "pipeline_timeline.json",
            "evaluation": "eval_threshold_table.json",
            "summary": "run_summary.json / codex_summary.json",
            "pseudolabel": f"{job.job_id}.accepted.geojson",
            "object_metrics": "object_metrics.json / object_matches.csv",
            "scene_lists": "train_scenes.txt / pseudolabel_scenes.txt",
            "prediction_examples": "prediction_examples.html",
        },
        warnings=result.get("warnings") or [],
    )
    mlflow_run.set_tags({MLFLOW_NOTE_TAG: final_note})
