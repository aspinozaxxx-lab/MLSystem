from __future__ import annotations
import csv
import json
import shutil
import socket
import time
from pathlib import Path
from typing import Any
from .codex_summary import build_codex_summary
from .io_utils import append_text, write_json
from .job_queue import claim_next, finish, heartbeat, utc_now
from .job_schema import JobSpec
from .mlflow_adapter import MLFLOW_NOTE_TAG, MLflowJobRun, build_run_note, compact_run_label, set_run_tags, start_job_run
from .pipeline_config import PipelineConfig, ensure_storage_layout
from .preprocess_inventory import build_preprocess_inventory
from .real_train import run_real_train
from .resource_manager import collect_status

IMPLEMENTED_TASKS = {"noop", "status_check", "inventory", "train"}

def _log(path: Path, message: str) -> None:
    append_text(path, f"{utc_now()} {message}\n")

def _inventory(config: PipelineConfig) -> dict[str, Any]:
    roots = []
    for root in config.known_data_roots:
        item: dict[str, Any] = {"path": str(root), "exists": root.exists(), "files": 0, "size_bytes_sampled": 0, "by_extension": {}}
        if root.exists():
            count = 0
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                count += 1
                if count > config.max_inventory_files:
                    item["truncated"] = True
                    break
                suffix = path.suffix.lower() or "<none>"
                item["by_extension"][suffix] = item["by_extension"].get(suffix, 0) + 1
                try:
                    item["size_bytes_sampled"] += path.stat().st_size
                except OSError:
                    pass
            item["files"] = min(count, config.max_inventory_files)
        roots.append(item)
    s3_inventory = build_preprocess_inventory(config, dry_run=True)
    return {"inventory_roots": roots, "s3_inventory": s3_inventory, "max_inventory_files": config.max_inventory_files}

def _is_smoke_train(job: JobSpec) -> bool:
    return bool(job.params.get("smoke") or job.params.get("smoke_train") or job.train.get("smoke"))

def _write_history(experiment_dir: Path, history: list[dict[str, float]]) -> list[Path]:
    json_path = experiment_dir / "history.json"
    csv_path = experiment_dir / "history.csv"
    json_path.write_text(json.dumps(history, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if history:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    return [json_path, csv_path]

def _write_epoch_table(experiment_dir: Path, history: list[dict[str, float]]) -> Path:
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
        "val/f1": "val_f1",
        "learning_rate": "learning_rate",
        "epoch_duration_sec": "epoch_duration_sec",
    }
    for item in history:
        rows.append({out_key: item.get(in_key) for in_key, out_key in key_map.items()})
    write_json(table_path, {"schema_version": 1, "rows": rows})
    return table_path

def _smoke_train(job: JobSpec, experiment_dir: Path, mlflow_run: MLflowJobRun, job_log: Path) -> dict[str, Any]:
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
            "val/f1": round(0.48 + 0.11 * progress, 6),
            "learning_rate": round(0.001 * (1.0 - 0.2 * (epoch - 1)), 8),
            "epoch_duration_sec": round(time.time() - started, 6),
        }
        mlflow_run.log_metrics(metrics, step=epoch)
        history.append({"epoch": float(epoch), **metrics})
        _log(job_log, f"smoke_train epoch={epoch} metrics_logged=true")
    artifacts = _write_history(experiment_dir, history)
    epoch_table_path = _write_epoch_table(experiment_dir, history)
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

def _execute(config: PipelineConfig, job: JobSpec, experiment_dir: Path, mlflow_run: MLflowJobRun, job_log: Path) -> dict[str, Any]:
    if job.task == "noop":
        return {"status": "done", "message": "noop completed"}
    if job.task == "status_check":
        status = collect_status(config, include_services=True)
        summary = build_codex_summary(config, status)
        return {"status": "done", "resource_status_path": str(config.system_root / "resource_status.json"), "codex_summary_path": str(config.system_root / "codex_summary.json"), "queue": summary.get("queue")}
    if job.task == "inventory":
        return {"status": "done", **_inventory(config)}
    if job.task == "train" and _is_smoke_train(job):
        return _smoke_train(job, experiment_dir, mlflow_run, job_log)
    if job.task == "train":
        return run_real_train(config, job, experiment_dir, mlflow_run, job_log, _log)
    return {"status": "not_implemented", "message": f"Task {job.task} is validated but not implemented in MVP executor"}

def _job_params(job: JobSpec) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "task": job.task,
        "class_name": job.class_name,
        "priority": job.priority,
        "description": job.description,
        "params": job.params,
        "data": job.data,
        "preprocess": job.preprocess,
        "train": job.train,
        "predict": job.predict,
        "postprocess": job.postprocess,
        "resources": job.resources.model_dump(),
    }

def run_once(config: PipelineConfig) -> dict[str, Any]:
    ensure_storage_layout(config)
    claimed = claim_next(config)
    if not claimed:
        status = collect_status(config, include_services=True)
        summary = build_codex_summary(config, status)
        return {"claimed": False, "message": "No pending jobs", "summary": summary}
    running_dir: Path = claimed["run_dir"]
    job: JobSpec = claimed["job"]
    claim_id = claimed["claim_id"]
    job_log = running_dir / "job.log"
    experiment_dir = config.experiments_root / claim_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(running_dir / "job.yml", experiment_dir / "job.yml")
    _log(job_log, f"claimed job_id={job.job_id} task={job.task}")
    mlflow_result: dict[str, Any] = {}
    try:
        heartbeat(running_dir)
        started = time.time()
        experiment_name = job.mlflow.experiment or config.mlflow_default_experiment
        model_cfg = (job.params.get("model") if isinstance(job.params.get("model"), dict) else {}) or {}
        model_name = model_cfg.get("name") or job.train.get("model_name")
        tile_size = job.preprocess.get("tile_size")
        stride = job.preprocess.get("stride")
        run_name = job.mlflow.tags.get("run_label") or compact_run_label(job.job_id, model_name, tile_size)
        existing_run_id = ((claimed.get("queue_metadata") or {}).get("mlflow") or {}).get("run_id")
        if existing_run_id:
            set_run_tags(
                config,
                existing_run_id,
                {
                    "job_status": "running",
                    "queue_state": "running",
                    "started_at": utc_now(),
                    "claim_id": claim_id,
                },
            )
        initial_note = build_run_note(
            job_id=job.job_id,
            run_label=run_name,
            model_name=model_name,
            tile_size=tile_size,
            stride=stride,
            data_uri=job.data.get("images_uri"),
            layout_uri=job.data.get("layout_uri"),
            warnings=["running"],
        )
        with start_job_run(
            config,
            experiment_name,
            run_name,
            params=_job_params(job),
            tags={
                "job_id": job.job_id,
                "task": job.task,
                "class_name": job.class_name or "",
                "job_status": "running",
                "queue_state": "running",
                "claim_id": claim_id,
                "run_label": run_name,
                "experiment_group": job.class_name or "",
                "model_name": model_name or "",
                "tile_size": tile_size or "",
                "stride": stride or "",
                "train_time_limit_sec": job.train.get("time_limit_sec") or "",
                "pseudolabel_enabled": str(bool((job.predict.get("pseudolabel") or {}).get("enabled"))).lower(),
                "postprocess_profile": "auto_tune" if job.postprocess.get("auto_tune") else "default",
                MLFLOW_NOTE_TAG: initial_note,
                **job.mlflow.tags,
            },
            run_id=existing_run_id,
        ) as mlflow_run:
            result = _execute(config, job, experiment_dir, mlflow_run, job_log)
            history_path_value = result.get("history_path")
            history_path = Path(str(history_path_value)) if history_path_value else None
            if history_path and history_path.exists():
                try:
                    history_payload = json.loads(history_path.read_text(encoding="utf-8"))
                    if isinstance(history_payload, list):
                        epoch_table_path = _write_epoch_table(experiment_dir, history_payload)
                        mlflow_run.log_table({"rows": history_payload}, "epoch_metrics_table_mlflow.json")
                        mlflow_run.log_artifacts([epoch_table_path])
                        result.setdefault("mlflow_artifacts", {})["epoch_metrics_table"] = "epoch_metrics_table.json"
                except Exception as exc:
                    _log(job_log, f"epoch_table_log_failed={type(exc).__name__}: {exc}")
            duration = round(time.time() - started, 3)
            mlflow_run.log_metrics({"duration_sec": duration})
            timeline_path = experiment_dir / "pipeline_timeline.json"
            write_json(
                timeline_path,
                {
                    "schema_version": 1,
                    "job_id": job.job_id,
                    "claim_id": claim_id,
                    "queued_at": (claimed.get("queue_metadata") or {}).get("created_at"),
                    "started_at": claimed["claim"].get("claimed_at"),
                    "finished_at": utc_now(),
                    "duration_sec": duration,
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
                    ] if post else [],
                },
            )
            mlflow_run.log_artifacts([timeline_path, eval_table_path])
            mlflow_result = mlflow_run.result()
            run_summary_status = result.get("status", "done")
            mlflow_run.set_tags({"job_status": run_summary_status, "queue_state": run_summary_status})
            final_metrics = {}
            final_metrics.update(result.get("last_epoch_metrics") or {})
            final_metrics.update(result.get("postprocess_metrics") or {})
            if result.get("best_val_iou") is not None:
                final_metrics["best_val_iou"] = result.get("best_val_iou")
            final_note = build_run_note(
                job_id=job.job_id,
                run_label=run_name,
                model_name=result.get("model_name") or model_name,
                tile_size=tile_size,
                stride=stride,
                data_uri=job.data.get("images_uri"),
                layout_uri=job.data.get("layout_uri"),
                metrics=final_metrics,
                artifacts={
                    "history": "history.csv / history.json / epoch_metrics_table.json",
                    "timeline": "pipeline_timeline.json",
                    "evaluation": "eval_threshold_table.json",
                    "summary": "run_summary.json / codex_summary.json",
                    "pseudolabel": "accepted.geojson / accepted.geojson.gz / accepted.gpkg",
                },
                warnings=result.get("warnings") or [],
            )
            mlflow_run.set_tags({MLFLOW_NOTE_TAG: final_note})
            run_summary = {
                "schema_version": 1,
                "claim_id": claim_id,
                "job_id": job.job_id,
                "task": job.task,
                "status": run_summary_status,
                "implemented": job.task in IMPLEMENTED_TASKS,
                "started_at": claimed["claim"].get("claimed_at"),
                "finished_at": utc_now(),
                "duration_sec": duration,
                "result": result,
                "mlflow": mlflow_result,
                "local_experiment_dir": str(experiment_dir),
            }
            write_json(experiment_dir / "run_summary.json", run_summary)
            write_json(running_dir / "result.json", {**result, "mlflow": mlflow_result})
            status = collect_status(config, include_services=True)
            codex = build_codex_summary(
                config,
                status,
                current_mlflow=mlflow_result,
                current_job={
                    "claim_id": claim_id,
                    "job_id": job.job_id,
                    "task": job.task,
                    "status": run_summary["status"],
                    "server": socket.gethostname(),
                    "metrics": result.get("last_epoch_metrics", {}),
                    "errors": [],
                },
            )
            write_json(experiment_dir / "codex_summary.json", codex)
            mlflow_run.log_artifacts([experiment_dir / "run_summary.json", experiment_dir / "codex_summary.json", experiment_dir / "job.yml"])
        _log(job_log, f"completed status={run_summary['status']} mlflow_ok={mlflow_result.get('ok')}")
        final_dir = finish(config, running_dir, success=True)
        return {"claimed": True, "success": True, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "final_dir": str(final_dir), "result": result, "mlflow": mlflow_result}
    except Exception as exc:
        error = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "job_id": job.job_id, "task": job.task, "failed_at": utc_now(), "mlflow": mlflow_result}
        existing_run_id = ((claimed.get("queue_metadata") or {}).get("mlflow") or {}).get("run_id")
        if existing_run_id:
            set_run_tags(config, existing_run_id, {"job_status": "failed", "queue_state": "failed", "error": error["error"]})
        write_json(running_dir / "error.json", error)
        _log(job_log, f"failed error={error['error']}")
        final_dir = finish(config, running_dir, success=False)
        collect_status(config, include_services=True)
        build_codex_summary(config)
        return {"claimed": True, "success": False, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "final_dir": str(final_dir), "error": error}
