from __future__ import annotations
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any
from .codex_summary import build_codex_summary
from .io_utils import append_text, write_json
from .job_queue import claim_next, finish, heartbeat, utc_now
from .job_schema import JobSpec
from .mlflow_adapter import MLflowJobRun, start_job_run
from .pipeline_config import PipelineConfig, ensure_storage_layout
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
    return {"inventory_roots": roots, "max_inventory_files": config.max_inventory_files}

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
    mlflow_run.log_artifacts(artifacts)
    return {
        "status": "done",
        "mode": "smoke_train",
        "epochs": epochs,
        "history_path": str(experiment_dir / "history.json"),
        "history_csv_path": str(experiment_dir / "history.csv"),
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
        return {"status": "not_implemented", "message": "Full train is not implemented in MVP executor; use train.smoke=true for MLflow smoke checks"}
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
        run_name = f"{job.task}:{job.job_id}"
        with start_job_run(
            config,
            experiment_name,
            run_name,
            params=_job_params(job),
            tags={"claim_id": claim_id, **job.mlflow.tags},
        ) as mlflow_run:
            result = _execute(config, job, experiment_dir, mlflow_run, job_log)
            duration = round(time.time() - started, 3)
            mlflow_run.log_metrics({"duration_sec": duration})
            mlflow_result = mlflow_run.result()
            run_summary = {
                "schema_version": 1,
                "claim_id": claim_id,
                "job_id": job.job_id,
                "task": job.task,
                "status": result.get("status", "done"),
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
            codex = build_codex_summary(config, status, current_mlflow=mlflow_result)
            write_json(experiment_dir / "codex_summary.json", codex)
            mlflow_run.log_artifacts([experiment_dir / "run_summary.json", experiment_dir / "codex_summary.json", experiment_dir / "job.yml"])
        _log(job_log, f"completed status={run_summary['status']} mlflow_ok={mlflow_result.get('ok')}")
        final_dir = finish(config, running_dir, success=True)
        return {"claimed": True, "success": True, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "final_dir": str(final_dir), "result": result, "mlflow": mlflow_result}
    except Exception as exc:
        error = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "job_id": job.job_id, "task": job.task, "failed_at": utc_now(), "mlflow": mlflow_result}
        write_json(running_dir / "error.json", error)
        _log(job_log, f"failed error={error['error']}")
        final_dir = finish(config, running_dir, success=False)
        collect_status(config, include_services=True)
        build_codex_summary(config)
        return {"claimed": True, "success": False, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "final_dir": str(final_dir), "error": error}
