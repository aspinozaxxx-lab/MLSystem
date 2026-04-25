from __future__ import annotations
import shutil, time
from pathlib import Path
from typing import Any
from .codex_summary import build_codex_summary
from .io_utils import append_text, write_json
from .job_queue import claim_next, finish, heartbeat, utc_now
from .mlflow_adapter import log_lightweight_run
from .pipeline_config import PipelineConfig, ensure_storage_layout
from .resource_manager import collect_status

IMPLEMENTED_TASKS = {"noop", "status_check", "inventory"}

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

def _execute(config: PipelineConfig, task: str) -> dict[str, Any]:
    if task == "noop":
        return {"status": "done", "message": "noop completed"}
    if task == "status_check":
        status = collect_status(config, include_services=True)
        summary = build_codex_summary(config, status)
        return {"status": "done", "resource_status_path": str(config.system_root / "resource_status.json"), "codex_summary_path": str(config.system_root / "codex_summary.json"), "queue": summary.get("queue")}
    if task == "inventory":
        return {"status": "done", **_inventory(config)}
    return {"status": "not_implemented", "message": f"Task {task} is validated but not implemented in MVP executor"}

def run_once(config: PipelineConfig) -> dict[str, Any]:
    ensure_storage_layout(config)
    claimed = claim_next(config)
    if not claimed:
        status = collect_status(config, include_services=True)
        summary = build_codex_summary(config, status)
        return {"claimed": False, "message": "No pending jobs", "summary": summary}
    running_dir: Path = claimed["run_dir"]
    job = claimed["job"]
    claim_id = claimed["claim_id"]
    job_log = running_dir / "job.log"
    experiment_dir = config.experiments_root / claim_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(running_dir / "job.yml", experiment_dir / "job.yml")
    _log(job_log, f"claimed job_id={job.job_id} task={job.task}")
    try:
        heartbeat(running_dir)
        started = time.time()
        result = _execute(config, job.task)
        duration = round(time.time() - started, 3)
        run_summary = {"schema_version": 1, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "status": result.get("status", "done"), "implemented": job.task in IMPLEMENTED_TASKS, "started_at": claimed["claim"].get("claimed_at"), "finished_at": utc_now(), "duration_sec": duration, "result": result, "local_experiment_dir": str(experiment_dir)}
        write_json(experiment_dir / "run_summary.json", run_summary)
        write_json(running_dir / "result.json", result)
        status = collect_status(config, include_services=True)
        codex = build_codex_summary(config, status)
        write_json(experiment_dir / "codex_summary.json", codex)
        mlflow_result = log_lightweight_run(config, job.mlflow.experiment or config.mlflow_experiment, f"{job.task}:{job.job_id}", params={"job_id": job.job_id, "task": job.task, "class_name": job.class_name, **job.params}, metrics={"duration_sec": duration}, artifacts=[experiment_dir / "run_summary.json", experiment_dir / "codex_summary.json", experiment_dir / "job.yml"], tags={"claim_id": claim_id, **job.mlflow.tags})
        run_summary["mlflow"] = mlflow_result
        write_json(experiment_dir / "run_summary.json", run_summary)
        write_json(running_dir / "result.json", {**result, "mlflow": mlflow_result})
        _log(job_log, f"completed status={run_summary['status']} mlflow_ok={mlflow_result.get('ok')}")
        final_dir = finish(config, running_dir, success=True)
        return {"claimed": True, "success": True, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "final_dir": str(final_dir), "result": result, "mlflow": mlflow_result}
    except Exception as exc:
        error = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "job_id": job.job_id, "task": job.task, "failed_at": utc_now()}
        write_json(running_dir / "error.json", error)
        _log(job_log, f"failed error={error['error']}")
        final_dir = finish(config, running_dir, success=False)
        collect_status(config, include_services=True)
        build_codex_summary(config)
        return {"claimed": True, "success": False, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "final_dir": str(final_dir), "error": error}
