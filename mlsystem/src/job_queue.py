from __future__ import annotations
import os, shutil, socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml
from .io_utils import write_json
from .job_schema import JobSpec
from .pipeline_config import PipelineConfig, ensure_storage_layout

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_job_file(path: Path) -> JobSpec:
    with path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    return JobSpec.model_validate(payload)

def queue_dirs(config: PipelineConfig) -> dict[str, Path]:
    return {s: config.jobs_root / s for s in ("pending", "running", "done", "failed")}

def enqueue(config: PipelineConfig, source_yaml: Path) -> dict[str, Any]:
    ensure_storage_layout(config)
    job = load_job_file(source_yaml)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = config.jobs_root / "pending" / f"{job.job_id}__{stamp}.yml"
    shutil.copy2(source_yaml, dest)
    return {"queued": True, "job_id": job.job_id, "queue_file": str(dest)}

def list_queue(config: PipelineConfig) -> dict[str, Any]:
    ensure_storage_layout(config)
    result: dict[str, Any] = {"counts": {}, "jobs": {}}
    for state, directory in queue_dirs(config).items():
        if state == "pending":
            entries = sorted(directory.glob("*.yml"), key=lambda p: p.stat().st_mtime)
            rows = [{"name": p.name, "path": str(p)} for p in entries]
        else:
            entries = sorted([p for p in directory.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
            rows = [{"name": p.name, "path": str(p)} for p in entries]
        result["jobs"][state] = rows
        result["counts"][state] = len(rows)
    return result

def claim_next(config: PipelineConfig, executor_id: str | None = None) -> dict[str, Any] | None:
    ensure_storage_layout(config)
    for pending_file in sorted((config.jobs_root / "pending").glob("*.yml"), key=lambda p: p.stat().st_mtime):
        try:
            job = load_job_file(pending_file)
        except Exception as exc:
            failed_dir = config.jobs_root / "failed" / (pending_file.stem + "__invalid")
            failed_dir.mkdir(parents=True, exist_ok=False)
            pending_file.replace(failed_dir / "job.yml")
            write_json(failed_dir / "error.json", {"error": f"validation_failed: {exc}", "claimed_at": utc_now()})
            continue
        claim_id = pending_file.stem
        target_dir = config.jobs_root / "running" / claim_id
        try:
            target_dir.mkdir(parents=True, exist_ok=False)
            os.replace(pending_file, target_dir / "job.yml")
        except (FileExistsError, OSError):
            continue
        claim = {
            "claim_id": claim_id, "job_id": job.job_id, "task": job.task,
            "executor_id": executor_id or f"{socket.gethostname()}:{os.getpid()}",
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "claimed_at": utc_now(), "heartbeat_at": utc_now(),
        }
        write_json(target_dir / "claim.json", claim)
        return {"claim_id": claim_id, "job": job, "run_dir": target_dir, "claim": claim}
    return None

def heartbeat(run_dir: Path) -> None:
    import json
    claim_path = run_dir / "claim.json"
    payload = json.loads(claim_path.read_text(encoding="utf-8")) if claim_path.exists() else {}
    payload["heartbeat_at"] = utc_now()
    write_json(claim_path, payload)

def finish(config: PipelineConfig, running_job_dir: Path, success: bool) -> Path:
    state = "done" if success else "failed"
    dest = config.jobs_root / state / running_job_dir.name
    if dest.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dest = config.jobs_root / state / f"{running_job_dir.name}__{stamp}"
    running_job_dir.replace(dest)
    return dest
