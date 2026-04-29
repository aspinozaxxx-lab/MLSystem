from __future__ import annotations
import os, shutil, socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml
from .io_utils import write_json
from .job_schema import JobSpec
from .pipeline_config import PipelineConfig, ensure_storage_layout
from .mlflow_adapter import MLFLOW_NOTE_TAG, build_run_note, compact_run_label, create_queued_job_run, trace_stage

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
    queue_position = len(list((config.jobs_root / "pending").glob("*.yml")))
    experiment_name = job.mlflow.experiment or (f"mlsystem-{job.class_name}" if job.class_name else "mlsystem-queue")
    model_cfg = (job.params.get("model") if isinstance(job.params.get("model"), dict) else {}) or {}
    model_name = model_cfg.get("name") or job.train.get("model_name")
    tile_size = job.preprocess.get("tile_size")
    stride = job.preprocess.get("stride")
    run_label = job.mlflow.tags.get("run_label") or compact_run_label(job.job_id, model_name, tile_size)
    run_note = build_run_note(
        job_id=job.job_id,
        run_label=run_label,
        model_name=model_name,
        tile_size=tile_size,
        stride=stride,
        data_uri=job.data.get("images_uri"),
        layout_uri=job.data.get("layout_uri"),
        warnings=["queued; training has not started yet"],
    )
    with trace_stage("enqueue_job", {"job_id": job.job_id, "task": job.task, "queue_position": queue_position}):
        mlflow_result = create_queued_job_run(
            config,
            experiment_name,
            run_label,
            params={
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
            },
            tags={
                "job_id": job.job_id,
                "task": job.task,
                "class_name": job.class_name or "",
                "priority": job.priority,
                "created_at": utc_now(),
                "run_label": run_label,
                "experiment_group": job.class_name or "",
                "model_name": model_name or "",
                "tile_size": tile_size or "",
                "stride": stride or "",
                "train_time_limit_sec": job.train.get("time_limit_sec") or "",
                "pseudolabel_enabled": str(bool((job.predict.get("pseudolabel") or {}).get("enabled"))).lower(),
                "postprocess_profile": "auto_tune" if job.postprocess.get("auto_tune") else "default",
                MLFLOW_NOTE_TAG: run_note,
                **job.mlflow.tags,
            },
            artifacts=[dest],
            queue_position=queue_position,
        )
    metadata = {
        "job_id": job.job_id,
        "task": job.task,
        "class_name": job.class_name,
        "queue_state": "pending",
        "created_at": utc_now(),
        "queue_file": str(dest),
        "mlflow": mlflow_result,
    }
    # Write metadata before exposing the YAML to the executor. This prevents a
    # fast polling loop from claiming the job before its MLflow run_id is known.
    write_json(dest.with_suffix(".json"), metadata)
    shutil.copy2(source_yaml, dest)
    return {"queued": True, "job_id": job.job_id, "queue_file": str(dest), "mlflow": mlflow_result}

def list_queue(config: PipelineConfig) -> dict[str, Any]:
    ensure_storage_layout(config)
    result: dict[str, Any] = {"counts": {}, "jobs": {}}
    for state, directory in queue_dirs(config).items():
        if state == "pending":
            entries = sorted(directory.glob("*.yml"), key=lambda p: p.stat().st_mtime)
            rows = []
            for p in entries:
                meta_path = p.with_suffix(".json")
                row = {"name": p.name, "path": str(p)}
                if meta_path.exists():
                    import json
                    row["metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
                rows.append(row)
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
            meta_path = pending_file.with_suffix(".json")
            if meta_path.exists():
                os.replace(meta_path, target_dir / "queue_metadata.json")
        except (FileExistsError, OSError):
            continue
        queue_metadata = {}
        queue_metadata_path = target_dir / "queue_metadata.json"
        if queue_metadata_path.exists():
            import json
            queue_metadata = json.loads(queue_metadata_path.read_text(encoding="utf-8"))
        claim = {
            "claim_id": claim_id, "job_id": job.job_id, "task": job.task,
            "executor_id": executor_id or f"{socket.gethostname()}:{os.getpid()}",
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "claimed_at": utc_now(), "heartbeat_at": utc_now(),
            "mlflow": queue_metadata.get("mlflow"),
        }
        write_json(target_dir / "claim.json", claim)
        return {"claim_id": claim_id, "job": job, "run_dir": target_dir, "claim": claim, "queue_metadata": queue_metadata}
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
