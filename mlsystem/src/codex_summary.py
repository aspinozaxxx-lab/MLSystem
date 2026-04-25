from __future__ import annotations
from typing import Any
from .io_utils import read_json, write_json
from .job_queue import list_queue
from .pipeline_config import PipelineConfig

def _recent_state_jobs(config: PipelineConfig, state: str, limit: int) -> list[dict[str, Any]]:
    root = config.jobs_root / state
    if not root.exists():
        return []
    entries = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    rows = []
    for entry in entries:
        rows.append({"name": entry.name, "state": state, "claim": read_json(entry / "claim.json", None), "result": read_json(entry / "result.json", None), "error": read_json(entry / "error.json", None)})
    return rows

def build_codex_summary(
    config: PipelineConfig,
    resource_status: dict[str, Any] | None = None,
    current_mlflow: dict[str, Any] | None = None,
    current_job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resource_status = resource_status or read_json(config.system_root / "resource_status.json", default={}) or {}
    queue = list_queue(config)
    recent = []
    for state in ("running", "failed", "done"):
        recent.extend(_recent_state_jobs(config, state, config.recent_jobs_limit))
    disk = resource_status.get("disk", {}) if isinstance(resource_status, dict) else {}
    recommendations: list[str] = []
    if disk.get("free_gb", 999) < config.disk_warning_free_gb:
        recommendations.append("Mount or expand a dedicated data disk before creating large caches or copying TIFF files.")
    if (resource_status.get("gpu") or {}).get("available") is False:
        recommendations.append("Keep max_gpu_train_jobs=0 and run only lightweight/CPU jobs until a real GPU backend is available.")
    if queue.get("counts", {}).get("pending", 0) == 0:
        recommendations.append("Queue is empty; enqueue a YAML job when ready.")
    mlflow_status = resource_status.get("mlflow")
    if current_mlflow:
        mlflow_status = {
            "tracking_uri_internal": current_mlflow.get("tracking_uri_internal", config.mlflow_tracking_uri_internal),
            "tracking_uri_external": current_mlflow.get("tracking_uri_external", config.mlflow_tracking_uri_external),
            "experiment_name": current_mlflow.get("experiment_name"),
            "experiment_id": current_mlflow.get("experiment_id"),
            "run_id": current_mlflow.get("run_id"),
            "run_url_external": current_mlflow.get("run_url_external") or current_mlflow.get("external_run_url"),
            "run_url_internal": current_mlflow.get("run_url_internal") or current_mlflow.get("internal_run_url"),
            "status": "ok" if current_mlflow.get("ok") else "error",
        }
    payload = {
        "schema_version": 1,
        "queue": queue.get("counts", {}),
        "recent_jobs": recent[: config.recent_jobs_limit],
        "mlflow": mlflow_status,
        "s3": resource_status.get("s3"),
        "disk_warning": disk.get("free_gb", 999) < config.disk_warning_free_gb,
        "warnings": resource_status.get("warnings", []),
        "recommendations": recommendations,
    }
    if current_job:
        payload.update(
            {
                "job_id": current_job.get("job_id"),
                "claim_id": current_job.get("claim_id"),
                "task": current_job.get("task"),
                "status": current_job.get("status"),
                "server": current_job.get("server"),
                "metrics": current_job.get("metrics", {}),
                "errors": current_job.get("errors", []),
            }
        )
    write_json(config.system_root / "codex_summary.json", payload)
    return payload
