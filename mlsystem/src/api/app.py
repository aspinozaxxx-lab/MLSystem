from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Response
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("FastAPI is required for mlsystem-api. Install fastapi and uvicorn.") from exc

from ..pipeline.stages.registry import known_stages
from . import __version__
from .job_runner import JobRunner
from .job_store import JobStore
from .models import JobStatusResponse, StageStartRequest, StageStartResponse
from .security import mask_text, masked_env_snapshot, verify_token_header
from .stage_routes import debug_run_stage_sync, job_status, run_summary, stages_payload, start_stage


def require_api_token(authorization: str | None = Header(default=None)) -> None:
    if not verify_token_header(authorization):
        raise HTTPException(status_code=401, detail="Invalid API token")


app = FastAPI(title="MLSystem API", version=__version__)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "mlsystem-api",
        "version": __version__,
        "commit": os.getenv("GIT_COMMIT") or os.getenv("MLSYSTEM_COMMIT") or "unknown",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    status_root = Path(os.getenv("MLSYSTEM_AIRFLOW_STATE_DIR", "/data/mlsystem/airflow/status"))
    job_root = Path(os.getenv("MLSYSTEM_API_JOB_ROOT", "/data/mlsystem/api/jobs"))
    try:
        status_root.mkdir(parents=True, exist_ok=True)
        checks["status_root"] = {"status": "ok", "path": str(status_root)}
    except Exception as exc:  # noqa: BLE001
        checks["status_root"] = {"status": "failed", "message": mask_text(str(exc))}
    try:
        job_root.mkdir(parents=True, exist_ok=True)
        checks["job_root"] = {"status": "ok", "path": str(job_root)}
    except Exception as exc:  # noqa: BLE001
        checks["job_root"] = {"status": "failed", "message": mask_text(str(exc))}
    try:
        checks["stage_registry"] = {"status": "ok", "stages": known_stages()}
    except Exception as exc:  # noqa: BLE001
        checks["stage_registry"] = {"status": "failed", "message": mask_text(str(exc))}
    env = masked_env_snapshot()
    checks["env"] = {"status": "ok", "keys": sorted(env)}
    required_env = [
        "MLSYSTEM_API_JOB_ROOT",
        "MLSYSTEM_API_TOKEN",
        "MLSYSTEM_AIRFLOW_STATE_DIR",
        "MLSYSTEM_AIRFLOW_EXECUTION_MODE",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLSYSTEM_TRITON_URL",
        "INFERENCE_ENGINE_API_URL",
    ]
    missing_env = [key for key in required_env if not os.getenv(key)]
    checks["required_env"] = {"status": "failed" if missing_env else "ok", "missing": missing_env}
    status = "ok" if all(item.get("status") == "ok" for item in checks.values()) else "degraded"
    if status != "ok":
        response.status_code = 503
    return {"status": status, "service": "mlsystem-api", "checks": checks}


@app.get("/api/v1/stages")
def list_stages() -> dict[str, Any]:
    return stages_payload()


@app.post("/api/v1/runs/{run_id}/stages/{stage_name}/start", response_model=StageStartResponse, dependencies=[Depends(require_api_token)])
def start_stage_endpoint(run_id: str, stage_name: str, request: StageStartRequest) -> StageStartResponse:
    try:
        return start_stage(run_id, stage_name, request, JobRunner(JobStore()))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_api_token)])
def job_status_endpoint(job_id: str) -> JobStatusResponse:
    try:
        return job_status(job_id, JobRunner(JobStore()))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc


@app.get("/api/v1/runs/{run_id}/summary")
def run_summary_endpoint(run_id: str) -> dict[str, Any]:
    return run_summary(run_id, os.getenv("MLSYSTEM_AIRFLOW_STATE_DIR", "/data/mlsystem/airflow/status"))


if str(os.getenv("MLSYSTEM_DEBUG_DATASET_ENDPOINTS") or "").lower() in {"1", "true", "yes", "on"}:

    @app.post("/api/debug/virtual-dataset/preview")
    def debug_virtual_dataset_preview_endpoint(request: dict[str, Any]) -> dict[str, Any]:
        dataset_manifest = request.get("dataset_manifest")
        annotation = request.get("annotation")
        images_dir = request.get("images_dir")
        if not dataset_manifest or not annotation or not images_dir:
            raise HTTPException(status_code=400, detail="dataset_manifest, annotation, and images_dir are required")
        try:
            from ..data.virtual_tile_sampling import preview_from_manifest

            return preview_from_manifest(
                dataset_manifest=str(dataset_manifest),
                annotation=str(annotation),
                images_dir=str(images_dir),
                config=request.get("config") or {},
                max_scenes=request.get("max_scenes"),
                max_records_preview=int(request.get("max_records_preview") or 20),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/api/debug/local-tile-report/preview")
    def debug_local_tile_report_preview_endpoint(request: dict[str, Any]) -> dict[str, Any]:
        images_dir = request.get("images_dir")
        if not images_dir:
            raise HTTPException(status_code=400, detail="images_dir is required")
        try:
            from ..data.local_tile_report import preview_local_tile_reports

            return preview_local_tile_reports(
                images_dir=str(images_dir),
                tile_size=int(request.get("tile_size") or 768),
                stride=int(request.get("stride") or 512),
                stride_factors=[float(item) for item in (request.get("stride_factors") or [1.0, 0.5, 0.25])],
                recursive=bool(request.get("recursive", False)),
                max_scenes=request.get("max_scenes"),
                max_records_preview=int(request.get("max_records_preview") or 20),
                include_augmentation_catalog=bool(request.get("include_augmentation_catalog", False)),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/api/debug/annotated-tile-report/preview")
    def debug_annotated_tile_report_preview_endpoint(request: dict[str, Any]) -> dict[str, Any]:
        input_dir = request.get("input_dir")
        if not input_dir:
            raise HTTPException(status_code=400, detail="input_dir is required")
        try:
            from ..tile_preparation.report import preview_annotated_tile_report

            return preview_annotated_tile_report(
                input_dir=str(input_dir),
                tile_size=int(request.get("tile_size") or 768),
                stride=int(request.get("stride") or 512),
                positive_stride_factor=float(request.get("positive_stride_factor") or 1.0),
                hard_negative_stride_factor=float(request.get("hard_negative_stride_factor") or 1.0),
                negative_stride_factor=float(request.get("negative_stride_factor") or 1.0),
                min_positive_pixels=int(request.get("min_positive_pixels") or 1),
                max_scenes=request.get("max_scenes"),
                max_records_preview=int(request.get("max_records_preview") or 20),
                include_annotation_summary=bool(request.get("include_annotation_summary", True)),
                include_augmentation_catalog=bool(request.get("include_augmentation_catalog", False)),
                annotation_crs=request.get("annotation_crs", "auto"),
                allow_inferred_annotation_crs=bool(request.get("allow_inferred_annotation_crs", True)),
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/v1/debug/run-stage-sync", response_model=JobStatusResponse, dependencies=[Depends(require_api_token)])
def debug_run_stage_sync_endpoint(request: StageStartRequest, run_id: str, stage_name: str) -> JobStatusResponse:
    try:
        return debug_run_stage_sync(run_id, stage_name, request, JobStore())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
