from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("FastAPI is required for mlsystem-api. Install fastapi and uvicorn.") from exc

from ..train_pipeline.api import (
    DEFAULT_PIPELINE_STAGES,
    JobStatusResponse,
    PipelineRunConfig,
    StageStartRequest,
    StageStartResponse,
    cancel_train_pipeline_run,
    create_run_store,
    create_stage_job_runner,
    create_stage_job_store,
    debug_run_stage_sync,
    get_train_pipeline_run,
    job_status,
    load_trace_payload,
    parse_pipeline_run_config,
    run_summary,
    stages_payload,
    start_train_pipeline_run,
    start_stage,
    tail_train_pipeline_log,
)
from ..inference_pipeline.api import (
    PseudolabelRunRequest,
    cancel_pseudolabel_run,
    get_pseudolabel_run,
    start_pseudolabel_run,
    tail_pseudolabel_log,
)
from . import __version__
from .security import mask_text, masked_env_snapshot, verify_token_header


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
    run_root = Path(os.getenv("MLSYSTEM_RUN_ROOT", "/data/mlsystem/runs"))
    job_root = Path(os.getenv("MLSYSTEM_API_JOB_ROOT", "/data/mlsystem/api/jobs"))
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        probe = run_root / ".ready-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks["run_root"] = {"status": "ok", "path": str(run_root)}
    except Exception as exc:  # noqa: BLE001
        checks["run_root"] = {"status": "failed", "message": mask_text(str(exc))}
    try:
        job_root.mkdir(parents=True, exist_ok=True)
        checks["job_root"] = {"status": "ok", "path": str(job_root)}
    except Exception as exc:  # noqa: BLE001
        checks["job_root"] = {"status": "failed", "message": mask_text(str(exc))}
    try:
        payload = stages_payload()
        checks["stage_registry"] = {
            "status": "ok",
            "registry_stages": payload["registry_stages"],
            "pipeline_stages": payload.get("pipeline_stages") or DEFAULT_PIPELINE_STAGES,
        }
    except Exception as exc:  # noqa: BLE001
        checks["stage_registry"] = {"status": "failed", "message": mask_text(str(exc))}
    try:
        from ..mlflow_adapter.api import check_mlflow
        from ..settings.api import load_config

        checks["mlflow"] = {"status": "ok", "details": check_mlflow(load_config())}
    except Exception as exc:  # noqa: BLE001
        checks["mlflow"] = {"status": "degraded", "message": mask_text(str(exc))}
    env = masked_env_snapshot()
    checks["env"] = {"status": "ok", "keys": sorted(env)}
    required_env = [
        "MLSYSTEM_RUN_ROOT",
        "MLSYSTEM_API_JOB_ROOT",
        "MLSYSTEM_API_TOKEN",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLSYSTEM_TRITON_URL",
        "INFERENCE_ENGINE_API_URL",
    ]
    missing_env = [key for key in required_env if not os.getenv(key)]
    checks["required_env"] = {"status": "failed" if missing_env else "ok", "missing": missing_env}
    critical_checks = {key: value for key, value in checks.items() if key != "mlflow"}
    status = "ok" if all(item.get("status") == "ok" for item in critical_checks.values()) else "degraded"
    if status != "ok":
        response.status_code = 503
    return {"status": status, "service": "mlsystem-api", "checks": checks}


@app.get("/api/v1/stages")
def list_stages() -> dict[str, Any]:
    return stages_payload()


@app.post("/api/v1/runs/{run_id}/stages/{stage_name}/start", response_model=StageStartResponse, dependencies=[Depends(require_api_token)])
def start_stage_endpoint(run_id: str, stage_name: str, request: StageStartRequest) -> StageStartResponse:
    try:
        return start_stage(run_id, stage_name, request, create_stage_job_runner())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, dependencies=[Depends(require_api_token)])
def job_status_endpoint(job_id: str) -> JobStatusResponse:
    try:
        return job_status(job_id, create_stage_job_runner())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc


@app.get("/api/v1/runs/{run_id}/summary")
def run_summary_endpoint(run_id: str) -> dict[str, Any]:
    return run_summary(run_id, os.getenv("MLSYSTEM_RUN_ROOT", "/data/mlsystem/runs"))


@app.post("/api/v1/pipeline-runs", dependencies=[Depends(require_api_token)])
async def start_pipeline_run_endpoint(request: Request) -> dict[str, Any]:
    try:
        config = await _pipeline_config_from_request(request)
        run = start_train_pipeline_run(config, source="api")
        return {
            "run_id": run.run_id,
            "state": run.state,
            "status_url": f"/api/v1/pipeline-runs/{run.run_id}",
            "log_url": f"/api/v1/pipeline-runs/{run.run_id}/log",
            "created_at": run.created_at,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/pipeline-runs/{run_id}", dependencies=[Depends(require_api_token)])
def pipeline_run_status_endpoint(run_id: str) -> dict[str, Any]:
    try:
        run = get_train_pipeline_run(run_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}") from exc
    run["log_tail"] = tail_train_pipeline_log(run_id, max_chars=20000)
    return run


@app.get("/api/v1/pipeline-runs/{run_id}/log", dependencies=[Depends(require_api_token)])
def pipeline_run_log_endpoint(run_id: str, tail: int = 20000) -> dict[str, Any]:
    try:
        run = get_train_pipeline_run(run_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}") from exc
    return {"run_id": run_id, "log_tail": tail_train_pipeline_log(run_id, max_chars=max(1, min(int(tail), 1_000_000))), "updated_at": run.get("updated_at")}


@app.get("/api/v1/pipeline-runs/{run_id}/stages", dependencies=[Depends(require_api_token)])
def pipeline_run_stages_endpoint(run_id: str) -> dict[str, Any]:
    store = create_run_store()
    try:
        run = store.read_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}") from exc
    reports = []
    for path in sorted((store.root / run_id / "stages").glob("*.json")):
        try:
            import json

            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            reports.append({"stage": path.stem, "status": "unreadable", "error": mask_text(str(exc))})
    return {"run_id": run_id, "stages": run.get("stages") or [], "reports": reports}


@app.post("/api/v1/pipeline-runs/{run_id}/cancel", dependencies=[Depends(require_api_token)])
def pipeline_run_cancel_endpoint(run_id: str) -> dict[str, Any]:
    try:
        return cancel_train_pipeline_run(run_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pipeline run not found: {run_id}") from exc


@app.post("/api/v1/pseudolabel-runs", dependencies=[Depends(require_api_token)])
def start_pseudolabel_run_endpoint(request: PseudolabelRunRequest) -> dict[str, Any]:
    try:
        run = start_pseudolabel_run(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "run_id": run.run_id,
        "state": run.state,
        "status_url": f"/api/v1/pseudolabel-runs/{run.run_id}",
        "log_url": f"/api/v1/pseudolabel-runs/{run.run_id}/log",
        "created_at": run.created_at,
    }


@app.get("/api/v1/pseudolabel-runs/{run_id}", dependencies=[Depends(require_api_token)])
def pseudolabel_run_status_endpoint(run_id: str) -> dict[str, Any]:
    try:
        return get_pseudolabel_run(run_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pseudolabel run not found: {run_id}") from exc


@app.get("/api/v1/pseudolabel-runs/{run_id}/log", dependencies=[Depends(require_api_token)])
def pseudolabel_run_log_endpoint(run_id: str, tail: int = 20000) -> dict[str, Any]:
    try:
        run = get_pseudolabel_run(run_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pseudolabel run not found: {run_id}") from exc
    return {"run_id": run_id, "log_tail": tail_pseudolabel_log(run_id, max_chars=max(1, min(int(tail), 1_000_000))), "updated_at": run.get("updated_at")}


@app.post("/api/v1/pseudolabel-runs/{run_id}/cancel", dependencies=[Depends(require_api_token)])
def pseudolabel_run_cancel_endpoint(run_id: str) -> dict[str, Any]:
    try:
        return cancel_pseudolabel_run(run_id).model_dump()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Pseudolabel run not found: {run_id}") from exc


async def _pipeline_config_from_request(request: Request) -> PipelineRunConfig:
    content_type = request.headers.get("content-type", "")
    dry_run = False
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("trace_file")
        if upload is None or not hasattr(upload, "read"):
            raise ValueError("multipart request must include trace_file")
        raw = await upload.read()  # type: ignore[attr-defined]
        filename = str(getattr(upload, "filename", None) or "trace.yaml")
        trace = load_trace_payload(raw.decode("utf-8-sig"), source_name=filename)
        dry_run = _truthy(form.get("dry_run"))
    else:
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"request body must be JSON or multipart trace_file: {exc}") from exc
        if not isinstance(body, dict):
            raise ValueError("request body must be an object")
        trace = body.get("trace") if "trace" in body else body
        if not isinstance(trace, dict):
            raise ValueError("trace must be an object")
        dry_run = _truthy(body.get("dry_run"))
    config = parse_pipeline_run_config(trace)
    if dry_run:
        config = config.model_copy(update={"pipeline": config.pipeline.model_copy(update={"dry_run": True})})
    return config


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.post("/api/v1/debug/run-stage-sync", response_model=JobStatusResponse, dependencies=[Depends(require_api_token)])
def debug_run_stage_sync_endpoint(request: StageStartRequest, run_id: str, stage_name: str) -> JobStatusResponse:
    try:
        return debug_run_stage_sync(run_id, stage_name, request, create_stage_job_store())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
