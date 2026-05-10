from __future__ import annotations

import asyncio
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException

from ..config.settings import InferenceEngineSettings
from ..queues.messages import QUEUE_NAMES, make_message, validate_queue_contracts
from ..queues.rabbitmq import RabbitMQClient
from ..queues.telemetry import rabbitmq_queue_metrics
from ..storage.job_store import JobStore
from ..workers.local_pipeline import run_job_local
from .schemas import JobCreated, JobRequest, JobStatus


settings = InferenceEngineSettings.from_env()
settings.ensure_dirs()
store = JobStore(settings.job_root)
app = FastAPI(title="MLSystem InferenceEngine", version="0.1.0")


def require_token(authorization: str | None = Header(default=None)) -> None:
    if not settings.api_token:
        return
    expected = f"Bearer {settings.api_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid InferenceEngine API token")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "inference-engine"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    settings.ensure_dirs()
    return {"status": "ready", "job_root": str(settings.job_root), "rabbitmq_enabled": settings.use_rabbitmq}


@app.get("/queues")
def queues() -> dict[str, Any]:
    return {**validate_queue_contracts(), "metrics": rabbitmq_queue_metrics()}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    aggregate: dict[str, float] = {}
    jobs: dict[str, Any] = {}
    for state_path in settings.job_root.glob("*/state.json"):
        try:
            state = store.read(state_path.parent.name)
        except Exception:
            continue
        job_metrics = state.get("metrics") or {}
        jobs[state_path.parent.name] = {"status": state.get("status"), "metrics": job_metrics}
        for key, value in job_metrics.items():
            if isinstance(value, (int, float)):
                aggregate[key] = float(aggregate.get(key, 0.0)) + float(value)
    return {"status": "ok", "queues": QUEUE_NAMES, "aggregate": aggregate, "jobs": jobs}


@app.post("/api/v1/jobs", response_model=JobCreated, dependencies=[Depends(require_token)])
async def create_job(request: JobRequest, background_tasks: BackgroundTasks) -> JobCreated:
    state = store.create(request.model_dump())
    job_id = str(state["job_id"])
    if settings.use_rabbitmq:
        await _publish_job_submit(job_id)
    else:
        background_tasks.add_task(run_job_local, job_id, store=store, settings=settings)
    return JobCreated(
        job_id=job_id,
        status=str(state["status"]),
        events_url=f"/api/v1/jobs/{job_id}/events",
        artifacts_url=f"/api/v1/jobs/{job_id}/artifacts",
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus, dependencies=[Depends(require_token)])
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return store.read(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/jobs/{job_id}/events", dependencies=[Depends(require_token)])
def get_events(job_id: str) -> dict[str, Any]:
    try:
        store.read(job_id)
        return {"job_id": job_id, "events": store.events(job_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/jobs/{job_id}/artifacts", dependencies=[Depends(require_token)])
def get_artifacts(job_id: str) -> dict[str, Any]:
    try:
        return {"job_id": job_id, "artifacts": store.artifacts(job_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobStatus, dependencies=[Depends(require_token)])
def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        return store.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _publish_job_submit(job_id: str) -> None:
    client = RabbitMQClient(settings.rabbitmq_url)
    try:
        await client.publish("ie.jobs.submit", make_message(job_id=job_id, stage="jobs.submit", payload={"job_id": job_id}))
    finally:
        await client.close()


def run_background_job(job_id: str) -> None:
    run_job_local(job_id, store=store, settings=settings)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference_engine.api.app:app", host="0.0.0.0", port=8095)
