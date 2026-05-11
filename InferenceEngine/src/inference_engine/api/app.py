from __future__ import annotations

import asyncio
import os
import urllib.request
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Response

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
    return {"status": "ok", "service": "inference-engine", "commit": os.getenv("MLSYSTEM_COMMIT") or os.getenv("GIT_COMMIT") or "unknown"}


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    settings.ensure_dirs()
    checks: dict[str, Any] = {
        "job_root": _path_check(settings.job_root),
        "spool_root": _path_check(settings.spool_root),
        "artifact_root": _path_check(settings.artifact_root),
        "logs_root": _path_check(settings.logs_root),
        "rabbitmq": {"status": "ok" if not settings.use_rabbitmq else "unknown", "enabled": settings.use_rabbitmq},
        "triton": _http_check(settings.triton_url.rstrip("/") + "/v2/health/ready"),
    }
    if settings.use_rabbitmq:
        queue_metrics = rabbitmq_queue_metrics()
        checks["rabbitmq"] = {
            "status": "ok" if settings.rabbitmq_url else "failed",
            "enabled": True,
            "management_metrics_available": bool(queue_metrics),
            "queues_seen": len(queue_metrics),
        }
    status = "ready" if all(item.get("status") == "ok" for item in checks.values()) else "degraded"
    if status != "ready":
        response.status_code = 503
    return {"status": status, "service": "inference-engine", "checks": checks}


@app.get("/queues")
def queues() -> dict[str, Any]:
    return {**validate_queue_contracts(), "metrics": rabbitmq_queue_metrics()}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return _collect_metrics()


@app.get("/metrics/prometheus")
def prometheus_metrics() -> Response:
    return Response(_render_prometheus_metrics(_collect_metrics()), media_type="text/plain; version=0.0.4; charset=utf-8")


def _collect_metrics() -> dict[str, Any]:
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
    return {"status": "ok", "queues": QUEUE_NAMES, "queue_metrics": rabbitmq_queue_metrics(), "aggregate": aggregate, "jobs": jobs}


def _render_prometheus_metrics(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# HELP inference_engine_jobs_total InferenceEngine jobs by status.",
        "# TYPE inference_engine_jobs_total gauge",
    ]
    jobs = payload.get("jobs") or {}
    status_counts: dict[str, int] = {}
    active_jobs = 0
    for job in jobs.values():
        status = str((job or {}).get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status in {"queued", "running"}:
            active_jobs += 1
    for status, count in sorted(status_counts.items()):
        lines.append(f'inference_engine_jobs_total{{status="{_label(status)}"}} {count}')
    lines.extend(
        [
            "# HELP inference_engine_active_jobs InferenceEngine queued or running jobs.",
            "# TYPE inference_engine_active_jobs gauge",
            f"inference_engine_active_jobs {active_jobs}",
        ]
    )
    aggregate = payload.get("aggregate") or {}
    metric_map = {
        "tiles_total": "inference_engine_tiles_total",
        "tiles_done": "inference_engine_tiles_done",
        "blocks_total": "inference_engine_blocks_total",
        "blocks_done": "inference_engine_blocks_done",
        "triton_batches": "inference_engine_triton_batches_total",
        "triton_batch_fill_ratio": "inference_engine_triton_batch_fill_ratio",
        "triton_request_duration_ms": "inference_engine_triton_request_duration_ms",
        "streaming_overlap_sec": "inference_engine_streaming_overlap_seconds",
        "spool_bytes": "inference_engine_spool_bytes",
        "preprocess_pauses_total": "inference_engine_preprocess_pauses_total",
        "preprocess_resumes_total": "inference_engine_preprocess_resumes_total",
    }
    for source, name in metric_map.items():
        value = _number(aggregate.get(source))
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")
    for job_id, job in sorted(jobs.items()):
        metrics = (job or {}).get("metrics") or {}
        status = str((job or {}).get("status") or "unknown")
        for source, name in metric_map.items():
            if source in metrics:
                lines.append(f'{name}{{job_id="{_label(job_id)}",status="{_label(status)}"}} {_number(metrics.get(source))}')
    lines.extend(
        [
            "# HELP inference_engine_queue_ready RabbitMQ ready messages per InferenceEngine queue.",
            "# TYPE inference_engine_queue_ready gauge",
            "# HELP inference_engine_queue_unacked RabbitMQ unacked messages per InferenceEngine queue.",
            "# TYPE inference_engine_queue_unacked gauge",
            "# HELP inference_engine_queue_consumers RabbitMQ consumers per InferenceEngine queue.",
            "# TYPE inference_engine_queue_consumers gauge",
        ]
    )
    dead_letter = 0.0
    for item in payload.get("queue_metrics") or []:
        queue = str(item.get("name") or "")
        ready = _number(item.get("messages_ready"))
        unacked = _number(item.get("messages_unacked"))
        consumers = _number(item.get("consumers"))
        lines.append(f'inference_engine_queue_ready{{queue="{_label(queue)}"}} {ready}')
        lines.append(f'inference_engine_queue_unacked{{queue="{_label(queue)}"}} {unacked}')
        lines.append(f'inference_engine_queue_consumers{{queue="{_label(queue)}"}} {consumers}')
        if queue == "ie.dead_letter":
            dead_letter = ready + unacked
    lines.extend(
        [
            "# HELP inference_engine_dead_letter_messages InferenceEngine dead-letter messages.",
            "# TYPE inference_engine_dead_letter_messages gauge",
            f"inference_engine_dead_letter_messages {dead_letter}",
            "",
        ]
    )
    return "\n".join(lines)


def _number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


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


def _path_check(path) -> dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ready"
        probe.write_text("ok", encoding="utf-8")
        try:
            probe.unlink()
        except OSError:
            pass
        return {"status": "ok", "path": str(path)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "path": str(path), "message": str(exc)}


def _http_check(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return {"status": "ok" if 200 <= response.status < 300 else "failed", "url": url, "http_status": response.status}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "url": url, "message": str(exc)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("inference_engine.api.app:app", host="0.0.0.0", port=8095)
