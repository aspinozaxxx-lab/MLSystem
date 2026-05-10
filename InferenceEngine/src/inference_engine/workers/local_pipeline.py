from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..api.schemas import JobRequest
from ..config.settings import InferenceEngineSettings
from ..planning.planner import ScenePlan, build_job_plan, write_plan
from ..queues.messages import make_message
from ..storage.job_store import JobStore
from ..telemetry.metrics import RuntimeMetrics, directory_size_bytes, gpu_util_snapshot
from ..triton.client import TritonEndpoint
from .backpressure import AdaptiveProducer
from .block_worker import materialize_expanded_block, vectorize_expanded_block
from .dependency_tracker import DependencyTracker
from .finalizer import finalize_job_artifacts, merge_scene_blocks
from .tile_workers import infer_tile_batch, preprocess_tile_descriptor


def run_job_local(job_id: str, *, store: JobStore, settings: InferenceEngineSettings) -> dict[str, Any]:
    request = JobRequest.model_validate(store.read_request(job_id))
    job_dir = store.job_dir(job_id)
    metrics = RuntimeMetrics()
    store.update(job_id, status="running")
    store.add_event(job_id, "job.running", {})
    try:
        plans = build_job_plan(job_id, request, job_dir)
        plan_path = write_plan(job_dir, plans)
        store.update(job_id, artifacts={"plan.json": str(plan_path)}, counters={"scenes_total": len(plans)})
        scene_summaries: list[dict[str, Any]] = []
        first_block_vectorized_at: float | None = None
        last_tile_inferred_at: float | None = None
        for plan in plans:
            summary, first_at, last_at = _run_scene(job_id, request, plan, job_dir, store, settings, metrics)
            scene_summaries.append(summary)
            if first_at is not None and (first_block_vectorized_at is None or first_at < first_block_vectorized_at):
                first_block_vectorized_at = first_at
            if last_at is not None and (last_tile_inferred_at is None or last_at > last_tile_inferred_at):
                last_tile_inferred_at = last_at
        if first_block_vectorized_at is not None:
            metrics.set("first_block_vectorized_at", first_block_vectorized_at)
        if last_tile_inferred_at is not None:
            metrics.set("last_tile_inferred_at", last_tile_inferred_at)
        overlap = None
        if first_block_vectorized_at is not None and last_tile_inferred_at is not None:
            overlap = max(0.0, last_tile_inferred_at - first_block_vectorized_at)
            metrics.set("streaming_overlap_sec", overlap)
        artifacts = finalize_job_artifacts(
            job_id=job_id,
            job_dir=job_dir,
            request=request,
            plans=plans,
            scene_summaries=scene_summaries,
            metrics=metrics.snapshot(),
        )
        acceptance = {
            "streaming_overlap_required": sum(len(plan.blocks) for plan in plans) > 1,
            "first_block_vectorized_at": first_block_vectorized_at,
            "last_tile_inferred_at": last_tile_inferred_at,
            "passed": True,
        }
        if acceptance["streaming_overlap_required"]:
            acceptance["passed"] = bool(first_block_vectorized_at is not None and last_tile_inferred_at is not None and first_block_vectorized_at < last_tile_inferred_at)
            if not acceptance["passed"]:
                raise RuntimeError("Streaming acceptance failed: first_block_vectorized_at must be less than last_tile_inferred_at")
        store.add_event(job_id, "job.finalize", {"artifacts": artifacts, "acceptance": acceptance})
        state = store.update(job_id, status="success", metrics=metrics.snapshot(), artifacts=artifacts, counters={"scenes_processed": len(plans)})
        return state
    except Exception as exc:  # noqa: BLE001 - errors are persisted for API consumers.
        store.add_event(job_id, "job.failed", {"error": f"{type(exc).__name__}: {exc}"})
        return store.update(job_id, status="failed", error=f"{type(exc).__name__}: {exc}", metrics=metrics.snapshot())


def _run_scene(
    job_id: str,
    request: JobRequest,
    plan: ScenePlan,
    job_dir: Path,
    store: JobStore,
    settings: InferenceEngineSettings,
    metrics: RuntimeMetrics,
) -> tuple[dict[str, Any], float | None, float | None]:
    store.add_event(job_id, "scene.plan", {"scene_id": plan.scene_id, "tiles": len(plan.tiles), "blocks": len(plan.blocks)})
    metrics.inc("tiles_total", len(plan.tiles))
    metrics.inc("blocks_total", len(plan.blocks))
    tracker = DependencyTracker.from_scene_plan(plan)
    tiles_by_id = {tile.tile_id: tile for tile in plan.tiles}
    producer = AdaptiveProducer(request.resource)
    endpoint = TritonEndpoint(
        url=settings.triton_url,
        model_name=request.model.triton_model_name or request.model.model_name or "segformer_b2",
        model_version=request.model.triton_model_version,
    )
    infer_queue: list[dict[str, Any]] = []
    block_summaries: list[dict[str, Any]] = []
    first_block_vectorized_at: float | None = None
    last_tile_inferred_at: float | None = None
    spool_dir = settings.spool_root / job_id / plan.scene_id

    def drain_infer(force: bool = False) -> None:
        nonlocal first_block_vectorized_at, last_tile_inferred_at
        batch_size = max(1, int(request.resource.triton_batch_size))
        while infer_queue and (force or len(infer_queue) >= batch_size):
            batch = infer_queue[:batch_size]
            del infer_queue[:batch_size]
            batch_outputs = infer_tile_batch(
                descriptors=batch,
                tiles_by_id=tiles_by_id,
                scene_plan=plan,
                request=request,
                endpoint=endpoint,
            )
            metrics.inc("triton_batches")
            metrics.inc("triton_batch_fill_ratio", len(batch_outputs) / batch_size)
            if batch_outputs:
                metrics.set("triton_request_duration_ms", batch_outputs[-1].get("triton_request_duration_ms"))
            for row in batch_outputs:
                tile_id = str(row["tile_id"])
                last_tile_inferred_at = time.time()
                metrics.inc("tiles_done")
                store.add_event(job_id, "tile.done", {"scene_id": plan.scene_id, "tile_id": tile_id})
                message = make_message(job_id=job_id, stage="tile.done", scene_id=plan.scene_id, tile_id=tile_id, payload=row)
                del message
                ready_blocks = tracker.mark_tile_done(tile_id)
                for block_id in ready_blocks:
                    block = tracker.block(block_id, plan)
                    store.add_event(job_id, "block.ready", {"scene_id": plan.scene_id, "block_id": block_id})
                    materialize_expanded_block(block, plan, tiles_by_id)
                    store.add_event(job_id, "block.vectorize", {"scene_id": plan.scene_id, "block_id": block_id})
                    block_summary = vectorize_expanded_block(block, plan, request.effective_vectorization())
                    block_summaries.append(block_summary)
                    metrics.inc("blocks_done")
                    if first_block_vectorized_at is None:
                        first_block_vectorized_at = time.time()
                    store.add_event(job_id, "block.done", {"scene_id": plan.scene_id, "block_id": block_id, "summary": block_summary})

    for tile in plan.tiles:
        spool_bytes = directory_size_bytes(spool_dir)
        metrics.set("spool_bytes", spool_bytes)
        metrics.set("infer_queue_depth", len(infer_queue))
        if producer.should_pause(infer_ready=len(infer_queue), infer_unacked=0, spool_bytes=spool_bytes):
            metrics.inc("preprocess_pauses_total")
            drain_infer(force=True)
            if producer.should_resume(infer_ready=len(infer_queue), infer_unacked=0, spool_bytes=directory_size_bytes(spool_dir)):
                metrics.inc("preprocess_resumes_total")
        store.add_event(job_id, "tile.preprocess", {"scene_id": plan.scene_id, "tile_id": tile.tile_id})
        descriptor = preprocess_tile_descriptor(tile=tile, scene_plan=plan, request=request, spool_dir=spool_dir)
        infer_queue.append(descriptor)
        store.add_event(job_id, "tile.infer", {"scene_id": plan.scene_id, "tile_id": tile.tile_id})
        drain_infer(force=False)
    drain_infer(force=True)
    store.add_event(job_id, "scene.merge", {"scene_id": plan.scene_id, "blocks_done": len(block_summaries)})
    scene_summary = merge_scene_blocks(job_dir=job_dir, plan=plan, request=request, block_summaries=block_summaries)
    metrics.counters.update(producer.state.__dict__)
    metrics.counters["gpu_util_snapshot_count"] = float(len(gpu_util_snapshot()))
    return scene_summary, first_block_vectorized_at, last_tile_inferred_at
