from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from ..api.schemas import JobRequest, SceneInput
from ..config.settings import InferenceEngineSettings
from ..planning.planner import build_scene_plan, read_scene_plan, resolve_scene_inputs, write_scene_plan
from ..queues.messages import QueueMessage, make_message
from ..queues.rabbitmq import RabbitMQClient
from ..storage.job_store import JobStore, TERMINAL_STATUSES
from ..storage.runtime_cleanup import cleanup_scene_runtime, cleanup_terminal_job_runtime, prune_missing_local_artifacts
from ..telemetry.metrics import directory_size_bytes, gpu_util_snapshot
from ..triton.client import TritonEndpoint
from .block_worker import materialize_expanded_block, vectorize_expanded_block
from .finalizer import finalize_job_artifacts, merge_scene_blocks
from .state import ProgressStore, SceneStateStore, initialize_progress, initialize_scene_state
from .tile_workers import infer_tile_batch, preprocess_tile_descriptor


ROLE_QUEUES = {
    "submit": "ie.jobs.submit",
    "planner": "ie.scene.plan",
    "preprocess": "ie.tile.preprocess",
    "triton": "ie.tile.infer",
    "aggregator": "ie.tile.done",
    "block-ready": "ie.block.ready",
    "block-vectorize": "ie.block.vectorize",
    "block-done": "ie.block.done",
    "merger": "ie.scene.merge",
    "finalizer": "ie.job.finalize",
}


class RabbitPipeline:
    def __init__(self, settings: InferenceEngineSettings) -> None:
        self.settings = settings
        self.store = JobStore(settings.job_root)
        self.client = RabbitMQClient(settings.rabbitmq_url, failure_handler=self._handle_message_failure)

    async def run_role(self, role: str) -> None:
        role = role.replace("_", "-")
        if role == "submit":
            await self.client.consume_forever("ie.jobs.submit", self.handle_submit)
        elif role == "planner":
            await asyncio.gather(
                RabbitPipeline(self.settings).run_role("submit"),
                RabbitPipeline(self.settings).client.consume_forever("ie.scene.plan", self.handle_scene_plan),
            )
        elif role == "preprocess":
            await self.client.consume_forever("ie.tile.preprocess", self.handle_tile_preprocess)
        elif role == "triton":
            await self.client.consume_batches_forever(
                "ie.tile.infer",
                max_batch_size=int(self._default_batch_size()),
                max_wait_ms=int(self.settings.max_wait_ms),
                handler=self.handle_tile_infer_batch,
            )
        elif role == "aggregator":
            await self.client.consume_forever("ie.tile.done", self.handle_tile_done)
        elif role == "block":
            await asyncio.gather(
                RabbitPipeline(self.settings).run_role("block-ready"),
                RabbitPipeline(self.settings).run_role("block-vectorize"),
            )
        elif role == "block-ready":
            await self.client.consume_forever("ie.block.ready", self.handle_block_ready)
        elif role == "block-vectorize":
            await self.client.consume_forever("ie.block.vectorize", self.handle_block_vectorize)
        elif role == "block-done":
            await self.client.consume_forever("ie.block.done", self.handle_block_done)
        elif role == "merger":
            await asyncio.gather(
                RabbitPipeline(self.settings).run_role("block-done"),
                RabbitPipeline(self.settings).client.consume_forever("ie.scene.merge", self.handle_scene_merge),
            )
        elif role == "finalizer":
            await self.client.consume_forever("ie.job.finalize", self.handle_job_finalize)
        elif role == "all":
            await asyncio.gather(*(RabbitPipeline(self.settings).run_role(item) for item in ["planner", "preprocess", "triton", "aggregator", "block", "merger", "finalizer"]))
        else:
            raise ValueError(f"Unknown InferenceEngine worker role: {role}")

    async def handle_submit(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        request = JobRequest.model_validate(self.store.read_request(job_id))
        job_dir = self.store.job_dir(job_id)
        scenes = resolve_scene_inputs(request)
        scene_payloads = [scene.model_dump() for scene in scenes]
        if not (job_dir / "progress.json").exists():
            initialize_progress(job_dir, job_id=job_id, scenes=scene_payloads, max_scenes_inflight=request.resource.max_scenes_inflight)
        self.store.update(job_id, status="running")
        await self._event(job_id, "job.running", {"scene_count": len(scenes)})
        await self._publish_more_scene_plans(job_id)

    async def handle_scene_plan(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        job_dir = self.store.job_dir(job_id)
        request = JobRequest.model_validate(self.store.read_request(job_id))
        scene_index = int(message.payload["scene_index"])
        scene = SceneInput.model_validate(message.payload["scene"])
        plan = build_scene_plan(job_id, scene_index, scene, request, job_dir)
        plan_path = write_scene_plan(job_dir, plan)
        if not (job_dir / "scenes" / plan.scene_id / "scene_state.json").exists():
            initialize_scene_state(job_dir, plan)
        await self._event(job_id, "scene.plan", {"scene_id": plan.scene_id, "tiles": len(plan.tiles), "blocks": len(plan.blocks), "plan": str(plan_path)})
        progress = ProgressStore(job_dir)

        def mutate(payload: dict[str, Any]) -> None:
            planned = set(payload.get("scene_plan_done") or [])
            if plan.scene_id not in planned:
                payload["scene_plan_done"] = sorted(planned | {plan.scene_id})
                payload["tiles_total"] = int(payload.get("tiles_total") or 0) + len(plan.tiles)
                payload["blocks_total"] = int(payload.get("blocks_total") or 0) + len(plan.blocks)

        progress.update(mutate)
        self.store.update(job_id, counters={"tiles_total": len(plan.tiles), "blocks_total": len(plan.blocks)}, artifacts={f"{plan.scene_id}.plan.json": str(plan_path)})
        await self._publish_more_preprocess(job_id, plan.scene_id)

    async def handle_tile_preprocess(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        scene_id = str(message.payload["scene_id"])
        tile_id = str(message.payload["tile_id"])
        job_dir = self.store.job_dir(job_id)
        request = JobRequest.model_validate(self.store.read_request(job_id))
        plan = read_scene_plan(job_dir, scene_id)
        tile = _tile_by_id(plan, tile_id)
        spool_dir = self.settings.spool_root / job_id / scene_id
        descriptor = preprocess_tile_descriptor(tile=tile, scene_plan=plan, request=request, spool_dir=spool_dir)
        await self._event(job_id, "tile.preprocess", {"scene_id": scene_id, "tile_id": tile_id, "spool_path": descriptor.get("spool_path")})
        await self.client.publish("ie.tile.infer", make_message(job_id=job_id, stage="tile.infer", scene_id=scene_id, tile_id=tile_id, payload={"scene_id": scene_id, **descriptor}))

    async def handle_tile_infer_batch(self, messages: list[QueueMessage]) -> None:
        if not messages:
            return
        terminal_cache: dict[str, bool] = {}
        active_messages = []
        for message in messages:
            terminal_cache.setdefault(message.job_id, self._job_is_terminal(message.job_id))
            if not terminal_cache[message.job_id]:
                active_messages.append(message)
        messages = active_messages
        if not messages:
            return
        by_scene: dict[tuple[str, str], list[QueueMessage]] = {}
        for message in messages:
            by_scene.setdefault((message.job_id, str(message.payload["scene_id"])), []).append(message)
        for (job_id, scene_id), rows in by_scene.items():
            job_dir = self.store.job_dir(job_id)
            request = JobRequest.model_validate(self.store.read_request(job_id))
            plan = read_scene_plan(job_dir, scene_id)
            tiles_by_id = {tile.tile_id: tile for tile in plan.tiles}
            endpoint = TritonEndpoint(
                url=self.settings.triton_url,
                model_name=request.model.triton_model_name or request.model.model_name or "segformer_b2",
                model_version=request.model.triton_model_version,
            )
            descriptors = [dict(row.payload) for row in rows]
            outputs = infer_tile_batch(descriptors=descriptors, tiles_by_id=tiles_by_id, scene_plan=plan, request=request, endpoint=endpoint)
            duration_ms = max([float(item.get("triton_request_duration_ms") or 0.0) for item in outputs] or [0.0])
            fill_ratio = len(outputs) / max(1, int(request.resource.triton_batch_size or len(outputs)))
            progress = ProgressStore(job_dir)

            def mutate(payload: dict[str, Any]) -> None:
                payload["triton_batches"] = int(payload.get("triton_batches") or 0) + 1
                payload["triton_batch_fill_sum"] = float(payload.get("triton_batch_fill_sum") or 0.0) + fill_ratio
                payload["triton_request_duration_ms_sum"] = float(payload.get("triton_request_duration_ms_sum") or 0.0) + duration_ms
                payload["triton_request_count"] = int(payload.get("triton_request_count") or 0) + 1

            progress.update(mutate)
            self.store.update(job_id, metrics=_metrics_from_progress(progress.read()))
            await self._event(job_id, "tile.infer", {"scene_id": scene_id, "batch_size": len(outputs), "duration_ms": duration_ms})
            for output in outputs:
                tile_id = str(output["tile_id"])
                await self.client.publish("ie.tile.done", make_message(job_id=job_id, stage="tile.done", scene_id=scene_id, tile_id=tile_id, payload={"scene_id": scene_id, **output}))

    async def handle_tile_done(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        scene_id = str(message.payload["scene_id"])
        tile_id = str(message.payload["tile_id"])
        job_dir = self.store.job_dir(job_id)
        plan = read_scene_plan(job_dir, scene_id)
        scene_state = SceneStateStore(job_dir, scene_id)
        ready_blocks: list[str] = []
        new_tile_done = False

        def mutate(payload: dict[str, Any]) -> None:
            nonlocal ready_blocks, new_tile_done
            done_tiles = set(payload.get("tiles_done") or [])
            if tile_id in done_tiles:
                ready_blocks = []
                return
            new_tile_done = True
            done_tiles.add(tile_id)
            payload["tiles_done"] = sorted(done_tiles)
            for block_id in payload.get("blocks_by_tile", {}).get(tile_id, []):
                remaining = set(payload.get("remaining_by_block", {}).get(block_id) or [])
                remaining.discard(tile_id)
                payload["remaining_by_block"][block_id] = sorted(remaining)
                if not remaining and block_id not in set(payload.get("blocks_ready_published") or []):
                    ready_blocks.append(block_id)
            if ready_blocks:
                published = set(payload.get("blocks_ready_published") or [])
                payload["blocks_ready_published"] = sorted(published | set(ready_blocks))

        scene_state.update(mutate)
        progress = ProgressStore(job_dir)

        def progress_mutate(payload: dict[str, Any]) -> None:
            if new_tile_done:
                payload["tiles_done"] = int(payload.get("tiles_done") or 0) + 1
                payload["last_tile_inferred_at"] = time.time()
            payload["spool_bytes"] = directory_size_bytes(self.settings.spool_root / job_id)

        progress.update(progress_mutate)
        await self._event(job_id, "tile.done", {"scene_id": scene_id, "tile_id": tile_id, "ready_blocks": ready_blocks})
        await self._publish_more_preprocess(job_id, scene_id)
        for block_id in ready_blocks:
            await self.client.publish("ie.block.ready", make_message(job_id=job_id, stage="block.ready", scene_id=scene_id, block_id=block_id, payload={"scene_id": scene_id, "block_id": block_id}))
        self.store.update(job_id, metrics=_metrics_from_progress(progress.read()))

    async def handle_block_ready(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        scene_id = str(message.payload["scene_id"])
        block_id = str(message.payload["block_id"])
        job_dir = self.store.job_dir(job_id)
        plan = read_scene_plan(job_dir, scene_id)
        block = _block_by_id(plan, block_id)
        materialize_expanded_block(block, plan, {tile.tile_id: tile for tile in plan.tiles})
        await self._event(job_id, "block.ready", {"scene_id": scene_id, "block_id": block_id, "artifact_path": block.artifact_path})
        await self.client.publish("ie.block.vectorize", make_message(job_id=job_id, stage="block.vectorize", scene_id=scene_id, block_id=block_id, payload={"scene_id": scene_id, "block_id": block_id}))

    async def handle_block_vectorize(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        scene_id = str(message.payload["scene_id"])
        block_id = str(message.payload["block_id"])
        job_dir = self.store.job_dir(job_id)
        request = JobRequest.model_validate(self.store.read_request(job_id))
        plan = read_scene_plan(job_dir, scene_id)
        block = _block_by_id(plan, block_id)
        summary = vectorize_expanded_block(block, plan, request.effective_vectorization())
        progress = ProgressStore(job_dir)

        def progress_mutate(payload: dict[str, Any]) -> None:
            payload.setdefault("first_block_vectorized_at", time.time())
            payload["last_block_vectorized_at"] = time.time()

        progress.update(progress_mutate)
        await self._event(job_id, "block.vectorize", {"scene_id": scene_id, "block_id": block_id, "summary": summary})
        await self.client.publish("ie.block.done", make_message(job_id=job_id, stage="block.done", scene_id=scene_id, block_id=block_id, payload={"scene_id": scene_id, "block_id": block_id, "summary": summary}))
        self.store.update(job_id, metrics=_metrics_from_progress(progress.read()))

    async def handle_block_done(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        scene_id = str(message.payload["scene_id"])
        block_id = str(message.payload["block_id"])
        job_dir = self.store.job_dir(job_id)
        plan = read_scene_plan(job_dir, scene_id)
        scene_state = SceneStateStore(job_dir, scene_id)
        publish_merge = False
        new_block_done = False

        def mutate(payload: dict[str, Any]) -> None:
            nonlocal publish_merge, new_block_done
            done = set(payload.get("blocks_done") or [])
            if block_id not in done:
                done.add(block_id)
                new_block_done = True
                payload["blocks_done"] = sorted(done)
            if len(done) >= len(plan.blocks) and not payload.get("scene_merge_published"):
                payload["scene_merge_published"] = True
                publish_merge = True

        scene_state.update(mutate)
        if new_block_done:
            progress_payload = ProgressStore(job_dir).update(lambda payload: payload.update({"blocks_done": int(payload.get("blocks_done") or 0) + 1}))
            self.store.update(job_id, metrics=_metrics_from_progress(progress_payload))
        await self._event(job_id, "block.done", {"scene_id": scene_id, "block_id": block_id})
        if publish_merge:
            await self.client.publish("ie.scene.merge", make_message(job_id=job_id, stage="scene.merge", scene_id=scene_id, payload={"scene_id": scene_id}))

    async def handle_scene_merge(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        scene_id = str(message.payload["scene_id"])
        job_dir = self.store.job_dir(job_id)
        request = JobRequest.model_validate(self.store.read_request(job_id))
        plan = read_scene_plan(job_dir, scene_id)
        block_summaries = []
        for block in plan.blocks:
            summary_path = Path(block.summary_path)
            if summary_path.exists():
                import json

                block_summaries.append(json.loads(summary_path.read_text(encoding="utf-8-sig")))
        summary = merge_scene_blocks(job_dir=job_dir, plan=plan, request=request, block_summaries=block_summaries)
        cleanup_report = cleanup_scene_runtime(job_id, scene_id, settings=self.settings, job_dir=job_dir)
        publish_finalize = False
        progress = ProgressStore(job_dir)

        def mutate(payload: dict[str, Any]) -> None:
            nonlocal publish_finalize
            done = set(payload.get("scene_done") or [])
            done.add(scene_id)
            payload["scene_done"] = sorted(done)
            payload["active_scenes"] = sorted(set(payload.get("active_scenes") or []) - {scene_id})
            payload.setdefault("scene_summaries", {})[scene_id] = summary
            publish_finalize = len(done) >= int(payload.get("scene_count") or 0)

        progress.update(mutate)
        await self._event(job_id, "scene.merge", {"scene_id": scene_id, "summary": summary})
        if cleanup_report.get("enabled"):
            await self._event(job_id, "scene.cleanup", {"scene_id": scene_id, "deleted_mb": cleanup_report.get("deleted_mb"), "deleted_files": cleanup_report.get("deleted_files")})
        await self._publish_more_scene_plans(job_id)
        if publish_finalize:
            await self.client.publish("ie.job.finalize", make_message(job_id=job_id, stage="job.finalize", payload={"job_id": job_id}))

    async def handle_job_finalize(self, message: QueueMessage) -> None:
        job_id = message.job_id
        if self._job_is_terminal(job_id):
            return
        job_dir = self.store.job_dir(job_id)
        request = JobRequest.model_validate(self.store.read_request(job_id))
        progress = ProgressStore(job_dir).read()
        scene_ids = [str((row.get("scene_id") or row.get("name") or f"scene_{idx:04d}")) for idx, row in enumerate(progress.get("scenes") or [])]
        plans = [read_scene_plan(job_dir, scene_id) for scene_id in scene_ids if (job_dir / "scenes" / scene_id / "plan.json").exists()]
        scene_summaries = [progress.get("scene_summaries", {}).get(plan.scene_id) for plan in plans]
        scene_summaries = [item for item in scene_summaries if item]
        metrics = _metrics_from_progress(progress)
        first_at = metrics.get("first_block_vectorized_at")
        last_at = metrics.get("last_tile_inferred_at")
        if len(plans) and sum(len(plan.blocks) for plan in plans) > 1 and (not first_at or not last_at or float(first_at) >= float(last_at)):
            error = "Streaming acceptance failed: first_block_vectorized_at must be less than last_tile_inferred_at"
            self.store.update(job_id, status="failed", error=error, metrics=metrics)
            await self._event(job_id, "job.failed", {"error": error})
            self._cleanup_terminal_job(job_id)
            return
        artifacts = finalize_job_artifacts(job_id=job_id, job_dir=job_dir, request=request, plans=plans, scene_summaries=scene_summaries, metrics=metrics)
        self.store.update(job_id, status="success", metrics=metrics, artifacts=artifacts, counters={"scenes_processed": len(plans)})
        await self._event(job_id, "job.finalize", {"artifacts": artifacts})
        self._cleanup_terminal_job(job_id)

    async def _publish_more_scene_plans(self, job_id: str) -> None:
        job_dir = self.store.job_dir(job_id)
        progress = ProgressStore(job_dir)
        to_publish: list[tuple[int, dict[str, Any]]] = []

        def mutate(payload: dict[str, Any]) -> None:
            nonlocal to_publish
            scenes = list(payload.get("scenes") or [])
            active = set(payload.get("active_scenes") or [])
            done = set(payload.get("scene_done") or [])
            next_index = int(payload.get("next_scene_index") or 0)
            max_inflight = max(1, int(payload.get("max_scenes_inflight") or 1))
            while next_index < len(scenes) and len(active - done) < max_inflight:
                scene = dict(scenes[next_index])
                scene_id = str(scene.get("scene_id") or scene.get("name") or f"scene_{next_index:04d}")
                active.add(scene_id)
                to_publish.append((next_index, scene))
                next_index += 1
            payload["next_scene_index"] = next_index
            payload["active_scenes"] = sorted(active - done)

        progress.update(mutate)
        for scene_index, scene in to_publish:
            scene_id = str(scene.get("scene_id") or scene.get("name") or f"scene_{scene_index:04d}")
            await self.client.publish("ie.scene.plan", make_message(job_id=job_id, stage="scene.plan", scene_id=scene_id, payload={"scene_index": scene_index, "scene": scene}))

    async def _publish_more_preprocess(self, job_id: str, scene_id: str) -> None:
        job_dir = self.store.job_dir(job_id)
        request = JobRequest.model_validate(self.store.read_request(job_id))
        plan = read_scene_plan(job_dir, scene_id)
        scene_state = SceneStateStore(job_dir, scene_id)
        batch_target = int(request.resource.triton_batch_size) * max(2, int(request.resource.triton_instance_count) * int(request.resource.batches_ahead))
        target = max(1, int(batch_target))
        queue_reserve = max(target, int(request.resource.max_preprocess_queue))
        high_watermark = queue_reserve
        low_watermark = max(1, min(target, queue_reserve // 2))
        infer_metrics = await self.client.queue_metrics("ie.tile.infer") if hasattr(self.client, "queue_metrics") else {"messages_ready": 0}
        infer_depth = int(infer_metrics.get("messages_ready") or 0)
        spool_bytes = directory_size_bytes(self.settings.spool_root / job_id)
        progress = ProgressStore(job_dir)
        if infer_depth >= high_watermark or spool_bytes > int(request.resource.max_spool_bytes):
            def pause(payload: dict[str, Any]) -> None:
                if not payload.get("preprocess_paused"):
                    payload["preprocess_pauses_total"] = int(payload.get("preprocess_pauses_total") or 0) + 1
                payload["preprocess_paused"] = True
                payload["infer_queue_depth"] = infer_depth
                payload["infer_queue_target"] = queue_reserve
                payload["infer_queue_low_watermark"] = low_watermark
                payload["infer_queue_high_watermark"] = high_watermark
                payload["spool_bytes"] = spool_bytes

            progress.update(pause)
            return
        if infer_depth <= low_watermark:
            def resume(payload: dict[str, Any]) -> None:
                if payload.get("preprocess_paused"):
                    payload["preprocess_resumes_total"] = int(payload.get("preprocess_resumes_total") or 0) + 1
                payload["preprocess_paused"] = False
                payload["infer_queue_depth"] = infer_depth
                payload["infer_queue_target"] = queue_reserve
                payload["infer_queue_low_watermark"] = low_watermark
                payload["infer_queue_high_watermark"] = high_watermark
                payload["spool_bytes"] = spool_bytes

            progress.update(resume)
        max_publish = max(0, queue_reserve - infer_depth)
        if max_publish <= 0:
            return
        to_publish = []

        def mutate(payload: dict[str, Any]) -> None:
            nonlocal to_publish
            next_index = int(payload.get("next_tile_index") or 0)
            published_now = 0
            while next_index < len(plan.tiles) and published_now < max_publish:
                tile = plan.tiles[next_index]
                to_publish.append(tile.tile_id)
                next_index += 1
                published_now += 1
            payload["next_tile_index"] = next_index
            payload["tiles_preprocess_published"] = int(payload.get("tiles_preprocess_published") or 0) + published_now

        scene_state.update(mutate)
        for tile_id in to_publish:
            await self.client.publish("ie.tile.preprocess", make_message(job_id=job_id, stage="tile.preprocess", scene_id=scene_id, tile_id=tile_id, payload={"scene_id": scene_id, "tile_id": tile_id}))

    async def _event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> None:
        self.store.add_event(job_id, event_type, payload)
        if not self.settings.publish_rabbitmq_events:
            return
        try:
            await self.client.publish("ie.events", make_message(job_id=job_id, stage="events", payload={"event_type": event_type, **payload}))
        except Exception:
            pass

    async def _handle_message_failure(self, message: QueueMessage, exc: Exception, source_queue: str, target_queue: str) -> None:
        if target_queue != "ie.dead_letter":
            return
        error = f"{message.stage} failed after retry in {source_queue}: {type(exc).__name__}: {exc}"
        try:
            self.store.update(message.job_id, status="failed", error=error)
            self.store.add_event(
                message.job_id,
                "job.failed",
                {
                    "stage": message.stage,
                    "source_queue": source_queue,
                    "target_queue": target_queue,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            self._cleanup_terminal_job(message.job_id)
        except Exception:
            pass

    def _default_batch_size(self) -> int:
        return int(self.settings.default_triton_batch_size)

    def _job_is_terminal(self, job_id: str) -> bool:
        try:
            return str(self.store.read(job_id).get("status")) in TERMINAL_STATUSES
        except Exception:
            return False

    def _cleanup_terminal_job(self, job_id: str) -> None:
        report = cleanup_terminal_job_runtime(job_id, settings=self.settings, job_dir=self.store.job_dir(job_id))
        if report.get("enabled"):
            try:
                self.store.replace_artifacts(job_id, prune_missing_local_artifacts(self.store.read(job_id).get("artifacts") or {}))
                self.store.update(job_id, cleanup=report)
            except Exception:
                pass


def _tile_by_id(plan, tile_id: str):
    for tile in plan.tiles:
        if tile.tile_id == tile_id:
            return tile
    raise KeyError(tile_id)


def _block_by_id(plan, block_id: str):
    for block in plan.blocks:
        if block.block_id == block_id:
            return block
    raise KeyError(block_id)


def _metrics_from_progress(progress: dict[str, Any]) -> dict[str, Any]:
    triton_batches = int(progress.get("triton_batches") or 0)
    first_at = progress.get("first_block_vectorized_at")
    last_tile = progress.get("last_tile_inferred_at")
    overlap = max(0.0, float(last_tile) - float(first_at)) if first_at and last_tile else None
    gpu_snapshot = gpu_util_snapshot()
    gpu_values = [float(item["gpu_util_pct"]) for item in gpu_snapshot if item.get("gpu_util_pct") is not None]
    metrics = {
        "tiles_total": int(progress.get("tiles_total") or 0),
        "tiles_done": int(progress.get("tiles_done") or 0),
        "triton_batches": triton_batches,
        "triton_batch_fill_ratio": (float(progress.get("triton_batch_fill_sum") or 0.0) / triton_batches if triton_batches else 0.0),
        "triton_request_duration_ms": (float(progress.get("triton_request_duration_ms_sum") or 0.0) / max(1, int(progress.get("triton_request_count") or 0))),
        "blocks_total": int(progress.get("blocks_total") or 0),
        "blocks_done": int(progress.get("blocks_done") or 0),
        "first_block_vectorized_at": first_at,
        "last_tile_inferred_at": last_tile,
        "streaming_overlap_sec": overlap,
        "infer_queue_depth": int(progress.get("infer_queue_depth") or 0),
        "infer_queue_target": int(progress.get("infer_queue_target") or 0),
        "infer_queue_low_watermark": int(progress.get("infer_queue_low_watermark") or 0),
        "infer_queue_high_watermark": int(progress.get("infer_queue_high_watermark") or 0),
        "spool_bytes": int(progress.get("spool_bytes") or 0),
        "preprocess_pauses_total": int(progress.get("preprocess_pauses_total") or 0),
        "preprocess_resumes_total": int(progress.get("preprocess_resumes_total") or 0),
        "gpu_util_snapshot_count": len(gpu_snapshot),
    }
    if gpu_values:
        metrics["gpu_utilization_mean"] = sum(gpu_values) / len(gpu_values)
        metrics["gpu_utilization_max"] = max(gpu_values)
    return metrics
