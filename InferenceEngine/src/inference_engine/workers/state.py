from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..planning.planner import ScenePlan
from ..storage.local_io import read_json, write_json


@contextmanager
def file_lock(path: Path, *, timeout_sec: float = 60.0) -> Iterator[None]:
    lock_dir = path.with_suffix(path.suffix + ".lock")
    deadline = time.time() + timeout_sec
    while True:
        try:
            lock_dir.mkdir(parents=True)
            break
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timed out waiting for lock {lock_dir}")
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


class ProgressStore:
    def __init__(self, job_dir: Path) -> None:
        self.job_dir = job_dir
        self.path = job_dir / "progress.json"

    def read(self) -> dict[str, Any]:
        return read_json(self.path, default={}) or {}

    def update(self, mutator) -> dict[str, Any]:
        with file_lock(self.path):
            payload = self.read()
            result = mutator(payload)
            if result is not None:
                payload = result
            payload["updated_at_unix"] = time.time()
            write_json(self.path, payload)
            return payload


class SceneStateStore:
    def __init__(self, job_dir: Path, scene_id: str) -> None:
        self.job_dir = job_dir
        self.scene_id = scene_id
        self.path = job_dir / "scenes" / scene_id / "scene_state.json"

    def read(self) -> dict[str, Any]:
        return read_json(self.path, default={}) or {}

    def update(self, mutator) -> dict[str, Any]:
        with file_lock(self.path):
            payload = self.read()
            result = mutator(payload)
            if result is not None:
                payload = result
            payload["updated_at_unix"] = time.time()
            write_json(self.path, payload)
            return payload


def initialize_progress(job_dir: Path, *, job_id: str, scenes: list[dict[str, Any]], max_scenes_inflight: int) -> dict[str, Any]:
    progress = {
        "schema_version": 1,
        "job_id": job_id,
        "status": "running",
        "scenes": scenes,
        "scene_count": len(scenes),
        "next_scene_index": 0,
        "max_scenes_inflight": max(1, int(max_scenes_inflight or 1)),
        "active_scenes": [],
        "scene_plan_done": [],
        "scene_done": [],
        "tiles_total": 0,
        "tiles_done": 0,
        "blocks_total": 0,
        "blocks_done": 0,
        "triton_batches": 0,
        "triton_batch_fill_sum": 0.0,
        "triton_request_duration_ms_sum": 0.0,
        "triton_request_count": 0,
        "preprocess_pauses_total": 0,
        "preprocess_resumes_total": 0,
    }
    write_json(job_dir / "progress.json", progress)
    return progress


def initialize_scene_state(job_dir: Path, plan: ScenePlan) -> dict[str, Any]:
    remaining = {block.block_id: list(block.dependency_tile_ids) for block in plan.blocks}
    blocks_by_tile: dict[str, list[str]] = {}
    for block in plan.blocks:
        for tile_id in block.dependency_tile_ids:
            blocks_by_tile.setdefault(tile_id, []).append(block.block_id)
    state = {
        "schema_version": 1,
        "scene_id": plan.scene_id,
        "status": "planned",
        "next_tile_index": 0,
        "tiles_total": len(plan.tiles),
        "tiles_preprocess_published": 0,
        "tiles_done": [],
        "blocks_total": len(plan.blocks),
        "blocks_done": [],
        "remaining_by_block": remaining,
        "blocks_by_tile": blocks_by_tile,
        "scene_merge_published": False,
    }
    write_json(job_dir / "scenes" / plan.scene_id / "scene_state.json", state)
    return state


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
