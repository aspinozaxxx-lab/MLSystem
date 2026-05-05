from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any

from ..storage.local_io import write_json
from .block_grid import build_vectorization_plan, write_block_tile_intersections, write_processing_blocks_geojson
from .block_vectorizer import vectorize_block
from .contracts import BlockVectorizationJob, BlockVectorizationResult
from .merge_vectors import merge_block_vectors
from .tile_index import build_prediction_tile_index


def run_block_parallel_vectorization(
    *,
    run_id: str,
    manifest_path: str | Path,
    output_dir: str | Path,
    accepted_geojson: str | Path,
    threshold: float,
    class_name: str,
    core_size_px: int = 4096,
    halo_px: int = 512,
    workers_requested: int = 30,
    memory_guard_enabled: bool = True,
    max_worker_memory_mb: int | None = None,
    local_min_area: float = 0.0,
    final_min_area: float = 0.0,
    merge_epsilon: float = 1.0,
    bad_block_policy: str = "fail",
) -> dict[str, Any]:
    started = time.time()
    out = Path(output_dir)
    blocks_dir = out / "blocks"
    out.mkdir(parents=True, exist_ok=True)
    blocks_dir.mkdir(parents=True, exist_ok=True)
    tiles = build_prediction_tile_index(manifest_path, out)
    memory_guard = _memory_guard(core_size_px, halo_px, workers_requested, memory_guard_enabled, max_worker_memory_mb)
    plan = build_vectorization_plan(
        tiles,
        core_size_px=core_size_px,
        halo_px=halo_px,
        workers_requested=workers_requested,
        workers_effective=memory_guard["workers_effective"],
        memory_guard=memory_guard,
    )
    write_json(out / "vectorization_plan.json", plan.to_dict())
    (out / "vectorization_plan.txt").write_text(_plan_text(plan.to_dict()), encoding="utf-8")
    write_processing_blocks_geojson(out / "processing_blocks.geojson", plan.blocks)
    write_block_tile_intersections(out / "block_tile_intersections.json", plan.blocks)

    jobs = [
        BlockVectorizationJob(
            run_id=run_id,
            block=block,
            tiles=tiles,
            threshold=float(threshold),
            local_min_area=float(local_min_area or 0.0),
            output_dir=str(blocks_dir),
            class_name=class_name,
        )
        for block in plan.blocks
    ]
    results = _run_jobs(jobs, workers_effective=memory_guard["workers_effective"])
    failed = [result for result in results if result.status != "success"]
    if failed:
        (out / "failed_blocks.txt").write_text("\n".join(result.block_id for result in failed) + "\n", encoding="utf-8")
        if str(bad_block_policy).lower() != "skip":
            write_json(out / "block_results.json", {"results": [result.to_dict() for result in results]})
            raise RuntimeError(f"block_parallel vectorization failed for {len(failed)} blocks; see {out / 'failed_blocks.txt'}")
    else:
        (out / "failed_blocks.txt").write_text("", encoding="utf-8")
    block_vector_paths = [str(result.output_vector_path) for result in results if result.status == "success" and result.output_vector_path]
    merge_summary = merge_block_vectors(
        block_vector_paths,
        output_geojson=accepted_geojson,
        final_min_area=float(final_min_area or 0.0),
        merge_epsilon=float(merge_epsilon or 0.0),
    )
    duration_sec = round(time.time() - started, 3)
    summary = {
        "mode": "block_parallel",
        "tiles_total": len(tiles),
        "prediction_tiles": len(tiles),
        "prediction_scenes": len({tile.scene_id for tile in tiles}),
        "blocks_total": len(results),
        "blocks_done": len([result for result in results if result.status == "success"]),
        "blocks_failed": len(failed),
        "workers_requested": int(workers_requested),
        "workers_effective": int(memory_guard["workers_effective"]),
        "memory_guard": memory_guard,
        "core_size_px": int(core_size_px),
        "halo_px": int(halo_px),
        "boundary_candidates_count": int(sum(result.boundary_candidates_count for result in results)),
        "polygons_before_merge": int(sum(result.polygons_after_clip for result in results)),
        "polygons_after_merge": merge_summary.get("total_polygons_after_final_filter"),
        "final_objects": merge_summary.get("final_objects"),
        "accepted_objects": merge_summary.get("final_objects"),
        "final_geojson_size_mb": merge_summary.get("final_geojson_size_mb"),
        "vectorization_duration_sec": duration_sec,
        "merge_duration_sec": merge_summary.get("merge_duration_sec"),
        "accepted_geojson": str(accepted_geojson),
    }
    write_json(out / "block_results.json", {"results": [result.to_dict() for result in results]})
    (out / "block_results.txt").write_text(_results_text(results), encoding="utf-8")
    write_json(out / "vectorization_summary.json", summary)
    (out / "vectorization_summary.txt").write_text(_summary_text(summary), encoding="utf-8")
    return summary


def _run_jobs(jobs: list[BlockVectorizationJob], *, workers_effective: int) -> list[BlockVectorizationResult]:
    if workers_effective <= 1 or len(jobs) <= 1:
        return [vectorize_block(job) for job in jobs]
    results: list[BlockVectorizationResult] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_effective) as executor:
        future_map = {executor.submit(vectorize_block, job): job.block.block_id for job in jobs}
        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())
    return sorted(results, key=lambda result: result.block_id)


def _memory_guard(core_size_px: int, halo_px: int, requested: int, enabled: bool, max_worker_memory_mb: int | None) -> dict[str, Any]:
    cpu_guard = max(1, (os.cpu_count() or 1) - 2)
    expanded = core_size_px + 2 * halo_px
    estimated_mb = max(1, int((expanded * expanded * (4 + 4 + 1 + 4)) / (1024 * 1024)))
    memory_workers = requested
    if enabled and max_worker_memory_mb:
        memory_workers = max(1, int(max_worker_memory_mb / estimated_mb))
    effective = max(1, min(int(requested), int(cpu_guard), int(memory_workers)))
    return {
        "enabled": bool(enabled),
        "estimated_worker_memory_mb": estimated_mb,
        "max_worker_memory_mb": max_worker_memory_mb,
        "cpu_guard_workers": cpu_guard,
        "memory_guard_workers": memory_workers,
        "workers_effective": effective,
        "reduced": effective < requested,
    }


def _plan_text(plan: dict[str, Any]) -> str:
    lines = ["Vectorization plan"]
    for key in ("tiles_total", "blocks_total", "crs", "bounds", "core_size_px", "halo_px", "workers_requested", "workers_effective"):
        lines.append(f"{key}={plan.get(key)}")
    return "\n".join(lines) + "\n"


def _results_text(results: list[BlockVectorizationResult]) -> str:
    lines = ["block_id\tstatus\tpolygons_after_clip\tboundary_candidates\tduration_sec"]
    for result in results:
        lines.append(f"{result.block_id}\t{result.status}\t{result.polygons_after_clip}\t{result.boundary_candidates_count}\t{result.duration_sec}")
    return "\n".join(lines) + "\n"


def _summary_text(summary: dict[str, Any]) -> str:
    return "\n".join(f"{key}={summary[key]}" for key in sorted(summary)) + "\n"
