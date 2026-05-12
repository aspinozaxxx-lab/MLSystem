from __future__ import annotations

import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any

from ..api.schemas import JobRequest, PseudolabelConfig
from ..planning.planner import ScenePlan
from ..storage.local_io import write_json
from ..vectorization.merge_vectors import merge_block_vectors


def merge_scene_blocks(
    *,
    job_dir: Path,
    plan: ScenePlan,
    request: JobRequest,
    block_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    vector_cfg = request.effective_vectorization()
    scene_out = job_dir / "scenes" / plan.scene_id
    accepted = scene_out / f"{plan.scene_id}.accepted.geojson"
    block_paths = [str(Path(block.vector_path)) for block in plan.blocks if Path(block.vector_path).exists()]
    merge_summary = merge_block_vectors(
        block_paths,
        output_geojson=accepted,
        final_min_area=float(vector_cfg.final_min_area or 0.0),
        merge_epsilon=float(vector_cfg.merge_epsilon or 0.0),
        crs_name=plan.crs,
    )
    summary = {
        "scene_id": plan.scene_id,
        "scene_name": plan.scene_name,
        "blocks_total": len(plan.blocks),
        "blocks_done": len(block_summaries),
        "block_summaries": block_summaries,
        **merge_summary,
    }
    write_json(scene_out / "scene_merge_summary.json", summary)
    return summary


def finalize_job_artifacts(
    *,
    job_id: str,
    job_dir: Path,
    request: JobRequest,
    plans: list[ScenePlan],
    scene_summaries: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, str]:
    started = time.time()
    run_dir = Path(request.run_dir) if request.run_dir else job_dir / "run_artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = request.experiment_id
    scene_vector_paths = [str(summary.get("accepted_geojson") or summary.get("accepted")) for summary in scene_summaries if summary.get("accepted_geojson") or summary.get("accepted")]
    accepted_geojson = run_dir / f"{experiment_id}.accepted.geojson"
    output_crs = _single_output_crs(plans)
    merge_summary = merge_block_vectors(
        scene_vector_paths,
        output_geojson=accepted_geojson,
        final_min_area=0.0,
        merge_epsilon=0.0,
        crs_name=output_crs,
    )
    accepted_gz = run_dir / "accepted.geojson.gz"
    with gzip.open(accepted_gz, "wt", encoding="utf-8") as fp:
        json.dump(json.loads(accepted_geojson.read_text(encoding="utf-8")), fp, ensure_ascii=False)

    coverage_report = {
        "schema_version": 1,
        "source": "inference_engine",
        "job_id": job_id,
        "scenes_processed": len(plans),
        "scenes_failed": 0,
        "total_expected_windows": sum(len(plan.tiles) for plan in plans),
        "total_predicted_windows": int(metrics.get("tiles_done") or 0),
        "total_skipped_windows": 0,
        "mean_coverage_fraction": 1.0 if plans else None,
        "min_coverage_fraction": 1.0 if plans else None,
        "accepted_objects_total": merge_summary.get("final_objects"),
        "accepted_geojson_mb": merge_summary.get("final_geojson_size_mb"),
        "streaming_overlap_sec": metrics.get("streaming_overlap_sec"),
        "first_block_vectorized_at": metrics.get("first_block_vectorized_at"),
        "last_tile_inferred_at": metrics.get("last_tile_inferred_at"),
    }
    write_json(run_dir / "coverage_report.json", coverage_report)

    scene_result_dir = run_dir / "pseudolabel_scene_results"
    scene_result_dir.mkdir(exist_ok=True)
    scene_rows = []
    for plan in plans:
        mosaic_npz = scene_result_dir / f"{plan.scene_id}.npz"
        if not mosaic_npz.exists():
            _write_placeholder_probability_mosaic(mosaic_npz, plan)
        meta_path = scene_result_dir / f"{plan.scene_id}.json"
        write_json(
            meta_path,
            {
                "scene_id": plan.scene_id,
                "scene_name": plan.scene_name,
                "npz_path": str(mosaic_npz),
                "transform": list(plan.transform),
                "crs": plan.crs,
                "coverage_fraction": 1.0,
                "source": "inference_engine",
            },
        )
        scene_rows.append(
            {
                "scene_id": plan.scene_id,
                "scene_name": plan.scene_name,
                "npz_path": str(mosaic_npz),
                "meta_path": str(meta_path),
                "coverage_fraction": 1.0,
            }
        )
    manifest = {
        "schema_version": 1,
        "source": "inference_engine",
        "scene_count": len(scene_rows),
        "scene_result_dir": str(scene_result_dir),
        "scenes": scene_rows,
    }
    write_json(run_dir / "pseudolabel_scene_results_manifest.json", manifest)
    write_json(run_dir / "probability_maps_index.json", manifest)
    (run_dir / "pseudolabel_scenes.txt").write_text("\n".join(plan.scene_name for plan in plans) + ("\n" if plans else ""), encoding="utf-8")

    vectorization_summary = {
        "source": "inference_engine",
        "mode": "rabbitmq_streaming_block_parallel",
        "accepted_geojson": str(accepted_geojson),
        "pseudolabel": {
            "accepted_objects": merge_summary.get("final_objects"),
            "accepted_geojson_mb": merge_summary.get("final_geojson_size_mb"),
        },
        "vectorization": {
            **metrics,
            "scenes": scene_summaries,
            "final_merge": merge_summary,
        },
    }
    postprocess_metrics = {
        "accepted_objects": merge_summary.get("final_objects"),
        "threshold_used": request.effective_vectorization().threshold,
        "min_object_area_m2_used": request.effective_vectorization().final_min_area,
        "simplify_tolerance_m_used": request.effective_vectorization().simplify_tolerance,
        "source": "inference_engine",
    }
    write_json(run_dir / "vectorization_summary.json", vectorization_summary)
    write_json(run_dir / "postprocess_summary.json", postprocess_metrics)
    write_json(
        run_dir / "pseudolabel_summary.json",
        {
            "status": "success",
            "source": "inference_engine",
            "metrics": postprocess_metrics,
            "coverage_report": coverage_report,
            "artifacts": [str(accepted_geojson), str(accepted_gz)],
        },
    )
    write_json(
        run_dir / "inference_results.json",
        {
            "source": "inference_engine",
            "job_id": job_id,
            "coverage": coverage_report,
            "pseudolabel": postprocess_metrics,
            "scene_results_manifest": str(run_dir / "pseudolabel_scene_results_manifest.json"),
            "probability_maps_index": str(run_dir / "probability_maps_index.json"),
        },
    )
    if not (run_dir / "prediction_examples.html").exists():
        (run_dir / "prediction_examples.html").write_text(
            "<!doctype html><html><body><h1>InferenceEngine pseudolabel</h1></body></html>",
            encoding="utf-8",
        )
    write_json(run_dir / "inference_timing_report.json", {"source": "inference_engine", **metrics, "finalize_duration_sec": round(time.time() - started, 3)})
    artifacts = {
        "accepted_geojson": str(accepted_geojson),
        "accepted.geojson.gz": str(accepted_gz),
        "coverage_report.json": str(run_dir / "coverage_report.json"),
        "pseudolabel_summary.json": str(run_dir / "pseudolabel_summary.json"),
        "postprocess_summary.json": str(run_dir / "postprocess_summary.json"),
        "vectorization_summary.json": str(run_dir / "vectorization_summary.json"),
        "pseudolabel_scene_results_manifest.json": str(run_dir / "pseudolabel_scene_results_manifest.json"),
        "probability_maps_index.json": str(run_dir / "probability_maps_index.json"),
        "inference_results.json": str(run_dir / "inference_results.json"),
        "prediction_examples.html": str(run_dir / "prediction_examples.html"),
        "run_dir": str(run_dir),
    }
    write_json(job_dir / "artifacts.json", artifacts)
    return artifacts


def _single_output_crs(plans: list[ScenePlan]) -> str | None:
    crs_values = {str(plan.crs) for plan in plans if plan.crs}
    if len(crs_values) == 1:
        return next(iter(crs_values))
    return None


def _write_placeholder_probability_mosaic(path: Path, plan: ScenePlan) -> None:
    import numpy as np

    # InferenceEngine keeps the real probability surface in per-tile artifacts.
    # The legacy manifest still needs a scene-level npz path, but downstream
    # stages are validate-only for source=inference_engine. Keep this artifact
    # compact so finalization does not write multi-GB zero mosaics for large
    # production scenes.
    prob = np.zeros((1, 1), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prob_uint8=prob, placeholder=np.array([1], dtype=np.uint8), source_width=np.array([plan.width]), source_height=np.array([plan.height]))
