from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import PseudolabelRunRequest


COMPATIBILITY_ARTIFACT_NAMES = [
    "accepted.geojson.gz",
    "coverage_report.json",
    "pseudolabel_summary.json",
    "postprocess_summary.json",
    "vectorization_summary.json",
    "pseudolabel_scene_results_manifest.json",
    "probability_maps_index.json",
    "inference_results.json",
    "inference_timing_report.json",
    "pseudolabel_scenes.txt",
    "prediction_examples.html",
]

DEFAULT_TILE_SIZE = 1024
DEFAULT_STRIDE = 768
DEFAULT_THRESHOLD = 0.5
DEFAULT_POLL_SEC = 10.0
DEFAULT_TIMEOUT_SEC = 24 * 3600.0


def build_inference_engine_payload(request: PseudolabelRunRequest, *, run_id: str, run_dir: Path) -> dict[str, Any]:
    scenes = [{"entry": scene, "name": scene} for scene in (request.scenes or [])]
    return {
        "run_id": run_id,
        "experiment_id": request.experiment_id,
        "class_name": request.class_name,
        "images_uri": request.images_uri,
        "layout_uri": request.layout_uri,
        "run_dir": str(run_dir),
        "scenes": scenes,
        "source": "inference_pipeline",
        "model": {
            "ref": request.model_ref,
            "model_ref": request.model_ref,
        },
        "preprocess": {
            "tile_size": DEFAULT_TILE_SIZE,
            "patch_size": DEFAULT_TILE_SIZE,
            "stride": DEFAULT_STRIDE,
            "stitch_mode": "weighted_overlap",
        },
        "pseudolabel": {
            "threshold": DEFAULT_THRESHOLD,
            "core_size_px": 4096,
            "halo_px": 512,
            "workers": 4,
            "local_min_area": 0,
            "final_min_area": 0,
            "merge_epsilon": 1.0,
            "simplify_tolerance": 0,
        },
        "resource": {
            "triton_batch_size": 8,
            "batches_ahead": 512,
            "max_preprocess_queue": 4096,
            "max_spool_bytes": 128 * 1024 * 1024 * 1024,
            "max_scenes_inflight": 32,
            "max_blocks_inflight": 32,
        },
        "compatibility_artifacts": list(COMPATIBILITY_ARTIFACT_NAMES),
    }
