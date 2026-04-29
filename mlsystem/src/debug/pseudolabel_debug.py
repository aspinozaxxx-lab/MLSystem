from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..inference.scene_inference import run_synthetic_scene_inference
from ..postprocessing.filtering import postprocess_vectorization_result
from ..postprocessing.pseudolabel_export import export_pseudolabel_artifacts
from ..postprocessing.vectorization import vectorize_probability_map
from ..storage.local_io import write_json


class ConstantProbabilityModel(torch.nn.Module):
    def __init__(self, logit: float = 2.0) -> None:
        super().__init__()
        self.logit = float(logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.shape[0], 1, x.shape[-2], x.shape[-1]), self.logit, dtype=x.dtype, device=x.device)


def write_probability_preview(prob_map: np.ndarray, scene_name: str, experiment_dir: Path, prefix: str = "probability_preview") -> Path | None:
    try:
        from PIL import Image
    except Exception:
        return None
    scale = max(1, int(max(prob_map.shape) / 1024))
    png = np.clip(prob_map[::scale, ::scale] * 255, 0, 255).astype("uint8")
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in scene_name)[:48] or "scene"
    path = experiment_dir / f"{prefix}_{safe_name}.png"
    Image.fromarray(png).save(path)
    return path


def build_postprocess_debug(postprocess_result: Any, *, geojson_mb: float, max_geojson_mb: float) -> dict[str, Any]:
    return {
        "threshold": postprocess_result.params.get("threshold"),
        "min_area_m2": postprocess_result.params.get("min_object_area_m2"),
        "simplify_tolerance": postprocess_result.params.get("simplify_tolerance_m"),
        "objects_before_filter": postprocess_result.objects_before_filter,
        "objects_after_filter": postprocess_result.objects_after_filter,
        "objects_after_top500": postprocess_result.objects_after_top,
        "geojson_size_mb": geojson_mb,
        "max_geojson_mb": max_geojson_mb,
        "warning": "geojson_size_exceeds_limit" if geojson_mb > max_geojson_mb else None,
    }


def run_synthetic_pseudolabel_smoke(output_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir or Path("results/reports/synthetic_pseudolabel_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = np.ones((4, 17, 13), dtype="float32")
    prob = run_synthetic_scene_inference(
        scene,
        ConstantProbabilityModel(logit=2.0),
        patch_size=6,
        stride=4,
        threshold=0.5,
        stitch_mode="weighted_overlap",
        context_bounds=2,
    )
    vectorized = vectorize_probability_map(prob, scene_name="synthetic", threshold=0.5)
    post = postprocess_vectorization_result(vectorized, min_area_m2=1.0, simplify_tolerance_m=0.0, max_objects=10)
    exported = export_pseudolabel_artifacts(
        experiment_dir=output_dir,
        job_id="synthetic_smoke",
        postprocess_result=post,
        windows_preview_features=[],
        tile_insert_features=[],
        tiling_debug={"schema_version": 1, "scenes": [{"coverage_fraction": prob.coverage_fraction}]},
        segformer_debug={"schema_version": 1, "tile_inserts": []},
    )
    preview = write_probability_preview(prob.prob, "synthetic", output_dir)
    summary = {
        "ok": True,
        "coverage_fraction": prob.coverage_fraction,
        "min_weight_sum": float(prob.weight_sum.min()),
        "accepted_objects": len(post.features),
        "accepted_geojson": str(exported["geojson_path"]),
        "preview": str(preview) if preview else None,
    }
    write_json(output_dir / "synthetic_smoke_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Local pseudolabel debug helpers")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.synthetic:
        print(json.dumps(run_synthetic_pseudolabel_smoke(Path(args.output_dir) if args.output_dir else None), ensure_ascii=False, indent=2, sort_keys=True))
    else:
        parser.error("Only --synthetic is available in the lightweight local debug entrypoint")


if __name__ == "__main__":
    main()
