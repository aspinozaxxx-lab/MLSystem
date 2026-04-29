from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from ..pipeline.contracts import PostprocessResult
from ..storage.local_io import write_json

def build_feature_collection(name: str, features: list[dict[str, Any]], crs_name: str = "urn:ogc:def:crs:EPSG::3857") -> dict[str, Any]:
    return {"type": "FeatureCollection", "name": name, "crs": {"type": "name", "properties": {"name": crs_name}}, "features": features}


def write_geojson(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def write_geojson_gz(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return path


def export_pseudolabel_artifacts(
    *,
    experiment_dir: Path,
    job_id: str,
    postprocess_result: PostprocessResult,
    windows_preview_features: list[dict[str, Any]],
    tile_insert_features: list[dict[str, Any]],
    tiling_debug: dict[str, Any],
    segformer_debug: dict[str, Any],
    debug_mode: bool = False,
) -> dict[str, Any]:
    payload = build_feature_collection(f"{job_id}_accepted", postprocess_result.features)
    geojson_path = write_geojson(experiment_dir / f"{job_id}.accepted.geojson", payload)

    debug_geojson_path = None
    if debug_mode:
        debug_geojson_path = write_geojson(experiment_dir / "accepted_debug.geojson", {**payload, "name": f"{job_id}_accepted_debug"})

    gz_path = write_geojson_gz(experiment_dir / "accepted.geojson.gz", payload)
    windows_preview_path = experiment_dir / "windows_preview.geojson"
    write_json(windows_preview_path, {"type": "FeatureCollection", "name": f"{job_id}_windows_preview", "features": windows_preview_features})
    tile_insert_debug_path = experiment_dir / "tile_insert_debug.geojson"
    write_json(tile_insert_debug_path, {"type": "FeatureCollection", "name": f"{job_id}_tile_insert_debug", "features": tile_insert_features})
    tiling_debug_path = experiment_dir / "tiling_debug.json"
    write_json(tiling_debug_path, tiling_debug)
    segformer_debug_path = experiment_dir / "segformer_tiling_debug.json"
    write_json(segformer_debug_path, segformer_debug)

    artifacts = [geojson_path, tiling_debug_path, segformer_debug_path]
    if debug_geojson_path:
        artifacts.append(debug_geojson_path)

    gpkg_path = experiment_dir / "accepted.gpkg"
    gpkg_error_path = None
    try:
        import geopandas as gpd

        if postprocess_result.features:
            gdf = gpd.GeoDataFrame.from_features(postprocess_result.features, crs="EPSG:3857")
        else:
            gdf = gpd.GeoDataFrame({"scene": [], "threshold": [], "area_m2": []}, geometry=[], crs="EPSG:3857")
        gdf.to_file(gpkg_path, driver="GPKG")
    except Exception as exc:
        gpkg_error_path = experiment_dir / "pseudolabel_gpkg_error.txt"
        gpkg_error_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        artifacts.append(gpkg_error_path)

    return {
        "payload": payload,
        "geojson_path": geojson_path,
        "gz_path": gz_path,
        "gpkg_path": gpkg_path,
        "gpkg_error_path": gpkg_error_path,
        "windows_preview_path": windows_preview_path,
        "tile_insert_debug_path": tile_insert_debug_path,
        "tiling_debug_path": tiling_debug_path,
        "segformer_debug_path": segformer_debug_path,
        "debug_geojson_path": debug_geojson_path,
        "artifacts": artifacts,
    }
