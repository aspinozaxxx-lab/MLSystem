from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..api.schemas import JobRequest
from ..planning.planner import ScenePlan, TileDescriptor
from ..preprocessing.normalization import normalize_image
from ..storage.artifacts import checksum_matches, write_checksum
from ..storage.local_io import write_json
from ..storage.probability_artifacts import write_probability_uint8
from ..storage.raster_paths import rasterio_path_for_uri
from ..triton.client import TritonEndpoint, infer_segmentation_batch


def preprocess_tile_descriptor(
    *,
    tile: TileDescriptor,
    scene_plan: ScenePlan,
    request: JobRequest,
    spool_dir: Path,
) -> dict[str, Any]:
    spool_dir.mkdir(parents=True, exist_ok=True)
    out_path = spool_dir / f"{tile.tile_id}.npy"
    checksum_path = spool_dir / f"{tile.tile_id}.sha256"
    if checksum_matches(out_path, checksum_path):
        return {"tile_id": tile.tile_id, "spool_path": str(out_path), "checksum_path": str(checksum_path), "idempotent_hit": True}

    arr, timings = read_normalized_tile(tile, scene_plan, request)
    write_started = time.perf_counter()
    np.save(out_path, arr)
    timings["spool_write_ms"] = _elapsed_ms(write_started)
    checksum = write_checksum(out_path, checksum_path)
    return {"tile_id": tile.tile_id, "spool_path": str(out_path), "checksum_path": str(checksum_path), "checksum": checksum, "timings": timings}


def infer_tile_batch(
    *,
    descriptors: list[dict[str, Any]],
    tiles_by_id: dict[str, TileDescriptor],
    scene_plan: ScenePlan,
    request: JobRequest,
    endpoint: TritonEndpoint | None,
) -> list[dict[str, Any]]:
    started = time.time()
    if not descriptors:
        return []
    if _scene_has_synthetic_probability(scene_plan):
        outputs = []
        for descriptor in descriptors:
            tile = tiles_by_id[str(descriptor["tile_id"])]
            prob = _synthetic_probability(tile, scene_plan).astype(np.float32, copy=False)
            outputs.append(_persist_probability_tile(tile, scene_plan, prob, logits_shape=[1, 1, tile.height, tile.width]))
            _cleanup_spool_descriptor(descriptor)
        duration_ms = (time.time() - started) * 1000.0
        for item in outputs:
            item["triton_request_duration_ms"] = duration_ms
        return outputs

    outputs = []
    pending_arrays = []
    pending_tiles = []
    pending_descriptors = []
    for descriptor in descriptors:
        tile = tiles_by_id[str(descriptor["tile_id"])]
        if checksum_matches(Path(tile.artifact_path), Path(tile.checksum_path)):
            outputs.append({"tile_id": tile.tile_id, "artifact_path": tile.artifact_path, "meta_path": tile.meta_path, "idempotent_hit": True, "triton_request_duration_ms": 0.0})
            _cleanup_spool_descriptor(descriptor)
            continue
        loaded, spool_read_ms = _load_spooled_image_with_timing(descriptor)
        pending_arrays.append(loaded)
        descriptor.setdefault("timings", {})["spool_read_ms"] = spool_read_ms
        pending_tiles.append(tile)
        pending_descriptors.append(descriptor)
    if endpoint is None:
        endpoint = TritonEndpoint(
            url=request.storage.get("triton_url") or request.model.model_dump().get("triton_url") or "http://triton:8000",
            model_name=request.model.triton_model_name or request.model.model_name or "segformer_b2",
            model_version=request.model.triton_model_version,
        )
    if pending_arrays:
        stack_started = time.perf_counter()
        batch = np.stack(pending_arrays)
        batch_stack_ms = _elapsed_ms(stack_started)
        infer_started = time.perf_counter()
        logits = infer_segmentation_batch(endpoint, batch)
        triton_request_ms = _elapsed_ms(infer_started)
        sigmoid_started = time.perf_counter()
        probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        cpu_sigmoid_ms = _elapsed_ms(sigmoid_started)
        duration_ms = (time.time() - started) * 1000.0
        persist_started = time.perf_counter()
        for tile, prob, descriptor in zip(pending_tiles, probs, pending_descriptors, strict=True):
            row = _persist_probability_tile(tile, scene_plan, prob.astype(np.float32, copy=False), logits_shape=list(logits.shape))
            row["triton_request_duration_ms"] = duration_ms
            row["timings"] = _merged_timings(
                descriptor.get("timings"),
                {
                    "batch_stack_ms": batch_stack_ms,
                    "triton_request_ms": triton_request_ms,
                    "cpu_sigmoid_ms": cpu_sigmoid_ms,
                },
            )
            outputs.append(row)
            _cleanup_spool_descriptor(descriptor)
        persist_ms = _elapsed_ms(persist_started)
        _set_output_timing(outputs[-len(pending_descriptors) :], "probability_persist_ms", persist_ms)
    return outputs


def infer_tile_batch_multi_scene(
    *,
    descriptors: list[dict[str, Any]],
    tiles_by_scene: dict[str, dict[str, TileDescriptor]],
    scene_plans: dict[str, ScenePlan],
    request: JobRequest,
    endpoint: TritonEndpoint | None,
) -> list[dict[str, Any]]:
    started = time.time()
    if not descriptors:
        return []

    outputs: list[dict[str, Any]] = []
    pending_arrays = []
    pending_rows: list[tuple[TileDescriptor, ScenePlan, dict[str, Any]]] = []
    for descriptor in descriptors:
        scene_id = str(descriptor["scene_id"])
        tile = tiles_by_scene[scene_id][str(descriptor["tile_id"])]
        scene_plan = scene_plans[scene_id]
        if _scene_has_synthetic_probability(scene_plan):
            prob = _synthetic_probability(tile, scene_plan).astype(np.float32, copy=False)
            row = _persist_probability_tile(tile, scene_plan, prob, logits_shape=[1, 1, tile.height, tile.width])
            row["scene_id"] = scene_id
            outputs.append(row)
            _cleanup_spool_descriptor(descriptor)
            continue
        if checksum_matches(Path(tile.artifact_path), Path(tile.checksum_path)):
            outputs.append(
                {
                    "scene_id": scene_id,
                    "tile_id": tile.tile_id,
                    "artifact_path": tile.artifact_path,
                    "meta_path": tile.meta_path,
                    "idempotent_hit": True,
                    "triton_request_duration_ms": 0.0,
                }
            )
            _cleanup_spool_descriptor(descriptor)
            continue
        loaded, spool_read_ms = _load_spooled_image_with_timing(descriptor)
        pending_arrays.append(loaded)
        descriptor.setdefault("timings", {})["spool_read_ms"] = spool_read_ms
        pending_rows.append((tile, scene_plan, descriptor))

    if endpoint is None:
        endpoint = TritonEndpoint(
            url=request.storage.get("triton_url") or request.model.model_dump().get("triton_url") or "http://triton:8000",
            model_name=request.model.triton_model_name or request.model.model_name or "segformer_b2",
            model_version=request.model.triton_model_version,
        )
    if pending_arrays:
        stack_started = time.perf_counter()
        batch = np.stack(pending_arrays)
        batch_stack_ms = _elapsed_ms(stack_started)
        infer_started = time.perf_counter()
        logits = infer_segmentation_batch(endpoint, batch)
        triton_request_ms = _elapsed_ms(infer_started)
        sigmoid_started = time.perf_counter()
        probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        cpu_sigmoid_ms = _elapsed_ms(sigmoid_started)
        duration_ms = (time.time() - started) * 1000.0
        persist_started = time.perf_counter()
        for (tile, scene_plan, descriptor), prob in zip(pending_rows, probs, strict=True):
            row = _persist_probability_tile(tile, scene_plan, prob.astype(np.float32, copy=False), logits_shape=list(logits.shape))
            row["scene_id"] = str(descriptor["scene_id"])
            row["triton_request_duration_ms"] = duration_ms
            row["timings"] = _merged_timings(
                descriptor.get("timings"),
                {
                    "batch_stack_ms": batch_stack_ms,
                    "triton_request_ms": triton_request_ms,
                    "cpu_sigmoid_ms": cpu_sigmoid_ms,
                },
            )
            outputs.append(row)
            _cleanup_spool_descriptor(descriptor)
        persist_ms = _elapsed_ms(persist_started)
        _set_output_timing(outputs[-len(pending_rows) :], "probability_persist_ms", persist_ms)
    return outputs


def infer_prepared_tile_batch_multi_scene(
    *,
    prepared_rows: list[dict[str, Any]],
    request: JobRequest,
    endpoint: TritonEndpoint,
) -> list[dict[str, Any]]:
    if not prepared_rows:
        return []
    started = time.time()
    outputs: list[dict[str, Any]] = []
    pending_arrays: list[np.ndarray] = []
    pending_rows: list[dict[str, Any]] = []
    for row in prepared_rows:
        scene_plan = row["scene_plan"]
        tile = row["tile"]
        if _scene_has_synthetic_probability(scene_plan):
            prob = _synthetic_probability(tile, scene_plan).astype(np.float32, copy=False)
            output = _persist_probability_tile(tile, scene_plan, prob, logits_shape=[1, 1, tile.height, tile.width])
            output["scene_id"] = str(row["scene_id"])
            output["timings"] = dict(row.get("timings") or {})
            outputs.append(output)
            continue
        if checksum_matches(Path(tile.artifact_path), Path(tile.checksum_path)):
            outputs.append(
                {
                    "scene_id": str(row["scene_id"]),
                    "tile_id": tile.tile_id,
                    "artifact_path": tile.artifact_path,
                    "meta_path": tile.meta_path,
                    "idempotent_hit": True,
                    "triton_request_duration_ms": 0.0,
                    "timings": dict(row.get("timings") or {}),
                }
            )
            continue
        pending_arrays.append(np.asarray(row["array"], dtype=np.float32))
        pending_rows.append(row)
    if pending_arrays:
        stack_started = time.perf_counter()
        batch = np.stack(pending_arrays)
        batch_stack_ms = _elapsed_ms(stack_started)
        infer_started = time.perf_counter()
        logits = infer_segmentation_batch(endpoint, batch)
        triton_request_ms = _elapsed_ms(infer_started)
        sigmoid_started = time.perf_counter()
        probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        cpu_sigmoid_ms = _elapsed_ms(sigmoid_started)
        duration_ms = (time.time() - started) * 1000.0
        persist_started = time.perf_counter()
        for row, prob in zip(pending_rows, probs, strict=True):
            tile = row["tile"]
            scene_plan = row["scene_plan"]
            output = _persist_probability_tile(tile, scene_plan, prob.astype(np.float32, copy=False), logits_shape=list(logits.shape))
            output["scene_id"] = str(row["scene_id"])
            output["triton_request_duration_ms"] = duration_ms
            output["timings"] = _merged_timings(
                row.get("timings"),
                {
                    "batch_stack_ms": batch_stack_ms,
                    "triton_request_ms": triton_request_ms,
                    "cpu_sigmoid_ms": cpu_sigmoid_ms,
                },
            )
            outputs.append(output)
        persist_ms = _elapsed_ms(persist_started)
        _set_output_timing(outputs[-len(pending_rows) :], "probability_persist_ms", persist_ms)
    return outputs


def _persist_probability_tile(tile: TileDescriptor, scene_plan: ScenePlan, prob: np.ndarray, *, logits_shape: list[int]) -> dict[str, Any]:
    out_path = Path(tile.artifact_path)
    checksum_path = Path(tile.checksum_path)
    if checksum_matches(out_path, checksum_path):
        return {"tile_id": tile.tile_id, "artifact_path": str(out_path), "meta_path": tile.meta_path, "idempotent_hit": True}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prob = prob[: tile.height, : tile.width]
    prob_uint8 = np.rint(np.clip(prob, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    write_probability_uint8(out_path, prob_uint8)
    meta = {
        "schema_version": 1,
        "tile_id": tile.tile_id,
        "scene_id": tile.scene_id,
        "scene_name": scene_plan.scene_name,
        "x": tile.x,
        "y": tile.y,
        "width": tile.width,
        "height": tile.height,
        "insert": tile.insert,
        "npz_path": str(out_path),
        "probability_band": "prob_uint8",
        "probability_storage": "npy_uint8",
        "transform": list(scene_plan.transform),
        "crs": scene_plan.crs,
        "logits_shape": logits_shape,
    }
    write_json(Path(tile.meta_path), meta)
    checksum = write_checksum(out_path, checksum_path)
    return {"tile_id": tile.tile_id, "artifact_path": str(out_path), "meta_path": tile.meta_path, "checksum": checksum}


def _load_spooled_image(descriptor: dict[str, Any]) -> np.ndarray:
    path = str(descriptor["spool_path"])
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            return payload["image"].astype(np.float32, copy=False)
        finally:
            payload.close()
    return np.asarray(payload, dtype=np.float32)


def _load_spooled_image_with_timing(descriptor: dict[str, Any]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    return _load_spooled_image(descriptor), _elapsed_ms(started)


def read_normalized_tile(tile: TileDescriptor, scene_plan: ScenePlan, request: JobRequest) -> tuple[np.ndarray, dict[str, float]]:
    read_started = time.perf_counter()
    arr = _read_or_synthetic_tile(tile, scene_plan, request)
    read_ms = _elapsed_ms(read_started)
    normalize_started = time.perf_counter()
    arr = normalize_image(arr).astype(np.float32, copy=False)
    normalize_ms = _elapsed_ms(normalize_started)
    return arr, {"input_read_ms": read_ms, "normalize_ms": normalize_ms}


def _read_or_synthetic_tile(tile: TileDescriptor, scene_plan: ScenePlan, request: JobRequest) -> np.ndarray:
    if _scene_has_synthetic_probability(scene_plan):
        prob = _synthetic_probability(tile, scene_plan)
        bands = max(1, len(request.preprocess.input_bands))
        return np.repeat(prob[None, :, :], bands, axis=0).astype(np.float32, copy=False)
    source = scene_plan.source or {}
    path = source.get("path") or source.get("uri") or source.get("image_uri")
    if not path:
        raise ValueError(f"Scene {scene_plan.scene_id} has no raster path and no synthetic probability")
    import rasterio
    from rasterio.windows import Window

    path = rasterio_path_for_uri(str(path))
    try:
        with rasterio.Env(**_rasterio_env_kwargs()):
            with rasterio.open(str(path)) as ds:
                return ds.read(request.preprocess.input_bands, window=Window(tile.x, tile.y, tile.patch_size, tile.patch_size), boundless=True, fill_value=0)
    except Exception as exc:
        raise RuntimeError(f"Failed to read raster for scene={scene_plan.scene_id} tile={tile.tile_id} path={path}: {type(exc).__name__}: {exc}") from exc


def _rasterio_env_kwargs() -> dict[str, str | int]:
    endpoint = os.getenv("AWS_S3_ENDPOINT") or os.getenv("MLFLOW_S3_ENDPOINT_URL") or os.getenv("AWS_ENDPOINT_URL")
    kwargs: dict[str, str | int] = {"AWS_HTTPS": "NO", "AWS_VIRTUAL_HOSTING": "FALSE", "GDAL_CACHEMAX": 128}
    if endpoint:
        endpoint = endpoint.replace("http://", "").replace("https://", "").rstrip("/")
        kwargs["AWS_S3_ENDPOINT"] = endpoint
    return kwargs


def _cleanup_spool_descriptor(descriptor: dict[str, Any]) -> None:
    for key in ("spool_path", "checksum_path"):
        value = descriptor.get(key)
        if not value:
            continue
        try:
            Path(str(value)).unlink(missing_ok=True)
        except OSError:
            pass


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _merged_timings(*items: Any) -> dict[str, float]:
    merged: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, (int, float)):
                merged[str(key)] = float(merged.get(str(key), 0.0)) + float(value)
    return merged


def _set_output_timing(outputs: list[dict[str, Any]], key: str, value: float) -> None:
    if not outputs:
        return
    per_output = float(value) / max(1, len(outputs))
    for output in outputs:
        output.setdefault("timings", {})[key] = per_output


def _scene_has_synthetic_probability(scene_plan: ScenePlan) -> bool:
    return bool((scene_plan.source or {}).get("probability_rects"))


def _synthetic_probability(tile: TileDescriptor, scene_plan: ScenePlan) -> np.ndarray:
    prob = np.zeros((tile.height, tile.width), dtype=np.float32)
    for rect in (scene_plan.source or {}).get("probability_rects") or []:
        if len(rect) < 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in rect[:4]]
        value = float(rect[4]) if len(rect) > 4 else 1.0
        ix0 = max(tile.x, x0)
        iy0 = max(tile.y, y0)
        ix1 = min(tile.x + tile.width, x1)
        iy1 = min(tile.y + tile.height, y1)
        if ix0 < ix1 and iy0 < iy1:
            prob[iy0 - tile.y : iy1 - tile.y, ix0 - tile.x : ix1 - tile.x] = value
    return prob


def descriptor_from_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
