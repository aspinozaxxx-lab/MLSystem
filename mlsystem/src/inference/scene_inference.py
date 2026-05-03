from __future__ import annotations

import time
import threading
import os
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from shapely.geometry import box, mapping

from ..pipeline.contracts import ProbabilityMap, TileWindow
from ..pipeline_config import PipelineConfig
from ..preprocessing.normalization import normalize_image
from ..storage.s3 import aws_session, raster_path_for_s3_key
from ..tiling.debug_tiling import probability_nonzero_bbox
from ..tiling.windows import window_grid
from .probability_map import ProbabilityMapAccumulator, ProbabilityMapConfig
from .triton_client import TritonEndpoint, infer_segmentation_batch


@dataclass(frozen=True)
class SceneInferenceConfig:
    input_bands: list[int]
    patch_size: int
    stride: int
    threshold: float
    crop_mode: str = "full"
    stitch_mode: str = "hard_insert"
    center_size: int | None = None
    context_bounds: int | None = None
    full_scene: bool = True
    max_windows_per_scene: int | None = None
    batch_size: int = 1
    collect_debug_features: bool = False
    gpu_forward_concurrency: int = 1
    tile_read_workers: int = 1
    tile_prefetch_batches: int = 4
    triton_endpoint: TritonEndpoint | None = None


@dataclass
class SceneInferenceResult:
    scene_index: int
    scene_id: str
    scene_name: str
    probability_map: ProbabilityMap
    windows_preview_features: list[dict[str, Any]] = field(default_factory=list)
    tile_insert_features: list[dict[str, Any]] = field(default_factory=list)
    tile_insert_debug_rows: list[dict[str, Any]] = field(default_factory=list)
    debug_row: dict[str, Any] = field(default_factory=dict)
    scene_duration_sec: float = 0.0


def _predict_probability(model: torch.nn.Module, device: torch.device, arr: np.ndarray) -> tuple[np.ndarray, list[int]]:
    sample = torch.from_numpy(normalize_image(arr)[None, ...]).to(device)
    with torch.no_grad():
        logits = model(sample)
        model_output_shape = list(logits.shape)
        if tuple(logits.shape[-2:]) != tuple(arr.shape[-2:]):
            logits = torch.nn.functional.interpolate(logits, size=arr.shape[-2:], mode="bilinear", align_corners=False)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob, model_output_shape


def _predict_probability_batch(model: torch.nn.Module, device: torch.device, arrays: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    batch = np.stack([normalize_image(arr) for arr in arrays])
    sample = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        logits = model(sample)
        model_output_shape = list(logits.shape)
        if tuple(logits.shape[-2:]) != tuple(arrays[0].shape[-2:]):
            logits = torch.nn.functional.interpolate(logits, size=arrays[0].shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()
    return probs, model_output_shape


def _predict_probability_batch_triton(endpoint: TritonEndpoint, arrays: list[np.ndarray]) -> tuple[np.ndarray, list[int]]:
    batch = np.stack([normalize_image(arr) for arr in arrays]).astype(np.float32, copy=False)
    logits = infer_segmentation_batch(endpoint, batch)
    if tuple(logits.shape[-2:]) != tuple(arrays[0].shape[-2:]):
        raise RuntimeError(
            f"Triton output spatial shape {tuple(logits.shape[-2:])} does not match input {tuple(arrays[0].shape[-2:])}"
        )
    probs = 1.0 / (1.0 + np.exp(-logits[:, 0]))
    return probs.astype(np.float32, copy=False), list(logits.shape)


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        isinstance(exc, torch.cuda.OutOfMemoryError)
        or "out of memory" in text
        or "cudaerrormemoryallocation" in text
        or "cublas_status_alloc_failed" in text
        or "memory allocation" in text
    )


def _window_feature(ds: Any, window: Any, props: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Feature", "properties": props, "geometry": mapping(box(*ds.window_bounds(window)))}


class SceneInferenceRunner:
    def __init__(self, config: PipelineConfig, model: torch.nn.Module, device: torch.device, inference_config: SceneInferenceConfig) -> None:
        self.config = config
        self.model = model
        self.device = device
        self.inference_config = inference_config
        self._predict_guard = threading.BoundedSemaphore(max(1, int(inference_config.gpu_forward_concurrency)))

    def run_scene(self, scene_index: int, match: Any) -> SceneInferenceResult:
        import rasterio
        from rasterio.windows import Window

        scene_started = time.time()
        cfg = self.inference_config
        path: str | None = None
        windows_preview_features: list[dict[str, Any]] = []
        tile_insert_features: list[dict[str, Any]] = []
        tile_insert_debug_rows: list[dict[str, Any]] = []
        aws = aws_session(self.config)

        try:
            with rasterio.Env(aws, AWS_HTTPS="NO", AWS_VIRTUAL_HOSTING="FALSE", GDAL_CACHEMAX=64):
                path = raster_path_for_s3_key(self.config, match.key)
                with rasterio.open(path) as ds:
                    all_windows = window_grid(ds.width, ds.height, cfg.patch_size, cfg.stride, scene_id=match.name)
                    windows = all_windows if cfg.full_scene else all_windows[: max(1, int(cfg.max_windows_per_scene or len(all_windows)))]
                    accumulator = ProbabilityMapAccumulator(
                        match.name,
                        ProbabilityMapConfig(
                            scene_width=ds.width,
                            scene_height=ds.height,
                            patch_size=cfg.patch_size,
                            mode=cfg.stitch_mode,
                            crop_mode=cfg.crop_mode,
                            center_size=cfg.center_size,
                            context_bounds=cfg.context_bounds,
                        ),
                    )
                    skipped_reasons: dict[str, int] = {}
                    predicted_count = 0
                    batch_size = max(1, int(cfg.batch_size or 1))
                    pending: list[tuple[Any, Any, dict[str, Any], np.ndarray]] = []
                    reader_handles: list[Any] = []
                    reader_handles_lock = threading.Lock()
                    reader_local = threading.local()
    
                    def reader_dataset() -> Any:
                        handle = getattr(reader_local, "dataset", None)
                        if handle is None:
                            handle = rasterio.open(path, sharing=False)
                            reader_local.dataset = handle
                            with reader_handles_lock:
                                reader_handles.append(handle)
                        return handle
    
                    def close_reader_handles() -> None:
                        with reader_handles_lock:
                            handles = list(reader_handles)
                            reader_handles.clear()
                        for handle in handles:
                            try:
                                handle.close()
                            except Exception:
                                pass
    
                    def read_window(tile_index: int, tile_window: TileWindow) -> tuple[Any, Any, dict[str, Any], np.ndarray | None]:
                        raster_window = Window(tile_window.x, tile_window.y, cfg.patch_size, cfg.patch_size)
                        props = {
                            "scene": match.name,
                            "tile_index": tile_index,
                            "x": int(tile_window.x),
                            "y": int(tile_window.y),
                            "width": int(tile_window.width),
                            "height": int(tile_window.height),
                            "predicted": False,
                            "skipped_reason": "",
                        }
                        read_ds = reader_dataset()
                        arr = read_ds.read(cfg.input_bands, window=raster_window, boundless=True, fill_value=0)
                        return tile_window, raster_window, props, arr
    
                    def flush_pending() -> None:
                        nonlocal predicted_count
                        if not pending:
                            return
                        arrays = [item[3] for item in pending]
                        with self._predict_guard:
                            try:
                                if cfg.triton_endpoint is not None:
                                    probs, model_output_shape = _predict_probability_batch_triton(cfg.triton_endpoint, arrays)
                                else:
                                    probs, model_output_shape = _predict_probability_batch(self.model, self.device, arrays)
                                effective_batch_size = len(arrays)
                            except Exception as exc:
                                if not _is_cuda_oom(exc):
                                    raise
                                if self.device.type == "cuda":
                                    torch.cuda.empty_cache()
                                probs_rows: list[np.ndarray] = []
                                model_output_shape = []
                                fallback_batch_size = max(1, len(arrays) // 2)
                                start = 0
                                while start < len(arrays):
                                    chunk = arrays[start : start + fallback_batch_size]
                                    try:
                                        if cfg.triton_endpoint is not None:
                                            chunk_probs, model_output_shape = _predict_probability_batch_triton(cfg.triton_endpoint, chunk)
                                        else:
                                            chunk_probs, model_output_shape = _predict_probability_batch(self.model, self.device, chunk)
                                    except Exception as chunk_exc:
                                        if not _is_cuda_oom(chunk_exc):
                                            raise
                                        if self.device.type == "cuda":
                                            torch.cuda.empty_cache()
                                        if fallback_batch_size <= 1:
                                            raise
                                        fallback_batch_size = max(1, fallback_batch_size // 2)
                                        continue
                                    probs_rows.extend(list(chunk_probs))
                                    start += fallback_batch_size
                                probs = np.stack(probs_rows)
                                effective_batch_size = fallback_batch_size
                        for (tile_window_item, raster_window_item, props_item, arr_item), prob in zip(pending, probs):
                            insert = accumulator.add_tile(tile_window_item, prob)
                            predicted_count += 1
                            props_item["predicted"] = True
                            if cfg.collect_debug_features:
                                windows_preview_features.append(_window_feature(ds, raster_window_item, props_item))
                                insert_props = {
                                    **props_item,
                                    "model_input_shape": [
                                        len(pending),
                                        int(arr_item.shape[0]),
                                        int(arr_item.shape[1]),
                                        int(arr_item.shape[2]),
                                    ],
                                    "model_output_shape": model_output_shape,
                                    "inference_batch_size": batch_size,
                                    "effective_inference_batch_size": effective_batch_size,
                                    "crop_mode": cfg.crop_mode,
                                    "stitch_mode": cfg.stitch_mode,
                                    "center_size": cfg.center_size,
                                    "context_bounds": cfg.context_bounds,
                                    **insert,
                                    "expected_insert_bounds": [
                                        insert["insert_x"],
                                        insert["insert_y"],
                                        insert["insert_x"] + insert["insert_width"],
                                        insert["insert_y"] + insert["insert_height"],
                                    ],
                                    "actual_insert_bounds": [
                                        insert["insert_x"],
                                        insert["insert_y"],
                                        insert["insert_x"] + insert["insert_width"],
                                        insert["insert_y"] + insert["insert_height"],
                                    ],
                                    "predicted_nonzero_fraction": float(
                                        np.count_nonzero(prob[: tile_window_item.height, : tile_window_item.width] > cfg.threshold)
                                        / max(1, tile_window_item.width * tile_window_item.height)
                                    ),
                                }
                                insert_window = Window(insert["insert_x"], insert["insert_y"], insert["insert_width"], insert["insert_height"])
                                tile_insert_features.append(_window_feature(ds, insert_window, insert_props))
                                tile_insert_debug_rows.append(insert_props)
                        pending.clear()
    
                    def consume_read(item: tuple[Any, Any, dict[str, Any], np.ndarray | None]) -> None:
                        tile_window_item, raster_window_item, props_item, arr_item = item
                        if arr_item is None or np.count_nonzero(arr_item) == 0:
                            skipped_reasons["all_zero"] = skipped_reasons.get("all_zero", 0) + 1
                            props_item["skipped_reason"] = "all_zero"
                            if cfg.collect_debug_features:
                                windows_preview_features.append(_window_feature(ds, raster_window_item, props_item))
                            return
                        pending.append((tile_window_item, raster_window_item, props_item, arr_item))
                        if len(pending) >= batch_size:
                            flush_pending()
    
                    tile_read_workers = max(1, int(cfg.tile_read_workers or 1))
                    if tile_read_workers > 1 and len(windows) > 1:
                        from concurrent.futures import ThreadPoolExecutor
    
                        max_inflight = max(batch_size, batch_size * max(1, int(cfg.tile_prefetch_batches or 1)))
                        window_iter = iter(enumerate(windows))
                        futures: set[Future[tuple[Any, Any, dict[str, Any], np.ndarray | None]]] = set()
                        try:
                            with ThreadPoolExecutor(max_workers=tile_read_workers) as executor:
                                while True:
                                    while len(futures) < max_inflight:
                                        try:
                                            tile_index, tile_window = next(window_iter)
                                        except StopIteration:
                                            break
                                        futures.add(executor.submit(read_window, tile_index, tile_window))
                                    if not futures:
                                        break
                                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                                    for future in done:
                                        consume_read(future.result())
                                for future in futures:
                                    consume_read(future.result())
                        finally:
                            close_reader_handles()
                    else:
                        for tile_index, tile_window in enumerate(windows):
                            consume_read(read_window(tile_index, tile_window))
                        close_reader_handles()
                    flush_pending()
    
                    probability_map = accumulator.finalize(transform=ds.transform, crs=str(ds.crs))
                    debug_row = build_scene_debug_row(
                        match=match,
                        image_uri=f"s3://{self.config.storage.s3_bucket}/{match.key}",
                        width=ds.width,
                        height=ds.height,
                        crs=str(ds.crs),
                        transform=list(ds.transform)[:6],
                        tile_size=cfg.patch_size,
                        stride=cfg.stride,
                        expected_window_count=len(all_windows),
                        actual_window_count=len(windows),
                        predicted_count=predicted_count,
                        skipped_reasons=skipped_reasons,
                        probability_map=probability_map,
                        scene_started=scene_started,
                    )

        finally:
            if path and str(os.getenv("MLSYSTEM_PURGE_S3_CACHE_AFTER_SCENE") or "").lower() in {"1", "true", "yes", "on"}:
                try:
                    local_path = Path(path).resolve()
                    cache_root = Path("/data/mlsystem/cache/s3").resolve()
                    if local_path.is_file() and str(local_path).startswith(str(cache_root) + os.sep):
                        local_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return SceneInferenceResult(
            scene_index=scene_index,
            scene_id=match.name,
            scene_name=match.name,
            probability_map=probability_map,
            windows_preview_features=windows_preview_features,
            tile_insert_features=tile_insert_features,
            tile_insert_debug_rows=tile_insert_debug_rows,
            debug_row=debug_row,
            scene_duration_sec=round(time.time() - scene_started, 3),
        )


def build_scene_debug_row(
    *,
    match: Any,
    image_uri: str,
    width: int,
    height: int,
    crs: str,
    transform: list[Any],
    tile_size: int,
    stride: int,
    expected_window_count: int,
    actual_window_count: int,
    predicted_count: int,
    skipped_reasons: dict[str, int],
    probability_map: ProbabilityMap,
    scene_started: float,
) -> dict[str, Any]:
    return {
        "scene_id": match.name,
        "image_uri": image_uri,
        "width": width,
        "height": height,
        "crs": crs,
        "transform": transform,
        "tile_size": tile_size,
        "stride": stride,
        "expected_window_count": expected_window_count,
        "actual_window_count": actual_window_count,
        "actual_predicted_window_count": predicted_count,
        "skipped_window_count": actual_window_count - predicted_count,
        "skip_reasons": skipped_reasons,
        "probability_map_shape": [int(height), int(width)],
        "nonzero_probability_bbox_pixels": probability_nonzero_bbox(probability_map.prob),
        "nonzero_probability_area_fraction": float(np.count_nonzero(probability_map.prob > 1e-6) / max(1, width * height)),
        "image_area_bbox_pixels": [0, 0, int(width), int(height)],
        "coverage_fraction": probability_map.coverage_fraction,
        "vector_bounds": None,
        "image_bounds": None,
        "vector_area_fraction_of_image_bbox": None,
        "objects_before_filter": 0,
        "objects_after_filter": 0,
        "scene_duration_sec": round(time.time() - scene_started, 3),
    }


def run_synthetic_scene_inference(
    scene: np.ndarray,
    model: torch.nn.Module,
    *,
    patch_size: int,
    stride: int,
    threshold: float = 0.5,
    crop_mode: str = "full",
    stitch_mode: str = "weighted_overlap",
    center_size: int | None = None,
    context_bounds: int | None = None,
    device: torch.device | None = None,
) -> ProbabilityMap:
    from rasterio.transform import from_origin

    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    _, height, width = scene.shape
    accumulator = ProbabilityMapAccumulator(
        "synthetic",
        ProbabilityMapConfig(
            scene_width=width,
            scene_height=height,
            patch_size=patch_size,
            mode=stitch_mode,
            crop_mode=crop_mode,
            center_size=center_size,
            context_bounds=context_bounds,
        ),
    )
    for tile_window in window_grid(width, height, patch_size, stride, scene_id="synthetic"):
        tile = np.zeros((scene.shape[0], patch_size, patch_size), dtype=scene.dtype)
        actual = scene[:, tile_window.y : tile_window.y + tile_window.height, tile_window.x : tile_window.x + tile_window.width]
        tile[:, : tile_window.height, : tile_window.width] = actual
        prob, _shape = _predict_probability(model, device, tile)
        accumulator.add_tile(tile_window, prob)
    return accumulator.finalize(transform=from_origin(0, height, 1, 1), crs="EPSG:3857")


def run_scene_inference(runner: SceneInferenceRunner, match: Any, scene_index: int = 0) -> SceneInferenceResult:
    return runner.run_scene(scene_index, match)
