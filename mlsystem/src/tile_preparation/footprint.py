from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from affine import Affine
from rasterio.features import rasterize, shapes
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import box, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union
from shapely import wkt

from .records import TileWindow
from .windows import generate_window_grid_for_scene


@dataclass
class SceneFootprint:
    scene_id: str
    image_path: str
    raster_width: int
    raster_height: int
    raster_crs: str | None
    source: str
    polygon_raster_crs: Any
    polygon_pixel: Any
    bounds_pixel: tuple[int, int, int, int]
    area_pixels_estimated: float
    valid_pixel_share_estimated: float
    warnings: list[str] = field(default_factory=list)
    build_sec: float = 0.0
    raster_transform: tuple[float, float, float, float, float, float] | None = None

    @property
    def is_empty(self) -> bool:
        try:
            return bool(self.polygon_pixel.is_empty)
        except Exception:  # noqa: BLE001
            return True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "image_path": self.image_path,
            "raster_width": int(self.raster_width),
            "raster_height": int(self.raster_height),
            "raster_crs": self.raster_crs,
            "source": self.source,
            "polygon_raster_crs_wkt": self.polygon_raster_crs.wkt if self.polygon_raster_crs is not None else None,
            "polygon_pixel_wkt": self.polygon_pixel.wkt,
            "bounds_pixel": list(self.bounds_pixel),
            "area_pixels_estimated": float(self.area_pixels_estimated),
            "valid_pixel_share_estimated": float(self.valid_pixel_share_estimated),
            "warnings": list(self.warnings),
            "build_sec": float(self.build_sec),
            "raster_transform": list(self.raster_transform) if self.raster_transform is not None else None,
        }

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "SceneFootprint":
        polygon_pixel = wkt.loads(str(payload["polygon_pixel_wkt"]))
        polygon_raster_text = payload.get("polygon_raster_crs_wkt")
        return cls(
            scene_id=str(payload.get("scene_id") or ""),
            image_path=str(payload.get("image_path") or ""),
            raster_width=int(payload.get("raster_width") or 0),
            raster_height=int(payload.get("raster_height") or 0),
            raster_crs=payload.get("raster_crs"),
            source=str(payload.get("source") or "metadata"),
            polygon_raster_crs=wkt.loads(str(polygon_raster_text)) if polygon_raster_text else None,
            polygon_pixel=polygon_pixel,
            bounds_pixel=tuple(int(value) for value in payload.get("bounds_pixel", (0, 0, 0, 0))),
            area_pixels_estimated=float(payload.get("area_pixels_estimated") or 0.0),
            valid_pixel_share_estimated=float(payload.get("valid_pixel_share_estimated") or 0.0),
            warnings=list(payload.get("warnings") or []),
            build_sec=float(payload.get("build_sec") or 0.0),
            raster_transform=_metadata_affine_tuple(payload.get("raster_transform")),
        )


@dataclass
class WindowFootprintInfo:
    intersects: bool
    fully_inside: bool
    valid_share_estimated: float
    is_boundary: bool


def build_scene_footprint(ds: Any, config: Any, *, scene_id: str | None = None, image_path: str | None = None) -> SceneFootprint:
    started = time.perf_counter()
    scene_id = str(scene_id or getattr(ds, "name", "") or "")
    image_path = str(image_path or getattr(ds, "name", "") or "")
    warnings: list[str] = []
    try:
        mask, source, source_warnings, pixel_transform = _read_footprint_mask(ds, str(getattr(config, "valid_pixel_mode", "auto") or "auto"))
        warnings.extend(source_warnings)
        if not np.any(mask):
            footprint = _empty_footprint(ds, scene_id, image_path, source=source, warnings=warnings)
        elif bool(np.all(mask)):
            footprint = _full_footprint(ds, scene_id, image_path, source=source, warnings=warnings)
        else:
            footprint = _mask_to_footprint(ds, mask, pixel_transform, scene_id, image_path, source=source, warnings=warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"footprint fallback to full raster: {type(exc).__name__}: {exc}")
        footprint = _full_footprint(ds, scene_id, image_path, source="full_raster_fallback", warnings=warnings)
    footprint.build_sec = time.perf_counter() - started
    return footprint


def window_footprint_intersection(window: Any, footprint: SceneFootprint) -> WindowFootprintInfo:
    if footprint.is_empty:
        return WindowFootprintInfo(False, False, 0.0, False)
    window_poly = box(int(window.x), int(window.y), int(window.x) + int(window.width), int(window.y) + int(window.height))
    try:
        if not footprint.polygon_pixel.intersects(window_poly):
            return WindowFootprintInfo(False, False, 0.0, False)
        fully_inside = bool(footprint.polygon_pixel.covers(window_poly))
        if fully_inside:
            return WindowFootprintInfo(True, True, 1.0, False)
        intersection_area = float(footprint.polygon_pixel.intersection(window_poly).area)
    except Exception:  # noqa: BLE001
        return WindowFootprintInfo(True, False, 1.0, True)
    window_area = max(1.0, float(int(window.width) * int(window.height)))
    share = max(0.0, min(1.0, intersection_area / window_area))
    return WindowFootprintInfo(share > 0.0, share >= 0.999, share, share < 0.999)


def rasterize_footprint_for_window(footprint: SceneFootprint, window: Any) -> np.ndarray:
    info = window_footprint_intersection(window, footprint)
    shape_hw = (int(window.height), int(window.width))
    if not info.intersects:
        return np.zeros(shape_hw, dtype="uint8")
    if info.fully_inside:
        return np.ones(shape_hw, dtype="uint8")
    transform = Affine.translation(float(window.x), float(window.y))
    return rasterize(
        [(footprint.polygon_pixel, 1)],
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype("uint8")


def window_polygon_pixel(window: Any) -> Any:
    return box(int(window.x), int(window.y), int(window.x) + int(window.width), int(window.y) + int(window.height))


def footprint_affine(footprint: SceneFootprint) -> Affine | None:
    if footprint.raster_transform is None:
        return None
    return Affine(*footprint.raster_transform)


def pixel_geometry_to_raster_crs(footprint: SceneFootprint, geometry: Any) -> Any | None:
    affine = footprint_affine(footprint)
    if affine is None or geometry is None:
        return None
    return shapely_transform(lambda x, y, z=None: affine * (x, y), geometry)


def window_polygon_raster_crs(footprint: SceneFootprint, window: Any) -> Any | None:
    return pixel_geometry_to_raster_crs(footprint, window_polygon_pixel(window))


def generate_windows_for_footprint(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    footprint: SceneFootprint,
    scene_id: str,
) -> list[TileWindow]:
    windows: list[TileWindow] = []
    for window in generate_window_grid_for_scene(width, height, tile_size, stride, scene_id=scene_id):
        if window_footprint_intersection(window, footprint).intersects:
            windows.append(
                TileWindow(
                    scene_id=window.scene_id,
                    x=window.x,
                    y=window.y,
                    width=window.width,
                    height=window.height,
                    tile_size=window.tile_size,
                    stride=window.stride,
                    index=len(windows),
                )
            )
    return windows


def footprint_window_counters(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    footprint: SceneFootprint,
    scene_id: str,
) -> dict[str, int]:
    rectangular = generate_window_grid_for_scene(width, height, tile_size, stride, scene_id=scene_id)
    intersecting = 0
    fully_inside = 0
    boundary = 0
    for window in rectangular:
        info = window_footprint_intersection(window, footprint)
        if not info.intersects:
            continue
        intersecting += 1
        if info.fully_inside:
            fully_inside += 1
        else:
            boundary += 1
    return {
        "candidate_windows_rectangular": int(len(rectangular)),
        "windows_intersecting_footprint": int(intersecting),
        "skipped_outside_footprint": int(len(rectangular) - intersecting),
        "fully_inside_footprint_windows": int(fully_inside),
        "boundary_footprint_windows": int(boundary),
    }


def footprint_record_metadata(footprint: SceneFootprint, info: WindowFootprintInfo) -> dict[str, Any]:
    return {
        "footprint_fully_inside": bool(info.fully_inside),
        "footprint_valid_share_estimated": float(info.valid_share_estimated),
        "footprint_source": footprint.source,
        "footprint_boundary": bool(info.is_boundary),
        "footprint_valid_mask_mode": "full" if info.fully_inside else "footprint_boundary",
    }


def _read_footprint_mask(ds: Any, mode: str) -> tuple[np.ndarray, str, list[str], Affine]:
    mode = str(mode or "auto").lower()
    warnings: list[str] = []
    out_shape, pixel_transform = _footprint_out_shape_and_transform(ds)
    if mode == "dataset_mask":
        return _dataset_mask(ds, out_shape), "dataset_mask", warnings, pixel_transform
    if mode == "alpha":
        return _alpha_mask(ds, out_shape), "alpha", warnings, pixel_transform
    if mode == "nodata":
        return _nodata_mask(ds, out_shape), "nodata", warnings, pixel_transform
    if mode == "nonzero_any":
        return _nonzero_mask(ds, out_shape, any_band=True), "nonzero_any", warnings, pixel_transform
    if mode == "nonzero_all":
        return _nonzero_mask(ds, out_shape, any_band=False), "nonzero_all", warnings, pixel_transform
    if mode != "auto":
        raise ValueError(f"Unsupported valid pixel mode: {mode}")

    for source, fn in (
        ("dataset_mask", _dataset_mask),
        ("alpha", _alpha_mask),
        ("nodata", _nodata_mask),
    ):
        try:
            mask = fn(ds, out_shape)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"footprint {source} unavailable: {type(exc).__name__}: {exc}")
            continue
        if np.any(mask) and not np.all(mask):
            return mask, source, warnings, pixel_transform
        if np.all(mask):
            warnings.append(f"footprint {source} covers full raster; trying nonzero_any fallback")
    return _nonzero_mask(ds, out_shape, any_band=True), "nonzero_any", warnings, pixel_transform


def _dataset_mask(ds: Any, out_shape: tuple[int, int]) -> np.ndarray:
    return (ds.dataset_mask(out_shape=out_shape) > 0).astype("uint8")


def _alpha_mask(ds: Any, out_shape: tuple[int, int]) -> np.ndarray:
    alpha_bands = _alpha_band_indexes(ds)
    if not alpha_bands:
        raise ValueError("dataset has no alpha band")
    band = alpha_bands[0]
    return (ds.read(band, out_shape=out_shape, resampling=Resampling.nearest) > 0).astype("uint8")


def _nodata_mask(ds: Any, out_shape: tuple[int, int]) -> np.ndarray:
    masks = ds.read_masks(out_shape=(int(ds.count), int(out_shape[0]), int(out_shape[1])), resampling=Resampling.nearest)
    if masks.size == 0:
        raise ValueError("dataset has no masks")
    return np.all(masks > 0, axis=0).astype("uint8")


def _nonzero_mask(ds: Any, out_shape: tuple[int, int], *, any_band: bool) -> np.ndarray:
    arr = ds.read(out_shape=(int(ds.count), int(out_shape[0]), int(out_shape[1])), resampling=Resampling.nearest)
    if arr.size == 0:
        return np.zeros(out_shape, dtype="uint8")
    if any_band:
        return np.any(arr != 0, axis=0).astype("uint8")
    return np.all(arr != 0, axis=0).astype("uint8")


def _footprint_out_shape_and_transform(ds: Any) -> tuple[tuple[int, int], Affine]:
    max_dim = max(1, int(float(os.getenv("MLSYSTEM_TILE_FOOTPRINT_MAX_DIM", "1024"))))
    width = int(ds.width)
    height = int(ds.height)
    scale = max(1.0, max(width, height) / float(max_dim))
    out_width = max(1, int(math.ceil(width / scale)))
    out_height = max(1, int(math.ceil(height / scale)))
    pixel_transform = Affine.scale(width / float(out_width), height / float(out_height))
    return (out_height, out_width), pixel_transform


def _mask_to_footprint(ds: Any, mask: np.ndarray, pixel_transform: Affine, scene_id: str, image_path: str, *, source: str, warnings: list[str]) -> SceneFootprint:
    pixel_geoms = []
    raster_geoms = []
    for geom, value in shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=pixel_transform):
        if int(value) == 1:
            pixel_geoms.append(shape(geom))
    raster_transform = ds.transform * pixel_transform
    for geom, value in shapes(mask.astype("uint8"), mask=mask.astype(bool), transform=raster_transform):
        if int(value) == 1:
            raster_geoms.append(shape(geom))
    polygon_pixel = _clean_union(pixel_geoms).simplify(1.0, preserve_topology=True)
    polygon_raster = _clean_union(raster_geoms)
    if polygon_pixel.is_empty:
        return _empty_footprint(ds, scene_id, image_path, source=source, warnings=warnings)
    minx, miny, maxx, maxy = polygon_pixel.bounds
    area = float(polygon_pixel.area)
    return SceneFootprint(
        scene_id=scene_id,
        image_path=image_path,
        raster_width=int(ds.width),
        raster_height=int(ds.height),
        raster_crs=str(ds.crs) if ds.crs else None,
        source=source,
        polygon_raster_crs=polygon_raster,
        polygon_pixel=polygon_pixel,
        bounds_pixel=(
            max(0, int(math.floor(minx))),
            max(0, int(math.floor(miny))),
            min(int(ds.width), int(math.ceil(maxx))),
            min(int(ds.height), int(math.ceil(maxy))),
        ),
        area_pixels_estimated=area,
        valid_pixel_share_estimated=area / max(1.0, float(int(ds.width) * int(ds.height))),
        warnings=warnings,
        raster_transform=_affine_tuple(ds.transform),
    )


def _full_footprint(ds: Any, scene_id: str, image_path: str, *, source: str, warnings: list[str]) -> SceneFootprint:
    pixel = box(0, 0, int(ds.width), int(ds.height))
    bounds = ds.bounds
    raster = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    area = float(int(ds.width) * int(ds.height))
    return SceneFootprint(
        scene_id=scene_id,
        image_path=image_path,
        raster_width=int(ds.width),
        raster_height=int(ds.height),
        raster_crs=str(ds.crs) if ds.crs else None,
        source=source,
        polygon_raster_crs=raster,
        polygon_pixel=pixel,
        bounds_pixel=(0, 0, int(ds.width), int(ds.height)),
        area_pixels_estimated=area,
        valid_pixel_share_estimated=1.0,
        warnings=warnings,
        raster_transform=_affine_tuple(ds.transform),
    )


def _empty_footprint(ds: Any, scene_id: str, image_path: str, *, source: str, warnings: list[str]) -> SceneFootprint:
    warnings = list(warnings) + ["footprint is empty; no tile windows will be generated"]
    pixel = box(0, 0, 0, 0)
    return SceneFootprint(
        scene_id=scene_id,
        image_path=image_path,
        raster_width=int(ds.width),
        raster_height=int(ds.height),
        raster_crs=str(ds.crs) if ds.crs else None,
        source=source,
        polygon_raster_crs=pixel,
        polygon_pixel=pixel,
        bounds_pixel=(0, 0, 0, 0),
        area_pixels_estimated=0.0,
        valid_pixel_share_estimated=0.0,
        warnings=warnings,
        raster_transform=_affine_tuple(ds.transform),
    )


def _affine_tuple(transform: Affine) -> tuple[float, float, float, float, float, float]:
    return (float(transform.a), float(transform.b), float(transform.c), float(transform.d), float(transform.e), float(transform.f))


def _metadata_affine_tuple(value: Any) -> tuple[float, float, float, float, float, float] | None:
    if value is None:
        return None
    try:
        values = [float(item) for item in value]
    except Exception:  # noqa: BLE001
        return None
    if len(values) >= 6:
        return (values[0], values[1], values[2], values[3], values[4], values[5])
    return None


def _clean_union(geometries: list[Any]) -> Any:
    if not geometries:
        return box(0, 0, 0, 0)
    geom = unary_union(geometries)
    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:  # noqa: BLE001
        pass
    return geom


def _alpha_band_indexes(ds: Any) -> list[int]:
    try:
        from rasterio.enums import ColorInterp

        return [idx + 1 for idx, value in enumerate(ds.colorinterp or []) if value == ColorInterp.alpha]
    except Exception:  # noqa: BLE001
        return []
