from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from .config import TilePreparationConfig
from .footprint import SceneFootprint, rasterize_footprint_for_window
from .records import TileSampleRecord, TileWindow
from .validity import read_valid_data_mask_with_source


@dataclass
class MosaicReadResult:
    image: np.ndarray
    valid_mask: np.ndarray
    source_map: np.ndarray | None
    source_scenes: list[str]
    filled_pixel_count: int
    unfilled_pixel_count: int
    warnings: list[str] = field(default_factory=list)
    anchor_valid_pixel_share: float = 0.0
    final_valid_pixel_share: float = 0.0
    valid_data_source: str = "unknown"
    candidate_neighbors: int = 0
    intersecting_neighbors: int = 0
    actually_used_neighbors: list[str] = field(default_factory=list)
    skipped_non_intersecting_neighbors: int = 0


def read_mosaic_window(
    anchor_ds: Any,
    neighbor_datasets: list[tuple[str, Any]],
    record: TileWindow | TileSampleRecord,
    config: TilePreparationConfig,
    *,
    anchor_footprint: SceneFootprint | None = None,
    neighbor_footprints: dict[str, SceneFootprint] | None = None,
) -> MosaicReadResult:
    raster_window = Window(int(record.x), int(record.y), int(record.width), int(record.height))
    image = anchor_ds.read(_bands(anchor_ds, config), window=raster_window, boundless=True, fill_value=0)
    if anchor_footprint is not None:
        valid = rasterize_footprint_for_window(anchor_footprint, record).astype("uint8", copy=True)
        anchor_valid_share = float(np.count_nonzero(valid)) / int(valid.size) if valid.size else 0.0
        anchor_valid_source = anchor_footprint.source
    else:
        anchor_valid = read_valid_data_mask_with_source(anchor_ds, raster_window, mode=config.valid_pixel_mode)
        valid = anchor_valid.mask.astype("uint8", copy=True)
        anchor_valid_share = anchor_valid.valid_pixel_share
        anchor_valid_source = anchor_valid.source
    source_map = np.zeros(valid.shape, dtype="uint16")
    source_map[valid > 0] = 1
    source_scenes = [str(getattr(record, "scene_id", "") or "anchor")]
    actually_used_neighbors: list[str] = []
    warnings: list[str] = []
    filled = 0
    candidate_neighbors = len(neighbor_datasets)
    intersecting_neighbors = 0
    skipped_non_intersecting_neighbors = 0

    if not config.mosaic_enabled or not config.mosaic_fill_nodata or valid.all():
        return MosaicReadResult(
            image=image,
            valid_mask=valid,
            source_map=source_map,
            source_scenes=source_scenes,
            filled_pixel_count=0,
            unfilled_pixel_count=int(valid.size - np.count_nonzero(valid)),
            warnings=warnings,
            anchor_valid_pixel_share=anchor_valid_share,
            final_valid_pixel_share=anchor_valid_share,
            valid_data_source=anchor_valid_source,
            candidate_neighbors=candidate_neighbors,
            intersecting_neighbors=0,
            actually_used_neighbors=[],
            skipped_non_intersecting_neighbors=0,
        )

    target_transform = anchor_ds.window_transform(raster_window)
    target_bounds = anchor_ds.window_bounds(raster_window)
    for source_index, (scene_id, neighbor_ds) in enumerate(neighbor_datasets, start=2):
        if config.mosaic_require_same_crs and neighbor_ds.crs != anchor_ds.crs:
            warnings.append(f"mosaic skipped {scene_id}: CRS differs from anchor")
            continue
        if not _neighbor_intersects_target(scene_id, neighbor_ds, anchor_ds, target_bounds, neighbor_footprints):
            skipped_non_intersecting_neighbors += 1
            continue
        intersecting_neighbors += 1
        try:
            with WarpedVRT(
                neighbor_ds,
                crs=anchor_ds.crs,
                transform=target_transform,
                width=int(record.width),
                height=int(record.height),
                resampling=_resampling(config.mosaic_resampling),
            ) as vrt:
                neighbor_image = vrt.read(_bands(vrt, config), window=Window(0, 0, int(record.width), int(record.height)))
                neighbor_valid = read_valid_data_mask_with_source(vrt, Window(0, 0, int(record.width), int(record.height)), mode=config.valid_pixel_mode)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"mosaic skipped {scene_id}: {type(exc).__name__}: {exc}")
            continue
        fill = (valid == 0) & (neighbor_valid.mask > 0)
        if not fill.any():
            continue
        image[:, fill] = neighbor_image[:, fill]
        valid[fill] = 1
        source_map[fill] = source_index
        filled += int(np.count_nonzero(fill))
        source_scenes.append(str(scene_id))
        actually_used_neighbors.append(str(scene_id))
        if valid.all():
            break

    final_share = float(np.count_nonzero(valid)) / int(valid.size) if valid.size else 0.0
    return MosaicReadResult(
        image=image,
        valid_mask=valid.astype("uint8"),
        source_map=source_map,
        source_scenes=source_scenes,
        filled_pixel_count=filled,
        unfilled_pixel_count=int(valid.size - np.count_nonzero(valid)),
        warnings=warnings,
        anchor_valid_pixel_share=anchor_valid_share,
        final_valid_pixel_share=final_share,
        valid_data_source=anchor_valid_source,
        candidate_neighbors=candidate_neighbors,
        intersecting_neighbors=intersecting_neighbors,
        actually_used_neighbors=actually_used_neighbors,
        skipped_non_intersecting_neighbors=skipped_non_intersecting_neighbors,
    )


def _bands(ds: Any, config: TilePreparationConfig) -> list[int]:
    requested = config.input_bands or list(range(1, int(ds.count) + 1))
    bands = [int(band) for band in requested if 1 <= int(band) <= int(ds.count)]
    if not bands:
        raise ValueError("no readable bands for mosaic")
    return bands


def _resampling(value: str) -> Resampling:
    name = str(value or "bilinear").lower()
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }.get(name, Resampling.bilinear)


def _neighbor_bounds_in_anchor_crs(neighbor_ds: Any, anchor_crs: Any) -> tuple[float, float, float, float]:
    bounds = neighbor_ds.bounds
    if neighbor_ds.crs and anchor_crs and neighbor_ds.crs != anchor_crs:
        return tuple(
            float(value)
            for value in transform_bounds(
                neighbor_ds.crs,
                anchor_crs,
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                densify_pts=21,
            )
        )
    return (float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top))


def _neighbor_intersects_target(
    scene_id: str,
    neighbor_ds: Any,
    anchor_ds: Any,
    target_bounds: tuple[float, float, float, float],
    neighbor_footprints: dict[str, SceneFootprint] | None,
) -> bool:
    footprint = (neighbor_footprints or {}).get(str(scene_id))
    target = box(*target_bounds)
    if footprint is not None and footprint.polygon_raster_crs is not None:
        polygon = footprint.polygon_raster_crs
        if neighbor_ds.crs and anchor_ds.crs and neighbor_ds.crs != anchor_ds.crs:
            try:
                from pyproj import Transformer

                transformer = Transformer.from_crs(neighbor_ds.crs, anchor_ds.crs, always_xy=True)
                polygon = shapely_transform(transformer.transform, polygon)
            except Exception:  # noqa: BLE001
                polygon = None
        if polygon is not None:
            try:
                return bool(polygon.intersects(target))
            except Exception:  # noqa: BLE001
                pass
    try:
        neighbor_bounds = _neighbor_bounds_in_anchor_crs(neighbor_ds, anchor_ds.crs)
    except Exception:  # noqa: BLE001
        return False
    return _bounds_intersect(target_bounds, neighbor_bounds)


def _bounds_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    left_a, bottom_a, right_a, top_a = a
    left_b, bottom_b, right_b, top_b = b
    return not (right_a <= left_b or right_b <= left_a or top_a <= bottom_b or top_b <= bottom_a)
