from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

from mlsystem.src.tile_preparation.config import TilePreparationConfig
from mlsystem.src.tile_preparation.footprint import (
    SceneFootprint,
    build_scene_footprint,
    generate_windows_for_footprint,
    rasterize_footprint_for_window,
    window_footprint_intersection,
)
from mlsystem.src.tile_preparation.records import TileWindow

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationFootprintTests(unittest.TestCase):
    def test_footprint_polygon_built_and_windows_outside_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raster_path = Path(tmp) / "scene.tif"
            _write_raster_with_valid_box(raster_path, valid_box=(20, 20, 180, 180))

            with rasterio.open(raster_path) as ds:
                config = TilePreparationConfig(tile_size=64, stride=64, valid_pixel_mode="nonzero_any")
                footprint = build_scene_footprint(ds, config, scene_id="scene", image_path=str(raster_path))
                windows = generate_windows_for_footprint(ds.width, ds.height, 64, 64, footprint, "scene")

            self.assertFalse(footprint.is_empty)
            self.assertEqual(footprint.source, "nonzero_any")
            self.assertLess(len(windows), 16)
            self.assertTrue(any(window_footprint_intersection(window, footprint).is_boundary for window in windows))
            self.assertTrue(any(window_footprint_intersection(window, footprint).fully_inside for window in windows))

            covered = unary_union([box(w.x, w.y, w.x + w.width, w.y + w.height) for w in windows])
            self.assertAlmostEqual(float(footprint.polygon_pixel.difference(covered).area), 0.0)

    def test_irregular_diagonal_footprint_keeps_all_intersecting_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raster_path = Path(tmp) / "diagonal.tif"
            _write_diagonal_valid_raster(raster_path)

            with rasterio.open(raster_path) as ds:
                config = TilePreparationConfig(tile_size=64, stride=64, valid_pixel_mode="nonzero_any")
                footprint = build_scene_footprint(ds, config, scene_id="diagonal", image_path=str(raster_path))
                kept = generate_windows_for_footprint(ds.width, ds.height, 64, 64, footprint, "diagonal")
                rectangular = [
                    TileWindow("diagonal", x, y, 64, 64, 64, 64)
                    for y in (0, 64, 128, 192)
                    for x in (0, 64, 128, 192)
                ]

            expected = [window for window in rectangular if window_footprint_intersection(window, footprint).intersects]
            self.assertEqual([(w.x, w.y) for w in kept], [(w.x, w.y) for w in expected])
            covered = unary_union([box(w.x, w.y, w.x + w.width, w.y + w.height) for w in kept])
            self.assertAlmostEqual(float(footprint.polygon_pixel.difference(covered).area), 0.0)

    def test_rasterize_footprint_for_offset_window_uses_window_origin(self) -> None:
        footprint = SceneFootprint(
            scene_id="scene",
            image_path="scene.tif",
            raster_width=256,
            raster_height=256,
            raster_crs="EPSG:3857",
            source="test",
            polygon_raster_crs=box(80, 80, 140, 140),
            polygon_pixel=box(80, 80, 140, 140),
            bounds_pixel=(80, 80, 140, 140),
            area_pixels_estimated=60 * 60,
            valid_pixel_share_estimated=(60 * 60) / (256 * 256),
            warnings=[],
        )
        window = TileWindow("scene", 64, 64, 64, 64, 64, 64)
        mask = rasterize_footprint_for_window(footprint, window)

        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(int(mask[:16, :].sum()), 0)
        self.assertEqual(int(mask[:, :16].sum()), 0)
        self.assertGreater(int(mask[16:, 16:].sum()), 0)


def _write_raster_with_valid_box(path: Path, *, valid_box: tuple[int, int, int, int]) -> None:
    data = np.zeros((3, 256, 256), dtype="uint8")
    x0, y0, x1, y1 = valid_box
    data[:, y0:y1, x0:x1] = 100
    with rasterio.open(path, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 256, 1, 1)) as ds:
        ds.write(data)


def _write_diagonal_valid_raster(path: Path) -> None:
    data = np.zeros((3, 256, 256), dtype="uint8")
    y, x = np.mgrid[0:256, 0:256]
    valid = (x > y - 20) & (x < y + 110)
    data[:, valid] = 100
    with rasterio.open(path, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 256, 1, 1)) as ds:
        ds.write(data)


def _write_geojson(path: Path, polygon: Polygon) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(polygon)}],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
