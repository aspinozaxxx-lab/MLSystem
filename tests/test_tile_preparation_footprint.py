from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box, mapping
from shapely.ops import unary_union

from mlsystem.src.tile_preparation.config import TilePreparationConfig
from mlsystem.src.tile_preparation.footprint import build_scene_footprint, generate_windows_for_footprint, window_footprint_intersection

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


def _write_raster_with_valid_box(path: Path, *, valid_box: tuple[int, int, int, int]) -> None:
    data = np.zeros((3, 256, 256), dtype="uint8")
    x0, y0, x1, y1 = valid_box
    data[:, y0:y1, x0:x1] = 100
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
