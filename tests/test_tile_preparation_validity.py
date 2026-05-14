from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation.config import TilePreparationConfig
from mlsystem.src.tile_preparation.config import AnnotationInput, SceneInput
from mlsystem.src.tile_preparation.iterator import build_validation_tile_records
from mlsystem.src.tile_preparation.mask_rasterizer import rasterize_mask_for_window
from mlsystem.src.tile_preparation.records import TileWindow
from mlsystem.src.tile_preparation.validity import read_valid_data_mask_with_source

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationValidityTests(unittest.TestCase):
    def test_mask_is_clipped_to_nonzero_valid_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path = root / "scene.tif"
            geojson_path = root / "scene.geojson"
            _write_half_valid_raster(raster_path)
            _write_geojson(geojson_path, Polygon([(0, 64), (64, 64), (64, 0), (0, 0)]))
            with rasterio.open(raster_path) as ds:
                geoms = [Polygon([(0, 64), (64, 64), (64, 0), (0, 0)])]
                window = TileWindow("scene", 0, 0, 64, 64, 64, 64)
                valid = read_valid_data_mask_with_source(ds, window, mode="nonzero_any")
                result = rasterize_mask_for_window(ds, geoms, window, valid_mask=valid.mask)

            self.assertEqual(valid.source, "nonzero_any")
            self.assertGreater(result.raw_positive_pixels, result.clipped_positive_pixels)
            self.assertEqual(int(result.mask[:, 32:].sum()), 0)
            self.assertGreater(int(result.raw_mask[:, 32:].sum()), 0)
            self.assertAlmostEqual(result.valid_pixel_share, 0.5)

    def test_fully_invalid_tiles_are_never_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path = root / "black.tif"
            geojson_path = root / "black.geojson"
            _write_black_raster(raster_path)
            _write_geojson(geojson_path, Polygon([(0, 64), (64, 64), (64, 0), (0, 0)]))

            result = build_validation_tile_records(
                [SceneInput(raster_path, "black")],
                AnnotationInput(geojson_path),
                TilePreparationConfig(tile_size=64, stride=64, valid_pixel_mode="nonzero_any", drop_fully_invalid_tiles=True),
            )

            self.assertEqual(len(result.records), 0)
            self.assertEqual(result.metadata["skipped_fully_invalid_tiles"], 1)


def _write_half_valid_raster(path: Path) -> None:
    data = np.zeros((3, 64, 64), dtype="uint8")
    data[:, :, :32] = 100
    with rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 64, 1, 1)) as ds:
        ds.write(data)


def _write_black_raster(path: Path) -> None:
    data = np.zeros((3, 64, 64), dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 64, 1, 1)) as ds:
        ds.write(data)


def _write_geojson(path: Path, polygon: Polygon) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(polygon)}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
