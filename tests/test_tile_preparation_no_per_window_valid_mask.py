from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation import AnnotationInput, SceneInput, TilePreparationConfig
from mlsystem.src.tile_preparation import iterator as iterator_module

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationNoPerWindowValidMaskTests(unittest.TestCase):
    def test_build_records_do_not_call_per_window_valid_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path = root / "scene.tif"
            geojson_path = root / "ann.geojson"
            _write_scene(raster_path)
            _write_geojson(geojson_path)
            scene = SceneInput(raster_path, "scene")
            annotation = AnnotationInput(geojson_path)
            config = TilePreparationConfig(tile_size=64, stride=64, valid_pixel_mode="nonzero_any")

            with mock.patch.object(iterator_module, "read_valid_data_mask_with_source", side_effect=AssertionError("per-window valid mask read")) as patched:
                train = iterator_module.build_tile_records([scene], annotation, config)
                val = iterator_module.build_validation_tile_records([scene], annotation, config)

            self.assertEqual(patched.call_count, 0)
            self.assertGreater(len(train.records), 0)
            self.assertGreater(len(val.records), 0)
            self.assertEqual(train.metadata["read_valid_mask_calls"], 0)
            self.assertEqual(val.metadata["read_valid_mask_calls"], 0)


def _write_scene(path: Path) -> None:
    data = np.zeros((3, 128, 128), dtype="uint8")
    data[:, 10:118, 10:118] = 100
    with rasterio.open(path, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 128, 1, 1)) as ds:
        ds.write(data)


def _write_geojson(path: Path) -> None:
    polygon = Polygon([(16, 112), (96, 112), (96, 32), (16, 32)])
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
