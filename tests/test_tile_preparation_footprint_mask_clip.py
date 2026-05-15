from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation import AnnotationInput, SceneInput, TilePreparationConfig, TrainingTileDataset
from mlsystem.src.tile_preparation.iterator import build_validation_tile_records

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationFootprintMaskClipTests(unittest.TestCase):
    def test_boundary_tile_mask_is_clipped_to_footprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path = root / "scene.tif"
            geojson_path = root / "scene.geojson"
            _write_partial_valid_raster(raster_path)
            _write_geojson(geojson_path, Polygon([(0, 128), (128, 128), (128, 0), (0, 0)]))
            config = TilePreparationConfig(tile_size=64, stride=64, valid_pixel_mode="nonzero_any", drop_fully_invalid_tiles=True)
            scene = SceneInput(raster_path, "scene")
            annotation = AnnotationInput(geojson_path)
            build_result = build_validation_tile_records([scene], annotation, config)
            dataset = TrainingTileDataset([scene], annotation, config, train=False, build_result=build_result)
            try:
                inside_index = next(index for index, record in enumerate(dataset.records) if record.x == 0 and record.y == 0)
                boundary_index = next(index for index, record in enumerate(dataset.records) if record.x == 64 and record.y == 0)
                inside = dataset[inside_index]
                boundary = dataset[boundary_index]

                self.assertTrue(inside.record.metadata["footprint_fully_inside"])
                self.assertFalse(boundary.record.metadata["footprint_fully_inside"])
                self.assertTrue(boundary.record.metadata["footprint_boundary"])
                self.assertGreater(int(boundary.mask.sum()), 0)
                self.assertEqual(int(boundary.mask[:, :, 16:].sum()), 0)
                self.assertEqual(int(inside.mask.sum()), 64 * 64)
            finally:
                dataset.close()


def _write_partial_valid_raster(path: Path) -> None:
    data = np.zeros((3, 128, 128), dtype="uint8")
    data[:, :, :80] = 100
    with rasterio.open(path, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 128, 1, 1)) as ds:
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
