from __future__ import annotations

import json
import tempfile
import unittest
from itertools import islice
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation import AnnotationInput, SceneInput, TilePreparationConfig
from mlsystem.src.tile_preparation.iterator import build_tile_records, iter_training_tiles

try:
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationIteratorTests(unittest.TestCase):
    def test_synthetic_raster_geojson_records_and_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path = root / "scene.tif"
            geojson_path = root / "scene.geojson"
            _write_raster(raster_path, width=1024, height=1024, crs="EPSG:3857", transform=from_origin(0, 1024, 1, 1))
            _write_geojson(geojson_path, Polygon([(300, 700), (460, 700), (460, 540), (300, 540)]), "EPSG:3857")
            config = TilePreparationConfig(tile_size=256, stride=256, output_format="hwc_uint8")

            result = build_tile_records(
                [SceneInput(raster_path, "scene")],
                AnnotationInput(geojson_path, annotation_crs="auto"),
                config,
            )

            self.assertEqual(len(result.base_records), 16)
            self.assertGreater(sum(record.kind == "positive" for record in result.base_records), 0)
            self.assertGreater(sum(record.kind == "negative" for record in result.base_records), 0)

            positive_record = next(record for record in result.base_records if record.kind == "positive")
            sample = next(
                item
                for item in iter_training_tiles([SceneInput(raster_path, "scene")], AnnotationInput(geojson_path), config)
                if item.record.x == positive_record.x and item.record.y == positive_record.y
            )
            self.assertEqual(sample.image.shape, (256, 256, 3))
            self.assertEqual(sample.mask.shape, (256, 256))
            self.assertEqual(sample.mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(sample.mask).tolist()).issubset({0, 1}))
            self.assertGreater(int(sample.mask.sum()), 0)

            negative = next(item for item in iter_training_tiles([SceneInput(raster_path, "scene")], AnnotationInput(geojson_path), config) if item.record.kind == "negative")
            self.assertEqual(int(negative.mask.sum()), 0)

    def test_crs_transform_keeps_mask_non_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path = root / "scene.tif"
            geojson_path = root / "scene.geojson"
            transform = from_origin(-1_000_000, 1_000_000, 1000, 1000)
            _write_raster(raster_path, width=1024, height=1024, crs="EPSG:3857", transform=transform)
            polygon_3857 = Polygon([(-700_000, 700_000), (-550_000, 700_000), (-550_000, 550_000), (-700_000, 550_000)])
            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            polygon_4326 = Polygon([transformer.transform(x, y) for x, y in polygon_3857.exterior.coords])
            _write_geojson(geojson_path, polygon_4326, "EPSG:4326")
            config = TilePreparationConfig(tile_size=256, stride=256, output_format="hwc_uint8")

            samples = list(islice(iter_training_tiles([SceneInput(raster_path, "scene")], AnnotationInput(geojson_path), config), 16))
            self.assertTrue(any(int(sample.mask.sum()) > 0 for sample in samples))

    def test_iterator_is_lazy_and_samples_are_ready_to_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path = root / "scene.tif"
            geojson_path = root / "scene.geojson"
            _write_raster(raster_path, width=1024, height=1024, crs="EPSG:3857", transform=from_origin(0, 1024, 1, 1))
            _write_geojson(geojson_path, Polygon([(100, 900), (250, 900), (250, 750), (100, 750)]), "EPSG:3857")
            config = TilePreparationConfig(tile_size=256, stride=256, output_format="chw_float32", max_records=3)

            iterator = iter_training_tiles([SceneInput(raster_path, "scene")], AnnotationInput(geojson_path), config)
            self.assertFalse(isinstance(iterator, list))
            samples = list(iterator)
            self.assertEqual(len(samples), 3)
            self.assertEqual(samples[0].image.shape, (3, 256, 256))
            self.assertEqual(samples[0].image.dtype, np.float32)
            self.assertEqual(samples[0].mask.shape, (1, 256, 256))


def _write_raster(path: Path, *, width: int, height: int, crs: str, transform: object) -> None:
    y, x = np.mgrid[0:height, 0:width]
    data = np.stack([(x % 255), (y % 255), ((x + y) % 255)], axis=0).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=3, dtype="uint8", crs=crs, transform=transform) as ds:
        ds.write(data)


def _write_geojson(path: Path, polygon: Polygon, crs: str) -> None:
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(polygon)}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
