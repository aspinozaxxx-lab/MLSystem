from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from shapely.geometry import Polygon, mapping

    from mlsystem.src.tile_preparation.api import build_datasets, train_dataloader
    from mlsystem.src.tile_preparation.contracts import SceneInputContract, TileDatasetRequest
    import rasterio
    from rasterio.transform import from_origin

    HAS_TILE_PREP_DEPS = True
except Exception:  # noqa: BLE001
    HAS_TILE_PREP_DEPS = False


@unittest.skipUnless(HAS_TILE_PREP_DEPS, "rasterio, shapely, and torch are required")
class TilePreparationDataloaderTests(unittest.TestCase):
    def test_workers_zero_returns_compatible_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp))
            try:
                indices, x, y = next(iter(train_dataloader(bundle, batch_size=2, workers=0)))
                self.assertIsInstance(indices, list)
                self.assertEqual(tuple(x.shape[1:]), (3, 128, 128))
                self.assertEqual(tuple(y.shape[1:]), (1, 128, 128))
                self.assertTrue(set(np.unique(y.numpy()).tolist()).issubset({0.0, 1.0}))
            finally:
                bundle.train_dataset.close()
                bundle.val_dataset.close()

    def test_workers_two_returns_compatible_batch_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp))
            try:
                indices, x, y = next(iter(train_dataloader(bundle, batch_size=2, workers=2, persistent_workers=False)))
                self.assertIsInstance(indices, list)
                self.assertEqual(tuple(x.shape[1:]), (3, 128, 128))
                self.assertEqual(tuple(y.shape[1:]), (1, 128, 128))
                self.assertTrue(set(np.unique(y.numpy()).tolist()).issubset({0.0, 1.0}))
            finally:
                bundle.train_dataset.close()
                bundle.val_dataset.close()

    def test_validation_dataloader_has_no_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp), augmentation_level=3)
            try:
                self.assertFalse(bundle.val_dataset.config.apply_random_augmentations)
                self.assertEqual(bundle.val_dataset.config.augmentations, {})
                sample_a = bundle.val_dataset[0]
                sample_b = bundle.val_dataset[0]
                self.assertNotIn("augmentation", sample_a.metadata)
                np.testing.assert_array_equal(sample_a.image, sample_b.image)
                np.testing.assert_array_equal(sample_a.mask, sample_b.mask)
            finally:
                bundle.train_dataset.close()
                bundle.val_dataset.close()


def _bundle(root: Path, *, augmentation_level: int = 1):
    raster_path = root / "scene.tif"
    geojson_path = root / "ann.geojson"
    _write_scene(raster_path)
    _write_geojson(geojson_path)
    return build_datasets(
        TileDatasetRequest(
            train_scenes=[SceneInputContract(raster_path, "scene")],
            val_scenes=[SceneInputContract(raster_path, "scene")],
            annotation_path=geojson_path,
            tile_size=128,
            stride=128,
            augmentation_level=augmentation_level,
            mosaic_enabled=False,
        )
    )


def _write_scene(path: Path) -> None:
    y, x = np.mgrid[0:256, 0:256]
    data = np.stack([(x % 255), (y % 255), ((x + y) % 255)], axis=0).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 256, 1, 1)) as ds:
        ds.write(data)


def _write_geojson(path: Path) -> None:
    polygon = Polygon([(32, 224), (160, 224), (160, 96), (32, 96)])
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
