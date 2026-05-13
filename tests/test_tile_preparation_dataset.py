from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation import AnnotationInput, SceneInput, TilePreparationConfig, TrainingTileDataset

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationDatasetTests(unittest.TestCase):
    def test_dataset_reads_tiles_lazily_and_returns_ready_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path, geojson_path = _write_fixture(root)
            config = TilePreparationConfig(tile_size=256, stride=256, input_bands=[1, 2, 3], seed=3)

            dataset = TrainingTileDataset([SceneInput(raster_path, "scene")], AnnotationInput(geojson_path), config, train=True)
            try:
                self.assertEqual(len(dataset), len(dataset.records))
                self.assertGreater(len(dataset), 0)
                self.assertFalse(hasattr(dataset, "samples"))
                self.assertEqual(dataset._datasets, {})

                sample = dataset[0]

                self.assertEqual(sample.image.shape, (3, 256, 256))
                self.assertEqual(sample.image.dtype, np.float32)
                self.assertEqual(sample.mask.shape, (1, 256, 256))
                self.assertTrue(set(np.unique(sample.mask).tolist()).issubset({0.0, 1.0}))
                self.assertTrue(dataset._datasets)
                self.assertIsNotNone(sample.record.record_id)
            finally:
                dataset.close()

    def test_train_augmentations_apply_and_validation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_path, geojson_path = _write_fixture(root)
            train_config = TilePreparationConfig(
                tile_size=256,
                stride=256,
                input_bands=[1, 2, 3],
                augmentations={"flips": True, "rot90": True, "brightness_contrast": True, "gamma": True, "noise": True, "blur": True, "cutout": True},
                apply_random_augmentations=True,
                seed=12,
            )
            val_config = TilePreparationConfig(tile_size=256, stride=256, input_bands=[1, 2, 3], apply_random_augmentations=False, seed=12)
            scene = SceneInput(raster_path, "scene")
            annotation = AnnotationInput(geojson_path)

            train_dataset = TrainingTileDataset([scene], annotation, train_config, train=True)
            val_dataset = TrainingTileDataset([scene], annotation, val_config, train=False)
            try:
                train_sample = train_dataset[0]
                val_a = val_dataset[0]
                val_b = val_dataset[0]

                self.assertIn("augmentation", train_sample.metadata)
                self.assertNotIn("augmentation", val_a.metadata)
                np.testing.assert_array_equal(val_a.image, val_b.image)
                np.testing.assert_array_equal(val_a.mask, val_b.mask)
                self.assertTrue(set(np.unique(train_sample.mask).tolist()).issubset({0.0, 1.0}))
            finally:
                train_dataset.close()
                val_dataset.close()

    def test_real_train_does_not_import_legacy_virtual_tile_sampling(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertNotIn("from .data.virtual_tile_sampling import", text)
        self.assertNotIn("def _apply_train_augmentations", text)
        self.assertNotIn("build_virtual_train_records", text)
        self.assertNotIn("build_validation_records", text)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    width = height = 1024
    y, x = np.mgrid[0:height, 0:width]
    data = np.stack([(x % 255), (y % 255), ((x + y) % 255)], axis=0).astype("uint8")
    raster_path = root / "scene.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 1024, 1, 1),
    ) as ds:
        ds.write(data)
    polygon = Polygon([(100, 900), (380, 900), (380, 620), (100, 620)])
    geojson_path = root / "scene.geojson"
    geojson_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(polygon)}],
            }
        ),
        encoding="utf-8",
    )
    return raster_path, geojson_path


if __name__ == "__main__":
    unittest.main()
