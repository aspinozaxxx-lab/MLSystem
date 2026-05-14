from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation import SceneInput, TilePreparationFacade

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationFacadeApiTests(unittest.TestCase):
    def test_from_scene_list_is_not_public_api(self) -> None:
        self.assertFalse(hasattr(TilePreparationFacade, "from_scene_list"))

    def test_default_config_has_minimal_signature(self) -> None:
        signature = inspect.signature(TilePreparationFacade.default_config)
        self.assertEqual(list(signature.parameters), ["tile_size", "stride", "augmentation_level"])

    def test_build_datasets_rejects_removed_public_parameters(self) -> None:
        signature = inspect.signature(TilePreparationFacade.build_datasets)
        forbidden = {"train_fraction", "stride_ratio", "mosaic_mode", "seed", "input_bands", "max_empty_tile_share"}
        self.assertFalse(forbidden.intersection(signature.parameters))

    def test_build_datasets_creates_bundle_from_prepared_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scene(root / "train_a.tif", offset=0)
            _write_scene(root / "train_b.tif", offset=10)
            _write_scene(root / "val_a.tif", offset=20)
            _write_annotation(root / "ann.geojson")

            bundle = TilePreparationFacade.build_datasets(
                train_scenes=[SceneInput(root / "train_a.tif", "train_a"), SceneInput(root / "train_b.tif", "train_b")],
                val_scenes=[SceneInput(root / "val_a.tif", "val_a")],
                annotation_path=root / "ann.geojson",
                tile_size=128,
                stride=128,
                augmentation_level=2,
            )
            try:
                self.assertGreater(len(bundle.train_dataset), 0)
                self.assertGreater(len(bundle.val_dataset), 0)
                self.assertTrue(bundle.train_dataset.config.mosaic_enabled)
                self.assertFalse(bundle.val_dataset.config.mosaic_enabled)
                self.assertEqual({scene.resolved_scene_id() for scene in bundle.train_dataset.scenes}, {"train_a", "train_b"})
                self.assertEqual({scene.resolved_scene_id() for scene in bundle.val_dataset.scenes}, {"val_a"})
            finally:
                bundle.train_dataset.close()
                bundle.val_dataset.close()

    def test_validation_config_is_deterministic_and_unexpanded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scene(root / "train.tif", offset=0)
            _write_scene(root / "val.tif", offset=20)
            _write_annotation(root / "ann.geojson")

            bundle = TilePreparationFacade.build_datasets(
                train_scenes=[SceneInput(root / "train.tif", "train")],
                val_scenes=[SceneInput(root / "val.tif", "val")],
                annotation_path=root / "ann.geojson",
                tile_size=128,
                stride=128,
                augmentation_level=3,
            )
            try:
                val_config = bundle.val_dataset.config
                self.assertFalse(val_config.apply_random_augmentations)
                self.assertEqual(val_config.augmentations, {})
                self.assertEqual(val_config.positive_repeat_factor, 1)
                self.assertEqual(val_config.hard_negative_repeat_factor, 1)
                self.assertEqual(val_config.negative_repeat_factor, 1)
                self.assertEqual(val_config.positive_stride_factor, 1.0)
                self.assertEqual(val_config.hard_negative_stride_factor, 1.0)
                self.assertEqual(val_config.negative_stride_factor, 1.0)
                self.assertIsNone(val_config.max_empty_tile_share)
                self.assertEqual(len(bundle.val_dataset.records), len(bundle.val_dataset.base_records))
            finally:
                bundle.train_dataset.close()
                bundle.val_dataset.close()


def _write_scene(path: Path, *, offset: int) -> None:
    size = 256
    y, x = np.mgrid[0:size, 0:size]
    data = np.stack([((x + offset) % 255), ((y + offset) % 255), ((x + y + offset) % 255)], axis=0).astype("uint8")
    data[:, 0, 0] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, size, 1, 1),
    ) as ds:
        ds.write(data)


def _write_annotation(path: Path) -> None:
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
