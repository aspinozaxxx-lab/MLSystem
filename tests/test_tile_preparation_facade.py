from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation import AnnotationInput, SceneInput, TilePreparationFacade, TilePreparationSimpleParams
from mlsystem.src.tile_preparation.iterator import build_validation_tile_records

try:
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio and pyproj are required")
class TilePreparationFacadeTests(unittest.TestCase):
    def test_default_config_accepts_all_augmentation_levels(self) -> None:
        for level in range(4):
            with self.subTest(level=level):
                config = TilePreparationFacade.default_config(tile_size=128, augmentation_level=level, seed=7)
                self.assertEqual(config.augmentation_level, level)
                self.assertEqual(config.cutout_mask_mode, "erase")
                self.assertTrue(config.drop_fully_invalid_tiles)

    def test_from_scene_list_builds_train_and_val_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scene(root / "scene_a.tif", crs="EPSG:3857", transform=from_origin(0, 256, 1, 1))
            _write_scene(root / "scene_b.tif", crs="EPSG:3857", transform=from_origin(0, 256, 1, 1))
            _write_geojson(root / "ann.geojson", Polygon([(32, 224), (160, 224), (160, 96), (32, 96)]), "EPSG:3857")
            scene_list = root / "scenes.txt"
            scene_list.write_text("scene_a\nscene_b.tif\n", encoding="utf-8")

            bundle = TilePreparationFacade.from_scene_list(
                images_root=root,
                scene_list_path=scene_list,
                annotation_path=root / "ann.geojson",
                tile_size=128,
                augmentation_level=1,
                mosaic_mode="auto",
                seed=5,
            )
            try:
                self.assertGreater(len(bundle.train_dataset), 0)
                self.assertGreater(len(bundle.val_dataset), 0)
                self.assertEqual(bundle.config.augmentation_level, 1)
                self.assertEqual(bundle.config.cutout_mask_mode, "erase")
                self.assertIn("skipped_fully_invalid_tiles", bundle.train_summary)
            finally:
                bundle.train_dataset.close()
                bundle.val_dataset.close()

    def test_multicrs_annotation_is_transformed_per_raster(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster_3857 = root / "scene_3857.tif"
            raster_4326 = root / "scene_4326.tif"
            _write_scene(raster_3857, crs="EPSG:3857", transform=from_origin(0, 1024, 2, 2), size=512)

            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            lon0, lat0 = transformer.transform(120, 820)
            lon1, lat1 = transformer.transform(300, 640)
            polygon_4326 = Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)])
            _write_scene(
                raster_4326,
                crs="EPSG:4326",
                transform=from_origin(min(lon0, lon1) - 0.001, max(lat0, lat1) + 0.001, 0.00001, 0.00001),
                size=512,
            )
            _write_geojson(root / "ann.geojson", polygon_4326, "EPSG:4326")
            params = TilePreparationSimpleParams(tile_size=256, stride=256, augmentation_level=0, mosaic_mode="off", seed=9)
            bundle = TilePreparationFacade.build_datasets(
                train_scenes=[SceneInput(raster_3857, "scene_3857"), SceneInput(raster_4326, "scene_4326")],
                val_scenes=[SceneInput(raster_3857, "scene_3857"), SceneInput(raster_4326, "scene_4326")],
                annotation=AnnotationInput(root / "ann.geojson", annotation_crs="auto", allow_inferred_annotation_crs=False),
                params=params,
            )
            try:
                reports = {item["scene"]: item for item in bundle.train_dataset.scene_reports}
                self.assertTrue(reports["scene_3857"]["transformed_to_raster_crs"])
                self.assertFalse(reports["scene_4326"]["transformed_to_raster_crs"])
                self.assertGreater(reports["scene_3857"]["positive_tiles"], 0)
                self.assertGreater(reports["scene_4326"]["positive_tiles"], 0)
            finally:
                bundle.train_dataset.close()
                bundle.val_dataset.close()

    def test_missing_geojson_crs_fails_when_inference_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scene(root / "scene.tif", crs="EPSG:3857", transform=from_origin(0, 256, 1, 1))
            payload = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": mapping(Polygon([(32, 224), (160, 224), (160, 96), (32, 96)])),
                    }
                ],
            }
            (root / "ann.geojson").write_text(json.dumps(payload), encoding="utf-8")
            params = TilePreparationSimpleParams(tile_size=128, stride=128, augmentation_level=0, mosaic_mode="off")
            with self.assertRaisesRegex(ValueError, "annotation CRS is missing"):
                TilePreparationFacade.build_datasets(
                    train_scenes=[SceneInput(root / "scene.tif", "scene")],
                    val_scenes=[SceneInput(root / "scene.tif", "scene")],
                    annotation=AnnotationInput(root / "ann.geojson", annotation_crs="auto", allow_inferred_annotation_crs=False),
                    params=params,
                )

    def test_cutout_erase_runs_through_facade_dataset_getitem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scene(root / "scene.tif", crs="EPSG:3857", transform=from_origin(0, 128, 1, 1), size=128)
            _write_geojson(root / "ann.geojson", Polygon([(0, 128), (128, 128), (128, 0), (0, 0)]), "EPSG:3857")
            found = False
            for seed in range(80):
                params = TilePreparationSimpleParams(tile_size=128, stride=128, augmentation_level=3, mosaic_mode="off", seed=seed)
                bundle = TilePreparationFacade.build_datasets(
                    train_scenes=[SceneInput(root / "scene.tif", "scene")],
                    val_scenes=[SceneInput(root / "scene.tif", "scene")],
                    annotation=AnnotationInput(root / "ann.geojson"),
                    params=params,
                )
                try:
                    sample = bundle.train_dataset[0]
                    aug = sample.metadata.get("augmentation") or {}
                    cutout = aug.get("cutout") or {}
                    if aug.get("cutout_applied") and int(cutout.get("mask_erased_pixels") or 0) > 0:
                        found = True
                        self.assertLess(float(sample.mask.sum()), float(sample.record.positive_pixels))
                        self.assertTrue(set(np.unique(sample.mask).tolist()).issubset({0.0, 1.0}))
                        self.assertEqual(cutout["cutout_mask_mode"], "erase")
                        break
                finally:
                    bundle.train_dataset.close()
                    bundle.val_dataset.close()
            self.assertTrue(found, "expected at least one deterministic seed to apply cutout over a positive mask")


def _write_scene(path: Path, *, crs: str, transform: object, size: int = 256) -> None:
    y, x = np.mgrid[0:size, 0:size]
    data = np.stack([(x % 255), (y % 255), ((x + y) % 255)], axis=0).astype("uint8")
    data[:, 0, 0] = 1
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs=crs, transform=transform) as ds:
        ds.write(data)


def _write_geojson(path: Path, polygon: Polygon, crs: str) -> None:
    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": crs}},
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(polygon)}],
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
