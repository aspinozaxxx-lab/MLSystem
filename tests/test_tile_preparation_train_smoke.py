from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from scripts.debug_tile_preparation_train_smoke import run_train_smoke

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationTrainSmokeTests(unittest.TestCase):
    def test_facade_smoke_runs_one_training_step_without_fastapi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scene(root / "scene_a.tif", offset=0)
            _write_scene(root / "scene_b.tif", offset=40)
            scene_list = root / "scenes.txt"
            scene_list.write_text("scene_a.tif\nscene_b.tif\n", encoding="utf-8")
            annotation = root / "deforestation.geojson"
            _write_annotation(annotation)
            output_json = root / "train_smoke_result.json"

            result = run_train_smoke(
                images_root=root,
                scene_list=scene_list,
                annotation=annotation,
                tile_size=128,
                augmentation_level=1,
                mosaic_mode="auto",
                batch_size=2,
                max_train_batches=1,
                max_val_batches=1,
                device="cpu",
                output_json=output_json,
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["entrypoint"], "TilePreparationFacade")
            self.assertFalse(result["uses_fastapi"])
            self.assertTrue(result["backward_optimizer_step"])
            self.assertTrue(result["train_batches"])
            self.assertTrue(result["val_batches"])
            self.assertEqual(result["train_batches"][0]["x_shape"][1], 3)
            self.assertEqual(result["train_batches"][0]["y_shape"][1], 1)
            self.assertTrue(set(result["train_batches"][0]["y_values"]).issubset({0.0, 1.0}))
            self.assertTrue(output_json.exists())
            saved = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(saved["bundle"]["config"]["cutout_mask_mode"], "erase")


def _write_scene(path: Path, *, offset: int) -> None:
    width = height = 256
    y, x = np.mgrid[0:height, 0:width]
    data = np.stack([((x + offset) % 255), ((y + offset) % 255), ((x + y + offset) % 255)], axis=0).astype("uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(0, 256, 1, 1),
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
