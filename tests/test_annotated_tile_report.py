from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation import TilePreparationConfig
from mlsystem.src.tile_preparation.report import generate_annotated_tile_report, preview_annotated_tile_report

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False

try:
    import importlib
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except Exception:  # noqa: BLE001
    HAS_FASTAPI = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class AnnotatedTileReportTests(unittest.TestCase):
    def test_annotated_report_generates_mask_and_augmentation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture(root)
            index = generate_annotated_tile_report(
                root,
                config=TilePreparationConfig(tile_size=256, stride=256, max_empty_tile_share=None),
                annotation_crs="auto",
                max_overview_size=512,
                max_tile_examples=8,
                max_augmentation_tiles=2,
                augmentation_mode="production-groups",
                augmentation_seed=9,
            )
            scene_dir = root / "scene"
            self.assertEqual(index["scene_count"], 1)
            self.assertTrue((root / "annotated_tile_sampling_index.html").exists())
            self.assertTrue((scene_dir / "annotated_tile_sampling_report.html").exists())
            self.assertTrue((scene_dir / "annotated_scene_summary.json").exists())
            self.assertTrue((scene_dir / "overview_raster_grid_mask.png").exists())
            self.assertTrue((scene_dir / "overview_mask_only.png").exists())
            self.assertTrue((scene_dir / "overview_grid_positive_negative.png").exists())
            summary = json.loads((scene_dir / "annotated_scene_summary.json").read_text(encoding="utf-8"))
            self.assertIn("classification_summary", summary)
            self.assertIn("augmentation_report", summary)
            self.assertEqual(summary["mask_visualization"]["mode"], "dashed_contour")
            self.assertGreater(summary["classification_summary"]["positive_tiles"], 0)
            self.assertGreater(summary["augmentation_report"]["checks_summary"]["passed"], 0)
            self.assertEqual(summary["augmentation_report"]["checks_summary"]["failed"], 0)
            self.assertTrue(list((scene_dir / "annotated_tiles").glob("*_overlay.png")))
            self.assertTrue(list((scene_dir / "annotated_augmentations").glob("*_overlay.png")))
            self.assertEqual(sorted(path.name for path in root.glob("*.tif")), ["scene.tif"])

    def test_preview_is_lightweight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture(root)
            body = preview_annotated_tile_report(
                root,
                tile_size=256,
                stride=256,
                max_records_preview=5,
                include_annotation_summary=True,
                include_augmentation_catalog=True,
            )
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["mask_visualization"]["mode"], "dashed_contour")
            self.assertIn("annotation", body)
            self.assertIn("classification_summary", body)
            self.assertIn("flip_horizontal", body["augmentation_operations_supported"])
            self.assertNotIn("array", json.dumps(body).lower())

    @unittest.skipUnless(HAS_FASTAPI, "fastapi is required")
    def test_annotated_preview_endpoint_is_env_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture(root)
            payload = {
                "input_dir": str(root),
                "tile_size": 256,
                "stride": 256,
                "positive_stride_factor": 1.0,
                "hard_negative_stride_factor": 1.0,
                "negative_stride_factor": 1.0,
                "min_positive_pixels": 1,
                "max_scenes": 1,
                "max_records_preview": 5,
                "include_annotation_summary": True,
                "include_augmentation_catalog": True,
            }

            os.environ.pop("MLSYSTEM_DEBUG_DATASET_ENDPOINTS", None)
            import mlsystem.src.api.app as api_app

            api_app = importlib.reload(api_app)
            self.assertEqual(TestClient(api_app.app).post("/api/debug/annotated-tile-report/preview", json=payload).status_code, 404)

            os.environ["MLSYSTEM_DEBUG_DATASET_ENDPOINTS"] = "1"
            api_app = importlib.reload(api_app)
            response = TestClient(api_app.app).post("/api/debug/annotated-tile-report/preview", json=payload)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["mask_visualization"]["mode"], "dashed_contour")
            self.assertIn("annotation", body)
            self.assertIn("classification_summary", body)
            self.assertIn("flip_horizontal", body["augmentation_operations_supported"])
            self.assertNotIn("array", json.dumps(body).lower())
            os.environ.pop("MLSYSTEM_DEBUG_DATASET_ENDPOINTS", None)
            importlib.reload(api_app)


def _write_fixture(root: Path) -> None:
    width = height = 1024
    y, x = np.mgrid[0:height, 0:width]
    data = np.stack([(x % 255), (y % 255), ((x + y) % 255)], axis=0).astype("uint8")
    with rasterio.open(
        root / "scene.tif",
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
    polygon = Polygon([(300, 700), (460, 700), (460, 540), (300, 540)])
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [{"type": "Feature", "properties": {}, "geometry": mapping(polygon)}],
    }
    (root / "scene.geojson").write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
