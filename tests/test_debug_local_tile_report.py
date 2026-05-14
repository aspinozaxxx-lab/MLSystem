from __future__ import annotations

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mlsystem.src.data.local_tile_report import (
    LocalTileReportConfig,
    build_tiling_check,
    generate_local_tile_reports,
)

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except Exception:  # noqa: BLE001
    HAS_FASTAPI = False


class DebugLocalTileReportTests(unittest.TestCase):
    def test_tiling_formula_for_dense_factors(self) -> None:
        cases = [
            (1.0, 256, 16),
            (0.5, 128, 49),
            (0.25, 64, 169),
        ]
        for factor, expected_stride, expected_total in cases:
            with self.subTest(factor=factor):
                check = build_tiling_check(1024, 1024, 256, 256, factor)
                self.assertEqual(check["effective_stride"], expected_stride)
                self.assertEqual(check["expected_total"], expected_total)
                self.assertEqual(check["actual_total"], expected_total)
                self.assertTrue(check["pass"])

    def test_edge_coverage_for_non_divisible_scene(self) -> None:
        check = build_tiling_check(1000, 1000, 256, 256, 1.0)
        self.assertEqual(check["last_x"], 744)
        self.assertEqual(check["last_y"], 744)
        self.assertEqual(check["coverage_x"], 1000)
        self.assertEqual(check["coverage_y"], 1000)
        self.assertEqual(check["out_of_bounds_windows"], 0)
        self.assertTrue(check["pass"])

    @unittest.skipUnless(HAS_RASTERIO, "rasterio is required for report generation")
    def test_html_report_generation_creates_expected_debug_artifacts_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "scene.tif"
            _write_raster(image, width=1024, height=1024)

            result = generate_local_tile_reports(
                LocalTileReportConfig(
                    images_dir=root,
                    tile_size=256,
                    stride=256,
                    max_overview_size=512,
                    max_tile_examples=4,
                    max_augmentation_tiles=2,
                    augmentation_mode="individual",
                    augmentations=["flip_horizontal", "brightness"],
                    augmentation_seed=7,
                    seed=7,
                )
            )

            scene_dir = root / "scene"
            self.assertEqual(result["scene_count"], 1)
            self.assertTrue((scene_dir / "tile_sampling_report.html").exists())
            self.assertTrue((scene_dir / "scene_summary.json").exists())
            self.assertTrue((scene_dir / "overview_grid_base.png").exists())
            self.assertTrue((root / "tile_sampling_index.html").exists())
            summary = json.loads((scene_dir / "scene_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(summary["example_tiles"]), 4)
            self.assertTrue((scene_dir / summary["example_tiles"][0]["preview_path"]).exists())
            self.assertIn("augmentation_report", summary)
            self.assertEqual(summary["augmentation_report"]["operations"], ["flip_horizontal", "brightness"])
            self.assertEqual(summary["augmentation_report"]["total_augmentation_previews"], 4)
            self.assertEqual(sum(len(tile["augmentations"]) for tile in summary["example_tiles"]), 4)
            html = (scene_dir / "tile_sampling_report.html").read_text(encoding="utf-8")
            self.assertIn("Аугментации: покрытие методов", html)
            self.assertIn("Примеры всех аугментаций", html)
            self.assertIn("Проверки аугментаций", html)
            self.assertEqual(sorted(path.name for path in root.rglob("*.tif")), ["scene.tif"])

    @unittest.skipUnless(HAS_RASTERIO and HAS_FASTAPI, "rasterio and fastapi are required for API smoke tests")
    def test_local_tile_report_endpoint_is_env_gated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_raster(root / "scene.tif", width=1024, height=1024)
            payload = {
                "images_dir": str(root),
                "tile_size": 256,
                "stride": 256,
                "stride_factors": [1.0, 0.5, 0.25],
                "max_scenes": 1,
                "max_records_preview": 3,
                "include_augmentation_catalog": True,
            }

            os.environ.pop("MLSYSTEM_DEBUG_DATASET_ENDPOINTS", None)
            import mlsystem.src.api.app as api_app

            api_app = importlib.reload(api_app)
            disabled_client = TestClient(api_app.app)
            self.assertEqual(disabled_client.post("/api/debug/local-tile-report/preview", json=payload).status_code, 404)

            os.environ["MLSYSTEM_DEBUG_DATASET_ENDPOINTS"] = "1"
            api_app = importlib.reload(api_app)
            enabled_client = TestClient(api_app.app)
            response = enabled_client.post("/api/debug/local-tile-report/preview", json=payload)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "ok")
            self.assertEqual(body["summary"]["scene_count"], 1)
            self.assertEqual(body["scenes"][0]["tiling_checks"][2]["actual_total"], 169)
            self.assertIn("flip_horizontal", body["summary"]["augmentation_operations_supported"])
            self.assertIn("flips", body["summary"]["training_augmentation_keys_supported"])
            self.assertFalse(body["summary"]["augmentation_report_available"])
            self.assertNotIn("array", json.dumps(body).lower())
            os.environ.pop("MLSYSTEM_DEBUG_DATASET_ENDPOINTS", None)
            importlib.reload(api_app)


def _write_raster(path: Path, *, width: int, height: int) -> None:
    data = np.ones((3, height, width), dtype="uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint16",
        transform=from_origin(0, height, 1, 1),
        crs="EPSG:3857",
    ) as ds:
        ds.write(data)


if __name__ == "__main__":
    unittest.main()
