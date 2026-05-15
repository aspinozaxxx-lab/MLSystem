from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import Polygon, mapping

    HAS_PROFILE_DEPS = True
except Exception:  # noqa: BLE001
    HAS_PROFILE_DEPS = False


@unittest.skipUnless(HAS_PROFILE_DEPS, "rasterio, shapely, and torch are required")
class TileTrainingProfilerTests(unittest.TestCase):
    def test_profiler_script_writes_summary_for_tiny_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scene(root / "scene.tif")
            _write_geojson(root / "deforestation.geojson")
            (root / "train.txt").write_text("scene.tif\n", encoding="utf-8")
            (root / "val.txt").write_text("scene.tif\n", encoding="utf-8")
            output_dir = root / "profile"

            subprocess.run(
                [
                    sys.executable,
                    "scripts/profile_tile_training_local.py",
                    "--images-root",
                    str(root),
                    "--train-scene-list",
                    str(root / "train.txt"),
                    "--val-scene-list",
                    str(root / "val.txt"),
                    "--annotation",
                    str(root / "deforestation.geojson"),
                    "--tile-size",
                    "128",
                    "--stride",
                    "128",
                    "--augmentation-level",
                    "1",
                    "--batch-size",
                    "1",
                    "--workers",
                    "0",
                    "--max-runtime-sec",
                    "15",
                    "--max-batches",
                    "2",
                    "--warmup-batches",
                    "0",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                timeout=60,
            )

            summary = json.loads((output_dir / "profile_summary.json").read_text(encoding="utf-8"))
            for key in ("batch_wait_mean_sec", "forward_mean_sec", "backward_mean_sec", "total_step_mean_sec"):
                self.assertIn(key, summary)
            self.assertGreaterEqual(summary["combined"]["batches"], 1)
            self.assertTrue((output_dir / "profile_batches.csv").exists())
            self.assertTrue((output_dir / "profile_functions.csv").exists())
            self.assertTrue((output_dir / "profile_report.md").exists())


def _write_scene(path: Path) -> None:
    y, x = np.mgrid[0:256, 0:256]
    data = np.stack([(x % 255), (y % 255), ((x + y) % 255), ((x * 2 + y) % 255)], axis=0).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", width=256, height=256, count=4, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 256, 1, 1)) as ds:
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
