from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, mapping

from mlsystem.src.tile_preparation import AnnotationInput, SceneInput, TilePreparationConfig
from mlsystem.src.tile_preparation.iterator import build_tile_records

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationParallelRecordsTests(unittest.TestCase):
    def test_parallel_and_sequential_record_build_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_a = root / "scene_a.tif"
            scene_b = root / "scene_b.tif"
            geojson_path = root / "ann.geojson"
            _write_scene(scene_a, value=90)
            _write_scene(scene_b, value=120)
            _write_geojson(geojson_path)
            scenes = [SceneInput(scene_a, "scene_a"), SceneInput(scene_b, "scene_b")]
            annotation = AnnotationInput(geojson_path)
            config = TilePreparationConfig(tile_size=64, stride=64, valid_pixel_mode="nonzero_any", max_empty_tile_share=None)

            old_value = os.environ.get("MLSYSTEM_TILE_RECORD_WORKERS")
            try:
                os.environ["MLSYSTEM_TILE_RECORD_WORKERS"] = "0"
                sequential = build_tile_records(scenes, annotation, config)
                os.environ["MLSYSTEM_TILE_RECORD_WORKERS"] = "2"
                parallel = build_tile_records(scenes, annotation, config)
            finally:
                if old_value is None:
                    os.environ.pop("MLSYSTEM_TILE_RECORD_WORKERS", None)
                else:
                    os.environ["MLSYSTEM_TILE_RECORD_WORKERS"] = old_value

            self.assertEqual([record.record_id for record in sequential.base_records], [record.record_id for record in parallel.base_records])
            self.assertEqual([record.kind for record in sequential.base_records], [record.kind for record in parallel.base_records])
            self.assertEqual(len(sequential.records), len(parallel.records))


def _write_scene(path: Path, *, value: int) -> None:
    data = np.zeros((3, 128, 128), dtype="uint8")
    data[:, 16:112, 16:112] = value
    with rasterio.open(path, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 128, 1, 1)) as ds:
        ds.write(data)


def _write_geojson(path: Path) -> None:
    polygon = Polygon([(24, 104), (72, 104), (72, 56), (24, 56)])
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
