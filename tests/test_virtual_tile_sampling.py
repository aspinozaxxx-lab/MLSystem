from __future__ import annotations

import json
import importlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box, mapping

from mlsystem.src.data.virtual_tile_sampling import (
    TileSampleRecord,
    TrainSamplingConfig,
    apply_virtual_repeats,
    build_balanced_epoch_indices,
    build_validation_records,
    classify_hard_negative,
    classify_tile_by_mask,
    generate_window_grid_for_scene,
    limit_empty_tile_share,
    preview_from_manifest,
    resolve_train_sampling_config,
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


class VirtualTileSamplingTests(unittest.TestCase):
    def test_dense_stride_factors_increase_window_count(self) -> None:
        base = len(generate_window_grid_for_scene(64, 64, 16, 16, scene_id="s"))
        dense = len(generate_window_grid_for_scene(64, 64, 16, 8, scene_id="s"))
        denser = len(generate_window_grid_for_scene(64, 64, 16, 4, scene_id="s"))

        self.assertEqual(base, 16)
        self.assertGreater(dense, base)
        self.assertGreater(denser, dense)

        cfg = resolve_train_sampling_config(
            {"train_sampling": {"enabled": True, "positive_stride_factor": 0.5, "negative_stride_factor": 0.25}}
        )
        self.assertEqual(cfg.effective_positive_stride(16), 8)
        self.assertEqual(cfg.effective_negative_stride(16), 4)

    def test_dense_stride_formula_for_1024_scene(self) -> None:
        cases = [
            (1.0, 256, 4, 16),
            (0.5, 128, 7, 49),
            (0.25, 64, 13, 169),
        ]
        for factor, effective_stride, expected_axis, expected_total in cases:
            with self.subTest(factor=factor):
                windows = generate_window_grid_for_scene(1024, 1024, 256, effective_stride, scene_id="s")
                xs = {window.x for window in windows}
                ys = {window.y for window in windows}
                self.assertEqual(len(xs), expected_axis)
                self.assertEqual(len(ys), expected_axis)
                self.assertEqual(len(windows), expected_total)

    def test_edge_origin_covers_non_divisible_scene_without_oob_windows(self) -> None:
        windows = generate_window_grid_for_scene(1000, 1000, 256, 256, scene_id="s")
        xs = sorted({window.x for window in windows})
        ys = sorted({window.y for window in windows})

        self.assertEqual(xs[-1], 744)
        self.assertEqual(ys[-1], 744)
        self.assertEqual(max(window.x + window.width for window in windows), 1000)
        self.assertEqual(max(window.y + window.height for window in windows), 1000)
        self.assertTrue(all(window.x + window.width <= 1000 for window in windows))
        self.assertTrue(all(window.y + window.height <= 1000 for window in windows))

    def test_classification_thresholds(self) -> None:
        zero = np.zeros((8, 8), dtype="uint8")
        self.assertEqual(classify_tile_by_mask(zero, min_positive_pixels=2)[0], "negative")

        partial = zero.copy()
        partial[1, 1] = 1
        self.assertEqual(classify_tile_by_mask(partial, min_positive_pixels=2)[0], "partial_positive")

        positive = zero.copy()
        positive[1, 1] = 1
        positive[2, 2] = 1
        self.assertEqual(classify_tile_by_mask(positive, min_positive_pixels=2)[0], "positive")

    def test_hard_negative_near_positive_bbox(self) -> None:
        positive_bboxes = [(10, 10, 20, 20)]
        self.assertTrue(classify_hard_negative((22, 10, 30, 18), positive_bboxes, 4))
        self.assertFalse(classify_hard_negative((40, 40, 48, 48), positive_bboxes, 4))

    def test_repeat_factors_do_not_expand_other_kinds(self) -> None:
        records = [
            _record("positive", "p"),
            _record("negative", "n"),
        ]
        cfg = TrainSamplingConfig(positive_repeat_factor=4, negative_repeat_factor=2)

        repeated = apply_virtual_repeats(records, cfg)
        positive = [record for record in repeated if record.kind == "positive"]
        negative = [record for record in repeated if record.kind == "negative"]

        self.assertEqual(len(positive), 4)
        self.assertEqual(len(negative), 2)
        self.assertTrue(all(record.base_record_id == records[0].record_id for record in positive))

    def test_max_empty_tile_share_limits_empty_records(self) -> None:
        records = [_record("positive", f"p{i}") for i in range(2)]
        records.extend(_record("negative", f"n{i}") for i in range(10))

        limited, warnings = limit_empty_tile_share(records, 0.5, seed=1)
        empty = [record for record in limited if record.kind in {"negative", "hard_negative"}]
        filled = [record for record in limited if record.kind not in {"negative", "hard_negative"}]

        self.assertEqual(len(filled), 2)
        self.assertLessEqual(len(empty) / len(limited), 0.5)
        self.assertTrue(warnings)

    def test_balanced_epoch_indices_follow_requested_fractions(self) -> None:
        records = [_record("positive", f"p{i}") for i in range(2)]
        records.extend(_record("hard_negative", f"h{i}") for i in range(2))
        records.extend(_record("negative", f"n{i}") for i in range(6))
        cfg = TrainSamplingConfig(
            batch_positive_fraction=0.5,
            batch_hard_negative_fraction=0.25,
            batch_negative_fraction=0.25,
        )

        indices, warnings = build_balanced_epoch_indices(records, cfg, seed=2)
        kinds = [records[index].kind for index in indices]

        self.assertEqual(len(indices), len(records))
        self.assertEqual(kinds.count("positive"), 5)
        self.assertIn(kinds.count("hard_negative"), {2, 3})
        self.assertIn(kinds.count("negative"), {2, 3})
        self.assertFalse(warnings)

    @unittest.skipUnless(HAS_RASTERIO, "rasterio is required for raster smoke tests")
    def test_validation_records_do_not_repeat_or_jitter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "scene.tif"
            _write_raster(image)
            shapes = [box(4, 4, 12, 12)]

            result = build_validation_records(
                [{"scene_id": "scene", "image_path": str(image)}],
                shapes,
                tile_size=16,
                stride=16,
            )

            self.assertTrue(result.records)
            self.assertTrue(all(record.source == "validation_grid" for record in result.records))
            self.assertTrue(all(record.repeat_index is None for record in result.records))
            self.assertTrue(all("jitter" not in record.metadata for record in result.records))

    @unittest.skipUnless(HAS_RASTERIO, "rasterio is required for raster smoke tests")
    def test_disabled_train_sampling_keeps_base_virtual_counts_equal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, annotation, images_dir = _write_preview_fixture(root)
            payload = preview_from_manifest(
                dataset_manifest=manifest,
                annotation=annotation,
                images_dir=images_dir,
                config={"preprocess": {"tile_size": 16, "stride": 16, "train_sampling": {"enabled": False}}},
            )

            summary = payload["summary"]
            self.assertFalse(summary["train_sampling_enabled"])
            self.assertEqual(summary["base_train_records"], summary["virtual_train_records"])
            self.assertEqual(summary["virtual_train_records"], summary["effective_train_samples_per_epoch"])

    @unittest.skipUnless(HAS_RASTERIO, "rasterio is required for raster smoke tests")
    def test_debug_script_smoke_does_not_materialize_tiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, annotation, images_dir = _write_preview_fixture(root)
            config = root / "config.json"
            output = root / "preview.json"
            config.write_text(
                json.dumps(
                    {
                        "preprocess": {
                            "tile_size": 16,
                            "stride": 16,
                            "train_sampling": {
                                "enabled": True,
                                "positive_stride_factor": 0.5,
                                "positive_repeat_factor": 2,
                                "max_empty_tile_share": 0.5,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            before_tifs = sorted(path.name for path in root.rglob("*.tif"))

            subprocess.run(
                [
                    sys.executable,
                    "scripts/debug_virtual_tile_dataset.py",
                    "--dataset-manifest",
                    str(manifest),
                    "--annotation",
                    str(annotation),
                    "--images-dir",
                    str(images_dir),
                    "--config-json",
                    str(config),
                    "--output-json",
                    str(output),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            after_tifs = sorted(path.name for path in root.rglob("*.tif"))
            self.assertEqual(after_tifs, before_tifs)
            self.assertTrue(output.exists())
            summary = json.loads(output.read_text(encoding="utf-8"))["summary"]
            self.assertTrue(summary["train_sampling_enabled"])
            self.assertGreaterEqual(summary["virtual_train_records"], summary["base_train_records"])
            self.assertEqual(list(root.rglob("*.npy")), [])
            self.assertEqual(list(root.rglob("*.npz")), [])

    @unittest.skipUnless(HAS_RASTERIO and HAS_FASTAPI, "rasterio and fastapi are required for API preview smoke tests")
    def test_fastapi_debug_endpoint_is_env_gated_and_returns_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, annotation, images_dir = _write_preview_fixture(root)
            payload = {
                "dataset_manifest": str(manifest),
                "annotation": str(annotation),
                "images_dir": str(images_dir),
                "config": {
                    "preprocess": {
                        "tile_size": 16,
                        "stride": 16,
                        "train_sampling": {"enabled": True, "positive_repeat_factor": 2},
                    }
                },
                "max_records_preview": 5,
            }

            os.environ.pop("MLSYSTEM_DEBUG_DATASET_ENDPOINTS", None)
            import mlsystem.src.api.app as api_app

            api_app = importlib.reload(api_app)
            disabled_client = TestClient(api_app.app)
            self.assertEqual(disabled_client.post("/api/debug/virtual-dataset/preview", json=payload).status_code, 404)

            os.environ["MLSYSTEM_DEBUG_DATASET_ENDPOINTS"] = "1"
            api_app = importlib.reload(api_app)
            enabled_client = TestClient(api_app.app)
            response = enabled_client.post("/api/debug/virtual-dataset/preview", json=payload)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "ok")
            self.assertGreater(body["summary"]["virtual_train_records"], 0)
            self.assertTrue(body["preview"])
            self.assertNotIn("array", json.dumps(body["preview"][0]).lower())
            os.environ.pop("MLSYSTEM_DEBUG_DATASET_ENDPOINTS", None)
            importlib.reload(api_app)


def _record(kind: str, record_id: str) -> TileSampleRecord:
    return TileSampleRecord(
        scene_id="scene",
        image_path=None,
        x=0,
        y=0,
        width=16,
        height=16,
        tile_size=16,
        stride=16,
        kind=kind,  # type: ignore[arg-type]
        positive_pixels=1 if kind in {"positive", "partial_positive"} else 0,
        source="test",
        metadata={"record_id": record_id},
    )


def _write_raster(path: Path) -> None:
    data = np.ones((4, 32, 32), dtype="uint16")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=4,
        dtype="uint16",
        transform=from_origin(0, 32, 1, 1),
        crs="EPSG:3857",
    ) as ds:
        ds.write(data)


def _write_preview_fixture(root: Path) -> tuple[Path, Path, Path]:
    images_dir = root / "images"
    images_dir.mkdir()
    train_image = images_dir / "train_scene.tif"
    val_image = images_dir / "val_scene.tif"
    _write_raster(train_image)
    _write_raster(val_image)
    annotation = root / "annotation.geojson"
    annotation.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": mapping(box(4, 4, 12, 12))}],
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "train_scenes": [{"entry": "train_scene.tif", "name": "train_scene.tif", "key": "train_scene.tif"}],
                "val_scenes": [{"entry": "val_scene.tif", "name": "val_scene.tif", "key": "val_scene.tif"}],
            }
        ),
        encoding="utf-8",
    )
    return manifest, annotation, images_dir


if __name__ == "__main__":
    unittest.main()
