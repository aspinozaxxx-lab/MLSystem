from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import box, mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlsystem.src.data.virtual_tile_sampling import preview_from_manifest
from mlsystem.src.tile_preparation.windows import generate_window_grid_for_scene


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible virtual tile sampling self-checks on synthetic data.")
    parser.add_argument("--output-json", default="outputs/debug_virtual_tile_sampling/parameter_matrix.json")
    args = parser.parse_args()

    payload = {
        "status": "ok",
        "tiling_checks": _tiling_checks(),
        "parameter_matrix": _parameter_matrix(),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    failed = [
        row
        for row in payload["tiling_checks"]
        if not bool(row.get("pass"))
    ]
    return 1 if failed else 0


def _tiling_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for factor, expected_stride, expected_n in ((1.0, 256, 4), (0.5, 128, 7), (0.25, 64, 13)):
        windows = generate_window_grid_for_scene(1024, 1024, 256, expected_stride, scene_id="synthetic")
        xs = sorted({window.x for window in windows})
        ys = sorted({window.y for window in windows})
        checks.append(
            {
                "case": "dense_stride",
                "scene_width": 1024,
                "scene_height": 1024,
                "tile_size": 256,
                "base_stride": 256,
                "stride_factor": factor,
                "effective_stride": expected_stride,
                "expected_nx": expected_n,
                "expected_ny": expected_n,
                "expected_total": expected_n * expected_n,
                "actual_nx": len(xs),
                "actual_ny": len(ys),
                "actual_total": len(windows),
                "pass": len(xs) == expected_n and len(ys) == expected_n and len(windows) == expected_n * expected_n,
            }
        )

    edge_windows = generate_window_grid_for_scene(1000, 1000, 256, 256, scene_id="edge")
    xs = sorted({window.x for window in edge_windows})
    ys = sorted({window.y for window in edge_windows})
    checks.append(
        {
            "case": "edge_coverage",
            "scene_width": 1000,
            "scene_height": 1000,
            "tile_size": 256,
            "base_stride": 256,
            "stride_factor": 1.0,
            "effective_stride": 256,
            "expected_last_origin": 744,
            "actual_x_origins": xs,
            "actual_y_origins": ys,
            "coverage_x": max(window.x + window.width for window in edge_windows),
            "coverage_y": max(window.y + window.height for window in edge_windows),
            "windows_out_of_bounds": [
                {"x": window.x, "y": window.y, "width": window.width, "height": window.height}
                for window in edge_windows
                if window.x < 0 or window.y < 0 or window.x + window.width > 1000 or window.y + window.height > 1000
            ],
            "pass": xs[-1] == 744
            and ys[-1] == 744
            and max(window.x + window.width for window in edge_windows) == 1000
            and max(window.y + window.height for window in edge_windows) == 1000
            and all(window.x + window.width <= 1000 and window.y + window.height <= 1000 for window in edge_windows),
        }
    )
    return checks


def _parameter_matrix() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest, annotation, images_dir = _write_fixture(root)
        configs = {
            "A_baseline": {
                "preprocess": {
                    "tile_size": 256,
                    "stride": 256,
                    "train_sampling": {"enabled": True},
                }
            },
            "B_positive_dense": {
                "preprocess": {
                    "tile_size": 256,
                    "stride": 256,
                    "train_sampling": {
                        "enabled": True,
                        "positive_stride_factor": 0.5,
                        "hard_negative_stride_factor": 1.0,
                        "negative_stride_factor": 1.0,
                    },
                }
            },
            "C_stronger_positive_dense": {
                "preprocess": {
                    "tile_size": 256,
                    "stride": 256,
                    "train_sampling": {
                        "enabled": True,
                        "positive_stride_factor": 0.25,
                        "hard_negative_stride_factor": 1.0,
                        "negative_stride_factor": 1.0,
                    },
                }
            },
            "D_repeats": {
                "preprocess": {
                    "tile_size": 256,
                    "stride": 256,
                    "train_sampling": {
                        "enabled": True,
                        "positive_repeat_factor": 4,
                        "hard_negative_repeat_factor": 2,
                        "negative_repeat_factor": 1,
                    },
                }
            },
            "E_empty_limit": {
                "preprocess": {
                    "tile_size": 256,
                    "stride": 256,
                    "train_sampling": {
                        "enabled": True,
                        "max_empty_tile_share": 0.35,
                    },
                }
            },
        }
        matrix: dict[str, Any] = {}
        for name, config in configs.items():
            result = preview_from_manifest(
                dataset_manifest=manifest,
                annotation=annotation,
                images_dir=images_dir,
                config=config,
                max_records_preview=5,
            )
            summary = result["summary"]
            matrix[name] = {
                "base_train_records": summary["base_train_records"],
                "virtual_train_records": summary["virtual_train_records"],
                "effective_train_samples_per_epoch": summary["effective_train_samples_per_epoch"],
                "train_positive_tiles": summary["train_positive_tiles"],
                "train_partial_positive_tiles": summary["train_partial_positive_tiles"],
                "train_hard_negative_tiles": summary["train_hard_negative_tiles"],
                "train_negative_tiles": summary["train_negative_tiles"],
                "positive_stride": summary["positive_stride"],
                "hard_negative_stride": summary["hard_negative_stride"],
                "negative_stride": summary["negative_stride"],
                "warnings": summary["warnings"],
            }
        return matrix


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    import rasterio
    from rasterio.transform import from_origin

    images_dir = root / "images"
    images_dir.mkdir()
    data = np.ones((4, 1024, 1024), dtype="uint16")
    for name in ("train_scene.tif", "val_scene.tif"):
        with rasterio.open(
            images_dir / name,
            "w",
            driver="GTiff",
            width=1024,
            height=1024,
            count=4,
            dtype="uint16",
            transform=from_origin(0, 1024, 1, 1),
            crs="EPSG:3857",
        ) as ds:
            ds.write(data)
    annotation = root / "annotation.geojson"
    annotation.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": mapping(box(96, 760, 180, 844))},
                    {"type": "Feature", "properties": {}, "geometry": mapping(box(578, 220, 690, 340))},
                    {"type": "Feature", "properties": {}, "geometry": mapping(box(820, 820, 900, 900))},
                ],
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
    raise SystemExit(main())
