from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from affine import Affine

from ..storage.local_io import write_json
from ..storage.probability_artifacts import load_probability_array, probability_artifact_info
from .contracts import PredictionTileInfo


def build_prediction_tile_index(manifest_path: str | Path, output_dir: str | Path) -> list[PredictionTileInfo]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    scene_rows = manifest.get("scenes") or manifest.get("scene_results") or []
    tiles: list[PredictionTileInfo] = []
    for idx, row in enumerate(scene_rows):
        meta_path = Path(str(row.get("meta_path") or ""))
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        npz_path = Path(str(row.get("npz_path") or meta_path.with_suffix(".npz")))
        if not npz_path.exists():
            continue
        height, width, band_name = probability_artifact_info(npz_path, probability_band=meta.get("probability_band"))
        transform_values = meta.get("transform")
        if not transform_values:
            continue
        transform = Affine(*[float(v) for v in transform_values[:6]])
        bounds = window_bounds(transform, 0, 0, width, height)
        tiles.append(
            PredictionTileInfo(
                tile_id=f"{idx:06d}",
                scene_id=str(meta.get("scene_id") or row.get("scene_id") or idx),
                scene_name=str(meta.get("scene_name") or row.get("scene_name") or meta.get("scene_id") or idx),
                npz_path=str(npz_path),
                meta_path=str(meta_path),
                bounds=bounds,
                transform=tuple(float(v) for v in transform_values[:6]),  # type: ignore[arg-type]
                crs=meta.get("crs"),
                width=int(width),
                height=int(height),
                probability_band=band_name,
                metadata={"coverage_fraction": meta.get("coverage_fraction"), "manifest_row": idx},
            )
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "prediction_tiles_manifest.json", {"tiles_total": len(tiles), "tiles": [tile.to_dict() for tile in tiles]})
    lines = ["tile_id\tscene_id\twidth\theight\tcrs\tnpz_path"]
    for tile in tiles:
        lines.append(f"{tile.tile_id}\t{tile.scene_id}\t{tile.width}\t{tile.height}\t{tile.crs}\t{tile.npz_path}")
    (output / "prediction_tiles_manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tiles


def window_bounds(transform: Affine, col_off: int, row_off: int, width: int, height: int) -> tuple[float, float, float, float]:
    x0, y0 = transform * (col_off, row_off)
    x1, y1 = transform * (col_off + width, row_off + height)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def load_tile_probability(tile: PredictionTileInfo) -> np.ndarray:
    probability, _band = load_probability_array(tile.npz_path, probability_band=tile.probability_band, as_float=True)
    return probability
