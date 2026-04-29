from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pipeline.contracts import ProbabilityMap, PredictionTile, TileWindow
from ..tiling.windows import tile_insert_slices, tile_weight_window


@dataclass(frozen=True)
class ProbabilityMapConfig:
    scene_width: int
    scene_height: int
    patch_size: int
    mode: str = "hard_insert"
    crop_mode: str = "full"
    center_size: int | None = None
    context_bounds: int | None = None


class ProbabilityMapAccumulator:
    def __init__(self, scene_id: str, config: ProbabilityMapConfig) -> None:
        self.scene_id = scene_id
        self.config = config
        self.prob_sum = np.zeros((config.scene_height, config.scene_width), dtype="float32")
        self.weight_sum = np.zeros((config.scene_height, config.scene_width), dtype="float32")

    def add_tile(self, window: TileWindow, prob: np.ndarray) -> dict[str, int]:
        actual_h = min(window.height, prob.shape[0], self.config.scene_height - window.y)
        actual_w = min(window.width, prob.shape[1], self.config.scene_width - window.x)
        prob_full = prob[:actual_h, :actual_w].astype("float32", copy=False)

        if self.config.mode == "weighted_overlap":
            insert = {
                "crop_x0": 0,
                "crop_y0": 0,
                "crop_x1": int(actual_w),
                "crop_y1": int(actual_h),
                "insert_x": int(window.x),
                "insert_y": int(window.y),
                "insert_width": int(actual_w),
                "insert_height": int(actual_h),
            }
            weight_insert = tile_weight_window(
                window.x,
                window.y,
                actual_w,
                actual_h,
                self.config.scene_width,
                self.config.scene_height,
                patch_size=self.config.patch_size,
                center_size=self.config.center_size,
                context_bounds=self.config.context_bounds,
            )
            prob_insert = prob_full
        else:
            insert = tile_insert_slices(
                window.x,
                window.y,
                actual_w,
                actual_h,
                self.config.scene_width,
                self.config.scene_height,
                patch_size=self.config.patch_size,
                crop_mode=self.config.crop_mode,
                center_size=self.config.center_size,
                context_bounds=self.config.context_bounds,
            )
            prob_insert = prob_full[insert["crop_y0"] : insert["crop_y1"], insert["crop_x0"] : insert["crop_x1"]]
            weight_insert = np.ones_like(prob_insert, dtype="float32")

        ix = insert["insert_x"]
        iy = insert["insert_y"]
        iw = insert["insert_width"]
        ih = insert["insert_height"]
        if iw > 0 and ih > 0:
            self.prob_sum[iy : iy + ih, ix : ix + iw] += prob_insert * weight_insert
            self.weight_sum[iy : iy + ih, ix : ix + iw] += weight_insert
        return insert

    def add_prediction_tile(self, tile: PredictionTile) -> dict[str, int]:
        return self.add_tile(tile.window, tile.prob)

    def finalize(self, transform: object | None = None, crs: str | None = None) -> ProbabilityMap:
        coverage_mask = self.weight_sum > 0
        prob = np.zeros_like(self.prob_sum, dtype="float32")
        prob[coverage_mask] = self.prob_sum[coverage_mask] / np.maximum(self.weight_sum[coverage_mask], 1e-6)
        coverage_fraction = float(np.count_nonzero(coverage_mask) / max(1, self.config.scene_width * self.config.scene_height))
        return ProbabilityMap(
            scene_id=self.scene_id,
            prob=prob,
            weight_sum=self.weight_sum.copy(),
            coverage_mask=coverage_mask,
            coverage_fraction=coverage_fraction,
            transform=transform,
            crs=crs,
            metadata={"stitch_mode": self.config.mode, "crop_mode": self.config.crop_mode},
        )


def coverage_stats(probability_map: ProbabilityMap) -> dict[str, float | int]:
    return {
        "coverage_fraction": probability_map.coverage_fraction,
        "covered_pixels": int(np.count_nonzero(probability_map.coverage_mask)),
        "total_pixels": int(probability_map.coverage_mask.size),
        "min_weight_sum": float(probability_map.weight_sum.min()) if probability_map.weight_sum.size else 0.0,
        "max_weight_sum": float(probability_map.weight_sum.max()) if probability_map.weight_sum.size else 0.0,
    }


__all__ = ["ProbabilityMap", "ProbabilityMapAccumulator", "ProbabilityMapConfig", "coverage_stats"]
