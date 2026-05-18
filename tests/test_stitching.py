from __future__ import annotations

import unittest

import numpy as np

from InferenceEngine.src.inference_engine.probability.map import ProbabilityMapAccumulator, ProbabilityMapConfig
from InferenceEngine.src.inference_engine.tiling.windows import window_grid


class StitchingTests(unittest.TestCase):
    def test_weight_sum_covers_non_divisible_scene(self) -> None:
        windows = window_grid(13, 11, 6, 4, scene_id="s")
        acc = ProbabilityMapAccumulator("s", ProbabilityMapConfig(scene_width=13, scene_height=11, patch_size=6, mode="weighted_overlap"))
        for window in windows:
            acc.add_tile(window, np.ones((window.height, window.width), dtype="float32"))
        result = acc.finalize()
        self.assertEqual(result.coverage_fraction, 1.0)
        self.assertTrue(np.all(result.weight_sum > 0))

    def test_full_tile_insert_has_no_holes(self) -> None:
        windows = window_grid(10, 9, 4, 3, scene_id="s")
        acc = ProbabilityMapAccumulator("s", ProbabilityMapConfig(scene_width=10, scene_height=9, patch_size=4, mode="hard_insert", crop_mode="full"))
        for window in windows:
            acc.add_tile(window, np.full((window.height, window.width), 0.25, dtype="float32"))
        result = acc.finalize()
        self.assertEqual(result.coverage_fraction, 1.0)
        self.assertTrue(np.allclose(result.prob, 0.25))

    def test_center_crop_insert_covers_scene_with_overlap_stride(self) -> None:
        windows = window_grid(10, 10, 6, 4, scene_id="s")
        acc = ProbabilityMapAccumulator(
            "s",
            ProbabilityMapConfig(scene_width=10, scene_height=10, patch_size=6, mode="hard_insert", crop_mode="center", center_size=4, context_bounds=1),
        )
        for window in windows:
            acc.add_tile(window, np.ones((window.height, window.width), dtype="float32"))
        result = acc.finalize()
        self.assertEqual(result.coverage_fraction, 1.0)
        self.assertTrue(np.all(result.weight_sum > 0))

    def test_synthetic_probability_map_has_no_constant_tile_edges(self) -> None:
        windows = window_grid(15, 12, 7, 5, scene_id="s")
        acc = ProbabilityMapAccumulator("s", ProbabilityMapConfig(scene_width=15, scene_height=12, patch_size=7, mode="weighted_overlap", context_bounds=2))
        for window in windows:
            acc.add_tile(window, np.full((window.height, window.width), 0.7, dtype="float32"))
        result = acc.finalize()
        self.assertTrue(np.allclose(result.prob, 0.7, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
