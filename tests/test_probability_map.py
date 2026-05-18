from __future__ import annotations

import unittest

import numpy as np

from mlsystem.src.inference.probability_map import ProbabilityMapAccumulator, ProbabilityMapConfig, coverage_stats
from InferenceEngine.src.inference_engine.tiling.windows import window_grid


class ProbabilityMapTests(unittest.TestCase):
    def test_accumulator_full_coverage(self) -> None:
        acc = ProbabilityMapAccumulator("s", ProbabilityMapConfig(scene_width=9, scene_height=7, patch_size=4, mode="weighted_overlap", context_bounds=1))
        for window in window_grid(9, 7, 4, 3, "s"):
            acc.add_tile(window, np.ones((window.height, window.width), dtype="float32"))
        result = acc.finalize()
        self.assertEqual(result.coverage_fraction, 1.0)
        self.assertTrue(np.all(result.weight_sum > 0))
        self.assertTrue(np.allclose(result.prob, 1.0))
        self.assertEqual(coverage_stats(result)["covered_pixels"], 63)


if __name__ == "__main__":
    unittest.main()
