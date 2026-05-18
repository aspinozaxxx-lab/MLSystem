from __future__ import annotations

import unittest

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mlsystem.src.debug.pseudolabel_debug import ConstantProbabilityModel
from InferenceEngine.src.inference_engine.workers.scene_inference import run_synthetic_scene_inference


class SceneInferenceTests(unittest.TestCase):
    def test_synthetic_scene_inference_full_coverage(self) -> None:
        scene = np.ones((4, 13, 11), dtype="float32")
        result = run_synthetic_scene_inference(
            scene,
            ConstantProbabilityModel(logit=2.0),
            patch_size=6,
            stride=4,
            stitch_mode="weighted_overlap",
            context_bounds=2,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.coverage_fraction, 1.0)
        self.assertTrue(np.all(result.weight_sum > 0))
        self.assertGreater(float(result.prob.mean()), 0.8)


if __name__ == "__main__":
    unittest.main()
