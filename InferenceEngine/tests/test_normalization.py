from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from InferenceEngine.src.inference_engine.preprocessing.normalization import normalize_image


class NormalizationTests(unittest.TestCase):
    def test_normalize_image_uses_bounded_percentile_sample(self) -> None:
        arr = np.arange(4 * 64 * 64, dtype=np.float32).reshape(4, 64, 64)
        with patch.dict(os.environ, {"INFERENCE_ENGINE_NORMALIZE_MAX_SAMPLES": "128"}):
            out = normalize_image(arr)

        self.assertEqual(out.shape, arr.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreater(float(out.max()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
