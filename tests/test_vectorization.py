from __future__ import annotations

import unittest

import numpy as np
from rasterio.transform import from_origin

from mlsystem.src.pipeline.contracts import ProbabilityMap
from mlsystem.src.postprocessing.vectorization import vectorize_probability_map


class VectorizationTests(unittest.TestCase):
    def test_simple_mask_to_polygon(self) -> None:
        prob = np.zeros((8, 8), dtype="float32")
        prob[2:6, 2:6] = 0.9
        result = vectorize_probability_map(
            ProbabilityMap(
                scene_id="s",
                prob=prob,
                weight_sum=np.ones_like(prob),
                coverage_mask=np.ones_like(prob, dtype=bool),
                coverage_fraction=1.0,
                transform=from_origin(0, 8, 1, 1),
                crs="EPSG:3857",
            ),
            scene_name="s",
            threshold=0.5,
        )
        self.assertEqual(result.raw_count, 1)
        self.assertGreater(result.vertices_before, 0)


if __name__ == "__main__":
    unittest.main()
