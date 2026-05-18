from __future__ import annotations

import unittest

import numpy as np
from rasterio.transform import from_origin

from mlsystem.src.contracts.prediction import ProbabilityMap
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

    def test_epsg4326_mask_is_reprojected_to_metric_crs(self) -> None:
        prob = np.ones((4, 4), dtype="float32")
        result = vectorize_probability_map(
            ProbabilityMap(
                scene_id="s",
                prob=prob,
                weight_sum=np.ones_like(prob),
                coverage_mask=np.ones_like(prob, dtype=bool),
                coverage_fraction=1.0,
                transform=from_origin(42.0, 49.0, 0.0001, 0.0001),
                crs="EPSG:4326",
            ),
            scene_name="s",
            threshold=0.5,
        )
        self.assertEqual(result.crs, "EPSG:3857")
        self.assertEqual(result.raw_count, 1)
        self.assertGreater(result.features_raw[0]["properties"]["area_m2"], 1000)


if __name__ == "__main__":
    unittest.main()
