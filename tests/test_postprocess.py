from __future__ import annotations

import unittest

import numpy as np
from shapely.geometry import box, mapping

from mlsystem.src.postprocessing.filtering import filter_features_by_area, limit_top_features
from mlsystem.src.postprocessing.simplification import simplify_features
from mlsystem.src.postprocessing.thresholding import threshold_probability_map


class PostprocessTests(unittest.TestCase):
    def test_threshold_probability_map(self) -> None:
        mask = threshold_probability_map(np.array([[0.2, 0.5], [0.7, 0.1]], dtype="float32"), 0.5)
        self.assertEqual(mask.tolist(), [[0, 1], [1, 0]])

    def test_area_filter(self) -> None:
        features = [
            {"type": "Feature", "properties": {}, "geometry": mapping(box(0, 0, 1, 1))},
            {"type": "Feature", "properties": {}, "geometry": mapping(box(0, 0, 4, 4))},
        ]
        filtered = filter_features_by_area(features, 5)
        self.assertEqual(len(filtered), 1)

    def test_top_limit_sorts_by_area(self) -> None:
        features = [
            {"type": "Feature", "properties": {"area_m2": 1}, "geometry": mapping(box(0, 0, 1, 1))},
            {"type": "Feature", "properties": {"area_m2": 9}, "geometry": mapping(box(0, 0, 3, 3))},
        ]
        top = limit_top_features(features, 1)
        self.assertEqual(top[0]["properties"]["area_m2"], 9)

    def test_simplify_keeps_valid_feature(self) -> None:
        feature = {"type": "Feature", "properties": {}, "geometry": mapping(box(0, 0, 10, 10))}
        simplified = simplify_features([feature], 1)
        self.assertEqual(len(simplified), 1)


if __name__ == "__main__":
    unittest.main()
