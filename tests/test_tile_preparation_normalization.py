from __future__ import annotations

import unittest

import numpy as np

try:
    from mlsystem.src.tile_preparation.config import TilePreparationConfig
    from mlsystem.src.tile_preparation.raster_reader import format_training_image, normalize_band_first, normalize_uint8_255

    HAS_TILE_PREP_DEPS = True
except Exception:  # noqa: BLE001
    HAS_TILE_PREP_DEPS = False


@unittest.skipUnless(HAS_TILE_PREP_DEPS, "tile_preparation dependencies are required")
class TilePreparationNormalizationTests(unittest.TestCase):
    def test_uint8_255_fast_path_outputs_unit_range(self) -> None:
        arr = np.array([[[0, 127], [255, 300]]], dtype="uint16")
        out = normalize_uint8_255(arr)
        self.assertEqual(out.shape, (1, 2, 2))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertAlmostEqual(float(out[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(out[0, 1, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1, 1]), 1.0)

    def test_format_training_image_preserves_chw_shape(self) -> None:
        arr = np.zeros((3, 8, 8), dtype="uint8")
        arr[0] = 255
        config = TilePreparationConfig(tile_size=8, stride=8, normalization_mode="uint8_255")
        out = format_training_image(arr, config)
        self.assertEqual(out.shape, (3, 8, 8))
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out[0].max()), 1.0)

    def test_tile_percentile_legacy_path_still_works(self) -> None:
        arr = np.arange(3 * 8 * 8, dtype="float32").reshape(3, 8, 8)
        out = normalize_band_first(arr)
        self.assertEqual(out.shape, (3, 8, 8))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
