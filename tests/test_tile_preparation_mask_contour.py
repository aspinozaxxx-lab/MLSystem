from __future__ import annotations

import unittest

import numpy as np

from mlsystem.src.tile_preparation.report import dashed_boundary, dilate_binary, mask_boundary, overlay_mask_contour


class TilePreparationMaskContourTests(unittest.TestCase):
    def test_mask_boundary_on_rectangle(self) -> None:
        mask = np.zeros((12, 12), dtype="uint8")
        mask[3:9, 4:10] = 1

        boundary = mask_boundary(mask)

        self.assertTrue(boundary.any())
        self.assertTrue(boundary[3, 4])
        self.assertTrue(boundary[8, 9])
        self.assertFalse(boundary[5, 6])
        self.assertFalse(boundary[0, 0])

    def test_dashed_contour_is_non_empty_subset(self) -> None:
        mask = np.zeros((32, 32), dtype="uint8")
        mask[6:26, 8:24] = 1
        boundary = mask_boundary(mask)
        dashed = dashed_boundary(boundary, dash=4, gap=3)

        self.assertTrue(dashed.any())
        self.assertLessEqual(int(dashed.sum()), int(boundary.sum()))

    def test_dilate_binary_increases_boundary_width(self) -> None:
        mask = np.zeros((16, 16), dtype="uint8")
        mask[4:12, 4:12] = 1
        boundary = mask_boundary(mask)
        dilated = dilate_binary(boundary, radius=1)

        self.assertGreater(int(dilated.sum()), int(boundary.sum()))

    def test_overlay_mask_contour_shape_dtype_and_pixels(self) -> None:
        rgb = np.full((24, 24, 3), 80, dtype="uint8")
        mask = np.zeros((24, 24), dtype="uint8")
        mask[5:19, 7:17] = 1

        overlay = overlay_mask_contour(rgb, mask, dash=5, gap=2, width=2)

        self.assertEqual(overlay.shape, rgb.shape)
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreater(int(np.count_nonzero(np.all(overlay == np.array([255, 0, 0], dtype="uint8"), axis=2))), 0)
        self.assertTrue(np.array_equal(overlay[0, 0], rgb[0, 0]))


if __name__ == "__main__":
    unittest.main()
