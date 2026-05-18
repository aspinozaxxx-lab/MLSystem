from __future__ import annotations

import unittest

from InferenceEngine.src.inference_engine.tiling.windows import origins, tile_insert_slices, window_grid


class WindowGenerationTests(unittest.TestCase):
    def test_edge_windows_are_added_for_non_divisible_scene(self) -> None:
        self.assertEqual(origins(10, 4, 3), [0, 3, 6])
        windows = window_grid(10, 9, 4, 3, scene_id="s")
        self.assertIn((6, 5), {(w.x, w.y) for w in windows})

    def test_stride_smaller_than_tile_covers_scene_edges(self) -> None:
        windows = window_grid(13, 11, 6, 4, scene_id="s")
        self.assertEqual(max(w.x + w.width for w in windows), 13)
        self.assertEqual(max(w.y + w.height for w in windows), 11)

    def test_small_scene_uses_origin_window(self) -> None:
        windows = window_grid(3, 2, 8, 4, scene_id="s")
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0].x, windows[0].y, windows[0].width, windows[0].height), (0, 0, 3, 2))

    def test_center_crop_offset_has_no_gap_between_neighbors(self) -> None:
        left = tile_insert_slices(0, 0, 6, 6, 10, 10, patch_size=6, crop_mode="center", center_size=4, context_bounds=1)
        right = tile_insert_slices(4, 0, 6, 6, 10, 10, patch_size=6, crop_mode="center", center_size=4, context_bounds=1)
        self.assertEqual(left["insert_x"] + left["insert_width"], right["insert_x"])
        self.assertEqual(right["insert_x"], 5)


if __name__ == "__main__":
    unittest.main()
