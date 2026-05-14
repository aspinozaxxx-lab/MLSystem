from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mlsystem.src.data.debug_augmentations import (
    AUGMENTATION_REGISTRY,
    DEFAULT_AUGMENTATION_OPERATIONS,
    TRAINING_AUGMENTATION_KEYS,
    apply_debug_augmentation,
    apply_debug_augmentation_with_mask,
    apply_training_augmentation,
)


REQUIRED_OPERATIONS = {
    "original",
    "flip_horizontal",
    "flip_vertical",
    "flip_horizontal_vertical",
    "rot90_90",
    "rot90_180",
    "rot90_270",
    "brightness",
    "contrast",
    "brightness_contrast",
    "gamma_low",
    "gamma_high",
    "noise_low",
    "noise_high",
    "blur_light",
    "blur_strong",
    "cutout_small",
    "cutout_medium",
    "coarse_dropout",
    "color_jitter",
    "training_random_all_enabled_seed_1",
    "training_random_all_enabled_seed_2",
    "training_random_all_enabled_seed_3",
}


class DebugAugmentationsTests(unittest.TestCase):
    def test_registry_contains_required_operations(self) -> None:
        self.assertTrue(REQUIRED_OPERATIONS.issubset(set(DEFAULT_AUGMENTATION_OPERATIONS)))
        self.assertTrue(REQUIRED_OPERATIONS.issubset(set(AUGMENTATION_REGISTRY)))

    def test_training_keys_match_real_train_augmentation_keys(self) -> None:
        source = Path("mlsystem/src/tile_preparation/augmentations.py").read_text(encoding="utf-8")
        for key in TRAINING_AUGMENTATION_KEYS:
            with self.subTest(key=key):
                self.assertIn(f'"{key}"', source)

    def test_geometric_operations_transform_image_and_mask_consistently(self) -> None:
        rgb = _rgb_pattern()
        mask = _mask_pattern()
        cases = {
            "flip_horizontal": (rgb[:, ::-1, :], mask[:, ::-1]),
            "flip_vertical": (rgb[::-1, :, :], mask[::-1, :]),
            "flip_horizontal_vertical": (rgb[::-1, ::-1, :], mask[::-1, ::-1]),
            "rot90_90": (np.rot90(rgb, k=1), np.rot90(mask, k=1)),
            "rot90_180": (np.rot90(rgb, k=2), np.rot90(mask, k=2)),
            "rot90_270": (np.rot90(rgb, k=3), np.rot90(mask, k=3)),
        }
        for operation, (expected_rgb, expected_mask) in cases.items():
            with self.subTest(operation=operation):
                augmented, augmented_mask, metadata = apply_debug_augmentation_with_mask(rgb, operation, seed=42, mask=mask)
                np.testing.assert_array_equal(augmented, expected_rgb)
                np.testing.assert_array_equal(augmented_mask, expected_mask)
                self.assertEqual(metadata["check_status"], "pass")
                self.assertTrue(metadata["checks"]["shape_preserved"])

    def test_image_only_operations_leave_mask_unchanged(self) -> None:
        rgb = _rgb_pattern()
        mask = _mask_pattern()
        operations = [
            "brightness",
            "contrast",
            "brightness_contrast",
            "gamma_low",
            "gamma_high",
            "noise_low",
            "noise_high",
            "blur_light",
            "blur_strong",
            "color_jitter",
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                augmented, augmented_mask, metadata = apply_debug_augmentation_with_mask(rgb, operation, seed=7, mask=mask)
                self.assertEqual(augmented.shape, rgb.shape)
                self.assertEqual(augmented.dtype, np.uint8)
                self.assertGreaterEqual(int(augmented.min()), 0)
                self.assertLessEqual(int(augmented.max()), 255)
                np.testing.assert_array_equal(augmented_mask, mask)
                self.assertNotEqual(metadata["changed_pixels_fraction"], 0.0)
                self.assertEqual(metadata["check_status"], "pass")

    def test_cutout_operations_erase_intersected_mask_pixels(self) -> None:
        rgb = _rgb_pattern()
        mask = np.ones((8, 8), dtype="uint8")
        for operation in ["cutout_small", "cutout_medium", "coarse_dropout"]:
            with self.subTest(operation=operation):
                augmented, augmented_mask, metadata = apply_debug_augmentation_with_mask(rgb, operation, seed=7, mask=mask)
                self.assertEqual(augmented.shape, rgb.shape)
                self.assertEqual(augmented_mask.shape, mask.shape)
                self.assertTrue(set(np.unique(augmented_mask).tolist()).issubset({0, 1}))
                self.assertEqual(metadata["cutout_mask_mode"], "erase")
                self.assertGreater(metadata["mask_erased_pixels"], 0)
                self.assertTrue(metadata["cutout_intersected_positive"])
                self.assertLess(int(augmented_mask.sum()), int(mask.sum()))
                self.assertEqual(metadata["check_status"], "pass")

    def test_cutout_preserve_mode_keeps_mask_for_compatibility(self) -> None:
        rgb = _rgb_pattern()
        mask = np.ones((8, 8), dtype="uint8")
        _augmented, augmented_mask, metadata = apply_debug_augmentation_with_mask(
            rgb,
            "cutout_medium",
            seed=7,
            mask=mask,
            cutout_mask_mode="preserve",
        )
        np.testing.assert_array_equal(augmented_mask, mask)
        self.assertEqual(metadata["cutout_mask_mode"], "preserve")
        self.assertEqual(metadata["mask_erased_pixels"], 0)

    def test_production_cutout_erases_mask_when_applied(self) -> None:
        image = np.ones((4, 32, 32), dtype="float32")
        mask = np.ones((32, 32), dtype="uint8")
        found = False
        for seed in range(100):
            augmented, augmented_mask, metadata = apply_training_augmentation(
                image,
                mask,
                {"cutout": True},
                seed=seed,
                cutout_mask_mode="erase",
            )
            self.assertEqual(augmented.shape, image.shape)
            self.assertEqual(augmented_mask.shape, mask.shape)
            self.assertTrue(set(np.unique(augmented_mask).tolist()).issubset({0, 1}))
            if metadata.get("cutout_applied"):
                found = True
                self.assertEqual(metadata["cutout"]["cutout_mask_mode"], "erase")
                self.assertGreater(metadata["cutout"]["mask_erased_pixels"], 0)
                self.assertLess(int(augmented_mask.sum()), int(mask.sum()))
                break
        self.assertTrue(found)

    def test_all_preview_operations_preserve_shape_dtype_and_range(self) -> None:
        rgb = _rgb_pattern()
        for operation in DEFAULT_AUGMENTATION_OPERATIONS:
            with self.subTest(operation=operation):
                augmented, metadata = apply_debug_augmentation(rgb, operation, seed=11)
                self.assertEqual(augmented.shape, rgb.shape)
                self.assertEqual(augmented.dtype, np.uint8)
                self.assertGreaterEqual(int(augmented.min()), 0)
                self.assertLessEqual(int(augmented.max()), 255)
                self.assertTrue(metadata["checks"]["shape_preserved"])
                self.assertTrue(metadata["checks"]["dtype_uint8"])
                self.assertTrue(metadata["checks"]["range_0_255"])
                self.assertTrue(metadata["checks"]["non_empty_output"])
                self.assertIn(metadata["check_status"], {"pass", "warning"})
                if operation == "original":
                    self.assertEqual(metadata["changed_pixels_fraction"], 0.0)
                else:
                    self.assertGreater(metadata["changed_pixels_fraction"], 0.0)


def _rgb_pattern() -> np.ndarray:
    y, x = np.mgrid[0:8, 0:8]
    return np.stack(
        [
            (x * 23 + y * 5) % 256,
            (x * 7 + y * 31) % 256,
            (x * 19 + y * 11 + 17) % 256,
        ],
        axis=2,
    ).astype("uint8")


def _mask_pattern() -> np.ndarray:
    mask = np.zeros((8, 8), dtype="uint8")
    mask[1:3, 2:6] = 1
    mask[5, 6] = 1
    return mask


if __name__ == "__main__":
    unittest.main()
