from __future__ import annotations

import random
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def apply_debug_augmentation(rgb: np.ndarray, operation: str, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply deterministic visual-only augmentations to an RGB uint8 preview."""
    rng = random.Random(seed)
    image = Image.fromarray(rgb.astype("uint8"), mode="RGB")
    metadata: dict[str, Any] = {"operation": operation, "seed": seed}

    if operation == "flip_rot90":
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            metadata["flip_horizontal"] = True
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            metadata["flip_vertical"] = True
        rotations = rng.randint(1, 3)
        image = image.rotate(90 * rotations, expand=False)
        metadata["rot90"] = rotations
        return np.asarray(image), metadata

    if operation == "brightness_gamma_noise":
        arr = np.asarray(image).astype("float32") / 255.0
        brightness = 0.85 + rng.random() * 0.30
        contrast = 0.85 + rng.random() * 0.35
        gamma = 0.80 + rng.random() * 0.45
        mean = arr.mean(axis=(0, 1), keepdims=True)
        arr = (arr - mean) * contrast + mean
        arr = np.clip(arr * brightness, 0.0, 1.0)
        arr = np.power(arr, gamma)
        noise = np.random.default_rng(seed).normal(0.0, 0.025, size=arr.shape).astype("float32")
        arr = np.clip(arr + noise, 0.0, 1.0)
        metadata.update({"brightness": round(brightness, 4), "contrast": round(contrast, 4), "gamma": round(gamma, 4), "noise_std": 0.025})
        return (arr * 255.0).astype("uint8"), metadata

    if operation == "blur_cutout":
        image = image.filter(ImageFilter.BoxBlur(radius=1))
        arr = np.asarray(image).copy()
        h, w = arr.shape[:2]
        cut_h = max(8, h // 8)
        cut_w = max(8, w // 8)
        x0 = rng.randint(0, max(0, w - cut_w))
        y0 = rng.randint(0, max(0, h - cut_h))
        arr[y0 : y0 + cut_h, x0 : x0 + cut_w, :] = 0
        metadata.update({"blur": "box_radius_1", "cutout": {"x": x0, "y": y0, "width": cut_w, "height": cut_h}})
        return arr.astype("uint8"), metadata

    raise ValueError(f"Unsupported debug augmentation operation: {operation}")
