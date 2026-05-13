from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


TRAINING_AUGMENTATION_KEYS = (
    "flips",
    "rot90",
    "brightness_contrast",
    "color_jitter",
    "gamma",
    "noise",
    "blur",
    "cutout",
    "coarse_dropout",
)


@dataclass(frozen=True)
class DebugAugmentationSpec:
    name: str
    group: str
    training_key: str
    description_ru: str
    parameters: dict[str, Any]
    matches_training_semantics: bool = True
    debug_note: str = "visual preview equivalent"
    expects_image_change: bool = True
    mask_transform: str = "identity"


def _spec(
    name: str,
    group: str,
    training_key: str,
    description_ru: str,
    parameters: dict[str, Any] | None = None,
    *,
    matches_training_semantics: bool = True,
    debug_note: str = "visual preview equivalent",
    expects_image_change: bool = True,
    mask_transform: str = "identity",
) -> DebugAugmentationSpec:
    return DebugAugmentationSpec(
        name=name,
        group=group,
        training_key=training_key,
        description_ru=description_ru,
        parameters=parameters or {},
        matches_training_semantics=matches_training_semantics,
        debug_note=debug_note,
        expects_image_change=expects_image_change,
        mask_transform=mask_transform,
    )


AUGMENTATION_REGISTRY: dict[str, DebugAugmentationSpec] = {
    "original": _spec("original", "baseline", "none", "Без аугментации.", expects_image_change=False),
    "flip_horizontal": _spec("flip_horizontal", "geometric", "flips", "Горизонтальное отражение.", mask_transform="flip_horizontal"),
    "flip_vertical": _spec("flip_vertical", "geometric", "flips", "Вертикальное отражение.", mask_transform="flip_vertical"),
    "flip_horizontal_vertical": _spec(
        "flip_horizontal_vertical",
        "geometric",
        "flips",
        "Горизонтальное и вертикальное отражение.",
        mask_transform="flip_horizontal_vertical",
    ),
    "rot90_90": _spec("rot90_90", "geometric", "rot90", "Поворот на 90 градусов.", {"k": 1}, mask_transform="rot90_90"),
    "rot90_180": _spec("rot90_180", "geometric", "rot90", "Поворот на 180 градусов.", {"k": 2}, mask_transform="rot90_180"),
    "rot90_270": _spec("rot90_270", "geometric", "rot90", "Поворот на 270 градусов.", {"k": 3}, mask_transform="rot90_270"),
    "brightness": _spec("brightness", "photometric", "brightness_contrast", "Изменение яркости.", {"factor": 1.18}),
    "contrast": _spec("contrast", "photometric", "brightness_contrast", "Изменение контраста.", {"factor": 1.25}),
    "brightness_contrast": _spec(
        "brightness_contrast",
        "photometric",
        "brightness_contrast",
        "Изменение яркости и контраста.",
        {"brightness": 1.15, "contrast": 1.25},
    ),
    "gamma_low": _spec("gamma_low", "photometric", "gamma", "Gamma ниже 1.0.", {"gamma": 0.8}),
    "gamma_high": _spec("gamma_high", "photometric", "gamma", "Gamma выше 1.0.", {"gamma": 1.25}),
    "noise_low": _spec("noise_low", "noise", "noise", "Слабый гауссов шум.", {"std": 0.015}),
    "noise_high": _spec("noise_high", "noise", "noise", "Сильный гауссов шум.", {"std": 0.045}),
    "blur_light": _spec("blur_light", "blur", "blur", "Легкое размытие.", {"radius": 1}),
    "blur_strong": _spec("blur_strong", "blur", "blur", "Сильное размытие.", {"radius": 2}),
    "cutout_small": _spec("cutout_small", "occlusion/dropout", "cutout", "Малый cutout только по изображению.", {"fraction": 0.12}),
    "cutout_medium": _spec("cutout_medium", "occlusion/dropout", "cutout", "Средний cutout только по изображению.", {"fraction": 0.22}),
    "coarse_dropout": _spec(
        "coarse_dropout",
        "occlusion/dropout",
        "coarse_dropout",
        "Coarse dropout только по изображению.",
        {"holes": 4, "fraction": 0.12},
    ),
    "color_jitter": _spec(
        "color_jitter",
        "photometric",
        "color_jitter",
        "Alias production-пути для brightness_contrast.",
        {"brightness": 1.12, "contrast": 1.20},
        debug_note="debug-only equivalent; production alias: brightness_contrast",
    ),
    "training_random_all_enabled_seed_1": _spec(
        "training_random_all_enabled_seed_1",
        "combined_random_training",
        "all",
        "Случайная train-like комбинация всех включенных групп, seed 1.",
        {"random_seed_alias": 1},
        matches_training_semantics=False,
        debug_note="debug-only numpy/Pillow equivalent of torch batch augmentation",
    ),
    "training_random_all_enabled_seed_2": _spec(
        "training_random_all_enabled_seed_2",
        "combined_random_training",
        "all",
        "Случайная train-like комбинация всех включенных групп, seed 2.",
        {"random_seed_alias": 2},
        matches_training_semantics=False,
        debug_note="debug-only numpy/Pillow equivalent of torch batch augmentation",
    ),
    "training_random_all_enabled_seed_3": _spec(
        "training_random_all_enabled_seed_3",
        "combined_random_training",
        "all",
        "Случайная train-like комбинация всех включенных групп, seed 3.",
        {"random_seed_alias": 3},
        matches_training_semantics=False,
        debug_note="debug-only numpy/Pillow equivalent of torch batch augmentation",
    ),
}


DEFAULT_AUGMENTATION_OPERATIONS = tuple(AUGMENTATION_REGISTRY)


def resolve_augmentation_operations(mode: str = "all", requested: str | list[str] | None = None) -> list[str]:
    if requested:
        values = requested.split(",") if isinstance(requested, str) else requested
        operations = [str(item).strip() for item in values if str(item).strip()]
    elif mode == "individual":
        operations = [name for name in DEFAULT_AUGMENTATION_OPERATIONS if not name.startswith("training_random_all_enabled")]
    elif mode == "production-groups":
        operations = [
            "flip_horizontal",
            "flip_vertical",
            "rot90_90",
            "brightness_contrast",
            "color_jitter",
            "gamma_low",
            "noise_low",
            "blur_light",
            "cutout_medium",
            "coarse_dropout",
        ]
    elif mode == "random-training":
        operations = [
            "training_random_all_enabled_seed_1",
            "training_random_all_enabled_seed_2",
            "training_random_all_enabled_seed_3",
        ]
    else:
        operations = list(DEFAULT_AUGMENTATION_OPERATIONS)
    unknown = [name for name in operations if name not in AUGMENTATION_REGISTRY]
    if unknown:
        raise ValueError(f"Unsupported debug augmentation operation(s): {unknown}")
    return operations


def augmentation_catalog(operations: list[str] | None = None) -> list[dict[str, Any]]:
    names = operations or list(DEFAULT_AUGMENTATION_OPERATIONS)
    return [augmentation_spec_to_dict(AUGMENTATION_REGISTRY[name]) for name in names]


def augmentation_spec_to_dict(spec: DebugAugmentationSpec) -> dict[str, Any]:
    return {
        "operation": spec.name,
        "group": spec.group,
        "training_key": spec.training_key,
        "description_ru": spec.description_ru,
        "parameters": dict(spec.parameters),
        "matches_training_semantics": spec.matches_training_semantics,
        "debug_note": spec.debug_note,
    }


def apply_debug_augmentation(rgb: np.ndarray, operation: str, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    augmented, _mask, metadata = apply_debug_augmentation_with_mask(rgb, operation, seed=seed, mask=None)
    return augmented, metadata


def apply_debug_augmentation_with_mask(
    rgb: np.ndarray,
    operation: str,
    *,
    seed: int,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if operation not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unsupported debug augmentation operation: {operation}")
    spec = AUGMENTATION_REGISTRY[operation]
    before = _ensure_rgb_uint8(rgb)
    before_mask = None if mask is None else np.asarray(mask).copy()
    augmented, after_mask, actual_parameters = _apply_operation(before, before_mask, spec, seed)
    augmented = _ensure_rgb_uint8(augmented)
    metadata = _metadata(before, augmented, spec, seed, actual_parameters)
    return augmented, after_mask, metadata


def _apply_operation(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    spec: DebugAugmentationSpec,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    name = spec.name
    rng = random.Random(seed + int(spec.parameters.get("random_seed_alias", 0)))
    arr = rgb.copy()
    out_mask = None if mask is None else mask.copy()

    if name == "original":
        return arr, out_mask, {}
    if name == "flip_horizontal":
        return arr[:, ::-1, :], None if out_mask is None else out_mask[:, ::-1], {}
    if name == "flip_vertical":
        return arr[::-1, :, :], None if out_mask is None else out_mask[::-1, :], {}
    if name == "flip_horizontal_vertical":
        return arr[::-1, ::-1, :], None if out_mask is None else out_mask[::-1, ::-1], {}
    if name.startswith("rot90_"):
        k = int(spec.parameters["k"])
        return np.rot90(arr, k=k).copy(), None if out_mask is None else np.rot90(out_mask, k=k).copy(), {"k": k}
    if name == "brightness":
        factor = float(spec.parameters["factor"])
        return _clip_uint8(arr.astype("float32") * factor), out_mask, {"factor": factor}
    if name == "contrast":
        factor = float(spec.parameters["factor"])
        mean = arr.astype("float32").mean(axis=(0, 1), keepdims=True)
        return _clip_uint8((arr.astype("float32") - mean) * factor + mean), out_mask, {"factor": factor}
    if name in {"brightness_contrast", "color_jitter"}:
        brightness = float(spec.parameters["brightness"])
        contrast = float(spec.parameters["contrast"])
        data = arr.astype("float32")
        mean = data.mean(axis=(0, 1), keepdims=True)
        return _clip_uint8(((data - mean) * contrast + mean) * brightness), out_mask, {"brightness": brightness, "contrast": contrast}
    if name.startswith("gamma_"):
        gamma = float(spec.parameters["gamma"])
        data = np.clip(arr.astype("float32") / 255.0, 0.0, 1.0)
        return _clip_uint8(np.power(data, gamma) * 255.0), out_mask, {"gamma": gamma}
    if name.startswith("noise_"):
        std = float(spec.parameters["std"])
        noise = np.random.default_rng(seed).normal(0.0, std * 255.0, size=arr.shape).astype("float32")
        return _clip_uint8(arr.astype("float32") + noise), out_mask, {"std": std}
    if name.startswith("blur_"):
        radius = int(spec.parameters["radius"])
        image = Image.fromarray(arr, mode="RGB").filter(ImageFilter.BoxBlur(radius=radius))
        return np.asarray(image), out_mask, {"radius": radius}
    if name.startswith("cutout_"):
        fraction = float(spec.parameters["fraction"])
        return _apply_cutout(arr, rng, fraction), out_mask, {"fraction": fraction}
    if name == "coarse_dropout":
        return _apply_coarse_dropout(arr, rng, int(spec.parameters["holes"]), float(spec.parameters["fraction"])), out_mask, dict(spec.parameters)
    if name.startswith("training_random_all_enabled_seed_"):
        return _apply_training_random_all(arr, rng, seed), out_mask, {"source_seed": seed, **spec.parameters}
    raise ValueError(f"Unsupported debug augmentation operation: {name}")


def _apply_training_random_all(arr: np.ndarray, rng: random.Random, seed: int) -> np.ndarray:
    result = arr.copy()
    if rng.random() < 0.5:
        result = result[:, ::-1, :]
    if rng.random() < 0.5:
        result = result[::-1, :, :]
    result = np.rot90(result, k=rng.randint(0, 3)).copy()
    brightness = 0.91 + rng.random() * 0.18
    contrast = 0.85 + rng.random() * 0.30
    mean = result.astype("float32").mean(axis=(0, 1), keepdims=True)
    result = _clip_uint8(((result.astype("float32") - mean) * contrast + mean) * brightness)
    gamma = 0.80 + rng.random() * 0.45
    result = _clip_uint8(np.power(np.clip(result.astype("float32") / 255.0, 0.0, 1.0), gamma) * 255.0)
    noise = np.random.default_rng(seed).normal(0.0, 0.02 * 255.0, size=result.shape).astype("float32")
    result = _clip_uint8(result.astype("float32") + noise)
    if rng.random() < 0.25:
        result = np.asarray(Image.fromarray(result, mode="RGB").filter(ImageFilter.BoxBlur(radius=1)))
    if rng.random() < 0.35:
        result = _apply_cutout(result, rng, 0.125)
    return result


def _apply_cutout(arr: np.ndarray, rng: random.Random, fraction: float) -> np.ndarray:
    result = arr.copy()
    h, w = result.shape[:2]
    cut_h = max(1, int(h * fraction))
    cut_w = max(1, int(w * fraction))
    x0 = rng.randint(0, max(0, w - cut_w))
    y0 = rng.randint(0, max(0, h - cut_h))
    result[y0 : y0 + cut_h, x0 : x0 + cut_w, :] = 0
    return result


def _apply_coarse_dropout(arr: np.ndarray, rng: random.Random, holes: int, fraction: float) -> np.ndarray:
    result = arr.copy()
    h, w = result.shape[:2]
    cut_h = max(1, int(h * fraction))
    cut_w = max(1, int(w * fraction))
    for _ in range(max(1, holes)):
        x0 = rng.randint(0, max(0, w - cut_w))
        y0 = rng.randint(0, max(0, h - cut_h))
        result[y0 : y0 + cut_h, x0 : x0 + cut_w, :] = 0
    return result


def _metadata(
    before: np.ndarray,
    after: np.ndarray,
    spec: DebugAugmentationSpec,
    seed: int,
    actual_parameters: dict[str, Any],
) -> dict[str, Any]:
    changed = np.any(before != after, axis=2)
    changed_fraction = float(np.count_nonzero(changed)) / float(changed.size) if changed.size else 0.0
    checks = {
        "shape_preserved": tuple(before.shape) == tuple(after.shape),
        "dtype_uint8": after.dtype == np.uint8,
        "range_0_255": bool(after.size == 0 or (int(after.min()) >= 0 and int(after.max()) <= 255)),
        "non_empty_output": bool(after.size > 0),
        "changed_when_expected": (changed_fraction > 0.0001) if spec.expects_image_change else (changed_fraction == 0.0),
    }
    failed = [key for key, value in checks.items() if not value and key not in {"changed_when_expected"}]
    warnings = [key for key, value in checks.items() if not value and key == "changed_when_expected"]
    status = "failed" if failed else ("warning" if warnings else "pass")
    return {
        **augmentation_spec_to_dict(spec),
        "operation": spec.name,
        "seed": seed,
        "parameters": {**dict(spec.parameters), **actual_parameters},
        "shape_before": list(before.shape),
        "shape_after": list(after.shape),
        "dtype_before": str(before.dtype),
        "dtype_after": str(after.dtype),
        "min_after": int(after.min()) if after.size else None,
        "max_after": int(after.max()) if after.size else None,
        "mean_before": round(float(before.mean()), 6) if before.size else None,
        "mean_after": round(float(after.mean()), 6) if after.size else None,
        "std_before": round(float(before.std()), 6) if before.size else None,
        "std_after": round(float(after.std()), 6) if after.size else None,
        "changed_pixels_fraction": round(changed_fraction, 6),
        "check_status": status,
        "checks": checks,
    }


def _ensure_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3 array, got shape={arr.shape}")
    if arr.dtype == np.uint8:
        return arr.copy()
    return _clip_uint8(arr.astype("float32"))


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype("uint8")
