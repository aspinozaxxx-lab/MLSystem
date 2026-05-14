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
    "flip_horizontal_vertical": _spec("flip_horizontal_vertical", "geometric", "flips", "Горизонтальное и вертикальное отражение.", mask_transform="flip_horizontal_vertical"),
    "rot90_90": _spec("rot90_90", "geometric", "rot90", "Поворот на 90 градусов.", {"k": 1}, mask_transform="rot90_90"),
    "rot90_180": _spec("rot90_180", "geometric", "rot90", "Поворот на 180 градусов.", {"k": 2}, mask_transform="rot90_180"),
    "rot90_270": _spec("rot90_270", "geometric", "rot90", "Поворот на 270 градусов.", {"k": 3}, mask_transform="rot90_270"),
    "brightness": _spec("brightness", "photometric", "brightness_contrast", "Изменение яркости.", {"factor": 1.18}),
    "contrast": _spec("contrast", "photometric", "brightness_contrast", "Изменение контраста.", {"factor": 1.25}),
    "brightness_contrast": _spec("brightness_contrast", "photometric", "brightness_contrast", "Изменение яркости и контраста.", {"brightness": 1.15, "contrast": 1.25}),
    "gamma_low": _spec("gamma_low", "photometric", "gamma", "Gamma ниже 1.0.", {"gamma": 0.8}),
    "gamma_high": _spec("gamma_high", "photometric", "gamma", "Gamma выше 1.0.", {"gamma": 1.25}),
    "noise_low": _spec("noise_low", "noise", "noise", "Слабый гауссов шум.", {"std": 0.015}),
    "noise_high": _spec("noise_high", "noise", "noise", "Сильный гауссов шум.", {"std": 0.045}),
    "blur_light": _spec("blur_light", "blur", "blur", "Легкое размытие.", {"radius": 1}),
    "blur_strong": _spec("blur_strong", "blur", "blur", "Сильное размытие.", {"radius": 2}),
    "cutout_small": _spec("cutout_small", "occlusion/dropout", "cutout", "Малый cutout только по изображению.", {"fraction": 0.12}),
    "cutout_medium": _spec("cutout_medium", "occlusion/dropout", "cutout", "Средний cutout только по изображению.", {"fraction": 0.22}),
    "coarse_dropout": _spec("coarse_dropout", "occlusion/dropout", "coarse_dropout", "Coarse dropout только по изображению.", {"holes": 4, "fraction": 0.12}),
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
GEOMETRIC_OPERATIONS = {"flip_horizontal", "flip_vertical", "flip_horizontal_vertical", "rot90_90", "rot90_180", "rot90_270"}


def resolve_augmentation_operations(mode: str = "all", requested: str | list[str] | None = None) -> list[str]:
    if requested:
        values = requested.split(",") if isinstance(requested, str) else requested
        operations = [str(item).strip() for item in values if str(item).strip()]
    elif mode == "individual":
        operations = [name for name in DEFAULT_AUGMENTATION_OPERATIONS if not name.startswith("training_random_all_enabled")]
    elif mode == "production-groups":
        operations = ["flip_horizontal", "flip_vertical", "rot90_90", "brightness_contrast", "color_jitter", "gamma_low", "noise_low", "blur_light", "cutout_medium", "coarse_dropout"]
    elif mode == "random-training":
        operations = ["training_random_all_enabled_seed_1", "training_random_all_enabled_seed_2", "training_random_all_enabled_seed_3"]
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


def apply_debug_augmentation(
    rgb: np.ndarray,
    operation: str,
    *,
    seed: int,
    cutout_mask_mode: str = "erase",
) -> tuple[np.ndarray, dict[str, Any]]:
    augmented, _mask, metadata = apply_debug_augmentation_with_mask(
        rgb,
        operation,
        seed=seed,
        mask=None,
        cutout_mask_mode=cutout_mask_mode,
    )
    return augmented, metadata


def apply_debug_augmentation_with_mask(
    rgb: np.ndarray,
    operation: str,
    *,
    seed: int,
    mask: np.ndarray | None = None,
    cutout_mask_mode: str = "erase",
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if operation not in AUGMENTATION_REGISTRY:
        raise ValueError(f"Unsupported debug augmentation operation: {operation}")
    spec = AUGMENTATION_REGISTRY[operation]
    before = _ensure_rgb_uint8(rgb)
    before_mask = None if mask is None else _ensure_mask_uint8(mask)
    augmented, after_mask, actual_parameters = _apply_operation(before, before_mask, spec, seed, cutout_mask_mode=cutout_mask_mode)
    augmented = _ensure_rgb_uint8(augmented)
    if after_mask is not None:
        after_mask = _ensure_mask_uint8(after_mask)
    metadata = _metadata(before, augmented, before_mask, after_mask, spec, seed, actual_parameters)
    return augmented, after_mask, metadata


def apply_random_training_augmentation(
    rgb: np.ndarray,
    mask: np.ndarray,
    augmentations: dict[str, Any],
    *,
    seed: int,
    cutout_mask_mode: str = "erase",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    enabled = bool(augmentations) and any(bool(value) for value in augmentations.values())
    if not enabled:
        augmented, out_mask, metadata = apply_debug_augmentation_with_mask(rgb, "original", seed=seed, mask=mask, cutout_mask_mode=cutout_mask_mode)
        return augmented, out_mask if out_mask is not None else mask, metadata
    operation = f"training_random_all_enabled_seed_{(seed % 3) + 1}"
    augmented, out_mask, metadata = apply_debug_augmentation_with_mask(rgb, operation, seed=seed, mask=mask, cutout_mask_mode=cutout_mask_mode)
    return augmented, out_mask if out_mask is not None else mask, metadata


def apply_training_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    augmentations: dict[str, Any],
    *,
    seed: int,
    cutout_mask_mode: str = "erase",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not isinstance(augmentations, dict) or not any(bool(value) for value in augmentations.values()):
        return image.copy(), _ensure_mask_uint8(mask), {"operation": "none", "seed": seed}
    arr = np.asarray(image).astype("float32", copy=True)
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW or HWC image, got shape={arr.shape}")
    chw = arr.shape[0] <= 16
    if not chw:
        arr = np.transpose(arr, (2, 0, 1))
    out_mask = _ensure_mask_uint8(mask)
    mask_positive_before_cutout = int(np.count_nonzero(out_mask))
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    cutout_mask_mode = _normalize_cutout_mask_mode(cutout_mask_mode)
    metadata: dict[str, Any] = {
        "operation": "production_random",
        "seed": seed,
        "applied": [],
        "cutout_mask_mode": cutout_mask_mode,
        "mask_positive_pixels_before": mask_positive_before_cutout,
    }

    if augmentations.get("flips"):
        h_flip = rng.random() < 0.5
        v_flip = rng.random() < 0.5
        if h_flip:
            arr = arr[:, :, ::-1]
            out_mask = out_mask[:, ::-1]
            metadata["applied"].append("flip_horizontal")
        if v_flip:
            arr = arr[:, ::-1, :]
            out_mask = out_mask[::-1, :]
            metadata["applied"].append("flip_vertical")

    if augmentations.get("rot90"):
        k = rng.randint(0, 3)
        if k:
            arr = np.rot90(arr, k=k, axes=(1, 2)).copy()
            out_mask = np.rot90(out_mask, k=k).copy()
            metadata["applied"].append(f"rot90_{k * 90}")
        metadata["rot90_k"] = k

    if augmentations.get("brightness_contrast") or augmentations.get("color_jitter"):
        brightness = 1.0 + (rng.random() - 0.5) * 0.18
        contrast = 1.0 + (rng.random() - 0.5) * 0.30
        mean = arr.mean(axis=(1, 2), keepdims=True)
        arr = (arr - mean) * contrast + mean
        arr = arr * brightness
        metadata["applied"].append("brightness_contrast")
        metadata["brightness"] = round(brightness, 6)
        metadata["contrast"] = round(contrast, 6)

    if augmentations.get("gamma"):
        gamma = 0.80 + rng.random() * 0.45
        arr = np.clip(arr, 0.0, 1.0) ** gamma
        metadata["applied"].append("gamma")
        metadata["gamma"] = round(gamma, 6)

    if augmentations.get("noise"):
        arr = arr + np_rng.normal(0.0, 0.02, size=arr.shape).astype("float32")
        metadata["applied"].append("noise")
        metadata["noise_std"] = 0.02

    if augmentations.get("blur"):
        blur_applied = rng.random() < 0.25
        if blur_applied:
            arr = _avg_blur_chw(arr)
            metadata["applied"].append("blur")
        metadata["blur_applied"] = blur_applied

    if augmentations.get("cutout") or augmentations.get("coarse_dropout"):
        cutout_applied = rng.random() < 0.35
        if cutout_applied:
            height = int(arr.shape[1])
            width = int(arr.shape[2])
            hole_count = 4 if augmentations.get("coarse_dropout") and not augmentations.get("cutout") else 1
            boxes: list[dict[str, int]] = []
            for _ in range(hole_count):
                cut_h = max(8, height // 8)
                cut_w = max(8, width // 8)
                y0 = rng.randint(0, max(0, height - cut_h))
                x0 = rng.randint(0, max(0, width - cut_w))
                arr[:, y0 : y0 + cut_h, x0 : x0 + cut_w] = 0.0
                boxes.append({"x": int(x0), "y": int(y0), "width": int(cut_w), "height": int(cut_h)})
            out_mask, mask_meta = _apply_cutout_to_mask(out_mask, boxes, cutout_mask_mode)
            metadata["applied"].append("coarse_dropout" if hole_count > 1 else "cutout")
            metadata["cutout"] = {"boxes": boxes, **mask_meta}
        metadata["cutout_applied"] = cutout_applied

    arr = np.clip(arr, 0.0, 1.0).astype("float32")
    if not chw:
        arr = np.transpose(arr, (1, 2, 0)).astype("float32")
    metadata["mask_positive_pixels_after"] = int(np.count_nonzero(out_mask))
    metadata.setdefault("cutout", {"boxes": [], "cutout_mask_mode": cutout_mask_mode, "mask_erased_pixels": 0, "cutout_intersected_positive": False})
    metadata["mask_behavior"] = (
        "geometric transforms image+mask; photometric/noise/blur keep mask unchanged; "
        f"cutout/coarse_dropout mask mode={cutout_mask_mode}"
    )
    return arr, out_mask, metadata


def _avg_blur_chw(arr: np.ndarray) -> np.ndarray:
    padded = np.pad(arr, ((0, 0), (1, 1), (1, 1)), mode="edge")
    acc = np.zeros_like(arr, dtype="float32")
    for dy in range(3):
        for dx in range(3):
            acc += padded[:, dy : dy + arr.shape[1], dx : dx + arr.shape[2]]
    return acc / 9.0


def _apply_operation(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    spec: DebugAugmentationSpec,
    seed: int,
    *,
    cutout_mask_mode: str = "erase",
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
        augmented, boxes = _apply_cutout(arr, rng, fraction)
        out_mask, mask_meta = _apply_cutout_to_mask(out_mask, boxes, cutout_mask_mode)
        return augmented, out_mask, {"fraction": fraction, "cutout_boxes": boxes, **mask_meta}
    if name == "coarse_dropout":
        augmented, boxes = _apply_coarse_dropout(arr, rng, int(spec.parameters["holes"]), float(spec.parameters["fraction"]))
        out_mask, mask_meta = _apply_cutout_to_mask(out_mask, boxes, cutout_mask_mode)
        return augmented, out_mask, {**dict(spec.parameters), "cutout_boxes": boxes, **mask_meta}
    if name.startswith("training_random_all_enabled_seed_"):
        out_arr, out_mask, params = _apply_training_random_all(arr, out_mask, rng, seed, cutout_mask_mode=cutout_mask_mode)
        return out_arr, out_mask, {"source_seed": seed, **spec.parameters, **params}
    raise ValueError(f"Unsupported debug augmentation operation: {name}")


def _apply_training_random_all(
    arr: np.ndarray,
    mask: np.ndarray | None,
    rng: random.Random,
    seed: int,
    *,
    cutout_mask_mode: str = "erase",
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    result = arr.copy()
    out_mask = None if mask is None else mask.copy()
    h_flip = rng.random() < 0.5
    v_flip = rng.random() < 0.5
    if h_flip:
        result = result[:, ::-1, :]
        if out_mask is not None:
            out_mask = out_mask[:, ::-1]
    if v_flip:
        result = result[::-1, :, :]
        if out_mask is not None:
            out_mask = out_mask[::-1, :]
    rot_k = rng.randint(0, 3)
    result = np.rot90(result, k=rot_k).copy()
    if out_mask is not None:
        out_mask = np.rot90(out_mask, k=rot_k).copy()
    brightness = 0.91 + rng.random() * 0.18
    contrast = 0.85 + rng.random() * 0.30
    mean = result.astype("float32").mean(axis=(0, 1), keepdims=True)
    result = _clip_uint8(((result.astype("float32") - mean) * contrast + mean) * brightness)
    gamma = 0.80 + rng.random() * 0.45
    result = _clip_uint8(np.power(np.clip(result.astype("float32") / 255.0, 0.0, 1.0), gamma) * 255.0)
    noise = np.random.default_rng(seed).normal(0.0, 0.02 * 255.0, size=result.shape).astype("float32")
    result = _clip_uint8(result.astype("float32") + noise)
    blur_applied = rng.random() < 0.25
    if blur_applied:
        result = np.asarray(Image.fromarray(result, mode="RGB").filter(ImageFilter.BoxBlur(radius=1)))
    cutout_applied = rng.random() < 0.35
    cutout_boxes: list[dict[str, int]] = []
    mask_meta = _cutout_mask_metadata(mask, mask, [], cutout_mask_mode)
    if cutout_applied:
        result, cutout_boxes = _apply_cutout(result, rng, 0.125)
        out_mask, mask_meta = _apply_cutout_to_mask(out_mask, cutout_boxes, cutout_mask_mode)
    return result, out_mask, {
        "h_flip": h_flip,
        "v_flip": v_flip,
        "rot90_k": rot_k,
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
        "gamma": round(gamma, 6),
        "noise_std": 0.02,
        "blur_applied": blur_applied,
        "cutout_applied": cutout_applied,
        "cutout_boxes": cutout_boxes,
        **mask_meta,
    }


def _apply_cutout(arr: np.ndarray, rng: random.Random, fraction: float) -> tuple[np.ndarray, list[dict[str, int]]]:
    result = arr.copy()
    h, w = result.shape[:2]
    cut_h = max(1, int(h * fraction))
    cut_w = max(1, int(w * fraction))
    x0 = rng.randint(0, max(0, w - cut_w))
    y0 = rng.randint(0, max(0, h - cut_h))
    result[y0 : y0 + cut_h, x0 : x0 + cut_w, :] = 0
    box = {"x": int(x0), "y": int(y0), "width": int(cut_w), "height": int(cut_h)}
    return result, [box]


def _apply_coarse_dropout(arr: np.ndarray, rng: random.Random, holes: int, fraction: float) -> tuple[np.ndarray, list[dict[str, int]]]:
    result = arr.copy()
    h, w = result.shape[:2]
    cut_h = max(1, int(h * fraction))
    cut_w = max(1, int(w * fraction))
    boxes: list[dict[str, int]] = []
    for _ in range(max(1, holes)):
        x0 = rng.randint(0, max(0, w - cut_w))
        y0 = rng.randint(0, max(0, h - cut_h))
        result[y0 : y0 + cut_h, x0 : x0 + cut_w, :] = 0
        boxes.append({"x": int(x0), "y": int(y0), "width": int(cut_w), "height": int(cut_h)})
    return result, boxes


def _normalize_cutout_mask_mode(mode: str) -> str:
    value = str(mode or "erase").lower()
    if value not in {"erase", "preserve", "ignore"}:
        raise ValueError("cutout_mask_mode must be one of: erase, preserve, ignore")
    return value


def _apply_cutout_to_mask(
    mask: np.ndarray | None,
    boxes: list[dict[str, int]],
    cutout_mask_mode: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    mode = _normalize_cutout_mask_mode(cutout_mask_mode)
    if mask is None:
        return None, _cutout_mask_metadata(None, None, boxes, mode)
    before = _ensure_mask_uint8(mask)
    after = before.copy()
    if mode == "erase":
        for box in boxes:
            x0 = max(0, int(box.get("x", 0)))
            y0 = max(0, int(box.get("y", 0)))
            x1 = min(after.shape[1], x0 + max(0, int(box.get("width", 0))))
            y1 = min(after.shape[0], y0 + max(0, int(box.get("height", 0))))
            if x1 > x0 and y1 > y0:
                after[y0:y1, x0:x1] = 0
    return after, _cutout_mask_metadata(before, after, boxes, mode)


def _cutout_mask_metadata(
    before: np.ndarray | None,
    after: np.ndarray | None,
    boxes: list[dict[str, int]],
    cutout_mask_mode: str,
) -> dict[str, Any]:
    before_count = int(np.count_nonzero(before)) if before is not None else None
    after_count = int(np.count_nonzero(after)) if after is not None else None
    erased = max(0, int(before_count - after_count)) if before_count is not None and after_count is not None else 0
    return {
        "cutout_mask_mode": _normalize_cutout_mask_mode(cutout_mask_mode),
        "mask_behavior": _normalize_cutout_mask_mode(cutout_mask_mode),
        "mask_positive_pixels_before": before_count,
        "mask_positive_pixels_after": after_count,
        "mask_erased_pixels": erased,
        "cutout_intersected_positive": bool(erased > 0),
        "cutout_boxes": list(boxes),
    }


def _metadata(
    before: np.ndarray,
    after: np.ndarray,
    before_mask: np.ndarray | None,
    after_mask: np.ndarray | None,
    spec: DebugAugmentationSpec,
    seed: int,
    actual_parameters: dict[str, Any],
) -> dict[str, Any]:
    changed = np.any(before != after, axis=2)
    changed_fraction = float(np.count_nonzero(changed)) / float(changed.size) if changed.size else 0.0
    image_checks = {
        "shape_preserved": tuple(before.shape) == tuple(after.shape),
        "dtype_uint8": after.dtype == np.uint8,
        "range_0_255": bool(after.size == 0 or (int(after.min()) >= 0 and int(after.max()) <= 255)),
        "non_empty_output": bool(after.size > 0),
        "changed_when_expected": (changed_fraction > 0.0001) if spec.expects_image_change else (changed_fraction == 0.0),
    }
    mask_checks = _mask_checks(before_mask, after_mask, spec, actual_parameters)
    mask_erasure_expected = _is_cutout_erasure_expected(spec, actual_parameters)
    failed = [key for key, value in image_checks.items() if not value and key not in {"changed_when_expected"}]
    failed.extend(
        key
        for key, value in mask_checks.items()
        if not value
        and key not in {"mask_alignment_check", "mask_unchanged"}
        and not (mask_erasure_expected and key == "mask_positive_pixels_preserved")
    )
    warnings = [key for key, value in image_checks.items() if not value and key == "changed_when_expected"]
    warnings.extend(key for key, value in mask_checks.items() if not value and key == "mask_alignment_check" and not mask_erasure_expected)
    status = "failed" if failed else ("warning" if warnings else "pass")
    mask_before_positive = int(np.count_nonzero(before_mask)) if before_mask is not None else None
    mask_after_positive = int(np.count_nonzero(after_mask)) if after_mask is not None else None
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
        "mask_positive_pixels_before": mask_before_positive,
        "mask_positive_pixels_after": mask_after_positive,
        "mask_erased_pixels": int(actual_parameters.get("mask_erased_pixels") or 0),
        "cutout_intersected_positive": bool(actual_parameters.get("cutout_intersected_positive", False)),
        "cutout_mask_mode": actual_parameters.get("cutout_mask_mode"),
        "cutout_boxes": actual_parameters.get("cutout_boxes", []),
        "mask_alignment_check": mask_checks.get("mask_alignment_check"),
        "check_status": status,
        "checks": image_checks,
        "mask_checks": mask_checks,
    }


def _mask_checks(
    before_mask: np.ndarray | None,
    after_mask: np.ndarray | None,
    spec: DebugAugmentationSpec,
    actual_parameters: dict[str, Any] | None = None,
) -> dict[str, bool]:
    if before_mask is None and after_mask is None:
        return {}
    if before_mask is None or after_mask is None:
        return {"mask_present": False, "mask_alignment_check": False}
    before_binary = set(np.unique(before_mask).tolist()).issubset({0, 1})
    after_binary = set(np.unique(after_mask).tolist()).issubset({0, 1})
    before_positive = int(np.count_nonzero(before_mask))
    after_positive = int(np.count_nonzero(after_mask))
    positive_preserved = before_positive == after_positive
    erasure_expected = _is_cutout_erasure_expected(spec, actual_parameters or {})
    if spec.name in GEOMETRIC_OPERATIONS:
        expected = _expected_mask_transform(before_mask, spec.name)
        alignment = bool(np.array_equal(expected, after_mask))
        unchanged = bool(np.array_equal(before_mask, after_mask))
    elif spec.name.startswith("training_random_all_enabled_seed_"):
        alignment = bool(after_positive <= before_positive) if erasure_expected else positive_preserved
        unchanged = bool(np.array_equal(before_mask, after_mask))
    elif erasure_expected:
        alignment = bool(after_positive <= before_positive)
        unchanged = bool(np.array_equal(before_mask, after_mask))
    else:
        alignment = bool(np.array_equal(before_mask, after_mask))
        unchanged = alignment
    return {
        "mask_shape_preserved": tuple(before_mask.shape) == tuple(after_mask.shape),
        "mask_binary": bool(before_binary and after_binary),
        "mask_positive_pixels_preserved": bool(positive_preserved),
        "mask_unchanged": bool(unchanged),
        "mask_alignment_check": bool(alignment),
    }


def _is_cutout_erasure_expected(spec: DebugAugmentationSpec, actual_parameters: dict[str, Any] | None) -> bool:
    params = actual_parameters or {}
    mode = str(params.get("cutout_mask_mode") or params.get("mask_behavior") or "").lower()
    has_cutout = bool(params.get("cutout_boxes")) or bool(params.get("cutout_applied"))
    is_cutout_operation = spec.name.startswith("cutout_") or spec.name == "coarse_dropout" or spec.name.startswith("training_random_all_enabled_seed_")
    return is_cutout_operation and has_cutout and mode == "erase"


def _expected_mask_transform(mask: np.ndarray, operation: str) -> np.ndarray:
    if operation == "flip_horizontal":
        return mask[:, ::-1]
    if operation == "flip_vertical":
        return mask[::-1, :]
    if operation == "flip_horizontal_vertical":
        return mask[::-1, ::-1]
    if operation == "rot90_90":
        return np.rot90(mask, k=1).copy()
    if operation == "rot90_180":
        return np.rot90(mask, k=2).copy()
    if operation == "rot90_270":
        return np.rot90(mask, k=3).copy()
    return mask.copy()


def _ensure_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3 array, got shape={arr.shape}")
    if arr.dtype == np.uint8:
        return arr.copy()
    return _clip_uint8(arr.astype("float32"))


def _ensure_mask_uint8(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Expected mask HxW array, got shape={arr.shape}")
    return (arr > 0).astype("uint8")


def _clip_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype("uint8")
