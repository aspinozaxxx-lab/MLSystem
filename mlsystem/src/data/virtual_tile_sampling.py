from __future__ import annotations

import math
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

import numpy as np
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box

from ..tile_preparation.windows import generate_window_grid_for_scene as _tile_preparation_window_grid


TileKind = Literal["positive", "partial_positive", "hard_negative", "negative"]


@dataclass
class TrainSamplingConfig:
    enabled: bool = False
    virtual_epoch_multiplier: int = 1
    positive_repeat_factor: int = 1
    hard_negative_repeat_factor: int = 1
    negative_repeat_factor: int = 1
    positive_stride_factor: float = 1.0
    hard_negative_stride_factor: float = 1.0
    negative_stride_factor: float = 1.0
    min_positive_pixels: int = 1
    include_partial_positive: bool = True
    partial_positive_fraction: float = 0.0
    max_empty_tile_share: float | None = None
    hard_negative_context_px: int | None = None
    batch_positive_fraction: float | None = None
    batch_hard_negative_fraction: float | None = None
    batch_negative_fraction: float | None = None
    random_jitter: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "max_shift_fraction": 0.25,
        "keep_positive_min_pixels": 1,
    })
    warnings: list[str] = field(default_factory=list)

    def effective_positive_stride(self, base_stride: int) -> int:
        return _effective_stride(base_stride, self.positive_stride_factor)

    def effective_hard_negative_stride(self, base_stride: int) -> int:
        return _effective_stride(base_stride, self.hard_negative_stride_factor)

    def effective_negative_stride(self, base_stride: int) -> int:
        return _effective_stride(base_stride, self.negative_stride_factor)


@dataclass
class TileSampleRecord:
    scene_id: str
    image_path: str | None
    x: int
    y: int
    width: int
    height: int
    tile_size: int
    stride: int
    kind: TileKind
    positive_pixels: int
    source: str
    base_record_id: str | None = None
    repeat_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        existing = self.metadata.get("record_id")
        if existing:
            return str(existing)
        return f"{self.scene_id}:x{self.x}:y{self.y}:w{self.width}:h{self.height}:{self.kind}"

    def to_dict(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.setdefault("record_id", self.record_id)
        return {
            "sample_id": metadata.get("sample_id") or self.record_id,
            "scene_id": self.scene_id,
            "scene": self.scene_id,
            "tile_id": f"x{self.x}_y{self.y}_w{self.width}_h{self.height}",
            "source_image_path": self.image_path,
            "window": {"x": int(self.x), "y": int(self.y), "width": int(self.width), "height": int(self.height)},
            "tile_size": int(self.tile_size),
            "stride": int(self.stride),
            "kind": self.kind,
            "positive_tile": self.kind in {"positive", "partial_positive"},
            "positive_pixels": int(self.positive_pixels),
            "gt_positive_pixels": int(self.positive_pixels),
            "source": self.source,
            "base_record_id": self.base_record_id,
            "repeat_index": self.repeat_index,
            "metadata": metadata,
        }


@dataclass(frozen=True)
class WindowCandidate:
    scene_id: str
    x: int
    y: int
    width: int
    height: int
    tile_size: int
    stride: int
    index: int


@dataclass
class TileRecordBuildResult:
    records: list[TileSampleRecord]
    scene_reports: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def resolve_train_sampling_config(
    preprocess: Mapping[str, Any] | None,
    train: Mapping[str, Any] | None = None,
) -> TrainSamplingConfig:
    preprocess = preprocess or {}
    raw = dict(preprocess.get("train_sampling") or {})
    random_jitter = {
        "enabled": False,
        "max_shift_fraction": 0.25,
        "keep_positive_min_pixels": 1,
    }
    random_jitter.update(raw.get("random_jitter") or {})

    max_empty_tile_share = raw.get("max_empty_tile_share")
    if max_empty_tile_share is None and "max_empty_tile_share" in preprocess:
        max_empty_tile_share = preprocess.get("max_empty_tile_share")

    cfg = TrainSamplingConfig(
        enabled=bool(raw.get("enabled", False)),
        virtual_epoch_multiplier=max(1, int(raw.get("virtual_epoch_multiplier", 1) or 1)),
        positive_repeat_factor=max(1, int(raw.get("positive_repeat_factor", 1) or 1)),
        hard_negative_repeat_factor=max(1, int(raw.get("hard_negative_repeat_factor", 1) or 1)),
        negative_repeat_factor=max(1, int(raw.get("negative_repeat_factor", 1) or 1)),
        positive_stride_factor=max(0.000001, float(raw.get("positive_stride_factor", 1.0) or 1.0)),
        hard_negative_stride_factor=max(0.000001, float(raw.get("hard_negative_stride_factor", 1.0) or 1.0)),
        negative_stride_factor=max(0.000001, float(raw.get("negative_stride_factor", 1.0) or 1.0)),
        min_positive_pixels=max(1, int(raw.get("min_positive_pixels", 1) or 1)),
        include_partial_positive=bool(raw.get("include_partial_positive", True)),
        partial_positive_fraction=max(0.0, float(raw.get("partial_positive_fraction", 0.0) or 0.0)),
        max_empty_tile_share=None if max_empty_tile_share is None else max(0.0, min(1.0, float(max_empty_tile_share))),
        hard_negative_context_px=None
        if raw.get("hard_negative_context_px") is None
        else max(0, int(raw.get("hard_negative_context_px") or 0)),
        batch_positive_fraction=_optional_fraction(raw.get("batch_positive_fraction")),
        batch_hard_negative_fraction=_optional_fraction(raw.get("batch_hard_negative_fraction")),
        batch_negative_fraction=_optional_fraction(raw.get("batch_negative_fraction")),
        random_jitter=random_jitter,
    )
    return cfg


def generate_window_grid_for_scene(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    *,
    scene_id: str = "",
) -> list[WindowCandidate]:
    return [
        WindowCandidate(
            scene_id=window.scene_id,
            x=window.x,
            y=window.y,
            width=window.width,
            height=window.height,
            tile_size=window.tile_size,
            stride=window.stride,
            index=index,
        )
        for index, window in enumerate(
            _tile_preparation_window_grid(
                int(width),
                int(height),
                int(tile_size),
                int(stride),
                scene_id=scene_id,
            )
        )
    ]


def classify_tile_by_mask(
    mask: np.ndarray,
    *,
    min_positive_pixels: int = 1,
    include_partial_positive: bool = True,
    partial_positive_fraction: float = 0.0,
) -> tuple[TileKind, int]:
    positive_pixels = int(np.count_nonzero(mask))
    if positive_pixels >= max(1, int(min_positive_pixels)):
        return "positive", positive_pixels
    if positive_pixels > 0 and include_partial_positive:
        if partial_positive_fraction <= 0:
            return "partial_positive", positive_pixels
        area = max(1, int(mask.shape[-1]) * int(mask.shape[-2]))
        if positive_pixels / area >= float(partial_positive_fraction):
            return "partial_positive", positive_pixels
    return "negative", positive_pixels


def classify_hard_negative(
    window_bbox: tuple[int, int, int, int],
    positive_bboxes: list[tuple[int, int, int, int]],
    hard_negative_context_px: int | None,
) -> bool:
    if hard_negative_context_px is None or hard_negative_context_px <= 0 or not positive_bboxes:
        return False
    x0, y0, x1, y1 = window_bbox
    current = box(x0, y0, x1, y1)
    context = int(hard_negative_context_px)
    for px0, py0, px1, py1 in positive_bboxes:
        expanded = box(px0 - context, py0 - context, px1 + context, py1 + context)
        if current.intersects(expanded):
            return True
    return False


def build_virtual_train_records(
    scenes: list[Any],
    shapes: list[Any],
    *,
    tile_size: int,
    stride: int,
    train_sampling: TrainSamplingConfig | Mapping[str, Any] | None = None,
    max_records_total: int | None = None,
    max_records_per_scene: int | None = None,
    seed: int = 0,
) -> TileRecordBuildResult:
    import rasterio

    cfg = train_sampling if isinstance(train_sampling, TrainSamplingConfig) else resolve_train_sampling_config({"train_sampling": train_sampling or {}})
    positive_stride = cfg.effective_positive_stride(stride)
    hard_negative_stride = cfg.effective_hard_negative_stride(stride)
    negative_stride = cfg.effective_negative_stride(stride)
    warnings = list(cfg.warnings)
    context_px = cfg.hard_negative_context_px
    if context_px is None:
        context_px = max(1, int(tile_size) // 2)
        warnings.append("train_sampling.hard_negative_context_px is not set; using tile_size // 2 for hard negatives")

    all_records: list[TileSampleRecord] = []
    scene_reports: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for scene_index, scene in enumerate(scenes):
        scene_id, image_path, scene_meta = _scene_fields(scene)
        if not image_path:
            warnings.append(f"scene {scene_id} has no image_path and was skipped")
            continue
        scene_records: list[TileSampleRecord] = []
        try:
            with rasterio.open(image_path) as ds:
                scene_bounds = box(*ds.bounds)
                positive_scene = any(
                    geom.is_valid and (not geom.is_empty) and geom.intersects(scene_bounds)
                    for geom in shapes
                )
                positive_windows = generate_window_grid_for_scene(
                    ds.width,
                    ds.height,
                    tile_size,
                    positive_stride,
                    scene_id=scene_id,
                )
                positive_bboxes: list[tuple[int, int, int, int]] = []
                by_window: dict[tuple[int, int, int, int], TileSampleRecord] = {}

                for window in positive_windows:
                    record = _record_for_window(
                        ds,
                        scene_id,
                        image_path,
                        window,
                        shapes,
                        cfg,
                        source="positive_grid",
                        hard_negative_context_px=None,
                        positive_bboxes=[],
                        scene_meta=scene_meta,
                    )
                    if record.kind in {"positive", "partial_positive"}:
                        _upsert_record(by_window, record)
                        if record.kind == "positive":
                            positive_bboxes.append(_record_bbox(record))

                for window in generate_window_grid_for_scene(ds.width, ds.height, tile_size, hard_negative_stride, scene_id=scene_id):
                    record = _record_for_window(
                        ds,
                        scene_id,
                        image_path,
                        window,
                        shapes,
                        cfg,
                        source="hard_negative_grid",
                        hard_negative_context_px=context_px,
                        positive_bboxes=positive_bboxes,
                        scene_meta=scene_meta,
                    )
                    if record.kind in {"positive", "partial_positive", "hard_negative"}:
                        _upsert_record(by_window, record)

                for window in generate_window_grid_for_scene(ds.width, ds.height, tile_size, negative_stride, scene_id=scene_id):
                    record = _record_for_window(
                        ds,
                        scene_id,
                        image_path,
                        window,
                        shapes,
                        cfg,
                        source="negative_grid",
                        hard_negative_context_px=context_px,
                        positive_bboxes=positive_bboxes,
                        scene_meta=scene_meta,
                    )
                    _upsert_record(by_window, record)

                scene_records = list(by_window.values())
                if bool((cfg.random_jitter or {}).get("enabled")):
                    jittered: list[TileSampleRecord] = []
                    for record in scene_records:
                        jittered.append(
                            _jitter_record(
                                ds,
                                record,
                                shapes,
                                cfg,
                                context_px,
                                positive_bboxes,
                                random.Random(rng.randint(0, 2**31 - 1)),
                            )
                        )
                    scene_records = jittered

                scene_records.sort(key=lambda item: (_kind_priority(item.kind), item.y, item.x, item.stride))
                if max_records_per_scene is not None and len(scene_records) > max_records_per_scene:
                    scene_records = _limit_records(scene_records, max_records_per_scene, rng)
                all_records.extend(scene_records)
                summary = summarize_tile_records(scene_records, prefix="")
                scene_reports.append(
                    {
                        "scene": scene_id,
                        "source_image_path": image_path,
                        "width": ds.width,
                        "height": ds.height,
                        "bands": ds.count,
                        "crs": str(ds.crs),
                        "positive_scene": bool(positive_scene),
                        "samples": len(scene_records),
                        "positive_tiles": summary["positive_tiles"],
                        "partial_positive_tiles": summary["partial_positive_tiles"],
                        "hard_negative_tiles": summary["hard_negative_tiles"],
                        "negative_tiles": summary["negative_tiles"],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"scene {scene_id} was skipped during virtual train sampling: {type(exc).__name__}: {exc}")
            continue

        if max_records_total is not None and len(all_records) >= max_records_total:
            all_records = _limit_records(all_records, max_records_total, rng)
            break

    if max_records_total is not None and len(all_records) > max_records_total:
        all_records = _limit_records(all_records, max_records_total, rng)

    return TileRecordBuildResult(
        records=all_records,
        scene_reports=scene_reports,
        warnings=warnings,
        metadata={
            "positive_stride": positive_stride,
            "hard_negative_stride": hard_negative_stride,
            "negative_stride": negative_stride,
            "tile_size": int(tile_size),
            "stride": int(stride),
        },
    )


def build_validation_records(
    scenes: list[Any],
    shapes: list[Any],
    *,
    tile_size: int,
    stride: int,
    max_records_total: int | None = None,
    max_records_per_scene: int | None = None,
    seed: int = 0,
    min_positive_pixels: int = 1,
    include_partial_positive: bool = True,
    partial_positive_fraction: float = 0.0,
) -> TileRecordBuildResult:
    import rasterio

    warnings: list[str] = []
    all_records: list[TileSampleRecord] = []
    scene_reports: list[dict[str, Any]] = []
    rng = random.Random(seed)
    cfg = TrainSamplingConfig(
        min_positive_pixels=max(1, int(min_positive_pixels)),
        include_partial_positive=bool(include_partial_positive),
        partial_positive_fraction=max(0.0, float(partial_positive_fraction)),
        hard_negative_context_px=0,
    )
    for scene in scenes:
        scene_id, image_path, scene_meta = _scene_fields(scene)
        if not image_path:
            warnings.append(f"validation scene {scene_id} has no image_path and was skipped")
            continue
        try:
            with rasterio.open(image_path) as ds:
                scene_records: list[TileSampleRecord] = []
                for window in generate_window_grid_for_scene(ds.width, ds.height, tile_size, stride, scene_id=scene_id):
                    record = _record_for_window(
                        ds,
                        scene_id,
                        image_path,
                        window,
                        shapes,
                        cfg,
                        source="validation_grid",
                        hard_negative_context_px=None,
                        positive_bboxes=[],
                        scene_meta=scene_meta,
                    )
                    scene_records.append(record)
                scene_records.sort(key=lambda item: (item.y, item.x))
                if max_records_per_scene is not None and len(scene_records) > max_records_per_scene:
                    scene_records = scene_records[:max_records_per_scene]
                all_records.extend(scene_records)
                summary = summarize_tile_records(scene_records, prefix="")
                scene_reports.append(
                    {
                        "scene": scene_id,
                        "source_image_path": image_path,
                        "width": ds.width,
                        "height": ds.height,
                        "bands": ds.count,
                        "crs": str(ds.crs),
                        "positive_scene": bool(summary["positive_tiles"] or summary["partial_positive_tiles"]),
                        "samples": len(scene_records),
                        "positive_tiles": summary["positive_tiles"],
                        "partial_positive_tiles": summary["partial_positive_tiles"],
                        "hard_negative_tiles": 0,
                        "negative_tiles": summary["negative_tiles"],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"validation scene {scene_id} was skipped: {type(exc).__name__}: {exc}")
            continue
        if max_records_total is not None and len(all_records) >= max_records_total:
            all_records = all_records[:max_records_total]
            break

    if max_records_total is not None and len(all_records) > max_records_total:
        all_records = all_records[:max_records_total]
    return TileRecordBuildResult(
        records=all_records,
        scene_reports=scene_reports,
        warnings=warnings,
        metadata={"positive_stride": int(stride), "hard_negative_stride": int(stride), "negative_stride": int(stride)},
    )


def apply_virtual_repeats(records: list[TileSampleRecord], cfg: TrainSamplingConfig) -> list[TileSampleRecord]:
    repeated: list[TileSampleRecord] = []
    for record in records:
        if record.kind in {"positive", "partial_positive"}:
            factor = cfg.positive_repeat_factor
        elif record.kind == "hard_negative":
            factor = cfg.hard_negative_repeat_factor
        else:
            factor = cfg.negative_repeat_factor
        base_id = record.base_record_id or record.record_id
        for repeat_index in range(max(1, int(factor))):
            metadata = dict(record.metadata)
            metadata.setdefault("base_record_id", base_id)
            metadata["record_id"] = f"{base_id}:repeat{repeat_index}" if factor > 1 else base_id
            repeated.append(
                TileSampleRecord(
                    scene_id=record.scene_id,
                    image_path=record.image_path,
                    x=record.x,
                    y=record.y,
                    width=record.width,
                    height=record.height,
                    tile_size=record.tile_size,
                    stride=record.stride,
                    kind=record.kind,
                    positive_pixels=record.positive_pixels,
                    source=record.source if repeat_index == 0 else "virtual_repeat",
                    base_record_id=base_id,
                    repeat_index=repeat_index if factor > 1 else None,
                    metadata=metadata,
                )
            )
    return repeated


def limit_empty_tile_share(
    records: list[TileSampleRecord],
    max_empty_tile_share: float | None,
    *,
    seed: int = 0,
) -> tuple[list[TileSampleRecord], list[str]]:
    if max_empty_tile_share is None:
        return list(records), []
    share = max(0.0, min(1.0, float(max_empty_tile_share)))
    if share >= 1.0 or not records:
        return list(records), []
    empty_indices = [index for index, record in enumerate(records) if record.kind in {"negative", "hard_negative"}]
    filled_indices = [index for index, record in enumerate(records) if record.kind not in {"negative", "hard_negative"}]
    warnings: list[str] = []
    if not empty_indices:
        return list(records), []
    if not filled_indices:
        warnings.append("max_empty_tile_share cannot be enforced because there are no positive or partial-positive records")
        return list(records), warnings
    max_empty = int(math.floor((share * len(filled_indices)) / max(1e-12, 1.0 - share)))
    if len(empty_indices) <= max_empty:
        return list(records), []
    rng = random.Random(seed)
    selected_empty = set(rng.sample(empty_indices, max(0, max_empty)))
    selected = set(filled_indices) | selected_empty
    limited = [record for index, record in enumerate(records) if index in selected]
    warnings.append(
        "max_empty_tile_share limited empty train records "
        f"from {len(empty_indices)} to {len(selected_empty)} with max_empty_tile_share={share}"
    )
    return limited, warnings


def build_balanced_epoch_indices(
    records: list[TileSampleRecord],
    cfg: TrainSamplingConfig,
    *,
    seed: int = 0,
) -> tuple[list[int], list[str]]:
    if not records:
        return [], ["cannot build balanced epoch indices for an empty record list"]
    total = max(1, len(records) * max(1, int(cfg.virtual_epoch_multiplier)))
    requested = {
        "positive": cfg.batch_positive_fraction,
        "hard_negative": cfg.batch_hard_negative_fraction,
        "negative": cfg.batch_negative_fraction,
    }
    any_fraction = any(value is not None for value in requested.values())
    rng = random.Random(seed)
    if not any_fraction:
        indices = list(range(len(records))) * max(1, int(cfg.virtual_epoch_multiplier))
        rng.shuffle(indices)
        return indices, []

    groups = _record_groups(records)
    warnings: list[str] = []
    fractions = _resolve_group_fractions(requested, groups, warnings)
    counts = _fraction_counts(total, fractions)
    indices: list[int] = []
    for group_name, count in counts.items():
        if count <= 0:
            continue
        group = list(groups.get(group_name) or [])
        if not group:
            warnings.append(f"batch balancing requested {count} {group_name} samples, but the group is empty")
            continue
        indices.extend(_sample_group_indices(group, count, rng))
    if len(indices) < total:
        fallback = list(range(len(records)))
        warnings.append(f"batch balancing filled {total - len(indices)} samples from all available groups")
        indices.extend(_sample_group_indices(fallback, total - len(indices), rng))
    rng.shuffle(indices)
    return indices[:total], warnings


def summarize_tile_records(records: list[TileSampleRecord], *, prefix: str = "") -> dict[str, Any]:
    counts = Counter(record.kind for record in records)
    total = len(records)
    empty = int(counts.get("negative", 0) + counts.get("hard_negative", 0))
    stem = f"{prefix}_" if prefix else ""
    return {
        f"{stem}tile_count": total,
        f"{stem}positive_tiles": int(counts.get("positive", 0)),
        f"{stem}partial_positive_tiles": int(counts.get("partial_positive", 0)),
        f"{stem}hard_negative_tiles": int(counts.get("hard_negative", 0)),
        f"{stem}negative_tiles": int(counts.get("negative", 0)),
        f"{stem}empty_tiles": empty,
        f"{stem}empty_tile_share": (float(empty) / total) if total else 0.0,
    }


def preview_from_manifest(
    *,
    dataset_manifest: str | Path,
    annotation: str | Path,
    images_dir: str | Path,
    config: Mapping[str, Any] | None = None,
    max_scenes: int | None = None,
    max_records_preview: int = 20,
) -> dict[str, Any]:
    import json

    manifest_path = Path(dataset_manifest)
    annotation_path = Path(annotation)
    images_root = Path(images_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotation_payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    shapes = [shape_from_feature(feature) for feature in annotation_payload.get("features") or [] if feature.get("geometry")]
    cfg_payload = dict(config or {})
    preprocess = dict(cfg_payload.get("preprocess") or {})
    train_sampling = resolve_train_sampling_config(preprocess, cfg_payload.get("train") or {})
    tile_size = int(preprocess.get("tile_size") or (cfg_payload.get("train") or {}).get("patch_size") or 768)
    stride = int(preprocess.get("stride") or 512)
    train_scenes = _manifest_scenes(payload, "train_scenes", images_root)
    val_scenes = _manifest_scenes(payload, "val_scenes", images_root)
    if max_scenes is not None:
        limit = max(1, int(max_scenes))
        train_scenes = train_scenes[:limit]
        val_scenes = val_scenes[:limit]

    if train_sampling.enabled:
        train_result = build_virtual_train_records(
            train_scenes,
            shapes,
            tile_size=tile_size,
            stride=stride,
            train_sampling=train_sampling,
            max_records_total=_optional_int(preprocess.get("max_train_tiles")),
            max_records_per_scene=_optional_int(preprocess.get("max_tiles_per_scene")),
            seed=int(preprocess.get("split_seed") or preprocess.get("seed") or 42),
        )
        base_records, limit_warnings = limit_empty_tile_share(
            train_result.records,
            train_sampling.max_empty_tile_share,
            seed=int(preprocess.get("split_seed") or preprocess.get("seed") or 42),
        )
        virtual_records = apply_virtual_repeats(base_records, train_sampling)
        virtual_records, virtual_limit_warnings = limit_empty_tile_share(
            virtual_records,
            train_sampling.max_empty_tile_share,
            seed=int(preprocess.get("split_seed") or preprocess.get("seed") or 42) + 1,
        )
        epoch_indices, balance_warnings = build_balanced_epoch_indices(
            virtual_records,
            train_sampling,
            seed=int(preprocess.get("split_seed") or preprocess.get("seed") or 42) + 2,
        )
        warnings = train_result.warnings + limit_warnings + virtual_limit_warnings + balance_warnings
    else:
        train_result = build_validation_records(
            train_scenes,
            shapes,
            tile_size=tile_size,
            stride=stride,
            max_records_total=_optional_int(preprocess.get("max_train_tiles")),
            max_records_per_scene=_optional_int(preprocess.get("max_tiles_per_scene")),
            seed=int(preprocess.get("split_seed") or preprocess.get("seed") or 42),
        )
        base_records = train_result.records
        virtual_records = train_result.records
        epoch_indices = list(range(len(virtual_records)))
        warnings = train_result.warnings

    val_result = build_validation_records(
        val_scenes,
        shapes,
        tile_size=tile_size,
        stride=stride,
        max_records_total=_optional_int(preprocess.get("max_val_tiles")),
        max_records_per_scene=_optional_int(preprocess.get("max_val_tiles_per_scene")),
        seed=int(preprocess.get("split_seed") or preprocess.get("seed") or 42) + 1000,
        min_positive_pixels=train_sampling.min_positive_pixels,
        include_partial_positive=train_sampling.include_partial_positive,
        partial_positive_fraction=train_sampling.partial_positive_fraction,
    )
    warnings.extend(val_result.warnings)
    train_summary = summarize_tile_records(virtual_records, prefix="train")
    base_summary = summarize_tile_records(base_records, prefix="base_train")
    val_summary = summarize_tile_records(val_result.records, prefix="val")
    summary = {
        "total_scenes": len(train_scenes) + len(val_scenes),
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "train_sampling_enabled": train_sampling.enabled,
        "base_train_records": len(base_records),
        "virtual_train_records": len(virtual_records),
        "effective_train_samples_per_epoch": len(epoch_indices),
        **base_summary,
        **train_summary,
        **val_summary,
        "positive_stride": train_sampling.effective_positive_stride(stride),
        "hard_negative_stride": train_sampling.effective_hard_negative_stride(stride),
        "negative_stride": train_sampling.effective_negative_stride(stride),
        "virtual_epoch_multiplier": train_sampling.virtual_epoch_multiplier,
        "positive_repeat_factor": train_sampling.positive_repeat_factor,
        "hard_negative_repeat_factor": train_sampling.hard_negative_repeat_factor,
        "negative_repeat_factor": train_sampling.negative_repeat_factor,
        "max_empty_tile_share": train_sampling.max_empty_tile_share,
        "batch_positive_fraction": train_sampling.batch_positive_fraction,
        "batch_hard_negative_fraction": train_sampling.batch_hard_negative_fraction,
        "batch_negative_fraction": train_sampling.batch_negative_fraction,
        "warnings": warnings,
    }
    return {
        "status": "ok",
        "summary": summary,
        "preview": [record.to_dict() for record in virtual_records[: max(0, int(max_records_preview))]],
        "validation_preview": [record.to_dict() for record in val_result.records[: max(0, int(max_records_preview))]],
    }


def shape_from_feature(feature: dict[str, Any]) -> Any:
    from shapely.geometry import shape

    return shape(feature["geometry"])


def _effective_stride(base_stride: int, factor: float) -> int:
    return max(1, int(max(1, int(base_stride)) * float(factor)))


def _optional_fraction(value: Any) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return max(1, int(value))


def _origins(length: int, tile: int, stride: int) -> list[int]:
    if tile <= 0:
        raise ValueError("tile_size must be positive")
    stride = max(1, int(stride))
    if length <= tile:
        return [0]
    values = list(range(0, max(1, length - tile + 1), stride))
    edge = max(0, length - tile)
    if not values or values[-1] != edge:
        values.append(edge)
    return sorted(set(max(0, int(value)) for value in values))


def _scene_fields(scene: Any) -> tuple[str, str | None, dict[str, Any]]:
    if isinstance(scene, Mapping):
        scene_id = str(scene.get("scene_id") or scene.get("name") or scene.get("entry") or scene.get("key") or "")
        image_path = scene.get("image_path") or scene.get("path") or scene.get("source_image_path")
        metadata = dict(scene)
    else:
        scene_id = str(getattr(scene, "scene_id", None) or getattr(scene, "name", None) or getattr(scene, "entry", None) or getattr(scene, "key", None) or "")
        image_path = getattr(scene, "image_path", None) or getattr(scene, "path", None) or getattr(scene, "source_image_path", None)
        metadata = dict(getattr(scene, "__dict__", {}) or {})
    if not scene_id and image_path:
        scene_id = PurePosixPath(str(image_path)).name
    return scene_id, str(image_path) if image_path else None, metadata


def _manifest_scenes(payload: dict[str, Any], key: str, images_root: Path) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    for item in payload.get(key) or []:
        row = dict(item)
        image_path = _resolve_manifest_image_path(row, images_root)
        row["image_path"] = str(image_path)
        row["scene_id"] = str(row.get("name") or row.get("entry") or image_path.name)
        scenes.append(row)
    return scenes


def _resolve_manifest_image_path(row: dict[str, Any], images_root: Path) -> Path:
    candidates = [
        row.get("image_path"),
        row.get("path"),
        row.get("source_image_path"),
        row.get("key"),
        row.get("name"),
        row.get("entry"),
    ]
    for raw in candidates:
        if not raw:
            continue
        value = str(raw)
        path = Path(value)
        if path.is_absolute() and path.exists():
            return path
        direct = images_root / value.replace("\\", "/")
        if direct.exists():
            return direct
        basename = images_root / PurePosixPath(value.replace("\\", "/")).name
        if basename.exists():
            return basename
    name = str(row.get("name") or row.get("entry") or row.get("key") or "")
    return images_root / PurePosixPath(name.replace("\\", "/")).name


def _record_for_window(
    ds: Any,
    scene_id: str,
    image_path: str,
    window: WindowCandidate,
    shapes: list[Any],
    cfg: TrainSamplingConfig,
    *,
    source: str,
    hard_negative_context_px: int | None,
    positive_bboxes: list[tuple[int, int, int, int]],
    scene_meta: dict[str, Any],
) -> TileSampleRecord:
    mask = _rasterize_window_mask(ds, window.x, window.y, window.width, window.height, shapes)
    kind, positive_pixels = classify_tile_by_mask(
        mask,
        min_positive_pixels=cfg.min_positive_pixels,
        include_partial_positive=cfg.include_partial_positive,
        partial_positive_fraction=cfg.partial_positive_fraction,
    )
    if kind == "negative" and positive_pixels == 0:
        if classify_hard_negative(_window_bbox(window), positive_bboxes, hard_negative_context_px):
            kind = "hard_negative"
    metadata = {
        "record_id": f"{scene_id}:x{window.x}:y{window.y}:w{window.width}:h{window.height}:{source}",
        "window_index": window.index,
        "source_grid": source,
        "scene_metadata": scene_meta,
    }
    if scene_meta.get("s3_key") or scene_meta.get("key"):
        metadata["s3_key"] = scene_meta.get("s3_key") or scene_meta.get("key")
    return TileSampleRecord(
        scene_id=scene_id,
        image_path=image_path,
        x=window.x,
        y=window.y,
        width=window.width,
        height=window.height,
        tile_size=window.tile_size,
        stride=window.stride,
        kind=kind,
        positive_pixels=positive_pixels,
        source=source,
        metadata=metadata,
    )


def _rasterize_window_mask(
    ds: Any,
    x: int,
    y: int,
    width: int,
    height: int,
    shapes: list[Any],
) -> np.ndarray:
    window = Window(int(x), int(y), int(width), int(height))
    window_bounds = box(*ds.window_bounds(window))
    geometries = [(geom, 1) for geom in shapes if geom.is_valid and (not geom.is_empty) and geom.intersects(window_bounds)]
    if not geometries:
        return np.zeros((int(height), int(width)), dtype="uint8")
    return rasterize(
        geometries,
        out_shape=(int(height), int(width)),
        transform=ds.window_transform(window),
        fill=0,
        dtype="uint8",
    )


def _upsert_record(records: dict[tuple[int, int, int, int], TileSampleRecord], candidate: TileSampleRecord) -> None:
    key = (candidate.x, candidate.y, candidate.width, candidate.height)
    existing = records.get(key)
    if existing is None or _kind_priority(candidate.kind) < _kind_priority(existing.kind):
        records[key] = candidate


def _kind_priority(kind: TileKind) -> int:
    return {"positive": 0, "partial_positive": 1, "hard_negative": 2, "negative": 3}[kind]


def _record_bbox(record: TileSampleRecord) -> tuple[int, int, int, int]:
    return int(record.x), int(record.y), int(record.x + record.width), int(record.y + record.height)


def _window_bbox(window: WindowCandidate) -> tuple[int, int, int, int]:
    return int(window.x), int(window.y), int(window.x + window.width), int(window.y + window.height)


def _limit_records(records: list[TileSampleRecord], max_records: int, rng: random.Random) -> list[TileSampleRecord]:
    if len(records) <= max_records:
        return records
    by_kind: dict[str, list[TileSampleRecord]] = defaultdict(list)
    for record in records:
        by_kind[record.kind].append(record)
    selected: list[TileSampleRecord] = []
    for kind in ("positive", "partial_positive", "hard_negative", "negative"):
        bucket = list(by_kind.get(kind) or [])
        rng.shuffle(bucket)
        take = min(len(bucket), max(0, max_records - len(selected)))
        selected.extend(bucket[:take])
        if len(selected) >= max_records:
            break
    selected.sort(key=lambda item: (item.scene_id, _kind_priority(item.kind), item.y, item.x))
    return selected


def _jitter_record(
    ds: Any,
    record: TileSampleRecord,
    shapes: list[Any],
    cfg: TrainSamplingConfig,
    hard_negative_context_px: int | None,
    positive_bboxes: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> TileSampleRecord:
    jitter_cfg = cfg.random_jitter or {}
    max_shift = max(0, int(record.tile_size * float(jitter_cfg.get("max_shift_fraction", 0.25))))
    if max_shift <= 0:
        return record
    keep_positive_min_pixels = max(1, int(jitter_cfg.get("keep_positive_min_pixels", 1) or 1))
    max_x = max(0, int(ds.width) - int(record.width))
    max_y = max(0, int(ds.height) - int(record.height))
    attempts = max(4, int(jitter_cfg.get("attempts", 8) or 8))
    for _attempt in range(attempts):
        dx = rng.randint(-max_shift, max_shift)
        dy = rng.randint(-max_shift, max_shift)
        x = max(0, min(max_x, int(record.x) + dx))
        y = max(0, min(max_y, int(record.y) + dy))
        if x == record.x and y == record.y:
            continue
        window = WindowCandidate(
            scene_id=record.scene_id,
            x=x,
            y=y,
            width=record.width,
            height=record.height,
            tile_size=record.tile_size,
            stride=record.stride,
            index=int(record.metadata.get("window_index") or 0),
        )
        jittered = _record_for_window(
            ds,
            record.scene_id,
            str(record.image_path or ""),
            window,
            shapes,
            cfg,
            source=f"{record.source}_jitter",
            hard_negative_context_px=hard_negative_context_px,
            positive_bboxes=positive_bboxes,
            scene_meta=dict((record.metadata or {}).get("scene_metadata") or {}),
        )
        if record.kind in {"positive", "partial_positive"} and jittered.positive_pixels < keep_positive_min_pixels:
            continue
        metadata = dict(jittered.metadata)
        metadata["jitter"] = {"enabled": True, "original_x": record.x, "original_y": record.y, "dx": x - record.x, "dy": y - record.y}
        jittered.metadata = metadata
        return jittered
    metadata = dict(record.metadata)
    metadata["jitter"] = {"enabled": True, "fallback_to_original": True}
    return TileSampleRecord(
        scene_id=record.scene_id,
        image_path=record.image_path,
        x=record.x,
        y=record.y,
        width=record.width,
        height=record.height,
        tile_size=record.tile_size,
        stride=record.stride,
        kind=record.kind,
        positive_pixels=record.positive_pixels,
        source=record.source,
        base_record_id=record.base_record_id,
        repeat_index=record.repeat_index,
        metadata=metadata,
    )


def _record_groups(records: list[TileSampleRecord]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {"positive": [], "hard_negative": [], "negative": []}
    for index, record in enumerate(records):
        if record.kind in {"positive", "partial_positive"}:
            groups["positive"].append(index)
        elif record.kind == "hard_negative":
            groups["hard_negative"].append(index)
        else:
            groups["negative"].append(index)
    return groups


def _resolve_group_fractions(
    requested: dict[str, float | None],
    groups: dict[str, list[int]],
    warnings: list[str],
) -> dict[str, float]:
    specified = {key: value for key, value in requested.items() if value is not None}
    specified_sum = sum(float(value) for value in specified.values())
    if specified_sum > 1.0:
        warnings.append(f"batch fractions sum to {specified_sum:.3f}; normalizing to 1.0")
        return {key: float(value) / specified_sum for key, value in specified.items()}
    result = {key: float(value) for key, value in specified.items()}
    remainder = max(0.0, 1.0 - specified_sum)
    unspecified = [key for key in ("positive", "hard_negative", "negative") if key not in result and groups.get(key)]
    if remainder > 0 and unspecified:
        total_available = sum(len(groups[key]) for key in unspecified)
        for key in unspecified:
            result[key] = remainder * (len(groups[key]) / total_available)
    return {key: result.get(key, 0.0) for key in ("positive", "hard_negative", "negative")}


def _fraction_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {key: max(0.0, float(value)) * total for key, value in fractions.items()}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda key: raw[key] - math.floor(raw[key]), reverse=True)
    for key in order:
        if remaining <= 0:
            break
        counts[key] += 1
        remaining -= 1
    return counts


def _sample_group_indices(group: list[int], count: int, rng: random.Random) -> list[int]:
    if count <= 0 or not group:
        return []
    result: list[int] = []
    while len(result) < count:
        shuffled = list(group)
        rng.shuffle(shuffled)
        result.extend(shuffled[: count - len(result)])
    return result[:count]


# Compatibility API below. The training path uses mlsystem.src.tile_preparation
# directly; these wrappers keep older debug scripts/tests on the same source of
# truth without importing this module from the train module.
from ..tile_preparation import (  # noqa: E402
    AnnotationInput as _TPAnnotationInput,
    SceneInput as _TPSceneInput,
    TilePreparationConfig as _TPTilePreparationConfig,
)
from ..tile_preparation.iterator import (  # noqa: E402
    apply_virtual_repeats as _tp_apply_virtual_repeats,
    build_balanced_epoch_records as _tp_build_balanced_epoch_records,
    build_tile_records as _tp_build_tile_records,
    build_validation_tile_records as _tp_build_validation_tile_records,
    limit_empty_tile_share as _tp_limit_empty_tile_share,
)
from ..tile_preparation.records import TileSampleRecord as _TPTileSampleRecord  # noqa: E402
from ..tile_preparation.summary import summarize_tile_records as _tp_summarize_tile_records  # noqa: E402


def _tp_config_from_sampling(cfg: TrainSamplingConfig, *, tile_size: int, stride: int, seed: int, max_records: int | None, max_records_per_scene: int | None) -> _TPTilePreparationConfig:
    return _TPTilePreparationConfig(
        tile_size=tile_size,
        stride=stride,
        positive_stride_factor=cfg.positive_stride_factor,
        hard_negative_stride_factor=cfg.hard_negative_stride_factor,
        negative_stride_factor=cfg.negative_stride_factor,
        min_positive_pixels=cfg.min_positive_pixels,
        include_partial_positive=cfg.include_partial_positive,
        partial_positive_fraction=cfg.partial_positive_fraction,
        max_empty_tile_share=cfg.max_empty_tile_share,
        hard_negative_context_px=cfg.hard_negative_context_px,
        virtual_epoch_multiplier=cfg.virtual_epoch_multiplier,
        positive_repeat_factor=cfg.positive_repeat_factor,
        hard_negative_repeat_factor=cfg.hard_negative_repeat_factor,
        negative_repeat_factor=cfg.negative_repeat_factor,
        batch_positive_fraction=cfg.batch_positive_fraction,
        batch_hard_negative_fraction=cfg.batch_hard_negative_fraction,
        batch_negative_fraction=cfg.batch_negative_fraction,
        max_records=max_records,
        max_records_per_scene=max_records_per_scene,
        seed=seed,
    )


def _to_tp_scene(scene: Any) -> _TPSceneInput:
    scene_id, image_path, metadata = _scene_fields(scene)
    if not image_path:
        raise ValueError(f"scene {scene_id} has no image_path")
    return _TPSceneInput(image_path=str(image_path), scene_id=scene_id, metadata=metadata)


def _tp_to_legacy_record(record: _TPTileSampleRecord) -> TileSampleRecord:
    return TileSampleRecord(
        scene_id=record.scene_id,
        image_path=record.image_path,
        x=record.x,
        y=record.y,
        width=record.width,
        height=record.height,
        tile_size=record.tile_size,
        stride=record.stride,
        kind=record.kind,
        positive_pixels=record.positive_pixels,
        source=record.source,
        base_record_id=record.base_record_id,
        repeat_index=record.repeat_index,
        metadata={**dict(record.metadata), "geometries_intersecting": record.geometries_intersecting},
    )


def _legacy_to_tp_record(record: TileSampleRecord) -> _TPTileSampleRecord:
    return _TPTileSampleRecord(
        scene_id=record.scene_id,
        image_path=record.image_path,
        x=record.x,
        y=record.y,
        width=record.width,
        height=record.height,
        tile_size=record.tile_size,
        stride=record.stride,
        kind=record.kind,
        positive_pixels=record.positive_pixels,
        source=record.source,
        base_record_id=record.base_record_id,
        repeat_index=record.repeat_index,
        metadata=dict(record.metadata),
    )


def _write_shapes_geojson_for_scene(shapes: list[Any], scene: _TPSceneInput) -> Path:
    import tempfile
    import rasterio
    from shapely.geometry import mapping as shapely_mapping

    with rasterio.open(str(scene.image_path)) as ds:
        crs = str(ds.crs) if ds.crs else None
    payload: dict[str, Any] = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {}, "geometry": shapely_mapping(geom)} for geom in shapes],
    }
    if crs:
        payload["crs"] = {"type": "name", "properties": {"name": crs}}
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".geojson", delete=False)
    with handle:
        json.dump(payload, handle)
    return Path(handle.name)


def build_virtual_train_records(  # type: ignore[no-redef]
    scenes: list[Any],
    shapes: list[Any],
    *,
    tile_size: int,
    stride: int,
    train_sampling: TrainSamplingConfig | Mapping[str, Any] | None = None,
    max_records_total: int | None = None,
    max_records_per_scene: int | None = None,
    seed: int = 0,
) -> TileRecordBuildResult:
    cfg = train_sampling if isinstance(train_sampling, TrainSamplingConfig) else resolve_train_sampling_config({"train_sampling": train_sampling or {}})
    tp_scenes = [_to_tp_scene(scene) for scene in scenes]
    if not tp_scenes:
        return TileRecordBuildResult(records=[], scene_reports=[], warnings=["no scenes were provided"])
    annotation_path = _write_shapes_geojson_for_scene(shapes, tp_scenes[0])
    try:
        result = _tp_build_tile_records(
            tp_scenes,
            _TPAnnotationInput(annotation_path, annotation_crs="auto", allow_inferred_annotation_crs=True),
            _tp_config_from_sampling(
                cfg,
                tile_size=tile_size,
                stride=stride,
                seed=seed,
                max_records=max_records_total,
                max_records_per_scene=max_records_per_scene,
            ),
        )
        return TileRecordBuildResult(
            records=[_tp_to_legacy_record(record) for record in result.records],
            scene_reports=result.scene_reports,
            warnings=result.warnings,
            metadata=result.metadata,
        )
    finally:
        try:
            annotation_path.unlink()
        except OSError:
            pass


def build_validation_records(  # type: ignore[no-redef]
    scenes: list[Any],
    shapes: list[Any],
    *,
    tile_size: int,
    stride: int,
    max_records_total: int | None = None,
    max_records_per_scene: int | None = None,
    seed: int = 0,
    min_positive_pixels: int = 1,
    include_partial_positive: bool = True,
    partial_positive_fraction: float = 0.0,
) -> TileRecordBuildResult:
    tp_scenes = [_to_tp_scene(scene) for scene in scenes]
    if not tp_scenes:
        return TileRecordBuildResult(records=[], scene_reports=[], warnings=["no scenes were provided"])
    annotation_path = _write_shapes_geojson_for_scene(shapes, tp_scenes[0])
    try:
        result = _tp_build_validation_tile_records(
            tp_scenes,
            _TPAnnotationInput(annotation_path, annotation_crs="auto", allow_inferred_annotation_crs=True),
            _TPTilePreparationConfig(
                tile_size=tile_size,
                stride=stride,
                min_positive_pixels=min_positive_pixels,
                include_partial_positive=include_partial_positive,
                partial_positive_fraction=partial_positive_fraction,
                max_records=max_records_total,
                max_records_per_scene=max_records_per_scene,
                seed=seed,
            ),
        )
        return TileRecordBuildResult(
            records=[_tp_to_legacy_record(record) for record in result.records],
            scene_reports=result.scene_reports,
            warnings=result.warnings,
            metadata=result.metadata,
        )
    finally:
        try:
            annotation_path.unlink()
        except OSError:
            pass


def apply_virtual_repeats(records: list[TileSampleRecord], cfg: TrainSamplingConfig) -> list[TileSampleRecord]:  # type: ignore[no-redef]
    tp_config = _tp_config_from_sampling(cfg, tile_size=1, stride=1, seed=0, max_records=None, max_records_per_scene=None)
    repeated = _tp_apply_virtual_repeats([_legacy_to_tp_record(record) for record in records], tp_config)
    return [_tp_to_legacy_record(record) for record in repeated]


def limit_empty_tile_share(  # type: ignore[no-redef]
    records: list[TileSampleRecord],
    max_empty_tile_share: float | None,
    *,
    seed: int = 0,
) -> tuple[list[TileSampleRecord], list[str]]:
    limited, warnings = _tp_limit_empty_tile_share([_legacy_to_tp_record(record) for record in records], max_empty_tile_share, seed=seed)
    return [_tp_to_legacy_record(record) for record in limited], warnings


def build_balanced_epoch_indices(  # type: ignore[no-redef]
    records: list[TileSampleRecord],
    cfg: TrainSamplingConfig,
    *,
    seed: int = 0,
) -> tuple[list[int], list[str]]:
    tp_records = [_legacy_to_tp_record(record) for record in records]
    tp_config = _tp_config_from_sampling(cfg, tile_size=1, stride=1, seed=seed, max_records=None, max_records_per_scene=None)
    balanced, warnings = _tp_build_balanced_epoch_records(tp_records, tp_config, seed=seed)
    available: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(tp_records):
        available[record.record_id].append(index)
    indices: list[int] = []
    for record in balanced:
        bucket = available.get(record.record_id) or []
        indices.append(bucket[0] if bucket else 0)
    return indices, warnings


def summarize_tile_records(records: list[TileSampleRecord], *, prefix: str = "") -> dict[str, Any]:  # type: ignore[no-redef]
    return _tp_summarize_tile_records([_legacy_to_tp_record(record) for record in records], prefix=prefix)


def preview_from_manifest(  # type: ignore[no-redef]
    *,
    dataset_manifest: str | Path,
    annotation: str | Path,
    images_dir: str | Path,
    config: Mapping[str, Any] | None = None,
    max_scenes: int | None = None,
    max_records_preview: int = 20,
) -> dict[str, Any]:
    import rasterio

    manifest_path = Path(dataset_manifest)
    images_root = Path(images_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg_payload = dict(config or {})
    preprocess = dict(cfg_payload.get("preprocess") or cfg_payload)
    tile_size = int(preprocess.get("tile_size") or 768)
    stride = int(preprocess.get("stride") or 512)
    raw_sampling = dict(preprocess.get("train_sampling") or {})
    train_cfg = resolve_train_sampling_config(preprocess)
    train_scenes = [_to_tp_scene(row) for row in _manifest_scenes(payload, "train_scenes", images_root)]
    val_scenes = [_to_tp_scene(row) for row in _manifest_scenes(payload, "val_scenes", images_root)]
    if max_scenes is not None:
        train_scenes = train_scenes[: max(1, int(max_scenes))]
        val_scenes = val_scenes[: max(1, int(max_scenes))]
    explicit_crs: str | None = None
    first_scene = (train_scenes or val_scenes or [None])[0]
    if first_scene is not None:
        with rasterio.open(str(first_scene.image_path)) as ds:
            explicit_crs = str(ds.crs) if ds.crs else None
    annotation_input = _TPAnnotationInput(Path(annotation), annotation_crs=explicit_crs or "auto", allow_inferred_annotation_crs=True)
    train_config = _tp_config_from_sampling(train_cfg, tile_size=tile_size, stride=stride, seed=0, max_records=None, max_records_per_scene=None)
    val_config = _TPTilePreparationConfig(
        tile_size=tile_size,
        stride=stride,
        min_positive_pixels=train_cfg.min_positive_pixels,
        include_partial_positive=train_cfg.include_partial_positive,
        partial_positive_fraction=train_cfg.partial_positive_fraction,
    )
    train_result = _tp_build_tile_records(train_scenes, annotation_input, train_config)
    val_result = _tp_build_validation_tile_records(val_scenes, annotation_input, val_config)
    train_summary = _tp_summarize_tile_records(train_result.records, prefix="train")
    base_summary = _tp_summarize_tile_records(train_result.base_records, prefix="base_train")
    val_summary = _tp_summarize_tile_records(val_result.records, prefix="val")
    summary = {
        "train_sampling_enabled": bool(raw_sampling.get("enabled", False)),
        "base_train_records": len(train_result.base_records),
        "virtual_train_records": len(train_result.records),
        "effective_train_samples_per_epoch": len(train_result.records),
        "positive_stride": train_config.positive_stride,
        "hard_negative_stride": train_config.hard_negative_stride,
        "negative_stride": train_config.negative_stride,
        **base_summary,
        **train_summary,
        **val_summary,
        "warnings": train_result.warnings + val_result.warnings,
    }
    return {
        "status": "ok",
        "summary": summary,
        "preview": [record.to_dict() for record in train_result.records[: max(0, int(max_records_preview))]],
        "warnings": summary["warnings"],
    }
