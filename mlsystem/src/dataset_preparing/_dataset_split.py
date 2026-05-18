from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any, TypeVar

T = TypeVar("T")

IMAGE_EXTENSIONS = (".tif", ".tiff", ".TIF", ".TIFF")
SCENE_PROPERTY_FIELDS = (
    "scene",
    "scene_id",
    "scene_name",
    "image",
    "image_name",
    "image_id",
    "filename",
    "file",
    "tif",
    "tiff",
    "raster",
    "source",
    "source_file",
    "src",
)


@dataclass
class SceneFilterResult:
    existing_scenes: list[str]
    missing_scenes: list[str]
    scene_to_image: dict[str, Path]
    matched_by: dict[str, str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class CleanSceneListResult:
    scene_list: Path
    backup_path: Path | None
    report_path: Path
    total_lines: int
    kept_count: int
    removed_count: int
    removed_scenes: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class LoadedFeature:
    properties: dict[str, Any]
    geometry: Any | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class SceneObjectCount:
    scene_name: str
    image_path: Path | None
    object_count: int
    matched_by: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class TrainValSplit:
    train: list[SceneObjectCount]
    val: list[SceneObjectCount]
    summary: dict[str, Any]


def train_val_split(items: list[T], train_fraction: float = 0.75) -> tuple[list[T], list[T]]:
    if not items:
        return [], []
    split_idx = max(1, int(math.ceil(len(items) * train_fraction)))
    train_items = items[:split_idx]
    val_items = items[split_idx:] or items[-1:]
    return train_items, val_items


def split_manifest_scene_rows(items: list[T], train_fraction: float = 0.75) -> tuple[list[T], list[T]]:
    if not items:
        return [], []
    split_idx = max(1, int(len(items) * train_fraction))
    train_items = items[:split_idx]
    val_items = items[split_idx:] or items[-1:]
    return train_items, val_items


def read_scene_list(txt_path: Path) -> list[str]:
    scenes: list[str] = []
    for raw_line in Path(txt_path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        scenes.append(line.split()[0])
    return scenes


def _append_index(bucket: dict[str, list[Path]], key: str, path: Path) -> None:
    if key:
        bucket[key].append(path)


def index_image_files(
    images_dir: Path,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
    *,
    recursive: bool = True,
) -> dict[str, Any]:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    normalized_ext = {item.lower() for item in extensions}
    paths = [
        path
        for path in (images_dir.rglob("*") if recursive else images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in normalized_ext
    ]
    paths.sort(key=lambda item: str(item).lower())

    by_name: dict[str, list[Path]] = defaultdict(list)
    by_name_lower: dict[str, list[Path]] = defaultdict(list)
    by_stem: dict[str, list[Path]] = defaultdict(list)
    by_stem_lower: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        _append_index(by_name, path.name, path)
        _append_index(by_name_lower, path.name.lower(), path)
        _append_index(by_stem, path.stem, path)
        _append_index(by_stem_lower, path.stem.lower(), path)

    warnings: list[str] = []
    for index_name, bucket in (("filename", by_name_lower), ("stem", by_stem_lower)):
        duplicates = {key: values for key, values in bucket.items() if len(values) > 1}
        for key, values in sorted(duplicates.items()):
            joined = "; ".join(str(path) for path in values)
            warnings.append(f"duplicate_{index_name}: {key}: {joined}")

    return {
        "images_dir": images_dir,
        "paths": paths,
        "by_name": dict(by_name),
        "by_name_lower": dict(by_name_lower),
        "by_stem": dict(by_stem),
        "by_stem_lower": dict(by_stem_lower),
        "warnings": warnings,
        "recursive": recursive,
        "extensions": tuple(extensions),
    }


def _scene_basename(value: str) -> str:
    text = str(value).strip().strip('"').strip("'")
    text = text.replace("\\", "/")
    return PurePath(text).name


def _scene_stem(value: str) -> str:
    name = _scene_basename(value)
    suffix = Path(name).suffix
    return Path(name).stem if suffix.lower() in {".tif", ".tiff"} else name


def _normalized_scene_key(value: str) -> str:
    name = _scene_stem(value).lower()
    name = re.sub(r"\.aux\.xml$", "", name)
    name = re.sub(r"[_\-. ]?cog$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def _pick_unique_match(scene: str, values: list[Path], matched_by: str, warnings: list[str]) -> tuple[Path | None, str | None]:
    if not values:
        return None, None
    if len(values) > 1:
        joined = "; ".join(str(path) for path in values)
        warnings.append(f"duplicate_match for {scene} by {matched_by}: {joined}; using first sorted path")
    return values[0], matched_by


def filter_existing_scenes(scene_names: list[str], image_index: dict[str, Any]) -> SceneFilterResult:
    existing: list[str] = []
    missing: list[str] = []
    mapping: dict[str, Path] = {}
    matched_by: dict[str, str] = {}
    warnings = list(image_index.get("warnings") or [])

    normalized_paths: dict[str, list[Path]] = defaultdict(list)
    for path in image_index.get("paths") or []:
        normalized_paths[_normalized_scene_key(path.name)].append(path)

    for scene in scene_names:
        name = _scene_basename(scene)
        stem = _scene_stem(scene)
        candidates: list[tuple[str, list[Path]]] = [
            ("filename_exact", image_index.get("by_name", {}).get(name, [])),
            ("stem_exact", image_index.get("by_stem", {}).get(stem, [])),
            ("filename_casefold", image_index.get("by_name_lower", {}).get(name.lower(), [])),
            ("stem_casefold", image_index.get("by_stem_lower", {}).get(stem.lower(), [])),
            ("normalized_scene", normalized_paths.get(_normalized_scene_key(scene), [])),
        ]
        selected_path: Path | None = None
        selected_reason: str | None = None
        for reason, values in candidates:
            selected_path, selected_reason = _pick_unique_match(scene, values, reason, warnings)
            if selected_path is not None:
                break
        if selected_path is None:
            missing.append(scene)
            continue
        existing.append(scene)
        mapping[scene] = selected_path
        matched_by[scene] = selected_reason or "unknown"

    return SceneFilterResult(existing, missing, mapping, matched_by, warnings)


def clean_scene_list_file(
    scene_list: Path,
    images_dir: Path,
    *,
    output_report: Path | None = None,
    in_place: bool = True,
    backup: bool = True,
    recursive: bool = True,
) -> CleanSceneListResult:
    scene_list = Path(scene_list)
    if not scene_list.exists():
        raise FileNotFoundError(f"Scene list does not exist: {scene_list}")
    scenes = read_scene_list(scene_list)
    image_index = index_image_files(Path(images_dir), recursive=recursive)
    filtered = filter_existing_scenes(scenes, image_index)

    backup_path: Path | None = None
    if in_place and backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = scene_list.with_name(f"{scene_list.name}.bak_{timestamp}")
        shutil.copy2(scene_list, backup_path)

    if in_place:
        scene_list.write_text("\n".join(filtered.existing_scenes) + ("\n" if filtered.existing_scenes else ""), encoding="utf-8")

    report_path = Path(output_report) if output_report else scene_list.with_name(f"{scene_list.stem}.missing_removed.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"scene_list={scene_list}",
        f"images_dir={Path(images_dir)}",
        f"backup_path={backup_path or ''}",
        f"total_lines={len(scenes)}",
        f"kept={len(filtered.existing_scenes)}",
        f"removed={len(filtered.missing_scenes)}",
        "",
        "[removed]",
        *filtered.missing_scenes,
        "",
        "[warnings]",
        *filtered.warnings,
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return CleanSceneListResult(
        scene_list=scene_list,
        backup_path=backup_path,
        report_path=report_path,
        total_lines=len(scenes),
        kept_count=len(filtered.existing_scenes),
        removed_count=len(filtered.missing_scenes),
        removed_scenes=filtered.missing_scenes,
        warnings=filtered.warnings,
    )


def load_geojson_features(annotation_path: Path) -> list[LoadedFeature]:
    try:
        from shapely.geometry import shape
    except Exception:  # noqa: BLE001 - property-based counting does not need geometry operations.
        shape = None

    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8-sig"))
    if payload.get("type") == "FeatureCollection":
        raw_features = payload.get("features") or []
    elif payload.get("type") == "Feature":
        raw_features = [payload]
    else:
        raw_features = [{"type": "Feature", "properties": {}, "geometry": payload}]

    loaded: list[LoadedFeature] = []
    for index, feature in enumerate(raw_features):
        properties = feature.get("properties") or {}
        warnings: list[str] = []
        geom = None
        geometry_payload = feature.get("geometry")
        if geometry_payload:
            if shape is None:
                geom = geometry_payload
            else:
                try:
                    candidate = shape(geometry_payload)
                    if candidate.is_empty or not candidate.is_valid:
                        warnings.append(f"feature_{index}: empty_or_invalid_geometry")
                    else:
                        geom = candidate
                except Exception as exc:  # noqa: BLE001 - report bad features without failing the whole file.
                    warnings.append(f"feature_{index}: bad_geometry: {exc}")
        else:
            warnings.append(f"feature_{index}: missing_geometry")
        loaded.append(LoadedFeature(properties=dict(properties), geometry=geom, warnings=warnings))
    return loaded


def load_geojson_crs(annotation_path: Path) -> str | None:
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8-sig"))
    crs = payload.get("crs")
    if not crs:
        return None
    if isinstance(crs, dict):
        properties = crs.get("properties") or {}
        name = properties.get("name") or crs.get("name")
        if name:
            value = str(name)
            match = re.search(r"EPSG[:/](\d+)", value, flags=re.IGNORECASE)
            return f"EPSG:{match.group(1)}" if match else value
    if isinstance(crs, str):
        return crs
    return None


def extract_scene_name_from_feature_properties(properties: dict[str, Any]) -> str | None:
    for field_name in SCENE_PROPERTY_FIELDS:
        if field_name not in properties or properties[field_name] is None:
            continue
        value = str(properties[field_name]).strip()
        if not value:
            continue
        embedded = re.search(r"([^\\/\s;,\"]+\.(?:tif|tiff))", value, flags=re.IGNORECASE)
        if embedded:
            return _scene_basename(embedded.group(1))
        basename = _scene_basename(value)
        if basename:
            return basename
    return None


def _scene_name_lookup(scene_names: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for scene in scene_names:
        for key in {
            _scene_basename(scene),
            _scene_basename(scene).lower(),
            _scene_stem(scene),
            _scene_stem(scene).lower(),
            _normalized_scene_key(scene),
        }:
            lookup.setdefault(key, scene)
    return lookup


def _feature_scene_match(feature_scene_name: str, lookup: dict[str, str]) -> str | None:
    return (
        lookup.get(_scene_basename(feature_scene_name))
        or lookup.get(_scene_basename(feature_scene_name).lower())
        or lookup.get(_scene_stem(feature_scene_name))
        or lookup.get(_scene_stem(feature_scene_name).lower())
        or lookup.get(_normalized_scene_key(feature_scene_name))
    )


def _count_by_properties(scene_names: list[str], features: list[LoadedFeature]) -> tuple[Counter[str], int, list[str]]:
    lookup = _scene_name_lookup(scene_names)
    counts: Counter[str] = Counter()
    unmatched = 0
    warnings: list[str] = []
    for index, feature in enumerate(features):
        if feature.geometry is None:
            unmatched += 1
            continue
        feature_scene = extract_scene_name_from_feature_properties(feature.properties)
        if not feature_scene:
            unmatched += 1
            continue
        matched_scene = _feature_scene_match(feature_scene, lookup)
        if matched_scene:
            counts[matched_scene] += 1
        else:
            unmatched += 1
            warnings.append(f"feature_{index}: property scene not in scene list: {feature_scene}")
    return counts, unmatched, warnings


def _count_by_geometry(
    scene_names: list[str],
    scene_to_image: dict[str, Path],
    features: list[LoadedFeature],
    *,
    annotation_crs: str | None = None,
    allow_inferred_crs: bool = True,
) -> tuple[Counter[str], list[str]]:
    warnings: list[str] = []
    counts: Counter[str] = Counter()
    try:
        import rasterio
        from shapely.geometry import box
        from rasterio.warp import transform_bounds
    except Exception as exc:  # noqa: BLE001
        return counts, [f"geometry fallback unavailable: rasterio import failed: {exc}"]

    valid_features = [(index, feature.geometry) for index, feature in enumerate(features) if feature.geometry is not None]
    if not valid_features:
        return counts, ["geometry fallback skipped: no valid feature geometries"]

    inferred_crs = False
    if not annotation_crs:
        if not allow_inferred_crs:
            return counts, ["geometry fallback unavailable: annotation CRS is missing and inference is disabled"]
        max_abs = max(max(map(abs, geom.bounds)) for _, geom in valid_features)
        annotation_crs = "EPSG:3857" if max_abs > 1000 else "EPSG:4326"
        inferred_crs = True
    seen_features: set[int] = set()
    for scene in scene_names:
        image_path = scene_to_image.get(scene)
        if not image_path:
            warnings.append(f"{scene}: no image path for geometry fallback")
            continue
        try:
            with rasterio.open(image_path) as ds:
                scene_crs = str(ds.crs) if ds.crs else annotation_crs
                bounds = tuple(ds.bounds)
                if scene_crs != annotation_crs:
                    bounds = transform_bounds(scene_crs, annotation_crs, *bounds, densify_pts=21)
                scene_bounds = box(*bounds)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{scene}: cannot read raster bounds: {exc}")
            continue
        for index, geom in valid_features:
            if index in seen_features:
                continue
            if geom.intersects(scene_bounds):
                counts[scene] += 1
                seen_features.add(index)
    if inferred_crs:
        warnings.append(f"geometry fallback used with inferred annotation CRS {annotation_crs}")
    else:
        warnings.append(f"geometry fallback used with annotation CRS {annotation_crs}")
    return counts, warnings


def count_objects_per_scene(
    scene_names: list[str],
    scene_to_image: dict[str, Path],
    annotation_path: Path,
    *,
    count_mode: str = "auto",
    annotation_crs: str | None = None,
    allow_inferred_crs: bool = True,
) -> list[SceneObjectCount]:
    if count_mode not in {"auto", "property", "geometry"}:
        raise ValueError(f"Unsupported count_mode: {count_mode}")
    features = load_geojson_features(annotation_path)
    annotation_crs = annotation_crs or load_geojson_crs(annotation_path)
    feature_warnings = [warning for feature in features for warning in feature.warnings]

    property_counts: Counter[str] = Counter()
    property_unmatched = len(features)
    property_warnings: list[str] = []
    if count_mode in {"auto", "property"}:
        property_counts, property_unmatched, property_warnings = _count_by_properties(scene_names, features)

    counts = property_counts
    matched_by = "property"
    warnings = feature_warnings + property_warnings
    if count_mode == "geometry" or (
        count_mode == "auto"
        and features
        and property_unmatched > max(2, int(len(features) * 0.25))
        and sum(property_counts.values()) == 0
    ):
        geometry_counts, geometry_warnings = _count_by_geometry(
            scene_names,
            scene_to_image,
            features,
            annotation_crs=annotation_crs,
            allow_inferred_crs=allow_inferred_crs,
        )
        if count_mode == "geometry" or sum(geometry_counts.values()) > sum(property_counts.values()):
            counts = geometry_counts
            matched_by = "geometry"
        warnings.extend(geometry_warnings)
    elif count_mode == "auto":
        matched_by = "property_auto"

    rows: list[SceneObjectCount] = []
    for scene in scene_names:
        scene_warnings = list(warnings) if scene == scene_names[0] else []
        rows.append(
            SceneObjectCount(
                scene_name=scene,
                image_path=scene_to_image.get(scene),
                object_count=int(counts.get(scene, 0)),
                matched_by=matched_by,
                warnings=scene_warnings,
            )
        )
    return rows


def _split_summary(train: list[SceneObjectCount], val: list[SceneObjectCount], seed: int, target_val_fraction: float) -> dict[str, Any]:
    train_objects = sum(item.object_count for item in train)
    val_objects = sum(item.object_count for item in val)
    total_files = len(train) + len(val)
    total_objects = train_objects + val_objects
    rows = train + val
    warnings = [warning for item in rows for warning in item.warnings + item.errors]
    return {
        "train_files": len(train),
        "train_objects": train_objects,
        "val_files": len(val),
        "val_objects": val_objects,
        "total_files": total_files,
        "total_objects": total_objects,
        "scenes_with_objects": sum(1 for item in rows if item.object_count > 0),
        "scenes_without_objects": sum(1 for item in rows if item.object_count <= 0),
        "warnings_count": len(warnings),
        "count_modes": sorted(set(item.matched_by for item in rows)),
        "crs_source": _crs_source_from_warnings(warnings),
        "val_fraction_by_files": (len(val) / total_files) if total_files else 0.0,
        "val_fraction_by_objects": (val_objects / total_objects) if total_objects else 0.0,
        "seed": seed,
        "target_val_fraction": target_val_fraction,
    }


def _crs_source_from_warnings(warnings: list[str]) -> str:
    for warning in warnings:
        match = re.search(r"inferred annotation CRS ([A-Za-z0-9:]+)", warning)
        if match:
            return f"inferred:{match.group(1)}"
    for warning in warnings:
        match = re.search(r"annotation CRS ([A-Za-z0-9:]+)", warning)
        if match:
            return f"explicit_or_geojson:{match.group(1)}"
    return "not_used"


def split_train_val_by_object_counts(
    counts: list[SceneObjectCount],
    *,
    target_val_fraction: float = 0.2,
    seed: int = 42,
    target_val_objects: int | None = None,
    min_val_scenes: int | None = None,
    include_zero_object_scenes: bool = True,
) -> TrainValSplit:
    if not counts:
        return TrainValSplit([], [], _split_summary([], [], seed, target_val_fraction))
    rng = random.Random(seed)
    positives = [item for item in counts if item.object_count > 0]
    zeros = [item for item in counts if item.object_count <= 0]
    tie_break = {id(item): rng.random() for item in counts}
    positives.sort(key=lambda item: (-item.object_count, tie_break[id(item)], item.scene_name))
    zeros.sort(key=lambda item: (tie_break[id(item)], item.scene_name))

    total_objects = sum(item.object_count for item in positives)
    target_objects = target_val_objects if target_val_objects is not None else int(round(total_objects * target_val_fraction))
    if total_objects > 0:
        target_objects = max(1, target_objects)
    target_scene_count = max(1, int(min_val_scenes)) if min_val_scenes is not None and len(counts) > 1 else 1

    val_ids: set[int] = set()
    val_objects = 0
    for index, item in enumerate(positives):
        if not val_ids:
            remaining = positives[index + 1 :]
            if target_objects > 0 and item.object_count > target_objects and any(candidate.object_count <= target_objects for candidate in remaining):
                continue
            val_ids.add(id(item))
            val_objects += item.object_count
            continue
        if val_objects >= target_objects and len(val_ids) >= target_scene_count:
            break
        current_gap = abs(target_objects - val_objects)
        next_gap = abs(target_objects - (val_objects + item.object_count))
        if (val_objects < target_objects and val_objects + item.object_count <= target_objects) or next_gap <= current_gap or len(val_ids) < target_scene_count:
            val_ids.add(id(item))
            val_objects += item.object_count

    if total_objects > 0 and positives and not val_ids:
        fallback = min(
            positives,
            key=lambda item: (
                abs(target_objects - item.object_count),
                item.object_count,
                tie_break[id(item)],
                item.scene_name,
            ),
        )
        val_ids.add(id(fallback))

    if include_zero_object_scenes:
        desired_zero_val = int(round(len(zeros) * target_val_fraction))
        if zeros and len(counts) > 1 and desired_zero_val == 0:
            desired_zero_val = 1
        for item in zeros[:desired_zero_val]:
            val_ids.add(id(item))

    if len(counts) > 1 and not val_ids:
        val_ids.add(id(counts[-1]))
    if len(val_ids) == len(counts) and len(counts) > 1:
        removable = next((item for item in zeros if id(item) in val_ids), None) or next(item for item in positives if id(item) in val_ids)
        val_ids.remove(id(removable))

    train = [item for item in counts if id(item) not in val_ids]
    val = [item for item in counts if id(item) in val_ids]
    train_names = {item.scene_name for item in train}
    val_names = {item.scene_name for item in val}
    if train_names & val_names:
        raise RuntimeError(f"Train/val split overlap: {sorted(train_names & val_names)}")

    return TrainValSplit(train=train, val=val, summary=_split_summary(train, val, seed, target_val_fraction))


def write_scene_object_counts_report(rows: list[SceneObjectCount], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_objects = sum(row.object_count for row in rows)
    warnings_count = sum(len(row.warnings) + len(row.errors) for row in rows)
    lines = ["scene_name\tobject_count\timage_path\tmatched_by"]
    for row in rows:
        lines.append(f"{row.scene_name}\t{row.object_count}\t{row.image_path or ''}\t{row.matched_by}")
    lines.extend(
        [
            "",
            "[summary]",
            f"total_scenes={len(rows)}",
            f"total_objects={total_objects}",
            f"scenes_with_objects={sum(1 for row in rows if row.object_count > 0)}",
            f"scenes_without_objects={sum(1 for row in rows if row.object_count <= 0)}",
            f"warnings_count={warnings_count}",
        ]
    )
    if warnings_count:
        lines.append("")
        lines.append("[warnings]")
        for row in rows:
            for warning in row.warnings + row.errors:
                lines.append(f"{row.scene_name}\t{warning}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_train_val_split_report(split: TrainValSplit, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["[train]", "scene_name\tobject_count"]
    lines.extend(f"{item.scene_name}\t{item.object_count}" for item in split.train)
    lines.extend(["", "[val]", "scene_name\tobject_count"])
    lines.extend(f"{item.scene_name}\t{item.object_count}" for item in split.val)
    lines.extend(["", "[summary]"])
    for key, value in split.summary.items():
        lines.append(f"{key}={value}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_split_outputs(rows: list[SceneObjectCount], split: TrainValSplit, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts_report = write_scene_object_counts_report(rows, output_dir / "scene_object_counts.txt")
    split_report = write_train_val_split_report(split, output_dir / "train_val_split.txt")
    train_path = output_dir / "train.txt"
    val_path = output_dir / "val.txt"
    summary_path = output_dir / "split_summary.json"
    train_path.write_text("\n".join(item.scene_name for item in split.train) + ("\n" if split.train else ""), encoding="utf-8")
    val_path.write_text("\n".join(item.scene_name for item in split.val) + ("\n" if split.val else ""), encoding="utf-8")
    payload = {
        "summary": split.summary,
        "warnings": [warning for item in rows for warning in item.warnings + item.errors],
        "train": [asdict(item) for item in split.train],
        "val": [asdict(item) for item in split.val],
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "scene_object_counts": counts_report,
        "train_val_split": split_report,
        "train": train_path,
        "val": val_path,
        "split_summary": summary_path,
    }


def run_split(
    scene_list: Path,
    images_dir: Path,
    annotation: Path,
    output_dir: Path,
    *,
    target_val_fraction: float = 0.2,
    seed: int = 42,
    count_mode: str = "auto",
) -> dict[str, Path]:
    scenes = read_scene_list(scene_list)
    image_index = index_image_files(images_dir)
    filtered = filter_existing_scenes(scenes, image_index)
    if filtered.missing_scenes:
        missing = ", ".join(filtered.missing_scenes[:10])
        raise RuntimeError(f"Scene list contains {len(filtered.missing_scenes)} missing images. Run clean-list first. First missing: {missing}")
    rows = count_objects_per_scene(filtered.existing_scenes, filtered.scene_to_image, annotation, count_mode=count_mode)
    for row in rows:
        row.warnings.extend(filtered.warnings if row is rows[0] else [])
        row.matched_by = f"{row.matched_by};image_{filtered.matched_by.get(row.scene_name, 'unknown')}"
    split = split_train_val_by_object_counts(rows, target_val_fraction=target_val_fraction, seed=seed)
    return write_split_outputs(rows, split, output_dir)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean scene lists and build deterministic train/val splits.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("clean-list")
    clean.add_argument("--scene-list", required=True, type=Path)
    clean.add_argument("--images-dir", required=True, type=Path)
    clean.add_argument("--output-report", type=Path)
    clean.add_argument("--in-place", action="store_true")
    clean.add_argument("--backup", action="store_true")
    clean.add_argument("--no-recursive", action="store_true")

    split = subparsers.add_parser("split")
    split.add_argument("--scene-list", required=True, type=Path)
    split.add_argument("--images-dir", required=True, type=Path)
    split.add_argument("--annotation", required=True, type=Path)
    split.add_argument("--output-dir", required=True, type=Path)
    split.add_argument("--target-val-fraction", type=float, default=0.2)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--count-mode", choices=("auto", "property", "geometry"), default="auto")

    handtest = subparsers.add_parser("run-handtest")
    handtest.add_argument("--scene-list", required=True, type=Path)
    handtest.add_argument("--images-dir", required=True, type=Path)
    handtest.add_argument("--annotation", required=True, type=Path)
    handtest.add_argument("--output-dir", required=True, type=Path)
    handtest.add_argument("--target-val-fraction", type=float, default=0.2)
    handtest.add_argument("--seed", type=int, default=42)
    handtest.add_argument("--count-mode", choices=("auto", "property", "geometry"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "clean-list":
        result = clean_scene_list_file(
            args.scene_list,
            args.images_dir,
            output_report=args.output_report,
            in_place=args.in_place,
            backup=args.backup,
            recursive=not args.no_recursive,
        )
        print(f"kept={result.kept_count} removed={result.removed_count}")
        print(f"report={result.report_path}")
        if result.backup_path:
            print(f"backup={result.backup_path}")
        return 0

    output_paths = run_split(
        args.scene_list,
        args.images_dir,
        args.annotation,
        args.output_dir,
        target_val_fraction=args.target_val_fraction,
        seed=args.seed,
        count_mode=args.count_mode,
    )
    print(f"scene_object_counts={output_paths['scene_object_counts']}")
    print(f"train_val_split={output_paths['train_val_split']}")
    print(f"train={output_paths['train']}")
    print(f"val={output_paths['val']}")
    print(f"split_summary={output_paths['split_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
