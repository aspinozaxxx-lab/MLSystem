from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.validation import make_valid


@dataclass(frozen=True)
class ObjectMatch:
    pred_index: int
    gt_index: int
    iou: float
    pred_area: float
    gt_area: float
    intersection_area: float
    union_area: float


def _clean_geometry(geom: BaseGeometry | None) -> BaseGeometry | None:
    if geom is None or geom.is_empty:
        return None
    try:
        cleaned = make_valid(geom)
    except Exception:
        cleaned = geom.buffer(0)
    if cleaned is None or cleaned.is_empty:
        return None
    if cleaned.geom_type == "GeometryCollection":
        parts = [part for part in cleaned.geoms if part.geom_type in {"Polygon", "MultiPolygon"} and not part.is_empty]
        if not parts:
            return None
        cleaned = parts[0] if len(parts) == 1 else make_valid(unary_union(parts))
    if cleaned.geom_type not in {"Polygon", "MultiPolygon"}:
        return None
    return cleaned


def clean_geometries(geoms: Iterable[BaseGeometry | None]) -> list[BaseGeometry]:
    out: list[BaseGeometry] = []
    for geom in geoms:
        cleaned = _clean_geometry(geom)
        if cleaned is not None and cleaned.area > 0:
            out.append(cleaned)
    return out


def reproject_geometries(
    geoms: Iterable[BaseGeometry],
    source_crs: str | None,
    target_crs: str | None,
) -> list[BaseGeometry]:
    if not source_crs or not target_crs or source_crs == target_crs:
        return list(geoms)
    from pyproj import Transformer

    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return [transform(transformer.transform, geom) for geom in geoms]


def prepare_geometries(
    geoms: Iterable[BaseGeometry | None],
    source_crs: str | None = None,
    target_crs: str | None = None,
) -> list[BaseGeometry]:
    cleaned = clean_geometries(geoms)
    return clean_geometries(reproject_geometries(cleaned, source_crs, target_crs))


def compute_pairwise_iou(pred_geoms: Iterable[BaseGeometry], gt_geoms: Iterable[BaseGeometry]) -> list[list[float]]:
    preds = list(pred_geoms)
    gts = list(gt_geoms)
    matrix: list[list[float]] = []
    for pred in preds:
        row: list[float] = []
        pred_area = float(pred.area)
        for gt in gts:
            gt_area = float(gt.area)
            if pred_area <= 0 or gt_area <= 0 or not pred.bounds or not gt.bounds:
                row.append(0.0)
                continue
            if not pred.intersects(gt):
                row.append(0.0)
                continue
            intersection = float(pred.intersection(gt).area)
            union = pred_area + gt_area - intersection
            row.append(intersection / union if union > 0 else 0.0)
        matrix.append(row)
    return matrix


def match_objects_by_iou(
    pred_geoms: Iterable[BaseGeometry | None],
    gt_geoms: Iterable[BaseGeometry | None],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    preds = clean_geometries(pred_geoms)
    gts = clean_geometries(gt_geoms)
    candidates: list[tuple[float, int, int, float, float, float, float]] = []
    for pred_index, pred in enumerate(preds):
        pred_area = float(pred.area)
        for gt_index, gt in enumerate(gts):
            gt_area = float(gt.area)
            if pred_area <= 0 or gt_area <= 0 or not pred.intersects(gt):
                continue
            intersection = float(pred.intersection(gt).area)
            union = pred_area + gt_area - intersection
            iou = intersection / union if union > 0 else 0.0
            if iou > iou_threshold:
                candidates.append((iou, pred_index, gt_index, pred_area, gt_area, intersection, union))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_preds: set[int] = set()
    used_gts: set[int] = set()
    matches: list[ObjectMatch] = []
    for iou, pred_index, gt_index, pred_area, gt_area, intersection, union in candidates:
        if pred_index in used_preds or gt_index in used_gts:
            continue
        used_preds.add(pred_index)
        used_gts.add(gt_index)
        matches.append(
            ObjectMatch(
                pred_index=pred_index,
                gt_index=gt_index,
                iou=float(iou),
                pred_area=float(pred_area),
                gt_area=float(gt_area),
                intersection_area=float(intersection),
                union_area=float(union),
            )
        )

    unmatched_pred_indices = [idx for idx in range(len(preds)) if idx not in used_preds]
    unmatched_gt_indices = [idx for idx in range(len(gts)) if idx not in used_gts]
    return {
        "matches": matches,
        "unmatched_pred_indices": unmatched_pred_indices,
        "unmatched_gt_indices": unmatched_gt_indices,
        "pred_geometries": preds,
        "gt_geometries": gts,
        "iou_threshold": float(iou_threshold),
        "matching_method": "greedy_descending_iou",
    }


def compute_object_f1(
    pred_geoms: Iterable[BaseGeometry | None],
    gt_geoms: Iterable[BaseGeometry | None],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    match_result = match_objects_by_iou(pred_geoms, gt_geoms, iou_threshold=iou_threshold)
    tp = len(match_result["matches"])
    fp = len(match_result["unmatched_pred_indices"])
    fn = len(match_result["unmatched_gt_indices"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {
        "object_tp": tp,
        "object_fp": fp,
        "object_fn": fn,
        "object_precision": float(precision),
        "object_recall": float(recall),
        "object_f1": float(f1),
        "object_iou_threshold": float(iou_threshold),
        "matching_method": match_result["matching_method"],
        "matches": [match.__dict__ for match in match_result["matches"]],
        "unmatched_pred_indices": match_result["unmatched_pred_indices"],
        "unmatched_gt_indices": match_result["unmatched_gt_indices"],
    }


def compute_object_metrics_by_scene(
    pred_by_scene: dict[str, Iterable[BaseGeometry | None]],
    gt_by_scene: dict[str, Iterable[BaseGeometry | None]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    scene_ids = sorted(set(pred_by_scene) | set(gt_by_scene))
    scene_metrics: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    for scene_id in scene_ids:
        metrics = compute_object_f1(
            pred_by_scene.get(scene_id, []),
            gt_by_scene.get(scene_id, []),
            iou_threshold=iou_threshold,
        )
        scene_metrics[scene_id] = metrics
        total_tp += int(metrics["object_tp"])
        total_fp += int(metrics["object_fp"])
        total_fn += int(metrics["object_fn"])
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) else 0.0
    return {
        "object_tp": total_tp,
        "object_fp": total_fp,
        "object_fn": total_fn,
        "object_precision": float(precision),
        "object_recall": float(recall),
        "object_f1": float(f1),
        "object_iou_threshold": float(iou_threshold),
        "matching_method": "greedy_descending_iou",
        "scenes": scene_metrics,
    }
