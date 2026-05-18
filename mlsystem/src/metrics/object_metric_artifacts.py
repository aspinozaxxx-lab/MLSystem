from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from shapely.geometry import shape

from ..storage.api import write_json
from .object_metrics import compute_object_f1


def write_object_metrics_artifacts(
    experiment_dir: Path,
    pred_features: list[dict[str, Any]],
    gt_shapes: list[Any] | None,
    prefix: str = "val",
    iou_threshold: float = 0.5,
) -> tuple[dict[str, float], list[Path]]:
    if not gt_shapes:
        return {}, []
    pred_geoms = [shape(feature["geometry"]) for feature in pred_features if feature.get("geometry")]
    metrics = compute_object_f1(pred_geoms, gt_shapes, iou_threshold=iou_threshold)
    metrics_payload = {
        "schema_version": 1,
        "method": "CHTZ Appendix G object F1: ObjF1 = 2TP / (2TP + FP + FN), TP if polygon IoU > 0.5",
        "matching_method": metrics["matching_method"],
        "object_iou_threshold": iou_threshold,
        "scope": prefix,
        "metrics": {
            "object_tp": metrics["object_tp"],
            "object_fp": metrics["object_fp"],
            "object_fn": metrics["object_fn"],
            "object_precision": metrics["object_precision"],
            "object_recall": metrics["object_recall"],
            "object_f1": metrics["object_f1"],
        },
        "matches": metrics["matches"],
        "unmatched_pred_indices": metrics["unmatched_pred_indices"],
        "unmatched_gt_indices": metrics["unmatched_gt_indices"],
    }
    metrics_path = experiment_dir / "object_metrics.json"
    matches_path = experiment_dir / "object_matches.csv"
    write_json(metrics_path, metrics_payload)
    with matches_path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["pred_index", "gt_index", "iou", "pred_area", "gt_area", "intersection_area", "union_area"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["matches"]:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return (
        {
            f"{prefix}/object_tp": float(metrics["object_tp"]),
            f"{prefix}/object_fp": float(metrics["object_fp"]),
            f"{prefix}/object_fn": float(metrics["object_fn"]),
            f"{prefix}/object_precision": float(metrics["object_precision"]),
            f"{prefix}/object_recall": float(metrics["object_recall"]),
            f"{prefix}/object_f1": float(metrics["object_f1"]),
            "best_val_object_f1" if prefix == "val" else f"{prefix}/best_object_f1": float(metrics["object_f1"]),
        },
        [metrics_path, matches_path],
    )
