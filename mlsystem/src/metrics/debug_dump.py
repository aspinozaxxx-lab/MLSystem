from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .segmentation import PixelCounts, metrics_from_counts, pixel_counts_from_masks


def metrics_debug_enabled(config: dict[str, Any] | None = None) -> bool:
    env_value = os.getenv("MLSYSTEM_METRICS_DEBUG", "").strip().lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    if not isinstance(config, dict):
        return False
    return bool(config.get("enabled", False))


def metrics_debug_config(job: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for source in (
        getattr(job, "params", None),
        getattr(job, "train", None),
        getattr(job, "evaluate", None),
    ):
        if isinstance(source, dict) and isinstance(source.get("metrics_debug"), dict):
            payload.update(source["metrics_debug"])
    env_class = os.getenv("MLSYSTEM_METRICS_DEBUG_CLASS")
    if env_class and not payload.get("class_name"):
        payload["class_name"] = env_class
    if metrics_debug_enabled(payload):
        payload["enabled"] = True
    payload.setdefault("class_name", getattr(job, "class_name", None) or "foreground")
    payload.setdefault("class_id", None)
    payload.setdefault("save_every_epoch", True)
    payload.setdefault("save_all_val_samples", True)
    payload.setdefault("save_arrays", True)
    payload.setdefault("save_png", True)
    payload.setdefault("save_object_matching", True)
    payload.setdefault("upload_to_mlflow", True)
    return payload


def write_epoch_debug(
    *,
    root_dir: Path,
    run_id: str,
    epoch: int,
    mlflow_run_id: str | None,
    model_checkpoint_path: str | None,
    val_manifest_path: str | None,
    val_manifest: list[dict[str, Any]],
    class_name: str,
    class_id: int | str | None,
    threshold: float,
    metric_row: dict[str, Any],
    train_loss: dict[str, Any],
    val_loss: dict[str, Any],
    per_sample_metrics: list[dict[str, Any]],
    sample_payloads: list[dict[str, Any]],
    logged_metrics: dict[str, Any],
    formulas: dict[str, str] | None = None,
) -> dict[str, Any]:
    epoch_dir = root_dir / str(run_id) / f"epoch_{int(epoch):04d}"
    samples_dir = epoch_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = epoch_dir / "per_sample_metrics.csv"
    _write_csv(per_sample_csv, per_sample_metrics)

    manifest_path = epoch_dir / "val_manifest_snapshot.json"
    _write_json(manifest_path, val_manifest)

    recompute = recompute_global_metrics(per_sample_metrics, metric_row)
    recompute_path = epoch_dir / "metrics_recompute_check.json"
    _write_json(recompute_path, recompute)

    logged_path = epoch_dir / "mlflow_logged_metrics.json"
    _write_json(logged_path, logged_metrics)

    for payload in sample_payloads:
        _write_sample_debug(samples_dir, payload)

    summary = {
        "epoch": int(epoch),
        "run_id": run_id,
        "mlflow_run_id": mlflow_run_id,
        "model_checkpoint_path": model_checkpoint_path,
        "val_manifest_path": val_manifest_path,
        "class_name": class_name,
        "class_id": class_id,
        "threshold": float(threshold),
        "metric_formulas": formulas or default_metric_formulas(),
        "global_pixel_tp": _int_metric(metric_row, "val/pixel_tp"),
        "global_pixel_fp": _int_metric(metric_row, "val/pixel_fp"),
        "global_pixel_fn": _int_metric(metric_row, "val/pixel_fn"),
        "global_pixel_tn": _int_metric(metric_row, "val/pixel_tn"),
        "pixel_precision": _float_metric(metric_row, "val/precision"),
        "pixel_recall": _float_metric(metric_row, "val/recall"),
        "pixel_f1": _float_metric(metric_row, "val/pixel_f1"),
        "pixel_iou": _float_metric(metric_row, "val/pixel_iou"),
        "object_tp": _int_metric(metric_row, "val/object_tp"),
        "object_fp": _int_metric(metric_row, "val/object_fp"),
        "object_fn": _int_metric(metric_row, "val/object_fn"),
        "object_precision": _float_metric(metric_row, "val/object_precision"),
        "object_recall": _float_metric(metric_row, "val/object_recall"),
        "object_f1": _float_metric(metric_row, "val/object_f1"),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "count_val_samples": len(per_sample_metrics),
        "count_positive_gt_pixels": sum(int(row.get("gt_positive_pixels") or 0) for row in per_sample_metrics),
        "count_positive_pred_pixels": sum(int(row.get("pred_positive_pixels") or 0) for row in per_sample_metrics),
        "count_gt_objects": sum(int(row.get("gt_objects_count") or 0) for row in per_sample_metrics),
        "count_pred_objects": sum(int(row.get("pred_objects_count") or 0) for row in per_sample_metrics),
        "per_sample_metrics": str(per_sample_csv),
        "val_manifest_snapshot": str(manifest_path),
        "metrics_recompute_check": str(recompute_path),
        "mlflow_logged_metrics": str(logged_path),
        "samples_dir": str(samples_dir),
    }
    summary_path = epoch_dir / "epoch_summary.json"
    _write_json(summary_path, summary)
    return {
        "epoch_dir": str(epoch_dir),
        "epoch_summary": str(summary_path),
        "per_sample_metrics": str(per_sample_csv),
        "val_manifest_snapshot": str(manifest_path),
        "metrics_recompute_check": str(recompute_path),
        "mlflow_logged_metrics": str(logged_path),
        "recompute": recompute,
    }


def recompute_global_metrics(per_sample_metrics: list[dict[str, Any]], metric_row: dict[str, Any]) -> dict[str, Any]:
    counts = PixelCounts()
    for row in per_sample_metrics:
        counts.tp += int(row.get("tp") or 0)
        counts.fp += int(row.get("fp") or 0)
        counts.fn += int(row.get("fn") or 0)
        counts.tn += int(row.get("tn") or 0)
    metrics = metrics_from_counts(counts)
    comparisons: dict[str, Any] = {}
    for source_key, recomputed_key in (
        ("val/pixel_tp", "pixel_tp"),
        ("val/pixel_fp", "pixel_fp"),
        ("val/pixel_fn", "pixel_fn"),
        ("val/pixel_tn", "pixel_tn"),
        ("val/precision", "pixel_precision"),
        ("val/recall", "pixel_recall"),
        ("val/pixel_f1", "pixel_f1"),
        ("val/pixel_iou", "pixel_iou"),
    ):
        logged = metric_row.get(source_key)
        recomputed = metrics.get(recomputed_key)
        if logged is None or recomputed is None:
            continue
        delta = abs(float(logged) - float(recomputed))
        comparisons[source_key] = {
            "logged": float(logged),
            "recomputed": float(recomputed),
            "abs_delta": delta,
            "ok": delta <= 1e-6,
        }
    return {
        "counts": metrics,
        "comparisons": comparisons,
        "ok": all(item["ok"] for item in comparisons.values()),
    }


def default_metric_formulas() -> dict[str, str]:
    return {
        "tp": "pred == 1 and gt == 1",
        "fp": "pred == 1 and gt == 0",
        "fn": "pred == 0 and gt == 1",
        "tn": "pred == 0 and gt == 0",
        "precision": "TP / (TP + FP)",
        "recall": "TP / (TP + FN)",
        "f1": "2TP / (2TP + FP + FN)",
        "iou": "TP / (TP + FP + FN)",
        "aggregation": "micro over all validation pixels",
    }


def sample_metrics_from_arrays(
    *,
    sample_id: str,
    scene_id: str,
    tile_id: str,
    source_image_path: str | None,
    gt_mask_path: str | None,
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    pred_mask: np.ndarray,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    counts = pixel_counts_from_masks(pred_mask, gt_mask)
    metrics = metrics_from_counts(counts)
    gt_objects = _objects_from_mask(gt_mask)
    pred_objects = _objects_from_mask(pred_mask)
    object_metrics = _object_metrics(pred_objects, gt_objects)
    prob = np.asarray(pred_prob, dtype=np.float32)
    return {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "tile_id": tile_id,
        "source_image_path": source_image_path,
        "gt_mask_path": gt_mask_path,
        "image_shape": "x".join(str(part) for part in np.asarray(image).shape),
        "gt_shape": "x".join(str(part) for part in np.asarray(gt_mask).shape),
        "pred_shape": "x".join(str(part) for part in np.asarray(pred_mask).shape),
        "prob_min": float(np.min(prob)) if prob.size else 0.0,
        "prob_max": float(np.max(prob)) if prob.size else 0.0,
        "prob_mean": float(np.mean(prob)) if prob.size else 0.0,
        "prob_std": float(np.std(prob)) if prob.size else 0.0,
        "gt_positive_pixels": int(np.count_nonzero(np.asarray(gt_mask) > 0.5)),
        "pred_positive_pixels": int(np.count_nonzero(np.asarray(pred_mask) > 0.5)),
        "tp": int(metrics["pixel_tp"]),
        "fp": int(metrics["pixel_fp"]),
        "fn": int(metrics["pixel_fn"]),
        "tn": int(metrics["pixel_tn"]),
        "pixel_precision": float(metrics["pixel_precision"]),
        "pixel_recall": float(metrics["pixel_recall"]),
        "pixel_f1": float(metrics["pixel_f1"]),
        "pixel_iou": float(metrics["pixel_iou"]),
        "gt_objects_count": len(gt_objects),
        "pred_objects_count": len(pred_objects),
        "matched_objects_count": int(object_metrics.get("object_tp") or 0),
        "object_precision": float(object_metrics.get("object_precision") or 0.0),
        "object_recall": float(object_metrics.get("object_recall") or 0.0),
        "object_f1": float(object_metrics.get("object_f1") or 0.0),
        "notes": "; ".join(notes or []),
    }


def build_sample_payload(
    *,
    metadata: dict[str, Any],
    image: torch.Tensor | np.ndarray,
    gt_mask: torch.Tensor | np.ndarray,
    pred_prob: torch.Tensor | np.ndarray,
    pred_mask: torch.Tensor | np.ndarray,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    image_np = _to_numpy(image)
    gt_np = _squeeze_mask(_to_numpy(gt_mask))
    prob_np = _squeeze_mask(_to_numpy(pred_prob))
    pred_np = _squeeze_mask(_to_numpy(pred_mask))
    sample_id = str(metadata.get("sample_id") or metadata.get("tile_id") or len(str(metadata)))
    scene_id = str(metadata.get("scene_id") or metadata.get("scene") or "unknown")
    tile_id = str(metadata.get("tile_id") or sample_id)
    metrics = sample_metrics_from_arrays(
        sample_id=sample_id,
        scene_id=scene_id,
        tile_id=tile_id,
        source_image_path=metadata.get("source_image_path") or metadata.get("s3_key"),
        gt_mask_path=metadata.get("gt_mask_path"),
        image=image_np,
        gt_mask=gt_np,
        pred_prob=prob_np,
        pred_mask=pred_np,
        notes=notes,
    )
    return {
        "metadata": {**metadata, **metrics},
        "image": image_np,
        "gt_mask": gt_np,
        "pred_prob": prob_np,
        "pred_mask": pred_np,
        "objects_gt": _objects_from_mask(gt_np),
        "objects_pred": _objects_from_mask(pred_np),
    }


def _write_sample_debug(samples_dir: Path, payload: dict[str, Any]) -> None:
    metadata = payload.get("metadata") or {}
    sample_id = _safe_name(str(metadata.get("sample_id") or "sample"))
    sample_dir = samples_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(payload.get("image"))
    gt_mask = np.asarray(payload.get("gt_mask"))
    pred_prob = np.asarray(payload.get("pred_prob"), dtype=np.float32)
    pred_mask = np.asarray(payload.get("pred_mask"))
    _write_json(sample_dir / "metadata.json", metadata)
    np.savez_compressed(sample_dir / "pred_prob.npz", pred_prob=pred_prob)
    _write_png(sample_dir / "image.png", _image_preview(image))
    _write_png(sample_dir / "gt_mask.png", _mask_preview(gt_mask))
    _write_png(sample_dir / "pred_prob.png", _prob_preview(pred_prob))
    _write_png(sample_dir / "pred_mask.png", _mask_preview(pred_mask))
    _write_png(sample_dir / "tp_mask.png", _mask_preview((pred_mask > 0.5) & (gt_mask > 0.5)))
    _write_png(sample_dir / "fp_mask.png", _mask_preview((pred_mask > 0.5) & ~(gt_mask > 0.5)))
    _write_png(sample_dir / "fn_mask.png", _mask_preview(~(pred_mask > 0.5) & (gt_mask > 0.5)))
    _write_png(sample_dir / "confusion_map.png", _confusion_preview(pred_mask, gt_mask))
    _write_png(sample_dir / "overlay_gt_pred.png", _overlay_preview(image, pred_mask, gt_mask))
    objects_gt = payload.get("objects_gt") or []
    objects_pred = payload.get("objects_pred") or []
    _write_geojson(sample_dir / "objects_gt.geojson", objects_gt)
    _write_geojson(sample_dir / "objects_pred.geojson", objects_pred)
    matches = _object_metrics(objects_pred, objects_gt)
    _write_json(sample_dir / "object_matches.json", matches)
    _write_iou_matrix(sample_dir / "object_iou_matrix.csv", objects_pred, objects_gt)


def _object_metrics(pred_objects: list[Any], gt_objects: list[Any]) -> dict[str, Any]:
    try:
        from ..object_metrics import compute_object_f1

        return compute_object_f1(pred_objects, gt_objects)
    except Exception as exc:
        return {
            "object_tp": 0,
            "object_fp": len(pred_objects),
            "object_fn": len(gt_objects),
            "object_precision": 0.0,
            "object_recall": 0.0,
            "object_f1": 0.0,
            "warning": f"{type(exc).__name__}: {exc}",
        }


def _objects_from_mask(mask: np.ndarray) -> list[Any]:
    arr = _squeeze_mask(np.asarray(mask))
    try:
        from rasterio.features import shapes
        from shapely.geometry import shape

        return [shape(geom) for geom, value in shapes((arr > 0.5).astype("uint8")) if int(value) == 1]
    except Exception:
        return []


def _write_iou_matrix(path: Path, pred_objects: list[Any], gt_objects: list[Any]) -> None:
    try:
        from ..object_metrics import compute_pairwise_iou

        matrix = compute_pairwise_iou(pred_objects, gt_objects)
    except Exception:
        matrix = []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["pred_index", *[f"gt_{idx}" for idx in range(len(gt_objects))]])
        for pred_index, row in enumerate(matrix):
            writer.writerow([pred_index, *row])


def _write_geojson(path: Path, geoms: list[Any]) -> None:
    features = []
    for idx, geom in enumerate(geoms):
        try:
            features.append({"type": "Feature", "properties": {"index": idx}, "geometry": geom.__geo_interface__})
        except Exception:
            continue
    _write_json(path, {"type": "FeatureCollection", "features": features})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key)) for key in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_png(path: Path, arr: np.ndarray) -> None:
    try:
        from PIL import Image

        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr.astype("uint8")).save(path)
    except Exception:
        return


def _image_preview(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4}:
        arr = np.moveaxis(arr[:3], 0, -1)
    elif arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = arr[..., :3]
    arr_min = float(np.nanmin(arr)) if arr.size else 0.0
    arr_max = float(np.nanmax(arr)) if arr.size else 1.0
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return np.clip(arr * 255.0, 0, 255).astype("uint8")


def _mask_preview(mask: np.ndarray) -> np.ndarray:
    arr = _squeeze_mask(np.asarray(mask))
    return ((arr > 0.5).astype("uint8") * 255)


def _prob_preview(prob: np.ndarray) -> np.ndarray:
    arr = _squeeze_mask(np.asarray(prob, dtype=np.float32))
    return np.clip(arr * 255.0, 0, 255).astype("uint8")


def _confusion_preview(pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    pred = _squeeze_mask(np.asarray(pred_mask)) > 0.5
    gt = _squeeze_mask(np.asarray(gt_mask)) > 0.5
    out = np.zeros((*gt.shape, 3), dtype="uint8")
    out[pred & gt] = [0, 180, 0]
    out[pred & ~gt] = [220, 50, 50]
    out[~pred & gt] = [40, 90, 220]
    return out


def _overlay_preview(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    base = _image_preview(image).astype(np.float32)
    pred = _squeeze_mask(np.asarray(pred_mask)) > 0.5
    gt = _squeeze_mask(np.asarray(gt_mask)) > 0.5
    overlay = base.copy()
    overlay[gt] = overlay[gt] * 0.45 + np.array([0, 200, 0], dtype=np.float32) * 0.55
    overlay[pred] = overlay[pred] * 0.45 + np.array([220, 50, 50], dtype=np.float32) * 0.55
    overlay[pred & gt] = overlay[pred & gt] * 0.35 + np.array([255, 220, 0], dtype=np.float32) * 0.65
    return np.clip(overlay, 0, 255).astype("uint8")


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_mask(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    while arr.ndim >= 3 and 1 in arr.shape[:-2]:
        axis = next(idx for idx, size in enumerate(arr.shape[:-2]) if size == 1)
        arr = np.squeeze(arr, axis=axis)
    return arr


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return value


def _safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return safe[:120] or "sample"


def _float_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def _int_metric(row: dict[str, Any], key: str) -> int | None:
    value = row.get(key)
    return None if value is None else int(value)
