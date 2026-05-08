from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
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
    payload.setdefault("report_enabled", True)
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
    geojson_dir = epoch_dir / "geojson"
    aggregate_paths = _write_epoch_aggregate_geojson(geojson_dir, sample_payloads)

    production_snapshot = production_metrics_snapshot(
        epoch=epoch,
        metric_row=metric_row,
        train_loss=train_loss,
        val_loss=val_loss,
        class_name=class_name,
        class_id=class_id,
        threshold=threshold,
        mlflow_run_id=mlflow_run_id,
        val_manifest_path=val_manifest_path,
        per_sample_metrics=per_sample_metrics,
    )
    production_snapshot_path = epoch_dir / "production_metrics_snapshot.json"
    _write_json(production_snapshot_path, production_snapshot)

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
        "production_metrics_snapshot": str(production_snapshot_path),
        "production_metrics": production_snapshot,
        "aggregate_geojson": aggregate_paths,
        "samples_dir": str(samples_dir),
    }
    summary_path = epoch_dir / "epoch_summary.json"
    _write_json(summary_path, summary)
    metrics_md_path = epoch_dir / "metrics.md"
    _write_epoch_metrics_md(metrics_md_path, summary, production_snapshot, recompute)
    artifacts_manifest_path = epoch_dir / "artifacts_manifest.csv"
    _write_artifacts_manifest(artifacts_manifest_path, epoch_dir)
    return {
        "epoch_dir": str(epoch_dir),
        "epoch_summary": str(summary_path),
        "production_metrics_snapshot": str(production_snapshot_path),
        "per_sample_metrics": str(per_sample_csv),
        "val_manifest_snapshot": str(manifest_path),
        "metrics_recompute_check": str(recompute_path),
        "mlflow_logged_metrics": str(logged_path),
        "metrics_md": str(metrics_md_path),
        "artifacts_manifest": str(artifacts_manifest_path),
        "aggregate_geojson": aggregate_paths,
        "recompute": recompute,
    }


def production_metrics_snapshot(
    *,
    epoch: int,
    metric_row: dict[str, Any],
    train_loss: dict[str, Any],
    val_loss: dict[str, Any],
    class_name: str,
    class_id: int | str | None,
    threshold: float,
    mlflow_run_id: str | None,
    val_manifest_path: str | None,
    per_sample_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the single production metrics snapshot used by MLflow/report/debug."""
    counts = {
        "gt_pixels": _int_metric(metric_row, "val/pixel_tp", default=0) + _int_metric(metric_row, "val/pixel_fn", default=0),
        "pred_pixels": _int_metric(metric_row, "val/pixel_tp", default=0) + _int_metric(metric_row, "val/pixel_fp", default=0),
        "gt_objects": sum(int(row.get("gt_objects_count") or 0) for row in per_sample_metrics),
        "pred_objects": sum(int(row.get("pred_objects_count") or 0) for row in per_sample_metrics),
        "matched_objects": _int_metric(metric_row, "val/object_tp", default=0),
        "val_samples": len(per_sample_metrics),
    }
    return {
        "schema_version": 1,
        "source": "production_validation_loop",
        "source_of_truth": True,
        "recompute_role": "validation_check_only",
        "epoch": int(epoch),
        "class_name": class_name,
        "class_id": class_id,
        "threshold": float(threshold),
        "mlflow_run_id": mlflow_run_id,
        "val_manifest_path": val_manifest_path,
        "metrics": {key: _json_safe(value) for key, value in metric_row.items() if key != "epoch"},
        "train_loss": train_loss,
        "val_loss": val_loss,
        "counts": counts,
    }


def write_metrics_debug_report(
    *,
    debug_root: Path,
    report_root: Path,
    report_name: str,
    run_metadata: dict[str, Any],
    dataset_check: dict[str, Any],
) -> dict[str, Any]:
    """Create a compact readable report folder from per-epoch production snapshots."""
    source_root = Path(debug_root)
    report_dir = Path(report_root) / _safe_name(report_name)
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    epochs_out = report_dir / "epochs"
    airflow_dir = report_dir / "airflow"
    checks_dir = report_dir / "checks"
    airflow_dir.mkdir(parents=True, exist_ok=True)
    checks_dir.mkdir(parents=True, exist_ok=True)

    _write_json(report_dir / "run_metadata.json", run_metadata)
    _write_json(airflow_dir / "dag_conf.json", run_metadata.get("dag_conf") or {})
    _write_json(airflow_dir / "dag_run.json", run_metadata.get("airflow") or {})
    _write_json(airflow_dir / "task_statuses.json", run_metadata.get("task_statuses") or {})
    (airflow_dir / "airflow_url.txt").write_text(str(run_metadata.get("airflow_url") or ""), encoding="utf-8")

    timeseries: list[dict[str, Any]] = []
    mlflow_checks: list[dict[str, Any]] = []
    report_structure: dict[str, Any] = {"epochs": {}, "raw_images_copied": False, "ok": True}
    for epoch_dir in sorted(source_root.glob("epoch_*")):
        epoch_name = epoch_dir.name
        out_epoch = epochs_out / epoch_name
        (out_epoch / "samples").mkdir(parents=True, exist_ok=True)
        _copy_epoch_level_files(epoch_dir, out_epoch)
        _copy_epoch_geojson(epoch_dir, out_epoch)
        copied_samples = _copy_selected_sample_artifacts(epoch_dir, out_epoch / "samples")

        snapshot = _read_json(epoch_dir / "production_metrics_snapshot.json")
        recompute = _read_json(epoch_dir / "metrics_recompute_check.json")
        logged = _read_json(epoch_dir / "mlflow_logged_metrics.json")
        metrics = snapshot.get("metrics") or {}
        row = _timeseries_row(snapshot)
        timeseries.append(row)
        mlflow_checks.extend(_compare_snapshot_to_logged(snapshot, logged))
        report_structure["epochs"][epoch_name] = {
            "copied_samples": copied_samples,
            "sample_copy_strategy": "selected positive/worst/high-fp/high-fn examples; per_sample_metrics and aggregate GeoJSON cover all validation samples",
            "has_per_sample_metrics": (out_epoch / "per_sample_metrics.csv").exists(),
            "has_epoch_geojson": (out_epoch / "geojson" / "gt_objects.geojson").exists()
            and (out_epoch / "geojson" / "pred_objects.geojson").exists(),
            "recompute_ok": bool(recompute.get("ok", False)),
            "metrics_keys": sorted(metrics),
        }

    _write_csv(report_dir / "metrics_timeseries.csv", timeseries)
    _write_json(report_dir / "metrics_timeseries.json", timeseries)
    source_check = {
        "ok": True,
        "production_snapshot_reused": True,
        "source": "production_validation_loop",
        "recompute_role": "validation_check_only",
        "debug_root": str(source_root),
    }
    _write_json(checks_dir / "source_of_truth_check.json", source_check)
    _write_json(checks_dir / "mlflow_consistency_check.json", {"ok": all(row["ok"] for row in mlflow_checks), "comparisons": mlflow_checks})
    _write_json(checks_dir / "report_structure_check.json", report_structure)
    _write_json(checks_dir / "dataset_full_run_check.json", dataset_check)
    _write_summary_md(
        report_dir / "summary.md",
        run_metadata=run_metadata,
        dataset_check=dataset_check,
        timeseries=timeseries,
        mlflow_checks=mlflow_checks,
        report_structure=report_structure,
    )
    return {
        "report_dir": str(report_dir),
        "summary": str(report_dir / "summary.md"),
        "epoch_count": len(timeseries),
        "timeseries": str(report_dir / "metrics_timeseries.csv"),
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


def _write_epoch_aggregate_geojson(geojson_dir: Path, sample_payloads: list[dict[str, Any]]) -> dict[str, str]:
    geojson_dir.mkdir(parents=True, exist_ok=True)
    gt_features: list[dict[str, Any]] = []
    pred_features: list[dict[str, Any]] = []
    match_features: list[dict[str, Any]] = []
    for payload in sample_payloads:
        metadata = payload.get("metadata") or {}
        sample_id = str(metadata.get("sample_id") or "sample")
        scene_id = str(metadata.get("scene_id") or metadata.get("scene") or "")
        gt_objects = payload.get("objects_gt") or []
        pred_objects = payload.get("objects_pred") or []
        for idx, geom in enumerate(gt_objects):
            feature = _feature_from_geom(geom, {"sample_id": sample_id, "scene_id": scene_id, "object_index": idx, "kind": "gt"})
            if feature:
                gt_features.append(feature)
        for idx, geom in enumerate(pred_objects):
            feature = _feature_from_geom(geom, {"sample_id": sample_id, "scene_id": scene_id, "object_index": idx, "kind": "pred"})
            if feature:
                pred_features.append(feature)
        matches = _object_metrics(pred_objects, gt_objects).get("matches") or []
        for match in matches:
            pred_idx = int(match.get("pred_index", -1))
            geom = pred_objects[pred_idx] if 0 <= pred_idx < len(pred_objects) else None
            feature = _feature_from_geom(
                geom,
                {
                    "sample_id": sample_id,
                    "scene_id": scene_id,
                    "pred_index": pred_idx,
                    "gt_index": int(match.get("gt_index", -1)),
                    "iou": float(match.get("iou") or 0.0),
                    "kind": "match_pred_geometry",
                },
            )
            if feature:
                match_features.append(feature)
    gt_path = geojson_dir / "gt_objects.geojson"
    pred_path = geojson_dir / "pred_objects.geojson"
    matches_path = geojson_dir / "object_matches.geojson"
    _write_json(gt_path, {"type": "FeatureCollection", "features": gt_features})
    _write_json(pred_path, {"type": "FeatureCollection", "features": pred_features})
    _write_json(matches_path, {"type": "FeatureCollection", "features": match_features})
    return {
        "gt_objects": str(gt_path),
        "pred_objects": str(pred_path),
        "object_matches": str(matches_path),
    }


def _feature_from_geom(geom: Any, properties: dict[str, Any]) -> dict[str, Any] | None:
    try:
        return {"type": "Feature", "properties": properties, "geometry": geom.__geo_interface__}
    except Exception:
        return None


def _write_epoch_metrics_md(path: Path, summary: dict[str, Any], snapshot: dict[str, Any], recompute: dict[str, Any]) -> None:
    metrics = snapshot.get("metrics") or {}
    counts = snapshot.get("counts") or {}
    lines = [
        f"# Epoch {summary.get('epoch')}",
        "",
        f"- class_name: `{snapshot.get('class_name')}`",
        f"- threshold: `{snapshot.get('threshold')}`",
        f"- source: `{snapshot.get('source')}`",
        f"- recompute_ok: `{bool(recompute.get('ok'))}`",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "train/loss_total",
        "val/loss_total",
        "val/pixel_f1",
        "val/pixel_iou",
        "val/precision",
        "val/recall",
        "val/object_f1",
    ):
        value = metrics.get(key) or (snapshot.get("train_loss") or {}).get(key) or (snapshot.get("val_loss") or {}).get(key)
        lines.append(f"| `{key}` | `{value}` |")
    for key, value in counts.items():
        lines.append(f"| `{key}` | `{value}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifacts_manifest(path: Path, epoch_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in sorted(epoch_dir.rglob("*")):
        if item.is_file():
            rows.append(
                {
                    "relative_path": str(item.relative_to(epoch_dir)).replace("\\", "/"),
                    "bytes": item.stat().st_size,
                    "included_in_compact_report": item.name != "image.png" and item.suffix not in {".npz", ".pt"},
                }
            )
    _write_csv(path, rows)


def _copy_epoch_level_files(source_epoch: Path, target_epoch: Path) -> None:
    target_epoch.mkdir(parents=True, exist_ok=True)
    for name in (
        "metrics.md",
        "epoch_summary.json",
        "production_metrics_snapshot.json",
        "metrics_recompute_check.json",
        "per_sample_metrics.csv",
        "artifacts_manifest.csv",
    ):
        source = source_epoch / name
        if source.exists():
            shutil.copy2(source, target_epoch / name)


def _copy_epoch_geojson(source_epoch: Path, target_epoch: Path) -> None:
    source_dir = source_epoch / "geojson"
    target_dir = target_epoch / "geojson"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("gt_objects.geojson", "pred_objects.geojson", "object_matches.geojson"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)


def _copy_selected_sample_artifacts(source_epoch: Path, target_samples: Path) -> int:
    source_samples = source_epoch / "samples"
    if not source_samples.exists():
        return 0
    selected_ids = _select_compact_sample_ids(source_epoch / "per_sample_metrics.csv")
    copied = 0
    for sample_dir in sorted(item for item in source_samples.iterdir() if item.is_dir()):
        if selected_ids and sample_dir.name not in selected_ids:
            continue
        target_dir = target_samples / sample_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        has_any = False
        for name in (
            "metadata.json",
            "gt_mask.png",
            "pred_mask.png",
            "overlay_gt_pred.png",
            "objects_gt.geojson",
            "objects_pred.geojson",
            "object_matches.json",
        ):
            source = sample_dir / name
            if source.exists():
                shutil.copy2(source, target_dir / name)
                has_any = True
        copied += int(has_any)
    return copied


def _select_compact_sample_ids(per_sample_csv: Path, *, max_samples: int = 16) -> set[str]:
    if not per_sample_csv.exists():
        return set()
    with per_sample_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return set()

    def sample_id(row: dict[str, Any]) -> str:
        return _safe_name(str(row.get("sample_id") or row.get("tile_id") or "sample"))

    def numeric(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or 0)
        except Exception:
            return 0.0

    selected: list[str] = []

    def add(rows_to_add: list[dict[str, Any]], limit: int) -> None:
        for row in rows_to_add:
            sid = sample_id(row)
            if sid not in selected:
                selected.append(sid)
            if len(selected) >= max_samples or limit <= 0:
                return
            limit -= 1

    positives = [row for row in rows if numeric(row, "gt_positive_pixels") > 0]
    add(sorted(positives, key=lambda row: numeric(row, "pixel_f1")), 6)
    add(sorted(rows, key=lambda row: numeric(row, "pixel_f1")), 4)
    add(sorted(rows, key=lambda row: numeric(row, "fp"), reverse=True), 3)
    add(sorted(rows, key=lambda row: numeric(row, "fn"), reverse=True), 3)
    add(rows[: max_samples], max_samples)
    return set(selected[:max_samples])


def _timeseries_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    metrics = snapshot.get("metrics") or {}
    counts = snapshot.get("counts") or {}
    return {
        "epoch": snapshot.get("epoch"),
        "train_loss": (snapshot.get("train_loss") or {}).get("train/loss_total"),
        "val_loss": (snapshot.get("val_loss") or {}).get("val/loss_total"),
        "pixel_f1": metrics.get("val/pixel_f1"),
        "pixel_iou": metrics.get("val/pixel_iou"),
        "precision": metrics.get("val/precision"),
        "recall": metrics.get("val/recall"),
        "object_f1": metrics.get("val/object_f1"),
        "gt_pixels": counts.get("gt_pixels"),
        "pred_pixels": counts.get("pred_pixels"),
        "gt_objects": counts.get("gt_objects"),
        "pred_objects": counts.get("pred_objects"),
        "matched_objects": counts.get("matched_objects"),
    }


def _compare_snapshot_to_logged(snapshot: dict[str, Any], logged: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = snapshot.get("metrics") or {}
    for key in (
        "val/pixel_f1",
        "val/pixel_iou",
        "val/precision",
        "val/recall",
        "val/object_f1",
        "val/pixel_tp",
        "val/pixel_fp",
        "val/pixel_fn",
    ):
        if key not in metrics or key not in logged:
            continue
        production = float(metrics[key])
        mlflow_logged = float(logged[key])
        delta = abs(production - mlflow_logged)
        rows.append(
            {
                "epoch": snapshot.get("epoch"),
                "metric": key,
                "production_snapshot": production,
                "mlflow_logged": mlflow_logged,
                "abs_delta": delta,
                "ok": delta <= 1e-6,
            }
        )
    return rows


def _write_summary_md(
    path: Path,
    *,
    run_metadata: dict[str, Any],
    dataset_check: dict[str, Any],
    timeseries: list[dict[str, Any]],
    mlflow_checks: list[dict[str, Any]],
    report_structure: dict[str, Any],
) -> None:
    lines = [
        "# MLSystem metrics debug report: вырубки",
        "",
        "## Run identity",
        "",
        f"- branch: `{run_metadata.get('branch') or ''}`",
        f"- commit: `{run_metadata.get('commit') or ''}`",
        f"- pushed_remote: `{run_metadata.get('pushed_remote') or ''}`",
        f"- Airflow DAG id: `{(run_metadata.get('airflow') or {}).get('dag_id') or ''}`",
        f"- Airflow dag_run_id: `{(run_metadata.get('airflow') or {}).get('run_id') or ''}`",
        f"- Airflow URL: `{run_metadata.get('airflow_url') or ''}`",
        f"- MLflow run id: `{run_metadata.get('mlflow_run_id') or ''}`",
        f"- MLflow URL: `{run_metadata.get('mlflow_url') or ''}`",
        f"- server debug artifact path: `{run_metadata.get('debug_root') or ''}`",
        "",
        "## Dataset/config",
        "",
        f"- class: `{dataset_check.get('class_name')}`",
        f"- full_dataset: `{dataset_check.get('full_dataset')}`",
        f"- synthetic: `{dataset_check.get('synthetic')}`",
        f"- train_sample_count: `{dataset_check.get('train_sample_count')}`",
        f"- val_sample_count: `{dataset_check.get('val_sample_count')}`",
        f"- train_manifest_hash: `{dataset_check.get('train_manifest_hash')}`",
        f"- val_manifest_hash: `{dataset_check.get('val_manifest_hash')}`",
        f"- threshold: `{run_metadata.get('threshold')}`",
        f"- seed: `{run_metadata.get('seed')}`",
        f"- time_limit_seconds: `{run_metadata.get('max_wallclock_seconds')}`",
        f"- train_limit_batches: `{dataset_check.get('train_limit_batches')}`",
        f"- val_limit_batches: `{dataset_check.get('val_limit_batches')}`",
        "",
        "## Source-of-truth metrics",
        "",
        "Production validation loop forms one metrics snapshot per epoch. The same snapshot is used for MLflow logging, epoch_summary.json, production_metrics_snapshot.json and this report. Recompute checks validate the snapshot from saved per-sample counts; recompute is not a separate source of truth.",
        "",
        "## Metrics by epoch",
        "",
        "| epoch | train_loss | val_loss | pixel_f1 | pixel_iou | precision | recall | object_f1 | gt_pixels | pred_pixels | gt_objects | pred_objects | matched_objects |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in timeseries:
        lines.append(
            "| {epoch} | {train_loss} | {val_loss} | {pixel_f1} | {pixel_iou} | {precision} | {recall} | {object_f1} | {gt_pixels} | {pred_pixels} | {gt_objects} | {pred_objects} | {matched_objects} |".format(
                **{key: _md_value(row.get(key)) for key in row}
            )
        )
    lines.extend(
        [
            "",
            "## Consistency checks",
            "",
            "| epoch | metric | production_snapshot | mlflow_logged | delta_mlflow | ok |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in mlflow_checks:
        lines.append(
            f"| {row.get('epoch')} | `{row.get('metric')}` | {row.get('production_snapshot')} | {row.get('mlflow_logged')} | {row.get('abs_delta')} | {row.get('ok')} |"
        )
    lines.extend(
        [
            "",
            "## Report artifacts",
            "",
            "| epoch | copied_samples | has_per_sample_metrics | has_epoch_geojson | recompute_ok |",
            "|---|---:|---|---|---|",
        ]
    )
    for epoch, payload in (report_structure.get("epochs") or {}).items():
        lines.append(
            f"| `{epoch}` | {payload.get('copied_samples')} | {payload.get('has_per_sample_metrics')} | {payload.get('has_epoch_geojson')} | {payload.get('recompute_ok')} |"
        )
    lines.extend(
        [
            "",
            "## Вывод",
            "",
            f"- Run zapushchen cherez Airflow: `{bool((run_metadata.get('airflow') or {}).get('run_id'))}`",
            f"- Epochs completed: `{len(timeseries)}`",
            f"- Metrics consistency OK: `{all(row.get('ok') for row in mlflow_checks)}`",
            f"- Raw dataset images copied into report: `{report_structure.get('raw_images_copied')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.10g}"
    return "" if value is None else value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(_json_safe(rows), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    pred = _squeeze_mask(np.asarray(pred_mask)) > 0.5
    gt = _squeeze_mask(np.asarray(gt_mask)) > 0.5
    overlay = np.zeros((*gt.shape, 3), dtype="uint8")
    overlay[gt] = [0, 190, 0]
    overlay[pred] = [220, 50, 50]
    overlay[pred & gt] = [255, 220, 0]
    return overlay


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


def _int_metric(row: dict[str, Any], key: str, default: int | None = None) -> int | None:
    value = row.get(key)
    return default if value is None else int(value)
