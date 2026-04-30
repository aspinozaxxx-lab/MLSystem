from __future__ import annotations

import csv
import json
import math
import random
import re
import time
import difflib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from rasterio.features import rasterize
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform

from .io_utils import write_json
from .data import scene_matching as scene_matching_mod
from .job_schema import JobSpec
from .mlflow_adapter import MLflowJobRun, trace_stage
from .models.factory import build_model as build_model_mod
from .models.factory import set_batchnorm_eval as set_batchnorm_eval_mod
from .object_metrics import compute_object_f1
from .pipeline.pseudolabel_pipeline import run_pseudolabel_pipeline
from .pipeline_config import PipelineConfig
from .preprocessing.normalization import normalize_image as normalize_image_mod
from .reporting.prediction_examples import write_prediction_examples_report as write_prediction_examples_report_mod
from .storage import s3 as s3_storage
from .tiling import windows as tiling_windows
from .training.losses import segmentation_loss


SceneMatch = scene_matching_mod.SceneMatch


def _s3_parts(uri: str) -> tuple[str, str]:
    return s3_storage.s3_parts(uri)


def _norm_scene_name(value: str) -> str:
    name = PurePosixPath(value.strip()).name.lower()
    name = re.sub(r"\.(tif|tiff)$", "", name)
    name = name.replace("_cog", "")
    return re.sub(r"[^a-z0-9а-я]+", "", name)


def _mc_credentials(config: PipelineConfig) -> tuple[str, str]:
    return s3_storage.credentials_from_mc(config)


def _s3_client(config: PipelineConfig):
    return s3_storage.s3_client(config)


def _aws_session(config: PipelineConfig) -> Any:
    return s3_storage.aws_session(config)


def _list_s3_objects(config: PipelineConfig, uri: str, suffixes: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    return s3_storage.list_s3_objects(config, uri, suffixes=suffixes)


def _read_s3_text(config: PipelineConfig, uri: str) -> str:
    return s3_storage.read_s3_text(config, uri)


def _read_s3_json(config: PipelineConfig, uri: str) -> Any:
    return s3_storage.read_s3_json(config, uri)


def _find_layout_files(config: PipelineConfig, layout_uri: str, scenes_file: str, annotation_file: str) -> tuple[str, str]:
    return s3_storage.find_layout_files(config, layout_uri, scenes_file, annotation_file)


def _norm_scene_name(value: str) -> str:
    return scene_matching_mod.norm_scene_name(value)


def _scene_signature(normalized: str) -> str | None:
    return scene_matching_mod.scene_signature(normalized)


def _scene_score(needle: str, candidate: str) -> tuple[float, str]:
    return scene_matching_mod.scene_score(needle, candidate)


def build_scene_matching_report(
    entries: list[str],
    images: list[dict[str, Any]],
    *,
    accept_threshold: float = 0.92,
    ambiguous_margin: float = 0.015,
) -> dict[str, Any]:
    return scene_matching_mod.build_scene_matching_report(
        entries,
        images,
        accept_threshold=accept_threshold,
        ambiguous_margin=ambiguous_margin,
    )
    normalized_images = [(item, _norm_scene_name(item["name"])) for item in images]
    matched: list[SceneMatch] = []
    ambiguous: list[dict[str, Any]] = []
    missing: list[str] = []
    rows: list[dict[str, Any]] = []

    for entry in entries:
        needle = _norm_scene_name(entry)
        scored: list[dict[str, Any]] = []
        for image, image_norm in normalized_images:
            score, score_reason = _scene_score(needle, image_norm)
            scored.append({"image": image, "normalized_image": image_norm, "score": round(float(score), 6), "reason": score_reason})
        scored.sort(key=lambda item: item["score"], reverse=True)
        best = scored[0] if scored else None
        second = scored[1] if len(scored) > 1 else None
        decision = "missing"
        reason = "no_reliable_candidate"
        if best and best["score"] >= accept_threshold:
            if second and second["score"] >= accept_threshold and (best["score"] - second["score"]) <= ambiguous_margin:
                decision = "ambiguous"
                reason = "multiple_close_candidates"
                ambiguous.append(
                    {
                        "entry": entry,
                        "normalized_scene_line": needle,
                        "candidates": [
                            {
                                "name": item["image"]["name"],
                                "key": item["image"]["key"],
                                "score": item["score"],
                                "reason": item["reason"],
                            }
                            for item in scored[:10]
                            if item["score"] >= accept_threshold
                        ],
                    }
                )
            else:
                decision = "matched"
                reason = str(best["reason"])
                image = best["image"]
                matched.append(SceneMatch(entry=entry, key=image["key"], name=image["name"], score=round(float(best["score"]), 4)))
        else:
            if best and best["score"] >= 0.85:
                decision = "likely_missing_file"
                reason = "best_candidate_below_accept_threshold"
            missing.append(entry)
        rows.append(
            {
                "original_scene_line": entry,
                "normalized_scene_line": needle,
                "scene_signature": _scene_signature(needle),
                "exact_match": bool(best and best["score"] == 1.0),
                "best_candidate_1": best["image"]["name"] if best else None,
                "best_candidate_1_key": best["image"]["key"] if best else None,
                "best_candidate_1_normalized": best["normalized_image"] if best else None,
                "best_candidate_1_score": best["score"] if best else None,
                "best_candidate_1_reason": best["reason"] if best else None,
                "best_candidate_2": second["image"]["name"] if second else None,
                "best_candidate_2_key": second["image"]["key"] if second else None,
                "best_candidate_2_score": second["score"] if second else None,
                "best_candidate_2_reason": second["reason"] if second else None,
                "decision": decision,
                "reason": reason,
            }
        )
    return {
        "schema_version": 1,
        "total_scenes_in_scenes_txt": len(entries),
        "total_images_available": len(images),
        "total_tif_images_available": len([item for item in images if item["name"].lower().endswith((".tif", ".tiff"))]),
        "matched_count": len(matched),
        "missing_count": len(missing),
        "ambiguous_count": len(ambiguous),
        "accept_threshold": accept_threshold,
        "ambiguous_margin": ambiguous_margin,
        "matched": [match.__dict__ for match in matched],
        "missing": missing,
        "ambiguous": ambiguous,
        "rows": rows,
    }


def _match_scenes(entries: list[str], images: list[dict[str, Any]]) -> tuple[list[SceneMatch], list[dict[str, Any]], list[str]]:
    return scene_matching_mod.match_scenes(entries, images)


def _match_scenes_legacy(entries: list[str], images: list[dict[str, Any]]) -> tuple[list[SceneMatch], list[dict[str, Any]], list[str]]:
    normalized_images = [(item, _norm_scene_name(item["name"])) for item in images]
    matched: list[SceneMatch] = []
    ambiguous: list[dict[str, Any]] = []
    missing: list[str] = []

    for entry in entries:
        needle = _norm_scene_name(entry)
        candidates: list[tuple[dict[str, Any], float]] = []
        for image, image_norm in normalized_images:
            if not needle:
                continue
            if needle == image_norm:
                candidates.append((image, 1.0))
            elif len(needle) >= 16 and (needle in image_norm or image_norm in needle):
                candidates.append((image, 0.98))
        if not candidates:
            # Very conservative fallback: allow small suffix/punctuation differences only.
            import difflib

            ratios = [
                (difflib.SequenceMatcher(None, needle, image_norm).ratio(), image)
                for image, image_norm in normalized_images
            ]
            ratios.sort(key=lambda item: item[0], reverse=True)
            if ratios and ratios[0][0] >= 0.92:
                candidates = [(ratios[0][1], ratios[0][0])]
        if len(candidates) == 1:
            image, score = candidates[0]
            matched.append(SceneMatch(entry=entry, key=image["key"], name=image["name"], score=round(float(score), 4)))
        elif len(candidates) > 1:
            ambiguous.append({"entry": entry, "candidates": [item[0]["name"] for item in candidates[:10]]})
        else:
            missing.append(entry)
    return matched, ambiguous, missing


def _load_shapes(config: PipelineConfig, annotation_uri: str) -> list[Any]:
    payload = _read_s3_json(config, annotation_uri)
    features = payload.get("features") or []
    return [shape(feature["geometry"]) for feature in features if feature.get("geometry")]


class TinyUNet(torch.nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base: int = 8) -> None:
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base, base, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.down = torch.nn.MaxPool2d(2)
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(base, base * 2, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base * 2, base * 2, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.up = torch.nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec = torch.nn.Sequential(
            torch.nn.Conv2d(base * 2, base, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc1(x)
        x = self.enc2(self.down(skip))
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec(torch.cat([x, skip], dim=1))


def _build_model(model_name: str, in_channels: int, out_channels: int, base_channels: int) -> torch.nn.Module:
    normalized = model_name.lower()
    if normalized in {"tiny", "tiny_unet", "tiny_unet_4ch"}:
        return TinyUNet(in_channels=in_channels, out_channels=out_channels, base=base_channels)
    encoders = {
        "unet_resnet18": "resnet18",
        "unet_resnet34": "resnet34",
        "unet_resnet50": "resnet50",
    }
    if normalized in encoders:
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name=encoders[normalized],
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )
    segformer_encoders = {
        "segformer_b0": "mit_b0",
        "segformer_b1": "mit_b1",
    }
    if normalized in segformer_encoders:
        import segmentation_models_pytorch as smp

        return smp.Segformer(
            encoder_name=segformer_encoders[normalized],
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )
    deeplab_encoders = {
        "deeplabv3plus_resnet34": "resnet34",
        "deeplabv3plus_resnet50": "resnet50",
        "deeplab_r34": "resnet34",
        "deeplab_r50": "resnet50",
    }
    if normalized in deeplab_encoders:
        import segmentation_models_pytorch as smp

        return smp.DeepLabV3Plus(
            encoder_name=deeplab_encoders[normalized],
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )
    supported = ", ".join(
        [
            "tiny_unet_4ch",
            "unet_resnet18",
            "unet_resnet34",
            "unet_resnet50",
            "segformer_b0",
            "segformer_b1",
            "deeplabv3plus_resnet34",
            "deeplabv3plus_resnet50",
        ]
    )
    raise ValueError(f"Unsupported model_name={model_name}. Supported: {supported}")


def _set_batchnorm_eval(model: torch.nn.Module) -> None:
    set_batchnorm_eval_mod(model)


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    return normalize_image_mod(arr)


def _dice_iou_precision_recall(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()
    target = target.float()
    eps = 1e-7
    tp = float((pred * target).sum().item())
    fp = float((pred * (1 - target)).sum().item())
    fn = float(((1 - pred) * target).sum().item())
    intersection = tp
    union = float(((pred + target) > 0).float().sum().item())
    dice = (2 * intersection + eps) / (float(pred.sum().item() + target.sum().item()) + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "f1": f1}


def _loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return segmentation_loss(logits, target)


def _sample_windows(
    ds: Any,
    shapes: list[Any],
    patch_size: int,
    max_tiles: int,
    empty_share: float,
    seed: int,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    width, height = ds.width, ds.height
    if width < patch_size or height < patch_size:
        return [(0, 0)]

    scene_bounds = box(*ds.bounds)
    positives: list[tuple[int, int]] = []
    for geom in shapes:
        if not geom.is_valid or not geom.intersects(scene_bounds):
            continue
        inter = geom.intersection(scene_bounds)
        if inter.is_empty:
            continue
        cx, cy = inter.centroid.x, inter.centroid.y
        row, col = ds.index(cx, cy)
        x = max(0, min(width - patch_size, int(col - patch_size / 2)))
        y = max(0, min(height - patch_size, int(row - patch_size / 2)))
        positives.append((x, y))

    rng.shuffle(positives)
    max_empty = max(1, int(max_tiles * empty_share))
    max_positive = max(1, max_tiles - max_empty)
    windows = positives[:max_positive]
    while len(windows) < max_tiles:
        windows.append((rng.randint(0, width - patch_size), rng.randint(0, height - patch_size)))
    return windows[:max_tiles]


def _read_samples(
    config: PipelineConfig,
    matches: list[SceneMatch],
    shapes: list[Any],
    input_bands: list[int],
    patch_size: int,
    max_tiles_per_scene: int,
    max_tiles_total: int,
    empty_share: float,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    import rasterio
    from rasterio.windows import Window

    samples: list[tuple[np.ndarray, np.ndarray]] = []
    report: list[dict[str, Any]] = []
    aws = _aws_session(config)
    with rasterio.Env(aws, AWS_HTTPS="NO", AWS_VIRTUAL_HOSTING="FALSE"):
        for scene_idx, match in enumerate(matches):
            if len(samples) >= max_tiles_total:
                break
            path = f"/vsis3/{config.storage.s3_bucket}/{match.key}"
            with rasterio.open(path) as ds:
                scene_bounds = box(*ds.bounds)
                positive_scene = any(
                    geom.is_valid and (not geom.is_empty) and geom.intersects(scene_bounds)
                    for geom in shapes
                )
                usable_bands = [band for band in input_bands if band <= ds.count]
                if len(usable_bands) != len(input_bands):
                    raise RuntimeError(f"{match.name} has {ds.count} bands, expected {input_bands}")
                windows = _sample_windows(
                    ds,
                    shapes,
                    patch_size,
                    max_tiles_per_scene,
                    empty_share,
                    seed + scene_idx,
                )
                scene_samples = 0
                positive_tiles = 0
                for x, y in windows:
                    if len(samples) >= max_tiles_total:
                        break
                    window = Window(x, y, patch_size, patch_size)
                    arr = ds.read(usable_bands, window=window, boundless=True, fill_value=0)
                    if arr.shape[-2:] != (patch_size, patch_size):
                        continue
                    mask = rasterize(
                        [(geom, 1) for geom in shapes if geom.intersects(box(*ds.window_bounds(window)))],
                        out_shape=(patch_size, patch_size),
                        transform=ds.window_transform(window),
                        fill=0,
                        dtype="uint8",
                    )
                    if np.count_nonzero(arr) == 0:
                        continue
                    samples.append((_normalize_image(arr), mask.astype("float32")[None, :, :]))
                    scene_samples += 1
                    positive_tiles += int(mask.sum() > 0)
                report.append(
                    {
                        "scene": match.name,
                        "s3_key": match.key,
                        "width": ds.width,
                        "height": ds.height,
                        "bands": ds.count,
                        "crs": str(ds.crs),
                        "positive_scene": bool(positive_scene),
                        "samples": scene_samples,
                        "positive_tiles": positive_tiles,
                        "negative_tiles": max(0, scene_samples - positive_tiles),
                    }
                )
    return samples, report


def _write_history(experiment_dir: Path, history: list[dict[str, float]]) -> list[Path]:
    json_path = experiment_dir / "history.json"
    csv_path = experiment_dir / "history.csv"
    write_json(json_path, history)
    if history:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    return [json_path, csv_path]


def _default_tile_limits(job: JobSpec) -> tuple[int, int, int]:
    use_all_scenes = bool(job.preprocess.get("use_all_matched_scenes"))
    default_train_tiles = 512 if use_all_scenes else 24
    default_val_tiles = 128 if use_all_scenes else 8
    default_tiles_per_scene = 32 if use_all_scenes else 4
    max_train_tiles = int(job.train.get("max_train_tiles") or job.preprocess.get("max_train_tiles") or default_train_tiles)
    max_val_tiles = int(job.train.get("max_val_tiles") or job.preprocess.get("max_val_tiles") or default_val_tiles)
    max_tiles_per_scene = int(
        job.train.get("max_tiles_per_scene")
        or job.preprocess.get("max_tiles_per_scene")
        or default_tiles_per_scene
    )
    return max_train_tiles, max_val_tiles, max_tiles_per_scene


def _auto_batch_size(model_name: str, patch_size: int, device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    name = model_name.lower()
    if patch_size <= 512:
        return 16
    if patch_size <= 768:
        return 8 if "deeplab" in name else 12
    return 4 if "segformer" in name else 6


def _scene_file_names(matches: list[SceneMatch]) -> list[str]:
    return [PurePosixPath(match.name or match.key).name for match in matches]


def _write_scene_list(path: Path, matches: list[SceneMatch]) -> Path:
    path.write_text("\n".join(_scene_file_names(matches)) + ("\n" if matches else ""), encoding="utf-8")
    return path


def _write_object_metrics_artifacts(
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
    json_path = experiment_dir / "object_metrics.json"
    write_json(json_path, metrics_payload)
    csv_path = experiment_dir / "object_matches.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["pred_index", "gt_index", "iou", "pred_area", "gt_area", "intersection_area", "union_area"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["matches"]:
            writer.writerow({key: row.get(key) for key in fieldnames})
    metric_prefix = f"{prefix}/object"
    flat_metrics = {
        f"{metric_prefix}_f1": metrics["object_f1"],
        f"{metric_prefix}_precision": metrics["object_precision"],
        f"{metric_prefix}_recall": metrics["object_recall"],
        f"{metric_prefix}_tp": float(metrics["object_tp"]),
        f"{metric_prefix}_fp": float(metrics["object_fp"]),
        f"{metric_prefix}_fn": float(metrics["object_fn"]),
    }
    if prefix == "val":
        flat_metrics.update(
            {
                "best_val_object_f1": metrics["object_f1"],
                "best_val_object_precision": metrics["object_precision"],
                "best_val_object_recall": metrics["object_recall"],
            }
        )
    return flat_metrics, [json_path, csv_path]


def _write_prediction_examples_report(experiment_dir: Path, job_id: str, preview_paths: list[Path], limit: int = 30) -> list[Path]:
    return write_prediction_examples_report_mod(experiment_dir, job_id, preview_paths, limit=limit)
    rows = []
    for idx, path in enumerate(preview_paths[:limit]):
        rows.append(
            {
                "index": idx,
                "scene": path.stem,
                "nrg_preview": path.name,
                "gt_overlay": None,
                "probability": path.name,
                "prediction_overlay": path.name,
                "tp_fp_fn_overlay": None,
            }
        )
    html_path = experiment_dir / "prediction_examples.html"
    table_path = experiment_dir / "prediction_examples_table.json"
    html_rows = "\n".join(
        "<tr>"
        f"<td>{row['index']}</td><td>{row['scene']}</td>"
        f"<td><img src='{row['nrg_preview']}' width='220'></td>"
        f"<td><img src='{row['probability']}' width='220'></td>"
        f"<td>{row['tp_fp_fn_overlay'] or 'not available yet'}</td>"
        "</tr>"
        for row in rows
    )
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'><title>Prediction examples</title>",
                "<style>body{font-family:Arial,sans-serif} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px;vertical-align:top}</style>",
                "</head><body>",
                f"<h1>Prediction examples: {job_id}</h1>",
                "<p>Small preview table for MLflow artifacts. Full TP/FP/FN overlays are populated when tile-level evaluation artifacts are available.</p>",
                "<table><thead><tr><th>#</th><th>scene</th><th>NRG/RGB preview</th><th>probability/prediction</th><th>TP/FP/FN overlay</th></tr></thead><tbody>",
                html_rows,
                "</tbody></table></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    write_json(table_path, {"schema_version": 1, "job_id": job_id, "rows": rows})
    return [html_path, table_path]


def _write_pseudolabel_outputs(
    config: PipelineConfig,
    job: JobSpec,
    experiment_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    matches: list[SceneMatch],
    input_bands: list[int],
    patch_size: int,
    threshold: float,
    seed: int,
    gt_shapes: list[Any] | None = None,
    object_metrics_prefix: str = "val",
) -> tuple[dict[str, Any], list[Path]]:
    return run_pseudolabel_pipeline(
        config=config,
        job=job,
        experiment_dir=experiment_dir,
        model=model,
        device=device,
        matches=matches,
        input_bands=input_bands,
        patch_size=patch_size,
        threshold=threshold,
        seed=seed,
        gt_shapes=gt_shapes,
        object_metrics_prefix=object_metrics_prefix,
    )


def run_debug_pseudolabel(
    config: PipelineConfig,
    job: JobSpec,
    experiment_dir: Path,
    mlflow_run: MLflowJobRun,
    job_log: Path,
    log_fn: Any,
) -> dict[str, Any]:
    data = job.data or {}
    model_cfg = (job.params.get("model") if isinstance(job.params.get("model"), dict) else {}) or {}
    model_cfg = {**model_cfg, **(job.predict.get("model") or {})}
    images_uri = data.get("images_uri") or config.s3_paths.images
    layout_uri = data.get("layout_uri") or f"{config.s3_paths.layouts.rstrip('/')}/{job.class_name or 'deforest'}/"
    scenes_file = data.get("scenes_file") or "scenes.txt"
    annotation_file = data.get("annotation_file") or "auto"
    seed = int(job.predict.get("seed") or job.params.get("seed") or 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prepare_started = time.time()
    with trace_stage("match_scenes", {"job_id": job.job_id, "images_uri": images_uri, "layout_uri": layout_uri}):
        images = _list_s3_objects(config, images_uri, suffixes=(".tif", ".tiff"))
        annotation_uri, scenes_uri = _find_layout_files(config, layout_uri, scenes_file, annotation_file)
        entries = [
            line.strip()
            for line in _read_s3_text(config, scenes_uri).splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        matching_report = build_scene_matching_report(entries, images)
    matches = [SceneMatch(**item) for item in matching_report["matched"]]
    ambiguous = matching_report["ambiguous"]
    missing = matching_report["missing"]
    strict_matching = str(data.get("scene_name_matching") or "").lower() == "strict_after_fuzzy_validation"
    if strict_matching and (missing or ambiguous or len(matches) != len(entries)):
        raise RuntimeError(
            "Scene matching is not complete: "
            f"total={len(entries)} matched={len(matches)} missing={len(missing)} ambiguous={len(ambiguous)}"
        )
    if not matches:
        raise RuntimeError("No scenes from scenes.txt matched available images")

    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    total_matched_count = len(matches)
    max_debug_scenes = pseudolabel_cfg.get("max_debug_scenes", job.predict.get("max_debug_scenes"))
    if max_debug_scenes is not None:
        matches = matches[: max(1, int(max_debug_scenes))]
    scene_report = {
        **matching_report,
        "images_uri": images_uri,
        "layout_uri": layout_uri,
        "annotation_uri": annotation_uri,
        "scenes_uri": scenes_uri,
        "matched_count": total_matched_count,
        "selected_matched_count": len(matches),
        "matched": [match.__dict__ for match in matches],
    }
    scenes_report_path = experiment_dir / "scenes_match_report.json"
    matching_report_path = experiment_dir / "scene_matching_report.json"
    write_json(scenes_report_path, scene_report)
    write_json(matching_report_path, scene_report)
    train_scenes_path = _write_scene_list(experiment_dir / "train_scenes.txt", matches)
    mlflow_run.log_artifacts([scenes_report_path, matching_report_path, train_scenes_path])
    mlflow_run.log_params(
        {
            "scene_count": len(matches),
            "scene_missing_count": len(missing),
            "scene_ambiguous_count": len(ambiguous),
            "annotation_uri": annotation_uri,
            "scenes_uri": scenes_uri,
            "debug_pseudolabel": True,
        }
    )
    log_fn(job_log, f"debug_pseudolabel matched={len(matches)} missing={len(missing)} ambiguous={len(ambiguous)}")

    input_bands = job.params.get("input_bands") or model_cfg.get("input_bands") or [1, 2, 3, 4]
    input_bands = [int(band) for band in input_bands]
    patch_size = int(job.preprocess.get("tile_size") or job.train.get("patch_size") or 1024)
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu_only else "cpu")
    model_name = str(model_cfg.get("name") or job.predict.get("model_name") or "tiny_unet_4ch")
    base_channels = int(model_cfg.get("base_channels") or job.train.get("base_channels") or 8)
    model = _build_model(model_name, len(input_bands), int(model_cfg.get("out_channels") or 1), base_channels).to(device)
    checkpoint_path = pseudolabel_cfg.get("checkpoint_path") or job.predict.get("checkpoint_path") or job.params.get("checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError("Debug pseudolabel job requires predict.pseudolabel.checkpoint_path")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint payload at {checkpoint_path}")
    model.load_state_dict(state)
    gt_shapes = _load_shapes(config, annotation_uri)
    prepare_duration_sec = round(time.time() - prepare_started, 3)

    postprocess_started = time.time()
    with trace_stage(
        "postprocess_vectors",
        {"job_id": job.job_id, "model_name": model_name, "scene_count": len(matches), "tile_size": patch_size},
    ):
        postprocess_metrics, artifacts = _write_pseudolabel_outputs(
            config,
            job,
            experiment_dir,
            model,
            device,
            matches,
            input_bands,
            patch_size,
            float((job.postprocess.get("thresholds") or [0.5])[0]),
            seed,
            gt_shapes=gt_shapes,
            object_metrics_prefix="val",
        )
    numeric_metrics = {key: value for key, value in postprocess_metrics.items() if isinstance(value, (int, float, bool))}
    if numeric_metrics:
        mlflow_run.log_metrics(numeric_metrics)
    mlflow_run.log_artifacts(artifacts)

    postprocess_duration_sec = round(time.time() - postprocess_started, 3)
    selected_postprocess_params = {
        "threshold": postprocess_metrics.get("threshold_used"),
        "min_object_area_m2": postprocess_metrics.get("min_object_area_m2_used"),
        "simplify_tolerance_m": postprocess_metrics.get("simplify_tolerance_m_used"),
        "max_objects": job.postprocess.get("max_objects"),
    }
    return {
        "status": "done",
        "mode": "debug_pseudolabel",
        "model_name": model_name,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "timing": {
            "prepare_duration_sec": prepare_duration_sec,
            "train_duration_sec": 0,
            "eval_duration_sec": None,
            "pseudolabel_duration_sec": postprocess_duration_sec,
            "postprocess_duration_sec": postprocess_duration_sec,
        },
        "postprocess_metrics": postprocess_metrics,
        "pseudolabel": {
            "status": "done" if postprocess_metrics.get("pseudolabel_enabled") else "skipped",
            "accepted_objects": postprocess_metrics.get("accepted_objects"),
            "accepted_geojson_mb": postprocess_metrics.get("accepted_geojson_mb"),
            "total_vertices": postprocess_metrics.get("total_vertices"),
            "total_area_m2": postprocess_metrics.get("total_area_m2"),
            "selected_postprocess_params": selected_postprocess_params,
            "mlflow_artifacts": {
                "accepted_geojson": f"{job.job_id}.accepted.geojson",
                "coverage_report": "coverage_report.json",
                "inference_timing_report": "inference_timing_report.json",
                "tiling_debug": "tiling_debug.json",
                "postprocess_debug": "postprocess_debug.json",
                "pseudolabel_summary": "pseudolabel_summary.json",
                "object_metrics": "object_metrics.json",
                "object_matches": "object_matches.csv",
                "prediction_examples": "prediction_examples.html",
                "pseudolabel_scenes": "pseudolabel_scenes.txt",
            },
            "s3_uris": {
                "accepted_geojson": None,
                "accepted_geojson_gz": None,
                "accepted_gpkg": None,
            },
        },
        "matched_scenes_count": len(matches),
        "total_matched_scenes_count": total_matched_count,
        "missing_scenes": missing,
        "ambiguous_scenes": ambiguous,
        "scenes_match_report": str(scenes_report_path),
        "warnings": [f"{len(missing)} scenes from scenes.txt were not matched"] if missing else [],
    }


def run_real_train(
    config: PipelineConfig,
    job: JobSpec,
    experiment_dir: Path,
    mlflow_run: MLflowJobRun,
    job_log: Path,
    log_fn: Any,
) -> dict[str, Any]:
    data = job.data or {}
    model_cfg = (job.params.get("model") if isinstance(job.params.get("model"), dict) else {}) or {}
    model_cfg = {**model_cfg, **(job.train.get("model") or {})}
    images_uri = data.get("images_uri") or config.s3_paths.images
    layout_uri = data.get("layout_uri") or f"{config.s3_paths.layouts.rstrip('/')}/{job.class_name or 'deforest'}/"
    scenes_file = data.get("scenes_file") or "scenes.txt"
    annotation_file = data.get("annotation_file") or "auto"
    seed = int(job.train.get("seed") or job.params.get("seed") or 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prepare_started = time.time()
    images = _list_s3_objects(config, images_uri, suffixes=(".tif", ".tiff"))
    annotation_uri, scenes_uri = _find_layout_files(config, layout_uri, scenes_file, annotation_file)
    entries = [
        line.strip()
        for line in _read_s3_text(config, scenes_uri).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    matching_report = build_scene_matching_report(entries, images)
    matches = [SceneMatch(**item) for item in matching_report["matched"]]
    ambiguous = matching_report["ambiguous"]
    missing = matching_report["missing"]
    strict_matching = str(data.get("scene_name_matching") or "").lower() == "strict_after_fuzzy_validation"
    if strict_matching and (missing or ambiguous or len(matches) != len(entries)):
        raise RuntimeError(
            "Scene matching is not complete: "
            f"total={len(entries)} matched={len(matches)} missing={len(missing)} ambiguous={len(ambiguous)}"
        )
    if not matches:
        raise RuntimeError("No scenes from scenes.txt matched available images")
    max_scenes = job.preprocess.get("max_scenes") or job.train.get("max_scenes") or data.get("max_scenes")
    if max_scenes is not None:
        matches = matches[: max(1, int(max_scenes))]

    scene_report = {
        **matching_report,
        "images_uri": images_uri,
        "layout_uri": layout_uri,
        "annotation_uri": annotation_uri,
        "scenes_uri": scenes_uri,
        "matched": [match.__dict__ for match in matches],
    }
    scenes_report_path = experiment_dir / "scenes_match_report.json"
    matching_report_path = experiment_dir / "scene_matching_report.json"
    write_json(scenes_report_path, scene_report)
    write_json(matching_report_path, scene_report)
    mlflow_run.log_artifacts([scenes_report_path, matching_report_path])
    mlflow_run.log_params(
        {
            "scene_count": len(matches),
            "scene_missing_count": len(missing),
            "scene_ambiguous_count": len(ambiguous),
            "annotation_uri": annotation_uri,
            "scenes_uri": scenes_uri,
        }
    )
    log_fn(job_log, f"real_train matched={len(matches)} missing={len(missing)} ambiguous={len(ambiguous)}")

    input_bands = job.params.get("input_bands") or model_cfg.get("input_bands") or [1, 2, 3, 4]
    input_bands = [int(band) for band in input_bands]
    patch_size = int(job.train.get("patch_size") or job.train.get("train_patch_size") or 256)
    max_train_tiles, max_val_tiles, max_tiles_per_scene = _default_tile_limits(job)
    empty_share = float(job.preprocess.get("max_empty_tile_share") or 0.5)
    with trace_stage("prepare_dataset", {"job_id": job.job_id, "scene_count": len(matches), "tile_size": patch_size}):
        shapes = _load_shapes(config, annotation_uri)

    split_idx = max(1, int(math.ceil(len(matches) * 0.75)))
    train_matches = matches[:split_idx]
    val_matches = matches[split_idx:] or matches[-1:]
    train_samples, train_report = _read_samples(
        config,
        train_matches,
        shapes,
        input_bands,
        patch_size,
        max_tiles_per_scene,
        max_train_tiles,
        empty_share,
        seed,
    )
    val_samples, val_report = _read_samples(
        config,
        val_matches,
        shapes,
        input_bands,
        patch_size,
        max(1, max_tiles_per_scene // 2),
        max_val_tiles,
        empty_share,
        seed + 1000,
    )
    if not val_samples and train_samples and bool(job.train.get("allow_train_val_sample_fallback", False)):
        fallback_count = max(1, min(max_val_tiles, max(1, len(train_samples) // 4)))
        if len(train_samples) > fallback_count:
            val_samples = train_samples[-fallback_count:]
            train_samples = train_samples[:-fallback_count]
        else:
            val_samples = train_samples[-fallback_count:]
        fallback_report = dict(train_report[-1]) if train_report else {"scene": "train_holdout"}
        fallback_report["validation_fallback_from_train_samples"] = True
        val_report = [fallback_report]
    if not train_samples or not val_samples:
        raise RuntimeError(f"Not enough samples: train={len(train_samples)} val={len(val_samples)}")

    train_positive_scene_count = sum(1 for row in train_report if row.get("positive_scene"))
    val_positive_scene_count = sum(1 for row in val_report if row.get("positive_scene"))
    train_negative_scene_count = max(0, len(train_report) - train_positive_scene_count)
    val_negative_scene_count = max(0, len(val_report) - val_positive_scene_count)
    train_positive_tiles = sum(int(mask.sum() > 0) for _, mask in train_samples)
    val_positive_tiles = sum(int(mask.sum() > 0) for _, mask in val_samples)
    train_negative_tiles = sum(int(mask.sum() == 0) for _, mask in train_samples)
    val_negative_tiles = sum(int(mask.sum() == 0) for _, mask in val_samples)
    dataset_report = {
        "train_scene_count": len(train_matches),
        "val_scene_count": len(val_matches),
        "positive_scene_count": train_positive_scene_count + val_positive_scene_count,
        "negative_scene_count": train_negative_scene_count + val_negative_scene_count,
        "train_positive_scene_count": train_positive_scene_count,
        "train_negative_scene_count": train_negative_scene_count,
        "val_positive_scene_count": val_positive_scene_count,
        "val_negative_scene_count": val_negative_scene_count,
        "train_tile_count": len(train_samples),
        "val_tile_count": len(val_samples),
        "train_positive_tiles": train_positive_tiles,
        "val_positive_tiles": val_positive_tiles,
        "train_negative_tiles": train_negative_tiles,
        "val_negative_tiles": val_negative_tiles,
        "positive_tile_count": train_positive_tiles + val_positive_tiles,
        "negative_tile_count": train_negative_tiles + val_negative_tiles,
        "patch_size": patch_size,
        "train_scenes": train_report,
        "val_scenes": val_report,
    }
    dataset_report_path = experiment_dir / "train_dataset_report.json"
    write_json(dataset_report_path, dataset_report)
    train_scenes_path = _write_scene_list(experiment_dir / "train_scenes.txt", train_matches)
    val_scenes_path = _write_scene_list(experiment_dir / "val_scenes.txt", val_matches)
    mlflow_run.log_params(
        {
            "positive_scene_count": dataset_report["positive_scene_count"],
            "negative_scene_count": dataset_report["negative_scene_count"],
            "positive_tile_count": dataset_report["positive_tile_count"],
            "negative_tile_count": dataset_report["negative_tile_count"],
        }
    )
    prepare_duration_sec = round(time.time() - prepare_started, 3)

    require_gpu = bool(job.train.get("require_gpu", False) or job.resources.requires_gpu)
    cuda_available = bool(torch.cuda.is_available())
    if require_gpu and (config.cpu_only or not cuda_available):
        raise RuntimeError(
            "GPU training was requested, but CUDA is not available "
            f"(config.cpu_only={config.cpu_only}, torch.cuda.is_available={cuda_available})."
        )
    device = torch.device("cuda" if cuda_available and not config.cpu_only else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    mlflow_run.log_params(
        {
            "train.require_gpu": require_gpu,
            "train.device": str(device),
            "train.cuda_available": cuda_available,
            "train.gpu_name": gpu_name or "",
        }
    )
    log_fn(job_log, f"real_train device={device} cuda_available={cuda_available} gpu_name={gpu_name or 'none'}")
    model_name = str(model_cfg.get("name") or job.train.get("model_name") or "tiny_unet_4ch")
    model = _build_model(model_name, len(input_bands), 1, int(job.train.get("base_channels") or 8)).to(device)
    freeze_batchnorm_default = model_name.lower().startswith(("deeplab", "deeplabv3plus"))
    freeze_batchnorm = bool(job.train.get("freeze_batchnorm", freeze_batchnorm_default))
    mlflow_run.log_params({"freeze_batchnorm": freeze_batchnorm})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(job.train.get("learning_rate") or 5e-4))
    batch_size = job.train.get("batch_size") or 2
    if batch_size == "auto":
        batch_size = _auto_batch_size(model_name, patch_size, device)
    batch_size = max(1, int(batch_size))
    mlflow_run.log_params(
        {
            "train.batch_size_resolved": batch_size,
            "train.max_train_tiles": max_train_tiles,
            "train.max_val_tiles": max_val_tiles,
            "train.max_tiles_per_scene": max_tiles_per_scene,
        }
    )
    epochs = int(job.train.get("epochs") or 20)
    time_limit_sec = int(job.train.get("time_limit_sec") or 600)
    early_cfg = job.train.get("early_stopping") or {}
    early_enabled = bool(early_cfg.get("enabled", job.train.get("early_stopping_enabled", False)))
    early_patience = int(early_cfg.get("patience") or job.train.get("early_stopping_patience") or 10)
    train_started = time.time()
    started = train_started
    history: list[dict[str, float]] = []

    def make_batches(samples: list[tuple[np.ndarray, np.ndarray]], shuffle: bool) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        rows = list(samples)
        if shuffle:
            random.shuffle(rows)
        return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    train_tensor_pair: tuple[torch.Tensor, torch.Tensor] | None = None
    val_tensor_pair: tuple[torch.Tensor, torch.Tensor] | None = None
    if device.type == "cuda" and bool(job.train.get("cache_samples_on_gpu", True)):
        train_x = torch.from_numpy(np.stack([item[0] for item in train_samples])).to(device)
        train_y = torch.from_numpy(np.stack([item[1] for item in train_samples])).to(device)
        val_x = torch.from_numpy(np.stack([item[0] for item in val_samples])).to(device)
        val_y = torch.from_numpy(np.stack([item[1] for item in val_samples])).to(device)
        train_tensor_pair = (train_x, train_y)
        val_tensor_pair = (val_x, val_y)
        cached_mb = (
            train_x.numel() * train_x.element_size()
            + train_y.numel() * train_y.element_size()
            + val_x.numel() * val_x.element_size()
            + val_y.numel() * val_y.element_size()
        ) / (1024 * 1024)
        mlflow_run.log_params({"train.cache_samples_on_gpu": True, "train.cached_samples_gpu_mb": round(cached_mb, 3)})
        log_fn(job_log, f"real_train cached_samples_on_gpu=true cached_mb={cached_mb:.1f} batch_size={batch_size}")

    def make_index_batches(count: int, shuffle: bool) -> list[list[int]]:
        indices = list(range(count))
        if shuffle:
            random.shuffle(indices)
        return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]

    best_val_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    train_trace = trace_stage("train_model", {"job_id": job.job_id, "model_name": model_name, "tile_size": patch_size, "epoch_count": epochs})
    train_trace.__enter__()
    try:
        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            model.train()
            if freeze_batchnorm:
                _set_batchnorm_eval(model)
            train_losses = []
            train_metrics = []
            if train_tensor_pair is not None:
                train_x, train_y = train_tensor_pair
                for batch_indices in make_index_batches(train_x.shape[0], shuffle=True):
                    idx = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
                    x = train_x.index_select(0, idx)
                    y = train_y.index_select(0, idx)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = _loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(float(loss.item()))
                    train_metrics.append(_dice_iou_precision_recall(logits.detach(), y))
                    if time.time() - started > time_limit_sec:
                        break
            else:
                for batch in make_batches(train_samples, shuffle=True):
                    x = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
                    y = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = _loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(float(loss.item()))
                    train_metrics.append(_dice_iou_precision_recall(logits.detach(), y))
                    if time.time() - started > time_limit_sec:
                        break

            model.eval()
            val_losses = []
            val_metrics = []
            with torch.no_grad():
                if val_tensor_pair is not None:
                    val_x, val_y = val_tensor_pair
                    for batch_indices in make_index_batches(val_x.shape[0], shuffle=False):
                        idx = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
                        x = val_x.index_select(0, idx)
                        y = val_y.index_select(0, idx)
                        logits = model(x)
                        val_losses.append(float(_loss_fn(logits, y).item()))
                        val_metrics.append(_dice_iou_precision_recall(logits, y))
                else:
                    for batch in make_batches(val_samples, shuffle=False):
                        x = torch.from_numpy(np.stack([item[0] for item in batch])).to(device)
                        y = torch.from_numpy(np.stack([item[1] for item in batch])).to(device)
                        logits = model(x)
                        val_losses.append(float(_loss_fn(logits, y).item()))
                        val_metrics.append(_dice_iou_precision_recall(logits, y))

            def avg_metric(rows: list[dict[str, float]], name: str) -> float:
                return float(np.mean([row[name] for row in rows])) if rows else 0.0

            row = {
                "epoch": float(epoch),
                "train/loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "train/dice": avg_metric(train_metrics, "dice"),
                "train/iou": avg_metric(train_metrics, "iou"),
                "val/loss": float(np.mean(val_losses)) if val_losses else 0.0,
                "val/dice": avg_metric(val_metrics, "dice"),
                "val/iou": avg_metric(val_metrics, "iou"),
                "val/pixel_dice": avg_metric(val_metrics, "dice"),
                "val/pixel_iou": avg_metric(val_metrics, "iou"),
                "val/precision": avg_metric(val_metrics, "precision"),
                "val/recall": avg_metric(val_metrics, "recall"),
                "val/pixel_f1": avg_metric(val_metrics, "f1"),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_duration_sec": round(time.time() - epoch_started, 4),
            }
            if device.type == "cuda":
                torch.cuda.synchronize()
                row["system/cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                row["system/cuda_memory_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 * 1024), 3)
                row["system/gpu_train_confirmed"] = 1.0
            history.append(row)
            mlflow_run.log_metrics({key: value for key, value in row.items() if key != "epoch"}, step=epoch)
            log_fn(job_log, f"real_train epoch={epoch} val_iou={row['val/iou']:.6f} duration={row['epoch_duration_sec']}")
            if row["val/iou"] > best_val_iou:
                best_val_iou = row["val/iou"]
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if early_enabled and epochs_without_improvement >= early_patience:
                log_fn(job_log, f"real_train early_stopping epoch={epoch} patience={early_patience}")
                break
            if time.time() - started > time_limit_sec:
                break
    finally:
        train_trace.__exit__(None, None, None)

    train_duration_sec = round(time.time() - train_started, 3)
    artifacts = _write_history(experiment_dir, history)
    checkpoint_path = experiment_dir / "tiny_unet_4ch.pt"
    checkpoint_path = experiment_dir / f"{model_name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "job_id": job.job_id, "model_name": model_name, "best_epoch": best_epoch}, checkpoint_path)
    artifacts.extend([dataset_report_path, train_scenes_path, val_scenes_path, checkpoint_path])
    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    pseudolabel_matches = matches
    if str(pseudolabel_cfg.get("run_on") or "").lower() in {"all_available_images", "all_images"}:
        pseudolabel_matches = [
            SceneMatch(entry=item["name"], key=item["key"], name=item["name"], score=1.0)
            for item in sorted(images, key=lambda row: row["key"])
        ]
    max_pseudolabel_scenes = pseudolabel_cfg.get("max_scenes")
    if max_pseudolabel_scenes is not None:
        pseudolabel_matches = pseudolabel_matches[: max(1, int(max_pseudolabel_scenes))]
    object_metrics_prefix = "val" if str(pseudolabel_cfg.get("run_on") or "").lower() not in {"all_available_images", "all_images"} else "pseudolabel"
    with trace_stage(
        "postprocess_vectors",
        {
            "job_id": job.job_id,
            "model_name": model_name,
            "scene_count": len(pseudolabel_matches),
            "tile_size": patch_size,
        },
    ):
        postprocess_metrics, postprocess_artifacts = _write_pseudolabel_outputs(
            config,
            job,
            experiment_dir,
            model,
            device,
            pseudolabel_matches,
            input_bands,
            patch_size,
            float((job.postprocess.get("thresholds") or [0.5])[0]),
            seed,
            gt_shapes=shapes,
            object_metrics_prefix=object_metrics_prefix,
        )
    if postprocess_metrics.get("pseudolabel_enabled"):
        mlflow_run.log_metrics(
            {
                key: value
                for key, value in postprocess_metrics.items()
                if isinstance(value, (int, float, bool))
            }
        )
        artifacts.extend(postprocess_artifacts)
    mlflow_run.log_artifacts(artifacts)

    last = history[-1] if history else {}
    postprocess_duration_sec = postprocess_metrics.get("postprocess_sec")
    pseudolabel_status = "done" if postprocess_metrics.get("pseudolabel_enabled") else "skipped"
    selected_postprocess_params = {
        "threshold": postprocess_metrics.get("threshold_used"),
        "min_object_area_m2": postprocess_metrics.get("min_object_area_m2_used"),
        "simplify_tolerance_m": postprocess_metrics.get("simplify_tolerance_m_used"),
        "max_objects": job.postprocess.get("max_objects"),
    }
    return {
        "status": "done",
        "mode": "real_train",
        "model_name": model_name,
        "device": str(device),
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "require_gpu": require_gpu,
        "timing": {
            "prepare_duration_sec": prepare_duration_sec,
            "train_duration_sec": train_duration_sec,
            "eval_duration_sec": None,
            "pseudolabel_duration_sec": postprocess_duration_sec,
            "postprocess_duration_sec": postprocess_duration_sec,
        },
        "epochs_completed": len(history),
        "time_limit_sec": time_limit_sec,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "last_epoch_metrics": last,
        "postprocess_metrics": postprocess_metrics,
        "pseudolabel": {
            "status": pseudolabel_status,
            "accepted_objects": postprocess_metrics.get("accepted_objects"),
            "accepted_geojson_mb": postprocess_metrics.get("accepted_geojson_mb"),
            "total_vertices": postprocess_metrics.get("total_vertices"),
            "total_area_m2": postprocess_metrics.get("total_area_m2"),
            "selected_postprocess_params": selected_postprocess_params,
            "mlflow_artifacts": {
                "accepted_geojson": f"{job.job_id}.accepted.geojson",
                "pseudolabel_summary": "pseudolabel_summary.json",
                "object_metrics": "object_metrics.json",
                "object_matches": "object_matches.csv",
                "prediction_examples": "prediction_examples.html",
                "train_scenes": "train_scenes.txt",
                "pseudolabel_scenes": "pseudolabel_scenes.txt",
            },
            "s3_uris": {
                "accepted_geojson": None,
                "accepted_geojson_gz": None,
                "accepted_gpkg": None,
            },
        },
        "matched_scenes_count": len(matches),
        "missing_scenes": missing,
        "ambiguous_scenes": ambiguous,
        "train_scene_count": len(train_matches),
        "val_scene_count": len(val_matches),
        "train_tile_count": len(train_samples),
        "val_tile_count": len(val_samples),
        "train_positive_tiles": dataset_report["train_positive_tiles"],
        "val_positive_tiles": dataset_report["val_positive_tiles"],
        "train_negative_tiles": dataset_report["train_negative_tiles"],
        "val_negative_tiles": dataset_report["val_negative_tiles"],
        "positive_scene_count": dataset_report["positive_scene_count"],
        "negative_scene_count": dataset_report["negative_scene_count"],
        "positive_tile_count": dataset_report["positive_tile_count"],
        "negative_tile_count": dataset_report["negative_tile_count"],
        "scenes_match_report": str(scenes_report_path),
        "train_dataset_report": str(dataset_report_path),
        "history_path": str(experiment_dir / "history.json"),
        "history_csv_path": str(experiment_dir / "history.csv"),
        "warnings": [f"{len(missing)} scenes from scenes.txt were not matched"] if missing else [],
    }
