from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import time
import difflib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from shapely.geometry import box, shape

from .io_utils import write_json
from .data import scene_matching as scene_matching_mod
from .job_schema import JobSpec
from .tile_preparation import (
    SceneInput,
    TilePreparationFacade,
    TrainingTileDataset,
)
from .tile_preparation.config import train_sampling_enabled as resolve_train_sampling_enabled
from .tile_preparation.dataset import iter_dataset_batches
from .tile_preparation.summary import summarize_tile_records
from .mlflow_adapter import MLflowJobRun, trace_stage
from .metrics.debug_dump import (
    build_sample_payload,
    manifest_hash,
    metrics_debug_config,
    metrics_debug_enabled,
    write_epoch_debug,
    write_metrics_debug_report,
)
from .metrics.segmentation import PixelMetricAccumulator, WeightedLossAccumulator, binary_segmentation_metrics_from_logits, probabilities_from_logits
from .models.factory import build_model as build_model_mod
from .models.factory import set_batchnorm_eval as set_batchnorm_eval_mod
from .object_metrics import compute_object_f1
from .pipeline.pseudolabel_pipeline import (
    run_pseudolabel_inference_stage,
    run_pseudolabel_pipeline,
    run_pseudolabel_postprocess_stage,
)
from .pipeline_config import PipelineConfig
from .reporting.prediction_examples import write_prediction_examples_report as write_prediction_examples_report_mod
from .storage import s3 as s3_storage
from .tiling import windows as tiling_windows
from .training.losses import segmentation_loss, segmentation_loss_components


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
    preferred_key_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    return scene_matching_mod.build_scene_matching_report(
        entries,
        images,
        accept_threshold=accept_threshold,
        ambiguous_margin=ambiguous_margin,
        preferred_key_prefixes=preferred_key_prefixes,
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


def _filter_shapes_to_matches(config: PipelineConfig, matches: list[SceneMatch], shapes: list[Any]) -> list[Any]:
    import rasterio
    from rasterio.warp import transform_bounds

    if not matches or not shapes:
        return []
    shape_crs = "EPSG:3857" if any(max(map(abs, geom.bounds)) > 1000 for geom in shapes if not geom.is_empty) else "EPSG:4326"
    selected: list[Any] = []
    seen: set[int] = set()
    aws = _aws_session(config)
    with rasterio.Env(aws, AWS_HTTPS="NO", AWS_VIRTUAL_HOSTING="FALSE"):
        for match in matches:
            path = s3_storage.raster_path_for_s3_key(config, match.key)
            with rasterio.open(path) as ds:
                scene_crs = str(ds.crs) if ds.crs else shape_crs
                bounds = tuple(ds.bounds)
                if scene_crs != shape_crs:
                    bounds = transform_bounds(scene_crs, shape_crs, *bounds, densify_pts=21)
                scene_bounds = box(*bounds)
            for index, geom in enumerate(shapes):
                if index in seen:
                    continue
                if geom.is_valid and not geom.is_empty and geom.intersects(scene_bounds):
                    selected.append(geom)
                    seen.add(index)
    return selected


def _split_train_val_matches(
    config: PipelineConfig,
    matches: list[SceneMatch],
    shapes: list[Any],
    *,
    train_fraction: float,
    seed: int,
    stratify_positive: bool,
) -> tuple[list[SceneMatch], list[SceneMatch]]:
    if not stratify_positive or len(matches) <= 1:
        split_idx = max(1, int(math.ceil(len(matches) * train_fraction)))
        return matches[:split_idx], matches[split_idx:] or matches[-1:]

    rng = random.Random(seed)
    positive: list[SceneMatch] = []
    negative: list[SceneMatch] = []
    for match in matches:
        if _filter_shapes_to_matches(config, [match], shapes):
            positive.append(match)
        else:
            negative.append(match)

    rng.shuffle(positive)
    rng.shuffle(negative)

    def split_group(group: list[SceneMatch]) -> tuple[list[SceneMatch], list[SceneMatch]]:
        if not group:
            return [], []
        if len(group) == 1:
            return group, []
        train_count = max(1, min(len(group) - 1, int(math.ceil(len(group) * train_fraction))))
        return group[:train_count], group[train_count:]

    train_positive, val_positive = split_group(positive)
    train_negative, val_negative = split_group(negative)
    train_matches = train_positive + train_negative
    val_matches = val_positive + val_negative
    if not val_matches:
        val_matches = train_matches[-1:]
        train_matches = train_matches[:-1] or train_matches
    return train_matches, val_matches


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
        "segformer_b2": "mit_b2",
        "segformer_b3": "mit_b3",
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
            "segformer_b2",
            "segformer_b3",
            "deeplabv3plus_resnet34",
            "deeplabv3plus_resnet50",
        ]
    )
    raise ValueError(f"Unsupported model_name={model_name}. Supported: {supported}")


def _set_batchnorm_eval(model: torch.nn.Module) -> None:
    set_batchnorm_eval_mod(model)


def _configure_dropout(model: torch.nn.Module, dropout_p: Any) -> int:
    if dropout_p is None:
        return 0
    probability = float(dropout_p)
    if probability < 0.0 or probability >= 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {probability}")
    updated = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.p = probability
            updated += 1
    return updated


def _build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer") or train_cfg.get("optimizer_name") or "adamw").strip().lower()
    lr_value = train_cfg.get("learning_rate")
    weight_decay_value = train_cfg.get("weight_decay")
    lr = float(5e-4 if lr_value in (None, "") else lr_value)
    weight_decay = float(0.0 if weight_decay_value in (None, "") else weight_decay_value)
    if name in {"adamw", "adam_w"}:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name in {"sgd", "momentum_sgd"}:
        momentum = float(train_cfg.get("momentum") or 0.9)
        nesterov = bool(train_cfg.get("nesterov", False))
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    raise ValueError(f"Unsupported optimizer={name}; supported: adamw, adam, sgd")


def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any], epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
    cfg = train_cfg.get("scheduler")
    if cfg is None:
        name = str(train_cfg.get("scheduler_name") or "none").strip().lower()
        cfg = {"name": name}
    elif isinstance(cfg, str):
        cfg = {"name": cfg}
    elif not isinstance(cfg, dict):
        cfg = {"name": "none"}
    name = str(cfg.get("name") or cfg.get("type") or "none").strip().lower()
    if name in {"", "none", "off", "disabled"}:
        return None
    if name in {"cosine", "cosine_annealing"}:
        t_max = int(cfg.get("t_max") or train_cfg.get("scheduler_t_max") or epochs)
        eta_min = float(cfg.get("eta_min") or train_cfg.get("scheduler_eta_min") or 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max), eta_min=eta_min)
    if name in {"step", "step_lr"}:
        step_size = int(cfg.get("step_size") or train_cfg.get("scheduler_step_size") or max(1, epochs // 3))
        gamma = float(cfg.get("gamma") or train_cfg.get("scheduler_gamma") or 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, step_size), gamma=gamma)
    raise ValueError(f"Unsupported scheduler={name}; supported: none, cosine, step")


def _metric_improved(value: float, best: float, *, maximize: bool) -> bool:
    return value > best if maximize else value < best


def _threshold_metric_suffix(threshold: float) -> str:
    text = f"{float(threshold):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "_") or "0"


def _coerce_threshold_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, dict):
        items = value.get("values") or value.get("thresholds") or []
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    thresholds: list[float] = []
    for item in items:
        if item is None or item == "":
            continue
        threshold = float(item)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"metric threshold must be in [0, 1], got {threshold}")
        thresholds.append(threshold)
    return thresholds


def _resolve_metric_thresholds(job: JobSpec, base_threshold: float) -> list[float]:
    raw = (
        job.train.get("metric_thresholds")
        or job.train.get("threshold_sweep")
        or job.evaluate.get("metric_thresholds")
        or job.params.get("metric_thresholds")
        or job.params.get("threshold_sweep")
    )
    values = [float(base_threshold), *_coerce_threshold_values(raw)]
    unique: dict[str, float] = {}
    for threshold in values:
        unique[f"{float(threshold):.6f}"] = float(threshold)
    return sorted(unique.values())


def _checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            state = payload.get(key)
            if isinstance(state, dict):
                return state
        if payload and all(isinstance(key, str) for key in payload.keys()):
            return payload
    raise RuntimeError("Unsupported checkpoint payload: expected model_state_dict/state_dict mapping")


def _load_initial_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, *, device: torch.device, strict: bool) -> dict[str, Any]:
    path = Path(str(checkpoint_path))
    if not path.exists():
        raise FileNotFoundError(f"Initial checkpoint is missing: {path}")
    payload = torch.load(str(path), map_location=device)
    state = _checkpoint_state_dict(payload)
    incompatible = model.load_state_dict(state, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []) or [])
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    return {
        "path": str(path),
        "strict": strict,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def _dice_iou_precision_recall(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    metrics = binary_segmentation_metrics_from_logits(logits, target, threshold=threshold)
    return {
        "dice": float(metrics["pixel_f1"]),
        "iou": float(metrics["pixel_iou"]),
        "precision": float(metrics["pixel_precision"]),
        "recall": float(metrics["pixel_recall"]),
        "f1": float(metrics["pixel_f1"]),
    }


def _loss_fn(logits: torch.Tensor, target: torch.Tensor, config: dict[str, Any] | None = None) -> torch.Tensor:
    return segmentation_loss(logits, target, config=config)


def _loss_components(logits: torch.Tensor, target: torch.Tensor, config: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    return segmentation_loss_components(logits, target, config=config)


def _metric_class_key(class_name: str) -> str:
    value = str(class_name or "class").strip().lower()
    if value in {"deforest", "cuttings", "clearcuts", "clear_cuts", "вырубки"}:
        return "cuttings"
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    return value.strip("_") or "class"


def _collect_val_debug_samples(
    *,
    batch_indices: list[int],
    x: torch.Tensor,
    y: torch.Tensor,
    logits: torch.Tensor,
    val_sample_records: list[dict[str, Any]],
    threshold: float,
    rows: list[dict[str, Any]],
    payloads: list[dict[str, Any]],
    save_all: bool,
) -> None:
    probs = probabilities_from_logits(logits.detach())
    pred_masks = probs >= float(threshold)
    max_payloads = len(batch_indices) if save_all else max(0, 10 - len(payloads))
    for local_index, sample_index in enumerate(batch_indices):
        metadata = dict(val_sample_records[sample_index]) if sample_index < len(val_sample_records) else {"sample_id": str(sample_index)}
        metadata.setdefault("sample_id", str(sample_index))
        payload = build_sample_payload(
            metadata=metadata,
            image=x[local_index],
            gt_mask=y[local_index],
            pred_prob=probs[local_index],
            pred_mask=pred_masks[local_index],
        )
        rows.append(dict(payload["metadata"]))
        if len(payloads) < len(rows) and (save_all or local_index < max_payloads):
            payloads.append(payload)


def _object_summary_from_sample_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    gt_total = sum(int(row.get("gt_objects_count") or 0) for row in rows)
    pred_total = sum(int(row.get("pred_objects_count") or 0) for row in rows)
    matched_total = sum(int(row.get("matched_objects_count") or 0) for row in rows)
    fp = max(0, pred_total - matched_total)
    fn = max(0, gt_total - matched_total)
    precision = matched_total / pred_total if pred_total else 0.0
    recall = matched_total / gt_total if gt_total else 0.0
    f1 = (2 * matched_total) / (2 * matched_total + fp + fn) if (2 * matched_total + fp + fn) else 0.0
    return {
        "object_tp": float(matched_total),
        "object_fp": float(fp),
        "object_fn": float(fn),
        "object_precision": float(precision),
        "object_recall": float(recall),
        "object_f1": float(f1),
        "gt_objects": float(gt_total),
        "pred_objects": float(pred_total),
    }


def _augmentation_profile(augmentations: Any) -> str:
    if not isinstance(augmentations, dict):
        return "none"
    enabled = sorted(str(key) for key, value in augmentations.items() if bool(value))
    return ",".join(enabled) if enabled else "none"


def _scene_inputs_for_matches(config: PipelineConfig, matches: list[SceneMatch]) -> list[SceneInput]:
    scenes: list[SceneInput] = []
    for match in matches:
        image_path = s3_storage.raster_path_for_s3_key(config, match.key)
        scenes.append(
            SceneInput(
                image_path=str(image_path),
                scene_id=match.name,
                metadata={"entry": match.entry, "s3_key": match.key, "score": match.score},
            )
        )
    return scenes


def _write_training_annotation_geojson(config: PipelineConfig, annotation_uri: str, experiment_dir: Path) -> Path:
    payload = _read_s3_json(config, annotation_uri)
    path = experiment_dir / "training_annotation.geojson"
    write_json(path, payload)
    return path


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


def _explicit_tile_total_limit(job: JobSpec, key: str) -> Any:
    if key in job.train:
        return job.train.get(key)
    if key in job.preprocess:
        return job.preprocess.get(key)
    return None


def _full_dataset_tiles_requested(job: JobSpec) -> bool:
    return bool(
        job.preprocess.get("use_full_dataset_tiles")
        or job.preprocess.get("full_dataset")
        or job.train.get("use_full_dataset_tiles")
    )


def _resolve_wallclock_limit(job: JobSpec) -> int | None:
    for source in (job.train, job.params.get("debug_run") if isinstance(job.params.get("debug_run"), dict) else {}):
        if not isinstance(source, dict):
            continue
        for key in ("max_wallclock_seconds", "max_training_seconds"):
            value = source.get(key)
            if value is not None:
                return max(0, int(value))
    if "time_limit_sec" in job.train:
        return max(0, int(job.train["time_limit_sec"]))
    return None


def _auto_batch_size(model_name: str, patch_size: int, device: torch.device) -> int:
    if device.type != "cuda":
        return 1
    name = model_name.lower()
    if "segformer_b3" in name:
        return 1 if patch_size >= 1024 else 2
    if "segformer_b2" in name:
        return 2 if patch_size >= 1024 else 4
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


def _load_prepared_dataset_split(
    experiment_dir: Path,
    matches: list[SceneMatch],
    job: JobSpec,
) -> tuple[list[SceneMatch], list[SceneMatch], dict[str, Any]] | None:
    manifest_value = job.preprocess.get("prepared_dataset_manifest")
    if not manifest_value and job.preprocess.get("use_prepared_dataset_manifest"):
        manifest_value = "dataset_manifest.json"
    if not manifest_value:
        return None
    manifest_path = Path(str(manifest_value))
    if not manifest_path.is_absolute():
        manifest_path = experiment_dir / manifest_path
    if not manifest_path.exists():
        raise RuntimeError(f"Prepared dataset manifest is missing: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_raw = payload.get("train_scenes") or []
    val_raw = payload.get("val_scenes") or []
    if not train_raw or not val_raw:
        raise RuntimeError("Prepared dataset manifest has empty train_scenes or val_scenes")

    lookup = _scene_match_lookup(matches)
    train_matches = _resolve_prepared_matches(train_raw, lookup, "train")
    val_matches = _resolve_prepared_matches(val_raw, lookup, "val")
    train_ids = {_scene_match_identity(match) for match in train_matches}
    val_ids = {_scene_match_identity(match) for match in val_matches}
    overlap = sorted(train_ids & val_ids)
    available_ids = {_scene_match_identity(match) for match in matches}
    if overlap:
        validation_kind = str(job.params.get("validation.kind") or job.preprocess.get("validation_kind") or "").lower()
        allow_sample_fallback = bool(job.train.get("allow_train_val_sample_fallback", False))
        limited_single_scene = (
            len(available_ids) <= 1
            or validation_kind in {"single_scene_tile_holdout_limited", "tile_holdout_limited", "limited"}
            or str(payload.get("split_strategy") or "").lower() == "legacy_75_25"
        )
        if allow_sample_fallback and limited_single_scene:
            val_matches = []
            val_ids = set()
        else:
            raise RuntimeError(f"Prepared dataset split overlap: {overlap[:10]}")
    selected_ids = train_ids | val_ids
    lost = sorted(available_ids - selected_ids)
    limited_manifest = _prepared_manifest_has_input_limit(payload)
    if lost and not limited_manifest:
        raise RuntimeError(f"Prepared dataset split lost {len(lost)} matched scenes: {lost[:10]}")
    return train_matches, val_matches, {
        "source": str(manifest_path),
        "split_strategy": payload.get("split_strategy"),
        "train_scene_count": len(train_matches),
        "val_scene_count": len(val_matches),
        "available_scene_count": len(available_ids),
        "selected_scene_count": len(selected_ids),
        "excluded_scene_count": len(lost),
        "input_limit_applied": limited_manifest,
        "split_summary": payload.get("split_summary") or {},
    }


def _prepared_manifest_has_input_limit(payload: dict[str, Any]) -> bool:
    lineage = payload.get("input_lineage") if isinstance(payload.get("input_lineage"), dict) else {}
    limits = payload.get("limits") if isinstance(payload.get("limits"), dict) else {}
    return bool(
        lineage.get("limit_applied")
        or lineage.get("dataset_input_limit") is not None
        or limits.get("dataset_input_limit") is not None
    )


def _scene_match_identity(match: SceneMatch) -> str:
    return str(match.key or match.name or match.entry)


def _scene_match_lookup(matches: list[SceneMatch]) -> dict[str, SceneMatch]:
    lookup: dict[str, SceneMatch] = {}
    for match in matches:
        for key in {
            str(match.key or ""),
            str(match.name or ""),
            str(match.entry or ""),
            PurePosixPath(str(match.key or "")).name,
            PurePosixPath(str(match.name or "")).name,
            PurePosixPath(str(match.entry or "")).name,
            _norm_scene_name(str(match.key or "")),
            _norm_scene_name(str(match.name or "")),
            _norm_scene_name(str(match.entry or "")),
        }:
            if key:
                lookup.setdefault(key, match)
    return lookup


def _resolve_prepared_matches(raw_items: list[Any], lookup: dict[str, SceneMatch], split_name: str) -> list[SceneMatch]:
    resolved: list[SceneMatch] = []
    missing: list[str] = []
    for item in raw_items:
        values: list[str]
        if isinstance(item, dict):
            values = [
                str(item.get("key") or ""),
                str(item.get("name") or ""),
                str(item.get("entry") or ""),
                PurePosixPath(str(item.get("key") or "")).name,
                PurePosixPath(str(item.get("name") or "")).name,
                PurePosixPath(str(item.get("entry") or "")).name,
                _norm_scene_name(str(item.get("key") or "")),
                _norm_scene_name(str(item.get("name") or "")),
                _norm_scene_name(str(item.get("entry") or "")),
            ]
        else:
            value = str(item)
            values = [value, PurePosixPath(value).name, _norm_scene_name(value)]
        match = next((lookup[value] for value in values if value and value in lookup), None)
        if match:
            resolved.append(match)
        else:
            missing.append(str(item))
    if missing:
        raise RuntimeError(f"Prepared dataset {split_name} scenes are not in current matching result: {missing[:10]}")
    return resolved


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
    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    run_on = str(pseudolabel_cfg.get("run_on") or "").lower()
    pseudolabel_images_uri = str(pseudolabel_cfg.get("images_uri") or images_uri)
    preferred_prefixes = list(job.preprocess.get("scene_matching_prefer_prefixes") or data.get("scene_matching_prefer_prefixes") or [])

    prepare_started = time.time()
    with trace_stage("match_scenes", {"job_id": job.job_id, "images_uri": pseudolabel_images_uri, "layout_uri": layout_uri}):
        images = _list_s3_objects(config, pseudolabel_images_uri, suffixes=(".tif", ".tiff"))
        if run_on in {"all_available_images", "all_images"}:
            annotation_uri = None
            scenes_uri = None
            entries = [str(item["name"]) for item in images]
            matching_report = {
                "matched": [
                    {"entry": item["name"], "key": item["key"], "name": item["name"], "score": 1.0}
                    for item in sorted(images, key=lambda row: row["key"])
                ],
                "missing": [],
                "ambiguous": [],
                "matched_count": len(images),
                "missing_count": 0,
                "ambiguous_count": 0,
            }
        else:
            annotation_uri, scenes_uri = _find_layout_files(config, layout_uri, scenes_file, annotation_file)
            configured_entries = pseudolabel_cfg.get("scene_entries") or pseudolabel_cfg.get("scenes")
            if configured_entries:
                entries = [str(item).strip() for item in configured_entries if str(item).strip()]
            elif run_on in {"validation_scenes", "val_scenes", "validation"} and (experiment_dir / "val_scenes.txt").exists():
                entries = [
                    line.strip()
                    for line in (experiment_dir / "val_scenes.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            elif run_on in {"train_scenes", "training_scenes", "train"} and (experiment_dir / "train_scenes.txt").exists():
                entries = [
                    line.strip()
                    for line in (experiment_dir / "train_scenes.txt").read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            else:
                entries = [
                    line.strip()
                    for line in _read_s3_text(config, scenes_uri).splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            matching_report = build_scene_matching_report(entries, images, preferred_key_prefixes=preferred_prefixes)
    matches = [SceneMatch(**item) for item in matching_report["matched"]]
    if run_on in {"dataset_scenes_plus_extra", "dataset_and_extra_images", "dataset_plus_extra"}:
        extra_images_uri = pseudolabel_cfg.get("extra_images_uri") or pseudolabel_cfg.get("additional_images_uri")
        if extra_images_uri:
            extra_images = _list_s3_objects(config, str(extra_images_uri), suffixes=(".tif", ".tiff"))
            seen_keys = {match.key for match in matches}
            for item in sorted(extra_images, key=lambda row: row["key"]):
                if item["key"] in seen_keys:
                    continue
                matches.append(SceneMatch(entry=item["name"], key=item["key"], name=item["name"], score=1.0))
                seen_keys.add(item["key"])
            matching_report["extra_images_uri"] = str(extra_images_uri)
            matching_report["extra_images_count"] = len(extra_images)
            matching_report["selected_with_extra_count"] = len(matches)
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

    total_matched_count = len(matches)
    max_debug_scenes = pseudolabel_cfg.get("max_debug_scenes", pseudolabel_cfg.get("max_scenes", job.predict.get("max_debug_scenes")))
    if max_debug_scenes is not None:
        matches = matches[: max(1, int(max_debug_scenes))]
    scene_report = {
        **matching_report,
        "images_uri": pseudolabel_images_uri,
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
    preserve_train_scenes = bool(
        pseudolabel_cfg.get("preserve_train_scenes")
        or job.predict.get("preserve_train_scenes")
        or job.params.get("preserve_train_scenes")
    )
    scene_list_path = (
        _write_scene_list(experiment_dir / "pseudolabel_candidate_scenes.txt", matches)
        if preserve_train_scenes
        else _write_scene_list(experiment_dir / "train_scenes.txt", matches)
    )
    mlflow_run.log_artifacts([scenes_report_path, matching_report_path, scene_list_path])
    mlflow_run.log_params(
        {
            "scene_count": len(matches),
            "scene_missing_count": len(missing),
            "scene_ambiguous_count": len(ambiguous),
            "annotation_uri": annotation_uri,
            "scenes_uri": scenes_uri,
            "pseudolabel.images_uri": pseudolabel_images_uri,
            "debug_pseudolabel": True,
        }
    )
    log_fn(job_log, f"debug_pseudolabel matched={len(matches)} missing={len(missing)} ambiguous={len(ambiguous)}")

    input_bands = job.params.get("input_bands") or model_cfg.get("input_bands") or [1, 2, 3, 4]
    input_bands = [int(band) for band in input_bands]
    patch_size = int(job.preprocess.get("tile_size") or job.train.get("patch_size") or 1024)
    model_name = str(model_cfg.get("name") or job.predict.get("model_name") or "tiny_unet_4ch")
    checkpoint_path = pseudolabel_cfg.get("checkpoint_path") or job.predict.get("checkpoint_path") or job.params.get("checkpoint_path")
    stage_mode = str(pseudolabel_cfg.get("_pipeline_stage_mode") or pseudolabel_cfg.get("_stage_mode") or "full").lower()
    if stage_mode in {"postprocess", "vectorize"}:
        gt_shapes = _load_shapes(config, annotation_uri) if annotation_uri else []
        if run_on in {"all_available_images", "all_images"}:
            # Full-pool pseudolabel targets do not have a one-to-one GT scene set;
            # using dataset GT here makes object-F1 tuning both misleading and slow.
            gt_shapes = []
        elif gt_shapes:
            gt_shapes = _filter_shapes_to_matches(config, matches, gt_shapes)
        prepare_duration_sec = round(time.time() - prepare_started, 3)
        postprocess_started = time.time()
        with trace_stage(
            "postprocess_vectors",
            {"job_id": job.job_id, "stage_mode": stage_mode, "scene_count": len(matches), "tile_size": patch_size},
        ):
            postprocess_metrics, artifacts = run_pseudolabel_postprocess_stage(
                config,
                job,
                experiment_dir,
                float((job.postprocess.get("thresholds") or [0.5])[0]),
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
            "mode": "debug_pseudolabel_postprocess",
            "model_name": model_name,
            "device": "cpu",
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
                "status": "done",
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
            },
            "matched_scenes_count": len(matches),
            "total_matched_scenes_count": total_matched_count,
            "missing_scenes": missing,
            "ambiguous_scenes": ambiguous,
            "scenes_match_report": str(scenes_report_path),
            "warnings": [f"{len(missing)} scenes from scenes.txt were not matched"] if missing else [],
        }
    require_gpu = bool(job.train.get("require_gpu", False) or job.resources.requires_gpu)
    cuda_available = bool(torch.cuda.is_available())
    if require_gpu and not cuda_available:
        raise RuntimeError(
            "GPU inference was requested, but CUDA is not available "
            f"(config.cpu_only={config.cpu_only}, torch.cuda.is_available={cuda_available})."
        )
    cpu_only_effective = bool(config.cpu_only) and not require_gpu
    device = torch.device("cuda" if cuda_available and not cpu_only_effective else "cpu")
    base_channels = int(model_cfg.get("base_channels") or job.train.get("base_channels") or 8)
    model = _build_model(model_name, len(input_bands), int(model_cfg.get("out_channels") or 1), base_channels).to(device)
    if not checkpoint_path:
        raise RuntimeError("Debug pseudolabel job requires predict.pseudolabel.checkpoint_path")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint payload at {checkpoint_path}")
    model.load_state_dict(state)
    gt_shapes = _load_shapes(config, annotation_uri) if annotation_uri else []
    if run_on in {"all_available_images", "all_images"}:
        gt_shapes = []
    elif gt_shapes:
        gt_shapes = _filter_shapes_to_matches(config, matches, gt_shapes)
    prepare_duration_sec = round(time.time() - prepare_started, 3)

    postprocess_started = time.time()
    with trace_stage(
        "postprocess_vectors",
        {"job_id": job.job_id, "model_name": model_name, "scene_count": len(matches), "tile_size": patch_size},
    ):
        if stage_mode in {"inference", "infer"}:
            postprocess_metrics, artifacts = run_pseudolabel_inference_stage(
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
            )
        else:
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
    preferred_prefixes = list(job.preprocess.get("scene_matching_prefer_prefixes") or data.get("scene_matching_prefer_prefixes") or [])
    matching_report = build_scene_matching_report(entries, images, preferred_key_prefixes=preferred_prefixes)
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
        "annotation_source": (data.get("annotations") or job.params.get("annotations") or {}).get("source") or "layout_uri",
        "annotations": data.get("annotations") or job.params.get("annotations") or {},
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
            "annotations.source": scene_report["annotation_source"],
            "mlmarkup.commit": (scene_report.get("annotations") or {}).get("commit"),
            "mlmarkup.repo_path": (scene_report.get("annotations") or {}).get("repo_path"),
            "pseudolabeling.enabled": False,
        }
    )
    log_fn(job_log, f"real_train matched={len(matches)} missing={len(missing)} ambiguous={len(ambiguous)}")

    input_bands = job.params.get("input_bands") or model_cfg.get("input_bands") or [1, 2, 3, 4]
    input_bands = [int(band) for band in input_bands]
    patch_size = int(job.train.get("patch_size") or job.train.get("train_patch_size") or job.preprocess.get("tile_size") or 256)
    max_train_tiles, max_val_tiles, max_tiles_per_scene = _default_tile_limits(job)
    explicit_max_train_tiles = _explicit_tile_total_limit(job, "max_train_tiles")
    explicit_max_val_tiles = _explicit_tile_total_limit(job, "max_val_tiles")
    empty_share = float(job.preprocess.get("max_empty_tile_share") or 0.5)
    with trace_stage("prepare_dataset", {"job_id": job.job_id, "scene_count": len(matches), "tile_size": patch_size}):
        shapes = _load_shapes(config, annotation_uri)

    prepared_split = _load_prepared_dataset_split(experiment_dir, matches, job)
    if prepared_split is not None:
        train_matches, val_matches, prepared_split_metadata = prepared_split
        log_fn(
            job_log,
            "real_train using prepared dataset manifest "
            f"train={len(train_matches)} val={len(val_matches)} source={prepared_split_metadata.get('source')}",
        )
    else:
        train_matches, val_matches = _split_train_val_matches(
            config,
            matches,
            shapes,
            train_fraction=float(job.preprocess.get("train_fraction") or 0.75),
            seed=seed,
            stratify_positive=bool(job.preprocess.get("stratify_positive_validation", True)),
        )
        prepared_split_metadata = None
    if _full_dataset_tiles_requested(job):
        if explicit_max_train_tiles is None:
            max_train_tiles = max(1, len(train_matches) * max_tiles_per_scene)
        if explicit_max_val_tiles is None:
            max_val_tiles = max(1, len(val_matches) * max_tiles_per_scene)
    annotation_geojson_path = _write_training_annotation_geojson(config, annotation_uri, experiment_dir)
    train_sampling_enabled = resolve_train_sampling_enabled(job.preprocess)
    base_stride = int(job.preprocess.get("stride") or job.preprocess.get("train_stride") or patch_size)
    train_sampling_cfg = job.preprocess.get("train_sampling") if isinstance(job.preprocess.get("train_sampling"), dict) else {}
    augmentation_level = int(
        train_sampling_cfg.get(
            "augmentation_level",
            job.preprocess.get("augmentation_level", 2 if train_sampling_enabled else 0),
        )
    )
    train_scenes = _scene_inputs_for_matches(config, train_matches)
    val_scenes = _scene_inputs_for_matches(config, val_matches)
    tile_bundle = TilePreparationFacade.build_datasets(
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        annotation_path=annotation_geojson_path,
        tile_size=patch_size,
        stride=base_stride,
        augmentation_level=augmentation_level,
    )
    train_dataset = tile_bundle.train_dataset
    val_dataset = tile_bundle.val_dataset
    train_tile_config = train_dataset.config
    val_tile_config = val_dataset.config
    if len(val_dataset) == 0 and len(train_dataset) > 0 and bool(job.train.get("allow_train_val_sample_fallback", False)):
        fallback_count = max(1, min(max_val_tiles, max(1, len(train_dataset.base_records) // 4)))
        fallback_records = list(train_dataset.base_records[-fallback_count:])
        val_dataset.close()
        val_dataset = TrainingTileDataset(train_scenes, train_dataset.annotation, val_tile_config, records=fallback_records, train=False, build_result=train_dataset.build_result)
        val_dataset.warnings.append("validation fallback used base train tile records; use a real val scene split for final runs")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Not enough samples: train={len(train_dataset)} val={len(val_dataset)}")
    dataset_input_channels = int(train_dataset[0].image.shape[0])
    input_bands = list(range(1, dataset_input_channels + 1))

    train_report = train_dataset.scene_reports
    val_report = val_dataset.scene_reports
    train_sample_records = train_dataset.sample_records()
    val_sample_records = val_dataset.sample_records()
    train_sampling_warnings: list[str] = list(train_dataset.warnings) + list(val_dataset.warnings)
    train_record_summary = summarize_tile_records(train_dataset.records, prefix="train")
    base_summary = summarize_tile_records(train_dataset.base_records, prefix="base_train")
    val_record_summary = summarize_tile_records(val_dataset.records, prefix="val")
    train_sampling_summary: dict[str, Any] = {
        "train_sampling_enabled": train_sampling_enabled,
        "tile_preparation_module": "mlsystem.src.tile_preparation",
        **base_summary,
        **train_record_summary,
        "base_train_tile_count": len(train_dataset.base_records),
        "virtual_train_tile_count": len(train_dataset.records),
        "effective_train_samples_per_epoch": len(train_dataset),
        "val_partial_positive_tiles": val_record_summary["val_partial_positive_tiles"],
        "val_hard_negative_tiles": val_record_summary["val_hard_negative_tiles"],
        "positive_stride": train_tile_config.positive_stride,
        "hard_negative_stride": train_tile_config.hard_negative_stride,
        "negative_stride": train_tile_config.negative_stride,
        "virtual_epoch_multiplier": train_tile_config.virtual_epoch_multiplier,
        "positive_repeat_factor": train_tile_config.positive_repeat_factor,
        "hard_negative_repeat_factor": train_tile_config.hard_negative_repeat_factor,
        "negative_repeat_factor": train_tile_config.negative_repeat_factor,
        "max_empty_tile_share": train_tile_config.max_empty_tile_share,
        "batch_positive_fraction": train_tile_config.batch_positive_fraction,
        "batch_hard_negative_fraction": train_tile_config.batch_hard_negative_fraction,
        "batch_negative_fraction": train_tile_config.batch_negative_fraction,
        "warnings": train_sampling_warnings,
    }
    log_fn(
        job_log,
        "real_train tile_preparation "
        f"base={len(train_dataset.base_records)} train={len(train_dataset)} "
        f"val={len(val_dataset)} warnings={len(train_sampling_warnings)}",
    )

    train_positive_scene_count = sum(1 for row in train_report if row.get("positive_scene"))
    val_positive_scene_count = sum(1 for row in val_report if row.get("positive_scene"))
    train_negative_scene_count = max(0, len(train_report) - train_positive_scene_count)
    val_negative_scene_count = max(0, len(val_report) - val_positive_scene_count)
    train_positive_tiles = train_record_summary["train_positive_tiles"] + train_record_summary["train_partial_positive_tiles"]
    train_negative_tiles = train_record_summary["train_negative_tiles"] + train_record_summary["train_hard_negative_tiles"]
    train_partial_positive_tiles = train_record_summary["train_partial_positive_tiles"]
    train_hard_negative_tiles = train_record_summary["train_hard_negative_tiles"]
    val_positive_tiles = val_record_summary["val_positive_tiles"] + val_record_summary["val_partial_positive_tiles"]
    val_negative_tiles = val_record_summary["val_negative_tiles"] + val_record_summary["val_hard_negative_tiles"]
    dataset_report = {
        "annotation_source": scene_report["annotation_source"],
        "annotations": scene_report.get("annotations") or {},
        "train_scene_count": len(train_matches),
        "val_scene_count": len(val_matches),
        "positive_scene_count": train_positive_scene_count + val_positive_scene_count,
        "negative_scene_count": train_negative_scene_count + val_negative_scene_count,
        "train_positive_scene_count": train_positive_scene_count,
        "train_negative_scene_count": train_negative_scene_count,
        "val_positive_scene_count": val_positive_scene_count,
        "val_negative_scene_count": val_negative_scene_count,
        "train_tile_count": len(train_dataset),
        "base_train_tile_count": train_sampling_summary["base_train_tile_count"],
        "virtual_train_tile_count": train_sampling_summary["virtual_train_tile_count"],
        "effective_train_samples_per_epoch": train_sampling_summary["effective_train_samples_per_epoch"],
        "val_tile_count": len(val_dataset),
        "train_positive_tiles": train_positive_tiles,
        "train_partial_positive_tiles": train_partial_positive_tiles,
        "train_hard_negative_tiles": train_hard_negative_tiles,
        "val_positive_tiles": val_positive_tiles,
        "train_negative_tiles": train_negative_tiles,
        "val_negative_tiles": val_negative_tiles,
        "positive_tile_count": train_positive_tiles + val_positive_tiles,
        "negative_tile_count": train_negative_tiles + val_negative_tiles,
        "patch_size": patch_size,
        "train_scenes": train_report,
        "val_scenes": val_report,
        "train_sample_records": train_sample_records,
        "val_sample_records": val_sample_records,
        "prepared_dataset_manifest": prepared_split_metadata,
        "train_sampling": train_sampling_summary,
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
            "train_sampling_enabled": train_sampling_enabled,
            "base_train_tile_count": dataset_report["base_train_tile_count"],
            "virtual_train_tile_count": dataset_report["virtual_train_tile_count"],
            "effective_train_samples_per_epoch": dataset_report["effective_train_samples_per_epoch"],
            "train_positive_tiles": dataset_report["train_positive_tiles"],
            "train_partial_positive_tiles": dataset_report["train_partial_positive_tiles"],
            "train_hard_negative_tiles": dataset_report["train_hard_negative_tiles"],
            "train_negative_tiles": dataset_report["train_negative_tiles"],
            "val_tile_count": dataset_report["val_tile_count"],
            "val_positive_tiles": dataset_report["val_positive_tiles"],
            "val_negative_tiles": dataset_report["val_negative_tiles"],
            "positive_stride": train_sampling_summary.get("positive_stride"),
            "hard_negative_stride": train_sampling_summary.get("hard_negative_stride"),
            "negative_stride": train_sampling_summary.get("negative_stride"),
            "virtual_epoch_multiplier": train_tile_config.virtual_epoch_multiplier,
            "positive_repeat_factor": train_tile_config.positive_repeat_factor,
            "hard_negative_repeat_factor": train_tile_config.hard_negative_repeat_factor,
            "negative_repeat_factor": train_tile_config.negative_repeat_factor,
            "max_empty_tile_share": train_tile_config.max_empty_tile_share,
            "batch_positive_fraction": train_tile_config.batch_positive_fraction,
            "batch_hard_negative_fraction": train_tile_config.batch_hard_negative_fraction,
            "batch_negative_fraction": train_tile_config.batch_negative_fraction,
            "train_sampling_warnings": "; ".join(train_sampling_warnings[:20]),
        }
    )
    prepare_duration_sec = round(time.time() - prepare_started, 3)

    require_gpu = bool(job.train.get("require_gpu", False) or job.resources.requires_gpu)
    cuda_available = bool(torch.cuda.is_available())
    if require_gpu and not cuda_available:
        raise RuntimeError(
            "GPU training was requested, but CUDA is not available "
            f"(config.cpu_only={config.cpu_only}, torch.cuda.is_available={cuda_available})."
        )
    cpu_only_effective = bool(config.cpu_only) and not require_gpu
    device = torch.device("cuda" if cuda_available and not cpu_only_effective else "cpu")
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
    initial_checkpoint_path = (
        job.train.get("initial_checkpoint_path")
        or job.train.get("checkpoint_path")
        or job.params.get("initial_checkpoint_path")
        or job.params.get("train.initial_checkpoint_path")
    )
    initial_checkpoint_info: dict[str, Any] | None = None
    if initial_checkpoint_path:
        initial_checkpoint_info = _load_initial_checkpoint(
            model,
            str(initial_checkpoint_path),
            device=device,
            strict=bool(job.train.get("initial_checkpoint_strict", True)),
        )
        mlflow_run.log_params(
            {
                "train.initial_checkpoint_path": initial_checkpoint_info["path"],
                "train.initial_checkpoint_strict": initial_checkpoint_info["strict"],
                "train.initial_checkpoint_missing_keys": len(initial_checkpoint_info["missing_keys"]),
                "train.initial_checkpoint_unexpected_keys": len(initial_checkpoint_info["unexpected_keys"]),
            }
        )
        log_fn(job_log, f"real_train loaded_initial_checkpoint={initial_checkpoint_info['path']}")
    freeze_batchnorm_default = model_name.lower().startswith(("deeplab", "deeplabv3plus"))
    freeze_batchnorm = bool(job.train.get("freeze_batchnorm", freeze_batchnorm_default))
    dropout_updated = _configure_dropout(model, job.train.get("dropout_p", job.train.get("dropout")))
    optimizer = _build_optimizer(model, job.train)
    batch_size = job.train.get("batch_size") or 2
    if batch_size == "auto":
        batch_size = _auto_batch_size(model_name, patch_size, device)
    batch_size = max(1, int(batch_size))
    epochs = int(job.train.get("epochs") or job.train.get("max_epochs") or 20)
    train_mode = str(job.train.get("mode") or job.train.get("train_mode") or "train").strip().lower()
    eval_only = train_mode in {"eval_only", "evaluate_only", "checkpoint_reeval", "reeval"}
    if eval_only:
        if not initial_checkpoint_path:
            raise RuntimeError("train.mode=eval_only requires train.initial_checkpoint_path")
        epochs = 0
    scheduler = _build_scheduler(optimizer, job.train, epochs)
    loss_cfg = job.train.get("loss") if isinstance(job.train.get("loss"), dict) else {}
    if isinstance(job.train.get("loss"), str):
        loss_cfg = {"name": job.train.get("loss")}
    loss_name = str((loss_cfg or {}).get("name") or (loss_cfg or {}).get("type") or "bce_dice")
    grad_clip_norm = job.train.get("grad_clip_norm")
    objective_metric = str(job.train.get("objective_metric") or job.params.get("objective_metric") or "val/iou")
    maximize_objective = bool(job.train.get("maximize", job.params.get("maximize", True)))
    learning_rate_value = job.train.get("learning_rate")
    weight_decay_value = job.train.get("weight_decay")
    mlflow_run.log_params(
        {
            "freeze_batchnorm": freeze_batchnorm,
            "train.dropout_p": job.train.get("dropout_p", job.train.get("dropout")),
            "train.dropout_modules_updated": dropout_updated,
            "train.optimizer": str(job.train.get("optimizer") or job.train.get("optimizer_name") or "adamw"),
            "train.learning_rate": float(5e-4 if learning_rate_value in (None, "") else learning_rate_value),
            "train.weight_decay": float(0.0 if weight_decay_value in (None, "") else weight_decay_value),
            "train.scheduler": str((job.train.get("scheduler") or {}).get("name") if isinstance(job.train.get("scheduler"), dict) else (job.train.get("scheduler") or job.train.get("scheduler_name") or "none")),
            "train.loss": loss_name,
            "train.mode": train_mode,
            "train.eval_only": eval_only,
            "train.loss_bce_weight": (loss_cfg or {}).get("bce_weight"),
            "train.loss_dice_weight": (loss_cfg or {}).get("dice_weight"),
            "train.loss_focal_weight": (loss_cfg or {}).get("focal_weight"),
            "train.loss_tversky_weight": (loss_cfg or {}).get("tversky_weight"),
            "train.loss_pos_weight": (loss_cfg or {}).get("pos_weight", (loss_cfg or {}).get("positive_weight")),
            "train.grad_clip_norm": grad_clip_norm,
            "train.objective_metric": objective_metric,
            "train.objective_maximize": maximize_objective,
            "train.batch_size_resolved": batch_size,
            "train.max_train_tiles": max_train_tiles,
            "train.max_val_tiles": max_val_tiles,
            "train.max_tiles_per_scene": max_tiles_per_scene,
        }
    )
    augmentations_cfg = job.train.get("augmentations") or {}
    mlflow_run.log_params({"train.augmentations_profile": _augmentation_profile(augmentations_cfg)})
    wallclock_limit_sec = _resolve_wallclock_limit(job)
    time_limit_sec = int(wallclock_limit_sec or 0)
    early_cfg = job.train.get("early_stopping") or {}
    early_enabled = bool(early_cfg.get("enabled", job.train.get("early_stopping_enabled", False)))
    early_patience = int(early_cfg.get("patience") or job.train.get("early_stopping_patience") or 10)
    train_started = time.time()
    started = train_started
    history: list[dict[str, float]] = []
    metric_threshold = float(
        job.train.get("metric_threshold")
        or job.evaluate.get("threshold")
        or job.params.get("metric_threshold")
        or job.params.get("threshold")
        or 0.5
    )
    metric_thresholds = _resolve_metric_thresholds(job, metric_threshold)
    metric_class_name = str(job.params.get("class_name") or job.train.get("class_name") or "deforest")
    metric_class_key = _metric_class_key(metric_class_name)
    metrics_debug_cfg = metrics_debug_config(job)
    debug_enabled = metrics_debug_enabled(metrics_debug_cfg)
    metrics_debug_root = experiment_dir / "metrics_debug"
    mlflow_run.log_params(
        {
            "metrics.threshold": metric_threshold,
            "metrics.threshold_sweep": ",".join(f"{item:.4f}".rstrip("0").rstrip(".") for item in metric_thresholds),
            "metrics.aggregation": "micro_global_pixel_counts",
            "metrics.source_of_truth": "mlsystem.src.metrics.segmentation",
            "metrics.val_shuffle": False,
            "metrics.val_augmentations": "none",
            "metrics.debug_enabled": debug_enabled,
            "metrics.debug_class_name": metrics_debug_cfg.get("class_name"),
            "train.max_wallclock_seconds": wallclock_limit_sec,
        }
    )

    if bool(job.train.get("cache_samples_on_gpu", False)):
        log_fn(job_log, "real_train ignored cache_samples_on_gpu because tile_preparation reads tiles lazily")
    mlflow_run.log_params(
        {
            "train.cache_samples_on_gpu": False,
            "train.dataset_adapter": "mlsystem.src.tile_preparation.TrainingTileDataset",
            "train.dataset_lazy_raster_reads": True,
        }
    )

    best_val_iou = -1.0
    best_objective_value = -math.inf if maximize_objective else math.inf
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    metrics_debug_artifacts: list[Path] = []
    train_trace = trace_stage("train_model", {"job_id": job.job_id, "model_name": model_name, "tile_size": patch_size, "epoch_count": epochs})
    train_trace.__enter__()
    try:
        for epoch in range(1, epochs + 1):
            if history and wallclock_limit_sec is not None and time.time() - started > wallclock_limit_sec:
                log_fn(job_log, f"real_train wallclock_stop before_epoch={epoch} limit_sec={wallclock_limit_sec}")
                break
            epoch_started = time.time()
            model.train()
            if freeze_batchnorm:
                _set_batchnorm_eval(model)
            train_loss_acc = WeightedLossAccumulator()
            train_metric_acc = PixelMetricAccumulator(threshold=metric_threshold)
            train_dataset.set_epoch(epoch)
            train_batch_count = 0
            for _batch_indices, x_cpu, y_cpu in iter_dataset_batches(
                train_dataset,
                batch_size,
                shuffle=True,
                seed=seed + epoch,
            ):
                train_batch_count += 1
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                components = _loss_components(logits, y, loss_cfg)
                loss = components["loss_total"]
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                optimizer.step()
                train_loss_acc.update({name: float(value.detach().item()) for name, value in components.items()}, weight=int(y.shape[0]))
                train_metric_acc.update_from_logits(logits.detach(), y)

            model.eval()
            val_loss_acc = WeightedLossAccumulator()
            val_metric_acc = PixelMetricAccumulator(threshold=metric_threshold)
            val_threshold_accs = {threshold: PixelMetricAccumulator(threshold=threshold) for threshold in metric_thresholds}
            val_sample_rows: list[dict[str, Any]] = []
            val_sample_payloads: list[dict[str, Any]] = []
            val_batch_count = 0
            with torch.no_grad():
                val_dataset.set_epoch(0)
                for batch_indices, x_cpu, y_cpu in iter_dataset_batches(
                    val_dataset,
                    batch_size,
                    shuffle=False,
                    seed=seed,
                ):
                    val_batch_count += 1
                    x = x_cpu.to(device)
                    y = y_cpu.to(device)
                    logits = model(x)
                    components = _loss_components(logits, y, loss_cfg)
                    val_loss_acc.update({name: float(value.detach().item()) for name, value in components.items()}, weight=int(y.shape[0]))
                    val_metric_acc.update_from_logits(logits, y)
                    for accumulator in val_threshold_accs.values():
                        accumulator.update_from_logits(logits, y)
                    if debug_enabled:
                        _collect_val_debug_samples(
                            batch_indices=batch_indices,
                            x=x,
                            y=y,
                            logits=logits,
                            val_sample_records=val_sample_records,
                            threshold=metric_threshold,
                            rows=val_sample_rows,
                            payloads=val_sample_payloads,
                            save_all=bool(metrics_debug_cfg.get("save_all_val_samples", True)),
                        )

            train_loss_values = train_loss_acc.averages("train")
            val_loss_values = val_loss_acc.averages("val")
            train_metrics = train_metric_acc.metrics()
            val_metrics = val_metric_acc.metrics()
            threshold_metrics = {threshold: accumulator.metrics() for threshold, accumulator in val_threshold_accs.items()}
            best_threshold, best_threshold_metrics = max(
                threshold_metrics.items(),
                key=lambda item: (float(item[1]["pixel_f1"]), -abs(float(item[0]) - metric_threshold)),
            )
            val_object_metrics = _object_summary_from_sample_rows(val_sample_rows) if debug_enabled else {}
            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_duration_sec = round(time.time() - epoch_started, 4)

            row = {
                "epoch": float(epoch),
                "train/loss": train_loss_values.get("train/loss_total", 0.0),
                "train/loss_total": train_loss_values.get("train/loss_total", 0.0),
                "train/loss_bce": train_loss_values.get("train/loss_bce", 0.0),
                "train/loss_dice": train_loss_values.get("train/loss_dice", 0.0),
                "train/loss_focal": train_loss_values.get("train/loss_focal", 0.0),
                "train/loss_tversky": train_loss_values.get("train/loss_tversky", 0.0),
                "train/dice": float(train_metrics["pixel_f1"]),
                "train/iou": float(train_metrics["pixel_iou"]),
                "train/pixel_f1": float(train_metrics["pixel_f1"]),
                "train/pixel_iou": float(train_metrics["pixel_iou"]),
                "train/precision": float(train_metrics["pixel_precision"]),
                "train/recall": float(train_metrics["pixel_recall"]),
                "train/pixel_tp": float(train_metrics["pixel_tp"]),
                "train/pixel_fp": float(train_metrics["pixel_fp"]),
                "train/pixel_fn": float(train_metrics["pixel_fn"]),
                "train/pixel_tn": float(train_metrics["pixel_tn"]),
                "val/loss": val_loss_values.get("val/loss_total", 0.0),
                "val/loss_total": val_loss_values.get("val/loss_total", 0.0),
                "val/loss_bce": val_loss_values.get("val/loss_bce", 0.0),
                "val/loss_dice": val_loss_values.get("val/loss_dice", 0.0),
                "val/loss_focal": val_loss_values.get("val/loss_focal", 0.0),
                "val/loss_tversky": val_loss_values.get("val/loss_tversky", 0.0),
                "val/dice": float(val_metrics["pixel_f1"]),
                "val/iou": float(val_metrics["pixel_iou"]),
                "val/pixel_dice": float(val_metrics["pixel_f1"]),
                "val/pixel_iou": float(val_metrics["pixel_iou"]),
                "val/precision": float(val_metrics["pixel_precision"]),
                "val/recall": float(val_metrics["pixel_recall"]),
                "val/pixel_f1": float(val_metrics["pixel_f1"]),
                "val/pixel_accuracy": float(val_metrics["pixel_accuracy"]),
                "val/pixel_tp": float(val_metrics["pixel_tp"]),
                "val/pixel_fp": float(val_metrics["pixel_fp"]),
                "val/pixel_fn": float(val_metrics["pixel_fn"]),
                "val/pixel_tn": float(val_metrics["pixel_tn"]),
                "val/threshold": metric_threshold,
                f"val/{metric_class_key}_pixel_f1": float(val_metrics["pixel_f1"]),
                f"val/{metric_class_key}_pixel_precision": float(val_metrics["pixel_precision"]),
                f"val/{metric_class_key}_pixel_recall": float(val_metrics["pixel_recall"]),
                f"val/{metric_class_key}_pixel_iou": float(val_metrics["pixel_iou"]),
                f"val/{metric_class_key}_gt_pixels": float(int(val_metrics["pixel_tp"]) + int(val_metrics["pixel_fn"])),
                f"val/{metric_class_key}_pred_pixels": float(int(val_metrics["pixel_tp"]) + int(val_metrics["pixel_fp"])),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_duration_sec": epoch_duration_sec,
                "train/epoch_duration_sec": epoch_duration_sec,
                "train/batches": float(train_batch_count),
                "val/batches": float(val_batch_count),
            }
            row.update(
                {
                    "val/best_threshold": float(best_threshold),
                    "val/pixel_f1_best_threshold": float(best_threshold_metrics["pixel_f1"]),
                    "val/pixel_iou_best_threshold": float(best_threshold_metrics["pixel_iou"]),
                    "val/precision_best_threshold": float(best_threshold_metrics["pixel_precision"]),
                    "val/recall_best_threshold": float(best_threshold_metrics["pixel_recall"]),
                }
            )
            for threshold, metrics_payload in threshold_metrics.items():
                suffix = _threshold_metric_suffix(threshold)
                row.update(
                    {
                        f"val/pixel_f1_at_threshold_{suffix}": float(metrics_payload["pixel_f1"]),
                        f"val/pixel_iou_at_threshold_{suffix}": float(metrics_payload["pixel_iou"]),
                        f"val/precision_at_threshold_{suffix}": float(metrics_payload["pixel_precision"]),
                        f"val/recall_at_threshold_{suffix}": float(metrics_payload["pixel_recall"]),
                    }
                )
            if val_object_metrics:
                row.update(
                    {
                        "val/object_tp": val_object_metrics["object_tp"],
                        "val/object_fp": val_object_metrics["object_fp"],
                        "val/object_fn": val_object_metrics["object_fn"],
                        "val/object_precision": val_object_metrics["object_precision"],
                        "val/object_recall": val_object_metrics["object_recall"],
                        "val/object_f1": val_object_metrics["object_f1"],
                        f"val/{metric_class_key}_object_precision": val_object_metrics["object_precision"],
                        f"val/{metric_class_key}_object_recall": val_object_metrics["object_recall"],
                        f"val/{metric_class_key}_object_f1": val_object_metrics["object_f1"],
                        f"val/{metric_class_key}_gt_objects": val_object_metrics["gt_objects"],
                        f"val/{metric_class_key}_pred_objects": val_object_metrics["pred_objects"],
                    }
                )
            row.update(
                {
                    "epoch/pixel_f1": row["val/pixel_f1"],
                    "epoch/pixel_iou": row["val/pixel_iou"],
                    "epoch/sec": row["epoch_duration_sec"],
                }
            )
            objective_value = float(row.get(objective_metric, row.get("val/iou", 0.0)))
            row["objective/value"] = objective_value
            if device.type == "cuda":
                torch.cuda.synchronize()
                row["system/cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
                row["system/cuda_memory_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 * 1024), 3)
                row["system/gpu_train_confirmed"] = 1.0
            history.append(row)
            logged_metrics = {key: value for key, value in row.items() if key != "epoch"}
            mlflow_run.log_metrics(logged_metrics, step=epoch)
            if debug_enabled:
                debug_result = write_epoch_debug(
                    root_dir=metrics_debug_root,
                    run_id=job.job_id,
                    epoch=epoch,
                    mlflow_run_id=mlflow_run.run_id,
                    model_checkpoint_path=None,
                    val_manifest_path=str(dataset_report_path),
                    val_manifest=val_sample_records,
                    class_name=str(metrics_debug_cfg.get("class_name") or metric_class_name),
                    class_id=metrics_debug_cfg.get("class_id"),
                    threshold=metric_threshold,
                    metric_row=row,
                    train_loss=train_loss_values,
                    val_loss=val_loss_values,
                    per_sample_metrics=val_sample_rows,
                    sample_payloads=val_sample_payloads,
                    logged_metrics=logged_metrics,
                )
                if not debug_result.get("recompute", {}).get("ok", False):
                    log_fn(job_log, f"real_train metrics_debug recompute mismatch epoch={epoch} path={debug_result.get('metrics_recompute_check')}")
                for artifact_key in (
                    "epoch_summary",
                    "production_metrics_snapshot",
                    "per_sample_metrics",
                    "val_manifest_snapshot",
                    "metrics_recompute_check",
                    "mlflow_logged_metrics",
                    "metrics_md",
                    "artifacts_manifest",
                ):
                    artifact_path = debug_result.get(artifact_key)
                    if artifact_path:
                        metrics_debug_artifacts.append(Path(str(artifact_path)))
            log_fn(job_log, f"real_train epoch={epoch} val_iou={row['val/iou']:.6f} duration={row['epoch_duration_sec']}")
            if scheduler is not None:
                scheduler.step()
            if row["val/iou"] > best_val_iou:
                best_val_iou = row["val/iou"]
            if _metric_improved(objective_value, best_objective_value, maximize=maximize_objective):
                best_objective_value = objective_value
                best_epoch = epoch
                best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if early_enabled and epochs_without_improvement >= early_patience:
                log_fn(job_log, f"real_train early_stopping epoch={epoch} patience={early_patience}")
                break
            if wallclock_limit_sec is not None and time.time() - started > wallclock_limit_sec:
                log_fn(job_log, f"real_train wallclock_stop after_epoch={epoch} limit_sec={wallclock_limit_sec}")
                break
    finally:
        train_trace.__exit__(None, None, None)

    if eval_only:
        eval_started = time.time()
        model.eval()
        val_loss_acc = WeightedLossAccumulator()
        val_metric_acc = PixelMetricAccumulator(threshold=metric_threshold)
        val_threshold_accs = {threshold: PixelMetricAccumulator(threshold=threshold) for threshold in metric_thresholds}
        val_sample_rows: list[dict[str, Any]] = []
        val_sample_payloads: list[dict[str, Any]] = []
        val_batch_count = 0
        with torch.no_grad():
            val_dataset.set_epoch(0)
            for batch_indices, x_cpu, y_cpu in iter_dataset_batches(
                val_dataset,
                batch_size,
                shuffle=False,
                seed=seed,
            ):
                val_batch_count += 1
                x = x_cpu.to(device)
                y = y_cpu.to(device)
                logits = model(x)
                components = _loss_components(logits, y, loss_cfg)
                val_loss_acc.update({name: float(value.detach().item()) for name, value in components.items()}, weight=int(y.shape[0]))
                val_metric_acc.update_from_logits(logits, y)
                for accumulator in val_threshold_accs.values():
                    accumulator.update_from_logits(logits, y)
                if debug_enabled:
                    _collect_val_debug_samples(
                        batch_indices=batch_indices,
                        x=x,
                        y=y,
                        logits=logits,
                        val_sample_records=val_sample_records,
                        threshold=metric_threshold,
                        rows=val_sample_rows,
                        payloads=val_sample_payloads,
                        save_all=bool(metrics_debug_cfg.get("save_all_val_samples", True)),
                    )
        val_loss_values = val_loss_acc.averages("val")
        val_metrics = val_metric_acc.metrics()
        threshold_metrics = {threshold: accumulator.metrics() for threshold, accumulator in val_threshold_accs.items()}
        best_threshold, best_threshold_metrics = max(
            threshold_metrics.items(),
            key=lambda item: (float(item[1]["pixel_f1"]), -abs(float(item[0]) - metric_threshold)),
        )
        val_object_metrics = _object_summary_from_sample_rows(val_sample_rows) if debug_enabled else {}
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_duration_sec = round(time.time() - eval_started, 4)
        row = {
            "epoch": 0.0,
            "train/loss": 0.0,
            "train/loss_total": 0.0,
            "train/loss_bce": 0.0,
            "train/loss_dice": 0.0,
            "train/loss_focal": 0.0,
            "train/loss_tversky": 0.0,
            "train/dice": 0.0,
            "train/iou": 0.0,
            "train/pixel_f1": 0.0,
            "train/pixel_iou": 0.0,
            "train/precision": 0.0,
            "train/recall": 0.0,
            "train/pixel_tp": 0.0,
            "train/pixel_fp": 0.0,
            "train/pixel_fn": 0.0,
            "train/pixel_tn": 0.0,
            "val/loss": val_loss_values.get("val/loss_total", 0.0),
            "val/loss_total": val_loss_values.get("val/loss_total", 0.0),
            "val/loss_bce": val_loss_values.get("val/loss_bce", 0.0),
            "val/loss_dice": val_loss_values.get("val/loss_dice", 0.0),
            "val/loss_focal": val_loss_values.get("val/loss_focal", 0.0),
            "val/loss_tversky": val_loss_values.get("val/loss_tversky", 0.0),
            "val/dice": float(val_metrics["pixel_f1"]),
            "val/iou": float(val_metrics["pixel_iou"]),
            "val/pixel_dice": float(val_metrics["pixel_f1"]),
            "val/pixel_iou": float(val_metrics["pixel_iou"]),
            "val/precision": float(val_metrics["pixel_precision"]),
            "val/recall": float(val_metrics["pixel_recall"]),
            "val/pixel_f1": float(val_metrics["pixel_f1"]),
            "val/pixel_accuracy": float(val_metrics["pixel_accuracy"]),
            "val/pixel_tp": float(val_metrics["pixel_tp"]),
            "val/pixel_fp": float(val_metrics["pixel_fp"]),
            "val/pixel_fn": float(val_metrics["pixel_fn"]),
            "val/pixel_tn": float(val_metrics["pixel_tn"]),
            "val/threshold": metric_threshold,
            f"val/{metric_class_key}_pixel_f1": float(val_metrics["pixel_f1"]),
            f"val/{metric_class_key}_pixel_precision": float(val_metrics["pixel_precision"]),
            f"val/{metric_class_key}_pixel_recall": float(val_metrics["pixel_recall"]),
            f"val/{metric_class_key}_pixel_iou": float(val_metrics["pixel_iou"]),
            f"val/{metric_class_key}_gt_pixels": float(int(val_metrics["pixel_tp"]) + int(val_metrics["pixel_fn"])),
            f"val/{metric_class_key}_pred_pixels": float(int(val_metrics["pixel_tp"]) + int(val_metrics["pixel_fp"])),
            "learning_rate": 0.0,
            "epoch_duration_sec": eval_duration_sec,
            "train/epoch_duration_sec": 0.0,
            "train/batches": 0.0,
            "val/batches": float(val_batch_count),
            "train/optimizer_steps": 0.0,
        }
        row.update(
            {
                "val/best_threshold": float(best_threshold),
                "val/pixel_f1_best_threshold": float(best_threshold_metrics["pixel_f1"]),
                "val/pixel_iou_best_threshold": float(best_threshold_metrics["pixel_iou"]),
                "val/precision_best_threshold": float(best_threshold_metrics["pixel_precision"]),
                "val/recall_best_threshold": float(best_threshold_metrics["pixel_recall"]),
            }
        )
        for threshold, metrics_payload in threshold_metrics.items():
            suffix = _threshold_metric_suffix(threshold)
            row.update(
                {
                    f"val/pixel_f1_at_threshold_{suffix}": float(metrics_payload["pixel_f1"]),
                    f"val/pixel_iou_at_threshold_{suffix}": float(metrics_payload["pixel_iou"]),
                    f"val/precision_at_threshold_{suffix}": float(metrics_payload["pixel_precision"]),
                    f"val/recall_at_threshold_{suffix}": float(metrics_payload["pixel_recall"]),
                }
            )
        if val_object_metrics:
            row.update(
                {
                    "val/object_tp": val_object_metrics["object_tp"],
                    "val/object_fp": val_object_metrics["object_fp"],
                    "val/object_fn": val_object_metrics["object_fn"],
                    "val/object_precision": val_object_metrics["object_precision"],
                    "val/object_recall": val_object_metrics["object_recall"],
                    "val/object_f1": val_object_metrics["object_f1"],
                    f"val/{metric_class_key}_object_precision": val_object_metrics["object_precision"],
                    f"val/{metric_class_key}_object_recall": val_object_metrics["object_recall"],
                    f"val/{metric_class_key}_object_f1": val_object_metrics["object_f1"],
                    f"val/{metric_class_key}_gt_objects": val_object_metrics["gt_objects"],
                    f"val/{metric_class_key}_pred_objects": val_object_metrics["pred_objects"],
                }
            )
        row.update(
            {
                "epoch/pixel_f1": row["val/pixel_f1"],
                "epoch/pixel_iou": row["val/pixel_iou"],
                "epoch/sec": row["epoch_duration_sec"],
            }
        )
        objective_value = float(row.get(objective_metric, row.get("val/iou", 0.0)))
        row["objective/value"] = objective_value
        if device.type == "cuda":
            row["system/cuda_memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024 * 1024), 3)
            row["system/cuda_memory_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024 * 1024), 3)
            row["system/gpu_train_confirmed"] = 1.0
        history.append(row)
        logged_metrics = {key: value for key, value in row.items() if key != "epoch"}
        mlflow_run.log_metrics(logged_metrics, step=0)
        best_val_iou = row["val/iou"]
        best_objective_value = objective_value
        best_epoch = 0
        best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        log_fn(job_log, f"real_train eval_only val_iou={row['val/iou']:.6f} duration={row['epoch_duration_sec']}")

    train_duration_sec = round(time.time() - train_started, 3)
    artifacts = _write_history(experiment_dir, history)
    checkpoint_path = experiment_dir / "tiny_unet_4ch.pt"
    checkpoint_path = experiment_dir / f"{model_name}.pt"
    state_to_save = best_state_dict or {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": state_to_save,
            "job_id": job.job_id,
            "model_name": model_name,
            "best_epoch": best_epoch,
            "best_objective_metric": objective_metric,
            "best_objective_value": best_objective_value,
            "final_epoch": len(history),
        },
        checkpoint_path,
    )
    metrics_debug_report: dict[str, Any] | None = None
    if debug_enabled and bool(metrics_debug_cfg.get("report_enabled", True)):
        pipeline_metadata = job.params.get("pipeline") if isinstance(job.params.get("pipeline"), dict) else {}
        report_root = Path(str(metrics_debug_cfg.get("report_root") or (experiment_dir / "metrics_debug_reports")))
        report_timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_name = str(
            metrics_debug_cfg.get("report_name")
            or f"metrics_debug_cuttings_pipeline_{pipeline_metadata.get('run_id') or job.job_id}_{report_timestamp}"
        )
        dataset_check = {
            "class_name": metrics_debug_cfg.get("class_name") or metric_class_name,
            "annotation_source": dataset_report.get("annotation_source"),
            "annotations": dataset_report.get("annotations") or {},
            "full_dataset": not any(
                value is not None
                for value in (
                    job.preprocess.get("max_scenes"),
                    job.preprocess.get("max_dataset_scenes"),
                    job.preprocess.get("dataset_limit"),
                    job.preprocess.get("scene_limit"),
                    job.preprocess.get("sample_size"),
                    job.train.get("max_train_batches"),
                    job.train.get("max_val_batches"),
                )
            )
            and len(train_matches) + len(val_matches) == len(matches),
            "synthetic": False,
            "train_limit_batches": job.train.get("max_train_batches"),
            "val_limit_batches": job.train.get("max_val_batches"),
            "train_sample_count": len(train_sample_records),
            "val_sample_count": len(val_sample_records),
            "train_manifest_hash": manifest_hash(train_sample_records),
            "val_manifest_hash": manifest_hash(val_sample_records),
            "matched_scenes_count": len(matches),
            "train_scene_count": len(train_matches),
            "val_scene_count": len(val_matches),
            "explicit_max_train_tiles": explicit_max_train_tiles,
            "explicit_max_val_tiles": explicit_max_val_tiles,
            "max_tiles_per_scene": max_tiles_per_scene,
            "train_sampling": train_sampling_summary,
        }
        run_metadata = {
            "branch": os.getenv("MLSYSTEM_GIT_BRANCH") or job.params.get("git_branch") or "",
            "commit": os.getenv("MLSYSTEM_COMMIT") or os.getenv("MLSYSTEM_GIT_COMMIT") or job.params.get("git_commit") or "",
            "pushed_remote": job.params.get("pushed_remote") or "",
            "pipeline": pipeline_metadata,
            "pipeline_url": pipeline_metadata.get("url") or "",
            "pipeline_trace": pipeline_metadata.get("trace") or {},
            "stage_statuses": pipeline_metadata.get("stage_statuses") or {},
            "mlflow_run_id": mlflow_run.run_id,
            "mlflow_url": (mlflow_run.result() or {}).get("run_url_external"),
            "debug_root": str(metrics_debug_root / job.job_id),
            "threshold": metric_threshold,
            "seed": seed,
            "max_wallclock_seconds": wallclock_limit_sec,
            "augmentations": _augmentation_profile(augmentations_cfg),
        }
        metrics_debug_report = write_metrics_debug_report(
            debug_root=metrics_debug_root / job.job_id,
            report_root=report_root,
            report_name=report_name,
            run_metadata=run_metadata,
            dataset_check=dataset_check,
        )
        artifacts.append(Path(str(metrics_debug_report["summary"])))
    artifacts.extend([dataset_report_path, train_scenes_path, val_scenes_path, checkpoint_path, *metrics_debug_artifacts])
    train_dataset.close()
    val_dataset.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    pseudolabel_run_on = str(pseudolabel_cfg.get("run_on") or "").lower()
    pseudolabel_matches = matches
    if pseudolabel_run_on in {"validation_scenes", "val_scenes", "validation"}:
        pseudolabel_matches = val_matches
    elif pseudolabel_run_on in {"train_scenes", "training_scenes", "train"}:
        pseudolabel_matches = train_matches
    elif pseudolabel_run_on in {"dataset_scenes_plus_extra", "dataset_and_extra_images", "dataset_plus_extra"}:
        extra_images_uri = pseudolabel_cfg.get("extra_images_uri") or pseudolabel_cfg.get("additional_images_uri")
        if extra_images_uri:
            extra_images = _list_s3_objects(config, str(extra_images_uri), suffixes=(".tif", ".tiff"))
            seen_keys = {match.key for match in pseudolabel_matches}
            for item in sorted(extra_images, key=lambda row: row["key"]):
                if item["key"] in seen_keys:
                    continue
                pseudolabel_matches.append(SceneMatch(entry=item["name"], key=item["key"], name=item["name"], score=1.0))
                seen_keys.add(item["key"])
            mlflow_run.log_params({"pseudolabel.extra_images_uri": str(extra_images_uri)})
    elif pseudolabel_run_on in {"all_available_images", "all_images"}:
        pseudolabel_images_uri = str(pseudolabel_cfg.get("images_uri") or images_uri)
        pseudolabel_images = _list_s3_objects(config, pseudolabel_images_uri, suffixes=(".tif", ".tiff"))
        pseudolabel_matches = [
            SceneMatch(entry=item["name"], key=item["key"], name=item["name"], score=1.0)
            for item in sorted(pseudolabel_images, key=lambda row: row["key"])
        ]
        mlflow_run.log_params({"pseudolabel.images_uri": pseudolabel_images_uri})
    max_pseudolabel_scenes = pseudolabel_cfg.get("max_scenes")
    if max_pseudolabel_scenes is not None:
        pseudolabel_matches = pseudolabel_matches[: max(1, int(max_pseudolabel_scenes))]
    object_metrics_prefix = (
        "pseudolabel"
        if pseudolabel_run_on in {"all_available_images", "all_images", "dataset_scenes_plus_extra", "dataset_and_extra_images", "dataset_plus_extra"}
        else "val"
    )
    metric_shapes = shapes
    if pseudolabel_run_on in {"validation_scenes", "val_scenes", "validation", "train_scenes", "training_scenes", "train"}:
        metric_shapes = _filter_shapes_to_matches(config, pseudolabel_matches, shapes)
    elif pseudolabel_run_on in {"dataset_scenes_plus_extra", "dataset_and_extra_images", "dataset_plus_extra"}:
        metric_shapes = None
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
            gt_shapes=metric_shapes,
            object_metrics_prefix=object_metrics_prefix,
        )
    if postprocess_metrics.get("pseudolabel_enabled"):
        mlflow_run.log_metrics(
            {
                key: value
                for key, value in postprocess_metrics.items()
                if isinstance(value, (int, float, bool))
            },
            step=len(history) if history else None,
        )
        artifacts.extend(postprocess_artifacts)
    mlflow_run.log_artifacts(artifacts)

    last = history[-1] if history else {}
    epoch_duration_values = sorted(float(row.get("epoch_duration_sec", 0.0)) for row in history if row.get("epoch_duration_sec") is not None)
    if epoch_duration_values:
        middle = len(epoch_duration_values) // 2
        if len(epoch_duration_values) % 2:
            median_epoch_duration_sec = epoch_duration_values[middle]
        else:
            median_epoch_duration_sec = (epoch_duration_values[middle - 1] + epoch_duration_values[middle]) / 2.0
        epoch_duration_stats = {
            "median_sec": round(median_epoch_duration_sec, 4),
            "min_sec": round(epoch_duration_values[0], 4),
            "max_sec": round(epoch_duration_values[-1], 4),
        }
        mlflow_run.log_metrics(
            {
                "train/epoch_duration_median_sec": epoch_duration_stats["median_sec"],
                "train/epoch_duration_min_sec": epoch_duration_stats["min_sec"],
                "train/epoch_duration_max_sec": epoch_duration_stats["max_sec"],
            },
            step=len(history),
        )
    else:
        epoch_duration_stats = {"median_sec": None, "min_sec": None, "max_sec": None}
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
        "mode": "checkpoint_reeval" if eval_only else "real_train",
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
        "max_wallclock_seconds": wallclock_limit_sec,
        "epoch_duration": epoch_duration_stats,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "best_objective_metric": objective_metric,
        "best_objective_value": best_objective_value,
        "best_val_pixel_f1": max((float(row.get("val/pixel_f1", 0.0)) for row in history), default=0.0),
        "best_val_pixel_f1_best_threshold": max((float(row.get("val/pixel_f1_best_threshold", 0.0)) for row in history), default=0.0),
        "metrics_source_of_truth": "micro_global_pixel_counts",
        "metrics_threshold": metric_threshold,
        "metrics_debug_enabled": debug_enabled,
        "metrics_debug_root": str(metrics_debug_root / job.job_id) if debug_enabled else None,
        "metrics_debug_report": metrics_debug_report,
        "checkpoint_path": str(checkpoint_path),
        "initial_checkpoint": initial_checkpoint_info,
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
        "train_tile_count": dataset_report["train_tile_count"],
        "base_train_tile_count": dataset_report["base_train_tile_count"],
        "virtual_train_tile_count": dataset_report["virtual_train_tile_count"],
        "effective_train_samples_per_epoch": dataset_report["effective_train_samples_per_epoch"],
        "val_tile_count": dataset_report["val_tile_count"],
        "train_positive_tiles": dataset_report["train_positive_tiles"],
        "train_partial_positive_tiles": dataset_report["train_partial_positive_tiles"],
        "train_hard_negative_tiles": dataset_report["train_hard_negative_tiles"],
        "val_positive_tiles": dataset_report["val_positive_tiles"],
        "train_negative_tiles": dataset_report["train_negative_tiles"],
        "val_negative_tiles": dataset_report["val_negative_tiles"],
        "train_sampling": train_sampling_summary,
        "positive_scene_count": dataset_report["positive_scene_count"],
        "negative_scene_count": dataset_report["negative_scene_count"],
        "positive_tile_count": dataset_report["positive_tile_count"],
        "negative_tile_count": dataset_report["negative_tile_count"],
        "scenes_match_report": str(scenes_report_path),
        "train_dataset_report": str(dataset_report_path),
        "history_path": str(experiment_dir / "history.json"),
        "history_csv_path": str(experiment_dir / "history.csv"),
        "warnings": ([f"{len(missing)} scenes from scenes.txt were not matched"] if missing else []) + train_sampling_warnings,
    }
