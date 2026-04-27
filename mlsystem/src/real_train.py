from __future__ import annotations

import csv
import gzip
import json
import math
import random
import re
import time
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
import geopandas as gpd
import numpy as np
import torch
from rasterio.features import rasterize
from rasterio.features import shapes as raster_shapes
from rasterio.session import AWSSession
from shapely.geometry import box, mapping, shape
from shapely.ops import transform as shapely_transform

from .io_utils import write_json
from .job_schema import JobSpec
from .mlflow_adapter import MLflowJobRun
from .pipeline_config import PipelineConfig


@dataclass
class SceneMatch:
    entry: str
    key: str
    name: str
    score: float


def _s3_parts(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def _norm_scene_name(value: str) -> str:
    name = PurePosixPath(value.strip()).name.lower()
    name = re.sub(r"\.(tif|tiff)$", "", name)
    name = name.replace("_cog", "")
    return re.sub(r"[^a-z0-9а-я]+", "", name)


def _mc_credentials(config: PipelineConfig) -> tuple[str, str]:
    import os

    access = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access and secret:
        return access, secret

    alias = config.s3_alias or "mlplatform"
    mc_config = Path.home() / ".mc" / "config.json"
    payload = json.loads(mc_config.read_text(encoding="utf-8"))
    item = (payload.get("aliases") or {}).get(alias) or {}
    access = item.get("accessKey")
    secret = item.get("secretKey")
    if not access or not secret:
        raise RuntimeError("S3 credentials were not found in env or mc alias")
    return access, secret


def _s3_client(config: PipelineConfig):
    from botocore.config import Config

    access, secret = _mc_credentials(config)
    return boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint_url,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


def _aws_session(config: PipelineConfig) -> AWSSession:
    access, secret = _mc_credentials(config)
    endpoint = config.s3_endpoint_url.replace("http://", "").replace("https://", "").rstrip("/")
    session = boto3.Session(
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
    )
    return AWSSession(session, endpoint_url=endpoint, aws_unsigned=False)


def _list_s3_objects(config: PipelineConfig, uri: str, suffixes: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    client = _s3_client(config)
    bucket, prefix = _s3_parts(uri)
    objects: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = item.get("Key", "")
            if suffixes and not key.lower().endswith(suffixes):
                continue
            objects.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "name": PurePosixPath(key).name,
                    "size": int(item.get("Size") or 0),
                    "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else None,
                    "etag": str(item.get("ETag", "")).strip('"'),
                }
            )
        if not response.get("IsTruncated"):
            return objects
        token = response.get("NextContinuationToken")


def _read_s3_text(config: PipelineConfig, uri: str) -> str:
    client = _s3_client(config)
    bucket, key = _s3_parts(uri)
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode("utf-8-sig")


def _read_s3_json(config: PipelineConfig, uri: str) -> Any:
    return json.loads(_read_s3_text(config, uri))


def _find_layout_files(config: PipelineConfig, layout_uri: str, scenes_file: str, annotation_file: str) -> tuple[str, str]:
    objects = _list_s3_objects(config, layout_uri)
    keys = [item["key"] for item in objects]
    bucket, _ = _s3_parts(layout_uri)

    if annotation_file and annotation_file != "auto":
        annotation_key = annotation_file
        if not annotation_key.startswith("layouts/"):
            _, layout_prefix = _s3_parts(layout_uri)
            annotation_key = layout_prefix.rstrip("/") + "/" + annotation_file
    else:
        geojsons = [key for key in keys if key.lower().endswith(".geojson")]
        if not geojsons:
            raise RuntimeError(f"No GeoJSON annotation found under {layout_uri}")
        annotation_key = sorted(geojsons)[-1]

    scene_candidates = [key for key in keys if PurePosixPath(key).name.lower() == scenes_file.lower()]
    if not scene_candidates:
        raise RuntimeError(f"No {scenes_file} found under {layout_uri}")
    scenes_key = sorted(scene_candidates)[-1]
    return f"s3://{bucket}/{annotation_key}", f"s3://{bucket}/{scenes_key}"


def _norm_scene_name(value: str) -> str:
    name = PurePosixPath(value.strip()).name.lower()
    name = re.sub(r"\.aux\.xml$", "", name)
    name = re.sub(r"\.(tif|tiff)$", "", name)
    name = re.sub(r"[_\-. ]?cog$", "", name)
    return re.sub(r"[^a-z0-9]+", "", name)


def _scene_signature(normalized: str) -> str | None:
    match = re.search(r"kanopus(\d{8})(\d{6}).*?scn(\d{1,2})", normalized)
    if not match:
        return None
    date, tm, scn = match.groups()
    return f"kanopus:{date}:{tm}:scn{int(scn):02d}"


def _scene_score(needle: str, candidate: str) -> tuple[float, str]:
    if not needle or not candidate:
        return 0.0, "empty"
    if needle == candidate:
        return 1.0, "normalized_exact"
    needle_sig = _scene_signature(needle)
    candidate_sig = _scene_signature(candidate)
    if needle_sig and candidate_sig and needle_sig == candidate_sig:
        return 0.995, "kanopus_datetime_scn_signature"
    if len(needle) >= 16 and (needle in candidate or candidate in needle):
        return 0.98, "normalized_substring"
    return difflib.SequenceMatcher(None, needle, candidate).ratio(), "sequence_ratio"


def build_scene_matching_report(
    entries: list[str],
    images: list[dict[str, Any]],
    *,
    accept_threshold: float = 0.92,
    ambiguous_margin: float = 0.015,
) -> dict[str, Any]:
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
    report = build_scene_matching_report(entries, images)
    return [SceneMatch(**item) for item in report["matched"]], report["ambiguous"], report["missing"]


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
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    arr[~np.isfinite(arr)] = 0.0
    out = np.zeros_like(arr, dtype="float32")
    for band in range(arr.shape[0]):
        data = arr[band]
        valid = data[data != 0]
        if valid.size < 16:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[band] = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return out


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
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    eps = 1e-7
    dice = 1 - ((2 * (probs * target).sum() + eps) / (probs.sum() + target.sum() + eps))
    return bce + dice


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
) -> tuple[dict[str, Any], list[Path]]:
    import rasterio
    from rasterio.windows import Window

    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    if not pseudolabel_cfg.get("enabled", False):
        return {"enabled": False}, []

    post_cfg = job.postprocess or {}
    thresholds = post_cfg.get("thresholds") or [threshold]
    threshold_used = float(thresholds[0])
    max_objects = int(post_cfg.get("max_objects") or 500)
    min_area_candidates = post_cfg.get("min_object_area_m2_candidates") or [1000]
    simplify_candidates = post_cfg.get("simplify_tolerance_m_candidates") or [5]
    min_area = float(min_area_candidates[0])
    simplify_tolerance = float(simplify_candidates[0])
    full_scene = bool(pseudolabel_cfg.get("full_scene", True))
    max_windows_per_scene = pseudolabel_cfg.get("max_windows_per_scene")
    max_debug_scenes = pseudolabel_cfg.get("max_debug_scenes")
    if max_debug_scenes is not None:
        matches = matches[: max(1, int(max_debug_scenes))]
    debug_mode = bool(pseudolabel_cfg.get("debug", False))
    features: list[dict[str, Any]] = []
    windows_preview_features: list[dict[str, Any]] = []
    tiling_debug_rows: list[dict[str, Any]] = []
    raw_feature_count = 0
    vertices_before = 0
    vertices_after = 0
    model.eval()
    started = time.time()
    parallel_cfg = ((job.predict.get("inference") or {}).get("parallel") or (job.params.get("inference") or {}).get("parallel") or {})
    parallel_enabled = bool(parallel_cfg.get("enabled", False))
    max_workers = max(1, int(parallel_cfg.get("max_workers") or 1))
    if not parallel_enabled:
        max_workers = 1
    torch_threads_per_worker = max(1, int(parallel_cfg.get("torch_threads_per_worker") or max(1, torch.get_num_threads() // max_workers)))
    previous_torch_threads = torch.get_num_threads()
    torch.set_num_threads(torch_threads_per_worker)

    def origins(length: int, tile: int, stride: int) -> list[int]:
        if length <= tile:
            return [0]
        values = list(range(0, max(1, length - tile + 1), max(1, stride)))
        edge = length - tile
        if values[-1] != edge:
            values.append(edge)
        return sorted(set(max(0, int(value)) for value in values))

    def window_grid(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
        return [(x, y) for y in origins(height, tile, stride) for x in origins(width, tile, stride)]

    def vertex_count(geom_mapping: dict[str, Any]) -> int:
        coords = geom_mapping.get("coordinates") or []
        if geom_mapping.get("type") == "Polygon":
            return sum(len(ring) for ring in coords)
        if geom_mapping.get("type") == "MultiPolygon":
            return sum(len(ring) for poly in coords for ring in poly)
        return 0

    def maybe_write_preview(prob_map: np.ndarray, scene_name: str) -> Path | None:
        if list(experiment_dir.glob("probability_preview_*.png")):
            return None
        try:
            from PIL import Image
        except Exception:
            return None
        scale = max(1, int(max(prob_map.shape) / 1024))
        png = np.clip(prob_map[::scale, ::scale] * 255, 0, 255).astype("uint8")
        path = experiment_dir / f"probability_preview_{_norm_scene_name(scene_name)[:48]}.png"
        Image.fromarray(png).save(path)
        return path

    def process_scene(scene_idx: int, match: SceneMatch) -> dict[str, Any]:
        scene_started = time.time()
        local_features: list[dict[str, Any]] = []
        local_windows_preview_features: list[dict[str, Any]] = []
        local_raw_feature_count = 0
        local_vertices_before = 0
        local_vertices_after = 0
        local_preview_path: Path | None = None
        aws = _aws_session(config)
        with rasterio.Env(aws, AWS_HTTPS="NO", AWS_VIRTUAL_HOSTING="FALSE"):
            path = f"/vsis3/{config.storage.s3_bucket}/{match.key}"
            with rasterio.open(path) as ds:
                stride = int(job.preprocess.get("stride") or patch_size)
                all_windows = window_grid(ds.width, ds.height, patch_size, stride)
                if full_scene:
                    windows = all_windows
                else:
                    windows = all_windows[: max(1, int(max_windows_per_scene or len(all_windows)))]
                prob_sum = np.zeros((ds.height, ds.width), dtype="float32")
                prob_count = np.zeros((ds.height, ds.width), dtype="uint16")
                skipped_reasons: dict[str, int] = {}
                predicted_count = 0
                for x, y in windows:
                    window = Window(x, y, patch_size, patch_size)
                    actual_w = min(patch_size, ds.width - x)
                    actual_h = min(patch_size, ds.height - y)
                    props = {
                        "scene": match.name,
                        "tile_index": len(local_windows_preview_features),
                        "x": int(x),
                        "y": int(y),
                        "width": int(actual_w),
                        "height": int(actual_h),
                        "predicted": False,
                        "skipped_reason": "",
                    }
                    arr = ds.read(input_bands, window=window, boundless=True, fill_value=0)
                    if np.count_nonzero(arr) == 0:
                        skipped_reasons["all_zero"] = skipped_reasons.get("all_zero", 0) + 1
                        props["skipped_reason"] = "all_zero"
                        local_windows_preview_features.append({"type": "Feature", "properties": props, "geometry": mapping(box(*ds.window_bounds(window)))})
                        continue
                    sample = torch.from_numpy(_normalize_image(arr)[None, ...]).to(device)
                    with torch.no_grad():
                        prob = torch.sigmoid(model(sample))[0, 0].detach().cpu().numpy()
                    prob_crop = prob[:actual_h, :actual_w]
                    prob_sum[y : y + actual_h, x : x + actual_w] += prob_crop
                    prob_count[y : y + actual_h, x : x + actual_w] += 1
                    predicted_count += 1
                    props["predicted"] = True
                    local_windows_preview_features.append({"type": "Feature", "properties": props, "geometry": mapping(box(*ds.window_bounds(window)))})

                coverage_mask = prob_count > 0
                prob_map = np.zeros_like(prob_sum, dtype="float32")
                prob_map[coverage_mask] = prob_sum[coverage_mask] / prob_count[coverage_mask]
                mask = (prob_map >= threshold_used).astype("uint8")
                scene_before = 0
                scene_after_filter = 0
                for geom, value in raster_shapes(mask, mask=mask.astype(bool), transform=ds.transform):
                    if value != 1:
                        continue
                    poly = shape(geom)
                    if poly.is_empty:
                        continue
                    scene_before += 1
                    local_raw_feature_count += 1
                    local_vertices_before += vertex_count(mapping(poly))
                    if float(poly.area) < min_area:
                        continue
                    scene_after_filter += 1
                    if simplify_tolerance > 0:
                        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
                    mapped = mapping(poly)
                    local_vertices_after += vertex_count(mapped)
                    local_features.append(
                        {
                            "type": "Feature",
                            "properties": {"scene": match.name, "threshold": threshold_used, "area_m2": float(poly.area)},
                            "geometry": mapped,
                        }
                    )

                nz = np.argwhere(prob_map > 1e-6)
                nonzero_bbox = None
                if nz.size:
                    y0, x0 = nz.min(axis=0)
                    y1, x1 = nz.max(axis=0)
                    nonzero_bbox = [int(x0), int(y0), int(x1) + 1, int(y1) + 1]
                scene_geoms = [shape(item["geometry"]) for item in local_features]
                vector_bounds = None
                if scene_geoms:
                    vector_bounds = [
                        min(g.bounds[0] for g in scene_geoms),
                        min(g.bounds[1] for g in scene_geoms),
                        max(g.bounds[2] for g in scene_geoms),
                        max(g.bounds[3] for g in scene_geoms),
                    ]
                image_bounds = list(ds.bounds)
                vector_area_fraction = None
                if vector_bounds:
                    image_area = max(1e-9, (image_bounds[2] - image_bounds[0]) * (image_bounds[3] - image_bounds[1]))
                    vector_area_fraction = ((vector_bounds[2] - vector_bounds[0]) * (vector_bounds[3] - vector_bounds[1])) / image_area
                debug_row = {
                    "scene_id": match.name,
                    "image_uri": f"s3://{config.storage.s3_bucket}/{match.key}",
                    "width": ds.width,
                    "height": ds.height,
                    "crs": str(ds.crs),
                    "transform": list(ds.transform)[:6],
                    "tile_size": patch_size,
                    "stride": stride,
                    "expected_window_count": len(all_windows),
                    "actual_window_count": len(windows),
                    "actual_predicted_window_count": predicted_count,
                    "skipped_window_count": len(windows) - predicted_count,
                    "skip_reasons": skipped_reasons,
                    "probability_map_shape": [int(ds.height), int(ds.width)],
                    "nonzero_probability_bbox_pixels": nonzero_bbox,
                    "nonzero_probability_area_fraction": float(np.count_nonzero(prob_map > 1e-6) / max(1, ds.width * ds.height)),
                    "image_area_bbox_pixels": [0, 0, int(ds.width), int(ds.height)],
                    "coverage_fraction": float(np.count_nonzero(coverage_mask) / max(1, ds.width * ds.height)),
                    "vector_bounds": vector_bounds,
                    "image_bounds": image_bounds,
                    "vector_area_fraction_of_image_bbox": vector_area_fraction,
                    "objects_before_filter": scene_before,
                    "objects_after_filter": scene_after_filter,
                    "scene_duration_sec": round(time.time() - scene_started, 3),
                }
                if scene_idx == 0:
                    local_preview_path = maybe_write_preview(prob_map, match.name)
        return {
            "scene_index": scene_idx,
            "features": local_features,
            "windows_preview_features": local_windows_preview_features,
            "raw_feature_count": local_raw_feature_count,
            "vertices_before": local_vertices_before,
            "vertices_after": local_vertices_after,
            "preview_path": str(local_preview_path) if local_preview_path else None,
            "scene_duration_sec": round(time.time() - scene_started, 3),
            "debug_row": debug_row,
        }

    scene_results: list[dict[str, Any]] = []
    try:
        if max_workers > 1 and len(matches) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_scene, idx, match) for idx, match in enumerate(matches)]
                for future in as_completed(futures):
                    scene_results.append(future.result())
        else:
            scene_results = [process_scene(idx, match) for idx, match in enumerate(matches)]
    finally:
        torch.set_num_threads(previous_torch_threads)

    for result in sorted(scene_results, key=lambda item: item["scene_index"]):
        features.extend(result["features"])
        windows_preview_features.extend(result["windows_preview_features"])
        tiling_debug_rows.append(result["debug_row"])
        raw_feature_count += int(result["raw_feature_count"])
        vertices_before += int(result["vertices_before"])
        vertices_after += int(result["vertices_after"])
    inference_duration_sec = round(time.time() - started, 3)

    features.sort(key=lambda item: float(item["properties"].get("area_m2") or 0), reverse=True)
    objects_after_filter = len(features)
    if len(features) > max_objects:
        features = features[:max_objects]
    objects_after_top = len(features)
    payload = {
        "type": "FeatureCollection",
        "name": f"{job.job_id}_accepted",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
        "features": features,
    }
    geojson_path = experiment_dir / "accepted.geojson"
    geojson_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    debug_geojson_path: Path | None = None
    if debug_mode:
        debug_payload = {**payload, "name": f"{job.job_id}_accepted_debug"}
        debug_geojson_path = experiment_dir / "accepted_debug.geojson"
        debug_geojson_path.write_text(json.dumps(debug_payload, ensure_ascii=False), encoding="utf-8")
    gz_path = experiment_dir / "accepted.geojson.gz"
    with gzip.open(gz_path, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    windows_preview_path = experiment_dir / "windows_preview.geojson"
    write_json(windows_preview_path, {"type": "FeatureCollection", "name": f"{job.job_id}_windows_preview", "features": windows_preview_features})
    tiling_debug_path = experiment_dir / "tiling_debug.json"
    write_json(tiling_debug_path, {"schema_version": 1, "full_scene": full_scene, "scenes": tiling_debug_rows})
    artifacts = [geojson_path, gz_path, windows_preview_path, tiling_debug_path]
    if debug_geojson_path:
        artifacts.append(debug_geojson_path)
    gpkg_path = experiment_dir / "accepted.gpkg"
    try:
        if features:
            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:3857")
        else:
            gdf = gpd.GeoDataFrame({"scene": [], "threshold": [], "area_m2": []}, geometry=[], crs="EPSG:3857")
        gdf.to_file(gpkg_path, driver="GPKG")
        artifacts.append(gpkg_path)
    except Exception as exc:
        (experiment_dir / "pseudolabel_gpkg_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        artifacts.append(experiment_dir / "pseudolabel_gpkg_error.txt")

    geojson_mb = geojson_path.stat().st_size / (1024 * 1024)
    postprocess_debug = {
        "threshold": threshold_used,
        "min_area_m2": min_area,
        "simplify_tolerance": simplify_tolerance,
        "objects_before_filter": raw_feature_count,
        "objects_after_filter": objects_after_filter,
        "objects_after_top500": objects_after_top,
        "vertices_before": vertices_before,
        "vertices_after": vertices_after,
        "geojson_size_mb": geojson_mb,
        "max_geojson_mb": float(post_cfg.get("max_geojson_mb") or 20),
        "max_objects": max_objects,
        "warning": "geojson_size_exceeds_limit" if geojson_mb > float(post_cfg.get("max_geojson_mb") or 20) else None,
    }
    postprocess_debug_path = experiment_dir / "postprocess_debug.json"
    write_json(postprocess_debug_path, postprocess_debug)
    artifacts.append(postprocess_debug_path)
    artifacts.extend(sorted(experiment_dir.glob("probability_preview_*.png"))[:2])
    gz_mb = gz_path.stat().st_size / (1024 * 1024)
    gpkg_mb = gpkg_path.stat().st_size / (1024 * 1024) if gpkg_path.exists() else None
    coverage_values = [float(row["coverage_fraction"]) for row in tiling_debug_rows]
    coverage_report = {
        "schema_version": 1,
        "job_id": job.job_id,
        "scenes_processed": len(tiling_debug_rows),
        "scenes_failed": 0,
        "total_expected_windows": int(sum(row["expected_window_count"] for row in tiling_debug_rows)),
        "total_predicted_windows": int(sum(row["actual_predicted_window_count"] for row in tiling_debug_rows)),
        "total_skipped_windows": int(sum(row["skipped_window_count"] for row in tiling_debug_rows)),
        "mean_coverage_fraction": float(np.mean(coverage_values)) if coverage_values else None,
        "min_coverage_fraction": float(np.min(coverage_values)) if coverage_values else None,
        "accepted_objects_total": len(features),
        "accepted_geojson_mb": geojson_mb,
        "accepted_geojson_gz_mb": gz_mb,
        "accepted_gpkg_mb": gpkg_mb,
        "top500_applied": objects_after_filter > max_objects,
        "max_objects": max_objects,
        "max_geojson_mb": float(post_cfg.get("max_geojson_mb") or 20),
        "inference_parallel_enabled": parallel_enabled,
        "inference_max_workers": max_workers,
        "torch_threads_per_worker": torch_threads_per_worker,
        "inference_duration_sec": inference_duration_sec,
        "scenes": tiling_debug_rows,
        "warnings": [
            f"low_coverage:{row['scene_id']}:{row['coverage_fraction']:.4f}"
            for row in tiling_debug_rows
            if float(row["coverage_fraction"]) < 0.5
        ],
    }
    coverage_report_path = experiment_dir / "coverage_report.json"
    write_json(coverage_report_path, coverage_report)
    artifacts.append(coverage_report_path)
    inference_timing_report = {
        "schema_version": 1,
        "job_id": job.job_id,
        "parallel": {
            "enabled": parallel_enabled,
            "strategy": "scene" if parallel_enabled else "sequential",
            "max_workers": max_workers,
            "torch_threads_per_worker": torch_threads_per_worker,
        },
        "total_inference_duration_sec": inference_duration_sec,
        "per_scene": [
            {
                "scene_id": row["scene_id"],
                "expected_window_count": row["expected_window_count"],
                "actual_predicted_window_count": row["actual_predicted_window_count"],
                "scene_duration_sec": row.get("scene_duration_sec"),
                "coverage_fraction": row["coverage_fraction"],
            }
            for row in tiling_debug_rows
        ],
    }
    inference_timing_path = experiment_dir / "inference_timing_report.json"
    write_json(inference_timing_path, inference_timing_report)
    artifacts.append(inference_timing_path)
    metrics = {
        "pseudolabel_enabled": True,
        "accepted_objects": len(features),
        "accepted_objects_total": len(features),
        "accepted_geojson_mb": geojson_mb,
        "accepted_geojson_gz_mb": gz_mb,
        "accepted_gpkg_mb": gpkg_mb,
        "total_area_m2": float(sum(float(item["properties"].get("area_m2") or 0) for item in features)),
        "total_vertices": int(sum(len(item["geometry"].get("coordinates", [[]])[0]) if item["geometry"].get("type") == "Polygon" else 0 for item in features)),
        "threshold_used": threshold_used,
        "min_object_area_m2_used": min_area,
        "simplify_tolerance_m_used": simplify_tolerance,
        "postprocess_sec": round(time.time() - started, 3),
        "inference_duration_sec": inference_duration_sec,
        "inference_parallel_enabled": float(parallel_enabled),
        "inference_max_workers": max_workers,
        "torch_threads_per_worker": torch_threads_per_worker,
    }
    if tiling_debug_rows:
        metrics["expected_window_count"] = int(sum(row["expected_window_count"] for row in tiling_debug_rows))
        metrics["actual_predicted_window_count"] = int(sum(row["actual_predicted_window_count"] for row in tiling_debug_rows))
        metrics["coverage_fraction"] = float(np.mean([row["coverage_fraction"] for row in tiling_debug_rows]))
        metrics["matched_scene_count"] = len(tiling_debug_rows)
        metrics["scenes_processed"] = len(tiling_debug_rows)
        metrics["total_expected_windows"] = coverage_report["total_expected_windows"]
        metrics["total_predicted_windows"] = coverage_report["total_predicted_windows"]
        metrics["mean_coverage_fraction"] = coverage_report["mean_coverage_fraction"]
        metrics["min_coverage_fraction"] = coverage_report["min_coverage_fraction"]
    summary_path = experiment_dir / "pseudolabel_summary.json"
    write_json(summary_path, {"metrics": metrics, "artifacts": [str(path) for path in artifacts], "postprocess_debug": postprocess_debug, "coverage_report": coverage_report})
    artifacts.append(summary_path)
    return metrics, artifacts


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
    mlflow_run.log_artifacts([scenes_report_path, matching_report_path])
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
    prepare_duration_sec = round(time.time() - prepare_started, 3)

    postprocess_started = time.time()
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
                "accepted_geojson": "accepted.geojson",
                "accepted_debug_geojson": "accepted_debug.geojson",
                "accepted_geojson_gz": "accepted.geojson.gz",
                "accepted_gpkg": "accepted.gpkg",
                "coverage_report": "coverage_report.json",
                "inference_timing_report": "inference_timing_report.json",
                "tiling_debug": "tiling_debug.json",
                "windows_preview": "windows_preview.geojson",
                "postprocess_debug": "postprocess_debug.json",
                "pseudolabel_summary": "pseudolabel_summary.json",
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
    max_train_tiles = int(job.train.get("max_train_tiles") or 24)
    max_val_tiles = int(job.train.get("max_val_tiles") or 8)
    max_tiles_per_scene = int(job.train.get("max_tiles_per_scene") or 4)
    empty_share = float(job.preprocess.get("max_empty_tile_share") or 0.5)
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
    mlflow_run.log_params(
        {
            "positive_scene_count": dataset_report["positive_scene_count"],
            "negative_scene_count": dataset_report["negative_scene_count"],
            "positive_tile_count": dataset_report["positive_tile_count"],
            "negative_tile_count": dataset_report["negative_tile_count"],
        }
    )
    prepare_duration_sec = round(time.time() - prepare_started, 3)

    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu_only else "cpu")
    model_name = str(model_cfg.get("name") or job.train.get("model_name") or "tiny_unet_4ch")
    model = _build_model(model_name, len(input_bands), 1, int(job.train.get("base_channels") or 8)).to(device)
    freeze_batchnorm_default = model_name.lower().startswith(("deeplab", "deeplabv3plus"))
    freeze_batchnorm = bool(job.train.get("freeze_batchnorm", freeze_batchnorm_default))
    mlflow_run.log_params({"freeze_batchnorm": freeze_batchnorm})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(job.train.get("learning_rate") or 5e-4))
    batch_size = job.train.get("batch_size") or 2
    if batch_size == "auto":
        batch_size = 1 if device.type == "cpu" else 2
    batch_size = max(1, int(batch_size))
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

    best_val_iou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        if freeze_batchnorm:
            _set_batchnorm_eval(model)
        train_losses = []
        train_metrics = []
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
            "val/precision": avg_metric(val_metrics, "precision"),
            "val/recall": avg_metric(val_metrics, "recall"),
            "val/f1": avg_metric(val_metrics, "f1"),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_duration_sec": round(time.time() - epoch_started, 4),
        }
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

    train_duration_sec = round(time.time() - train_started, 3)
    artifacts = _write_history(experiment_dir, history)
    checkpoint_path = experiment_dir / "tiny_unet_4ch.pt"
    checkpoint_path = experiment_dir / f"{model_name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "job_id": job.job_id, "model_name": model_name, "best_epoch": best_epoch}, checkpoint_path)
    artifacts.extend([dataset_report_path, checkpoint_path])
    postprocess_metrics, postprocess_artifacts = _write_pseudolabel_outputs(
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
                "accepted_geojson": "accepted.geojson",
                "accepted_geojson_gz": "accepted.geojson.gz",
                "accepted_gpkg": "accepted.gpkg",
                "pseudolabel_summary": "pseudolabel_summary.json",
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
