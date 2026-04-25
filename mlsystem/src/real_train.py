from __future__ import annotations

import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
import numpy as np
import torch
from rasterio.features import rasterize
from rasterio.session import AWSSession
from shapely.geometry import box, shape
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


def _match_scenes(entries: list[str], images: list[dict[str, Any]]) -> tuple[list[SceneMatch], list[dict[str, Any]], list[str]]:
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
                        "samples": scene_samples,
                        "positive_tiles": positive_tiles,
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

    images = _list_s3_objects(config, images_uri, suffixes=(".tif", ".tiff"))
    annotation_uri, scenes_uri = _find_layout_files(config, layout_uri, scenes_file, annotation_file)
    entries = [
        line.strip()
        for line in _read_s3_text(config, scenes_uri).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    matches, ambiguous, missing = _match_scenes(entries, images)
    if not matches:
        raise RuntimeError("No scenes from scenes.txt matched available images")

    scene_report = {
        "images_uri": images_uri,
        "layout_uri": layout_uri,
        "annotation_uri": annotation_uri,
        "scenes_uri": scenes_uri,
        "total_images_available": len(images),
        "scenes_entries": len(entries),
        "matched_count": len(matches),
        "ambiguous_count": len(ambiguous),
        "missing_count": len(missing),
        "matched": [match.__dict__ for match in matches],
        "ambiguous": ambiguous,
        "missing": missing,
    }
    scenes_report_path = experiment_dir / "scenes_match_report.json"
    write_json(scenes_report_path, scene_report)
    mlflow_run.log_artifacts([scenes_report_path])
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

    dataset_report = {
        "train_scene_count": len(train_matches),
        "val_scene_count": len(val_matches),
        "train_tile_count": len(train_samples),
        "val_tile_count": len(val_samples),
        "train_positive_tiles": sum(int(mask.sum() > 0) for _, mask in train_samples),
        "val_positive_tiles": sum(int(mask.sum() > 0) for _, mask in val_samples),
        "patch_size": patch_size,
        "train_scenes": train_report,
        "val_scenes": val_report,
    }
    dataset_report_path = experiment_dir / "train_dataset_report.json"
    write_json(dataset_report_path, dataset_report)

    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu_only else "cpu")
    model = TinyUNet(in_channels=len(input_bands), out_channels=1, base=int(job.train.get("base_channels") or 8)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(job.train.get("learning_rate") or 5e-4))
    batch_size = job.train.get("batch_size") or 2
    if batch_size == "auto":
        batch_size = 1 if device.type == "cpu" else 2
    batch_size = max(1, int(batch_size))
    epochs = int(job.train.get("epochs") or 20)
    time_limit_sec = int(job.train.get("time_limit_sec") or 600)
    started = time.time()
    history: list[dict[str, float]] = []

    def make_batches(samples: list[tuple[np.ndarray, np.ndarray]], shuffle: bool) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        rows = list(samples)
        if shuffle:
            random.shuffle(rows)
        return [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    best_val_iou = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
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
        if time.time() - started > time_limit_sec:
            break

    artifacts = _write_history(experiment_dir, history)
    checkpoint_path = experiment_dir / "tiny_unet_4ch.pt"
    torch.save({"model_state_dict": model.state_dict(), "job_id": job.job_id, "best_epoch": best_epoch}, checkpoint_path)
    artifacts.extend([dataset_report_path, checkpoint_path])
    mlflow_run.log_artifacts(artifacts)

    last = history[-1] if history else {}
    return {
        "status": "done",
        "mode": "real_train",
        "device": str(device),
        "epochs_completed": len(history),
        "time_limit_sec": time_limit_sec,
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "last_epoch_metrics": last,
        "matched_scenes_count": len(matches),
        "missing_scenes": missing,
        "ambiguous_scenes": ambiguous,
        "train_scene_count": len(train_matches),
        "val_scene_count": len(val_matches),
        "train_tile_count": len(train_samples),
        "val_tile_count": len(val_samples),
        "train_positive_tiles": dataset_report["train_positive_tiles"],
        "val_positive_tiles": dataset_report["val_positive_tiles"],
        "scenes_match_report": str(scenes_report_path),
        "train_dataset_report": str(dataset_report_path),
        "history_path": str(experiment_dir / "history.json"),
        "history_csv_path": str(experiment_dir / "history.csv"),
        "warnings": [f"{len(missing)} scenes from scenes.txt were not matched"] if missing else [],
    }
