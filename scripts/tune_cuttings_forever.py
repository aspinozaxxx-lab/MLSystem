from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"succeeded", "failed", "cancelled"}
DEFAULT_RUN_ROOT = Path("/data/mlsystem/runs")
THRESHOLDS = [round(value / 100.0, 2) for value in range(20, 81, 5)]
LEADERBOARD_FIELDS = [
    "rank",
    "run_id",
    "mlflow_run_id",
    "parent_checkpoint_id",
    "parent_checkpoint_path",
    "status",
    "valid_status",
    "val_pixel_f1",
    "val_pixel_iou",
    "val_precision",
    "val_recall",
    "object_f1",
    "best_epoch",
    "best_threshold",
    "epoch_duration_median_sec",
    "model_name",
    "tile_size",
    "stride",
    "augmentation_level",
    "loss",
    "lr",
    "weight_decay",
    "batch_size",
    "train_positive_tiles",
    "val_positive_tiles",
    "MLMarkup commit",
    "notes",
]
PREVIOUS_CHECKPOINT_FIELDS = [
    "source",
    "mlflow_experiment",
    "mlflow_run_id",
    "run_name",
    "checkpoint_path_or_uri",
    "model_name",
    "architecture",
    "input_bands",
    "tile_size",
    "stride",
    "loss",
    "lr",
    "old_val_pixel_f1",
    "old_val_pixel_iou",
    "old_object_f1",
    "old_best_epoch",
    "old_dataset_info",
    "artifact_exists",
    "checkpoint_loadable",
    "notes",
]
CHECKPOINT_REEVAL_FIELDS = [
    "checkpoint_id",
    "old_val_pixel_f1",
    "new_val_pixel_f1",
    "new_val_pixel_iou",
    "new_precision",
    "new_recall",
    "new_object_f1",
    "threshold",
    "val_scene_count",
    "val_positive_tiles",
    "duration",
    "valid_status",
    "notes",
]
RESOURCE_FIELDS = [
    "timestamp",
    "active_run_id",
    "gpu_index",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "cpu_load1",
    "ram_used_gb",
    "disk_free_gb",
    "active_jobs",
    "pending_jobs",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_csv(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_command(args: list[str], *, cwd: Path | None = None, timeout: int = 15) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def parse_env_file_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"")
    return ""


def api_json(api_url: str, token: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = api_url.rstrip("/") + path
    body = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


def ensure_work_dirs(work_dir: Path) -> None:
    for name in ("queue", "traces", "runs", "reports", "checkpoints"):
        (work_dir / name).mkdir(parents=True, exist_ok=True)


def git_metadata(repo_path: Path) -> dict[str, Any]:
    commit = run_command(["git", "-C", str(repo_path), "rev-parse", "HEAD"])
    branch = run_command(["git", "-C", str(repo_path), "branch", "--show-current"])
    status = run_command(["git", "-C", str(repo_path), "status", "--short"])
    return {"commit": commit, "branch": branch, "dirty": bool(status), "status_short": status}


def candidate_class_dirs(repo_path: Path, requested: str | None) -> list[Path]:
    candidates: list[Path] = []
    if requested:
        requested_path = Path(requested)
        candidates.append(requested_path if requested_path.is_absolute() else repo_path / requested)
    if repo_path.exists():
        for child in sorted(repo_path.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir():
                candidates.append(child)
    return candidates


def class_dir_score(path: Path) -> int:
    if not path.is_dir():
        return -100
    txt_count = len(list(path.glob("*.txt")))
    geojson_count = len(list(path.glob("*.geojson")))
    score = txt_count + 3 * geojson_count
    lowered = path.name.lower()
    if any(marker in lowered for marker in ("cut", "deforest", "clear", "rub", "vyr")):
        score += 5
    if (path / "deforestation.txt").exists():
        score += 5
    if (path / "deforestation.geojson").exists():
        score += 5
    return score


def resolve_class_dir(repo_path: Path, requested: str | None) -> Path:
    candidates = candidate_class_dirs(repo_path, requested)
    existing = [path for path in candidates if path.exists()]
    if requested and existing and existing[0].is_dir():
        return existing[0]
    scored = sorted(((class_dir_score(path), path) for path in existing), key=lambda item: (item[0], str(item[1])), reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    raise RuntimeError(f"MLMarkup class directory was not found under {repo_path}")


def find_data_file(class_dir: Path, requested: str | None, suffix: str, preferred_names: tuple[str, ...]) -> Path:
    if requested:
        requested_path = Path(requested)
        candidate = requested_path if requested_path.is_absolute() else class_dir / requested
        if candidate.exists():
            return candidate
    for name in preferred_names:
        candidate = class_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(class_dir.glob(f"*{suffix}"), key=lambda item: item.name.lower())
    if matches:
        return matches[0]
    raise RuntimeError(f"No {suffix} file found in {class_dir}")


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def count_scenes(path: Path) -> int:
    try:
        return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip() and not line.strip().startswith("#"))
    except Exception:
        return 0


def count_geojson_features(path: Path) -> int:
    payload = read_json(path, default={}) or {}
    features = payload.get("features") if isinstance(payload, dict) else None
    return len(features) if isinstance(features, list) else 0


def snapshot_mlmarkup(work_dir: Path, repo_path: Path, class_dir_arg: str | None, scenes_file: str | None, annotation_file: str | None) -> dict[str, Any]:
    metadata = git_metadata(repo_path)
    class_dir = resolve_class_dir(repo_path, class_dir_arg)
    scenes_path = find_data_file(class_dir, scenes_file, ".txt", ("deforestation.txt", "scenes.txt"))
    annotation_path = find_data_file(class_dir, annotation_file, ".geojson", ("deforestation.geojson", "annotations.geojson"))
    snapshot = {
        "repo_path": str(repo_path),
        "branch": metadata.get("branch") or "",
        "commit": metadata.get("commit") or "",
        "dirty": bool(metadata.get("dirty")),
        "class_dir": relative_or_absolute(class_dir, repo_path),
        "class_dir_path": str(class_dir),
        "scenes_file": scenes_path.name,
        "scenes_file_path": str(scenes_path),
        "annotation_file": annotation_path.name,
        "annotation_file_path": str(annotation_path),
        "scene_count": count_scenes(scenes_path),
        "geojson_feature_count": count_geojson_features(annotation_path),
    }
    write_json(work_dir / "mlmarkup_snapshot.json", snapshot)
    if snapshot["dirty"]:
        (work_dir / "reports" / "mlmarkup_dirty_warning.md").write_text(
            "# MLMarkup dirty warning\n\n"
            "MLMarkup has uncommitted changes. Only diagnostic runs are comparable until the annotation repo is clean.\n",
            encoding="utf-8",
        )
    return snapshot


def metric_value(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics and metrics[key] not in (None, ""):
            return metrics[key]
    return ""


def trace_for_run_dir(run_dir: Path) -> dict[str, Any]:
    for name in ("trace.json", "pipeline_trace.json", "run.json"):
        payload = read_json(run_dir / name, default={})
        if isinstance(payload, dict):
            if "trace" in payload and isinstance(payload["trace"], dict):
                return payload["trace"]
            if "experiment_id" in payload or "model" in payload or "train" in payload:
                return payload
    return {}


def looks_like_cuttings(trace: dict[str, Any], training_result: dict[str, Any], run_dir: Path) -> bool:
    text = json.dumps({"trace": trace, "training": training_result, "path": str(run_dir)}, ensure_ascii=False).lower()
    return any(marker in text for marker in ("cuttings", "deforest", "clearcut", "clear_cut", "deforestation"))


def discover_checkpoint_file(run_dir: Path, training_result: dict[str, Any]) -> Path | None:
    configured = training_result.get("checkpoint_path")
    if configured:
        path = Path(str(configured))
        if path.exists():
            return path
        relative = run_dir / str(configured)
        if relative.exists():
            return relative
    names = (
        "best_model.pt",
        "best_checkpoint.pt",
        "checkpoint_best.pt",
        "model_best.pth",
        "tiny_unet_4ch.pt",
        "unet_resnet34.pt",
        "deeplabv3plus_resnet34.pt",
        "segformer_b0.pt",
    )
    for name in names:
        matches = sorted(run_dir.rglob(name))
        if matches:
            return matches[0]
    generic = sorted([*run_dir.rglob("*.pt"), *run_dir.rglob("*.pth")], key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
    return generic[0] if generic else None


def checkpoint_id_for(path: str, run_id: str | None = None) -> str:
    source = f"{run_id or ''}_{Path(path).stem}"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source).strip("_")[:120] or f"checkpoint_{utc_stamp()}"


def infer_model_name_from_path(path: str) -> str:
    name = Path(path).stem.lower()
    for candidate in (
        "deeplabv3plus_resnet50",
        "deeplabv3plus_resnet34",
        "unet_resnet50",
        "unet_resnet34",
        "unet_resnet18",
        "segformer_b3",
        "segformer_b2",
        "segformer_b1",
        "segformer_b0",
        "tiny_unet_4ch",
    ):
        if candidate in name:
            return candidate
    return ""


def normalize_previous_checkpoint_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field, "") for field in PREVIOUS_CHECKPOINT_FIELDS}
    if not normalized.get("model_name"):
        normalized["model_name"] = infer_model_name_from_path(str(normalized.get("checkpoint_path_or_uri") or ""))
    normalized["checkpoint_id"] = row.get("checkpoint_id") or checkpoint_id_for(str(row.get("checkpoint_path_or_uri") or ""), str(row.get("mlflow_run_id") or ""))
    return normalized


def run_dir_checkpoint_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not run_root.exists():
        return rows
    for result_path in sorted(run_root.rglob("training_result.json")):
        run_dir = result_path.parent
        training_result = read_json(result_path, default={}) or {}
        trace = trace_for_run_dir(run_dir)
        if not looks_like_cuttings(trace, training_result, run_dir):
            continue
        checkpoint_path = discover_checkpoint_file(run_dir, training_result)
        if checkpoint_path is None:
            continue
        last_metrics = training_result.get("last_epoch_metrics") or {}
        model_cfg = trace.get("model") if isinstance(trace.get("model"), dict) else {}
        train_cfg = trace.get("train") if isinstance(trace.get("train"), dict) else {}
        preprocess_cfg = trace.get("preprocess") if isinstance(trace.get("preprocess"), dict) else {}
        row = {
            "source": "run_dir",
            "mlflow_experiment": ((trace.get("mlflow") or {}) if isinstance(trace.get("mlflow"), dict) else {}).get("experiment", ""),
            "mlflow_run_id": ((training_result.get("mlflow") or {}) if isinstance(training_result.get("mlflow"), dict) else {}).get("run_id", ""),
            "run_name": trace.get("experiment_id") or run_dir.name,
            "checkpoint_path_or_uri": str(checkpoint_path),
            "model_name": model_cfg.get("name") or training_result.get("model_name") or "",
            "architecture": model_cfg.get("architecture", ""),
            "input_bands": json.dumps(model_cfg.get("input_bands") or [1, 2, 3, 4]),
            "tile_size": preprocess_cfg.get("tile_size") or train_cfg.get("patch_size") or "",
            "stride": preprocess_cfg.get("stride") or "",
            "loss": (train_cfg.get("loss") or {}).get("name") if isinstance(train_cfg.get("loss"), dict) else train_cfg.get("loss", ""),
            "lr": train_cfg.get("learning_rate", ""),
            "old_val_pixel_f1": training_result.get("best_val_pixel_f1_best_threshold")
            or training_result.get("best_val_pixel_f1")
            or metric_value(last_metrics, "val/pixel_f1_best_threshold", "val/pixel_f1"),
            "old_val_pixel_iou": metric_value(last_metrics, "val/pixel_iou_best_threshold", "val/pixel_iou"),
            "old_object_f1": metric_value(last_metrics, "val/object_f1", "object_f1"),
            "old_best_epoch": training_result.get("best_epoch", ""),
            "old_dataset_info": json.dumps(
                {
                    "train_tile_count": training_result.get("train_tile_count"),
                    "val_tile_count": training_result.get("val_tile_count"),
                    "train_positive_tiles": training_result.get("train_positive_tiles"),
                    "val_positive_tiles": training_result.get("val_positive_tiles"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            "artifact_exists": checkpoint_path.exists(),
            "checkpoint_loadable": "",
            "notes": "",
        }
        row["checkpoint_id"] = checkpoint_id_for(str(checkpoint_path), str(row["mlflow_run_id"] or run_dir.name))
        rows.append(row)
    return rows


def mlflow_tracking_base(mlflow_uri: str) -> str:
    return mlflow_uri.rstrip("/")


def mlflow_get_json(mlflow_uri: str, path: str, query: dict[str, Any] | None = None) -> dict[str, Any]:
    if not mlflow_uri:
        return {}
    url = mlflow_tracking_base(mlflow_uri) + path
    if query:
        encoded = urllib.parse.urlencode({key: value for key, value in query.items() if value is not None})
        url = f"{url}?{encoded}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def mlflow_post_json(mlflow_uri: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not mlflow_uri:
        return {}
    url = mlflow_tracking_base(mlflow_uri) + path
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers={"Accept": "application/json", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}


def mlflow_list_checkpoint_artifacts(mlflow_uri: str, run_id: str, path: str = "", depth: int = 0) -> list[str]:
    if depth > 4:
        return []
    payload = mlflow_get_json(mlflow_uri, "/api/2.0/mlflow/artifacts/list", {"run_id": run_id, "path": path or None})
    result: list[str] = []
    for item in payload.get("files") or []:
        artifact_path = str(item.get("path") or "")
        if item.get("is_dir"):
            result.extend(mlflow_list_checkpoint_artifacts(mlflow_uri, run_id, artifact_path, depth + 1))
        elif artifact_path.lower().endswith((".pt", ".pth")):
            result.append(artifact_path)
    return result


def mlflow_checkpoint_rows(mlflow_uri: str) -> list[dict[str, Any]]:
    experiments = mlflow_get_json(mlflow_uri, "/api/2.0/mlflow/experiments/search", {"max_results": 200}).get("experiments") or []
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        name = str(experiment.get("name") or "")
        if not any(marker in name.lower() for marker in ("deforest", "cutting", "clearcut")):
            continue
        experiment_id = str(experiment.get("experiment_id") or "")
        runs_payload = mlflow_post_json(
            mlflow_uri,
            "/api/2.0/mlflow/runs/search",
            {
                "experiment_ids": [experiment_id],
                "max_results": 50,
                "order_by": ["metrics.`val/pixel_f1_best_threshold` DESC", "metrics.`val/pixel_f1` DESC"],
            },
        )
        for run in runs_payload.get("runs") or []:
            info = run.get("info") or {}
            data = run.get("data") or {}
            params = {item.get("key"): item.get("value") for item in data.get("params") or []}
            metrics = {item.get("key"): item.get("value") for item in data.get("metrics") or []}
            checkpoint_path = (
                params.get("train.initial_checkpoint_path")
                or params.get("checkpoint_path")
                or params.get("model.checkpoint_path")
                or ""
            )
            artifact_path = ""
            if not checkpoint_path:
                artifacts = mlflow_list_checkpoint_artifacts(mlflow_uri, str(info.get("run_id") or ""))
                artifact_path = artifacts[0] if artifacts else ""
                checkpoint_path = f"runs:/{info.get('run_id')}/{artifact_path}" if artifact_path else ""
            if not checkpoint_path:
                continue
            row = {
                "source": "mlflow",
                "mlflow_experiment": name,
                "mlflow_run_id": info.get("run_id", ""),
                "run_name": info.get("run_name") or params.get("mlflow.runName") or "",
                "checkpoint_path_or_uri": checkpoint_path,
                "model_name": params.get("model.name") or params.get("model_name") or "",
                "architecture": params.get("model.architecture") or "",
                "input_bands": params.get("model.input_bands") or "[1, 2, 3, 4]",
                "tile_size": params.get("preprocess.tile_size") or params.get("train.patch_size") or "",
                "stride": params.get("preprocess.stride") or "",
                "loss": params.get("train.loss") or "",
                "lr": params.get("train.learning_rate") or "",
                "old_val_pixel_f1": metric_value(metrics, "val/pixel_f1_best_threshold", "val/pixel_f1"),
                "old_val_pixel_iou": metric_value(metrics, "val/pixel_iou_best_threshold", "val/pixel_iou"),
                "old_object_f1": metric_value(metrics, "val/object_f1", "object_f1"),
                "old_best_epoch": params.get("best_epoch") or "",
                "old_dataset_info": "",
                "artifact_exists": Path(str(checkpoint_path)).exists() or bool(artifact_path),
                "checkpoint_loadable": "",
                "notes": f"mlflow_artifact={artifact_path}" if artifact_path else "",
            }
            row["checkpoint_id"] = checkpoint_id_for(str(checkpoint_path), str(row["mlflow_run_id"]))
            rows.append(row)
    return rows


def validate_checkpoint(row: dict[str, Any], api_container: str) -> dict[str, Any]:
    path = str(row.get("checkpoint_path_or_uri") or "")
    model_name = str(row.get("model_name") or "")
    if not path or not Path(path).exists():
        row["checkpoint_loadable"] = False
        row["notes"] = "; ".join(filter(None, [str(row.get("notes") or ""), "checkpoint file missing or not local"]))
        return row
    if not model_name:
        row["checkpoint_loadable"] = False
        row["notes"] = "; ".join(filter(None, [str(row.get("notes") or ""), "model_name is unknown"]))
        return row
    script = (
        "import json, torch; "
        "from src.real_train import _build_model; "
        f"path={path!r}; model_name={model_name!r}; "
        "payload=torch.load(path, map_location='cpu'); "
        "state=payload.get('model_state_dict') if isinstance(payload, dict) else payload; "
        "state=state or (payload.get('state_dict') if isinstance(payload, dict) else None); "
        "model=_build_model(model_name, in_channels=4, out_channels=1, base_channels=8); "
        "incomp=model.load_state_dict(state, strict=True); "
        "out=model(torch.zeros(1,4,32,32)); "
        "print(json.dumps({'ok': True, 'missing': len(incomp.missing_keys), 'unexpected': len(incomp.unexpected_keys), 'out_shape': list(out.shape)}))"
    )
    output = run_command(["docker", "exec", api_container, "python", "-c", script], timeout=120)
    payload = read_json_from_string(output)
    row["checkpoint_loadable"] = bool(payload.get("ok")) if isinstance(payload, dict) else False
    if payload:
        row["notes"] = "; ".join(
            filter(
                None,
                [
                    str(row.get("notes") or ""),
                    f"missing={payload.get('missing', '')}",
                    f"unexpected={payload.get('unexpected', '')}",
                ],
            )
        )
    else:
        row["notes"] = "; ".join(filter(None, [str(row.get("notes") or ""), "checkpoint validation failed"]))
    return row


def read_json_from_string(text: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text.splitlines()[-1])
    except Exception:
        return {}


def materialize_checkpoint(row: dict[str, Any], args: argparse.Namespace, work_dir: Path) -> dict[str, Any]:
    path = str(row.get("checkpoint_path_or_uri") or "")
    if not path.startswith("runs:/"):
        return row
    parts = path[len("runs:/") :].split("/", 1)
    if len(parts) != 2:
        row["notes"] = "; ".join(filter(None, [str(row.get("notes") or ""), "invalid runs:/ checkpoint uri"]))
        return row
    run_id, artifact_path = parts
    dst = work_dir / "checkpoints" / run_id
    dst.mkdir(parents=True, exist_ok=True)
    tracking_stmt = ""
    if args.mlflow_container_uri:
        tracking_stmt = f"mlflow.set_tracking_uri({args.mlflow_container_uri!r}); "
    script = (
        "import mlflow; "
        f"{tracking_stmt}"
        f"p=mlflow.artifacts.download_artifacts(run_id={run_id!r}, artifact_path={artifact_path!r}, dst_path={str(dst)!r}); "
        "print(p)"
    )
    output = run_command(["docker", "exec", args.api_container, "python", "-c", script], timeout=300)
    downloaded = Path(output.splitlines()[-1].strip()) if output else None
    if downloaded and downloaded.exists():
        row["checkpoint_path_or_uri"] = str(downloaded)
        row["artifact_exists"] = True
        row["notes"] = "; ".join(filter(None, [str(row.get("notes") or ""), f"downloaded_from={path}"]))
    else:
        row["notes"] = "; ".join(filter(None, [str(row.get("notes") or ""), f"download_failed={path}"]))
    return row


def discover_previous_checkpoints(args: argparse.Namespace, work_dir: Path, state: dict[str, Any]) -> list[dict[str, Any]]:
    if state.get("checkpoint_discovery_done") and (work_dir / "previous_checkpoints.csv").exists():
        rows_payload = read_json(work_dir / "previous_checkpoints.json", default=[]) or []
        return [row for row in rows_payload if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in [*run_dir_checkpoint_rows(Path(args.run_root)), *mlflow_checkpoint_rows(args.mlflow_uri)]:
        normalized = normalize_previous_checkpoint_row(row)
        key = str(normalized.get("checkpoint_path_or_uri") or normalized.get("mlflow_run_id"))
        if not key or key in seen:
            continue
        seen.add(key)
        rows.append(normalized)
    rows.sort(
        key=lambda row: (
            float(row.get("old_val_pixel_f1") or 0.0),
            float(row.get("old_object_f1") or 0.0),
        ),
        reverse=True,
    )
    validated: list[dict[str, Any]] = []
    for row in rows[: max(1, int(args.previous_checkpoint_limit) * 3)]:
        row = materialize_checkpoint(row, args, work_dir)
        validated.append(validate_checkpoint(row, args.api_container))
    validated.sort(
        key=lambda row: (
            bool(row.get("checkpoint_loadable")),
            float(row.get("old_val_pixel_f1") or 0.0),
            float(row.get("old_object_f1") or 0.0),
        ),
        reverse=True,
    )
    selected = validated[: max(1, int(args.previous_checkpoint_limit))]
    write_json(work_dir / "previous_checkpoints.json", selected)
    with (work_dir / "previous_checkpoints.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREVIOUS_CHECKPOINT_FIELDS)
        writer.writeheader()
        for row in selected:
            writer.writerow({field: row.get(field, "") for field in PREVIOUS_CHECKPOINT_FIELDS})
    state["checkpoint_discovery_done"] = True
    state["previous_checkpoint_count"] = len(selected)
    return selected


def base_trace(experiment_id: str, snapshot: dict[str, Any], experiment_name: str, *, max_epochs: int, dry_run: bool = False) -> dict[str, Any]:
    commit = str(snapshot.get("commit") or "")
    branch = str(snapshot.get("branch") or "")
    dirty = bool(snapshot.get("dirty"))
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "class_name": "cuttings",
        "task": "train_predict_pseudolabel",
        "pipeline": {
            "stages": ["inventory", "prepare-dataset", "train", "evaluate", "compute-f1", "finalize"],
            "stop_on_failure": True,
            "dry_run": dry_run,
            "log_mlflow": True,
        },
        "images_uri": "s3://mlsystems/images/",
        "annotations": {
            "source": "MLMarkup",
            "repo_path": snapshot["repo_path"],
            "class_dir": snapshot["class_dir"],
            "scenes_file": snapshot["scenes_file"],
            "annotation_file": snapshot["annotation_file"],
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        },
        "pseudolabel": {"enabled": False},
        "preprocess": {
            "tile_size": 512,
            "stride": 512,
            "train_sampling": {"enabled": True},
        },
        "train": {
            "enabled": True,
            "max_epochs": max_epochs,
            "batch_size": 2,
            "augmentation_level": 1,
            "learning_rate": 5e-4,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "loss": {"name": "bce_dice"},
            "metric_thresholds": THRESHOLDS,
            "objective_metric": "val/pixel_f1_best_threshold",
            "objective_mode": "max",
            "require_gpu": True,
            "max_train_batches": None,
            "max_val_batches": None,
        },
        "model": {"name": "tiny_unet_4ch", "input_bands": [1, 2, 3, 4]},
        "mlflow": {"experiment": experiment_name},
        "params": {
            "tuning": True,
            "tuning.primary_metric": "val/pixel_f1",
            "tuning.reject_epoch_time_below_sec": 10,
            "tuning.suspicious_best_epoch_before": 10,
            "mlmarkup.commit": commit,
            "mlmarkup.branch": branch,
            "mlmarkup.dirty": dirty,
            "tile_preparation.enabled": True,
            "pipeline_runner.no_airflow": True,
            "validity.epoch_time_guard": True,
        },
    }


def diagnostic_candidate(snapshot: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    experiment_id = f"cuttings_dataset_diagnostic_{utc_stamp()}"
    trace = base_trace(experiment_id, snapshot, experiment_name, max_epochs=1)
    trace["pipeline"]["stages"] = ["inventory", "prepare-dataset", "train"]
    trace["train"].update({"max_train_batches": 3, "max_val_batches": 2, "augmentation_level": 1})
    trace["model"]["name"] = "tiny_unet_4ch"
    trace["params"]["tuning.hypothesis"] = "diagnostic dataset run: verify MLMarkup split, tile preparation and real GPU training."
    return {"kind": "diagnostic", "experiment_id": experiment_id, "trace": trace, "hypothesis": trace["params"]["tuning.hypothesis"]}


def candidate_library() -> list[dict[str, Any]]:
    candidates = [
        ("A", "tiny_unet_4ch", 512, 512, 1, "bce_dice", 5e-4, 1e-4, "cosine", 30, "baseline tiny model at 512 tiles"),
        ("A", "unet_resnet34", 512, 512, 1, "bce_dice", 5e-4, 1e-4, "cosine", 30, "resnet34 encoder baseline"),
        ("A", "deeplabv3plus_resnet34", 512, 512, 1, "bce_dice", 5e-4, 1e-4, "cosine", 30, "deeplab baseline for shape boundaries"),
        ("A", "segformer_b0", 512, 512, 1, "bce_dice", 5e-4, 1e-4, "cosine", 30, "segformer b0 baseline"),
        ("A", "unet_resnet34", 768, 512, 2, "bce_dice", 5e-4, 1e-4, "cosine", 30, "larger context with moderate overlap"),
        ("A", "tiny_unet_4ch", 768, 512, 2, "bce_dice", 5e-4, 1e-4, "cosine", 30, "tiny model larger context sanity"),
        ("B", "unet_resnet34", 512, 512, 2, "dice_focal", 5e-4, 1e-4, "cosine", 40, "loss search dice focal"),
        ("B", "unet_resnet34", 512, 512, 2, "focal_tversky", 5e-4, 1e-4, "cosine", 40, "loss search focal tversky"),
        ("B", "deeplabv3plus_resnet34", 512, 512, 2, "dice_focal", 5e-4, 1e-4, "cosine", 40, "deeplab loss search"),
        ("B", "segformer_b0", 512, 512, 2, "dice_focal", 5e-4, 1e-4, "cosine", 40, "segformer loss search"),
        ("C", "unet_resnet34", 768, 384, 2, "bce_dice", 5e-4, 1e-4, "cosine", 40, "overlap sampling at 768 tiles"),
        ("C", "unet_resnet34", 1024, 512, 2, "bce_dice", 2e-4, 1e-4, "cosine", 40, "large context if VRAM allows"),
        ("C", "deeplabv3plus_resnet34", 768, 512, 2, "bce_dice", 2e-4, 1e-4, "cosine", 40, "deeplab larger context"),
        ("C", "segformer_b0", 768, 512, 2, "dice_focal", 2e-4, 1e-4, "cosine", 40, "segformer larger context"),
        ("D", "unet_resnet34", 768, 512, 2, "bce_dice", 1e-3, 1e-4, "cosine", 40, "lr search high"),
        ("D", "unet_resnet34", 768, 512, 2, "bce_dice", 2e-4, 1e-4, "cosine", 40, "lr search low"),
        ("D", "unet_resnet34", 768, 512, 2, "bce_dice", 1e-4, 1e-5, "step", 40, "lr scheduler step"),
        ("D", "unet_resnet34", 768, 512, 2, "bce_dice", 5e-4, 1e-5, "none", 40, "no scheduler baseline"),
    ]
    result: list[dict[str, Any]] = []
    for phase, model, tile, stride, aug, loss, lr, weight_decay, scheduler, epochs, hypothesis in candidates:
        result.append(
            {
                "kind": "tuning",
                "phase": phase,
                "model_name": model,
                "tile_size": tile,
                "stride": stride,
                "augmentation_level": aug,
                "loss": loss,
                "lr": lr,
                "weight_decay": weight_decay,
                "scheduler": scheduler,
                "max_epochs": epochs,
                "batch_size": 2,
                "hypothesis": hypothesis,
            }
        )
    return result


def build_tuning_trace(candidate: dict[str, Any], snapshot: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    short = (
        f"{candidate['phase']}_{candidate['model_name']}_t{candidate['tile_size']}_s{candidate['stride']}_"
        f"aug{candidate['augmentation_level']}_{candidate['loss']}_lr{candidate['lr']}"
    )
    experiment_id = candidate.get("experiment_id") or f"cuttings_tune_{utc_stamp()}_{short}".replace(".", "p")
    candidate["experiment_id"] = experiment_id
    trace = base_trace(experiment_id, snapshot, experiment_name, max_epochs=int(candidate["max_epochs"]))
    trace["preprocess"].update({"tile_size": int(candidate["tile_size"]), "stride": int(candidate["stride"])})
    trace["train"].update(
        {
            "batch_size": int(candidate["batch_size"]),
            "augmentation_level": int(candidate["augmentation_level"]),
            "learning_rate": float(candidate["lr"]),
            "weight_decay": float(candidate["weight_decay"]),
            "scheduler": str(candidate["scheduler"]),
            "loss": {"name": str(candidate["loss"])},
            "max_train_batches": None,
            "max_val_batches": None,
        }
    )
    trace["model"]["name"] = str(candidate["model_name"])
    trace["params"]["tuning.phase"] = str(candidate["phase"])
    trace["params"]["tuning.hypothesis"] = str(candidate["hypothesis"])
    return trace


def build_checkpoint_reeval_trace(checkpoint: dict[str, Any], snapshot: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    checkpoint_id = str(checkpoint.get("checkpoint_id") or checkpoint_id_for(str(checkpoint.get("checkpoint_path_or_uri") or ""), str(checkpoint.get("mlflow_run_id") or "")))
    experiment_id = f"cuttings_reeval_{utc_stamp()}_{checkpoint_id}".replace(".", "p")
    tile_size = int(float(checkpoint.get("tile_size") or 512))
    stride = int(float(checkpoint.get("stride") or tile_size))
    model_name = str(checkpoint.get("model_name") or "tiny_unet_4ch")
    trace = base_trace(experiment_id, snapshot, experiment_name, max_epochs=0)
    trace["preprocess"].update({"tile_size": tile_size, "stride": stride})
    trace["train"].update(
        {
            "mode": "eval_only",
            "max_epochs": 0,
            "epochs": 0,
            "augmentation_level": 0,
            "learning_rate": 0.0,
            "weight_decay": 0.0,
            "initial_checkpoint_path": str(checkpoint.get("checkpoint_path_or_uri") or ""),
            "checkpoint_source_run_id": str(checkpoint.get("mlflow_run_id") or ""),
            "checkpoint_source_metric": checkpoint.get("old_val_pixel_f1") or "",
            "checkpoint_finetune": False,
            "initial_checkpoint_strict": True,
            "max_train_batches": None,
            "max_val_batches": None,
        }
    )
    trace["model"]["name"] = model_name
    trace["params"].update(
        {
            "tuning.phase": "checkpoint_reeval",
            "tuning.hypothesis": "Re-evaluate a previous cuttings checkpoint on the fresh MLMarkup validation split before fine-tuning.",
            "checkpoint.parent_run_id": str(checkpoint.get("mlflow_run_id") or ""),
            "checkpoint.path": str(checkpoint.get("checkpoint_path_or_uri") or ""),
            "checkpoint.loaded": bool(checkpoint.get("checkpoint_loadable")),
            "checkpoint.id": checkpoint_id,
            "checkpoint.old_val_pixel_f1": checkpoint.get("old_val_pixel_f1") or "",
        }
    )
    return trace


def build_finetune_trace(parent: dict[str, Any], snapshot: dict[str, Any], experiment_name: str, candidate: dict[str, Any]) -> dict[str, Any]:
    checkpoint_id = str(parent.get("checkpoint_id") or parent.get("parent_checkpoint_id") or "checkpoint")
    model_name = str(parent.get("model_name") or candidate.get("model_name") or "tiny_unet_4ch")
    short = f"{checkpoint_id}_{model_name}_t{candidate['tile_size']}_s{candidate['stride']}_lr{candidate['lr']}_e{candidate['max_epochs']}"
    experiment_id = f"cuttings_ft_{utc_stamp()}_{short}".replace(".", "p")
    trace = base_trace(experiment_id, snapshot, experiment_name, max_epochs=int(candidate["max_epochs"]))
    trace["preprocess"].update({"tile_size": int(candidate["tile_size"]), "stride": int(candidate["stride"])})
    trace["train"].update(
        {
            "mode": "train",
            "batch_size": int(candidate["batch_size"]),
            "augmentation_level": int(candidate["augmentation_level"]),
            "learning_rate": float(candidate["lr"]),
            "weight_decay": float(candidate["weight_decay"]),
            "scheduler": str(candidate["scheduler"]),
            "loss": {"name": str(candidate["loss"])},
            "initial_checkpoint_path": str(parent.get("checkpoint_path_or_uri") or parent.get("parent_checkpoint_path") or ""),
            "checkpoint_source_run_id": str(parent.get("mlflow_run_id") or parent.get("parent_run_id") or ""),
            "checkpoint_source_metric": parent.get("new_val_pixel_f1") or parent.get("old_val_pixel_f1") or "",
            "checkpoint_finetune": True,
            "initial_checkpoint_strict": True,
            "max_train_batches": None,
            "max_val_batches": None,
        }
    )
    trace["model"]["name"] = model_name
    trace["params"].update(
        {
            "tuning.phase": str(candidate.get("phase") or "short_finetune"),
            "tuning.hypothesis": str(candidate["hypothesis"]),
            "tuning.parent_checkpoint": str(parent.get("checkpoint_path_or_uri") or parent.get("parent_checkpoint_path") or ""),
            "tuning.parent_run_id": str(parent.get("mlflow_run_id") or parent.get("parent_run_id") or ""),
            "checkpoint.parent_run_id": str(parent.get("mlflow_run_id") or parent.get("parent_run_id") or ""),
            "checkpoint.path": str(parent.get("checkpoint_path_or_uri") or parent.get("parent_checkpoint_path") or ""),
            "checkpoint.id": checkpoint_id,
            "checkpoint.loaded": True,
        }
    )
    return trace


def short_finetune_candidates(parent: dict[str, Any]) -> list[dict[str, Any]]:
    tile_size = int(float(parent.get("tile_size") or 512))
    stride = int(float(parent.get("stride") or tile_size))
    model_name = str(parent.get("model_name") or "tiny_unet_4ch")
    return [
        {
            "kind": "fine_tune",
            "phase": "short_finetune",
            "model_name": model_name,
            "tile_size": tile_size,
            "stride": stride,
            "augmentation_level": 1,
            "loss": "bce_dice",
            "lr": lr,
            "weight_decay": wd,
            "scheduler": "cosine",
            "max_epochs": 15,
            "batch_size": 2,
            "hypothesis": f"Short fine-tune from checkpoint {parent.get('checkpoint_id')} with lr={lr} wd={wd}.",
        }
        for lr, wd in ((1e-5, 1e-5), (2e-5, 1e-5), (5e-5, 1e-4))
    ]


def load_state(work_dir: Path) -> dict[str, Any]:
    return read_json(work_dir / "tuning_state.json", default={}) or {}


def save_state(work_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(work_dir / "tuning_state.json", state)


def top_reevaluated_checkpoints(work_dir: Path, limit: int = 3) -> list[dict[str, Any]]:
    rows = read_json(work_dir / "checkpoint_reeval_leaderboard.json", default=[]) or []
    valid = [row for row in rows if row.get("valid_status") == "valid"]
    valid.sort(key=lambda row: float(row.get("new_val_pixel_f1") or 0.0), reverse=True)
    return valid[:limit]


def fill_pending_queue(
    state: dict[str, Any],
    snapshot: dict[str, Any],
    experiment_name: str,
    pending_size: int,
    previous_checkpoints: list[dict[str, Any]],
    *,
    start_from_checkpoints: bool,
    work_dir: Path,
) -> None:
    active = state.setdefault("active", [])
    pending = state.setdefault("pending", [])
    if state.get("diagnostic_failed"):
        return
    if not state.get("diagnostic_submitted") and not any(item.get("kind") == "diagnostic" for item in active + pending):
        pending.append(diagnostic_candidate(snapshot, experiment_name))
        return
    if not state.get("diagnostic_done"):
        return
    if snapshot.get("dirty"):
        state["paused_reason"] = "MLMarkup is dirty; full tuning is paused after diagnostic."
        return
    if start_from_checkpoints:
        active_or_pending_reeval = any(item.get("kind") == "checkpoint_reeval" for item in active + pending)
        if not state.get("checkpoint_reeval_queued"):
            loadable = [row for row in previous_checkpoints if str(row.get("checkpoint_loadable")).lower() in {"true", "1"}]
            if not loadable:
                state["paused_reason"] = "No loadable previous checkpoints found; falling back to baseline search."
            else:
                for checkpoint in loadable[: max(1, int(state.get("previous_checkpoint_limit") or 5))]:
                    trace = build_checkpoint_reeval_trace(checkpoint, snapshot, experiment_name)
                    pending.append(
                        {
                            "kind": "checkpoint_reeval",
                            "experiment_id": trace["experiment_id"],
                            "trace": trace,
                            "checkpoint": checkpoint,
                            "hypothesis": trace["params"]["tuning.hypothesis"],
                        }
                    )
                state["checkpoint_reeval_queued"] = True
                return
        if state.get("checkpoint_reeval_queued") and not state.get("checkpoint_reeval_done") and not active_or_pending_reeval:
            state["checkpoint_reeval_done"] = True
        if state.get("checkpoint_reeval_queued") and not state.get("checkpoint_reeval_done"):
            return
        if state.get("checkpoint_reeval_done") and not state.get("finetune_queue_seeded"):
            parents = top_reevaluated_checkpoints(work_dir, limit=3)
            if not parents:
                state["paused_reason"] = "No valid checkpoint re-evaluation result; falling back to baseline search."
            else:
                for parent in parents:
                    for candidate in short_finetune_candidates(parent):
                        trace = build_finetune_trace(parent, snapshot, experiment_name, candidate)
                        candidate["trace"] = trace
                        candidate["parent_checkpoint"] = parent
                        pending.append(candidate)
                state["finetune_queue_seeded"] = True
                return
    library = candidate_library()
    cursor = int(state.get("candidate_cursor") or 0)
    while len(pending) < pending_size and cursor < len(library):
        candidate = dict(library[cursor])
        trace = build_tuning_trace(candidate, snapshot, experiment_name)
        candidate["trace"] = trace
        pending.append(candidate)
        cursor += 1
    state["candidate_cursor"] = cursor


def submit_run(args: argparse.Namespace, work_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    trace = candidate["trace"]
    experiment_id = str(trace["experiment_id"])
    trace_path = work_dir / "traces" / f"{experiment_id}.json"
    write_json(trace_path, trace)
    response = api_json(args.api_url, args.token, "POST", "/api/v1/pipeline-runs", {"trace": trace})
    run_id = str(response["run_id"])
    write_json(work_dir / "runs" / f"{run_id}.api_response.json", response)
    return {
        "kind": candidate.get("kind", "tuning"),
        "run_id": run_id,
        "experiment_id": experiment_id,
        "trace_path": str(trace_path),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {key: value for key, value in candidate.items() if key != "trace"},
    }


def status_for_run(args: argparse.Namespace, run_id: str) -> dict[str, Any]:
    return api_json(args.api_url, args.token, "GET", f"/api/v1/pipeline-runs/{run_id}")


def run_dir_for(run_id: str) -> Path:
    root = Path(os.getenv("MLSYSTEM_RUN_ROOT") or DEFAULT_RUN_ROOT)
    return root / run_id


def load_trace(active: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    trace_path = active.get("trace_path")
    if trace_path:
        trace = read_json(Path(str(trace_path)), default={})
        if trace:
            return trace
    return read_json(run_dir / "trace.json", default={}) or {}


def load_history(training_result: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    candidates = []
    if training_result.get("history_path"):
        candidates.append(Path(str(training_result["history_path"])))
    candidates.append(run_dir / "history.json")
    for path in candidates:
        payload = read_json(path, default=None)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
    return []


def median(values: list[float]) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2.0


def extract_scene_id(row: dict[str, Any]) -> str:
    for key in ("entry", "name", "scene", "scene_id", "key"):
        value = row.get(key)
        if value:
            return str(value)
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def evaluate_run(work_dir: Path, active: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    run_id = str(active["run_id"])
    run_dir = run_dir_for(run_id)
    summary = read_json(run_dir / "summary.json", default={}) or {}
    training_result = summary.get("training_result") or read_json(run_dir / "training_result.json", default={}) or {}
    trace = load_trace(active, run_dir)
    dataset_report = read_json(run_dir / "train_dataset_report.json", default={}) or {}
    history = load_history(training_result, run_dir)
    mlflow_info = status.get("mlflow") or summary.get("mlflow") or training_result.get("mlflow") or {}
    candidate = active.get("candidate") or {}
    is_diagnostic = active.get("kind") == "diagnostic"
    is_checkpoint_reeval = active.get("kind") == "checkpoint_reeval"
    is_full_training = not is_diagnostic and not is_checkpoint_reeval
    errors: list[str] = []
    warnings: list[str] = []
    state = str(status.get("state") or summary.get("status") or "").lower()
    if state != "succeeded":
        errors.append(f"state={state or 'unknown'}")
    pipeline = trace.get("pipeline") or {}
    if pipeline.get("dry_run"):
        errors.append("dry_run=true")
    if not (mlflow_info or {}).get("run_id"):
        errors.append("MLflow run missing")
    if not history:
        errors.append("history missing")
    train_cfg = trace.get("train") or {}
    if is_full_training and (train_cfg.get("max_train_batches") is not None or train_cfg.get("max_val_batches") is not None):
        errors.append("full tuning run has max_train_batches/max_val_batches")
    train_tile_count = int(training_result.get("train_tile_count") or dataset_report.get("train_tile_count") or 0)
    val_tile_count = int(training_result.get("val_tile_count") or dataset_report.get("val_tile_count") or 0)
    train_positive_tiles = int(training_result.get("train_positive_tiles") or dataset_report.get("train_positive_tiles") or 0)
    val_positive_tiles = int(training_result.get("val_positive_tiles") or dataset_report.get("val_positive_tiles") or 0)
    if train_tile_count <= 0 or val_tile_count <= 0:
        errors.append("empty train or validation tile count")
    if train_positive_tiles <= 0 or val_positive_tiles <= 0:
        errors.append("empty train or validation positive tile count")
    train_scenes = {extract_scene_id(row) for row in dataset_report.get("train_scenes") or [] if isinstance(row, dict)}
    val_scenes = {extract_scene_id(row) for row in dataset_report.get("val_scenes") or [] if isinstance(row, dict)}
    overlap = sorted(train_scenes.intersection(val_scenes))
    if overlap:
        errors.append(f"train/val scenes overlap: {overlap[:5]}")
    annotations = (trace.get("annotations") or {}) if isinstance(trace.get("annotations"), dict) else {}
    if not annotations.get("commit"):
        errors.append("MLMarkup commit not logged")
    best_row: dict[str, Any] = {}
    best_pixel_f1 = None
    if history:
        best_row = max(history, key=lambda row: float(row.get("val/pixel_f1_best_threshold", row.get("val/pixel_f1", -1.0)) or -1.0))
        best_pixel_f1 = float(best_row.get("val/pixel_f1_best_threshold", best_row.get("val/pixel_f1", 0.0)) or 0.0)
    if best_pixel_f1 is None:
        errors.append("no val/pixel_f1")
    epoch_durations = [float(row.get("epoch_duration_sec") or 0.0) for row in history if row.get("epoch_duration_sec") is not None]
    median_epoch_duration = training_result.get("epoch_duration", {}).get("median_sec") if isinstance(training_result.get("epoch_duration"), dict) else None
    if median_epoch_duration is None:
        median_epoch_duration = median(epoch_durations)
    if is_full_training and median_epoch_duration is not None and float(median_epoch_duration) < 10.0:
        errors.append(f"median epoch duration below guard: {median_epoch_duration}")
    best_epoch = int(float(best_row.get("epoch", training_result.get("best_epoch") or 0) or 0))
    max_epochs = int(train_cfg.get("max_epochs") or train_cfg.get("epochs") or 0)
    if is_full_training:
        if best_epoch <= 5:
            warnings.append("best epoch is <= 5")
        if max_epochs >= 50 and best_epoch < 10:
            warnings.append("best epoch is < 10 for a long run")
        if max_epochs >= 50 and best_epoch < max(1, int(max_epochs * 0.1)):
            warnings.append("best epoch is within the first 10% of a long run")
        if val_positive_tiles < 10:
            warnings.append("validation positive tile count is very low")
    if is_checkpoint_reeval and best_pixel_f1 is not None and best_pixel_f1 > 0.8:
        warnings.append("previous checkpoint has unusually high fresh validation F1; verify leakage")
    valid_status = "invalid" if errors else ("suspicious" if warnings else "valid")
    parent_checkpoint = candidate.get("parent_checkpoint") or candidate.get("checkpoint") or {}
    object_f1 = best_row.get("val/object_f1", (training_result.get("postprocess_metrics") or {}).get("val_object_f1", ""))
    row = {
        "rank": "",
        "run_id": run_id,
        "mlflow_run_id": (mlflow_info or {}).get("run_id", ""),
        "parent_checkpoint_id": parent_checkpoint.get("checkpoint_id", ""),
        "parent_checkpoint_path": parent_checkpoint.get("checkpoint_path_or_uri", ""),
        "status": state,
        "valid_status": valid_status,
        "val_pixel_f1": best_pixel_f1 if best_pixel_f1 is not None else "",
        "val_pixel_iou": best_row.get("val/pixel_iou_best_threshold", best_row.get("val/pixel_iou", "")),
        "val_precision": best_row.get("val/precision_best_threshold", best_row.get("val/precision", "")),
        "val_recall": best_row.get("val/recall_best_threshold", best_row.get("val/recall", "")),
        "object_f1": object_f1,
        "best_epoch": best_epoch,
        "best_threshold": best_row.get("val/best_threshold", best_row.get("val/threshold", "")),
        "epoch_duration_median_sec": median_epoch_duration if median_epoch_duration is not None else "",
        "model_name": trace.get("model", {}).get("name") or training_result.get("model_name") or candidate.get("model_name", ""),
        "tile_size": trace.get("preprocess", {}).get("tile_size", candidate.get("tile_size", "")),
        "stride": trace.get("preprocess", {}).get("stride", candidate.get("stride", "")),
        "augmentation_level": train_cfg.get("augmentation_level", candidate.get("augmentation_level", "")),
        "loss": (train_cfg.get("loss") or {}).get("name") if isinstance(train_cfg.get("loss"), dict) else train_cfg.get("loss", candidate.get("loss", "")),
        "lr": train_cfg.get("learning_rate", candidate.get("lr", "")),
        "weight_decay": train_cfg.get("weight_decay", candidate.get("weight_decay", "")),
        "batch_size": train_cfg.get("batch_size", candidate.get("batch_size", "")),
        "train_positive_tiles": train_positive_tiles,
        "val_positive_tiles": val_positive_tiles,
        "MLMarkup commit": annotations.get("commit", ""),
        "notes": "; ".join(errors + warnings),
    }
    decision = {
        "run_id": run_id,
        "kind": active.get("kind"),
        "state": state,
        "valid_status": valid_status,
        "errors": errors,
        "warnings": warnings,
        "leaderboard_row": row,
        "parent_checkpoint": parent_checkpoint,
        "mlflow": mlflow_info,
        "training_result": {
            "best_epoch": best_epoch,
            "best_pixel_f1": best_pixel_f1,
            "epoch_duration_median_sec": median_epoch_duration,
            "train_tile_count": train_tile_count,
            "val_tile_count": val_tile_count,
            "val_scene_count": training_result.get("val_scene_count") or dataset_report.get("val_scene_count"),
            "train_positive_tiles": train_positive_tiles,
            "val_positive_tiles": val_positive_tiles,
        },
        "status": status,
        "trace_path": active.get("trace_path"),
    }
    write_json(work_dir / "runs" / f"{run_id}.status.json", status)
    write_json(work_dir / "runs" / f"{run_id}.decision.json", decision)
    report = [
        f"# {run_id}",
        "",
        f"- kind: {active.get('kind')}",
        f"- state: {state}",
        f"- validity: {valid_status}",
        f"- val_pixel_f1: {row['val_pixel_f1']}",
        f"- best_threshold: {row['best_threshold']}",
        f"- epoch_duration_median_sec: {row['epoch_duration_median_sec']}",
        f"- mlflow_run_id: {row['mlflow_run_id']}",
        f"- notes: {row['notes']}",
        "",
    ]
    (work_dir / "reports" / f"{run_id}.md").write_text("\n".join(report), encoding="utf-8")
    return decision


def rebuild_leaderboard(work_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = []
    for path in sorted((work_dir / "runs").glob("*.decision.json")):
        decision = read_json(path, default={})
        if decision and decision.get("leaderboard_row"):
            decisions.append(decision)
    rows = [dict(item["leaderboard_row"]) for item in decisions]
    valid_rows = [row for row in rows if row.get("valid_status") == "valid"]
    valid_rows.sort(key=lambda row: float(row.get("val_pixel_f1") or -1.0), reverse=True)
    for index, row in enumerate(valid_rows, start=1):
        row["rank"] = index
    suspicious_rows = [row for row in rows if row.get("valid_status") == "suspicious"]
    suspicious_rows.sort(key=lambda row: float(row.get("val_pixel_f1") or -1.0), reverse=True)
    invalid_rows = [row for row in rows if row.get("valid_status") == "invalid"]
    leaderboard = valid_rows + suspicious_rows
    write_json(work_dir / "leaderboard.json", leaderboard)
    with (work_dir / "leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for row in leaderboard:
            writer.writerow({field: row.get(field, "") for field in LEADERBOARD_FIELDS})
    with (work_dir / "rejected_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for row in invalid_rows:
            writer.writerow({field: row.get(field, "") for field in LEADERBOARD_FIELDS})
    with (work_dir / "suspicious_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for row in suspicious_rows:
            writer.writerow({field: row.get(field, "") for field in LEADERBOARD_FIELDS})
    lines = ["# Cuttings Tuning Leaderboard", ""]
    if leaderboard:
        lines.append("| rank | run_id | validity | val_pixel_f1 | threshold | model | tile/stride | loss | notes |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |")
        for row in leaderboard[:10]:
            lines.append(
                "| {rank} | {run_id} | {valid_status} | {val_pixel_f1} | {best_threshold} | {model_name} | "
                "{tile_size}/{stride} | {loss} | {notes} |".format(**{key: str(row.get(key, "")) for key in LEADERBOARD_FIELDS})
            )
    else:
        lines.append("No valid or suspicious runs yet.")
    (work_dir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return leaderboard, invalid_rows


def rebuild_checkpoint_reeval_leaderboard(work_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((work_dir / "runs").glob("*.decision.json")):
        decision = read_json(path, default={})
        if not decision or decision.get("kind") != "checkpoint_reeval":
            continue
        parent = decision.get("parent_checkpoint") or {}
        row = decision.get("leaderboard_row") or {}
        training = decision.get("training_result") or {}
        rows.append(
            {
                "checkpoint_id": parent.get("checkpoint_id") or row.get("parent_checkpoint_id") or "",
                "old_val_pixel_f1": parent.get("old_val_pixel_f1", ""),
                "new_val_pixel_f1": row.get("val_pixel_f1", ""),
                "new_val_pixel_iou": row.get("val_pixel_iou", ""),
                "new_precision": row.get("val_precision", ""),
                "new_recall": row.get("val_recall", ""),
                "new_object_f1": row.get("object_f1", ""),
                "threshold": row.get("best_threshold", ""),
                "val_scene_count": training.get("val_scene_count", ""),
                "val_positive_tiles": row.get("val_positive_tiles", ""),
                "duration": row.get("epoch_duration_median_sec", ""),
                "valid_status": decision.get("valid_status", ""),
                "notes": row.get("notes", ""),
                "checkpoint_path_or_uri": parent.get("checkpoint_path_or_uri", ""),
                "model_name": parent.get("model_name", ""),
                "mlflow_run_id": parent.get("mlflow_run_id", ""),
            }
        )
    rows.sort(key=lambda row: float(row.get("new_val_pixel_f1") or 0.0), reverse=True)
    write_json(work_dir / "checkpoint_reeval_leaderboard.json", rows)
    with (work_dir / "checkpoint_reeval_leaderboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_REEVAL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CHECKPOINT_REEVAL_FIELDS})
    return rows


def collect_gpu_rows() -> list[dict[str, Any]]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "gpu_index": parts[0],
                "gpu_util_pct": parts[1],
                "gpu_mem_used_mb": parts[2],
                "gpu_mem_total_mb": parts[3],
            }
        )
    return rows or [{"gpu_index": "", "gpu_util_pct": "", "gpu_mem_used_mb": "", "gpu_mem_total_mb": ""}]


def cpu_load1() -> str:
    try:
        return str(round(os.getloadavg()[0], 3))
    except Exception:
        return ""


def ram_used_gb() -> str:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return ""
    values: dict[str, int] = {}
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            values[parts[0][:-1]] = int(parts[1])
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return ""
    return str(round((total - available) / 1024 / 1024, 3))


def disk_free_gb(path: Path) -> str:
    usage = shutil.disk_usage(path)
    return str(round(usage.free / 1024**3, 3))


def write_resource_log(work_dir: Path, active: list[dict[str, Any]], pending_count: int) -> None:
    active_ids = ",".join(str(item.get("run_id")) for item in active)
    base = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_run_id": active_ids,
        "cpu_load1": cpu_load1(),
        "ram_used_gb": ram_used_gb(),
        "disk_free_gb": disk_free_gb(work_dir),
        "active_jobs": len(active),
        "pending_jobs": pending_count,
    }
    for gpu in collect_gpu_rows():
        row = {**base, **gpu}
        append_csv(work_dir / "resource_log.csv", RESOURCE_FIELDS, row)


def print_progress(work_dir: Path, state: dict[str, Any], leaderboard: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    best_valid = next((row for row in leaderboard if row.get("valid_status") == "valid"), None)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_runs": [item.get("run_id") for item in state.get("active", [])],
        "pending_jobs": len(state.get("pending", [])),
        "best_valid_pixel_f1": best_valid.get("val_pixel_f1") if best_valid else None,
        "best_valid_run_id": best_valid.get("run_id") if best_valid else None,
        "rejected_count": len(rejected),
        "suspicious_count": sum(1 for row in leaderboard if row.get("valid_status") == "suspicious"),
        "paused_reason": state.get("paused_reason"),
        "work_dir": str(work_dir),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def loop(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir)
    ensure_work_dirs(work_dir)
    snapshot = snapshot_mlmarkup(work_dir, Path(args.mlmarkup_repo), args.class_dir, args.scenes_file, args.annotation_file)
    state = load_state(work_dir)
    state.setdefault("active", [])
    state.setdefault("pending", [])
    state.setdefault("completed", [])
    state.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    if args.start_from_checkpoints and not state.get("checkpoint_first_initialized"):
        state["pending"] = [item for item in state.get("pending", []) if item.get("kind") in {"diagnostic", "checkpoint_reeval", "fine_tune"}]
        state["candidate_cursor"] = 0
        state["checkpoint_first_initialized"] = True
    state["previous_checkpoint_limit"] = int(args.previous_checkpoint_limit)
    previous_checkpoints = discover_previous_checkpoints(args, work_dir, state) if args.start_from_checkpoints else []
    while True:
        if (work_dir / "STOP_TUNING").exists():
            state["stopped_at"] = datetime.now(timezone.utc).isoformat()
            state["paused_reason"] = "STOP_TUNING file exists."
            save_state(work_dir, state)
            print_progress(work_dir, state, *rebuild_leaderboard(work_dir))
            return
        snapshot = snapshot_mlmarkup(work_dir, Path(args.mlmarkup_repo), args.class_dir, args.scenes_file, args.annotation_file)
        if args.start_from_checkpoints:
            previous_checkpoints = discover_previous_checkpoints(args, work_dir, state)
        active_next: list[dict[str, Any]] = []
        for active in state.get("active", []):
            run_id = str(active.get("run_id"))
            try:
                status = status_for_run(args, run_id)
            except Exception as exc:
                active["last_poll_error"] = str(exc)
                active_next.append(active)
                continue
            write_json(work_dir / "runs" / f"{run_id}.status.json", status)
            if str(status.get("state", "")).lower() in TERMINAL_STATES:
                decision = evaluate_run(work_dir, active, status)
                state.setdefault("completed", []).append(
                    {
                        "run_id": run_id,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "valid_status": decision.get("valid_status"),
                        "kind": active.get("kind"),
                    }
                )
                if active.get("kind") == "diagnostic":
                    if decision.get("valid_status") == "invalid":
                        state["diagnostic_failed"] = True
                        state["paused_reason"] = "Diagnostic dataset run is invalid; full tuning is blocked."
                    else:
                        state["diagnostic_done"] = True
                        state.pop("paused_reason", None)
            else:
                active_next.append(active)
        state["active"] = active_next
        fill_pending_queue(
            state,
            snapshot,
            args.experiment_name,
            int(args.pending_queue_size),
            previous_checkpoints,
            start_from_checkpoints=bool(args.start_from_checkpoints),
            work_dir=work_dir,
        )
        while len(state["active"]) < int(args.max_active_train_runs) and state.get("pending") and not state.get("diagnostic_failed"):
            candidate = state["pending"].pop(0)
            if candidate.get("kind") == "diagnostic":
                state["diagnostic_submitted"] = True
            active = submit_run(args, work_dir, candidate)
            state["active"].append(active)
            print(json.dumps({"submitted": active}, ensure_ascii=False, sort_keys=True), flush=True)
        write_resource_log(work_dir, state.get("active", []), len(state.get("pending", [])))
        rebuild_checkpoint_reeval_leaderboard(work_dir)
        leaderboard, rejected = rebuild_leaderboard(work_dir)
        print_progress(work_dir, state, leaderboard, rejected)
        save_state(work_dir, state)
        time.sleep(float(args.poll_sec))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operate cuttings tuning through MLSystem pipeline runs.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8088")
    parser.add_argument("--token", default=os.getenv("MLSYSTEM_API_TOKEN", ""))
    parser.add_argument("--token-file", default="")
    parser.add_argument("--mlmarkup-repo", default="/data/mlsystem/MLMarkup")
    parser.add_argument("--class-dir", default="")
    parser.add_argument("--scenes-file", default="")
    parser.add_argument("--annotation-file", default="")
    parser.add_argument("--run-root", default="/data/mlsystem/runs")
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI") or "http://127.0.0.1:5000/mlflow")
    parser.add_argument("--mlflow-container-uri", default="")
    parser.add_argument("--api-container", default="mlsystem-gpu-api")
    parser.add_argument("--experiment-name", default="mlsystem-cuttings-tuning")
    parser.add_argument("--work-dir", default="/data/mlsystem/tuning/cuttings")
    parser.add_argument("--max-active-train-runs", type=int, default=1)
    parser.add_argument("--pending-queue-size", type=int, default=3)
    parser.add_argument("--previous-checkpoint-limit", type=int, default=5)
    parser.add_argument("--start-from-checkpoints", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=45.0)
    args = parser.parse_args(argv)
    if not args.token and args.token_file:
        args.token = parse_env_file_value(Path(args.token_file), "MLSYSTEM_API_TOKEN")
    if not args.token:
        raise SystemExit("MLSYSTEM API token is required via --token, --token-file or MLSYSTEM_API_TOKEN.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
