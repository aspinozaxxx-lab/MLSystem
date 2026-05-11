from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any

from .dataset_inventory import SUPPORTED_CLASSES, class_slug_for_text


PIXEL_F1_PRIORITY = (
    "best_val_pixel_f1",
    "val_pixel_f1",
    "pixel_f1",
    "best_pixel_f1",
    "val_dice",
    "dice",
    "f1",
    "pixel_dice",
    "metrics.pixel_f1",
)

THRESHOLD_RE = re.compile(
    r"(?:^|[./_])(?:best_)?(?:val_)?(?:pixel_)?(?:f1|dice)(?:[./_@:-]*(?:threshold|thr|t))?[./_@:-]*([01](?:\.\d+)?)$",
    re.IGNORECASE,
)


class MLflowReader:
    def __init__(self, tracking_uri: str, *, timeout_seconds: int = 12) -> None:
        self.tracking_uri = tracking_uri.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._working_base: str | None = None

    def read_class_runs(self) -> list[dict[str, Any]]:
        experiments = self.search_experiments()
        experiment_ids = [str(item.get("experiment_id")) for item in experiments if item.get("experiment_id") is not None]
        if not experiment_ids:
            return []
        runs = self.search_runs(experiment_ids)
        result: list[dict[str, Any]] = []
        for run in runs:
            entry = normalize_run(run)
            if entry.get("class_slug"):
                result.append(entry)
        return result

    def search_experiments(self) -> list[dict[str, Any]]:
        payload = self._post_json("/api/2.0/mlflow/experiments/search", {"max_results": 1000, "view_type": "ALL"})
        experiments = payload.get("experiments") if isinstance(payload, dict) else None
        return experiments if isinstance(experiments, list) else []

    def search_runs(self, experiment_ids: list[str]) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            payload: dict[str, Any] = {
                "experiment_ids": experiment_ids,
                "max_results": 1000,
                "run_view_type": "ALL",
                "order_by": ["attributes.start_time DESC"],
            }
            if page_token:
                payload["page_token"] = page_token
            response = self._post_json("/api/2.0/mlflow/runs/search", payload)
            batch = response.get("runs") if isinstance(response, dict) else None
            if isinstance(batch, list):
                runs.extend(batch)
            page_token = response.get("next_page_token") if isinstance(response, dict) else None
            if not page_token:
                return runs

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for base in self._candidate_bases():
            try:
                return self._request_json(base, path, payload)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        if last_error:
            raise last_error
        raise RuntimeError("MLflow tracking URI is empty")

    def _request_json(self, base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = base.rstrip("/") + path
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            decoded = response.read().decode("utf-8")
        if self._working_base is None:
            self._working_base = base
        return json.loads(decoded) if decoded else {}

    def _candidate_bases(self) -> list[str]:
        if self._working_base:
            return [self._working_base]
        bases = [self.tracking_uri]
        if not self.tracking_uri.rstrip("/").endswith("/mlflow"):
            bases.append(self.tracking_uri.rstrip("/") + "/mlflow")
        else:
            bases.append(self.tracking_uri.rsplit("/mlflow", 1)[0])
        seen: set[str] = set()
        result: list[str] = []
        for base in bases:
            if base and base not in seen:
                seen.add(base)
                result.append(base)
        return result


def normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    info = run.get("info") or {}
    data = run.get("data") or {}
    metrics = _kv_list_to_dict(data.get("metrics") or [])
    params = _kv_list_to_dict(data.get("params") or [])
    tags = _kv_list_to_dict(data.get("tags") or [])
    class_slug = _detect_class_slug(tags=tags, params=params, info=info)
    class_name = _class_name_for_slug(class_slug) if class_slug else None
    f1 = normalize_pixel_f1(metrics)
    split_strategy = _first_value(
        (tags, params),
        "validation.split_strategy",
        "split_strategy",
        "preprocess.split_strategy",
        "params.split_strategy",
    )
    model_name = _first_value((tags, params), "model.name", "model_name", "model", "architecture", "model.architecture")
    train_date = _date_from_ms(info.get("start_time")) or _date_from_iso(tags.get("mlsystem.train_date") or tags.get("train_date"))
    duration_sec = _duration_sec(info.get("start_time"), info.get("end_time"))
    epochs_completed = _first_float(
        (metrics, params),
        "epochs_completed",
        "object_safe_scene_group/epoch",
        "epoch",
        "best_epoch",
        "train.epochs_completed",
        "train.epochs",
        "train.epochs_planned",
    )
    best_epoch = _first_float((metrics, params), "best_epoch", "best_val_epoch", "object_safe_scene_group/epoch", "epoch")
    epochs_planned = _first_float((params, metrics), "train.epochs", "train.epochs_planned", "epochs", "max_epochs")
    if best_epoch is None:
        best_epoch = epochs_completed
    return {
        "run_id": info.get("run_id") or run.get("run_id"),
        "experiment_id": str(info.get("experiment_id") or ""),
        "run_status": info.get("status"),
        "run_name": tags.get("mlflow.runName") or info.get("run_name") or "",
        "run_url": run_url(info.get("experiment_id"), info.get("run_id") or run.get("run_id")),
        "class_name": class_name,
        "class_slug": class_slug,
        "pixel_f1": f1["pixel_f1"],
        "metric_name_source": f1["metric_name_source"],
        "best_threshold": f1["best_threshold"],
        "train_date": train_date,
        "split_strategy": split_strategy or "unknown",
        "validation_kind": validation_kind_from_split(split_strategy),
        "model_name": model_name or "unknown",
        "params_summary": summarize_params(params, model_name=model_name, split_strategy=split_strategy),
        "training_duration_sec": duration_sec,
        "epochs_completed": epochs_completed,
        "epochs_planned": epochs_planned,
        "best_epoch": best_epoch,
        "metrics": metrics,
        "params": params,
        "tags": tags,
    }


def normalize_pixel_f1(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized = {_normalize_metric_name(key): (key, _float_or_none(value)) for key, value in metrics.items()}
    threshold_candidates: list[tuple[float, float, str]] = []
    for key, value in metrics.items():
        numeric = _float_or_none(value)
        if numeric is None:
            continue
        match = THRESHOLD_RE.search(key)
        if match and ("f1" in key.casefold() or "dice" in key.casefold()):
            try:
                threshold = float(match.group(1))
            except ValueError:
                continue
            threshold_candidates.append((numeric, threshold, key))
    if threshold_candidates:
        value, threshold, source = max(threshold_candidates, key=lambda item: item[0])
        return {"pixel_f1": value, "metric_name_source": source, "best_threshold": threshold}

    for candidate in PIXEL_F1_PRIORITY:
        normalized_candidate = _normalize_metric_name(candidate)
        item = normalized.get(normalized_candidate)
        if item and item[1] is not None:
            source, value = item
            return {"pixel_f1": value, "metric_name_source": source, "best_threshold": None}
    return {"pixel_f1": None, "metric_name_source": None, "best_threshold": None}


def run_url(experiment_id: Any, run_id: Any) -> str | None:
    if experiment_id is None or run_id is None:
        return None
    return f"/mlflow/#/experiments/{experiment_id}/runs/{run_id}"


def validation_kind_from_split(split_strategy: str | None) -> str:
    value = (split_strategy or "").casefold()
    if any(marker in value for marker in ("scene_level", "leave_one_scene", "scene_group")):
        return "scene_level"
    if "spatial" in value or "block" in value:
        return "spatial_holdout"
    if "tile" in value:
        return "tile_holdout"
    return "unknown"


def summarize_params(params: dict[str, Any], *, model_name: str | None = None, split_strategy: str | None = None) -> str:
    pieces = []
    model = model_name or _first_from_dict(params, ("model.name", "model_name", "architecture", "model"))
    if model:
        pieces.append(str(model))
    for label, keys in (
        ("lr", ("learning_rate", "train.learning_rate", "lr")),
        ("bs", ("batch_size", "train.batch_size")),
        ("patch", ("patch_size", "tile_size", "preprocess.patch_size", "preprocess.tile_size")),
        ("epochs", ("epochs", "max_epochs", "train.epochs", "train.max_epochs")),
    ):
        value = _first_from_dict(params, keys)
        if value not in (None, ""):
            pieces.append(f"{label}={value}")
    split = split_strategy or _first_from_dict(params, ("split_strategy", "preprocess.split_strategy"))
    if split:
        pieces.append(str(split))
    return ", ".join(pieces[:6])


def _detect_class_slug(*, tags: dict[str, Any], params: dict[str, Any], info: dict[str, Any]) -> str | None:
    for key in ("mlsystem.class_name", "class_name", "dataset.class_name"):
        slug = class_slug_for_text(str(tags.get(key) or ""))
        if slug:
            return slug
    for key in ("class_name", "params.class_name", "dataset.class_name"):
        slug = class_slug_for_text(str(params.get(key) or ""))
        if slug:
            return slug
    run_name = str(tags.get("mlflow.runName") or info.get("run_name") or "")
    slug = class_slug_for_text(run_name)
    if slug:
        return slug
    text = " ".join(str(value) for value in [*params.values(), *tags.values()] if value is not None)
    return class_slug_for_text(text)


def _class_name_for_slug(class_slug: str | None) -> str | None:
    if not class_slug:
        return None
    for spec in SUPPORTED_CLASSES:
        if spec.class_slug == class_slug:
            return spec.class_name
    return None


def _kv_list_to_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        key = item.get("key")
        if not key:
            continue
        result[str(key)] = item.get("value")
    return result


def _normalize_metric_name(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_value(dicts: tuple[dict[str, Any], ...], *keys: str) -> str | None:
    for key in keys:
        for item in dicts:
            value = item.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _first_float(dicts: tuple[dict[str, Any], ...], *keys: str) -> float | None:
    for key in keys:
        for item in dicts:
            value = item.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _first_from_dict(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _duration_sec(start_ms: Any, end_ms: Any) -> float | None:
    try:
        start = int(start_ms)
        end = int(end_ms)
    except (TypeError, ValueError):
        return None
    if start <= 0 or end <= start:
        return None
    return round((end - start) / 1000, 3)


def _date_from_ms(value: Any) -> str | None:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date().isoformat()


def _date_from_iso(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
            return text
        return None
