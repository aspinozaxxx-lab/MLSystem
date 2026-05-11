from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import FrontendConfig
from .cache import TrainingReportCache, utc_now_iso
from .dataset_inventory import SUPPORTED_CLASSES, discover_mlmarkup_path, git_commit, inventory_all_classes
from .mlflow_reader import MLflowReader, normalize_pixel_f1, run_url, summarize_params, validation_kind_from_split


logger = logging.getLogger("mlsystem.frontend.training_report")


class TrainingReportCollector:
    def __init__(self, config: FrontendConfig, cache: TrainingReportCache) -> None:
        self.config = config
        self.cache = cache

    def collect(self) -> dict[str, Any]:
        state = self.cache.read_state()
        existing_index = self.cache.read_index()
        mlmarkup_path = discover_mlmarkup_path(self.config.mlmarkup_path)
        mlmarkup_commit = git_commit(mlmarkup_path) if mlmarkup_path else None
        inventories = inventory_all_classes(mlmarkup_path, images_uri=self.config.default_images_uri)
        errors: list[str] = []
        try:
            mlflow_runs = MLflowReader(self.config.mlflow_tracking_uri).read_class_runs()
        except Exception as exc:  # noqa: BLE001
            logger.warning("training report MLflow scan failed: %s", exc)
            errors.append(f"MLflow scan failed: {type(exc).__name__}: {exc}")
            if existing_index:
                state.update(
                    {
                        "last_mlflow_scan_at": utc_now_iso(),
                        "mlmarkup_commit": mlmarkup_commit,
                        "last_error": "; ".join(errors),
                    }
                )
                self.cache.write_state(state)
                self.cache.append_log("; ".join(errors))
                stale = dict(existing_index)
                stale["status"] = "stale"
                return stale
            mlflow_runs = []
        runtime_runs = _read_runtime_report_runs(Path("/data/mlsystem/reports/class_training"))
        runs_by_id = _merge_runs_by_id([*runtime_runs, *mlflow_runs])
        known_run_ids = sorted(runs_by_id)
        class_rows = self._build_class_rows(inventories, list(runs_by_id.values()))
        updated_at = utc_now_iso()
        payload = {
            "status": "ok" if not errors else "stale" if self.cache.read_index() else "ok",
            "updated_at": updated_at,
            "source": {
                "mlflow_tracking_uri": self.config.mlflow_tracking_uri,
                "mlmarkup_path": str(mlmarkup_path) if mlmarkup_path else None,
                "cache_path": str(self.cache.index_path),
            },
            "classes": class_rows,
        }
        self.cache.write_index(payload)
        state.update(
            {
                "last_update_at": updated_at,
                "known_run_ids": known_run_ids,
                "last_mlflow_scan_at": updated_at,
                "mlmarkup_commit": mlmarkup_commit,
                "per_class_dataset_fingerprints": {
                    slug: item.get("dataset_fingerprint")
                    for slug, item in inventories.items()
                    if item.get("dataset_fingerprint")
                },
                "last_error": "; ".join(errors) if errors else None,
            }
        )
        self.cache.write_state(state)
        if errors:
            self.cache.append_log("; ".join(errors))
        return payload

    def _build_class_rows(self, inventories: dict[str, dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        runs_by_class: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            slug = run.get("class_slug")
            if not slug:
                continue
            runs_by_class.setdefault(str(slug), []).append(run)

        rows: list[dict[str, Any]] = []
        for spec in SUPPORTED_CLASSES:
            inventory = inventories.get(spec.class_slug) or {}
            class_runs = [_enrich_run(run, inventory) for run in runs_by_class.get(spec.class_slug, [])]
            class_runs.sort(key=lambda item: (_sort_f1(item.get("pixel_f1")), item.get("train_date") or ""))
            top_runs = []
            for index, run in enumerate(class_runs[:10], start=1):
                run = dict(run)
                run["rank"] = index
                top_runs.append(_public_run(run))

            best = top_runs[0] if top_runs else None
            dataset_objects = _int_or_zero((best or {}).get("dataset_objects"), inventory.get("objects_count"))
            dataset_scenes = _int_or_zero((best or {}).get("dataset_scenes"), inventory.get("scenes_count"))
            validation_kind = (best or {}).get("validation_kind") or "unknown"
            if validation_kind == "unknown":
                validation_kind = _validation_kind_from_inventory(dataset_scenes)
            warning = None
            if not top_runs:
                warning = "no runs"
            elif validation_kind != "scene_level" or dataset_scenes < 4:
                warning = "limited validation"
            rows.append(
                {
                    "class_name": spec.class_name,
                    "class_slug": spec.class_slug,
                    "best_pixel_f1": (best or {}).get("pixel_f1"),
                    "dataset_date": (best or {}).get("dataset_date") or inventory.get("dataset_date"),
                    "dataset_objects": dataset_objects,
                    "dataset_scenes": dataset_scenes,
                    "best_run_id": (best or {}).get("run_id"),
                    "best_run_url": (best or {}).get("run_url"),
                    "best_train_date": (best or {}).get("train_date"),
                    "validation_kind": validation_kind,
                    "warning": warning,
                    "top_runs": top_runs,
                }
            )
        rows.sort(key=lambda item: (item.get("class_name") or ""))
        return rows


class TrainingReportService:
    def __init__(self, config: FrontendConfig) -> None:
        self.config = config
        self.cache = TrainingReportCache(config.training_report_root)
        self.collector = TrainingReportCollector(config, self.cache)
        self._refresh_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start_background(self) -> None:
        if not self.config.training_report_background_enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="training-report-updater", daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def get_report(self) -> dict[str, Any]:
        index = self.cache.read_index()
        if index is None:
            result = self.refresh_now(reason="cache_miss")
            index = self.cache.read_index()
            if index is None:
                return _empty_report(self.config, error=result.get("error"))
        return self._with_stale_status(index)

    def refresh_now(self, *, reason: str = "manual") -> dict[str, Any]:
        if not self._refresh_lock.acquire(blocking=False):
            return {"status": "busy", "message": "refresh already in progress"}
        try:
            self.cache.append_log(f"refresh start reason={reason}")
            payload = self.collector.collect()
            self.cache.append_log(f"refresh complete reason={reason} classes={len(payload.get('classes') or [])}")
            return {"status": "ok", "updated_at": payload.get("updated_at")}
        except Exception as exc:  # noqa: BLE001
            logger.exception("training report refresh failed")
            state = self.cache.read_state()
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            self.cache.write_state(state)
            self.cache.append_log(f"refresh failed reason={reason} error={type(exc).__name__}: {exc}")
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._refresh_lock.release()

    def status(self) -> dict[str, Any]:
        state = self.cache.read_state()
        index = self.cache.read_index()
        updated_at = (index or {}).get("updated_at") or state.get("last_update_at")
        return {
            "status": "stale" if _is_stale(updated_at, self.config.training_report_stale_minutes) else "ok",
            "updated_at": updated_at,
            "cache_path": str(self.cache.index_path),
            "state_path": str(self.cache.state_path),
            "refresh_in_progress": self._refresh_lock.locked(),
            "known_run_ids": state.get("known_run_ids") or [],
            "last_error": state.get("last_error"),
        }

    def _loop(self) -> None:
        self.refresh_now(reason="startup")
        interval = max(60, int(self.config.training_report_refresh_seconds))
        while not self._stop_event.wait(interval):
            self.refresh_now(reason="background")

    def _with_stale_status(self, index: dict[str, Any]) -> dict[str, Any]:
        payload = dict(index)
        if _is_stale(payload.get("updated_at"), self.config.training_report_stale_minutes):
            payload["status"] = "stale"
        return payload


def _enrich_run(run: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    tags = run.get("tags") if isinstance(run.get("tags"), dict) else {}
    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    dicts = (tags, params)
    dataset_objects = _first_number(dicts, "dataset.objects", "dataset_objects", "objects_count")
    dataset_scenes = _first_number(dicts, "dataset.scenes", "dataset_scenes", "scenes_count")
    dataset_date = _first_text(dicts, "dataset.date", "dataset_date", "mlmarkup_commit_date", "dataset_published_date")
    split_strategy = run.get("split_strategy") or _first_text(dicts, "validation.split_strategy", "split_strategy", "preprocess.split_strategy")
    validation_kind = run.get("validation_kind") or _first_text(dicts, "validation.kind", "validation_kind") or validation_kind_from_split(split_strategy)
    model_name = run.get("model_name") or _first_text(dicts, "model.name", "model_name", "architecture", "model")
    enriched = dict(run)
    enriched.update(
        {
            "dataset_date": _date_only(dataset_date) or inventory.get("dataset_date"),
            "dataset_objects": dataset_objects if dataset_objects is not None else inventory.get("objects_count", 0),
            "dataset_scenes": dataset_scenes if dataset_scenes is not None else inventory.get("scenes_count", 0),
            "split_strategy": split_strategy or "unknown",
            "validation_kind": validation_kind or "unknown",
            "model_name": model_name or "unknown",
            "params_summary": run.get("params_summary") or summarize_params(params, model_name=model_name, split_strategy=split_strategy),
        }
    )
    return enriched


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": run.get("rank"),
        "run_id": run.get("run_id"),
        "run_url": run.get("run_url"),
        "pixel_f1": run.get("pixel_f1"),
        "dataset_date": run.get("dataset_date"),
        "dataset_objects": run.get("dataset_objects"),
        "dataset_scenes": run.get("dataset_scenes"),
        "train_date": run.get("train_date"),
        "class_name": run.get("class_name"),
        "class_slug": run.get("class_slug"),
        "split_strategy": run.get("split_strategy"),
        "validation_kind": run.get("validation_kind"),
        "model_name": run.get("model_name"),
        "params_summary": run.get("params_summary") or "",
        "best_threshold": run.get("best_threshold"),
        "metric_name_source": run.get("metric_name_source"),
    }


def _read_runtime_report_runs(report_root: Path) -> list[dict[str, Any]]:
    path = report_root / "class_training_report.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("classes") or payload.get("results") or payload.get("runs") or []
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        return []
    runs: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        run_id = _first_existing(item, "run_id", "mlflow_run_id", "best_run_id")
        class_name = _first_existing(item, "class_name", "name")
        if not run_id or not class_name:
            continue
        class_slug = _class_slug_from_runtime(item)
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else item
        f1 = normalize_pixel_f1({str(key): value for key, value in metrics.items()})
        experiment_id = item.get("experiment_id") or item.get("mlflow_experiment_id")
        runs.append(
            {
                "run_id": run_id,
                "experiment_id": str(experiment_id or ""),
                "run_url": run_url(experiment_id, run_id) if experiment_id else None,
                "class_name": class_name,
                "class_slug": class_slug,
                "pixel_f1": f1["pixel_f1"] if f1["pixel_f1"] is not None else _float_or_none(_first_existing(item, "best_val_f1", "best_f1", "f1")),
                "metric_name_source": f1["metric_name_source"],
                "best_threshold": f1["best_threshold"],
                "train_date": _date_only(_first_existing(item, "train_date", "start_time", "started_at")),
                "split_strategy": _first_existing(item, "split_strategy", "validation_kind") or "unknown",
                "validation_kind": _first_existing(item, "validation_kind") or validation_kind_from_split(str(_first_existing(item, "split_strategy") or "")),
                "model_name": _first_existing(item, "model_name", "model", "architecture") or "unknown",
                "params_summary": _first_existing(item, "params_summary", "config_summary") or "",
                "metrics": metrics,
                "params": item.get("params") if isinstance(item.get("params"), dict) else {},
                "tags": item.get("tags") if isinstance(item.get("tags"), dict) else {},
            }
        )
    return runs


def _merge_runs_by_id(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for run in runs:
        run_id = run.get("run_id")
        if not run_id:
            continue
        current = result.get(str(run_id), {})
        merged = {**current, **{key: value for key, value in run.items() if value not in (None, "", [])}}
        result[str(run_id)] = merged
    return result


def _empty_report(config: FrontendConfig, *, error: str | None = None) -> dict[str, Any]:
    return {
        "status": "stale" if error else "ok",
        "updated_at": None,
        "source": {
            "mlflow_tracking_uri": config.mlflow_tracking_uri,
            "mlmarkup_path": str(config.mlmarkup_path),
            "cache_path": str(config.training_report_root / "index.json"),
        },
        "classes": [],
        "error": error,
    }


def _is_stale(updated_at: str | None, stale_minutes: int) -> bool:
    if not updated_at:
        return True
    try:
        updated = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds() > stale_minutes * 60


def _first_text(dicts: tuple[dict[str, Any], ...], *keys: str) -> str | None:
    for key in keys:
        for payload in dicts:
            value = payload.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def _first_number(dicts: tuple[dict[str, Any], ...], *keys: str) -> int | None:
    text = _first_text(dicts, *keys)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _int_or_zero(*values: Any) -> int:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_existing(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _class_slug_from_runtime(item: dict[str, Any]) -> str | None:
    from .dataset_inventory import class_slug_for_text

    for key in ("class_slug", "slug", "class_name", "name"):
        slug = class_slug_for_text(str(item.get(key) or ""))
        if slug:
            return slug
    return None


def _date_only(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    if len(text) >= 10 and text[4:5] == "-" and text[7:8] == "-":
        return text[:10]
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def _validation_kind_from_inventory(scene_count: int) -> str:
    return "scene_level" if scene_count >= 4 else "unknown"


def _sort_f1(value: Any) -> float:
    numeric = _float_or_none(value)
    return float("inf") if numeric is None else -numeric
