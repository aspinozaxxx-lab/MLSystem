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


MIN_TRUSTED_TRAIN_DATE = "2026-05-06"
MIN_TRUSTED_BEST_EPOCH = 10.0


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
        overall_best = _overall_best(class_rows)
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
            "overall_best": overall_best,
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
            enriched_runs = [_enrich_run(run, inventory) for run in runs_by_class.get(spec.class_slug, [])]
            version_windows = _dataset_version_windows(enriched_runs, inventory)
            exclusion_reasons: dict[str, int] = {}
            class_runs: list[dict[str, Any]] = []
            for run in enriched_runs:
                reason = _training_run_exclusion_reason(run, version_windows)
                if reason:
                    exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1
                    continue
                class_runs.append(run)
            dataset_versions = []
            for index, run in enumerate(_best_runs_by_dataset_version(class_runs), start=1):
                run = dict(run)
                run["rank"] = index
                dataset_versions.append(_public_run(run))

            best = _best_current_dataset_run(dataset_versions, inventory) or (dataset_versions[0] if dataset_versions else None)
            dataset_objects = _int_or_zero((best or {}).get("dataset_objects"), inventory.get("objects_count"))
            dataset_scenes = _int_or_zero((best or {}).get("dataset_scenes"), inventory.get("scenes_count"))
            validation_kind = (best or {}).get("validation_kind") or "unknown"
            if validation_kind == "unknown":
                validation_kind = _validation_kind_from_inventory(dataset_scenes)
            warning = None
            if not dataset_versions:
                warning = "no runs"
            elif validation_kind != "scene_level" or dataset_scenes < 4:
                warning = "limited validation"
            rows.append(
                {
                    "class_name": spec.class_name,
                    "class_slug": spec.class_slug,
                    "best_pixel_f1": (best or {}).get("pixel_f1"),
                    "dataset_version": (best or {}).get("dataset_version"),
                    "dataset_version_source": (best or {}).get("dataset_version_source"),
                    "dataset_fingerprint": (best or {}).get("dataset_fingerprint"),
                    "dataset_date": (best or {}).get("dataset_date") or inventory.get("dataset_date"),
                    "dataset_objects": dataset_objects,
                    "dataset_scenes": dataset_scenes,
                    "best_run_id": (best or {}).get("run_id"),
                    "best_run_url": (best or {}).get("run_url"),
                    "best_train_date": (best or {}).get("train_date"),
                    "validation_kind": validation_kind,
                    "warning": warning,
                    "quality_filter": {
                        "excluded_runs": sum(exclusion_reasons.values()),
                        "reasons": exclusion_reasons,
                        "rules": [
                            "exclude missing or perfect pixel F1",
                            f"exclude runs before {MIN_TRUSTED_TRAIN_DATE}",
                            f"exclude runs with fewer than {int(MIN_TRUSTED_BEST_EPOCH)} completed epochs",
                            "exclude runs without explicit dataset.version or dataset.fingerprint",
                            "exclude runs outside their dataset version publication window",
                            "exclude runs without reliable dataset scene/object counts",
                        ],
                    },
                    "dataset_versions": dataset_versions,
                    "top_runs": dataset_versions,
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
    dataset_objects = _int_or_none(run.get("dataset_objects")) or _first_number(dicts, "dataset.objects", "dataset_objects", "objects_count")
    dataset_scenes = _int_or_none(run.get("dataset_scenes")) or _first_number(dicts, "dataset.scenes", "dataset_scenes", "scenes_count")
    dataset_train_scenes = _int_or_none(run.get("dataset_train_scenes")) or _first_number(dicts, "dataset.train_scenes", "dataset_train_scenes", "train_scenes")
    dataset_val_scenes = _int_or_none(run.get("dataset_val_scenes")) or _first_number(dicts, "dataset.val_scenes", "dataset_val_scenes", "val_scenes")
    dataset_version = run.get("dataset_version") or _first_text(dicts, "dataset.version", "dataset_version")
    dataset_version_source = run.get("dataset_version_source") or _first_text(dicts, "dataset.version_source", "dataset_version_source")
    dataset_fingerprint = run.get("dataset_fingerprint") or _first_text(dicts, "dataset.fingerprint", "dataset_fingerprint")
    dataset_git_commit_date = run.get("dataset_git_commit_date") or _first_text(dicts, "dataset.git_commit_date", "dataset_git_commit_date", "mlmarkup_commit_date")
    dataset_date = run.get("dataset_date") or _date_only(dataset_git_commit_date) or _first_text(dicts, "dataset.date", "dataset_date", "dataset_published_date")
    split_strategy = run.get("split_strategy") or _first_text(dicts, "validation.split_strategy", "split_strategy", "preprocess.split_strategy")
    validation_kind = run.get("validation_kind") or _first_text(dicts, "validation.kind", "validation_kind") or validation_kind_from_split(split_strategy)
    model_name = run.get("model_name") or _first_text(dicts, "model.name", "model_name", "architecture", "model")
    explicit_version = dataset_version or dataset_fingerprint
    inventory_matches = _inventory_matches_dataset_version(inventory, dataset_version, dataset_fingerprint)
    if dataset_objects is None and inventory_matches:
        dataset_objects = _int_or_none(inventory.get("objects_count"))
    if dataset_scenes is None and inventory_matches:
        dataset_scenes = _int_or_none(inventory.get("scenes_count"))
    if dataset_date is None and inventory_matches:
        dataset_date = inventory.get("dataset_date")
    enriched = dict(run)
    enriched.update(
        {
            "dataset_date": _date_only(dataset_date),
            "dataset_version": explicit_version,
            "dataset_version_source": dataset_version_source or ("fingerprint" if dataset_fingerprint and not dataset_version else None),
            "dataset_fingerprint": dataset_fingerprint or (inventory.get("dataset_fingerprint") if inventory_matches else None),
            "dataset_objects": dataset_objects,
            "dataset_scenes": dataset_scenes,
            "dataset_train_scenes": dataset_train_scenes,
            "dataset_val_scenes": dataset_val_scenes,
            "dataset_git_commit_date": dataset_git_commit_date,
            "split_strategy": split_strategy or "unknown",
            "validation_kind": validation_kind or "unknown",
            "model_name": model_name or "unknown",
            "params_summary": run.get("params_summary") or summarize_params(params, model_name=model_name, split_strategy=split_strategy),
            "training_duration_sec": run.get("training_duration_sec") or _first_float(dicts, "duration_sec", "training_duration_sec"),
            "epochs_completed": run.get("epochs_completed") or _first_float(dicts, "epochs_completed", "train.epochs_completed", "epoch"),
            "epochs_planned": run.get("epochs_planned") or _first_float(dicts, "train.epochs", "train.epochs_planned", "epochs", "max_epochs"),
            "best_epoch": run.get("best_epoch") or _first_float(dicts, "best_epoch", "best_val_epoch", "epoch"),
        }
    )
    return enriched


def _public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": run.get("rank"),
        "run_id": run.get("run_id"),
        "run_url": run.get("run_url"),
        "pixel_f1": run.get("pixel_f1"),
        "dataset_version": run.get("dataset_version"),
        "dataset_version_source": run.get("dataset_version_source"),
        "dataset_fingerprint": run.get("dataset_fingerprint"),
        "dataset_date": run.get("dataset_date"),
        "dataset_objects": run.get("dataset_objects"),
        "dataset_scenes": run.get("dataset_scenes"),
        "dataset_train_scenes": run.get("dataset_train_scenes"),
        "dataset_val_scenes": run.get("dataset_val_scenes"),
        "dataset_git_commit_date": run.get("dataset_git_commit_date"),
        "train_date": run.get("train_date"),
        "class_name": run.get("class_name"),
        "class_slug": run.get("class_slug"),
        "split_strategy": run.get("split_strategy"),
        "validation_kind": run.get("validation_kind"),
        "model_name": run.get("model_name"),
        "params_summary": run.get("params_summary") or "",
        "best_threshold": run.get("best_threshold"),
        "metric_name_source": run.get("metric_name_source"),
        "training_duration_sec": run.get("training_duration_sec"),
        "epochs_completed": run.get("epochs_completed"),
        "epochs_planned": run.get("epochs_planned"),
        "best_epoch": run.get("best_epoch"),
    }


def _best_runs_by_dataset_version(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        key = f"{run.get('class_slug') or ''}:{run.get('dataset_version') or run.get('dataset_fingerprint') or 'unknown'}"
        grouped.setdefault(key, []).append(run)
    best: list[dict[str, Any]] = []
    for group in grouped.values():
        group.sort(key=lambda item: (_sort_f1(item.get("pixel_f1")), item.get("train_date") or ""))
        best.append(group[0])
    best.sort(key=lambda item: (_sort_f1(item.get("pixel_f1")), item.get("train_date") or ""))
    return best


def _best_current_dataset_run(runs: list[dict[str, Any]], inventory: dict[str, Any]) -> dict[str, Any] | None:
    current_ids = {
        str(value)
        for value in (
            inventory.get("dataset_version"),
            inventory.get("dataset_fingerprint"),
            inventory.get("git_commit"),
            inventory.get("mlmarkup_commit"),
        )
        if value
    }
    if not current_ids:
        return None
    matches = [
        run
        for run in runs
        if str(run.get("dataset_version") or "") in current_ids or str(run.get("dataset_fingerprint") or "") in current_ids
    ]
    if not matches:
        inventory_objects = _int_or_none(inventory.get("objects_count"))
        inventory_scenes = _int_or_none(inventory.get("scenes_count"))
        if inventory_objects is not None and inventory_scenes is not None:
            matches = [
                run
                for run in runs
                if _int_or_none(run.get("dataset_objects")) == inventory_objects
                and _int_or_none(run.get("dataset_scenes")) == inventory_scenes
            ]
        if not matches and inventory_objects is not None:
            matches = [
                run
                for run in runs
                if _int_or_none(run.get("dataset_objects")) == inventory_objects
            ]
    if not matches:
        return None
    matches.sort(key=lambda item: (_sort_f1(item.get("pixel_f1")), item.get("train_date") or ""))
    return matches[0]


def _overall_best(class_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in class_rows:
        for version in row.get("dataset_versions") or []:
            item = dict(version)
            item.setdefault("class_name", row.get("class_name"))
            item.setdefault("class_slug", row.get("class_slug"))
            candidates.append(item)
    if not candidates:
        return None
    candidates.sort(key=lambda item: (_sort_f1(item.get("pixel_f1")), item.get("train_date") or ""))
    return candidates[0]


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
                "dataset_version": _first_existing(item, "dataset_version", "dataset.version"),
                "dataset_version_source": _first_existing(item, "dataset_version_source", "dataset.version_source"),
                "dataset_fingerprint": _first_existing(item, "dataset_fingerprint", "dataset.fingerprint"),
                "dataset_objects": _first_existing(item, "dataset_objects", "dataset.objects", "objects_count"),
                "dataset_scenes": _first_existing(item, "dataset_scenes", "dataset.scenes", "scenes_count"),
                "dataset_train_scenes": _first_existing(item, "dataset_train_scenes", "dataset.train_scenes"),
                "dataset_val_scenes": _first_existing(item, "dataset_val_scenes", "dataset.val_scenes"),
                "dataset_git_commit_date": _first_existing(item, "dataset_git_commit_date", "dataset.git_commit_date"),
                "dataset_date": _date_only(_first_existing(item, "dataset_date", "dataset.git_commit_date", "dataset_git_commit_date")),
                "train_date": _date_only(_first_existing(item, "train_date", "start_time", "started_at")),
                "split_strategy": _first_existing(item, "split_strategy", "validation_kind") or "unknown",
                "validation_kind": _first_existing(item, "validation_kind") or validation_kind_from_split(str(_first_existing(item, "split_strategy") or "")),
                "model_name": _first_existing(item, "model_name", "model", "architecture") or "unknown",
                "params_summary": _first_existing(item, "params_summary", "config_summary") or "",
                "training_duration_sec": _float_or_none(_first_existing(item, "duration_sec", "training_duration_sec")),
                "epochs_completed": _float_or_none(_first_existing(item, "epochs_completed", "completed_epochs")),
                "epochs_planned": _float_or_none(_first_existing(item, "epochs_planned", "epochs_planned", "epochs")),
                "best_epoch": _float_or_none(_first_existing(item, "best_epoch", "best_val_epoch")),
                "run_status": _first_existing(item, "run_status", "status"),
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


def _first_float(dicts: tuple[dict[str, Any], ...], *keys: str) -> float | None:
    text = _first_text(dicts, *keys)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int_or_zero(*values: Any) -> int:
    for value in values:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


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


def _inventory_matches_dataset_version(inventory: dict[str, Any], *versions: Any) -> bool:
    inventory_version = str(inventory.get("dataset_fingerprint") or "")
    if not inventory_version:
        return False
    return any(str(value or "") == inventory_version for value in versions)


def _dataset_version_windows(runs: list[dict[str, Any]], inventory: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    version_dates: dict[str, str] = {}
    inventory_version = str(inventory.get("dataset_fingerprint") or "")
    inventory_date = _date_only(inventory.get("dataset_date"))
    if inventory_version and inventory_date:
        version_dates[inventory_version] = inventory_date
    for run in runs:
        version = str(run.get("dataset_version") or run.get("dataset_fingerprint") or "")
        date_value = _date_only(run.get("dataset_date"))
        if not version or not date_value:
            continue
        current = version_dates.get(version)
        if current is None or date_value < current:
            version_dates[version] = date_value
    ordered = sorted(version_dates.items(), key=lambda item: item[1])
    windows: dict[str, dict[str, str | None]] = {}
    for index, (version, start) in enumerate(ordered):
        next_start = ordered[index + 1][1] if index + 1 < len(ordered) else None
        windows[version] = {"start": start, "end": next_start}
    return windows


def _report_exclude_tag(run: dict[str, Any]) -> bool:
    tags = run.get("tags") if isinstance(run.get("tags"), dict) else {}
    params = run.get("params") if isinstance(run.get("params"), dict) else {}
    value = tags.get("mlsystem.report.exclude") or params.get("mlsystem.report.exclude")
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _basic_training_exclusion_reason(run: dict[str, Any]) -> str | None:
    f1 = _float_or_none(run.get("pixel_f1"))
    if f1 is None or f1 < 0.0 or f1 > 1.0 or _is_perfect_pixel_f1(f1):
        return "invalid_pixel_f1"
    status = str(run.get("run_status") or "").upper()
    if status and status not in {"FINISHED", "SUCCEEDED", "SUCCESS", "OK", "KILLED"}:
        return "non_terminal_run_status"
    train_date = _date_only(run.get("train_date"))
    if not train_date or train_date < MIN_TRUSTED_TRAIN_DATE:
        return "train_date_before_cutoff"
    epochs_completed = _float_or_none(run.get("epochs_completed"))
    if epochs_completed is None:
        epochs_completed = _float_or_none(run.get("epochs_planned"))
    if epochs_completed is None:
        epochs_completed = _float_or_none(run.get("best_epoch"))
    if epochs_completed is None or epochs_completed < MIN_TRUSTED_BEST_EPOCH:
        return "insufficient_epochs"
    metric_source = str(run.get("metric_name_source") or "").casefold()
    if metric_source and "object" in metric_source:
        return "object_metric_source"
    return None


def _training_run_exclusion_reason(run: dict[str, Any], version_windows: dict[str, dict[str, str | None]]) -> str | None:
    if _report_exclude_tag(run):
        return "report_excluded"
    basic = _basic_training_exclusion_reason(run)
    if basic:
        return basic
    version = str(run.get("dataset_version") or run.get("dataset_fingerprint") or "")
    if not version:
        return "missing_dataset_version"
    train_date = _date_only(run.get("train_date"))
    window = version_windows.get(version)
    if train_date and window:
        start = window.get("start")
        end = window.get("end")
        if start and train_date < start:
            return "train_date_before_dataset_date"
        if end and train_date >= end:
            return "train_date_after_next_dataset_version"
    scenes = _int_or_none(run.get("dataset_scenes"))
    if scenes is None or scenes <= 0:
        return "missing_dataset_scene_count"
    objects = _int_or_none(run.get("dataset_objects"))
    if objects is None or objects <= 0:
        return "missing_dataset_object_count"
    return None


def _is_perfect_pixel_f1(value: Any) -> bool:
    numeric = _float_or_none(value)
    return numeric is not None and abs(numeric - 1.0) <= 1e-12


def _is_trusted_training_run(run: dict[str, Any]) -> bool:
    return _basic_training_exclusion_reason(run) is None
