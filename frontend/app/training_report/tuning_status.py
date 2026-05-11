from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_tuning_status(root: Path) -> dict[str, Any]:
    classes: list[dict[str, Any]] = []
    for slug in ("lakes", "abrasion"):
        state_path = root / slug / "state.json"
        stop_path = root / slug / "STOP"
        payload = _read_json(state_path)
        status = payload.get("status") if payload else "unknown"
        if (root / "STOP_ALL").exists() or stop_path.exists():
            status = "stopped"
        classes.append(
            {
                "class_slug": slug,
                "status": status,
                "state_path": str(state_path),
                "stop_path": str(stop_path),
                "current_trial_id": payload.get("current_trial_id"),
                "last_run_id": payload.get("last_run_id"),
                "best_f1": payload.get("best_f1"),
                "last_update_at": payload.get("last_update_at"),
            }
        )
    return {
        "status": "ok",
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "stop_all_path": str(root / "STOP_ALL"),
        "classes": classes,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}

