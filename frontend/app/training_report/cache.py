from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TrainingReportCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.index_path = root / "index.json"
        self.state_path = root / "state.json"
        self.log_path = root / "training_report.log"

    def ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def read_index(self) -> dict[str, Any] | None:
        return self._read_json(self.index_path)

    def read_state(self) -> dict[str, Any]:
        return self._read_json(self.state_path) or {
            "last_update_at": None,
            "known_run_ids": [],
            "last_mlflow_scan_at": None,
            "mlmarkup_commit": None,
            "per_class_dataset_fingerprints": {},
            "last_error": None,
        }

    def write_index(self, payload: dict[str, Any]) -> None:
        self._atomic_write_json(self.index_path, payload)

    def write_state(self, payload: dict[str, Any]) -> None:
        self._atomic_write_json(self.state_path, payload)

    def append_log(self, message: str) -> None:
        self.ensure_root()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now_iso()} {message.rstrip()}\n")

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _atomic_write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self.ensure_root()
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_name, path)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass

