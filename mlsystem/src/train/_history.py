from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_history(output_dir: Path, rows: list[dict[str, Any]]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "history.json"
    csv_path = output_dir / "history.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if rows:
        keys = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return [json_path, csv_path]
