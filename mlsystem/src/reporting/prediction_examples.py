from __future__ import annotations

import html
from pathlib import Path

from ..storage.api import write_json


def write_prediction_examples_report(experiment_dir: Path, job_id: str, preview_paths: list[Path], limit: int = 30) -> list[Path]:
    rows = []
    html_rows = []
    for path in preview_paths[:limit]:
        rel = path.name
        rows.append({"path": str(path), "name": path.name})
        html_rows.append(f"<tr><td>{html.escape(path.name)}</td><td><img src='{html.escape(rel)}' style='max-width:640px'></td></tr>")
    html_path = experiment_dir / "prediction_examples.html"
    table_path = experiment_dir / "prediction_examples_table.json"
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'><title>Prediction examples</title></head><body>",
                f"<h1>{html.escape(job_id)} prediction examples</h1>",
                "<table><tbody>",
                *html_rows,
                "</tbody></table></body></html>",
            ]
        ),
        encoding="utf-8",
    )
    write_json(table_path, {"schema_version": 1, "job_id": job_id, "rows": rows})
    return [html_path, table_path]
