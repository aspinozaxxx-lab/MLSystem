from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlsystem.src.mlflow_adapter.api import search_runs  # noqa: E402
from mlsystem.src.pipeline_config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a lightweight MLflow leaderboard without launching experiments.")
    parser.add_argument("--experiment", default="mlsystem-cuttings-tuning")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-results", type=int, default=1000)
    parser.add_argument("--metric", default="val/pixel_f1")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _leaderboard_rows(
        search_runs(
            load_config(),
            experiment_name=args.experiment,
            max_results=args.max_results,
            order_by=["attributes.start_time DESC"],
        ),
        metric=args.metric,
    )
    _write_csv(output_dir / "leaderboard.csv", rows)
    (output_dir / "leaderboard.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state = {
        "schema_version": 1,
        "experiment": args.experiment,
        "metric": args.metric,
        "run_count": len(rows),
        "best_run": rows[0] if rows else None,
        "source": "MLflow",
    }
    (output_dir / "tuning_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, ensure_ascii=False, indent=2))


def _leaderboard_rows(runs: list[dict[str, Any]], *, metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run.get("metrics") or {}
        params = run.get("params") or {}
        tags = run.get("tags") or {}
        rows.append(
            {
                "run_id": run.get("run_id"),
                "status": run.get("status"),
                "run_name": tags.get("mlflow.runName") or tags.get("run_name") or "",
                "metric": metric,
                "metric_value": _float_or_none(metrics.get(metric)),
                "val_pixel_f1": _float_or_none(metrics.get("val/pixel_f1")),
                "best_val_pixel_f1": _float_or_none(metrics.get("val/best_pixel_f1") or metrics.get("best_val_pixel_f1")),
                "epoch": _float_or_none(metrics.get("epoch")),
                "epoch_duration_sec": _float_or_none(metrics.get("epoch_duration_sec")),
                "data_samples_per_sec": _float_or_none(metrics.get("data/samples_per_sec")),
                "data_batch_wait_p95_sec": _float_or_none(metrics.get("data/batch_wait_p95_sec")),
                "model": params.get("model.name") or params.get("train.model_name") or "",
                "tile_size": params.get("preprocess.tile_size") or params.get("train.patch_size") or "",
                "batch_size": params.get("train.batch_size") or "",
                "learning_rate": params.get("train.learning_rate") or params.get("learning_rate") or "",
                "run_url": run.get("url_mlflow_run") or "",
            }
        )
    return sorted(rows, key=lambda row: (row.get("metric_value") is not None, row.get("metric_value") or float("-inf")), reverse=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else [
        "run_id",
        "status",
        "run_name",
        "metric",
        "metric_value",
        "run_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
