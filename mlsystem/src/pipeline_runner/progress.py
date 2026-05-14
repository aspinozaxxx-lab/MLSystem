from __future__ import annotations

from typing import Any


STAGE_WEIGHTS = {
    "train_model": 5.0,
    "predict_validation_scenes": 3.0,
    "inference_engine_pipeline": 3.0,
    "vectorize_validation_predictions": 2.0,
    "compute_f1": 2.0,
}


def stage_weight(stage: str) -> float:
    return float(STAGE_WEIGHTS.get(stage, 1.0))


def progress_percent(stages: list[str], completed: set[str], current_stage: str | None = None, current_stage_progress: float = 0.0) -> int:
    total = sum(stage_weight(stage) for stage in stages) or 1.0
    done = sum(stage_weight(stage) for stage in completed)
    if current_stage and current_stage not in completed:
        done += max(0.0, min(1.0, current_stage_progress)) * stage_weight(current_stage)
    return int(round(100.0 * done / total))


def stage_progress_from_file(payload: Any) -> float:
    if not isinstance(payload, dict):
        return 0.0
    for key in ("progress", "progress_fraction", "fraction"):
        value = payload.get(key)
        if value is not None:
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                return 0.0
    percent = payload.get("progress_percent")
    if percent is not None:
        try:
            return max(0.0, min(1.0, float(percent) / 100.0))
        except (TypeError, ValueError):
            return 0.0
    return 0.0
