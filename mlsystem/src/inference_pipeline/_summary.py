from __future__ import annotations

from typing import Any

from .contracts import PseudolabelRunRequest


def build_final_summary(
    *,
    request: PseudolabelRunRequest,
    run_id: str,
    job_id: str | None,
    final_state: dict[str, Any],
    artifacts: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_id": request.experiment_id,
        "source": "inference_engine",
        "submitted_via": "inference_pipeline",
        "inference_engine_job_id": job_id,
        "inference_engine_job": final_state,
        "status": final_state.get("status") or "success",
        "metrics": final_state.get("metrics") or {},
        "artifacts": artifacts,
        "warnings": warnings,
        "request": request.model_dump(mode="json"),
    }
