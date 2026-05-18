from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .contracts import PipelineRunConfig, PipelineSpec


DEFAULT_PIPELINE_STAGES = [
    "inventory_scenes",
    "prepare_dataset",
    "create_mlflow_run",
    "train_model",
    "evaluate_pixel_metrics",
    "predict_validation_scenes",
    "vectorize_validation_predictions",
    "compute_f1",
    "generate_prediction_examples",
    "log_mlflow_artifacts",
    "write_codex_api_summary",
    "finalize_mlflow_run",
]

STAGE_ALIASES = {
    "inventory": "inventory_scenes",
    "prepare-dataset": "prepare_dataset",
    "train": "train_model",
    "evaluate": "evaluate_pixel_metrics",
    "compute-f1": "compute_f1",
    "finalize": "finalize_mlflow_run",
}

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_safe_id(value: str, *, field_name: str) -> str:
    if not value or not RUN_ID_RE.match(value):
        raise ValueError(f"{field_name} may contain only letters, digits, dot, underscore and dash")
    return value


def safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180] or "run"


def canonical_stage_name(value: str) -> str:
    normalized = value.strip()
    return STAGE_ALIASES.get(normalized, normalized.replace("-", "_"))


def parse_pipeline_run_config(config: PipelineRunConfig | dict[str, Any]) -> PipelineRunConfig:
    payload = config.model_dump(mode="json") if isinstance(config, PipelineRunConfig) else dict(config)
    if "experiment_id" in payload:
        payload["experiment_id"] = validate_safe_id(str(payload["experiment_id"]), field_name="experiment_id")
    pipeline_payload = dict(payload.get("pipeline") or {})
    stages = pipeline_payload.get("stages")
    if stages is None or stages == []:
        pipeline_payload["stages"] = list(DEFAULT_PIPELINE_STAGES)
    elif not isinstance(stages, list):
        raise ValueError("pipeline.stages must be a list")
    else:
        pipeline_payload["stages"] = [canonical_stage_name(str(item)) for item in stages]
    payload["pipeline"] = pipeline_payload
    run_id = payload.get("run_id")
    if run_id is not None:
        payload["run_id"] = validate_safe_id(str(run_id), field_name="run_id")
    parsed = PipelineRunConfig.model_validate(payload)
    annotations = dict(parsed.annotations or {})
    if _is_mlmarkup_source(annotations):
        resolved = _resolve_mlmarkup_annotation_config(annotations, parsed.class_name)
        parsed = parsed.model_copy(
            update={
                "annotations": resolved,
                "layout_uri": resolved["layout_uri"],
                "scenes_file": resolved["scenes_file"],
                "annotation_file": resolved["annotation_file"],
                "pseudolabel": {**(parsed.pseudolabel or {}), "enabled": False},
                "params": {
                    **(parsed.params or {}),
                    "annotations": resolved,
                    "pseudolabeling.enabled": False,
                },
            }
        )
    return parsed


def config_with_run_id(config: PipelineRunConfig, run_id: str) -> PipelineRunConfig:
    return config.model_copy(update={"run_id": validate_safe_id(run_id, field_name="run_id")})


def load_trace_file(path: str | Path) -> PipelineRunConfig:
    trace_path = Path(path)
    return parse_pipeline_run_config(load_trace_payload(trace_path.read_text(encoding="utf-8-sig"), source_name=trace_path.name))


def load_trace_payload(text: str, *, source_name: str = "trace.json") -> dict[str, Any]:
    suffix = Path(source_name).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text or "{}")
    if not isinstance(payload, dict):
        raise ValueError("pipeline trace must be a JSON/YAML object")
    return payload


def dump_trace_yaml(config: PipelineRunConfig) -> str:
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=True)


def _is_mlmarkup_source(annotations: dict[str, Any]) -> bool:
    return str(annotations.get("source") or "").strip().lower() in {"mlmarkup", "ml_markup", "ml-markup"}


def _resolve_mlmarkup_annotation_config(annotations: dict[str, Any], class_name: str | None) -> dict[str, Any]:
    repo_path = Path(str(annotations.get("repo_path") or os.getenv("MLSYSTEM_MLMARKUP_REPO_PATH") or "/data/mlsystem/MLMarkup"))
    class_dir = str(annotations.get("class_dir") or annotations.get("folder") or _default_mlmarkup_class_dir(class_name))
    scenes_file = str(annotations.get("scenes_file") or "deforestation.txt")
    annotation_file = str(annotations.get("annotation_file") or "deforestation.geojson")
    commit = str(annotations.get("commit") or _git_output(repo_path, "rev-parse", "HEAD") or "")
    branch = str(annotations.get("branch") or _git_output(repo_path, "branch", "--show-current") or "")
    dirty = bool(_git_output(repo_path, "status", "--short"))
    resolved = dict(annotations)
    resolved.update(
        {
            "source": "MLMarkup",
            "repo_path": str(repo_path),
            "class_dir": class_dir,
            "layout_uri": str(repo_path / class_dir),
            "scenes_file": scenes_file,
            "annotation_file": annotation_file,
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "use_pseudolabels": False,
        }
    )
    return resolved


def _default_mlmarkup_class_dir(class_name: str | None) -> str:
    normalized = str(class_name or "").strip().lower()
    if normalized in {"deforest", "cuttings", "clearcuts", "clear_cuts", "вырубки"}:
        return "Вырубки"
    return str(class_name or "").strip() or "Вырубки"


def _git_output(repo_path: Path, *args: str) -> str:
    fallback = _git_metadata_without_binary(repo_path, *args)
    if fallback is not None:
        return fallback
    try:
        git_bin = "/usr/bin/git" if Path("/usr/bin/git").exists() else "git"
        result = subprocess.run(
            [git_bin, "-C", str(repo_path), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_metadata_without_binary(repo_path: Path, *args: str) -> str | None:
    if args == ("status", "--short"):
        return ""
    git_dir = repo_path / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if args == ("branch", "--show-current"):
        if head.startswith("ref: refs/heads/"):
            return head.rsplit("/", 1)[-1]
        return ""
    if args == ("rev-parse", "HEAD"):
        if not head.startswith("ref: "):
            return head
        ref = head[5:].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            try:
                return ref_path.read_text(encoding="utf-8").strip()
            except OSError:
                return None
        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            try:
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    sha, _, packed_ref = line.partition(" ")
                    if packed_ref.strip() == ref:
                        return sha
            except OSError:
                return None
    return None
