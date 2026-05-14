from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DEFAULT_PIPELINE_STAGES = [
    "inventory_scenes",
    "prepare_dataset",
    "create_mlflow_run",
    "train_model",
    "evaluate_pixel_metrics",
    "predict_validation_scenes",
    "vectorize_validation_predictions",
    "compute_f1",
    "inference_engine_pipeline",
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
    "inference-engine": "inference_engine_pipeline",
    "pseudolabel": "inference_engine_pipeline",
}

RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class PipelineSpec(BaseModel):
    stages: list[str] = Field(default_factory=lambda: list(DEFAULT_PIPELINE_STAGES))
    stop_on_failure: bool = True
    dry_run: bool = False
    log_mlflow: bool = True

    @field_validator("stages", mode="before")
    @classmethod
    def normalize_stages(cls, value: Any) -> list[str]:
        if value is None:
            return list(DEFAULT_PIPELINE_STAGES)
        if not isinstance(value, list):
            raise ValueError("pipeline.stages must be a list")
        return [canonical_stage_name(str(item)) for item in value]


class PipelineRunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    run_id: str | None = None
    experiment_id: str
    class_name: str | None = None
    task: str = "train_predict_pseudolabel"
    smoke: bool = False

    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)

    images_uri: str = "s3://mlsystems/images/"
    layout_uri: str = "s3://mlsystems/layouts/deforest/"
    scenes_file: str = "scenes.txt"
    annotation_file: str = "auto"

    model: dict[str, Any] = Field(default_factory=dict)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    train: dict[str, Any] = Field(default_factory=dict)
    evaluate: dict[str, Any] = Field(default_factory=dict)
    pseudolabel: dict[str, Any] = Field(default_factory=dict)
    postprocess: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    predict: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    mlflow: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        return validate_safe_id(value, field_name="experiment_id")

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_safe_id(value, field_name="run_id")

    @model_validator(mode="after")
    def resolve_annotation_source(self) -> "PipelineRunConfig":
        annotations = dict(self.annotations or {})
        if _is_mlmarkup_source(annotations):
            resolved = _resolve_mlmarkup_annotation_config(annotations, self.class_name)
            self.annotations = resolved
            self.layout_uri = resolved["layout_uri"]
            self.scenes_file = resolved["scenes_file"]
            self.annotation_file = resolved["annotation_file"]
            self.pseudolabel = {**(self.pseudolabel or {}), "enabled": False}
            self.params = {
                **(self.params or {}),
                "annotations": resolved,
                "pseudolabeling.enabled": False,
            }
        return self

    def with_run_id(self, run_id: str) -> "PipelineRunConfig":
        return self.model_copy(update={"run_id": validate_safe_id(run_id, field_name="run_id")})


def validate_safe_id(value: str, *, field_name: str) -> str:
    if not value or not RUN_ID_RE.match(value):
        raise ValueError(f"{field_name} may contain only letters, digits, dot, underscore and dash")
    return value


def safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180] or "run"


def canonical_stage_name(value: str) -> str:
    normalized = value.strip()
    return STAGE_ALIASES.get(normalized, normalized.replace("-", "_"))


def load_trace_file(path: str | Path) -> PipelineRunConfig:
    trace_path = Path(path)
    return PipelineRunConfig.model_validate(load_trace_payload(trace_path.read_text(encoding="utf-8-sig"), source_name=trace_path.name))


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
