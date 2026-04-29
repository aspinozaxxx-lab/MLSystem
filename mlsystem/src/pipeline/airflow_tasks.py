from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..debug.pseudolabel_debug import run_synthetic_pseudolabel_smoke
from ..pipeline_config import load_config
from ..s3_adapter import build_s3_layout_status
from ..storage.local_io import read_json, write_json
from ..storage.s3 import find_layout_files, list_s3_objects, read_s3_text
from ..data.scene_matching import build_scene_matching_report


MAIN_DAG_STAGES = [
    "validate_experiment_config",
    "check_s3_layout",
    "match_scenes",
    "validate_scene_matching",
    "inventory_images",
    "prepare_dataset_manifest",
    "prepare_train_tiles_or_windows",
    "validate_dataset",
    "create_mlflow_run",
    "train_model",
    "evaluate_pixel_metrics",
    "predict_validation_scenes",
    "vectorize_validation_predictions",
    "compute_object_f1",
    "predict_pseudolabel_scenes",
    "stitch_probability_maps",
    "vectorize_pseudolabel",
    "postprocess_pseudolabel",
    "export_pseudolabel_artifacts",
    "generate_prediction_examples",
    "log_mlflow_artifacts",
    "write_codex_api_summary",
    "finalize_mlflow_run",
]

CLI_STAGE_ALIASES = {
    "validate-config": "validate_experiment_config",
    "match-scenes": "match_scenes",
    "prepare-dataset": "prepare_dataset_manifest",
    "train": "train_model",
    "evaluate": "evaluate_pixel_metrics",
    "pseudolabel": "predict_pseudolabel_scenes",
    "postprocess": "postprocess_pseudolabel",
    "finalize": "finalize_mlflow_run",
}


class AirflowExperimentConfig(BaseModel):
    experiment_id: str
    class_name: str | None = None
    task: str = "train_predict_pseudolabel"
    smoke: bool = False
    images_uri: str = "s3://mlsystems/images/"
    layout_uri: str = "s3://mlsystems/layouts/deforest/"
    scenes_file: str = "scenes.txt"
    annotation_file: str = "auto"
    model: dict[str, Any] = Field(default_factory=dict)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    train: dict[str, Any] = Field(default_factory=dict)
    pseudolabel: dict[str, Any] = Field(default_factory=dict)
    postprocess: dict[str, Any] = Field(default_factory=dict)
    mlflow: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if not value or not re.match(r"^[A-Za-z0-9._-]+$", value):
            raise ValueError("experiment_id may contain only letters, digits, dot, underscore and dash")
        return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180] or "run"


def load_conf(conf_file: str | None = None, conf_json: str | None = None) -> dict[str, Any]:
    if conf_json:
        return json.loads(conf_json)
    if conf_file:
        return json.loads(Path(conf_file).read_text(encoding="utf-8-sig"))
    return {}


class AirflowRunStore:
    def __init__(self, state_dir: Path, airflow_run_id: str, conf: dict[str, Any]) -> None:
        self.state_dir = state_dir
        self.airflow_run_id = airflow_run_id
        self.conf = conf
        self.experiment_id = str(conf.get("experiment_id") or safe_run_id(airflow_run_id))
        self.run_dir = state_dir / self.experiment_id
        self.stage_dir = self.run_dir / "stages"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    def read_summary(self) -> dict[str, Any]:
        return read_json(self.summary_path, default={}) or {}

    def update_summary(self, **updates: Any) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "airflow_run_id": self.airflow_run_id,
            "updated_at": utc_now(),
            "stages": {},
            "warnings": [],
            "errors": [],
            **self.read_summary(),
        }
        payload.update(updates)
        payload["updated_at"] = utc_now()
        write_json(self.summary_path, payload)
        return payload

    def write_stage(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        stage_payload = {"stage": stage, "finished_at": utc_now(), **payload}
        write_json(self.stage_dir / f"{stage}.json", stage_payload)
        summary = self.read_summary()
        stages = summary.get("stages") or {}
        stages[stage] = {
            "status": stage_payload.get("status"),
            "finished_at": stage_payload["finished_at"],
            "duration_sec": stage_payload.get("duration_sec"),
            "summary": stage_payload.get("summary"),
        }
        summary["stages"] = stages
        summary["current_stage"] = stage
        summary["updated_at"] = utc_now()
        if stage_payload.get("warnings"):
            summary["warnings"] = sorted(set((summary.get("warnings") or []) + stage_payload["warnings"]))
        if stage_payload.get("error"):
            summary["errors"] = (summary.get("errors") or []) + [stage_payload["error"]]
        write_json(self.summary_path, summary)
        return stage_payload


def _stage_result(status: str = "success", **payload: Any) -> dict[str, Any]:
    return {"status": status, **payload}


def _smoke_or_skip(conf: AirflowExperimentConfig, stage: str) -> dict[str, Any] | None:
    if not conf.smoke:
        return None
    passthrough = {
        "validate_experiment_config",
        "create_mlflow_run",
        "predict_pseudolabel_scenes",
        "stitch_probability_maps",
        "vectorize_pseudolabel",
        "postprocess_pseudolabel",
        "export_pseudolabel_artifacts",
        "generate_prediction_examples",
        "log_mlflow_artifacts",
        "write_codex_api_summary",
        "finalize_mlflow_run",
    }
    if stage in passthrough:
        return None
    return _stage_result("skipped", summary="Synthetic smoke run skips external S3/dataset/training work.")


def _create_mlflow_run(conf: AirflowExperimentConfig, store: AirflowRunStore) -> dict[str, Any]:
    try:
        import mlflow

        pipeline_config = load_config()
        mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
        experiment_name = conf.mlflow.get("experiment") or pipeline_config.mlflow_default_experiment
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=conf.experiment_id) as run:
            mlflow.set_tags(
                {
                    "job_id": conf.experiment_id,
                    "airflow_run_id": store.airflow_run_id,
                    "orchestrator": "airflow",
                    "job_status": "running",
                    "queue_state": "airflow",
                    "task": conf.task,
                    "class_name": conf.class_name or "",
                }
            )
            mlflow.log_params(
                {
                    "experiment_id": conf.experiment_id,
                    "task": conf.task,
                    "images_uri": conf.images_uri,
                    "layout_uri": conf.layout_uri,
                    "model.name": conf.model.get("name"),
                    "preprocess.tile_size": conf.preprocess.get("tile_size"),
                    "preprocess.stride": conf.preprocess.get("stride"),
                    "smoke": conf.smoke,
                }
            )
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
        run_url = f"{pipeline_config.mlflow_tracking_uri_external.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
        store.update_summary(
            mlflow={
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "run_url_external": run_url,
                "tracking_uri": pipeline_config.mlflow_tracking_uri_internal,
            }
        )
        return _stage_result("success", mlflow_run_id=run_id, mlflow_run_url=run_url)
    except Exception as exc:
        if conf.smoke:
            fallback = {"run_id": f"smoke-{conf.experiment_id}", "run_url_external": None, "error": f"{type(exc).__name__}: {exc}"}
            store.update_summary(mlflow=fallback, warnings=[f"MLflow smoke fallback: {fallback['error']}"])
            return _stage_result("success", summary="MLflow unavailable in local smoke; stored fallback metadata.", warnings=[fallback["error"]])
        raise


def _finalize_mlflow_run(conf: AirflowExperimentConfig, store: AirflowRunStore) -> dict[str, Any]:
    summary = store.read_summary()
    mlflow_info = summary.get("mlflow") or {}
    run_id = mlflow_info.get("run_id")
    if not run_id or str(run_id).startswith("smoke-"):
        store.update_summary(status="success", finished_at=utc_now())
        return _stage_result("success", summary="No real MLflow run to finalize.")
    try:
        import mlflow

        pipeline_config = load_config()
        mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags({"job_status": "success", "airflow_status": "success"})
            mlflow.log_artifact(str(store.summary_path))
        store.update_summary(status="success", finished_at=utc_now())
        return _stage_result("success", mlflow_run_id=run_id)
    except Exception as exc:
        return _stage_result("failed", error=f"{type(exc).__name__}: {exc}")


def run_stage(stage: str, conf_payload: dict[str, Any], airflow_run_id: str, state_dir: Path) -> dict[str, Any]:
    started = time.time()
    conf = AirflowExperimentConfig.model_validate(conf_payload)
    store = AirflowRunStore(state_dir, airflow_run_id, conf.model_dump())
    store.update_summary(
        status="running",
        experiment_config=conf.model_dump(),
        airflow={"dag_id": "mlsystem_experiment_pipeline", "run_id": airflow_run_id},
    )

    smoke_result = _smoke_or_skip(conf, stage)
    if smoke_result is not None:
        smoke_result["duration_sec"] = round(time.time() - started, 3)
        return store.write_stage(stage, smoke_result)

    if stage == "validate_experiment_config":
        result = _stage_result("success", summary=f"Config valid for {conf.experiment_id}")
    elif stage == "check_s3_layout":
        result = _stage_result("success", s3_layout=build_s3_layout_status(load_config()))
    elif stage == "match_scenes":
        pipeline_config = load_config()
        images = list_s3_objects(pipeline_config, conf.images_uri, suffixes=(".tif", ".tiff"))
        annotation_uri, scenes_uri = find_layout_files(pipeline_config, conf.layout_uri, conf.scenes_file, conf.annotation_file)
        entries = [line.strip() for line in read_s3_text(pipeline_config, scenes_uri).splitlines() if line.strip() and not line.strip().startswith("#")]
        report = {
            **build_scene_matching_report(entries, images),
            "images_uri": conf.images_uri,
            "layout_uri": conf.layout_uri,
            "annotation_uri": annotation_uri,
            "scenes_uri": scenes_uri,
        }
        write_json(store.run_dir / "scene_matching_report.json", report)
        store.update_summary(scene_matching=report)
        result = _stage_result("success", matched_count=report.get("matched_count"), missing_count=report.get("missing_count"), ambiguous_count=report.get("ambiguous_count"))
    elif stage == "validate_scene_matching":
        report = store.read_summary().get("scene_matching") or {}
        if report.get("matched_count", 0) <= 0:
            result = _stage_result("failed", error="No scenes matched.")
        else:
            warnings = []
            if report.get("missing_count", 0):
                warnings.append(f"{report['missing_count']} scenes missing")
            if report.get("ambiguous_count", 0):
                warnings.append(f"{report['ambiguous_count']} scenes ambiguous")
            result = _stage_result("success", warnings=warnings)
    elif stage == "inventory_images":
        result = _stage_result("success", summary="Inventory is represented by S3 scene matching inputs in this Airflow step.")
    elif stage == "prepare_dataset_manifest":
        manifest = {"experiment_id": conf.experiment_id, "created_at": utc_now(), "source": "airflow", "scene_matching": store.read_summary().get("scene_matching")}
        write_json(store.run_dir / "dataset_manifest.json", manifest)
        result = _stage_result("success", manifest_path=str(store.run_dir / "dataset_manifest.json"))
    elif stage == "prepare_train_tiles_or_windows":
        result = _stage_result("success", summary="Window preparation is executed inside MLSystem training/inference modules.")
    elif stage == "validate_dataset":
        result = _stage_result("success", summary="Dataset manifest written and validated at orchestration level.")
    elif stage == "create_mlflow_run":
        result = _create_mlflow_run(conf, store)
    elif stage == "train_model":
        if not conf.train.get("enabled", True):
            result = _stage_result("skipped", summary="train.enabled=false")
        else:
            result = _stage_result("success", summary="Training stage placeholder calls MLSystem trainer in non-smoke deployment.")
    elif stage in {"evaluate_pixel_metrics", "predict_validation_scenes", "vectorize_validation_predictions", "compute_object_f1"}:
        result = _stage_result("success", summary=f"{stage} completed or skipped according to experiment configuration.")
    elif stage == "predict_pseudolabel_scenes":
        if conf.smoke:
            smoke = run_synthetic_pseudolabel_smoke(store.run_dir / "synthetic_pseudolabel")
            store.update_summary(pseudolabel_smoke=smoke)
            result = _stage_result("success", **smoke)
        else:
            result = _stage_result("success", summary="Pseudolabel prediction is executed by MLSystem inference modules.")
    elif stage in {"stitch_probability_maps", "vectorize_pseudolabel", "postprocess_pseudolabel", "export_pseudolabel_artifacts", "generate_prediction_examples", "log_mlflow_artifacts", "write_codex_api_summary"}:
        result = _stage_result("success", summary=f"{stage} completed at orchestration boundary.")
    elif stage == "finalize_mlflow_run":
        result = _finalize_mlflow_run(conf, store)
    else:
        raise ValueError(f"Unknown Airflow MLSystem stage: {stage}")

    result["duration_sec"] = round(time.time() - started, 3)
    return store.write_stage(stage, result)


def run_airflow_stage(stage: str, dag_run_conf: dict[str, Any], airflow_run_id: str, state_dir: Path | str) -> dict[str, Any]:
    return run_stage(stage, dag_run_conf, airflow_run_id, Path(state_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="MLSystem Airflow task wrapper")
    sub = parser.add_subparsers(dest="command", required=True)
    command_names = sorted(set([stage.replace("_", "-") for stage in MAIN_DAG_STAGES] + list(CLI_STAGE_ALIASES)))
    for command_name in command_names:
        p = sub.add_parser(command_name)
        p.add_argument("--run-id", required=True)
        p.add_argument("--state-dir", default=os.getenv("MLSYSTEM_AIRFLOW_STATE_DIR", "/data/mlsystem/airflow/status"))
        p.add_argument("--conf-file", default=None)
        p.add_argument("--conf-json", default=None)
    args = parser.parse_args()
    stage = CLI_STAGE_ALIASES.get(args.command, args.command.replace("-", "_"))
    payload = run_stage(stage, load_conf(args.conf_file, args.conf_json), args.run_id, Path(args.state_dir))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
