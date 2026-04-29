# Airflow acceptance 2026-04-29

## Access

- Local server UI: `http://127.0.0.1:8081` returns Airflow redirect.
- External UI: `http://172.26.12.169:8081` returns HTTP 200 from the workstation.
- Health: `http://127.0.0.1:8081/api/v1/health` reports healthy metadatabase, scheduler, and triggerer.
- `http://mlserver:8081` is an SSH/local alias, not a guaranteed external DNS name.

## Legacy Queue

Checked services:

- `mlsystem-executor.service`: disabled, inactive.
- `mlsystem-web.service`: disabled, inactive.
- `mlsystem-preprocess.service`: disabled, inactive.

No legacy executor/preprocess process was found outside the check command itself.

## Workflows

- `cicd-code`: `workflow_dispatch` plus push on `mlsystem/**`, `airflow/**`, `configs/**`, and its own workflow file.
- `cicd-ansible`: `workflow_dispatch` plus push on `ansible/**`, `airflow/**`, and its own workflow file.
- `cicd-queue`: manual-only deprecated.
- `sync-results`: manual-only deprecated.

## Non-Synthetic Mini Run

- DAG: `mlsystem_experiment_pipeline`.
- Run: `AIR-mini-deforest-r4`.
- Airflow status: success.
- MLflow run: `1257f447516c416dab09fc89d80e74b3`.
- MLflow URL: `http://172.26.12.169:5000/#/experiments/3/runs/1257f447516c416dab09fc89d80e74b3`.

The run used real S3 URIs:

- `s3://mlsystems/images/`
- `s3://mlsystems/layouts/deforest/`

Real outputs produced by Airflow tasks:

- `/data/mlsystem/storage/system/s3_layout.json`
- `/data/mlsystem/airflow/status/AIR-mini-deforest-r4/scene_matching_report.json`
- `/data/mlsystem/airflow/status/AIR-mini-deforest-r4/dataset_manifest.json`
- `/data/mlsystem/airflow/status/AIR-mini-deforest-r4/summary.json`
- MLflow artifact: `summary.json`

MLflow params were logged. MLflow metrics were empty.

## Stage Classification

| task_id | status | function | artifacts |
|---|---|---|---|
| `validate_experiment_config` | real | `run_stage()` Pydantic validation | stage JSON |
| `check_s3_layout` | real | `build_s3_layout_status()` | `s3_layout.json`, stage JSON |
| `match_scenes` | real | `list_s3_objects()`, `find_layout_files()`, `read_s3_text()`, `build_scene_matching_report()` | `scene_matching_report.json`, summary scene_matching |
| `validate_scene_matching` | real | validates matched/missing/ambiguous counts | stage JSON |
| `inventory_images` | wrapper | summary boundary | stage JSON |
| `prepare_dataset_manifest` | wrapper | writes manifest from scene matching | `dataset_manifest.json` |
| `prepare_train_tiles_or_windows` | placeholder | no tile/window generation | stage JSON |
| `validate_dataset` | wrapper | manifest-level validation only | stage JSON |
| `create_mlflow_run` | real | `_create_mlflow_run()` | MLflow run + params |
| `train_model` | placeholder | no training loop | stage JSON |
| `evaluate_pixel_metrics` | placeholder | no metrics | stage JSON |
| `predict_validation_scenes` | placeholder | no inference | stage JSON |
| `vectorize_validation_predictions` | placeholder | no vectorization | stage JSON |
| `compute_object_f1` | placeholder | no object F1 | stage JSON |
| `predict_pseudolabel_scenes` | placeholder for non-smoke | no model inference | stage JSON |
| `stitch_probability_maps` | placeholder | no stitching | stage JSON |
| `vectorize_pseudolabel` | placeholder | no vectorization | stage JSON |
| `postprocess_pseudolabel` | placeholder | no postprocess | stage JSON |
| `export_pseudolabel_artifacts` | placeholder | no accepted GeoJSON | stage JSON |
| `generate_prediction_examples` | placeholder | no HTML examples | stage JSON |
| `log_mlflow_artifacts` | placeholder | no domain artifacts | stage JSON |
| `write_codex_api_summary` | wrapper | status JSON already written | stage JSON |
| `finalize_mlflow_run` | real | `_finalize_mlflow_run()` | MLflow `summary.json` artifact |

## Artifact Check

Expected production artifacts were not produced by the non-synthetic run:

- `train_scenes.txt`: missing.
- `pseudolabel_scenes.txt`: missing.
- accepted GeoJSON: missing.
- `codex_summary.json`: missing.
- `prediction_examples.html`: missing.
- `object_f1`: missing.

Forbidden legacy artifacts were not produced:

- `accepted.geojson.gz`
- `accepted.gpkg`
- `accepted_debug.geojson`
- `windows_preview.geojson`

XCom payloads were small; the largest checked XCom value for `AIR-mini-deforest-r4` was 2785 bytes.

## Verdict

Airflow is operational as the replacement queue/orchestration shell: UI, API, logs, DAG runs, S3 scene matching, MLflow run creation, and Codex JSON status work.

It is not yet accepted as a full real ML orchestrator. The core ML stages are still placeholders and must be wired to the refactored `training`, `inference`, `tiling`, `postprocessing`, `metrics`, and `reporting` modules.
