# Airflow experiments

## UI

Open Airflow:

```text
http://172.26.12.169:8081
```

On the server or through SSH alias, `http://mlserver:8081` may also work.

Trigger experiments from DAG `mlsystem_experiment_pipeline` with JSON `dag_run.conf`.

## Status

Primary status source:

```text
GET /api/v1/dags/mlsystem_experiment_pipeline/dagRuns
GET /api/v1/dags/mlsystem_experiment_pipeline/dagRuns/{dag_run_id}
GET /api/v1/dags/mlsystem_experiment_pipeline/dagRuns/{dag_run_id}/taskInstances
```

Compact Codex-readable status is also written on the server:

```text
/data/mlsystem/airflow/status/<experiment_id>/summary.json
/data/mlsystem/airflow/status/<experiment_id>/stages/*.json
```

## Real Mini Acceptance Config

This config was used for the real ML acceptance run `AIR-real-mini-deforest-r5`.

```json
{
  "experiment_id": "AIR-real-mini-deforest-r5",
  "class_name": "deforest",
  "task": "train_predict_pseudolabel",
  "images_uri": "s3://mlsystems/images/",
  "layout_uri": "s3://mlsystems/layouts/deforest/",
  "scenes_file": "scenes.txt",
  "annotation_file": "auto",
  "model": {
    "name": "tiny_unet_4ch",
    "input_bands": [1, 2, 3, 4],
    "preview_bands": [4, 1, 2]
  },
  "preprocess": {
    "tile_size": 512,
    "stride": 384,
    "context": 64,
    "include_negative_scenes": true,
    "max_scenes": 2,
    "max_train_tiles": 64,
    "max_val_tiles": 16
  },
  "train": {
    "enabled": true,
    "time_limit_sec": 600,
    "max_epochs": 2,
    "batch_size": 1,
    "workers": 0,
    "augmentations": {
      "flips": true,
      "rot90": true
    }
  },
  "pseudolabel": {
    "enabled": true,
    "run_on": "validation_scenes",
    "max_scenes": 1,
    "full_scene": true
  },
  "postprocess": {
    "enabled": true,
    "max_geojson_mb": 20,
    "max_objects": 500,
    "keep_largest_if_too_many": true,
    "thresholds": [0.35, 0.5],
    "min_object_area_m2_candidates": [500, 1000],
    "simplify_tolerance_m_candidates": [2, 5]
  },
  "mlflow": {
    "experiment": "mlsystem-deforest"
  }
}
```

## Current Status

As of `AIR-real-mini-deforest-r5`, Airflow runs a real MLSystem mini pipeline:

- real S3 layout check and scene matching;
- dataset manifest and train/validation scene selection;
- `TrainingPipeline -> real_train.run_real_train` for real CPU training;
- real full-scene pseudolabel inference through `SceneInferenceRunner`;
- real stitching, vectorization, postprocess and object F1;
- MLflow metrics/artifacts logging;
- Airflow/Codex summaries under `/data/mlsystem/airflow/status/<experiment_id>/`.

The remaining compatibility seam is intentional: `training_pipeline.py` still delegates the heavy implementation to `real_train.run_real_train`. Airflow task IDs after `train_model` are real validation/reporting gates over files and metrics produced by the pipeline, but the physical ML computation is still executed inside the `train_model` stage until the legacy facade is split further.

Acceptance run:

```text
Airflow run: http://172.26.12.169:8081/dags/mlsystem_experiment_pipeline/grid?dag_run_id=AIR-real-mini-deforest-r5
MLflow run: http://172.26.12.169:5000/#/experiments/1/runs/6ba597b9d1e34200a5894a96cbd3873c
```
