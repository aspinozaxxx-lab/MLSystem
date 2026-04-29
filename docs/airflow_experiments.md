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

## Mini Acceptance Config

This config was used for the non-synthetic acceptance run `AIR-mini-deforest-r4`.

```json
{
  "experiment_id": "AIR-mini-deforest-r4",
  "class_name": "deforest",
  "task": "train_predict_pseudolabel",
  "smoke": false,
  "images_uri": "s3://mlsystems/images/",
  "layout_uri": "s3://mlsystems/layouts/deforest/",
  "scenes_file": "scenes.txt",
  "annotation_file": "auto",
  "model": {
    "name": "unet_small",
    "input_bands": [1, 2, 3, 4],
    "preview_bands": [4, 1, 2]
  },
  "preprocess": {
    "tile_size": 512,
    "stride": 384,
    "context": 64,
    "include_negative_scenes": false
  },
  "train": {
    "enabled": true,
    "time_limit_sec": 300,
    "epochs": 1,
    "batch_size": 1,
    "workers": 0
  },
  "pseudolabel": {
    "enabled": true,
    "run_on": "matched_scenes",
    "max_scenes": 2,
    "full_scene": false
  },
  "postprocess": {
    "enabled": true,
    "max_geojson_mb": 20,
    "max_objects": 500,
    "thresholds": [0.45],
    "min_object_area_m2_candidates": [1000],
    "simplify_tolerance_m_candidates": [5]
  },
  "mlflow": {
    "experiment": "mlsystem-airflow-acceptance"
  }
}
```

## Current Limitation

As of the acceptance run, Airflow executes real S3 layout checking, scene matching, status JSON, and MLflow run creation/finalization. Training, inference, stitching, vectorization, postprocessing, object F1, prediction examples, and artifact export are still placeholders in `src.pipeline.airflow_tasks`.
