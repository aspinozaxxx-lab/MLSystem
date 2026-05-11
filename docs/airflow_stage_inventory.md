# Airflow Stage Inventory

## Production Flow

```text
inventory_scenes
prepare_dataset
create_mlflow_run
train_model
evaluate_pixel_metrics
predict_validation_scenes
vectorize_validation_predictions
compute_f1
inference_engine_pipeline
generate_prediction_examples
log_mlflow_artifacts
write_codex_api_summary
finalize_mlflow_run
```

Airflow has one pseudolabel stage: `inference_engine_pipeline`. It submits an HTTP job to InferenceEngine, polls the job status, validates the compatibility artifacts in the Airflow run directory, and writes a compact `StageReport`.

The retired internal pseudolabel stages are not part of `MAIN_DAG_STAGES`, Airflow task lists, or `/api/v1/stages` production output:

- `prepare_inference_scenes`
- `run_pseudolabel_inference`
- `validate_probability_maps`
- `vectorize_pseudolabel`
- `postprocess_pseudolabel`
- `export_pseudolabel_artifacts`

Deprecated modules with those names may remain only as compatibility imports/helpers. They are not registered as production stages.

## Stages

| # | Stage | Entrypoint | Pool | Notes |
| --- | --- | --- | --- | --- |
| 1 | `inventory_scenes` | `mlsystem.src.pipeline.stages.inventory_scenes.run` | `io_light` | Validates scene/layout inputs and writes inventory artifacts. |
| 2 | `prepare_dataset` | `mlsystem.src.pipeline.stages.prepare_dataset.run` | `cpu_heavy` | Builds the dataset manifest and train/val split. |
| 3 | `create_mlflow_run` | dispatcher in `airflow_tasks.py` | `io_light` | Creates the MLflow run and public URLs. |
| 4 | `train_model` | dispatcher in `airflow_tasks.py` | `gpu_training` | Runs training in `mlsystem`; training/MLflow semantics are unchanged. |
| 5 | `evaluate_pixel_metrics` | dispatcher in `airflow_tasks.py` | `cpu_light` | Extracts pixel metrics from `training_result.json`. |
| 6 | `predict_validation_scenes` | dispatcher in `airflow_tasks.py` | `gpu_inference` | Validation prediction compatibility gate. |
| 7 | `vectorize_validation_predictions` | dispatcher in `airflow_tasks.py` | `cpu_heavy` | Writes validation vectorization summary from available artifacts. |
| 8 | `compute_f1` | dispatcher in `airflow_tasks.py` | `cpu_heavy` | Writes pixel/object metric summary. |
| 9 | `inference_engine_pipeline` | `mlsystem.src.pipeline.stages.inference_engine_pipeline.run` | `io_light` | Calls `INFERENCE_ENGINE_API_URL`, polls `/api/v1/jobs/{job_id}`, and validates ready pseudolabel artifacts. |
| 10 | `generate_prediction_examples` | dispatcher in `airflow_tasks.py` | `cpu_heavy` | Checks `prediction_examples.html` written by InferenceEngine finalization. |
| 11 | `log_mlflow_artifacts` | dispatcher in `airflow_tasks.py` | `io_light` | Logs summary artifacts to MLflow. |
| 12 | `write_codex_api_summary` | dispatcher in `airflow_tasks.py` | `io_light` | Writes run summary files. |
| 13 | `finalize_mlflow_run` | dispatcher in `airflow_tasks.py` | `io_light` | Finalizes MLflow run and cleanup. |

## InferenceEngine Stage Contract

`inference_engine_pipeline` sends an HTTP request to:

```text
POST /api/v1/jobs
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/artifacts
```

The stage report includes:

- `source=inference_engine`
- `request_submitted_via_http=true`
- `inference_engine_api_url`
- `inference_engine_job_id`
- `inference_engine_status_url`
- `inference_engine_artifacts_url`

It validates these artifacts in `/data/mlsystem/airflow/status/<run_id>/`:

- `<experiment_id>.accepted.geojson`
- `accepted.geojson.gz`
- `coverage_report.json`
- `pseudolabel_summary.json`
- `postprocess_summary.json`
- `vectorization_summary.json`
- `pseudolabel_scene_results_manifest.json`
- `probability_maps_index.json`
- `inference_results.json`
- `inference_timing_report.json`
- `pseudolabel_scenes.txt`
- `prediction_examples.html`

Only small scalar XCom values are pushed. Heavy raster, probability, vectorization, and postprocess work is done by InferenceEngine workers through RabbitMQ and Triton.

## External Services

- `mlsystem-api`: Airflow stage execution endpoint and job persistence.
- InferenceEngine API: pseudolabel pipeline owner.
- RabbitMQ: InferenceEngine descriptor queues and dead-letter routing.
- Triton: model serving for pseudolabel inference.
- MinIO/S3: scene data and artifacts.
- MLflow: training run lifecycle and metrics.
- Airflow metadata DB: DAG/task state and scalar XCom.
