# Airflow stage inventory

## Flow

```text
inventory_scenes
prepare_dataset
create_mlflow_run
train_model
evaluate_pixel_metrics
predict_validation_scenes
vectorize_validation_predictions
compute_f1
prepare_inference_scenes
run_pseudolabel_inference
validate_probability_maps
vectorize_pseudolabel
postprocess_pseudolabel
export_pseudolabel_artifacts
generate_prediction_examples
log_mlflow_artifacts
write_codex_api_summary
finalize_mlflow_run
```

## Stages

| # | Stage | Entrypoint | Pool | Notes |
| --- | --- | --- | --- | --- |
| 1 | `inventory_scenes` | `mlsystem.src.pipeline.stages.inventory_scenes.run` | `io_light` | Проверяет config/S3 layout/scenes/annotations и пишет inventory artifacts. |
| 2 | `prepare_dataset` | `mlsystem.src.pipeline.stages.prepare_dataset.run` | `cpu_heavy` | Читает inventory текущего run, считает objects per scene и пишет object-balanced dataset manifest. |
| 3 | `create_mlflow_run` | dispatcher in `airflow_tasks.py` | `io_light` | Создает MLflow run и URL. |
| 4 | `train_model` | dispatcher in `airflow_tasks.py` | `gpu_training` | Запускает текущий training pipeline без переписывания MLflow/training semantics. |
| 5 | `evaluate_pixel_metrics` | dispatcher in `airflow_tasks.py` | `cpu_light` | Извлекает доступные pixel metrics из `training_result.json`. |
| 6 | `predict_validation_scenes` | dispatcher in `airflow_tasks.py` | `gpu_inference` | Сейчас gate над доступными validation artifacts; если отдельный prediction не выполнен, status/summary говорят об этом явно. |
| 7 | `vectorize_validation_predictions` | dispatcher in `airflow_tasks.py` | `cpu_heavy` | Пишет validation vectorization summary по доступным artifacts. |
| 8 | `compute_f1` | dispatcher in `airflow_tasks.py` | `cpu_heavy` | Пишет pixel/object metric summary; unavailable object metrics не пушатся как metric XCom. |
| 9 | `prepare_inference_scenes` | `mlsystem.src.pipeline.stages.prepare_inference_scenes.run` | `io_light` | Формирует `inference_manifest.json`. |
| 10 | `run_pseudolabel_inference` | `mlsystem.src.pipeline.stages.pseudolabel_inference.run` | `gpu_inference` | Запускает direct Triton inference через текущий pipeline. |
| 11 | `validate_probability_maps` | `mlsystem.src.pipeline.stages.probability_maps.run` | `cpu_heavy` | Валидирует probability map index и coverage artifacts. |
| 12 | `vectorize_pseudolabel` | `mlsystem.src.pipeline.stages.vectorize_pseudolabel.run` | `cpu_heavy` | Поддерживает `legacy` и `block_parallel` modes. |
| 13 | `postprocess_pseudolabel` | `mlsystem.src.pipeline.stages.postprocess_pseudolabel.run` | `cpu_heavy` | Проверяет postprocess summary/artifacts. |
| 14 | `export_pseudolabel_artifacts` | `mlsystem.src.pipeline.stages.export_pseudolabel.run` | `io_light` | Проверяет export artifacts. |
| 15 | `generate_prediction_examples` | dispatcher in `airflow_tasks.py` | `cpu_heavy` | Проверяет `prediction_examples.html`. |
| 16 | `log_mlflow_artifacts` | dispatcher in `airflow_tasks.py` | `io_light` | Логирует summary artifacts в MLflow. |
| 17 | `write_codex_api_summary` | dispatcher in `airflow_tasks.py` | `io_light` | Пишет run summary files. |
| 18 | `finalize_mlflow_run` | dispatcher in `airflow_tasks.py` | `io_light` | Завершает MLflow run и cleanup. |

## External services

- MinIO/S3: scene discovery, layout files, training/inference artifacts.
- MLflow: runs, params, metrics, summary artifacts.
- Triton HTTP: production inference path.
- InferenceEngine API: pseudolabel pipeline owner when `pseudolabel.source=inference_engine`.
- RabbitMQ: InferenceEngine descriptor queues and dead-letter routing.
- Airflow metadata DB: DAG/task state and small scalar XCom only.
- Local status root: `/data/mlsystem/airflow/status/<run_id>/`.
- API job root: `/data/mlsystem/api/jobs/<job_id>/`.

## InferenceEngine Compatibility Mode

When DAG config sets `pseudolabel.source=inference_engine`, Airflow keeps the same stage list but changes pseudolabel stage responsibilities:

- `run_pseudolabel_inference` calls `INFERENCE_ENGINE_API_URL`, submits `/api/v1/jobs`, polls until terminal status, and expects compatibility artifacts in `/data/mlsystem/airflow/status/<run_id>/<experiment_id>/`.
- `validate_probability_maps` validates `coverage_report.json` and `probability_maps_index.json`.
- `vectorize_pseudolabel` is validate-only and skips mlsystem CPU vectorization.
- `postprocess_pseudolabel` is validate-only and reads `postprocess_summary.json`.
- `export_pseudolabel_artifacts` keeps the old artifact presence checks.

Training, MLflow run lifecycle, validation predictions, metrics, and final MLflow logging remain in `mlsystem`.
