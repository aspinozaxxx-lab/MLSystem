# Модуль mlflow_adapter

## Назначение

`mlsystem/src/mlflow_adapter/api.py` - единственная граница доступа к MLflow для production-кода MLSystem.
Остальные модули не импортируют `mlflow` напрямую.
`mlflow_adapter` — единственный владелец политики логирования в MLflow.

## Public методы

Основные методы:
- `MLflowJobRun`;
- `start_job_run`;
- `create_run`;
- `set_run_tags`;
- `set_experiment_tags`;
- `log_metrics_to_run`;
- `log_params_to_run`;
- `log_artifacts_to_run`;
- `log_dataset_input_to_run`;
- `log_training_result_to_run`;
- `download_run_artifacts`;
- `get_run`;
- `list_artifact_paths`;
- `search_child_runs`;
- `search_runs`;
- `check_mlflow`;
- `log_lightweight_run`.

Для training runs в MLflow metrics table разрешены только:
- `f1_pixel`;
- `epochs_total`;
- `epoch_time_sec`;
- `training_time_sec`.





