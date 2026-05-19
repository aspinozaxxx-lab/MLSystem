# Модуль mlflow_adapter

## Назначение

`mlsystem/src/mlflow_adapter/api.py` - единственная граница доступа к MLflow для production-кода MLSystem.

Остальные модули не импортируют `mlflow` напрямую.

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

## MLflow logging policy

`mlflow_adapter` — единственный владелец политики логирования в MLflow.

Он решает:

- какие training values логируются как MLflow metrics и видны в run table;
- какие values логируются как params/tags;
- какие values сохраняются artifacts;
- какие dataset metadata логируются через MLflow Dataset input.

Для training runs в MLflow metrics table разрешены только:

- `f1_pixel`;
- `epochs_total`;
- `epoch_time_sec`;
- `training_time_sec`.

Запрещено логировать как MLflow metrics:

- `val/*`;
- `train/*`;
- `diagnostics/*`;
- `system/*`;
- `resource/*`;
- `resources/*`;
- threshold sweep metrics;
- `*_best_threshold`;
- pixel counters/debug/microtiming metrics.

Эти данные должны сохраняться как artifacts:

- `history.json`;
- `history.csv`;
- `training_diagnostics.json`;
- `threshold_sweep_summary.json`, если есть;
- `system_metrics.json` / `resource_samples.json`, если есть.

Dataset column/input логируется через `log_dataset_input_to_run(...)`.

Public helper `log_training_result_to_run(config, run_id, train_result, *, dataset_identity=None, context="training")` принимает сырой `TrainResult`, извлекает 4 UI metrics, логирует history/diagnostics/system/threshold data как artifacts и не перекладывает эту ответственность на `train` или `train_pipeline`.

## Что запрещено другим модулям

- `import mlflow`;
- `mlflow.start_run`;
- `mlflow.log_metric`;
- `mlflow.log_artifact`;
- `MlflowClient`;
- прямое скачивание artifacts через `mlflow.artifacts`.

Исключение: тесты самого adapter.

## Lightweight tuning files

MLflow остаётся источником истины. Для удобства tuning можно вести derived files:

- `leaderboard.csv`;
- `leaderboard.json`;
- `tuning_state.json`.

Эти файлы не заменяют MLflow и должны строиться из MLflow/artifacts.

Утилита:

```bash
python scripts/export_mlflow_leaderboard.py --experiment mlsystem-cuttings-tuning --output-dir <dir>
```
