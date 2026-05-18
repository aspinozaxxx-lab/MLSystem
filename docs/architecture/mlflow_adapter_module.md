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
- `download_run_artifacts`;
- `get_run`;
- `list_artifact_paths`;
- `search_child_runs`;
- `search_runs`;
- `check_mlflow`;
- `log_lightweight_run`.

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
