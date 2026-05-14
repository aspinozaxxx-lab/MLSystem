# MLSystem Pipeline Runner

Airflow больше не является orchestration layer. Эксперименты запускаются напрямую через `mlsystem-api`, который сохраняет trace, стартует worker-процесс и последовательно выполняет stages.

## Запуск API

```bash
python -m uvicorn mlsystem.src.api.app:app --host 0.0.0.0 --port 8088
```

Основные переменные:

- `MLSYSTEM_RUN_ROOT=/data/mlsystem/runs`
- `MLSYSTEM_API_JOB_ROOT=/data/mlsystem/api/jobs`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_S3_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Trace

JSON/YAML trace принимает схему:

```yaml
schema_version: 1
run_id: cuttings_exp_001
experiment_id: cuttings_exp_001
class_name: cuttings
task: train_predict_pseudolabel
pipeline:
  stages:
    - inventory
    - prepare-dataset
    - train
    - evaluate
    - compute-f1
    - finalize
  stop_on_failure: true
  dry_run: false
  log_mlflow: true
images_uri: s3://mlsystems/images/
layout_uri: s3://mlsystems/layouts/deforest/
scenes_file: scenes.txt
annotation_file: auto
pseudolabel:
  enabled: false
train:
  max_epochs: 1
```

Aliases нормализуются в stage names: `inventory -> inventory_scenes`, `prepare-dataset -> prepare_dataset`, `train -> train_model`, `evaluate -> evaluate_pixel_metrics`, `compute-f1 -> compute_f1`, `finalize -> finalize_mlflow_run`.

## API

```bash
curl -fsS -X POST http://127.0.0.1:8088/api/v1/pipeline-runs \
  -H "Authorization: Bearer $MLSYSTEM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"trace":{"experiment_id":"smoke","pipeline":{"dry_run":true}}}'
```

Endpoints:

- `POST /api/v1/pipeline-runs`
- `GET /api/v1/pipeline-runs/{run_id}`
- `GET /api/v1/pipeline-runs/{run_id}/log?tail=20000`
- `GET /api/v1/pipeline-runs/{run_id}/stages`
- `POST /api/v1/pipeline-runs/{run_id}/cancel`

Все endpoints по умолчанию используют `require_api_token`.

## Файлы run

Runner пишет в `<MLSYSTEM_RUN_ROOT>/<run_id>/`:

- `trace.json`, `trace.yaml`
- `run.json`, `status.json`, `summary.json`
- `logs/pipeline.log`, `logs/stdout_tail.txt`, `logs/stderr_tail.txt`, `logs/<stage>.log`
- `stages/<stage>.json`, `stages/<stage>.report.md`
- `artifacts/`

## Progress

Progress считается по весам stages. По умолчанию вес stage равен `1`; `train_model` имеет вес `5`, inference stages имеют повышенный вес. Первая версия обновляет progress на границах stages, а структура `progress/<stage>.json` зарезервирована для epoch-level progress.

## MLflow

Существующие stages `create_mlflow_run`, `train_model`, `compute_f1`, `log_mlflow_artifacts`, `finalize_mlflow_run` сохраняют текущую MLflow-схему. Runner дополнительно best-effort логирует `pipeline.run_id`, список stages, stage reports и итоговый `summary.json`, не создавая дубль MLflow run.

## CLI

```bash
python -m mlsystem.src.pipeline_runner.cli run --trace configs/pipeline.smoke.local.yaml
python -m mlsystem.src.pipeline_runner.cli status --run-id smoke_pipeline_local
python -m mlsystem.src.pipeline_runner.cli log --run-id smoke_pipeline_local
```

Для локального smoke:

```bash
python scripts/debug_run_pipeline_without_airflow.py --trace configs/pipeline.smoke.local.yaml --wait
```

`dry_run` проверяет orchestration без GPU и без тяжёлой обработки данных.
