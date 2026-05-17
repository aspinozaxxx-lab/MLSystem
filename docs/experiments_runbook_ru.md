# Runbook запуска экспериментов без Airflow

## Контур запуска

Эксперименты запускаются через `mlsystem-api` и `pipeline_runner`.
Airflow для tuning/loader smoke не используется.

Основной поток:

1. Подготовить trace JSON/YAML.
2. Отправить `POST /api/v1/pipeline-runs`.
3. Проверять `GET /api/v1/pipeline-runs/{run_id}`.
4. Читать лог через `GET /api/v1/pipeline-runs/{run_id}/log`.
5. Читать stage reports через `GET /api/v1/pipeline-runs/{run_id}/stages`.
6. Проверить MLflow run, metrics, artifacts и локальные run files.

Run files лежат в:

```text
/data/mlsystem/runs/<run_id>/
```

MLflow UI:

```text
http://31.192.104.147/mlflow/
```

Derived leaderboard files можно выгрузить без запуска экспериментов:

```bash
python scripts/export_mlflow_leaderboard.py \
  --experiment mlsystem-cuttings-tuning \
  --output-dir /data/mlsystem/tuning/cuttings/leaderboard
```

Скрипт пишет `leaderboard.csv`, `leaderboard.json`, `tuning_state.json`. Источник истины остаётся MLflow.

## Минимальный trace

```yaml
schema_version: 1
experiment_id: cuttings_smoke_YYYYMMDD_HHMM
class_name: cuttings
pipeline:
  stages: [inventory, prepare-dataset, train, finalize]
  stop_on_failure: true
  dry_run: false
  log_mlflow: true
annotations:
  source: MLMarkup
  repo_path: /data/mlsystem/MLMarkup
  class_dir: Вырубки
  scenes_file: deforestation.txt
  annotation_file: deforestation.geojson
pseudolabel:
  enabled: false
preprocess:
  tile_size: 512
  stride: 512
  train_sampling:
    enabled: true
train:
  max_epochs: 2
  batch_size: 4
  augmentation_level: 1
  require_gpu: true
model:
  name: tiny_unet_4ch
mlflow:
  experiment: mlsystem-cuttings-tuning
```

## Запуск через FastAPI

```bash
curl -fsS -X POST http://127.0.0.1:8088/api/v1/pipeline-runs \
  -H "Authorization: Bearer $MLSYSTEM_API_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @trace.json
```

Статус:

```bash
curl -fsS http://127.0.0.1:8088/api/v1/pipeline-runs/<run_id> \
  -H "Authorization: Bearer $MLSYSTEM_API_TOKEN"
```

Лог:

```bash
curl -fsS "http://127.0.0.1:8088/api/v1/pipeline-runs/<run_id>/log?tail=40000" \
  -H "Authorization: Bearer $MLSYSTEM_API_TOKEN"
```

Stage reports:

```bash
curl -fsS http://127.0.0.1:8088/api/v1/pipeline-runs/<run_id>/stages \
  -H "Authorization: Bearer $MLSYSTEM_API_TOKEN"
```

## Запуск через CLI

```bash
python -m mlsystem.src.pipeline_runner.cli run --trace trace.yaml
python -m mlsystem.src.pipeline_runner.cli status --run-id <run_id>
python -m mlsystem.src.pipeline_runner.cli log --run-id <run_id>
```

CLI удобен для локальной отладки. Серверные tuning runs предпочтительно запускать через API, чтобы состояние было видно в общем run store.

## Валидация run

Run считается пригодным к анализу, если:

- `state=succeeded`;
- есть MLflow `run_id`;
- `training_result.json` содержит `epochs_completed`;
- `history.csv/json` содержит все эпохи;
- train/val scene counts и tile counts соответствуют trace;
- нет скрытых `max_train_batches` / `max_val_batches`;
- validation без augmentation, random jitter и virtual repeats;
- epoch duration правдоподобна;
- есть DataLoader metrics:
  - `data/batch_wait_total_sec`;
  - `data/batch_wait_mean_sec`;
  - `data/batch_wait_p95_sec`;
  - `data/samples_per_sec`;
  - `data/batches_per_sec`.

Guardrails для вырубок:

- epoch duration меньше 10 секунд на полном датасете считается invalid и требует расследования;
- best epoch `<= 5` для длинного run считается suspicious, если нет отдельного объяснения;
- pseudo-labeling должен быть disabled;
- оптимизируем pixel-level F1, но смотрим precision/recall и threshold.

## Scratch, fine-tune, eval-only

Scratch:

- не задавать `initial_checkpoint_path`;
- не задавать `checkpoint_finetune=true`;
- указать model/loss/LR/scheduler/batch/augmentation явно.

Fine-tune:

- явно указать checkpoint;
- в `params.tuning` записать гипотезу и reference checkpoint;
- проверить, что MLflow tags/params отражают источник checkpoint.

Eval-only:

- указать checkpoint;
- включить eval-only режим через поддерживаемый trace параметр;
- не включать train augmentation.

## Параметры, которые можно менять в tuning

- `preprocess.tile_size`;
- `preprocess.stride`;
- `train.augmentation_level`;
- `train.batch_size`;
- `train.loss`;
- `train.learning_rate`;
- `train.weight_decay`;
- `train.scheduler`.

DataLoader workers, prefetch, pin memory and persistent worker policy are internal `tile_preparation` settings, not experiment trace parameters. Emergency diagnostics can override them with environment variables, but normal traces should not set them.

## Что не менять без согласования

- публичный API `tile_preparation`;
- train/val split semantics;
- validation behavior;
- перенос tile reading/rasterization/augmentation из `tile_preparation` в training loop;
- Airflow automation, supervisor loops, shell loops и повторные автоматические запуски экспериментов.
