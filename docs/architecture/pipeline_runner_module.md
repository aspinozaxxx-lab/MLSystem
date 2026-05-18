# Модуль pipeline_runner

## Назначение

`mlsystem/src/pipeline_runner` - единственный orchestration модуль для lifecycle pipeline run.

Он отвечает за:

- создание run;
- хранение trace;
- запуск worker process;
- run status;
- stage lifecycle;
- logs;
- progress;
- cancellation;
- final summary;
- MLflow metadata logging через `mlflow_adapter`.

## Public API

- `PipelineRunConfig`;
- `PipelineRunner`;
- `PipelineRunStore`;
- CLI `python -m mlsystem.src.pipeline_runner.cli`;
- API endpoints в `mlsystem/src/api/app.py`, которые делегируют в `PipelineRunner`.

## Lifecycle

1. API/CLI валидирует trace.
2. `PipelineRunStore.create_run` создаёт `/data/mlsystem/runs/<run_id>/`.
3. `PipelineRunner.start_run` запускает worker process.
4. Worker последовательно выполняет stages.
5. Каждый stage пишет JSON/report/log.
6. Финальный статус и MLflow metadata фиксируются после завершения.

## Граница с pipeline

`pipeline_runner` не содержит domain stage logic.

`mlsystem/src/pipeline` содержит implementations stages и registry:

- inventory;
- prepare_dataset;
- train_model;
- evaluate;
- inference/pseudolabel stages;
- finalize implementation.

`pipeline` не должен владеть run lifecycle, store, API status или worker process.

## Worker diagnostics

Если worker умер без terminal state, runner должен записать:

- `WorkerExited`;
- pid;
- last stage;
- tails worker stdout/stderr;
- tail stage log;
- OOM/log hints, если они видны.
