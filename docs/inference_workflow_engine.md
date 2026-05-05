# Inference workflow engine

Каркас inference workflow engine находится в:

```text
mlsystem/src/workflow/inference/
```

Цель - постепенно вынести orchestration inference из `pseudolabel_pipeline.py`, не ломая текущий direct Triton production path.

## Backends

| Backend | Статус | Назначение |
|---|---|---|
| `direct_triton` | default | Обертка над текущим `SceneInferenceRunner` / Triton direct path. |
| `rabbitmq_triton` | experimental | Заготовка под RabbitMQ/Triton fan-out. По умолчанию выключена. |

## Feature flags

```json
{
  "pseudolabel": {
    "workflow_engine": {
      "enabled": false,
      "backend": "direct_triton",
      "max_scene_workers": 1,
      "rabbitmq_queue_prefix": "mlsystem.inference",
      "bad_scene_policy": "fail"
    }
  }
}
```

RabbitMQ backend не является production default.

## Vectorization jobs are Rabbit-ready

Новый `block_parallel` vectorization path использует JSON-serializable contracts:

- `PredictionTileInfo`;
- `ProcessingBlock`;
- `BlockVectorizationJob`;
- `BlockVectorizationResult`;
- `VectorizationPlan`.

Сейчас jobs исполняются через `ProcessPoolExecutor` внутри API/stage container. Контракт позволяет позже заменить локальный executor на RabbitMQ queue без изменения stage-level artifacts.

## RabbitMQ smoke

Локально или внутри контейнера:

```bash
python -m src.workflow.inference.rabbitmq_backend --smoke --enable
```

Через API:

```bash
curl -fsS -X POST http://mlsystem-api:8088/api/v1/debug/inference-rabbit-smoke
```

Endpoint требует API token, если `MLSYSTEM_API_TOKEN` задан.

## Что пока не production

- RabbitMQ backend не выполняет полный all-images inference.
- Нет отдельных production Rabbit workers для tile/block fan-out.
- Direct Triton path не переписан специально, чтобы не сломать текущий рабочий inference.
- `block_parallel` vectorization доступен через feature flag, но legacy остается default до серверной валидации на реальных сценах.
