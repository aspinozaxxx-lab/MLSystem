# Inference workflow engine

Добавлен каркас inference workflow engine:

```text
mlsystem/src/workflow/inference/
```

Цель - постепенно вынести orchestration inference из `pseudolabel_pipeline.py`, не ломая текущий direct Triton path.

## Backends

| Backend | Статус | Назначение |
|---|---|---|
| `direct_triton` | default | Обертка над текущим безопасным direct Triton path. Production inference пока остается в `SceneInferenceRunner` / `pseudolabel_pipeline.py`. |
| `rabbitmq_triton` | experimental | Заготовка для параллельной обработки через RabbitMQ/Triton. По умолчанию выключена. |

## Feature flags

Планируемая конфигурация:

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

RabbitMQ backend не включается как default.

## RabbitMQ smoke

Локально/в контейнере:

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
- Нет production workers для tile/scene fan-out.
- Direct Triton path не переписан, чтобы не ломать текущий рабочий inference.
