# MLSystem InferenceEngine

InferenceEngine is the pseudolabel service extracted from `mlsystem`. It owns scene planning, tile preprocessing, Triton inference descriptors, block/core/halo vectorization, final merge, and compatibility artifact export.

## API

- `GET /health`
- `GET /ready`
- `POST /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/events`
- `GET /api/v1/jobs/{job_id}/artifacts`
- `POST /api/v1/jobs/{job_id}/cancel`
- `GET /queues`
- `GET /metrics`

`POST /api/v1/jobs` returns `job_id` immediately. Airflow polls the status endpoint and reads final artifacts after `status=success`.

## Queues

RabbitMQ descriptors use durable JSON messages. Large arrays are written under `INFERENCE_ENGINE_JOB_ROOT` or `INFERENCE_ENGINE_SPOOL_ROOT`; queue messages contain only IDs, paths, checksums, and stage descriptors.

Queue names are defined in `inference_engine.queues.messages.QUEUE_NAMES`.

## Local Mode

When `INFERENCE_ENGINE_USE_RABBITMQ` is false, the API runs the same pipeline in a background task. This is intended for unit tests and smoke handtests without RabbitMQ.

## Run

```bash
pip install -e InferenceEngine
uvicorn inference_engine.api.app:app --host 0.0.0.0 --port 8095
```

Worker:

```bash
inference-engine-worker worker
```
