# InferenceEngine Validation Report

Status: completed on the GPU server through GitHub Actions, Ansible-managed Docker services, curl, Airflow, RabbitMQ, Triton, and the authenticated frontend gateway.

## Commit And Workflows

- Runtime code validated on the server: `034123d1fa8953ed354c74bde7a0ed94acc40a5b`.
- Server health after validation:
  - `mlsystem-api /health`: `034123d1fa8953ed354c74bde7a0ed94acc40a5b`
  - `InferenceEngine /health`: `034123d1fa8953ed354c74bde7a0ed94acc40a5b`
- Validation artifact on the server: `/tmp/inference_engine_server_validation_gateway.json`.
- GitHub Actions observed for the runtime commit:
  - `mlservice` run `29`: success.
  - `InferenceEngine service` run `40`: success.
- Ansible infrastructure rollout was verified through the repo workflow path and server checks; the relevant `ansible` run `35` succeeded, and all affected containers were recreated from repository-managed compose/templates. No server compose, env, or secrets were edited by hand.

## Deployed Services

Docker validation observed these service containers running:

- `mlsystem-gpu-inference-engine-api`
- `mlsystem-gpu-inference-engine-planner`
- `mlsystem-gpu-inference-engine-preprocess-worker`
- `mlsystem-gpu-inference-engine-triton-worker`
- `mlsystem-gpu-inference-engine-aggregator`
- `mlsystem-gpu-inference-engine-block-worker`
- `mlsystem-gpu-inference-engine-merger`
- `mlsystem-gpu-inference-engine-finalizer`
- `mlsystem-gpu-rabbitmq`
- `mlsystem-gpu-triton`
- `mlsystem-gpu-api`
- Airflow scheduler/webserver/triggerer
- `mlsystem-gpu-frontend` and `mlsystem-gpu-frontend-proxy`
- `mlsystem-gpu-mlflow`
- `mlsystem-gpu-minio`

`/ready` reported job, spool, artifact, and log roots available, RabbitMQ management metrics available, all InferenceEngine queues visible, and Triton ready.

## InferenceEngine API

OpenAPI exposed these endpoints:

- `GET /health`
- `GET /ready`
- `GET /queues`
- `GET /metrics`
- `POST /api/v1/jobs`
- `GET /api/v1/jobs/{job_id}`
- `GET /api/v1/jobs/{job_id}/events`
- `GET /api/v1/jobs/{job_id}/artifacts`
- `POST /api/v1/jobs/{job_id}/cancel`
- `GET /openapi.json`
- `GET /docs`
- `GET /redoc`

## Authenticated Admin Gateway

Public entrypoint:

- `http://31.192.104.147/`

Frontend cards and authenticated UI routes:

- `Airflow`: `http://31.192.104.147/airflow/`
- `MLflow`: `http://31.192.104.147/mlflow/`
- `MinIO artifacts`: `http://31.192.104.147/minio-browser/`
- `Очереди RabbitMQ`: `http://31.192.104.147/rabbitmq/`

Gateway checks:

- With a valid frontend session, `/auth/proxy-check` returned `204`.
- With a valid frontend session:
  - `/airflow/home` returned `200` and did not show the Airflow login form.
  - `/mlflow/` returned `200`.
  - `/rabbitmq/` returned `200`.
  - `/minio-browser/` returned `200`.
- Without a frontend session, `/airflow/`, `/mlflow/`, `/rabbitmq/`, and `/minio-browser/` redirected to frontend login or returned an unauthorized gateway response.

Implementation:

- Nginx protects admin UI locations with `auth_request /auth/proxy-check`.
- Airflow runs behind `/airflow/` with remote-user auth from the trusted frontend proxy, so `mluser` does not see a second Airflow login.
- MLflow remains behind frontend auth; its direct UI is not the public entrypoint.
- RabbitMQ Management runs under `/rabbitmq/`; Nginx injects the Basic Authorization header server-side from environment, so RabbitMQ credentials are not present in HTML, JavaScript, URLs, docs, or git.
- Native MinIO Console SSO was not enabled because the current frontend session is not an OIDC/LDAP identity provider. The safe alternative is `/minio-browser/`, a frontend-authenticated read-only browser that uses S3 credentials only server-side.

## Airflow And MLSystem API

`mlsystem-api /api/v1/stages` and the Airflow task list expose one pseudolabel production task:

```text
inventory_scenes
prepare_dataset
create_mlflow_run
train_model
evaluate_pixel_metrics
predict_validation_scenes
vectorize_validation_predictions
compute_f1
inference_engine_pipeline
generate_prediction_examples
log_mlflow_artifacts
write_codex_api_summary
finalize_mlflow_run
```

The old internal pseudolabel stages are absent from the production DAG and API stage list:

```text
prepare_inference_scenes
run_pseudolabel_inference
validate_probability_maps
vectorize_pseudolabel
postprocess_pseudolabel
export_pseudolabel_artifacts
```

`inference_engine_pipeline` submits InferenceEngine over HTTP, polls the job status, reads artifacts, validates the compatibility files in `/data/mlsystem/airflow/status/<run_id>/`, and writes only compact stage metadata. The heavy pseudolabel domain work runs in InferenceEngine workers through RabbitMQ and Triton.

## Airflow DAG Validation

The real 2-scene validation was executed as an Airflow DAG run, not as a direct stage call.

- DAG id: `mlsystem_experiment_pipeline`
- DAG run id: `ie_airflow_real_2_1778511884`
- UI link: `/airflow/dags/mlsystem_experiment_pipeline/grid?dag_run_id=ie_airflow_real_2_1778511884`
- DAG state: `success`
- InferenceEngine job id: `ie_airflow_real_2_1778511884-8ec69761f05a`

Task states:

```text
compute_f1: success
create_mlflow_run: success
evaluate_pixel_metrics: success
finalize_mlflow_run: success
generate_prediction_examples: success
inference_engine_pipeline: success
inventory_scenes: success
log_mlflow_artifacts: success
predict_validation_scenes: success
prepare_dataset: success
train_model: success
vectorize_validation_predictions: success
write_codex_api_summary: success
```

The run is visible in Airflow via `airflow dags list-runs -d mlsystem_experiment_pipeline` and the `/airflow/` UI.

## RabbitMQ

Final queue snapshot after validation:

```text
name                  ready  unacked  consumers
ie.jobs.submit        0      0        1
ie.scene.plan         0      0        1
ie.tile.preprocess    0      0        8
ie.tile.infer         0      0        2
ie.tile.done          0      0        1
ie.block.ready        0      0        4
ie.block.vectorize    0      0        4
ie.block.done         0      0        1
ie.scene.merge        0      0        1
ie.job.finalize       0      0        1
ie.events             41003  0        0
ie.dead_letter        0      0        0
```

All stage queues had consumers. `ie.dead_letter` stayed empty.

## Triton

Triton `/v2/health/ready` returned ready. The validation used:

- MLflow run: `a7838f91528a47e1931b685c2ea06686`
- Triton model: `segformer_b2`
- Interface: `INPUT__0` / `OUTPUT__0`

The `identity_python` smoke model remained in the repository and was not removed.

## 2 Scene Airflow Run

- Path: real Airflow DAG run -> `mlsystem-api` -> `inference_engine_pipeline` -> InferenceEngine API.
- DAG run id: `ie_airflow_real_2_1778511884`.
- InferenceEngine job id: `ie_airflow_real_2_1778511884-8ec69761f05a`.
- Result: success.
- Metrics:
  - `tiles_done=760/760`
  - `blocks_done=30/30`
  - `triton_batches=131`
  - `triton_batch_fill_ratio=0.7251908396946565`
  - mean Triton request `413.46 ms`
  - `spool_bytes=0`
- Streaming proof:
  - `first_block_vectorized_at=1778512159.3154213`
  - `last_tile_inferred_at=1778512206.045915`
  - `streaming_overlap_sec=46.73049354553223`

Compatibility artifacts checked in `/data/mlsystem/airflow/status/ie_airflow_real_2_1778511884/`:

- `ie_airflow_real_2_1778511884.accepted.geojson`
- `accepted.geojson.gz`
- `coverage_report.json`
- `pseudolabel_summary.json`
- `postprocess_summary.json`
- `vectorization_summary.json`
- `pseudolabel_scene_results_manifest.json`
- `probability_maps_index.json`
- `inference_results.json`
- `inference_timing_report.json`
- `pseudolabel_scenes.txt`
- `prediction_examples.html`

## 20 Scene Performance Run

- Path: direct InferenceEngine API job through RabbitMQ workers after Airflow integration was validated.
- Job id: `ie_real_20-1e97c074a989`.
- Result: success.
- Metrics:
  - `tiles_done=6638/6638`
  - `blocks_done=272/272`
  - `triton_batches=1143`
  - `triton_batch_fill_ratio=0.7259405074365705`
  - mean Triton request `393.08 ms`
  - `spool_bytes=0`
- Stage events:
  - `scene.plan=20`
  - `tile.preprocess=6638`
  - `tile.infer=1143`
  - `tile.done=6638`
  - `block.ready=272`
  - `block.vectorize=272`
  - `block.done=272`
  - `scene.merge=20`
  - `job.finalize=1`
- Streaming proof:
  - `first_block_vectorized_at=1778512310.8842049`
  - `last_tile_inferred_at=1778513295.547272`
  - `streaming_overlap_sec=984.6630671024323`

The 20-scene run sampled GPU utilization up to `100%` during the active phase. The final stored job-level GPU metric was reset to `0` after completion, so the validation report uses the live sampled maximum for utilization. CPU preprocess and block/vectorization workers were active while tile inference was still running.

## Tuned Parameters

Server defaults:

- `INFERENCE_ENGINE_PREPROCESS_CONCURRENCY=8`
- `INFERENCE_ENGINE_TRITON_CONCURRENCY=2`
- `INFERENCE_ENGINE_BLOCK_CONCURRENCY=4`
- `INFERENCE_ENGINE_DEFAULT_TRITON_BATCH_SIZE=8`
- `INFERENCE_ENGINE_MAX_WAIT_MS=75`

20-scene resource config:

- `batches_ahead=12`
- `max_preprocess_queue=1024`
- `max_scenes_inflight=4`
- `max_blocks_inflight=32`
- `max_spool_bytes=30 GiB`

The configuration kept the spool bounded, kept all stage queues draining, and proved overlap between inference and block/vectorization work.

## Fixes Applied During Validation

- InferenceEngine code deploy was preserving `mlsystem/configs/pipeline.server.yaml` after rsync, preventing missing server config.
- `inference_engine_pipeline` now passes the shared Airflow status path `/data/mlsystem/airflow/status/<run_id>` to InferenceEngine, so compatibility artifacts appear where downstream stages expect them.
- Final cleanup now treats permission-denied cleanup of root-owned InferenceEngine intermediates as best-effort and does not fail a successful DAG run.
- The frontend session cookie was separated from Airflow cookies to avoid route-level auth collisions.
- Airflow, MLflow, RabbitMQ, and MinIO artifact access were moved behind the authenticated frontend gateway.

## Acceptance

- InferenceEngine is deployed as a separate service with dedicated worker containers.
- Airflow and `mlsystem-api` call InferenceEngine through HTTP.
- Airflow shows one production pseudolabel task: `inference_engine_pipeline`.
- A real Airflow DAG run is visible in the Airflow UI and completed successfully.
- Frontend home exposes Airflow, MLflow, MinIO artifacts, and RabbitMQ Management through authenticated routes.
- RabbitMQ stage queues are used by all pipeline steps.
- Triton performs inference with `segformer_b2`.
- Block/core/halo postprocessing starts before all tile inference completes.
- 2-scene Airflow and 20-scene performance validations completed successfully.
- Compatibility artifacts are present.
- `ie.dead_letter` stayed empty.
- No runtime artifacts, probability tiles, accepted GeoJSON outputs, server env files, or secrets are committed.
