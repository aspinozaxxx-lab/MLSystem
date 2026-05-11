# InferenceEngine Validation Report

Status: completed on the GPU server through GitHub Actions, Ansible, Docker, curl, Airflow, RabbitMQ, and Triton.

## Commit And Workflows

- Validated service commit: `a4cbd0ca2ecd3ce509d8a97dab70bb8ce13564fe`.
- Validation generated at: `2026-05-11T10:40:06Z`.
- `InferenceEngine service` workflow run `25665009150`: `test`, `build-artifact`, `deploy-service-code`, `validate-service`, and `validate-real-server` succeeded.
- `ansible` workflow run `25664452452`: syntax, plan/check, apply infra, and validation succeeded.
- `InferenceEngine infra` workflow run `25664452478`: syntax, plan/check, apply infra, and validation succeeded.
- `frontend-ansible` workflow run `25664452417`: frontend deploy and validation succeeded.
- Validation artifact: `inference-engine-server-validation`, artifact id `6916474727`.

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

`/health` returned commit `a4cbd0ca2ecd3ce509d8a97dab70bb8ce13564fe`. `/ready` returned `ready` with job, spool, artifact, and log roots available, RabbitMQ management metrics available, 12 queues seen, and Triton ready at `http://triton:8000/v2/health/ready`.

## API Endpoints

OpenAPI exposed:

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

## Airflow And MLSystem API

`mlsystem-api /api/v1/stages` returned one production pseudolabel stage:

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

The Airflow task list for `mlsystem_experiment_pipeline` matched that list and did not include the retired internal stages `prepare_inference_scenes`, `run_pseudolabel_inference`, `validate_probability_maps`, `vectorize_pseudolabel`, `postprocess_pseudolabel`, or `export_pseudolabel_artifacts`.

The 2-scene validation executed `inference_engine_pipeline` through `mlsystem-api`. The stage report recorded:

- `inference_engine_api_url=http://inference-engine-api:8095`
- `inference_engine_job_id=ie_real_2_1778496009-a8461eb048bb`
- `inference_engine_status_url=http://inference-engine-api:8095/api/v1/jobs/ie_real_2_1778496009-a8461eb048bb`
- `request_submitted_via_http=true`
- `source=inference_engine`

## RabbitMQ UI

- RabbitMQ Management UI URL: `http://31.192.104.147:15672/`.
- The management API requires authentication; unauthenticated access is rejected.
- The frontend home page contains the `Очереди RabbitMQ` card, which links to the native RabbitMQ UI and reads compact queue metrics from InferenceEngine `/queues`.

## Triton

Triton `/v2/health/ready` was ready. Repository index contained:

- `segformer_b2`, version `1`, `READY`
- `identity_python`, version `1`, `READY`
- `deforest_segformer_b1_t1024`

The validation jobs used MLflow run `a7838f91528a47e1931b685c2ea06686` with Triton model `segformer_b2` and `INPUT__0` / `OUTPUT__0`.

## 2 Scene Run

- Path: Airflow-compatible `mlsystem-api` stage `inference_engine_pipeline`.
- Run id: `ie_real_2_1778496009`.
- InferenceEngine job id: `ie_real_2_1778496009-a8461eb048bb`.
- Result: success.
- Metrics: `tiles_done=760/760`, `blocks_done=30/30`, `triton_batches=114`, `triton_batch_fill_ratio=0.833`, mean Triton request `451.79 ms`, `spool_bytes=0`.
- Streaming proof: `first_block_vectorized_at=1778496032.195401`, `last_tile_inferred_at=1778496050.6956096`, overlap `18.500 s`.
- GPU sample in job metrics: max `49%`.

Artifacts checked in `/data/mlsystem/airflow/status/ie_real_2_1778496009/`:

- `ie_real_2_1778496009.accepted.geojson`
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

## 20 Scene Run

- Path: direct InferenceEngine API job through RabbitMQ workers after Airflow integration had been validated.
- Job id: `ie_real_20-147aa0db6e1a`.
- Result: success.
- Metrics: `tiles_done=6638/6638`, `blocks_done=272/272`, `triton_batches=983`, `triton_batch_fill_ratio=0.844`, mean Triton request `453.02 ms`, `spool_bytes=0`.
- Stage events: `scene.plan=20`, `tile.preprocess=6638`, `tile.infer=983`, `tile.done=6638`, `block.ready=272`, `block.vectorize=272`, `block.done=272`, `scene.merge=20`, `job.finalize=1`.
- Streaming proof: `first_block_vectorized_at=1778496090.6759727`, `last_tile_inferred_at=1778496464.001578`, overlap `373.326 s`.
- Queue state at completion: all stage queues drained; `ie.dead_letter` had `messages_ready=0`, `messages_unacknowledged=0`.

Final queue consumers:

- `ie.tile.preprocess`: `8`
- `ie.tile.infer`: `2`
- `ie.tile.done`: `1`
- `ie.block.ready`: `4`
- `ie.block.vectorize`: `4`
- `ie.block.done`: `1`
- `ie.scene.plan`: `1`
- `ie.scene.merge`: `1`
- `ie.job.finalize`: `1`
- `ie.jobs.submit`: `1`

## Resource Use And Tuning

Applied tuning:

- `INFERENCE_ENGINE_PREPROCESS_CONCURRENCY=8`
- `INFERENCE_ENGINE_TRITON_CONCURRENCY=2`
- `INFERENCE_ENGINE_BLOCK_CONCURRENCY=4`
- `INFERENCE_ENGINE_DEFAULT_TRITON_BATCH_SIZE=8`
- `INFERENCE_ENGINE_MAX_WAIT_MS=75`
- 20-scene resource config: `batches_ahead=12`, `max_preprocess_queue=1024`, `max_scenes_inflight=4`, `max_blocks_inflight=32`, `max_spool_bytes=30 GiB`.

The tuning replaced serialized async consumers with real per-role worker threads. During the green 20-scene run, preprocess used up to `501%` CPU, Triton worker up to `109%` CPU, and block worker memory/CPU increased while inference was still running. Batch fill improved to `0.844`.

GPU utilization is bursty because raster read/preprocess and final block work still create short CPU/IO-bound intervals. The immediately preceding tuning validation on the same code path sampled `gpu_utilization_max=100%` and a final active sample of `89%`; the final green artifact sampled lower external GPU points, with InferenceEngine job metrics showing mid-run internal samples up to `76%` and final metrics retaining `26%`. The selected configuration is stable, keeps spool bounded, drains all queues, and gives proven CPU/GPU overlap without dead letters.

## Cleanup

- Production Airflow now exposes only `inference_engine_pipeline` for pseudolabels.
- Old internal pseudolabel stage modules for run/validate/vectorize/postprocess/export were removed from production registration.
- `mlsystem/src/pipeline/airflow_tasks.py` no longer contains the old direct pseudolabel dispatcher path.
- Thin `mlsystem` wrappers remain only for backwards-compatible imports; source of truth for pseudolabel inference, vectorization, postprocessing, and export is `InferenceEngine/src/inference_engine`.
- No runtime artifacts, probability tiles, accepted GeoJSON outputs, env files, or secrets are committed.

## Acceptance

- InferenceEngine is deployed as a separate service with dedicated worker containers.
- Airflow and `mlsystem-api` call InferenceEngine through HTTP.
- RabbitMQ stage queues are used by all pipeline steps.
- Triton performs inference with `segformer_b2`.
- Block/core/halo postprocessing starts before all tile inference completes.
- 2-scene and 20-scene real runs completed successfully.
- Compatibility artifacts are present.
- `ie.dead_letter` stayed empty.
