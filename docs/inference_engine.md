# InferenceEngine

InferenceEngine is a separate FastAPI service for pseudolabel inference. Airflow calls its API and receives ready artifacts instead of running pseudolabel domain logic inside `mlsystem`.

## Architecture

Pipeline:

1. `scene.plan`: build inference tile plan and block/core/halo plan.
2. `tile.preprocess`: read raster window, normalize, write bounded spool artifact.
3. `tile.infer`: batch descriptors, call Triton with `INPUT__0`, persist probability tiles.
4. `tile.done`: mark tile durable after checksum write.
5. `block.ready`: dependency tracker releases a block when all expanded-window tiles are ready.
6. `block.vectorize`: materialize expanded probability block, threshold/vectorize, clip to core.
7. `scene.merge`: dissolve block GeoJSON, apply `merge_epsilon` and final area filter.
8. `job.finalize`: write old-pipeline-compatible artifacts into Airflow run dir.

The acceptance metric is:

`first_block_vectorized_at < last_tile_inferred_at`

for jobs with more than one block.

## Queues

RabbitMQ queues:

- `ie.jobs.submit`
- `ie.scene.plan`
- `ie.tile.preprocess`
- `ie.tile.infer`
- `ie.tile.done`
- `ie.block.ready`
- `ie.block.vectorize`
- `ie.block.done`
- `ie.scene.merge`
- `ie.job.finalize`
- `ie.events`
- `ie.dead_letter`

Messages contain descriptors only: `job_id`, `scene_id`, `tile_id`, `block_id`, paths, checksums, retry attempt, and stage name. Numpy arrays are stored in `/data/mlsystem/inference-engine/jobs` or `/data/mlsystem/inference-engine/spool`.

## Worker Roles

Production uses separate RabbitMQ consumers:

- `inference-engine-worker planner`: consumes `ie.jobs.submit` and `ie.scene.plan`.
- `inference-engine-worker preprocess`: consumes `ie.tile.preprocess`, writes spool descriptors, publishes `ie.tile.infer`.
- `inference-engine-worker triton`: batches `ie.tile.infer`, calls Triton, writes probability tile artifacts, publishes `ie.tile.done`.
- `inference-engine-worker aggregator`: consumes `ie.tile.done`, updates dependency counters, publishes `ie.block.ready`.
- `inference-engine-worker block`: consumes `ie.block.ready` and `ie.block.vectorize`, materializes expanded blocks, vectorizes, clips to core, publishes `ie.block.done`.
- `inference-engine-worker merger`: consumes `ie.block.done` and `ie.scene.merge`, merges scene outputs, publishes `ie.job.finalize`.
- `inference-engine-worker finalizer`: consumes `ie.job.finalize`, writes compatibility artifacts and marks the job successful.

Ack happens after durable state/artifact writes. Retry republishes with incremented attempt count; exhausted messages go to `ie.dead_letter`.
Workers check terminal job state before heavy work, so cancelled or already-failed jobs ack stale descriptors without continuing raster reads, Triton requests, or vectorization.
Worker role concurrency is real parallelism: each configured consumer runs in its own worker thread with an independent asyncio loop, so blocking rasterio, NumPy, Shapely, and Triton HTTP calls do not serialize a role inside one event loop.

## Backpressure

Adaptive producer tracks:

- `infer_queue_depth`
- `spool_bytes`
- `preprocess_pauses_total`
- `preprocess_resumes_total`
- `triton_batch_fill_ratio`
- `triton_request_duration_ms`
- `gpu_util_snapshot_count`

Target ready tiles:

`triton_batch_size * max(2, triton_instance_count * batches_ahead)`

Publishing pauses when infer queue depth exceeds the configured high watermark or spool bytes exceed `max_spool_bytes`.

Current tuned server defaults:

- `INFERENCE_ENGINE_DEFAULT_TRITON_BATCH_SIZE=8`
- `INFERENCE_ENGINE_MAX_WAIT_MS=75`
- `INFERENCE_ENGINE_PREPROCESS_CONCURRENCY=8`
- `INFERENCE_ENGINE_TRITON_CONCURRENCY=2`
- `INFERENCE_ENGINE_BLOCK_CONCURRENCY=4`

The 20-scene validation job uses `batches_ahead=12`, `max_preprocess_queue=1024`, `max_scenes_inflight=4`, and `max_blocks_inflight=32`.

Tile spool files are deleted after probability tile artifacts and checksums are durable. Duplicate `tile.infer` messages are idempotent: if the probability artifact checksum is already valid, the worker skips the spool read and publishes `tile.done`.

## Compatibility Artifacts

The finalizer writes the old Airflow artifact names into `run_dir`. Real probability data remains tile-backed under the InferenceEngine job tree; the legacy per-scene `npz_path` entries are compact placeholder files because downstream stages are validate-only for `pseudolabel.source=inference_engine`. This keeps validators compatible without writing multi-GB zero mosaics during 20-scene finalization.

## API

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

`/health` reports the deployed commit through `MLSYSTEM_COMMIT` or `GIT_COMMIT`. `/ready` checks job/spool/artifact/log roots, RabbitMQ configuration, and Triton health.

## Airflow Integration

Airflow does not model internal InferenceEngine stages. The only Airflow pseudolabel task is `inference_engine_pipeline`; it calls `POST /api/v1/jobs`, polls `GET /api/v1/jobs/{job_id}`, reads `GET /api/v1/jobs/{job_id}/artifacts`, and validates the final artifacts in the Airflow run directory.

The InferenceEngine job itself fans out through RabbitMQ queues and worker containers. Stage transitions are visible through job events and queue metrics, not as Airflow tasks.

## Testing

Unit tests live in `InferenceEngine/tests`.

```bash
python -m unittest discover -s InferenceEngine/tests
```

Synthetic handtest:

```bash
python scripts/handtest_inference_engine_synthetic.py --api http://127.0.0.1:8095
```

Real handtests:

```bash
python scripts/handtest_inference_engine_real.py --manifest /path/to/inference_manifest.json --max-scenes 2
python scripts/handtest_inference_engine_20_scenes.py --manifest /path/to/inference_manifest.json
```

Production worker startup:

```bash
inference-engine-worker planner
inference-engine-worker preprocess
inference-engine-worker triton
inference-engine-worker aggregator
inference-engine-worker block
inference-engine-worker merger
inference-engine-worker finalizer
```
