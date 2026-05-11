# InferenceEngine Validation Report

Status: completed on the GPU server through GitHub Actions and Ansible deploy.

## Commit And Workflow

- Validated service commit: `5f490b7` (`Compact InferenceEngine compatibility probability artifacts`).
- Workflow: `InferenceEngine service`, run `25650559696`.
- Service jobs: `test`, `build-artifact`, `deploy-service-code`, `validate-service`, and `validate-real-server` succeeded.
- Validation artifact: `inference-engine-server-validation`, artifact id `6910610629`.
- Validation generated at: `2026-05-11T04:42:08Z`.
- Real manifest: auto-discovered `inference_manifest.json`, 996 scenes available.

## Deployed Services

The server validation observed these containers up and healthy/reachable:

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
- MinIO and MLflow

Triton health was ready. Repository index contained `segformer_b2` version `1` in `READY` state and `identity_python` stayed ready.

## 2 Scene Run

- Path: `mlsystem-api` compatibility path, equivalent to Airflow stage execution.
- Run id: `ie_real_2_1778474528`.
- InferenceEngine job: `ie_real_2_1778474528-242a3dfd9d7b`.
- Model: MLflow run `a7838f91528a47e1931b685c2ea06686`, `segformer_b2`, Triton model `segformer_b2`.
- Config: `patch_size=1024`, `stride=768`, bands `[1,2,3,4]`, `threshold=0.5`, `core_size_px=4096`, `halo_px=512`, `local_min_area=0`, `final_min_area=0`, `merge_epsilon=1.0`, `triton_batch_size=8`, `batches_ahead=4`, `max_scenes_inflight=1`.
- Result: success.
- Stage results: `run_pseudolabel_inference`, `validate_probability_maps`, `vectorize_pseudolabel`, `postprocess_pseudolabel`, and `export_pseudolabel_artifacts` all succeeded.
- Metrics: `tiles_done=760/760`, `blocks_done=30/30`, `triton_batches=293`, `triton_batch_fill_ratio=0.324`, mean Triton request `110.5 ms`, `spool_bytes=0`.
- Streaming proof: `first_block_vectorized_at=2026-05-11T04:42:20.188782Z`; `last_tile_inferred_at=2026-05-11T04:43:18.006213Z`; overlap `57.817 s`.
- Artifacts present: `<experiment_id>.accepted.geojson`, `accepted.geojson.gz`, `coverage_report.json`, `pseudolabel_summary.json`, `postprocess_summary.json`, `vectorization_summary.json`, `pseudolabel_scene_results_manifest.json`, `probability_maps_index.json`, `inference_results.json`, `pseudolabel_scenes.txt`, `prediction_examples.html`.

## 20 Scene Run

- Path: direct InferenceEngine API job through RabbitMQ workers.
- Job id: `ie_real_20-5f171920cdb1`.
- Config: `max_scenes=20`, `patch_size=1024`, `stride=768`, bands `[1,2,3,4]`, `threshold=0.5`, `core_size_px=4096`, `halo_px=512`, `local_min_area=0`, `final_min_area=0`, `merge_epsilon=1.0`, `triton_batch_size=8`, `batches_ahead=6`, `max_scenes_inflight=2`, `max_blocks_inflight=16`, `max_preprocess_queue=512`, `max_spool_bytes=30 GiB`.
- Result: success.
- Metrics: `tiles_done=6638/6638`, `blocks_done=272/272`, `triton_batches=1806`, `triton_batch_fill_ratio=0.459`, mean Triton request `159.68 ms`, `spool_bytes=0`, `preprocess_pauses_total=0`, `preprocess_resumes_total=0`.
- Stage event counts: `scene.plan=20`, `tile.preprocess=6638`, `tile.infer=1806`, `tile.done=6638`, `block.ready=272`, `block.vectorize=272`, `block.done=272`, `scene.merge=20`, `job.finalize=1`.
- Streaming proof: `first_block_vectorized_at=2026-05-11T04:44:32.803125Z`; `last_tile_inferred_at=2026-05-11T04:53:22.850786Z`; overlap `530.048 s`.
- RabbitMQ final queue state: all stage queues drained; `ie.dead_letter` had `messages_ready=0`, `messages_unacknowledged=0`; consumers were present on `ie.tile.preprocess`, `ie.tile.infer`, `ie.block.vectorize`, `ie.scene.merge`, and `ie.job.finalize`.
- Resource summary: RTX 5090 memory held about `15512 MiB`; workflow samples saw GPU utilization up to `100%` during active inference and queue/event metrics showed sustained Triton batching. CPU block workers were active while inference continued, proven by block completion counts increasing before all tiles finished.

## Backpressure And Bottlenecks

- The successful 20-scene run did not hit spool backpressure: `preprocess_pauses_total=0`, `spool_bytes=0` at completion.
- Earlier validation exposed stale spool cleanup and full-scene placeholder probability mosaics as bottlenecks. Fixes applied:
  - tile spool descriptors are deleted after durable probability artifacts are written;
  - duplicate `tile.infer` messages skip spool reads when probability artifact checksum already exists;
  - job state JSON writes use atomic replace;
  - workers ack and skip stale messages for terminal jobs;
  - final compatibility probability artifacts are compact placeholders because real probabilities are tile-backed.
- Remaining utilization bottleneck is batch fill, not correctness: final `triton_batch_fill_ratio=0.459`. The selected stable parameters are `triton_batch_size=8`, `batches_ahead=6`, `max_scenes_inflight=2`, `max_blocks_inflight=16`. These produced successful streaming and bounded spool without dead letters.

## Acceptance

- RabbitMQ was used by every production stage queue.
- Triton performed inference with `INPUT__0` / `OUTPUT__0` compatible `segformer_b2`.
- CPU block vectorization/postprocessing started before all tile inference completed.
- Airflow compatibility path via `mlsystem-api` produced all required artifacts.
- `ie.dead_letter` stayed empty at completion.
- No runtime artifacts, probability maps, GeoJSON outputs, env files, or secrets are stored in git.
