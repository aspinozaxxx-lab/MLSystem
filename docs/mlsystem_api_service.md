# MLSystem API Service

`mlsystem-api` is the production pipeline execution endpoint used by the MLSystem pipeline runner and frontend tools. It owns MLSystem orchestration, run persistence, training dispatch, MLflow lifecycle, dataset preparation, and metric summaries. It does not own pseudolabel inference domain logic.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process health and deployed commit. |
| `GET /ready` | Checks job root, run root, env, and stage registry. |
| `GET /api/v1/stages` | Current pipeline stages, registry stages, dispatcher stages, and pools. |
| `POST /api/v1/pipeline-runs` | Starts a complete pipeline run from JSON/YAML trace. |
| `GET /api/v1/pipeline-runs/{run_id}` | Returns run status, progress, stages, artifacts, MLflow info, and log tail. |
| `GET /api/v1/pipeline-runs/{run_id}/log` | Returns the current pipeline log tail. |
| `POST /api/v1/pipeline-runs/{run_id}/cancel` | Requests cancellation before the next stage. |
| `POST /api/v1/runs/{run_id}/stages/{stage}/start` | Creates a persisted API job for a stage. |
| `GET /api/v1/jobs/{job_id}` | Returns job state, stage report, artifacts, and error details. |
| `GET /api/v1/runs/{run_id}/summary` | Reads run summary from status artifacts. |
| `POST /api/v1/debug/run-stage-sync` | Synchronous debug endpoint for tests and handtests. |

## Pseudolabel Boundary

The production pseudolabel stage is:

```text
inference_engine_pipeline
```

It is implemented by `mlsystem.src.pipeline.stages.inference_engine_pipeline.run` and calls InferenceEngine over HTTP:

```text
POST /api/v1/jobs
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/artifacts
```

The stage report records:

- `inference_engine_api_url`
- `inference_engine_job_id`
- `inference_engine_status_url`
- `source=inference_engine`
- `request_submitted_via_http=true`

The old internal pseudolabel stages are no longer production stages and should not appear in `/api/v1/stages` production output:

- `prepare_inference_scenes`
- `run_pseudolabel_inference`
- `validate_probability_maps`
- `vectorize_pseudolabel`
- `postprocess_pseudolabel`
- `export_pseudolabel_artifacts`

## Required Environment

- `MLSYSTEM_API_JOB_ROOT`
- `MLSYSTEM_RUN_ROOT`
- `MLSYSTEM_API_TOKEN`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_S3_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `MLSYSTEM_TRITON_URL`
- `INFERENCE_ENGINE_API_URL`
- `INFERENCE_ENGINE_API_TOKEN`

## Job Storage

API job state is outside git:

```text
/data/mlsystem/api/jobs/<job_id>/
```

Pipeline run artifacts are outside git:

```text
/data/mlsystem/runs/<run_id>/
```

Runtime outputs, probability maps, accepted GeoJSON, env files, and secrets must not be committed.
