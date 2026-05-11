# MLSystem API Service

`mlsystem-api` is the production stage execution endpoint used by Airflow and frontend tools. It owns MLSystem orchestration, job persistence, training dispatch, MLflow lifecycle, dataset preparation, and metric summaries. It does not own pseudolabel inference domain logic.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process health and deployed commit. |
| `GET /ready` | Checks job root, status root, env, and stage registry. |
| `GET /api/v1/stages` | Current `MAIN_DAG_STAGES`, registry stages, dispatcher stages, and pools. |
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

The old internal pseudolabel stages are no longer production stages and should not appear in `/api/v1/stages` `main_dag_stages` or in the Airflow task list:

- `prepare_inference_scenes`
- `run_pseudolabel_inference`
- `validate_probability_maps`
- `vectorize_pseudolabel`
- `postprocess_pseudolabel`
- `export_pseudolabel_artifacts`

## Required Environment

- `MLSYSTEM_API_JOB_ROOT`
- `MLSYSTEM_API_TOKEN`
- `MLSYSTEM_AIRFLOW_STATE_DIR`
- `MLSYSTEM_AIRFLOW_EXECUTION_MODE`
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

Airflow stage artifacts are outside git:

```text
/data/mlsystem/airflow/status/<run_id>/
```

Runtime outputs, probability maps, accepted GeoJSON, env files, and secrets must not be committed.
