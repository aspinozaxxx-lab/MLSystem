# MLSystem Code Architecture Rules

## Current Boundary

```text
MLSystem pipeline runner -> mlsystem-api -> production stage/dispatcher
inference_engine_pipeline -> InferenceEngine API -> RabbitMQ workers -> Triton
```

Runtime artifacts are never stored in git.

## Rules

| Priority | Rule | Target |
| --- | --- | --- |
| high | The pipeline runner does not execute pseudolabel domain logic. | `inference_engine_pipeline` submits/polls InferenceEngine over HTTP. |
| high | InferenceEngine is the source of truth for pseudolabel scene planning, tiling, Triton inference, vectorization, postprocess, merge, and export. | `InferenceEngine/src/inference_engine/**` |
| high | `mlsystem` keeps training, MLflow lifecycle, dataset preparation, pixel/object metrics, API job persistence, and report formatting. | `mlsystem/src/**` |
| high | Stage artifacts live under `/data/mlsystem/runs/<run_id>/`; API job state lives under `/data/mlsystem/api/jobs/<job_id>/`. | `PipelineRunStore`, `JobStore` |
| high | `/api/v1/stages` exposes one pseudolabel stage: `inference_engine_pipeline`. | `DEFAULT_PIPELINE_STAGES`, stage registry |
| medium | Deprecated compatibility wrappers must be marked and must not be registered as production pipeline stages. | `mlsystem/src/**` |
| medium | Runtime probability maps, accepted GeoJSON, queue spool, env files, and secrets stay outside git. | `.gitignore`, validation scripts |

## Deliberately Kept

- `real_train.py` and `mlsystem/src/pipeline/pseudolabel_pipeline.py` are kept for training-era compatibility and are not the production pseudolabel path.
- Thin wrappers under `mlsystem/src/inference`, `mlsystem/src/vectorization`, `mlsystem/src/postprocessing`, and `mlsystem/src/tiling` re-export InferenceEngine implementations for older imports.
- `prepare_inference_scenes.py` remains as a small manifest helper used by `inference_engine_pipeline` before HTTP submission.
