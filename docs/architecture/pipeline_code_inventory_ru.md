# Inventory pipeline/orchestration code

## pipeline_runner

| Файл | Что делает | Решение |
|---|---|---|
| `pipeline_runner/config.py` | trace schema, stage aliases | keep |
| `pipeline_runner/runner.py` | start/refresh/cancel run, worker process | keep as единственный lifecycle runner |
| `pipeline_runner/worker.py` | sequential stage worker | keep |
| `pipeline_runner/run_store.py` | run/status/summary/log storage | keep |
| `pipeline_runner/stages.py` | validation/dispatch bridge to pipeline stages | keep |
| `pipeline_runner/progress.py` | progress weights | keep |
| `pipeline_runner/logging.py` | stage log tee | keep |
| `pipeline_runner/mlflow_logging.py` | final metadata logging via adapter | keep |
| `pipeline_runner/cli.py` | CLI wrapper | keep |

## pipeline

| Файл | Что делает | Решение |
|---|---|---|
| `pipeline/experiment_stages.py` | domain dispatcher stage implementations | keep, but no run lifecycle ownership |
| `pipeline/stages/*` | typed stage implementations: inventory, prepare_dataset, inference_engine | keep |
| `pipeline/training_pipeline.py` | compatibility wrapper to `real_train` | keep until training internals extracted |
| `pipeline/prediction_pipeline.py` | prediction compatibility wrapper | keep |
| `pipeline/pseudolabel_pipeline.py` | pseudolabel domain logic | keep |
| `pipeline/pseudolabel_vectorize_worker.py` | vectorization worker logic | keep |
| `pipeline/pipeline_factory.py` | lightweight factory | keep |
| `pipeline/contracts.py` | stage contracts | keep |
| `pipeline/stage_report_formatter.py` | report formatting | keep |

## orchestration

| Файл | Что делает | Решение |
|---|---|---|
| `orchestration/maintenance_cleanup.py` | maintenance cleanup utility | keep or move to pipeline_runner maintenance later |
| `orchestration/inference_engine_client.py` | InferenceEngine client wrapper | move candidate to inference_engine or pipeline/stages |
| `orchestration/__init__.py` | package marker | keep until moves complete |

## Решение

- `pipeline_runner` - единственный orchestration module для run lifecycle, status, worker, logs, progress.
- `pipeline` - только domain stage implementations и registry/dispatch.
- `orchestration` не должен развиваться как второй runner. Оставшиеся utilities нужно либо перенести, либо оставить как compatibility до отдельного cleanup.
- Не создавать второй store, второй runner или второй API layer для pipeline lifecycle.
