# Архитектурные правила MLSystem

Дата обновления: 2026-05-05.

Документ фиксирует правила для текущей архитектуры:

```text
Airflow -> mlsystem-api -> stage registry -> production modules
```

## Модули и текущая роль

| Модуль | Роль | Замечания |
|---|---|---|
| `mlsystem/src/api` | FastAPI facade и persistent API jobs. | Должен оставаться тонким слоем над stage registry. |
| `mlsystem/src/pipeline/stages` | Явные entrypoints Airflow stages. | Каждый stage получает `StageContext` и возвращает `StageReport`. |
| `mlsystem/src/pipeline/airflow_tasks.py` | Совместимость, legacy wrappers, CLI/local mode. | Все еще крупный файл; training/MLflow/reporting stages постепенно нужно вынести. |
| `mlsystem/src/data` | Scene matching, dataset split, manifests. | Production-функции используются и Airflow, и handtests. |
| `mlsystem/src/storage` | S3/local IO adapters. | Domain code не должен работать с S3 напрямую без adapter. |
| `mlsystem/src/training` и `real_train.py` | Обучение. | Семантика обучения пока не переписывается. |
| `mlsystem/src/inference` | Scene inference, Triton client, probability maps. | Direct Triton path остается production default. |
| `mlsystem/src/vectorization` | Block/core/halo vectorization. | Новый feature-flagged путь для `vectorize_pseudolabel`. |
| `mlsystem/src/workflow/inference` | Каркас workflow engine. | RabbitMQ backend experimental. |
| `mlsystem/src/postprocessing` | Базовые операции vectorization/filter/export. | Orchestration postprocess еще частично в `pseudolabel_pipeline.py`. |
| `docs` | Stage inventory, API execution, validation docs. | Менять вместе с DAG/stage контрактами. |
| `tests` и `tests/handtests` | Unit и ручные проверки. | Handtests не копируют TIFF/TIF в repo. |

## Правила

| Приоритет | Правило | Зачем | Риск сейчас | Целевой паттерн |
|---|---|---|---|---|
| high | Airflow stage имеет один entrypoint. | По task_id должно быть понятно, кто выполняет работу. | Legacy stages еще живут в `airflow_tasks.py`. | `pipeline/stages/<stage>.py::run(ctx) -> StageReport`. |
| high | Airflow не содержит domain logic. | DAG отвечает за scheduling, retries, pools, XCom. | Compatibility fallback еще крупный. | Airflow -> API -> registry -> use case. |
| high | Каждый stage пишет `StageReport`. | Оператор должен видеть checks/counters/metrics/artifacts. | Новые stages уже делают это, legacy wrappers частично адаптированы. | StageReport + markdown report + scalar XCom. |
| high | XCom только scalar key/value. | Airflow metadata DB не должен хранить большие reports. | Исправлено через `xcom_safe_value` и `safe_xcom_push`. | Полный payload только в artifacts. |
| high | У каждого stage явный input/output contract. | Downstream не должен угадывать файлы. | Typed contracts есть не для всех artifacts. | Dataclass/Pydantic-like schema + status JSON. |
| high | Handtests вызывают production-функции. | Нет копий алгоритмов в ручных тестах. | `dataset_split.py` и split handtest соответствуют правилу. | CLI/production module shared by Airflow and handtest. |
| high | Storage access идет через adapters. | S3/MinIO/local cache можно менять независимо. | MLflow calls еще частично в `airflow_tasks.py`. | `storage/*`, `tracking/*`, clients/adapters. |
| high | Infrastructure clients отдельно от domain logic. | URL/секреты не должны попадать в data/training/postprocessing. | Triton client отделен; MLflow lifecycle требует дальнейшего выноса. | `inference/triton_client.py`, tracking adapter, S3 adapter. |
| high | GPU stages не удерживают CPU-heavy работу. | GPU pool должен освобождаться до vectorization/postprocess. | `run_pseudolabel_inference` делает inference; `vectorize_pseudolabel` CPU stage. | GPU writes probability artifacts, CPU stages read them. |
| high | Pools отражают нагрузку. | Airflow scheduler должен различать train и inference. | Добавлен `gpu_inference`; training остается `gpu_training`. | train -> `gpu_training`, inference -> `gpu_inference`, vectorize/postprocess -> `cpu_heavy`. |
| high | Метрики не выдумываются. | Report должен быть эксплуатационно честным. | Если pixel/object metric недоступна, пишется `None` и warning. | Считать метрику или явно писать `not_available` с причиной. |
| high | Artifacts объявляются явно. | Ошибки проще диагностировать без чтения кода. | Новые reports перечисляют artifacts. | Каждый stage пишет artifact map. |
| high | Нет неявных глобальных side effects. | Повторный запуск и retry должны быть предсказуемыми. | Cleanup работает по allowlist; env/secrets не менять руками. | Все output paths и policies передаются явно. |
| medium | Ошибки и skip policy явные. | Missing/bad scene должна иметь понятное поведение. | `prepare_inference_scenes` пишет `bad_scene_policy`; глубокая TIFF-проверка downstream. | `fail`/`skip_with_warning` policy в config и report. |
| medium | Stage inventory обновляется с DAG. | Документация должна соответствовать коду. | Правило остается ручным. | Любой PR со stage list обновляет `docs/airflow_stage_inventory.md`. |

## Следующие цели

1. Вынести legacy training/reporting stages из `airflow_tasks.py` в StageReport-compatible entrypoints без смены поведения.
2. Разделить `pseudolabel_pipeline.py` на inference, probability map validation/stitching, vectorization, postprocess, export.
3. Расширить typed contracts для `inventory_scenes.json`, `dataset_manifest.json`, `inference_manifest.json`, `f1_summary.json`.
4. Вынести MLflow lifecycle в отдельный tracking adapter.
5. Провалидировать `block_parallel` vectorization на реальных сценах и только после этого обсуждать default.
6. Подготовить pinned Airflow/API image вместо runtime `_PIP_ADDITIONAL_REQUIREMENTS`.
