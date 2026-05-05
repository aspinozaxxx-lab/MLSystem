# Архитектурные правила кода MLSystem

Дата обновления: 2026-05-05.

Правила основаны на текущем состоянии Airflow pipeline после введения `mlsystem/src/pipeline/stages`. Цель правил: постепенно отделять orchestration от domain logic без рискованного переписывания обучения, MLflow и `real_train.py` за один патч.

## Инвентаризация модулей

| Пакет | Текущая роль | Что лежит правильно | Риски смешения ответственности |
|---|---|---|---|
| `mlsystem/src/data` | Scene matching, чтение scene list, object counting, train/val split. | `scene_matching.py` и `dataset_split.py` содержат reusable production-функции, которые уже используются handtest и новым `prepare_dataset`. | Geometry fallback зависит от raster metadata и CRS policy; storage details не должны расползаться глубже data layer. |
| `mlsystem/src/storage` | S3/MinIO, локальный JSON I/O, raster path/cache helpers. | `storage/s3.py` централизует boto3/S3 и `/vsis3`/cache policy. | `raster_path_for_s3_key` совмещает storage backend и cache policy; нужна явная конфигурация на уровне stage/use case. |
| `mlsystem/src/pipeline` | Pipeline contracts, wrappers, Airflow adapter, pseudolabel orchestration. | `pipeline/stages` теперь дает явные entrypoints для новых stages; `contracts.py` хранит reusable dataclasses. | `airflow_tasks.py` все еще содержит status store, resource monitor, MLflow, cleanup и legacy stage logic. `pseudolabel_pipeline.py` смешивает inference, vectorization, postprocess, export. |
| `mlsystem/src/pipeline/stages` | Stage entrypoints для Airflow/API. | Каждый новый stage получает `StageContext`, возвращает `StageReport`, пишет понятные artifacts. | Часть stages пока compatibility wrappers над старой логикой, а не полностью самостоятельные use cases. |
| `mlsystem/src/api` | FastAPI facade и persistent job runner для вызова stages из Airflow. | API не содержит бизнес-логики, запускает stage worker subprocess и сохраняет job artifacts. | Нужно следить, чтобы long-running work не возвращался в uvicorn worker и чтобы секреты маскировались. |
| `mlsystem/src/workflow/inference` | Каркас workflow engine для inference. | `direct_triton` остается default, `rabbitmq_triton` experimental и feature-flagged. | Полный RabbitMQ production inference еще не реализован. |
| `mlsystem/src/training` | Training helpers. | Losses, early stopping, train loop helpers изолированы. | Основная training-семантика все еще в `real_train.py`; выносить постепенно. |
| `mlsystem/src/inference` | Scene inference, Triton client, probability maps. | Triton HTTP клиент изолирован в `triton_client.py`; probability map accumulation локализован. | Actual stitching пока встроен в inference runner, а Airflow stage `validate_probability_maps` только валидирует outputs; старое имя `stitch_probability_maps` оставлено alias. |
| `mlsystem/src/postprocessing` | Vectorization, filtering, simplification, export. | Базовые операции разделены по focused modules. | Orchestration postprocess workers и tuning пока в `pseudolabel_pipeline.py`. |
| `mlsystem/src/tiling` | Window generation, stitching primitives, tile reading. | Примитивы окон и stitching отделены. | Не все primitive-level возможности представлены отдельными production use cases. |
| `mlsystem/src/metrics` | Pixel/object metrics. | Metrics переиспользуемы и изолированы. | Airflow object metric stage пока читает поздние compatibility artifacts. |
| `mlsystem/src/reporting` | Summaries, prediction examples, artifact reports. | HTML/examples и summary writers отделены. | MLflow artifact logging пока частично в `airflow_tasks.py`. |
| `mlsystem/src/orchestration` | Старые compatibility re-exports. | Сохраняет старые import paths. | Это не основной новый orchestration layer. Новая маршрутизация живет в `pipeline/stages`. |
| `airflow` | DAG definitions. | DAG file тонкий: строит `PythonOperator` по stage list. | При изменении stage list надо синхронно обновлять inventory docs и pools. |
| `tests` / `tests/handtests` | Unit tests и ручные проверки. | Handtests вызывают production CLI/functions, TIFF не копируются в repo. | Handtest output может разрастаться; нужен отдельный policy, если появятся большие generated artifacts. |

## Правила

| Приоритет | Правило | Зачем нужно | Где сейчас риск | Целевой паттерн |
|---|---|---|---|---|
| high | Каждый Airflow stage обязан иметь один явный entrypoint. | По task_id должно быть понятно, какая функция маршрутизирует работу. | Новые stages уже в `pipeline/stages`; старые training/MLflow stages пока compatibility fallback в `airflow_tasks.py`. | `pipeline/stages/<stage>.py::run(ctx) -> StageReport`. |
| high | Airflow tasks должны быть тонкими адаптерами. | DAG отвечает за scheduling, retries, pools и передачу config, а не за domain logic. | `airflow_tasks.py` все еще крупный и содержит legacy branches. | DAG -> `run_airflow_stage` -> registry -> stage entrypoint -> production modules. |
| high | У каждого stage должен быть `StageContext`, `StageReport`, fail policy и понятный Airflow log. | При ошибке оператор должен видеть checks, counters, warnings, errors и artifacts. | Новые stages уже используют `StageReport`; legacy fallback пока пишет старый payload. | Расширить StageReport-compatible wrappers на training/reporting stages без смены семантики. |
| high | У каждого stage должен быть явный input/output contract. | Это упрощает retries, handtests и downstream compatibility. | `dataset_manifest.json`, `inventory_scenes.json`, `inference_manifest.json` появились, но typed schemas еще не оформлены. | Dataclass/Pydantic-like `StageInput`/`StageOutput` плюс status JSON. |
| high | Production-функции должны переиспользоваться handtests. | Handtests не должны иметь копии алгоритмов. | `dataset_split.py` уже используется handtest и `prepare_dataset`; это правильный путь. | Любой новый handtest вызывает production module или CLI entrypoint. |
| high | Storage access должен идти через adapters. | S3/MinIO/local cache/MLflow меняются независимо от domain code. | Новые stages используют `storage.s3`; MLflow calls пока в `airflow_tasks.py`. | `storage` и infrastructure clients как отдельные adapters/services. |
| high | Infrastructure clients отдельно от domain logic. | URL, секреты, endpoint policy не должны попадать в data/training/postprocessing. | Triton client отделен, но endpoint orchestration в `pseudolabel_pipeline.py`; MLflow частично в `airflow_tasks.py`. | `inference/triton_client.py`, `tracking/mlflow_client.py`, `storage/s3.py`, stage-level config normalization. |
| high | Конфигурация отдельно от логики выполнения. | Defaults и normalization должны тестироваться без запуска stage. | `_build_airflow_job` и config merge пока в `airflow_tasks.py`. | `pipeline/config_normalization.py` или `AirflowJobBuilder`. |
| high | Воспроизводимость должна попадать в artifacts. | Seed, scene list, annotation, model checkpoint и params должны быть восстановимы. | `prepare_dataset` теперь пишет split seed/strategy; training/inference пишут часть runtime данных. | Каждый stage summary фиксирует input URIs, seed, version, checkpoint, params. |
| high | Artifacts должны быть явными. | Downstream не должен угадывать, кто создал файл. | `validate_probability_maps` честно валидирует/индексирует probability maps, а не создает actual stitched maps. | Stage либо владеет artifact creation, либо называется validate/collect и документирует это. |
| medium | Совместимость wrappers сохраняется, новая логика не прячется в `real_train.py`. | Снижает риск регрессий обучения. | Training split внутри `real_train.py` пока остается legacy. | Новые use cases выносятся в production modules; wrappers вызывают их постепенно. |
| high | ML pipeline stages должны быть модульными. | Нужны независимые retries и resource pools. | Pseudolabel vectorize/postprocess/export частично остаются в одном compatibility вызове. | Разделить discovery, matching, dataset prep, train, evaluation, inference, stitching, vectorization, postprocess, export, reporting, tracking. |
| high | GPU stages не должны удерживать CPU-heavy работу. | GPU pool должен освобождаться перед vectorization/postprocess. | `run_pseudolabel_inference` вызывает `stage_mode="inference"`; это корректно. Полное разделение downstream еще не завершено. | GPU inference пишет probability artifacts, CPU stages читают их отдельно. |
| medium | Для каждого production use case нужны unit test и handtest/smoke. | Synthetic tests ловят контрактные ошибки, handtests ловят реальные пути. | Добавлены unit tests для StageReport/registry/inventory/prepare_dataset/prepare_inference. | При добавлении stage добавлять unit test, handtest без копирования тяжелых данных и smoke при возможности. |
| high | Никаких неявных глобальных side effects. | Функции не должны молча писать в неожиданные места или менять env без восстановления. | Legacy pseudolabel code меняет некоторые env policy в `finally`; cleanup должен оставаться allowlisted. | Все output paths и policies передаются явно и пишутся в StageReport. |
| medium | Ошибки и skip policy должны быть явными. | Missing/bad scene должен обрабатываться предсказуемо. | `prepare_inference_scenes` фиксирует `bad_scene_policy`, но глубокая проверка TIFF остается в inference. | `fail_fast`/`skip_with_warning` policy в stage input и report. |
| medium | Stage inventory надо обновлять при изменении DAG. | Документация должна соответствовать коду. | Документ ручной. | PR, который меняет stage list, обновляет `docs/airflow_stage_inventory.md`. |

## Приоритетные цели следующего рефакторинга

1. Вынести legacy training/reporting stages в StageReport-compatible entrypoints без смены поведения.
2. Разделить `pseudolabel_pipeline.py` на отдельные use cases: inference, probability map collection/stitching, vectorization, postprocess, export.
3. Полностью развести `vectorize_pseudolabel`, `postprocess_pseudolabel` и `export_pseudolabel_artifacts`: сейчас `vectorize_pseudolabel` все еще compatibility wrapper над `stage_mode="postprocess"`.
4. Оформить typed contracts для `inventory_scenes.json`, `dataset_manifest.json`, `inference_manifest.json`.
5. Вынести MLflow lifecycle из `airflow_tasks.py` в отдельный tracking adapter.
6. Уточнить CRS policy для geometry fallback: обязательный `annotation_crs` в config для production, если GeoJSON не содержит CRS.
