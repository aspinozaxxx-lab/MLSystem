# MLSystem Airflow migration

## Цель

Airflow становится основным оркестратором MLSystem вместо legacy filesystem queue:

- Airflow UI показывает DAG runs, стадии, логи и статусы.
- Эксперименты запускаются через Airflow UI/API с JSON `dag_run.conf`.
- MLflow продолжает хранить experiments, metrics, params и artifacts.
- S3/MinIO остается heavy storage.
- Legacy queue/services отключаются только после успешного Airflow smoke.

## Основные файлы

- `airflow/dags/mlsystem_experiment_pipeline.py` - основной DAG.
- `airflow/dags/mlsystem_smoke_pipeline.py` - lightweight synthetic smoke DAG.
- `mlsystem/src/pipeline/airflow_tasks.py` - публичные task wrappers для DAG и CLI.
- `ansible/playbooks/deploy_airflow.yml` - deploy Airflow.
- `ansible/playbooks/disable_legacy_queue.yml` - backup legacy root и disable legacy services.
- `ansible/roles/airflow/` - Docker Compose deployment role.

## Deploy

Секреты не хранятся в repo. Перед deploy в protected environment должны быть заданы:

```bash
export AIRFLOW_ADMIN_PASSWORD='...'
export AIRFLOW_FERNET_KEY='...'
export AIRFLOW_SECRET_KEY='...'
export AIRFLOW_POSTGRES_PASSWORD='...'
```

Если protected env vars отсутствуют при первом deploy, Ansible role сгенерирует first-run значения и сохранит их в `/etc/mlsystem/airflow.env` с mode `0600`. Последующие deploy переиспользуют этот protected env file, если явные env vars не переданы.

Deploy:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/deploy_airflow.yml
```

Airflow UI:

```text
http://mlserver:8081
```

Если внешний доступ закрыт, использовать SSH tunnel:

```bash
ssh -L 8081:127.0.0.1:8081 mlserver
```

## Запуск эксперимента

В Airflow UI открыть DAG `mlsystem_experiment_pipeline`, нажать `Trigger DAG`, передать JSON config.

Минимальный smoke можно запустить DAG `mlsystem_smoke_pipeline`; он trigger-ит основной DAG с synthetic конфигом.

Пример API:

```bash
curl -u "$AIRFLOW_USER:$AIRFLOW_PASSWORD" \
  -H "Content-Type: application/json" \
  -X POST \
  http://127.0.0.1:8081/api/v1/dags/mlsystem_experiment_pipeline/dagRuns \
  -d '{"conf":{"experiment_id":"deforest_segformer_b0_t1024_v1","class_name":"deforest","task":"train_predict_pseudolabel","images_uri":"s3://mlsystems/images/","layout_uri":"s3://mlsystems/layouts/deforest/","scenes_file":"scenes.txt","annotation_file":"auto","model":{"name":"segformer_b0","input_bands":[1,2,3,4],"preview_bands":[4,1,2]},"preprocess":{"tile_size":1024,"stride":768,"context":128,"include_negative_scenes":true},"train":{"enabled":true,"time_limit_sec":14400,"early_stopping":true,"batch_size":"auto","workers":0},"pseudolabel":{"enabled":true,"run_on":"all_available_images","full_scene":true},"postprocess":{"enabled":true,"max_geojson_mb":20,"max_objects":500,"keep_largest_if_too_many":true,"thresholds":[0.35,0.45,0.55,0.65],"min_object_area_m2_candidates":[500,1000,2000,5000],"simplify_tolerance_m_candidates":[2,5,10,20]},"mlflow":{"experiment":"mlsystem-deforest"}}}'
```

## Статус для Codex/API

Основной источник статуса - Airflow REST API:

- `GET /api/v1/dags/mlsystem_experiment_pipeline/dagRuns`
- `GET /api/v1/dags/mlsystem_experiment_pipeline/dagRuns/{dag_run_id}`
- `GET /api/v1/dags/mlsystem_experiment_pipeline/dagRuns/{dag_run_id}/taskInstances`

Дополнительно task wrappers пишут compact JSON summaries:

```text
/data/mlsystem/airflow/status/<experiment_id>/summary.json
/data/mlsystem/airflow/status/<experiment_id>/stages/*.json
```

В Docker эти файлы смонтированы как:

```text
/opt/airflow/mlsystem_runs/<experiment_id>/summary.json
```

## DAG stages

Основной DAG содержит стадии:

1. `validate_experiment_config`
2. `check_s3_layout`
3. `match_scenes`
4. `validate_scene_matching`
5. `inventory_images`
6. `prepare_dataset_manifest`
7. `prepare_train_tiles_or_windows`
8. `validate_dataset`
9. `create_mlflow_run`
10. `train_model`
11. `evaluate_pixel_metrics`
12. `predict_validation_scenes`
13. `vectorize_validation_predictions`
14. `compute_object_f1`
15. `predict_pseudolabel_scenes`
16. `stitch_probability_maps`
17. `vectorize_pseudolabel`
18. `postprocess_pseudolabel`
19. `export_pseudolabel_artifacts`
20. `generate_prediction_examples`
21. `log_mlflow_artifacts`
22. `write_codex_api_summary`
23. `finalize_mlflow_run`

Некоторые стадии пока являются orchestration-boundary wrappers, если физическая ML-логика еще выполняется внутри существующих MLSystem modules.

## Legacy disable

После успешного Airflow smoke:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/disable_legacy_queue.yml
```

Playbook:

- делает backup `/home/worker/mlsystem` в `/data/mlsystem/backups/mlsystem_legacy_<timestamp>.tar.gz`;
- исключает heavy `storage`, `logs`, `.git`;
- stop/disable:
  - `mlsystem-executor.service`
  - `mlsystem-web.service`
  - `mlsystem-preprocess.service`

MLflow metadata, S3/MinIO artifacts и старые experiments не удаляются.

## Ограничения текущего шага

- Реальные heavy train/pseudolabel jobs не запускаются автоматически.
- Smoke DAG использует synthetic data.
- Для production training task wrappers еще нужно постепенно заменить placeholders на прямые вызовы refactored MLSystem pipeline modules.
