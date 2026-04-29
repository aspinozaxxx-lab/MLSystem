# MLSystem Airflow migration summary

Дата: 2026-04-29

## Что сделано в repo

- Добавлен Airflow deployment через Ansible role `airflow`.
- Добавлен основной DAG `mlsystem_experiment_pipeline`.
- Добавлен smoke DAG `mlsystem_smoke_pipeline`.
- Добавлены task wrappers `mlsystem/src/pipeline/airflow_tasks.py`.
- Smoke-ветка Airflow task wrappers сделана lightweight: DAG импортируется без `torch` и тяжелого ML-окружения.
- Добавлен lightweight status store для Codex/API:
  - `/data/mlsystem/airflow/status/<experiment_id>/summary.json`
  - `/data/mlsystem/airflow/status/<experiment_id>/stages/*.json`
- Обновлены GitHub workflows:
  - `cicd-code` деплоит MLSystem code и Airflow DAGs.
  - `cicd-ansible` проверяет новые playbooks и деплоит Airflow.
  - `cicd-queue` оставлен manual-only deprecated.
  - `sync-results` оставлен manual-only deprecated.
- Добавлен playbook `disable_legacy_queue.yml` с backup legacy root перед остановкой services.
- Airflow UI port по умолчанию: `8081`, чтобы не конфликтовать с legacy Streamlit на `8080` до smoke success.

## Airflow DAG stages

Основной DAG содержит 23 stage:

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

## Локальные проверки

Пройдены:

```powershell
python -m compileall mlsystem\src tests airflow\dags
python -m unittest discover -s tests
python -m mlsystem.src.pipeline.airflow_tasks validate-config --run-id manual__cli_smoke --state-dir .tmp\airflow-status --conf-file .tmp\airflow-smoke-conf.json
python -m mlsystem.src.pipeline.airflow_tasks check-s3-layout --run-id manual__cli_smoke --state-dir .tmp\airflow-status --conf-file .tmp\airflow-smoke-conf.json
python -m mlsystem.src.pipeline.airflow_tasks predict-pseudolabel-scenes --run-id manual__airflow_smoke --state-dir .tmp\airflow-status --conf-file .tmp\airflow-cli-smoke.json
```

`unittest`: 34 tests OK.

YAML parse для Ansible/GitHub/Airflow config пройден.

`ansible-playbook` не установлен на локальной Windows машине, поэтому локальный syntax-check Ansible пропущен.

## Секреты

Airflow secrets не коммитятся. Deploy ожидает protected env vars:

- `AIRFLOW_ADMIN_PASSWORD`
- `AIRFLOW_FERNET_KEY`
- `AIRFLOW_SECRET_KEY`
- `AIRFLOW_POSTGRES_PASSWORD`

Если protected env vars отсутствуют, Ansible role генерирует first-run значения и сохраняет их в `/etc/mlsystem/airflow.env` с mode `0600`; следующие deploy переиспользуют этот файл.

## Legacy policy

Legacy queue services отключаются только отдельным playbook после smoke success:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/disable_legacy_queue.yml
```

Playbook сначала делает backup `/home/worker/mlsystem` в `/data/mlsystem/backups/mlsystem_legacy_<timestamp>.tar.gz`, исключая heavy `storage`, `logs`, `.git`.

## Что не удаляется

- MLflow metadata/runs.
- S3/MinIO artifacts.
- Старые experiment directories.
- Старые logs/summaries.
- Legacy Python code очереди пока остается в repo как deprecated compatibility.

## Ограничения

- Smoke DAG synthetic и не запускает тяжелое обучение.
- Некоторые stages пока являются orchestration wrappers, потому что физическая ML-логика еще постепенно выносится из legacy runner.
- Airflow deploy требует GitHub/environment secrets для admin/db keys.
