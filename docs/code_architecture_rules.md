# Правила архитектуры кода MLSystem

## Текущий контур

```text
Airflow -> mlsystem-api -> persisted API job -> stage registry/dispatcher -> production modules
```

Runtime artifacts не хранятся в git.

## Правила

| Priority | Rule | Why | Target |
| --- | --- | --- | --- |
| high | Airflow не исполняет domain logic напрямую. | Airflow отвечает за scheduling, retries, pools и XCom. | `airflow_tasks.run_airflow_stage` вызывает API. |
| high | Каждый stage пишет `StageReport` или совместимый report payload. | Оператор должен видеть status/checks/counters/metrics/artifacts. | `stages/*`, dispatcher stages in `airflow_tasks.py`. |
| high | XCom содержит только small scalar key/value. | Airflow metadata DB не должен хранить reports/artifacts. | `push_stage_xcom`, `safe_xcom_push`. |
| high | Источник истины по stage - files under `/data/mlsystem/airflow/status/<run_id>/`. | Отчеты должны переживать перезапуск API/Airflow. | `AirflowRunStore`, API job reports. |
| high | API job state лежит вне git. | Runtime jobs не являются исходным кодом. | `/data/mlsystem/api/jobs/<job_id>/`. |
| high | Stage input/output artifacts объявляются явно. | Downstream stages не должны гадать, откуда читать данные. | artifact contracts and stage reports. |
| medium | Storage access идет через adapters. | S3/MinIO/local cache можно менять независимо. | `storage/*`, S3 adapters. |
| medium | MLflow URL/report formatting централизуется. | Логи Airflow должны быть одинаково читаемы. | `stage_report_formatter`, MLflow helpers. |
| medium | GPU stages отделены от CPU-heavy stages. | GPU не должен удерживаться vectorization/postprocess работой. | Airflow pools and stage split. |
| medium | Handtests вызывают production functions. | Ручная отладка не должна дублировать алгоритмы. | `tests/handtests/*` -> `mlsystem/src/*`. |
| medium | Runtime artifacts не возвращаются в git. | История есть в git, актуальный repo должен быть чистым. | `.gitignore`, cleanup checks. |

## Что оставлено

- `mlsystem/src/pipeline/airflow_tasks.py` пока содержит dispatcher stages для training/MLflow/reporting, потому что эти stages реально входят в текущий DAG.
- `real_train.py` и тяжелая MLflow/training логика не переписаны в этом cleanup pass.
- `vectorize_pseudolabel` сохраняет `legacy` mode как текущий совместимый режим выполнения; `block_parallel` включается явно config-ом.
