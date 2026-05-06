from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.orchestration.maintenance_cleanup import (
    cleanup_airflow_logs,
    cleanup_local_cache,
    cleanup_runtime_intermediates,
)


with DAG(
    dag_id="mlsystem_maintenance_cleanup",
    description="Cleans old MLSystem Airflow runtime intermediates, local cache and old Airflow logs",
    start_date=datetime(2026, 1, 1),
    schedule="17 3 * * *",
    catchup=False,
    max_active_runs=1,
    tags=["mlsystem", "maintenance", "cleanup"],
) as dag:
    cleanup_intermediates = PythonOperator(
        task_id="cleanup_runtime_intermediates",
        python_callable=cleanup_runtime_intermediates,
        pool="io_light",
        pool_slots=1,
    )
    cleanup_cache = PythonOperator(
        task_id="cleanup_local_cache",
        python_callable=cleanup_local_cache,
        pool="io_light",
        pool_slots=1,
    )
    cleanup_logs = PythonOperator(
        task_id="cleanup_airflow_logs",
        python_callable=cleanup_airflow_logs,
        pool="io_light",
        pool_slots=1,
    )

    cleanup_intermediates >> cleanup_cache >> cleanup_logs
