from __future__ import annotations

from datetime import datetime
from inspect import signature
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.airflow_tasks import MAIN_DAG_STAGES, STAGE_POOLS, push_stage_xcom, run_airflow_stage, stage_return_message


AIRFLOW_STATE_DIR = Path("/opt/airflow/mlsystem_runs")

def _run_stage(stage: str, **context):
    dag_run = context["dag_run"]
    summary = run_airflow_stage(
        stage=stage,
        dag_run_conf=dag_run.conf or {},
        airflow_run_id=dag_run.run_id,
        state_dir=AIRFLOW_STATE_DIR,
    )
    push_stage_xcom(summary, context.get("ti") or context.get("task_instance"))
    return stage_return_message(summary)


with DAG(
    dag_id="mlsystem_experiment_pipeline",
    description="MLSystem experiment pipeline orchestrated by Airflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=3,
    tags=["mlsystem", "ml", "deforest"],
) as dag:
    previous = None
    for stage_name in MAIN_DAG_STAGES:
        pool_name, pool_slots = STAGE_POOLS.get(stage_name, ("cpu_light", 1))
        task_kwargs = {
            "task_id": stage_name,
            "python_callable": _run_stage,
            "op_kwargs": {"stage": stage_name},
            "pool": pool_name,
            "pool_slots": pool_slots,
            "do_xcom_push": False,
        }
        if "show_return_value_in_logs" in signature(PythonOperator).parameters:
            task_kwargs["show_return_value_in_logs"] = False
        task = PythonOperator(**task_kwargs)
        if previous:
            previous >> task
        previous = task
