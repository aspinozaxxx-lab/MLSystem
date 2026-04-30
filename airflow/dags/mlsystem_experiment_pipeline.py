from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.airflow_tasks import MAIN_DAG_STAGES, run_airflow_stage


AIRFLOW_STATE_DIR = Path("/opt/airflow/mlsystem_runs")
GPU_POOL_STAGES = {"train_model", "predict_pseudolabel_scenes"}


def _run_stage(stage: str, **context):
    dag_run = context["dag_run"]
    return run_airflow_stage(
        stage=stage,
        dag_run_conf=dag_run.conf or {},
        airflow_run_id=dag_run.run_id,
        state_dir=AIRFLOW_STATE_DIR,
    )


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
        task = PythonOperator(
            task_id=stage_name,
            python_callable=_run_stage,
            op_kwargs={"stage": stage_name},
            pool="gpu_training" if stage_name in GPU_POOL_STAGES else "default_pool",
        )
        if previous:
            previous >> task
        previous = task
