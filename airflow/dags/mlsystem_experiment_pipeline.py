from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.pipeline.airflow_tasks import MAIN_DAG_STAGES, run_airflow_stage


AIRFLOW_STATE_DIR = Path("/opt/airflow/mlsystem_runs")

STAGE_RESOURCES = {
    "validate_experiment_config": ("cpu_light", 1),
    "check_s3_layout": ("io_light", 1),
    "match_scenes": ("cpu_light", 1),
    "validate_scene_matching": ("cpu_light", 1),
    "inventory_images": ("io_light", 1),
    "prepare_dataset_manifest": ("cpu_light", 1),
    "prepare_train_tiles_or_windows": ("cpu_light", 1),
    "validate_dataset": ("cpu_light", 1),
    "create_mlflow_run": ("io_light", 1),
    "train_model": ("gpu_training", 1),
    "evaluate_pixel_metrics": ("cpu_light", 1),
    "predict_validation_scenes": ("gpu_training", 1),
    "vectorize_validation_predictions": ("cpu_heavy", 2),
    "compute_object_f1": ("cpu_heavy", 1),
    "predict_pseudolabel_scenes": ("gpu_training", 1),
    "stitch_probability_maps": ("cpu_heavy", 2),
    "vectorize_pseudolabel": ("cpu_heavy", 3),
    "postprocess_pseudolabel": ("cpu_heavy", 3),
    "export_pseudolabel_artifacts": ("io_light", 1),
    "generate_prediction_examples": ("cpu_heavy", 1),
    "log_mlflow_artifacts": ("io_light", 1),
    "write_codex_api_summary": ("io_light", 1),
    "finalize_mlflow_run": ("io_light", 1),
}


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
        pool_name, pool_slots = STAGE_RESOURCES.get(stage_name, ("cpu_light", 1))
        task = PythonOperator(
            task_id=stage_name,
            python_callable=_run_stage,
            op_kwargs={"stage": stage_name},
            pool=pool_name,
            pool_slots=pool_slots,
        )
        if previous:
            previous >> task
        previous = task
