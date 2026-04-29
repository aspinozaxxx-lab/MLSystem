from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


SMOKE_CONF = {
    "experiment_id": "airflow_smoke_synthetic",
    "class_name": "deforest",
    "task": "smoke",
    "smoke": True,
    "images_uri": "s3://mlsystems/images/",
    "layout_uri": "s3://mlsystems/layouts/deforest/",
    "scenes_file": "scenes.txt",
    "annotation_file": "auto",
    "model": {"name": "constant_probability", "input_bands": [1, 2, 3, 4], "preview_bands": [4, 1, 2]},
    "preprocess": {"tile_size": 6, "stride": 4, "context": 2, "include_negative_scenes": True},
    "train": {"enabled": False, "time_limit_sec": 30, "batch_size": "auto", "workers": 0},
    "pseudolabel": {"enabled": True, "run_on": "synthetic", "full_scene": True},
    "postprocess": {
        "enabled": True,
        "max_geojson_mb": 20,
        "max_objects": 10,
        "keep_largest_if_too_many": True,
        "thresholds": [0.5],
        "min_object_area_m2_candidates": [1],
        "simplify_tolerance_m_candidates": [0],
    },
    "mlflow": {"experiment": "mlsystem-airflow-smoke"},
}


with DAG(
    dag_id="mlsystem_smoke_pipeline",
    description="Triggers the main MLSystem Airflow DAG with a synthetic lightweight smoke config",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlsystem", "smoke"],
) as dag:
    trigger_main = TriggerDagRunOperator(
        task_id="trigger_mlsystem_experiment_pipeline_smoke",
        trigger_dag_id="mlsystem_experiment_pipeline",
        conf=SMOKE_CONF,
        wait_for_completion=True,
        poke_interval=10,
        reset_dag_run=False,
    )
