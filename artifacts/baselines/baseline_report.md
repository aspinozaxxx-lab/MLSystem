# Baseline report

Generated: 2026-05-01T13:19:40.457266+00:00

## IRKUTSK-pseudolabel-unet-r50-t512-v4

- Accepted GeoJSON: `/data/mlsystem/airflow/status/IRKUTSK-pseudolabel-unet-r50-t512-v4/IRKUTSK-pseudolabel-unet-r50-t512-v4.accepted.geojson`
- MLflow experiment: `mlsystem-deforest-irkutsk` / `6`
- MLflow run: `be71380e3b9b4e3a9004a6c9ea6165d9`
- MLflow URL: http://31.192.104.147:5000/#/experiments/6/runs/be71380e3b9b4e3a9004a6c9ea6165d9
- Airflow run: `manual__IRKUTSK-pseudolabel-unet-r50-t512-v4__20260501T_irkutsk_v4`
- Model: `unet_resnet50`, backbone `resnet50`
- Checkpoint: `/opt/airflow/mlsystem_runs/Q2-deforest-unet-r50-t512-f1-v7/unet_resnet50.pt`
- Tiling: tile `512`, stride `384`, context `64`
- Train: enabled `False`, planned epochs `None`, completed `None`, batch `None`, lr `None`
- Postprocess: threshold `0.7`, min_area `70000.0`, simplify `18.0`, max_geojson_mb `20.0`, top500 `0.0`
- Pixel metrics: `{}`
- Object metrics: `{}`
- Pseudolabel metrics: `{'object_count': 7092, 'vertices_count': 210918, 'accepted_geojson_mb': 9.64661693572998, 'coverage_fraction': 0.7619547605067737}`

Source training run:
- Baseline id: `Q2-deforest-unet-r50-t512-f1-v7`
- MLflow run: `dda5fc17f36449118aad07dee3e39bdd`
- MLflow URL: http://31.192.104.147:5000/#/experiments/3/runs/dda5fc17f36449118aad07dee3e39bdd
- Airflow run: `manual__Q2-deforest-unet-r50-t512-f1-v7__20260501T_quality_v7`
- Source model: `unet_resnet50`
- Source checkpoint: `/opt/airflow/mlsystem_runs/Q2-deforest-unet-r50-t512-f1-v7/unet_resnet50.pt`
- Source object metrics: `{'object_f1': 0.07401812688821752, 'object_fn': 89, 'object_fp': 1137, 'object_precision': 0.04131534569983136, 'object_recall': 0.35507246376811596, 'object_tp': 49}`

## GPU-deforest-segformer-b0-t1024-gpu-v13

- Accepted GeoJSON: `/data/mlsystem/airflow/status/GPU-deforest-segformer-b0-t1024-gpu-v13/GPU-deforest-segformer-b0-t1024-gpu-v13.accepted.geojson`
- MLflow experiment: `mlsystem-deforest` / `2`
- MLflow run: `ac140723852b4208b0e76ada65fd2de1`
- MLflow URL: http://31.192.104.147:5000/#/experiments/2/runs/ac140723852b4208b0e76ada65fd2de1
- Airflow run: `manual__GPU-deforest-segformer-b0-t1024-gpu-v13__20260430T_gpu_util_v13`
- Model: `segformer_b0`, backbone `b0`
- Checkpoint: `/opt/airflow/mlsystem_runs/GPU-deforest-segformer-b0-t1024-gpu-v13/segformer_b0.pt`
- Tiling: tile `1024`, stride `768`, context `128`
- Train: enabled `True`, planned epochs `100`, completed `12`, batch `6`, lr `0.0003`
- Postprocess: threshold `0.35`, min_area `1000.0`, simplify `5.0`, max_geojson_mb `20.0`, top500 `0.0`
- Pixel metrics: `{'train/loss': 0.769782786630094, 'val/loss': 1.0020568455968584, 'val/dice': 0.8571428571443025, 'val/iou': 0.8571428571443025, 'val/pixel_dice': 0.8571428571443025, 'val/pixel_iou': 0.8571428571443025, 'val/pixel_f1': 0.8571428714314607}`
- Object metrics: `{'object_f1': 0.06674427629025999, 'object_fn': 214, 'object_fp': 2191, 'object_precision': 0.037768994290733424, 'object_recall': 0.2866666666666667, 'object_tp': 86}`
- Pseudolabel metrics: `{'object_count': 2277, 'vertices_count': 78641, 'accepted_geojson_mb': 3.5225143432617188, 'coverage_fraction': 0.7991342956628209}`
