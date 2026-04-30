# Migration From Old ML Server

Old server:

- SSH alias: `mlserver`
- existing Airflow/MLflow/MinIO are left intact
- old MLflow experiments and S3 artifacts were not deleted

New server:

- SSH alias: `gpu-mlserver`
- host: `31.192.104.147`

## What Was Migrated

The required MinIO bucket data was mirrored from old MinIO to new MinIO using `mc mirror`.

Migrated prefixes:

- `s3://mlsystems/images/`
- `s3://mlsystems/layouts/`
- `s3://mlsystems/system/`
- `s3://mlsystems/models/`
- `s3://mlsystems/reports/`

Inventory after migration:

- `images`: about 42 GiB / 332 objects on the new server
- `layouts`: about 338 KiB / deforest layout and `scenes.txt`
- `system`: manifests copied
- `models`: only `.keep` existed on the old server, no checkpoints found

Prefixes intentionally not bulk-migrated yet:

- old generated predictions
- old pseudolabel debug output
- old temporary probability maps
- old queue state

## MLflow

Old MLflow was not migrated destructively. The old server remains the source of truth for old experiment history.

New MLflow is used for new GPU-server experiments:

- tracking UI: `http://31.192.104.147:5000`
- backend store: PostgreSQL in the GPU Docker Compose stack
- artifact store: MinIO/S3 in the GPU Docker Compose stack

If old MLflow history must appear in the new UI, do a separate DB/artifact migration with downtime planning. Do not blindly overwrite the new MLflow database.

## Verification

Real mini run on the new server:

- Airflow DAG run: `manual__AIR-gpu-mini-deforest__20260430T115951Z`
- MLflow run: `http://31.192.104.147:5000/#/experiments/2/runs/f0105ffa612444068fad8975f99d7ca1`
- model: `tiny_unet_4ch`
- training device: `cuda`
- pseudolabel scene count: 1

The old server was not modified except read-only inventory and MinIO mirror reads.
