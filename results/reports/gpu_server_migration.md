# GPU Server Migration Report

Date: 2026-04-30

## New Server Inventory

- host: `31.192.104.147`
- SSH alias: `gpu-mlserver`
- OS: Ubuntu 24.04.4 LTS
- kernel: `6.8.0-110-generic`
- CPU: AMD Ryzen 9 9950X, 16 cores / 32 threads
- RAM: 123 GiB
- root disk: 1.9 TiB, about 1.8 TiB free before deployment
- GPU: NVIDIA GeForce RTX 5090
- VRAM: 32607 MiB
- NVIDIA driver: `580.142`
- CUDA reported by driver: `13.0`

Initial state: Docker and `nvidia-smi` were absent, and the GPU needed the open NVIDIA kernel module. The Ansible bootstrap installs Docker, NVIDIA Container Toolkit, UFW rules, and `nvidia-driver-580-open`.

## Deployed Platform

Deployed through Ansible + Docker Compose:

- MinIO
- MLflow + PostgreSQL
- Airflow webserver/scheduler/triggerer + PostgreSQL
- Triton Inference Server
- MLSystem code mounted into Airflow containers

External URLs:

- Airflow: `http://31.192.104.147:8081`
- MLflow: `http://31.192.104.147:5000`
- MinIO Console: `http://31.192.104.147:9001`
- Triton ready: `http://31.192.104.147:8000/v2/health/ready`

Health checks passed:

- Airflow `/api/v1/health`
- MLflow `/health`
- MinIO `/minio/health/live`
- Triton `/v2/health/ready`

PyTorch inside Airflow scheduler:

```text
torch.cuda.is_available() = True
device = NVIDIA GeForce RTX 5090
```

Triton model repository:

```json
[{"name": "identity_python", "state": "READY", "version": "1"}]
```

Triton smoke inference returned `[42.0]`.

## GitHub Deploy

Added workflow:

- `.github/workflows/deploy-gpu-server.yml`

It runs on `ubuntu-latest` and deploys over SSH + Ansible. It does not require a self-hosted runner.

Required secrets are documented in `docs/gpu_server_setup.md`.

The workflow was committed and pushed, but actual GitHub-side execution depends on GitHub Secrets being configured in the repository settings.

## S3 / MinIO Migration

Created bucket layout for `mlsystems`.

Migrated from old server:

- `images`: about 42 GiB, 332 objects
- `layouts`: deforest layout and `scenes.txt`
- `system`: manifests
- `models`: no checkpoints existed on the old server, only `.keep`
- `reports`: `.keep`

Not migrated:

- old generated predictions
- old pseudolabel debug/probability maps
- deprecated queue state
- old MLflow database

Old server was not deleted or disabled by this migration.

## Smoke Runs

Synthetic smoke DAG:

- DAG: `mlsystem_experiment_pipeline`
- run id: `manual__AIR-gpu-smoke-synthetic__20260430T115829Z`
- status: success
- all 23 stages succeeded
- MLflow run: `http://31.192.104.147:5000/#/experiments/1/runs/95f0be755c624befbdc93ec3f6f926c3`

Real mini GPU DAG:

- DAG: `mlsystem_experiment_pipeline`
- run id: `manual__AIR-gpu-mini-deforest__20260430T115951Z`
- status: success
- all 23 stages succeeded
- MLflow run: `http://31.192.104.147:5000/#/experiments/2/runs/f0105ffa612444068fad8975f99d7ca1`
- model: `tiny_unet_4ch`
- epochs completed: 1
- device: `cuda`
- train/loss: `1.7174495855967205`
- val/iou: `3.814697265623545e-13`
- val/object_f1: `0.0`
- pseudolabel accepted objects: 1
- coverage fraction: `0.7723794744217091`

Real mini artifacts:

- `train_scenes.txt`
- `val_scenes.txt`
- `pseudolabel_scenes.txt`
- `AIR-gpu-mini-deforest.accepted.geojson`
- `object_metrics.json`
- `coverage_report.json`
- `pseudolabel_summary.json`
- `prediction_examples.html`
- `run_summary.json`
- `codex_summary.json`
- `history.json`
- `history.csv`
- checkpoint `tiny_unet_4ch.pt`

Forbidden artifacts not logged to MLflow:

- `accepted.geojson.gz`
- `accepted.gpkg`
- `accepted_debug.geojson`
- `windows_preview.geojson`

Current reporter still logs some debug JSON/PNG artifacts, for example `segformer_tiling_debug.json` and probability preview images. That is a cleanup item, not a deployment blocker.

## Remaining Limits

- Airflow image currently installs ML dependencies at startup. Build a custom image before long production training.
- Real model export to Triton is not implemented yet; Triton is deployed and smoke-tested with a dummy model.
- Old MLflow history remains on the old server. New experiments use the new MLflow.
- GitHub deploy workflow still needs repository secrets configured before it can be considered fully operational from GitHub Actions.
