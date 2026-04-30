# GPU MLSystem Server

Primary training server:

- host: `31.192.104.147`
- SSH alias: `gpu-mlserver`
- Airflow: `http://31.192.104.147:8081`
- MLflow: `http://31.192.104.147:5000`
- MinIO Console: `http://31.192.104.147:9001`
- Triton HTTP: `http://31.192.104.147:8000`

The server is configured through Ansible roles in `ansible/roles/gpu_server` and `ansible/roles/gpu_platform`.

## Hardware Inventory

Current inventory from the first deployment:

- OS: Ubuntu 24.04.4 LTS
- kernel: `6.8.0-110-generic`
- CPU: AMD Ryzen 9 9950X, 16 cores / 32 threads
- RAM: 123 GiB
- root storage: 1.9 TiB, mounted under `/`
- GPU: NVIDIA GeForce RTX 5090, 32607 MiB VRAM
- NVIDIA driver: `580.142`
- CUDA reported by driver: `13.0`

The RTX 5090 requires the open NVIDIA kernel module on this host. The bootstrap role installs `nvidia-driver-580-open`.

## Deployment

Manual local deployment, when needed:

```bash
ansible-playbook -i ansible/inventory/gpu.ini ansible/playbooks/bootstrap_gpu_server.yml
ansible-playbook -i ansible/inventory/gpu.ini ansible/playbooks/deploy_gpu_stack.yml
```

Normal deployment should use GitHub Actions workflow `deploy-gpu-server`.

Required GitHub Secrets:

- `GPU_SERVER_HOST`
- `GPU_SERVER_USER`
- `GPU_SERVER_SSH_KEY`
- `MINIO_ROOT_USER`
- `MINIO_ROOT_PASSWORD`
- `MLFLOW_DB_PASSWORD`
- `AIRFLOW_ADMIN_USER`
- `AIRFLOW_ADMIN_PASSWORD`
- `AIRFLOW_POSTGRES_PASSWORD`
- `AIRFLOW_FERNET_KEY`
- `AIRFLOW_SECRET_KEY`

Secrets must not be committed to the repository. On the server they are materialized into `/etc/mlsystem/gpu-platform.env`.

## Data Layout

Main persistent root:

```text
/data/mlsystem/
  airflow/
  artifacts/
  backups/
  cache/
  minio/
  mlflow/
  models/
  platform/
  storage/
  triton/
```

Docker Compose file deployed by Ansible:

```text
/data/mlsystem/platform/docker-compose.yml
```

MLSystem repository copy:

```text
/opt/mlsystem/repo
```

## Health Checks

```bash
curl http://31.192.104.147:8081/api/v1/health
curl http://31.192.104.147:5000/health
curl http://31.192.104.147:9000/minio/health/live
curl http://31.192.104.147:8000/v2/health/ready
```

GPU checks:

```bash
ssh gpu-mlserver nvidia-smi
ssh gpu-mlserver "docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi"
```

PyTorch CUDA inside Airflow:

```bash
ssh gpu-mlserver "cd /data/mlsystem/platform && docker compose --env-file /etc/mlsystem/gpu-platform.env -f docker-compose.yml exec -T airflow-scheduler python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))\""
```

## Notes

Airflow currently installs ML Python packages at container startup through `_PIP_ADDITIONAL_REQUIREMENTS`. This works for smoke and mini runs, but a custom image should replace runtime installation before long production training.
