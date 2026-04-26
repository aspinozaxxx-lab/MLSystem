# Server Setup

Target server:

- SSH alias: `mlserver`
- hostname: `roskadastr-ml`
- OS: Ubuntu 22.04.5
- runtime venv: `/home/worker/ml-training/.venv`
- application root: `/home/worker/mlsystem`
- data mount: `/data`
- MLSystem storage: `/data/mlsystem/storage`
- MinIO data: `/data/minio/data`
- MLflow: `http://127.0.0.1:5000`
- MinIO/S3: `http://127.0.0.1:9000`
- current mode: CPU-only, `max_gpu_train_jobs: 0`

## Management Rule

GitHub self-hosted runner registration is the only manual exception. It requires a short-lived GitHub registration token, so the token must not be stored in Git, docs, configs, logs, or Ansible vars.

Everything else is managed through Ansible:

- `/home/worker/mlsystem`
- `/data/mlsystem/storage`
- `/data/mlsystem/logs`
- compatibility symlinks `/home/worker/mlsystem/storage` and `/home/worker/mlsystem/logs`
- non-secret env file `/etc/mlsystem/mlsystem.env`
- `mlsystem-web.service`
- `mlsystem-executor.service`
- `mlsystem-preprocess.service`
- systemd reload and service restart
- MLflow Docker compose/image, MinIO, and web API health checks

## Disk And /data

Initial manual disk setup is intentionally not automated in Ansible because it is destructive if pointed at the wrong disk. The server was prepared with:

```bash
sudo parted /dev/sda unit GiB print free
sudo parted /dev/sda --script mkpart primary ext4 256GiB 100%
sudo partprobe /dev/sda
sudo mkfs.ext4 -F -L data /dev/sda3
sudo mkdir -p /data
sudo mount /dev/sda3 /data
uuid=$(sudo blkid -s UUID -o value /dev/sda3)
sudo cp /etc/fstab /etc/fstab.backup-$(date +%Y%m%dT%H%M%S)
echo "UUID=$uuid /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sudo mount -a
df -h / /data
lsblk -f
```

After `/data` exists, Ansible manages the application data directories:

```text
/data/mlsystem/storage
/data/mlsystem/logs
/data/mlsystem/artifacts
/data/minio/data
```

MinIO and MLflow are managed by `/home/worker/ml-platform/docker-compose.yml`. The compose file is rendered by Ansible from:

```text
ansible/roles/mlplatform/templates/docker-compose.yml.j2
```

The template does not contain secrets; it expects the existing server-side `/home/worker/ml-platform/.env`. MLflow image updates are applied through:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/deploy_mlflow.yml
```

Before updating MLflow, Ansible writes a lightweight PostgreSQL metadata dump to `/data/mlsystem/artifacts/mlflow-backups/`.

MinIO storage was moved from Docker volume `ml-platform_minio-data` to bind mount:

```yaml
volumes:
  - /data/minio/data:/data
```

The old Docker volume was intentionally left as a backup and must not be removed until a separate cleanup is approved.

## Local Commands

Bootstrap:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/bootstrap.yml
```

Deploy infrastructure:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/deploy_infra.yml
```

Deploy MLflow platform:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/deploy_mlflow.yml
```

Deploy application code:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/deploy_code.yml
```

Full deploy wrapper:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/deploy.yml \
  -e mlsystem_manage_services=true
```

On the server, GitHub Actions uses local inventory:

```bash
ansible-playbook -i 'localhost,' -c local ansible/playbooks/deploy_code.yml
ansible-playbook -i 'localhost,' -c local ansible/playbooks/deploy_mlflow.yml
ansible-playbook -i 'localhost,' -c local ansible/playbooks/deploy_infra.yml
```

## Services

Ansible creates and manages:

- `mlsystem-web.service`
- `mlsystem-executor.service`

Templates:

- `ansible/roles/mlsystem/templates/mlsystem-web.service.j2`
- `ansible/roles/mlsystem/templates/mlsystem-executor.service.j2`
- `ansible/roles/mlsystem/templates/mlsystem-preprocess.service.j2`
- `ansible/roles/mlsystem/templates/mlsystem.env.j2`

Logs:

- application logs directory: `/data/mlsystem/logs`
- service logs: `journalctl -u mlsystem-web.service -u mlsystem-executor.service -u mlsystem-preprocess.service`
- runner logs: `/home/worker/actions-runner/_diag/`

## Status Checks

```bash
systemctl is-active mlsystem-web.service
systemctl is-active mlsystem-executor.service
systemctl is-active mlsystem-preprocess.service
curl -fsS http://127.0.0.1:8010/api/state
curl -fsS http://127.0.0.1:8010/api/preprocess
curl -fsS http://127.0.0.1:9000/minio/health/live
curl -fsS http://127.0.0.1:5000
curl -fsS http://127.0.0.1:5000/version
mc ls mlplatform
mc ls mlplatform/mlsystems
cd /home/worker/mlsystem
source /home/worker/ml-training/.venv/bin/activate
python -m src.cli status
```

## Runner Audit

`ansible/playbooks/runner.yml` is audit-only. It checks whether the runner directory, runner config marker, and runner service exist. It does not register or reconfigure the runner.
