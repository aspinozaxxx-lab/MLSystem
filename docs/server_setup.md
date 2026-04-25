# Server setup

Целевой сервер:

- SSH alias: `mlserver`
- hostname: `roskadastr-ml`
- OS: Ubuntu 22.04.5
- runtime venv: `/home/worker/ml-training/.venv`
- application root: `/home/worker/mlsystem`
- MLflow: `http://127.0.0.1:5000`
- MinIO/S3: `http://127.0.0.1:9000`
- режим: CPU-only, `max_gpu_train_jobs: 0`

## Правило управления

GitHub self-hosted runner устанавливается и регистрируется один раз вручную, потому что GitHub выдает временный registration token. Этот token нельзя хранить в репозитории, README, changelog, workflow logs или Ansible vars.

Все остальное должно быть воспроизводимо через Ansible:

- создание `/home/worker/mlsystem`
- деплой легкого кода из `mlsystem/`
- создание `storage/` и `logs/`
- установка non-secret env-файла `/etc/mlsystem/mlsystem.env`
- установка systemd unit files
- `systemctl daemon-reload`
- enable/start/restart services
- health checks MLflow, MinIO и MLSystem web API

## Локальные команды

Bootstrap:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/bootstrap.yml
```

Deploy:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/deploy.yml \
  -e mlsystem_manage_services=true
```

Deploy на самом сервере, как это делает GitHub Actions runner:

```bash
ansible-playbook -i 'localhost,' -c local ansible/playbooks/deploy.yml \
  -e mlsystem_manage_services=true
```

## Services

Ansible создает и управляет:

- `mlsystem-web.service`
- `mlsystem-executor.service`

Templates:

- `ansible/roles/mlsystem/templates/mlsystem-web.service.j2`
- `ansible/roles/mlsystem/templates/mlsystem-executor.service.j2`
- `ansible/roles/mlsystem/templates/mlsystem.env.j2`

Logs:

- application logs directory: `/home/worker/mlsystem/logs`
- service logs: `journalctl -u mlsystem-web.service -u mlsystem-executor.service`
- runner logs: `/home/worker/actions-runner/_diag/`

## Проверка состояния

```bash
systemctl is-active mlsystem-web.service
systemctl is-active mlsystem-executor.service
curl -fsS http://127.0.0.1:8010/api/state
cd /home/worker/mlsystem
source /home/worker/ml-training/.venv/bin/activate
python -m src.cli status
```

## Runner

Runner registration remains the only manual exception. Use a fresh token from GitHub UI and do not persist it.

The repository contains `ansible/playbooks/runner.yml` as documentation and optional helper for runner installation, but runner registration must still receive the temporary token only at execution time.
