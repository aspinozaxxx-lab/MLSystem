# Server setup

Target server:

- SSH alias: `mlserver`
- Hostname: `roskadastr-ml`
- OS: Ubuntu 22.04.5
- Runtime venv: `/home/worker/ml-training/.venv`
- App root: `/home/worker/mlsystem`
- MLflow: `http://127.0.0.1:5000`
- MinIO/S3: `http://127.0.0.1:9000`
- Current mode: CPU-only, `max_gpu_train_jobs: 0`

Bootstrap:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/bootstrap.yml
```

Deploy:

```bash
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/deploy.yml
```

Self-hosted runner:

1. In GitHub, open repository settings, Actions, Runners, New self-hosted runner.
2. Generate a fresh registration token.
3. Do not paste the token into repository files, README, changelog, or logs.
4. Run `ansible/playbooks/runner.yml` only with the token supplied through a temporary environment variable or interactive shell.

Example pattern:

```bash
export RUNNER_TOKEN='<fresh-token-from-github-ui>'
ansible-playbook -i ansible/inventory/hosts.ini ansible/playbooks/runner.yml \
  -e install_runner_archive=true \
  -e configure_runner=true \
  -e runner_registration_token="$RUNNER_TOKEN"
unset RUNNER_TOKEN
```

For untrusted fork pull requests, do not run repository code on the self-hosted runner. CI uses GitHub-hosted runners; deploy/enqueue/sync are restricted to `main`, manual, or schedule triggers.
