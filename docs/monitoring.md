# MLSystem Monitoring

MLSystem monitoring is deployed as a repository-managed Docker stack through Ansible and GitHub Actions. Runtime configs and secrets stay on the server; no dashboard credentials or service passwords are committed.

## Architecture

Services:

- `prometheus` scrapes machine, container, GPU, RabbitMQ, Triton, InferenceEngine, and service-health metrics.
- `grafana` serves dashboards under the authenticated frontend gateway at `/grafana/`.
- `node-exporter` exposes host CPU, memory, filesystem, network, and disk metrics.
- `cadvisor` exposes Docker container CPU, memory, network, and disk IO metrics.
- `dcgm-exporter` exposes NVIDIA GPU utilization, memory, power, temperature, and PCIe/DCGM counters.
- `mlsystem-monitor-exporter` converts InferenceEngine JSON metrics and service health checks into Prometheus metrics.
- RabbitMQ has the `rabbitmq_prometheus` plugin enabled at internal port `15692`.
- Triton is scraped from its metrics endpoint on internal port `8002`.

Public routes:

- `http://31.192.104.147/grafana/`
- `http://31.192.104.147/prometheus/`

Both routes are protected by the frontend session through nginx `auth_request /auth/proxy-check`. Grafana uses auth-proxy mode with `X-MLSystem-User`; after frontend login as `mluser`, Grafana opens without a second login.

## Dashboards

Provisioned dashboards:

- `mlsystem-overview`: main dashboard and Grafana home dashboard.
- `mlsystem-inference-engine`: InferenceEngine jobs, queues, Triton batches, spool, overlap, and failures.
- `mlsystem-rabbitmq`: per-queue ready/unacked/consumers and dead-letter state.

The overview dashboard is linked from the frontend home page and embedded as a preview:

```text
/grafana/d/mlsystem-overview/mlsystem-overview?orgId=1&kiosk
```

## Metrics Sources

Prometheus scrape jobs:

- `node-exporter`
- `cadvisor`
- `dcgm-exporter`
- `rabbitmq`
- `triton`
- `inference-engine`
- `mlsystem-monitor-exporter`

InferenceEngine keeps the existing JSON endpoint:

```text
GET /metrics
```

It also exposes Prometheus text format:

```text
GET /metrics/prometheus
```

Key InferenceEngine metrics include:

- `inference_engine_jobs_total{status=...}`
- `inference_engine_active_jobs`
- `inference_engine_tiles_done`
- `inference_engine_blocks_done`
- `inference_engine_triton_batches_total`
- `inference_engine_triton_batch_fill_ratio`
- `inference_engine_streaming_overlap_seconds`
- `inference_engine_queue_ready{queue=...}`
- `inference_engine_dead_letter_messages`

## Deployment

Monitoring is part of the GPU platform compose file and is applied through:

```text
.github/workflows/monitoring.yml
ansible/playbooks/deploy_ml_platform.yml
```

Ansible creates:

```text
/data/mlsystem/prometheus
/data/mlsystem/grafana
/data/mlsystem/monitoring
```

The generated server env includes `FRONTEND_GRAFANA_URL`, `FRONTEND_PROMETHEUS_URL`, `GF_SECURITY_SECRET_KEY`, and exporter ports.

## Validation

Server checks:

```bash
curl -fsS http://127.0.0.1:9090/prometheus/-/ready
curl -fsS http://127.0.0.1:3000/api/health
curl -fsS http://127.0.0.1:9100/metrics | head
curl -fsS http://127.0.0.1:8080/metrics | head
curl -fsS http://127.0.0.1:9400/metrics | head
curl -fsS http://127.0.0.1:9200/metrics | head
curl -fsS http://127.0.0.1:8095/metrics/prometheus | head
curl -fsS http://127.0.0.1:8002/metrics | head
```

Prometheus targets must be `up` for node-exporter, cAdvisor, RabbitMQ, Triton, InferenceEngine, and the MLSystem monitor exporter. DCGM is expected to be `up` on NVIDIA runtime hosts; if the server driver/runtime rejects the DCGM container, this is treated as a deployment issue to fix rather than silently ignoring GPU metrics.

## Security

- Grafana and Prometheus raw ports bind to localhost by default.
- Public access goes through the frontend gateway.
- Grafana uses auth-proxy headers from nginx; there is no separate browser-visible Grafana password.
- Prometheus is protected by the same frontend session.
- RabbitMQ, MinIO, MLflow, Grafana, and Prometheus secrets remain in server env or container env and are not sent to the browser.
