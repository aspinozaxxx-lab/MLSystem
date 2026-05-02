# Triton Integration

Triton is deployed on the GPU server as part of `deploy/docker-compose.gpu.yml`.

Endpoints:

- HTTP: `http://31.192.104.147:8000`
- gRPC: `31.192.104.147:8001`
- metrics: `http://31.192.104.147:8002`
- readiness: `http://31.192.104.147:8000/v2/health/ready`

Model repository:

```text
/data/mlsystem/triton/model_repository
```

The Ansible role installs a small `identity_python` model for deployment smoke checks.

## Smoke Inference

From the Airflow scheduler container:

```bash
ssh gpu-mlserver "cd /data/mlsystem/platform && docker compose --env-file /etc/mlsystem/gpu-platform.env -f docker-compose.yml exec -T airflow-scheduler python - <<'PY'
import numpy as np
from src.inference.triton_client import TritonEndpoint, infer_identity_smoke, triton_ready
print(triton_ready('http://triton:8000'))
print(infer_identity_smoke(TritonEndpoint(url='http://triton:8000'), np.array([42.0], dtype=np.float32)).tolist())
PY"
```

Expected output:

```text
True
[42.0]
```

## MLSystem Adapter

The client adapter is in `mlsystem/src/inference/triton_client.py`.

Current status:

- Triton deployment and dummy inference are working.
- Real model export to Triton is not yet wired into the training pipeline.
- Training/evaluation still use PyTorch directly.

Next integration step: export a trained UNet/DeepLab checkpoint to a Triton-compatible format and switch pseudolabel inference to `inference_backend: triton` when an exported model is available.

## RabbitMQ Inference Queue

The GPU platform can also run RabbitMQ as the control plane between CPU tile
preparation and Triton inference.

The intended flow is:

1. CPU producers read rasters, build tile tensors, and publish lightweight tile
   metadata to RabbitMQ.
2. GPU inference consumers pull batches, call Triton HTTP/gRPC, and write
   probability tile outputs to local run storage or S3.
3. CPU stitch/vectorize/postprocess stages consume saved outputs after GPU
   inference is complete.

RabbitMQ must not carry full scenes or large probability maps. Use it for
metadata, backpressure, and retry control. Heavy arrays should stay in shared
local storage, S3, or Triton shared-memory integration.

Infrastructure:

- Compose service: `rabbitmq`
- Internal AMQP URL: `MLSYSTEM_RABBITMQ_URL`
- Management UI: bind-mounted to `127.0.0.1:${RABBITMQ_MANAGEMENT_PORT}` and
  intended for SSH tunneling only.

Smoke check inside Airflow:

```bash
python -m mlsystem.src.inference.rabbitmq_triton_queue smoke-produce --count 8
python -m mlsystem.src.inference.rabbitmq_triton_queue smoke-consume --count 8 --triton-url http://triton:8000
python -m mlsystem.src.inference.rabbitmq_triton_queue status --include-result
```

Current limitation: real SegFormer/UNet/DeepLab model export is still required
before production pseudolabel inference can be fully switched from PyTorch
runtime to Triton.
