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
