from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TritonEndpoint:
    url: str = "http://triton:8000"
    model_name: str = "identity_python"
    model_version: str | None = None


def triton_ready(url: str = "http://triton:8000", timeout_sec: float = 5.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v2/health/ready", timeout=timeout_sec) as response:
            return response.status == 200
    except Exception:
        return False


def infer_identity_smoke(endpoint: TritonEndpoint, values: np.ndarray | None = None) -> np.ndarray:
    """Run a tiny Triton HTTP smoke inference against the identity test model."""
    import tritonclient.http as httpclient

    payload = np.asarray(values if values is not None else [1.0], dtype=np.float32)
    client = httpclient.InferenceServerClient(url=endpoint.url.replace("http://", "").replace("https://", ""))
    infer_input = httpclient.InferInput("INPUT0", payload.shape, "FP32")
    infer_input.set_data_from_numpy(payload)
    output = httpclient.InferRequestedOutput("OUTPUT0")
    result = client.infer(endpoint.model_name, model_version=endpoint.model_version or "", inputs=[infer_input], outputs=[output])
    return result.as_numpy("OUTPUT0")


def infer_segmentation_batch(endpoint: TritonEndpoint, batch: np.ndarray) -> np.ndarray:
    """Run a BCHW float32 segmentation batch and return raw logits as BCHW."""
    import tritonclient.http as httpclient

    payload = np.asarray(batch, dtype=np.float32)
    if payload.ndim != 4:
        raise ValueError(f"Triton segmentation input must be BCHW, got shape={payload.shape}")
    client = httpclient.InferenceServerClient(url=endpoint.url.replace("http://", "").replace("https://", ""))
    infer_input = httpclient.InferInput("INPUT__0", payload.shape, "FP32")
    infer_input.set_data_from_numpy(payload)
    output = httpclient.InferRequestedOutput("OUTPUT__0")
    result = client.infer(endpoint.model_name, model_version=endpoint.model_version or "", inputs=[infer_input], outputs=[output])
    logits = result.as_numpy("OUTPUT__0")
    if logits is None:
        raise RuntimeError(f"Triton model {endpoint.model_name} did not return OUTPUT__0")
    return np.asarray(logits, dtype=np.float32)


def build_triton_config(job_predict: dict[str, Any] | None = None) -> TritonEndpoint | None:
    predict = job_predict or {}
    if str(predict.get("inference_backend") or "").lower() != "triton":
        return None
    return TritonEndpoint(
        url=str(predict.get("triton_url") or "http://triton:8000"),
        model_name=str(predict.get("triton_model_name") or predict.get("model_name") or "identity_python"),
        model_version=str(predict["triton_model_version"]) if predict.get("triton_model_version") is not None else None,
    )
