from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TritonEndpoint:
    url: str = "http://triton:8000"
    model_name: str = "identity_python"
    model_version: str | None = None
    transport: str = "http"
    grpc_url: str | None = None
    shared_memory: str = "off"


_READY_MODELS: set[tuple[str, str, str]] = set()
_CLIENTS = threading.local()


def triton_ready(url: str = "http://triton:8000", timeout_sec: float = 5.0) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/v2/health/ready", timeout=timeout_sec) as response:
            return response.status == 200
    except Exception:
        return False


def load_model(endpoint: TritonEndpoint, timeout_sec: float = 60.0) -> None:
    """Load a model when Triton runs with explicit model control."""
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=endpoint.url.replace("http://", "").replace("https://", ""))
    client.load_model(endpoint.model_name)
    if not client.is_model_ready(endpoint.model_name, model_version=endpoint.model_version or ""):
        raise RuntimeError(f"Triton model {endpoint.model_name} was loaded but is not ready")


def ensure_model_ready(endpoint: TritonEndpoint, timeout_sec: float = 120.0) -> None:
    """Ensure the configured model is ready under Triton explicit model control."""
    import time

    import tritonclient.http as httpclient

    key = (endpoint.url.rstrip("/"), endpoint.model_name, endpoint.model_version or "")
    if key in _READY_MODELS:
        return
    client = httpclient.InferenceServerClient(url=endpoint.url.replace("http://", "").replace("https://", ""))
    if not client.is_model_ready(endpoint.model_name, model_version=endpoint.model_version or ""):
        try:
            client.load_model(endpoint.model_name)
        except Exception:
            pass
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        if client.is_model_ready(endpoint.model_name, model_version=endpoint.model_version or ""):
            _READY_MODELS.add(key)
            return
        time.sleep(1.0)
    raise RuntimeError(f"Triton model {endpoint.model_name} is not ready at {endpoint.url}")


def unload_model(endpoint: TritonEndpoint) -> None:
    """Unload a model to release GPU memory after an inference stage."""
    import tritonclient.http as httpclient

    client = httpclient.InferenceServerClient(url=endpoint.url.replace("http://", "").replace("https://", ""))
    client.unload_model(endpoint.model_name)


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
    payload = np.asarray(batch, dtype=np.float32)
    if payload.ndim != 4:
        raise ValueError(f"Triton segmentation input must be BCHW, got shape={payload.shape}")
    ensure_model_ready(endpoint)
    if endpoint.shared_memory not in {"", "off"}:
        raise NotImplementedError("Triton shared memory is exposed for benchmarking config but not enabled in production client yet")
    if endpoint.transport == "grpc":
        return _infer_segmentation_batch_grpc(endpoint, payload)
    return _infer_segmentation_batch_http(endpoint, payload)


def _infer_segmentation_batch_http(endpoint: TritonEndpoint, payload: np.ndarray) -> np.ndarray:
    import tritonclient.http as httpclient

    client = _client_for_endpoint(endpoint)
    infer_input = httpclient.InferInput("INPUT__0", payload.shape, "FP32")
    infer_input.set_data_from_numpy(payload)
    output = httpclient.InferRequestedOutput("OUTPUT__0")
    result = client.infer(endpoint.model_name, model_version=endpoint.model_version or "", inputs=[infer_input], outputs=[output])
    logits = result.as_numpy("OUTPUT__0")
    if logits is None:
        raise RuntimeError(f"Triton model {endpoint.model_name} did not return OUTPUT__0")
    return np.asarray(logits, dtype=np.float32)


def _infer_segmentation_batch_grpc(endpoint: TritonEndpoint, payload: np.ndarray) -> np.ndarray:
    import tritonclient.grpc as grpcclient

    client = _client_for_endpoint(endpoint)
    infer_input = grpcclient.InferInput("INPUT__0", payload.shape, "FP32")
    infer_input.set_data_from_numpy(payload)
    output = grpcclient.InferRequestedOutput("OUTPUT__0")
    result = client.infer(endpoint.model_name, model_version=endpoint.model_version or "", inputs=[infer_input], outputs=[output])
    logits = result.as_numpy("OUTPUT__0")
    if logits is None:
        raise RuntimeError(f"Triton model {endpoint.model_name} did not return OUTPUT__0")
    return np.asarray(logits, dtype=np.float32)


def _client_for_endpoint(endpoint: TritonEndpoint):
    transport = endpoint.transport or "http"
    url = _client_url(endpoint)
    clients = getattr(_CLIENTS, transport, None)
    if clients is None:
        clients = {}
        setattr(_CLIENTS, transport, clients)
    client = clients.get(url)
    if client is None:
        if transport == "grpc":
            import tritonclient.grpc as grpcclient

            client = grpcclient.InferenceServerClient(url=url)
        else:
            import tritonclient.http as httpclient

            client = httpclient.InferenceServerClient(url=url)
        clients[url] = client
    return client


def _client_url(endpoint: TritonEndpoint) -> str:
    if endpoint.transport == "grpc":
        url = endpoint.grpc_url or endpoint.url
        url = url.replace("http://", "").replace("https://", "").rstrip("/")
        if url.endswith(":8000"):
            return url[:-5] + ":8001"
        return url
    return endpoint.url.replace("http://", "").replace("https://", "").rstrip("/")


def build_triton_config(job_predict: dict[str, Any] | None = None) -> TritonEndpoint | None:
    predict = job_predict or {}
    if str(predict.get("inference_backend") or "").lower() != "triton":
        return None
    return TritonEndpoint(
        url=str(predict.get("triton_url") or "http://triton:8000"),
        model_name=str(predict.get("triton_model_name") or predict.get("model_name") or "identity_python"),
        model_version=str(predict["triton_model_version"]) if predict.get("triton_model_version") is not None else None,
        transport=str(predict.get("triton_transport") or "http").lower(),
        grpc_url=str(predict["triton_grpc_url"]) if predict.get("triton_grpc_url") else None,
        shared_memory=str(predict.get("triton_shared_memory") or "off").lower(),
    )
