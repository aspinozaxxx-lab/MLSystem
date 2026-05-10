from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


TERMINAL = {"success", "failed", "cancelled"}


class InferenceEngineClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout_sec: float = 30.0) -> None:
        self.base_url = (base_url or os.getenv("INFERENCE_ENGINE_API_URL") or "http://inference-engine-api:8095").rstrip("/")
        self.token = token if token is not None else os.getenv("INFERENCE_ENGINE_API_TOKEN")
        self.timeout_sec = timeout_sec

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/api/v1/jobs", payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/v1/jobs/{job_id}")

    def get_artifacts(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/api/v1/jobs/{job_id}/artifacts")

    def wait(self, job_id: str, *, poll_sec: float = 10.0, timeout_sec: float = 24 * 3600) -> dict[str, Any]:
        deadline = time.time() + timeout_sec
        while True:
            state = self.get_job(job_id)
            if str(state.get("status")) in TERMINAL:
                return state
            if time.time() >= deadline:
                raise TimeoutError(f"InferenceEngine job {job_id} did not finish within {timeout_sec} sec")
            time.sleep(poll_sec)

    def _json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.token}"} if self.token else {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"InferenceEngine API {method} {path} failed: HTTP {exc.code}: {body}") from exc
        return json.loads(body) if body else {}
