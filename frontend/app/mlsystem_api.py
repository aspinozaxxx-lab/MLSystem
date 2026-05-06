from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class MLSystemApiError(RuntimeError):
    pass


@dataclass
class MLSystemApiClient:
    base_url: str
    token: str | None = None
    timeout_sec: float = 30.0

    def start_stage(self, run_id: str, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/runs/{run_id}/stages/{stage}/start", payload)

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/jobs/{job_id}")

    def run_summary(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/runs/{run_id}/summary")

    def wait_for_job(self, job_id: str, poll_sec: float = 2.0, timeout_sec: float = 3600.0) -> dict[str, Any]:
        started = time.time()
        while True:
            status = self.job_status(job_id)
            state = status.get("state")
            if state in {"succeeded", "failed", "cancelled", "timed_out"}:
                return status
            if time.time() - started > timeout_sec:
                raise MLSystemApiError(f"Timed out waiting for MLSystem API job {job_id}")
            time.sleep(poll_sec)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}{path}", data=body, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise MLSystemApiError(f"MLSystem API HTTP {exc.code}: {text}") from exc
        except urllib.error.URLError as exc:
            raise MLSystemApiError(f"MLSystem API request failed: {exc}") from exc

