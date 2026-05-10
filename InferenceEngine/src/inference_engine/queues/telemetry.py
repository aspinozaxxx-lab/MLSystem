from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Any


def rabbitmq_queue_metrics() -> list[dict[str, Any]]:
    management_url = os.getenv("INFERENCE_ENGINE_RABBITMQ_MANAGEMENT_URL")
    if not management_url:
        return []
    user = os.getenv("RABBITMQ_DEFAULT_USER") or "guest"
    password = os.getenv("RABBITMQ_DEFAULT_PASS") or "guest"
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        management_url.rstrip("/") + "/api/queues/%2F",
        headers={"Authorization": f"Basic {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            rows = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    metrics = []
    for row in rows:
        name = str(row.get("name") or "")
        if not name.startswith("ie."):
            continue
        metrics.append(
            {
                "name": name,
                "messages_ready": int(row.get("messages_ready") or 0),
                "messages_unacked": int(row.get("messages_unacknowledged") or 0),
                "consumers": int(row.get("consumers") or 0),
            }
        )
    return sorted(metrics, key=lambda item: item["name"])
