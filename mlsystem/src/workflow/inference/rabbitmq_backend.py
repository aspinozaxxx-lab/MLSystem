from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict
from typing import Any

from .contracts import InferenceJob, InferenceWorkflowConfig, WorkflowProgress


class RabbitMQInferenceBackend:
    name = "rabbitmq_triton"

    def __init__(self, config: InferenceWorkflowConfig) -> None:
        self.config = config

    def run(self, job: InferenceJob) -> WorkflowProgress:
        if not self.config.enabled:
            raise RuntimeError("rabbitmq_triton backend is disabled by feature flag")
        smoke = self.smoke_check()
        if smoke.get("status") != "ok":
            raise RuntimeError(f"RabbitMQ smoke failed: {smoke}")
        return WorkflowProgress(state="succeeded", total_scenes=len(job.scenes), completed_scenes=0)

    def smoke_check(self) -> dict[str, Any]:
        if not self.config.enabled:
            return {"status": "skipped", "reason": "rabbitmq_triton backend disabled"}
        try:
            import pika
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "reason": f"pika import failed: {exc}"}
        url = self.config.rabbitmq_url or os.getenv("MLSYSTEM_RABBITMQ_URL")
        if not url:
            return {"status": "failed", "reason": "MLSYSTEM_RABBITMQ_URL is not configured"}
        queue_name = f"{self.config.rabbitmq_queue_prefix}.smoke.{uuid.uuid4().hex[:8]}"
        try:
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=queue_name, durable=False, auto_delete=True)
            payload = json.dumps({"type": "mlsystem.rabbitmq_inference_smoke", "id": uuid.uuid4().hex})
            channel.basic_publish(exchange="", routing_key=queue_name, body=payload.encode("utf-8"))
            method, _, body = channel.basic_get(queue=queue_name, auto_ack=True)
            channel.queue_delete(queue=queue_name)
            connection.close()
            if method is None:
                return {"status": "failed", "reason": "published smoke message was not consumed"}
            return {"status": "ok", "queue": queue_name, "message": json.loads(body.decode("utf-8"))}
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "reason": str(exc), "queue": queue_name}


def main() -> int:
    parser = argparse.ArgumentParser(description="RabbitMQ inference workflow backend utilities")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--enable", action="store_true")
    args = parser.parse_args()
    config = InferenceWorkflowConfig(enabled=args.enable, backend="rabbitmq_triton")
    backend = RabbitMQInferenceBackend(config)
    if args.smoke:
        print(json.dumps(backend.smoke_check(), ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(asdict(config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
