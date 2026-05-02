from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .triton_client import TritonEndpoint, infer_identity_smoke, triton_ready


DEFAULT_QUEUE = "mlsystem.inference.tiles"
DEFAULT_RESULT_QUEUE = "mlsystem.inference.results"


@dataclass(frozen=True)
class RabbitMQEndpoint:
    url: str
    queue: str = DEFAULT_QUEUE
    result_queue: str = DEFAULT_RESULT_QUEUE


def rabbitmq_endpoint_from_env(queue: str | None = None, result_queue: str | None = None) -> RabbitMQEndpoint:
    url = os.getenv("MLSYSTEM_RABBITMQ_URL") or os.getenv("RABBITMQ_URL")
    if not url:
        raise RuntimeError("MLSYSTEM_RABBITMQ_URL/RABBITMQ_URL is not set")
    return RabbitMQEndpoint(
        url=url,
        queue=queue or os.getenv("MLSYSTEM_INFERENCE_QUEUE") or DEFAULT_QUEUE,
        result_queue=result_queue or os.getenv("MLSYSTEM_INFERENCE_RESULT_QUEUE") or DEFAULT_RESULT_QUEUE,
    )


def _connect(endpoint: RabbitMQEndpoint):
    import pika

    params = pika.URLParameters(endpoint.url)
    params.heartbeat = 60
    params.blocked_connection_timeout = 300
    return pika.BlockingConnection(params)


def declare_queues(endpoint: RabbitMQEndpoint) -> None:
    connection = _connect(endpoint)
    try:
        channel = connection.channel()
        for queue in (endpoint.queue, endpoint.result_queue):
            channel.queue_declare(queue=queue, durable=True)
    finally:
        connection.close()


def publish_json(endpoint: RabbitMQEndpoint, queue: str, payload: dict[str, Any]) -> None:
    import pika

    connection = _connect(endpoint)
    try:
        channel = connection.channel()
        channel.queue_declare(queue=queue, durable=True)
        channel.basic_publish(
            exchange="",
            routing_key=queue,
            body=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
        )
    finally:
        connection.close()


def queue_depth(endpoint: RabbitMQEndpoint, queue: str | None = None) -> int:
    connection = _connect(endpoint)
    try:
        channel = connection.channel()
        method = channel.queue_declare(queue=queue or endpoint.queue, durable=True, passive=True)
        return int(method.method.message_count)
    finally:
        connection.close()


def publish_identity_smoke_jobs(endpoint: RabbitMQEndpoint, count: int) -> list[dict[str, Any]]:
    declare_queues(endpoint)
    jobs = []
    for idx in range(count):
        payload = {
            "schema_version": 1,
            "kind": "identity_smoke",
            "tile_id": f"smoke-{idx:04d}",
            "values": [float(idx + 1)],
            "created_at": time.time(),
        }
        publish_json(endpoint, endpoint.queue, payload)
        jobs.append(payload)
    return jobs


def consume_identity_smoke_jobs(endpoint: RabbitMQEndpoint, triton: TritonEndpoint, max_jobs: int, timeout_sec: float) -> list[dict[str, Any]]:
    if not triton_ready(triton.url, timeout_sec=timeout_sec):
        raise RuntimeError(f"Triton is not ready: {triton.url}")

    connection = _connect(endpoint)
    results: list[dict[str, Any]] = []
    started = time.time()
    try:
        channel = connection.channel()
        channel.queue_declare(queue=endpoint.queue, durable=True)
        channel.queue_declare(queue=endpoint.result_queue, durable=True)
        channel.basic_qos(prefetch_count=max(1, min(32, max_jobs)))
        while len(results) < max_jobs and time.time() - started < timeout_sec:
            method, _properties, body = channel.basic_get(endpoint.queue, auto_ack=False)
            if method is None:
                time.sleep(0.1)
                continue
            payload = json.loads(body.decode("utf-8"))
            try:
                if payload.get("kind") != "identity_smoke":
                    raise RuntimeError(f"Unsupported message kind: {payload.get('kind')}")
                output = infer_identity_smoke(triton, np.asarray(payload["values"], dtype=np.float32))
                result = {
                    "schema_version": 1,
                    "kind": "identity_smoke_result",
                    "tile_id": payload.get("tile_id"),
                    "values": output.astype(float).tolist(),
                    "duration_sec": round(time.time() - float(payload.get("created_at") or started), 6),
                }
                publish_json(endpoint, endpoint.result_queue, result)
                channel.basic_ack(method.delivery_tag)
                results.append(result)
            except Exception:
                channel.basic_nack(method.delivery_tag, requeue=True)
                raise
    finally:
        connection.close()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MLSystem RabbitMQ -> Triton queue utilities")
    parser.add_argument("--queue", default=None)
    parser.add_argument("--result-queue", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    produce = subparsers.add_parser("smoke-produce")
    produce.add_argument("--count", type=int, default=8)

    consume = subparsers.add_parser("smoke-consume")
    consume.add_argument("--count", type=int, default=8)
    consume.add_argument("--timeout-sec", type=float, default=30.0)
    consume.add_argument("--triton-url", default=os.getenv("MLSYSTEM_TRITON_URL") or "http://triton:8000")
    consume.add_argument("--triton-model", default="identity_python")

    status = subparsers.add_parser("status")
    status.add_argument("--include-result", action="store_true")

    args = parser.parse_args(argv)
    endpoint = rabbitmq_endpoint_from_env(args.queue, args.result_queue)
    if args.command == "smoke-produce":
        jobs = publish_identity_smoke_jobs(endpoint, count=max(1, args.count))
        print(json.dumps({"published": len(jobs), "queue": endpoint.queue}, ensure_ascii=False))
        return 0
    if args.command == "smoke-consume":
        triton = TritonEndpoint(url=args.triton_url, model_name=args.triton_model)
        results = consume_identity_smoke_jobs(endpoint, triton, max_jobs=max(1, args.count), timeout_sec=args.timeout_sec)
        print(json.dumps({"consumed": len(results), "results": results}, ensure_ascii=False))
        return 0
    if args.command == "status":
        payload = {"queue": endpoint.queue, "depth": queue_depth(endpoint, endpoint.queue)}
        if args.include_result:
            payload["result_queue"] = endpoint.result_queue
            payload["result_depth"] = queue_depth(endpoint, endpoint.result_queue)
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
