from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .messages import QUEUE_NAMES, QueueMessage


Handler = Callable[[QueueMessage], Awaitable[None]]
BatchHandler = Callable[[list[QueueMessage]], Awaitable[None]]


@dataclass
class QueueCounters:
    published: dict[str, int] = field(default_factory=dict)
    consumed: dict[str, int] = field(default_factory=dict)

    def inc_publish(self, queue: str) -> None:
        self.published[queue] = int(self.published.get(queue, 0)) + 1

    def inc_consume(self, queue: str) -> None:
        self.consumed[queue] = int(self.consumed.get(queue, 0)) + 1


class RabbitMQClient:
    def __init__(self, url: str, *, prefetch_count: int = 16, max_attempts: int = 3) -> None:
        self.url = url
        self.prefetch_count = prefetch_count
        self.max_attempts = max_attempts
        self.connection: Any | None = None
        self.channel: Any | None = None
        self.counters = QueueCounters()

    async def connect(self) -> None:
        import aio_pika

        self.connection = await aio_pika.connect_robust(self.url)
        self.channel = await self.connection.channel()
        await self.channel.set_qos(prefetch_count=self.prefetch_count)
        await declare_topology(self.channel)

    async def close(self) -> None:
        if self.connection is not None:
            await self.connection.close()

    async def publish(self, queue: str, message: QueueMessage) -> None:
        import aio_pika

        if self.channel is None:
            await self.connect()
        assert self.channel is not None
        body = message.to_json().encode("utf-8")
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body,
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                message_id=message.message_id,
                timestamp=int(time.time()),
                headers={"schema_version": message.schema_version, "stage": message.stage},
            ),
            routing_key=queue,
        )
        self.counters.inc_publish(queue)

    async def consume_forever(self, queue: str, handler: Handler) -> None:
        if self.channel is None:
            await self.connect()
        assert self.channel is not None
        q = await self.channel.get_queue(queue)

        async with q.iterator() as queue_iter:
            async for raw in queue_iter:
                async with raw.process(requeue=False):
                    message = QueueMessage.from_json(raw.body)
                    try:
                        await handler(message)
                    except Exception:
                        target_queue = retry_target(message, queue, self.max_attempts)
                        if target_queue != "ie.dead_letter":
                            await self.publish(target_queue, QueueMessage.from_dict({**message.to_dict(), "attempt": message.attempt + 1}))
                        else:
                            await self.publish(target_queue, message)
                        continue
                    self.counters.inc_consume(queue)

    async def consume_batches_forever(self, queue: str, *, max_batch_size: int, max_wait_ms: int, handler: BatchHandler) -> None:
        if self.channel is None:
            await self.connect()
        assert self.channel is not None
        q = await self.channel.get_queue(queue)
        max_batch_size = max(1, int(max_batch_size))
        max_wait_sec = max(0.001, float(max_wait_ms) / 1000.0)
        async with q.iterator() as queue_iter:
            while True:
                first = await queue_iter.__anext__()
                raw_messages = [first]
                started = time.monotonic()
                while len(raw_messages) < max_batch_size:
                    remaining = max(0.001, max_wait_sec - (time.monotonic() - started))
                    try:
                        item = await asyncio.wait_for(queue_iter.__anext__(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    raw_messages.append(item)
                messages = [QueueMessage.from_json(raw.body) for raw in raw_messages]
                try:
                    await handler(messages)
                except Exception:
                    for raw, message in zip(raw_messages, messages, strict=True):
                        target_queue = retry_target(message, queue, self.max_attempts)
                        if target_queue != "ie.dead_letter":
                            await self.publish(target_queue, QueueMessage.from_dict({**message.to_dict(), "attempt": message.attempt + 1}))
                        else:
                            await self.publish(target_queue, message)
                        await raw.ack()
                    continue
                for raw in raw_messages:
                    await raw.ack()
                for _ in raw_messages:
                    self.counters.inc_consume(queue)


async def declare_topology(channel: Any) -> None:
    dead_letter = await channel.declare_queue("ie.dead_letter", durable=True)
    del dead_letter
    args = {
        "x-dead-letter-exchange": "",
        "x-dead-letter-routing-key": "ie.dead_letter",
        "x-message-ttl": 7 * 24 * 60 * 60 * 1000,
    }
    for name in QUEUE_NAMES:
        if name == "ie.dead_letter":
            continue
        await channel.declare_queue(name, durable=True, arguments=args)


def run_consumer(url: str, queue: str, handler: Handler) -> None:
    async def _main() -> None:
        client = RabbitMQClient(url)
        await client.consume_forever(queue, handler)

    asyncio.run(_main())


def retry_target(message: QueueMessage, source_queue: str, max_attempts: int) -> str:
    return source_queue if message.attempt + 1 < int(max_attempts) else "ie.dead_letter"
