from __future__ import annotations

import asyncio
import unittest

from InferenceEngine.src.inference_engine.queues.messages import QueueMessage, make_message, validate_queue_contracts
from InferenceEngine.src.inference_engine.queues.rabbitmq import RabbitMQClient, message_with_failure, retry_target


class MessageContractTests(unittest.TestCase):
    def test_message_roundtrip_and_deterministic_id(self) -> None:
        message = make_message(job_id="job1", stage="block.vectorize", scene_id="scene", block_id="block")
        decoded = QueueMessage.from_json(message.to_json())
        self.assertEqual(decoded.message_id, "job1/scene/block/block.vectorize")
        self.assertEqual(decoded.stage, "block.vectorize")

    def test_queue_contract_validation(self) -> None:
        result = validate_queue_contracts()
        self.assertIn("ie.tile.infer", result["queues"])
        self.assertEqual(result["sample_message_id"], "job/scene/tile/tile.infer")

    def test_retry_dead_letter_policy(self) -> None:
        message = make_message(job_id="job", stage="tile.infer", scene_id="scene", tile_id="tile")
        self.assertEqual(retry_target(message, "ie.tile.infer", 3), "ie.tile.infer")
        exhausted = QueueMessage.from_dict({**message.to_dict(), "attempt": 2})
        self.assertEqual(retry_target(exhausted, "ie.tile.infer", 3), "ie.dead_letter")

    def test_retry_message_carries_failure_metadata(self) -> None:
        message = make_message(job_id="job", stage="jobs.submit")
        failed = message_with_failure(message, ValueError("bad manifest"), source_queue="ie.jobs.submit", target_queue="ie.dead_letter", max_attempts=3)
        self.assertEqual(failed.payload["last_error"]["error_type"], "ValueError")
        self.assertIn("bad manifest", failed.payload["last_error"]["error"])
        self.assertEqual(failed.payload["last_error"]["source_queue"], "ie.jobs.submit")

    def test_batch_consumer_reopens_after_iterator_stop(self) -> None:
        raw = _FakeRaw(make_message(job_id="job", stage="tile.infer", scene_id="scene", tile_id="tile"))
        client = RabbitMQClient("amqp://guest:guest@localhost/")
        client.channel = _FakeChannel(raw)
        seen: list[list[QueueMessage]] = []

        async def handler(messages: list[QueueMessage]) -> None:
            seen.append(messages)

        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(client.consume_batches_forever("ie.tile.infer", max_batch_size=4, max_wait_ms=1, handler=handler))
        self.assertEqual([[item.message_id for item in batch] for batch in seen], [["job/scene/tile/tile.infer"]])
        self.assertTrue(raw.acked)


class _FakeRaw:
    def __init__(self, message: QueueMessage) -> None:
        self.body = message.to_json().encode("utf-8")
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


class _FakeQueueIterator:
    def __init__(self, raw: _FakeRaw) -> None:
        self.raw = raw
        self.done = False

    async def __aenter__(self) -> "_FakeQueueIterator":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def __anext__(self) -> _FakeRaw:
        if self.done:
            raise StopAsyncIteration
        self.done = True
        return self.raw


class _FakeQueue:
    def __init__(self, raw: _FakeRaw) -> None:
        self.raw = raw

    def iterator(self) -> _FakeQueueIterator:
        return _FakeQueueIterator(self.raw)


class _FakeChannel:
    def __init__(self, raw: _FakeRaw) -> None:
        self.raw = raw
        self.calls = 0

    async def get_queue(self, _queue: str) -> _FakeQueue:
        self.calls += 1
        if self.calls > 1:
            raise asyncio.CancelledError
        return _FakeQueue(self.raw)


if __name__ == "__main__":
    unittest.main()
