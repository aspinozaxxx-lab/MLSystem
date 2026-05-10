from __future__ import annotations

import unittest

from InferenceEngine.src.inference_engine.queues.messages import QueueMessage, make_message, validate_queue_contracts
from InferenceEngine.src.inference_engine.queues.rabbitmq import retry_target


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


if __name__ == "__main__":
    unittest.main()
