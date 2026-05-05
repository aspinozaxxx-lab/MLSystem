from __future__ import annotations

import unittest

from mlsystem.src.workflow.inference.contracts import InferenceJob, InferenceWorkflowConfig, SceneTask
from mlsystem.src.workflow.inference.direct_triton_backend import DirectTritonBackend
from mlsystem.src.workflow.inference.engine import InferenceWorkflowEngine
from mlsystem.src.workflow.inference.rabbitmq_backend import RabbitMQInferenceBackend


class InferenceWorkflowEngineTests(unittest.TestCase):
    def test_default_backend_is_direct_triton(self) -> None:
        engine = InferenceWorkflowEngine(InferenceWorkflowConfig())
        self.assertIsInstance(engine.backend(), DirectTritonBackend)

    def test_direct_backend_returns_delegated_progress(self) -> None:
        config = InferenceWorkflowConfig(enabled=True)
        job = InferenceJob(job_id="job", scenes=[SceneTask("scene", "image.tif")], config=config)
        progress = InferenceWorkflowEngine(config).run(job)
        self.assertEqual(progress.state, "succeeded")
        self.assertEqual(progress.completed_scenes, 1)

    def test_rabbit_backend_disabled_smoke_is_skipped(self) -> None:
        backend = RabbitMQInferenceBackend(InferenceWorkflowConfig(enabled=False, backend="rabbitmq_triton"))
        self.assertEqual(backend.smoke_check()["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
