from __future__ import annotations

from .contracts import InferenceJob, InferenceWorkflowConfig, WorkflowProgress
from .direct_triton_backend import DirectTritonBackend
from .rabbitmq_backend import RabbitMQInferenceBackend


class InferenceWorkflowEngine:
    def __init__(self, config: InferenceWorkflowConfig) -> None:
        self.config = config

    def backend(self):
        if self.config.backend == "direct_triton":
            return DirectTritonBackend(self.config)
        if self.config.backend == "rabbitmq_triton":
            return RabbitMQInferenceBackend(self.config)
        raise ValueError(f"Unsupported inference workflow backend: {self.config.backend}")

    def run(self, job: InferenceJob) -> WorkflowProgress:
        return self.backend().run(job)
