from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

WorkflowState = Literal["created", "running", "succeeded", "failed", "cancelled"]
InferenceBackendName = Literal["direct_triton", "rabbitmq_triton"]
BadScenePolicy = Literal["fail", "skip"]


@dataclass
class InferenceWorkflowConfig:
    enabled: bool = False
    backend: InferenceBackendName = "direct_triton"
    max_scene_workers: int = 1
    rabbitmq_queue_prefix: str = "mlsystem.inference"
    bad_scene_policy: BadScenePolicy = "fail"
    rabbitmq_url: str | None = None


@dataclass
class SceneTask:
    scene_id: str
    image_ref: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TileTask:
    scene_id: str
    tile_id: str
    window: tuple[int, int, int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceJob:
    job_id: str
    scenes: list[SceneTask]
    config: InferenceWorkflowConfig


@dataclass
class InferenceResultRef:
    scene_id: str
    result_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbabilityMapRef:
    scene_id: str
    probability_map_uri: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneFailure:
    scene_id: str
    message: str
    recoverable: bool = True


@dataclass
class WorkflowProgress:
    state: WorkflowState = "created"
    total_scenes: int = 0
    completed_scenes: int = 0
    failed_scenes: int = 0
    failures: list[SceneFailure] = field(default_factory=list)
    results: list[InferenceResultRef] = field(default_factory=list)

    @property
    def completion_fraction(self) -> float:
        if self.total_scenes <= 0:
            return 0.0
        return (self.completed_scenes + self.failed_scenes) / self.total_scenes
