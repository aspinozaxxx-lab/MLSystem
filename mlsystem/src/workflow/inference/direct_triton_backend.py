from __future__ import annotations

from .contracts import InferenceJob, InferenceResultRef, InferenceWorkflowConfig, WorkflowProgress
from .progress import mark_scene_completed, mark_started


class DirectTritonBackend:
    name = "direct_triton"

    def __init__(self, config: InferenceWorkflowConfig) -> None:
        self.config = config

    def run(self, job: InferenceJob) -> WorkflowProgress:
        progress = mark_started(WorkflowProgress(), len(job.scenes))
        for scene in job.scenes:
            # The production direct Triton path remains in SceneInferenceRunner/pseudolabel_pipeline.
            # This wrapper is a stable workflow boundary for the next extraction step.
            mark_scene_completed(
                progress,
                InferenceResultRef(scene_id=scene.scene_id, result_uri=scene.image_ref, metadata={"backend": self.name, "mode": "delegated"}),
            )
        return progress
