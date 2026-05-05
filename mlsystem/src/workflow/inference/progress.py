from __future__ import annotations

from .contracts import InferenceResultRef, SceneFailure, WorkflowProgress


def mark_started(progress: WorkflowProgress, total_scenes: int) -> WorkflowProgress:
    progress.state = "running"
    progress.total_scenes = total_scenes
    return progress


def mark_scene_completed(progress: WorkflowProgress, result: InferenceResultRef) -> WorkflowProgress:
    progress.completed_scenes += 1
    progress.results.append(result)
    if progress.completed_scenes + progress.failed_scenes >= progress.total_scenes:
        progress.state = "succeeded" if not progress.failures else "failed"
    return progress


def mark_scene_failed(progress: WorkflowProgress, failure: SceneFailure, *, fail_workflow: bool) -> WorkflowProgress:
    progress.failed_scenes += 1
    progress.failures.append(failure)
    if fail_workflow:
        progress.state = "failed"
    elif progress.completed_scenes + progress.failed_scenes >= progress.total_scenes:
        progress.state = "succeeded"
    return progress
