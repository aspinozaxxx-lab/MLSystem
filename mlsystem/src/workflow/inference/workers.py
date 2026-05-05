from __future__ import annotations

from .contracts import SceneTask


def scene_task_name(task: SceneTask) -> str:
    return task.scene_id
