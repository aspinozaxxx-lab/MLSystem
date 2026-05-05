from __future__ import annotations

import unittest

from mlsystem.src.workflow.inference.contracts import InferenceWorkflowConfig, SceneTask, WorkflowProgress


class InferenceWorkflowContractsTests(unittest.TestCase):
    def test_defaults_are_direct_triton_disabled(self) -> None:
        config = InferenceWorkflowConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.backend, "direct_triton")

    def test_progress_fraction(self) -> None:
        progress = WorkflowProgress(total_scenes=4, completed_scenes=1, failed_scenes=1)
        self.assertEqual(progress.completion_fraction, 0.5)

    def test_scene_task(self) -> None:
        task = SceneTask(scene_id="scene", image_ref="s3://bucket/key.tif")
        self.assertEqual(task.scene_id, "scene")


if __name__ == "__main__":
    unittest.main()
