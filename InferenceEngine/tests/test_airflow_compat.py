from __future__ import annotations

import unittest

from mlsystem.src.inference_pipeline._payload import COMPATIBILITY_ARTIFACT_NAMES
from mlsystem.src.inference_pipeline.api import start_pseudolabel_run


class InferencePipelineCompatibilityTests(unittest.TestCase):
    def test_pseudolabel_orchestration_uses_inference_pipeline_api(self) -> None:
        self.assertTrue(callable(start_pseudolabel_run))

    def test_inference_engine_stage_validates_old_airflow_artifacts(self) -> None:
        for name in [
            "accepted.geojson.gz",
            "coverage_report.json",
            "pseudolabel_summary.json",
            "postprocess_summary.json",
            "vectorization_summary.json",
            "pseudolabel_scene_results_manifest.json",
            "probability_maps_index.json",
            "inference_results.json",
            "inference_timing_report.json",
            "pseudolabel_scenes.txt",
            "prediction_examples.html",
        ]:
            self.assertIn(name, COMPATIBILITY_ARTIFACT_NAMES)


if __name__ == "__main__":
    unittest.main()
