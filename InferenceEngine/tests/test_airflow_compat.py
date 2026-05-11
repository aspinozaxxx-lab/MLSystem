from __future__ import annotations

import unittest

from mlsystem.src.pipeline.stages.inference_engine_pipeline import COMPATIBILITY_ARTIFACTS
from mlsystem.src.pipeline.stages.registry import known_stages


class AirflowCompatibilityTests(unittest.TestCase):
    def test_production_registry_uses_single_inference_engine_stage(self) -> None:
        stages = known_stages()
        self.assertIn("inference_engine_pipeline", stages)
        self.assertNotIn("prepare_inference_scenes", stages)
        self.assertNotIn("run_pseudolabel_inference", stages)
        self.assertNotIn("validate_probability_maps", stages)
        self.assertNotIn("vectorize_pseudolabel", stages)
        self.assertNotIn("postprocess_pseudolabel", stages)
        self.assertNotIn("export_pseudolabel_artifacts", stages)

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
            self.assertIn(name, COMPATIBILITY_ARTIFACTS)


if __name__ == "__main__":
    unittest.main()
