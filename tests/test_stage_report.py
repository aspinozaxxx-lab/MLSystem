from __future__ import annotations

import unittest

from mlsystem.src.pipeline.stages.report import StageCheck, StageReport


class StageReportTests(unittest.TestCase):
    def test_success_log_contains_checks_counters_and_artifacts(self) -> None:
        report = StageReport(
            "inventory_scenes",
            checks=[StageCheck("config", "ok", "valid")],
            counters={"matched_scenes": 3},
            artifacts={"inventory_scenes.json": "/tmp/inventory_scenes.json"},
        )
        text = report.to_pipeline_log()
        self.assertIn("=== inventory_scenes summary ===", text)
        self.assertIn("config - OK: valid", text)
        self.assertIn("- matched_scenes: 3", text)
        self.assertIn("inventory_scenes.json", text)

    def test_failed_log_contains_warnings_and_errors(self) -> None:
        report = StageReport(
            "prepare_dataset",
            status="failed",
            warnings=["zero-object scene"],
            errors=["validation split is empty"],
        )
        text = report.to_pipeline_log()
        self.assertIn("=== prepare_dataset FAILED ===", text)
        self.assertIn("zero-object scene", text)
        self.assertIn("validation split is empty", text)


if __name__ == "__main__":
    unittest.main()
