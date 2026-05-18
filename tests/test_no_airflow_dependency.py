from __future__ import annotations

import unittest
from pathlib import Path


class NoAirflowDependencyTests(unittest.TestCase):
    def test_new_runner_and_api_do_not_import_removed_stage_wrapper(self) -> None:
        files = list(Path("mlsystem/src/pipeline_runner").glob("*.py")) + [
            Path("mlsystem/src/api/app.py"),
        ]
        for path in files:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("pipeline.airflow_tasks", text, path)
            self.assertNotIn("airflow_tasks", text, path)

    def test_pipeline_runner_package_has_no_airflow_terms(self) -> None:
        for path in Path("mlsystem/src/pipeline_runner").glob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("airflow", text, path)

    def test_ready_does_not_require_airflow_env(self) -> None:
        text = Path("mlsystem/src/api/app.py").read_text(encoding="utf-8")
        self.assertNotIn("MLSYSTEM_AIRFLOW_", text)
        self.assertIn("MLSYSTEM_RUN_ROOT", text)


if __name__ == "__main__":
    unittest.main()
