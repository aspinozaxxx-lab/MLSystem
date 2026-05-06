from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frontend.app.report_builder import build_annotation_report


class ReportBuilderTests(unittest.TestCase):
    def test_report_uses_existing_artifacts_without_recomputing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root / "run1"
            (run / "stages").mkdir(parents=True)
            (run / "inventory_scenes.json").write_text(
                json.dumps(
                    {
                        "scene_count": 2,
                        "matched_count": 1,
                        "missing_count": 1,
                        "matched": [{"entry": "a.tif"}],
                        "missing": ["missing.tif"],
                    }
                ),
                encoding="utf-8",
            )
            (run / "dataset_manifest.json").write_text(
                json.dumps(
                    {
                        "split_strategy": "object_balanced",
                        "train_scenes": [{"entry": "a.tif"}],
                        "val_scenes": [],
                        "scene_object_counts": [{"scene_name": "a.tif", "object_count": 3}],
                    }
                ),
                encoding="utf-8",
            )
            (run / "stages" / "inventory_scenes.json").write_text(json.dumps({"status": "success", "counters": {"matched_scenes": 1}}), encoding="utf-8")
            (run / "stages" / "prepare_dataset.json").write_text(json.dumps({"status": "success", "counters": {"total_objects": 3, "train_scenes": 1}}), encoding="utf-8")
            report = build_annotation_report("run1", root)
            self.assertEqual(report["status"], "succeeded")
            self.assertEqual(report["summary"]["matched_scenes"], 1)
            self.assertEqual(report["summary"]["missing_scenes"], 1)
            self.assertEqual(report["summary"]["total_objects"], 3)
            self.assertEqual(report["scene_rows"][0]["objects"], 3)
            self.assertEqual(report["scene_rows"][1]["storage_status"], "missing")


if __name__ == "__main__":
    unittest.main()

