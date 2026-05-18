from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlsystem.src.train_pipeline.api import (
    StageJobRunner as JobRunner,
    StageJobStore as JobStore,
    StageStartRequest,
    debug_run_stage_sync,
    stages_payload,
    start_stage,
    validate_stage_name,
)


class ApiStageRoutesTests(unittest.TestCase):
    def test_stages_payload_contains_main_stages(self) -> None:
        payload = stages_payload()
        self.assertIn("inventory_scenes", payload["pipeline_stages"])
        self.assertIn("train_model", payload["pipeline_stages"])
        self.assertNotIn("inference_engine_pipeline", payload["pipeline_stages"])
        self.assertNotIn("run_pseudolabel_inference", payload["pipeline_stages"])
        self.assertEqual(payload["aliases"]["inventory"], "inventory_scenes")

    def test_unknown_stage_fails_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown stage"):
            validate_stage_name("unknown_stage")

    def test_start_stage_dry_run_creates_succeeded_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner = JobRunner(JobStore(Path(tmp)))
            request = StageStartRequest(experiment_config={"experiment_id": "unit"}, dry_run=True)
            response = start_stage("run1", "inventory_scenes", request, runner)
            self.assertEqual(response.state, "succeeded")
            status = runner.store.read_job(response.job_id)
            self.assertEqual(status.state, "succeeded")

    def test_debug_sync_inventory_missing_scene_persists_report_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JobStore(Path(tmp) / "jobs")
            status_root = Path(tmp) / "status"
            request = StageStartRequest(
                experiment_config={
                    "experiment_id": "unit_inventory_missing",
                    "images_uri": "s3://b/images/",
                    "layout_uri": "s3://b/layouts/",
                    "scenes_file": "scenes.txt",
                    "annotation_file": "auto",
                },
                status_root=str(status_root),
            )
            images = [{"bucket": "b", "key": "images/scene_a.tif", "name": "scene_a.tif", "size": 1}]
            with patch.multiple(
                "mlsystem.src.train_pipeline.stages.inventory_scenes",
                load_config=lambda: SimpleNamespace(),
                build_s3_layout_status=lambda _cfg: {"ok": True},
                list_s3_objects=lambda _cfg, _uri, suffixes=None: images,
                find_layout_files=lambda _cfg, _layout, _scenes, _ann: ("s3://b/layouts/ann.geojson", "s3://b/layouts/scenes.txt"),
                read_s3_text=lambda _cfg, _uri: "scene_a.tif\nmissing_scene.tif\n",
            ):
                status = debug_run_stage_sync("unit_inventory_missing", "inventory_scenes", request, store)
            self.assertEqual(status.state, "failed")
            self.assertIn("missing_scene.tif", (status.error.message + str(status.report)))
            run_dir = status_root / "unit_inventory_missing"
            self.assertTrue((run_dir / "missing_scenes.txt").exists())
            self.assertIn("missing_scene.tif", (run_dir / "missing_scenes.txt").read_text(encoding="utf-8"))
            self.assertTrue((run_dir / "stages" / "inventory_scenes.report.md").exists())
            self.assertIn("report_path", status.report)


if __name__ == "__main__":
    unittest.main()
