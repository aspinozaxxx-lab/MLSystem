from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shapely.geometry import box, mapping

from mlsystem.src.mlflow_adapter.api import MLFLOW_EXCLUDED_ARTIFACT_NAMES
from mlsystem.src.pipeline.contracts import PostprocessResult
from mlsystem.src.postprocessing.pseudolabel_export import export_pseudolabel_artifacts


class PseudolabelExportTests(unittest.TestCase):
    def test_accepted_geojson_named_by_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            post = PostprocessResult(
                features=[{"type": "Feature", "properties": {"area_m2": 1.0}, "geometry": mapping(box(0, 0, 1, 1))}],
                objects_before_filter=1,
                objects_after_filter=1,
                objects_after_top=1,
                params={"threshold": 0.5},
            )
            result = export_pseudolabel_artifacts(
                experiment_dir=root,
                job_id="job_123",
                postprocess_result=post,
                windows_preview_features=[],
                tile_insert_features=[],
                tiling_debug={"schema_version": 1},
                segformer_debug={"schema_version": 1},
            )
            self.assertEqual(result["geojson_path"].name, "job_123.accepted.geojson")
            self.assertTrue(result["geojson_path"].exists())

    def test_mlflow_excluded_artifact_policy_keeps_heavy_debug_out(self) -> None:
        self.assertIn("accepted.geojson.gz", MLFLOW_EXCLUDED_ARTIFACT_NAMES)
        self.assertIn("accepted.gpkg", MLFLOW_EXCLUDED_ARTIFACT_NAMES)
        self.assertIn("windows_preview.geojson", MLFLOW_EXCLUDED_ARTIFACT_NAMES)


if __name__ == "__main__":
    unittest.main()
