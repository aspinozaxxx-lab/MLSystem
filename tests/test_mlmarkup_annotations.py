from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mlsystem.src.mlflow_adapter import _is_excluded_metric_key
from mlsystem.src.pipeline.experiment_stages import ExperimentStageConfig, _build_pipeline_job, _git_output
from mlsystem.src.pipeline_config import PipelineConfig
from mlsystem.src.storage.s3 import find_layout_files, read_s3_text


class MLMarkupAnnotationTests(unittest.TestCase):
    def test_pipeline_conf_resolves_mlmarkup_annotation_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "MLMarkup"
            class_dir = repo / "Вырубки"
            class_dir.mkdir(parents=True)
            (class_dir / "deforestation.txt").write_text("scene_a.tif\n", encoding="utf-8")
            (class_dir / "deforestation.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

            conf = ExperimentStageConfig.model_validate(
                {
                    "experiment_id": "mlmarkup_unit",
                    "class_name": "вырубки",
                    "annotations": {"source": "MLMarkup", "repo_path": str(repo)},
                    "model": {"architecture": "segformer", "backbone": "segformer-b1"},
                    "pseudolabel": {"enabled": True},
                }
            )
            self.assertEqual(conf.layout_uri, str(class_dir))
            self.assertEqual(conf.scenes_file, "deforestation.txt")
            self.assertEqual(conf.annotation_file, "deforestation.geojson")
            self.assertFalse(conf.pseudolabel["enabled"])

            job = _build_pipeline_job(conf)
            self.assertEqual(job.data["layout_uri"], str(class_dir))
            self.assertEqual(job.data["annotations"]["source"], "MLMarkup")
            self.assertEqual(job.train["model_name"], "segformer_b1")
            self.assertEqual(job.params["annotations.source"], "MLMarkup")
            self.assertFalse(job.params["pseudolabeling.enabled"])

    def test_local_layout_files_are_read_without_s3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "deforestation.txt").write_text("scene_a.tif\n", encoding="utf-8")
            (root / "deforestation.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            config = PipelineConfig(project_root=root, storage_root=root / "storage", logs_root=root / "logs")
            annotation_uri, scenes_uri = find_layout_files(config, str(root), "deforestation.txt", "deforestation.geojson")
            self.assertEqual(Path(annotation_uri).name, "deforestation.geojson")
            self.assertEqual(read_s3_text(config, scenes_uri), "scene_a.tif\n")

    def test_recource_and_resource_metrics_are_excluded_from_mlflow(self) -> None:
        for key in ("recource/cpu", "recource_cpu", "resource/final_ram", "resources.disk", "resourse/foo"):
            self.assertTrue(_is_excluded_metric_key(key))
        self.assertFalse(_is_excluded_metric_key("val/pixel_f1"))

    def test_mlmarkup_git_metadata_can_be_read_without_git_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "MLMarkup"
            ref_dir = repo / ".git" / "refs" / "heads"
            ref_dir.mkdir(parents=True)
            sha = "d68e23153078f32fa6479a55d2c0cb7e5d16c12d"
            (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (ref_dir / "main").write_text(f"{sha}\n", encoding="utf-8")

            self.assertEqual(_git_output(repo, "rev-parse", "HEAD"), sha)
            self.assertEqual(_git_output(repo, "branch", "--show-current"), "main")
            self.assertEqual(_git_output(repo, "status", "--short"), "")


if __name__ == "__main__":
    unittest.main()
