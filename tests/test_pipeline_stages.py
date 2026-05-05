from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlsystem.src.storage.local_io import write_json
from mlsystem.src.pipeline.airflow_tasks import AirflowRunStore
from mlsystem.src.pipeline.stages.context import StageContext
from mlsystem.src.pipeline.stages.inventory_scenes import run as run_inventory_scenes
from mlsystem.src.pipeline.stages.export_pseudolabel import run as run_export_pseudolabel
from mlsystem.src.pipeline.stages.postprocess_pseudolabel import run as run_postprocess_pseudolabel
from mlsystem.src.pipeline.stages.prepare_dataset import run as run_prepare_dataset
from mlsystem.src.pipeline.stages.prepare_inference_scenes import run as run_prepare_inference_scenes
from mlsystem.src.pipeline.stages.probability_maps import run as run_probability_maps
from mlsystem.src.pipeline.stages.pseudolabel_inference import run as run_pseudolabel_inference
from mlsystem.src.pipeline.stages.report import StageFailure
from mlsystem.src.pipeline.stages.vectorize_pseudolabel import run as run_vectorize_pseudolabel


class PipelineStagesTests(unittest.TestCase):
    def test_inventory_scenes_success_and_missing_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp)
            images = [{"bucket": "b", "key": "images/scene_a.tif", "name": "scene_a.tif", "size": 1}]
            with self._inventory_patches(images, "scene_a.tif\n"):
                report = run_inventory_scenes(ctx)
            self.assertEqual(report.status, "success")
            self.assertTrue((ctx.store.run_dir / "inventory_scenes.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp)
            images = [{"bucket": "b", "key": "images/scene_a.tif", "name": "scene_a.tif", "size": 1}]
            with self._inventory_patches(images, "scene_a.tif\nmissing_scene.tif\n"):
                with self.assertRaises(StageFailure) as raised:
                    run_inventory_scenes(ctx)
            self.assertIn("missing_scene.tif", "\n".join(raised.exception.report.errors + (ctx.store.run_dir / "missing_scenes.txt").read_text(encoding="utf-8").splitlines()))

    def test_prepare_dataset_object_balanced_split_keeps_zero_scenes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, preprocess={"split_strategy": "object_balanced", "count_mode": "property", "target_val_fraction": 0.34, "split_seed": 3})
            self._write_inventory(ctx)
            annotation = {
                "type": "FeatureCollection",
                "features": [
                    self._feature("scene_a.tif"),
                    self._feature("scene_a.tif"),
                    self._feature("scene_b.tif"),
                ],
            }
            with patch("mlsystem.src.pipeline.stages.prepare_dataset.load_config", return_value=SimpleNamespace(storage=SimpleNamespace(heavy_backend="local", s3_bucket="b"), known_data_roots=[])), \
                patch("mlsystem.src.pipeline.stages.prepare_dataset.read_s3_text", return_value=json.dumps(annotation)), \
                patch("mlsystem.src.pipeline.stages.prepare_dataset.raster_path_for_s3_key", side_effect=lambda _cfg, key: str(ctx.store.run_dir / Path(key).name)):
                report = run_prepare_dataset(ctx)
            self.assertEqual(report.status, "success")
            split_summary = json.loads((ctx.store.run_dir / "split_summary.json").read_text(encoding="utf-8"))
            all_scenes = set((ctx.store.run_dir / "train_scenes.txt").read_text(encoding="utf-8").split()) | set((ctx.store.run_dir / "val_scenes.txt").read_text(encoding="utf-8").split())
            self.assertEqual(all_scenes, {"scene_a.tif", "scene_b.tif", "scene_c.tif"})
            self.assertEqual(split_summary["total_files"], 3)
            self.assertEqual(split_summary["total_objects"], 3)

    def test_prepare_dataset_default_legacy_without_schema_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, preprocess={"count_mode": "property"})
            self._write_inventory(ctx)
            annotation = {
                "type": "FeatureCollection",
                "features": [self._feature("scene_a.tif"), self._feature("scene_b.tif")],
            }
            with patch("mlsystem.src.pipeline.stages.prepare_dataset.load_config", return_value=SimpleNamespace(storage=SimpleNamespace(heavy_backend="local", s3_bucket="b"), known_data_roots=[])), \
                patch("mlsystem.src.pipeline.stages.prepare_dataset.read_s3_text", return_value=json.dumps(annotation)), \
                patch("mlsystem.src.pipeline.stages.prepare_dataset.raster_path_for_s3_key", side_effect=lambda _cfg, key: str(ctx.store.run_dir / Path(key).name)):
                report = run_prepare_dataset(ctx)
            self.assertEqual(report.counters["split_strategy"], "legacy_75_25")

    def test_prepare_inference_scenes_dataset_and_explicit_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, pseudolabel={"run_on": "dataset_scenes", "bad_scene_policy": "skip"})
            self._write_inventory(ctx)
            write_json(ctx.store.run_dir / "dataset_manifest.json", {"train_scenes": [], "val_scenes": []})
            report = run_prepare_inference_scenes(ctx)
            self.assertEqual(report.counters["inference_scenes"], 3)
            self.assertTrue((ctx.store.run_dir / "inference_manifest.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, pseudolabel={"run_on": "explicit_scene_list", "scene_list": ["missing.tif"]})
            self._write_inventory(ctx)
            with self.assertRaises(StageFailure):
                run_prepare_inference_scenes(ctx)

        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, pseudolabel={"run_on": "all_images"})
            self._write_inventory(ctx)
            report = run_prepare_inference_scenes(ctx)
            self.assertEqual(report.counters["inference_scenes"], 3)

        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, pseudolabel={"run_on": "synthetic"})
            report = run_prepare_inference_scenes(ctx)
            self.assertEqual(report.counters["inference_scenes"], 1)
            manifest = json.loads((ctx.store.run_dir / "inference_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["scenes"][0]["entry"], "synthetic")

    def test_run_pseudolabel_inference_uses_inference_manifest_scene_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, pseudolabel={"enabled": True, "run_on": "dataset_scenes"})
            write_json(
                ctx.store.run_dir / "inference_manifest.json",
                {
                    "run_on": "dataset_scenes",
                    "bad_scene_policy": "skip",
                    "scenes": [
                        {"entry": "scene_a.tif", "name": "scene_a.tif", "key": "images/scene_a.tif"},
                        {"entry": "scene_b.tif", "name": "scene_b.tif", "key": "images/scene_b.tif"},
                    ],
                },
            )
            write_json(ctx.store.run_dir / "coverage_report.json", {"scenes_processed": 2, "total_predicted_windows": 4})
            write_json(ctx.store.run_dir / "pseudolabel_scene_results_manifest.json", {"scenes": ["scene_a.tif", "scene_b.tif"]})
            captured: dict[str, object] = {}

            def fake_pipeline(conf, _store, *, stage_mode: str):
                captured["stage_mode"] = stage_mode
                captured["pseudolabel"] = conf.pseudolabel
                return {"pseudolabel": {"accepted_objects": 0}}

            with patch("mlsystem.src.pipeline.airflow_tasks._run_pseudolabel_pipeline", side_effect=fake_pipeline):
                report = run_pseudolabel_inference(ctx)
            self.assertEqual(report.status, "success")
            self.assertEqual(captured["stage_mode"], "inference")
            self.assertEqual(captured["pseudolabel"]["scene_entries"], ["scene_a.tif", "scene_b.tif"])

    def test_synthetic_smoke_writes_compatibility_artifacts_for_downstream_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, pseudolabel={"enabled": True, "run_on": "synthetic"}, smoke=True)
            self.assertEqual(run_pseudolabel_inference(ctx).status, "success")
            self.assertTrue((ctx.store.run_dir / "coverage_report.json").exists())
            self.assertTrue((ctx.store.run_dir / "pseudolabel_summary.json").exists())
            self.assertEqual(run_probability_maps(ctx).status, "success")
            self.assertEqual(run_vectorize_pseudolabel(ctx).status, "success")
            self.assertEqual(run_postprocess_pseudolabel(ctx).status, "success")
            self.assertEqual(run_export_pseudolabel(ctx).status, "success")

    def _context(self, tmp: str, *, preprocess: dict | None = None, pseudolabel: dict | None = None, smoke: bool = False) -> StageContext:
        conf = SimpleNamespace(
            schema_version=None,
            experiment_id="unit_stage",
            smoke=smoke,
            images_uri="s3://b/images/",
            layout_uri="s3://b/layouts/",
            scenes_file="scenes.txt",
            annotation_file="auto",
            preprocess=preprocess or {},
            pseudolabel=pseudolabel or {},
        )
        store = AirflowRunStore(Path(tmp), "manual__unit", {"experiment_id": "unit_stage"})
        return StageContext("unit_stage", "manual__unit", conf, {}, Path(tmp), store, logging.getLogger("test"))

    def _inventory_patches(self, images: list[dict], scenes_text: str):
        return patch.multiple(
            "mlsystem.src.pipeline.stages.inventory_scenes",
            load_config=lambda: SimpleNamespace(),
            build_s3_layout_status=lambda _cfg: {"ok": True},
            list_s3_objects=lambda _cfg, _uri, suffixes=None: images,
            find_layout_files=lambda _cfg, _layout, _scenes, _ann: ("s3://b/layouts/ann.geojson", "s3://b/layouts/scenes.txt"),
            read_s3_text=lambda _cfg, _uri: scenes_text,
        )

    def _write_inventory(self, ctx: StageContext) -> None:
        matched = [
            {"entry": "scene_a.tif", "name": "scene_a.tif", "key": "images/scene_a.tif", "score": 1.0},
            {"entry": "scene_b.tif", "name": "scene_b.tif", "key": "images/scene_b.tif", "score": 1.0},
            {"entry": "scene_c.tif", "name": "scene_c.tif", "key": "images/scene_c.tif", "score": 1.0},
        ]
        inventory = {
            "annotation_uri": "s3://b/layouts/ann.geojson",
            "matched": matched,
            "available_images": matched,
        }
        write_json(ctx.store.run_dir / "inventory_scenes.json", inventory)
        write_json(ctx.store.run_dir / "scene_matching_report.json", {"matched": matched})

    @staticmethod
    def _feature(scene_name: str) -> dict:
        return {
            "type": "Feature",
            "properties": {"image_name": scene_name},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        }


if __name__ == "__main__":
    unittest.main()
