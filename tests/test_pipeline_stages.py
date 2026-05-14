from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mlsystem.src.storage.local_io import write_json
from mlsystem.src.pipeline_runner.run_store import PipelineRunStore
from mlsystem.src.pipeline.stages.context import StageContext
from mlsystem.src.pipeline.stages.inventory_scenes import run as run_inventory_scenes
from mlsystem.src.pipeline.stages.inference_engine_pipeline import _shared_run_dir_for_inference_engine, run as run_inference_engine_pipeline
from mlsystem.src.pipeline.stages.prepare_dataset import run as run_prepare_dataset
from mlsystem.src.pipeline.stages.report import StageFailure


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

    def test_inventory_scenes_expands_folder_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp)
            images = [
                {"bucket": "b", "key": "images/Hilokskij/a.tif", "name": "a.tif", "size": 1},
                {"bucket": "b", "key": "images/Hilokskij/b.TIF", "name": "b.TIF", "size": 1},
                {"bucket": "b", "key": "images/Toguchinskij/c.tif", "name": "c.tif", "size": 1},
            ]
            with self._inventory_patches(images, "hilokskij\nToguchinskij/c.tif\n"):
                report = run_inventory_scenes(ctx)
            self.assertEqual(report.status, "success")
            self.assertEqual(report.counters["requested_entries_count"], 2)
            self.assertEqual(report.counters["requested_folders_count"], 1)
            self.assertEqual(report.counters["expanded_scene_count"], 3)
            inventory = json.loads((ctx.store.run_dir / "inventory_scenes.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["scene_count"], 3)
            self.assertEqual([item["entry"] for item in inventory["matched"]], ["images/Hilokskij/a.tif", "images/Hilokskij/b.TIF", "Toguchinskij/c.tif"])

    def test_inventory_scenes_expands_user_folder_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp)
            images = [
                {"bucket": "b", "key": "images/Hilokskij/a.tif", "name": "a.tif", "size": 1},
                {"bucket": "b", "key": "images/Toguchinskij/b.tif", "name": "b.tif", "size": 1},
                {"bucket": "b", "key": "images/Irkutsk/c.TIFF", "name": "c.TIFF", "size": 1},
                {"bucket": "b", "key": "images/Other/d.tif", "name": "d.tif", "size": 1},
            ]
            with self._inventory_patches(images, "Hilokskij\nToguchinskij\nirkutsk\n"):
                report = run_inventory_scenes(ctx)
            self.assertEqual(report.status, "success")
            self.assertEqual(report.counters["requested_entries_count"], 3)
            self.assertEqual(report.counters["requested_folders_count"], 3)
            self.assertEqual(report.counters["expanded_scene_count"], 3)
            inventory = json.loads((ctx.store.run_dir / "inventory_scenes.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["missing"], [])
            self.assertEqual(set(inventory["folder_expansions"]), {"Hilokskij", "Toguchinskij", "irkutsk"})

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
            self.assertEqual(report.counters["upstream_inventory_matched_scenes"], 3)
            self.assertEqual(report.counters["selected_dataset_scenes"], 3)
            self.assertFalse(report.counters["limit_applied"])

    def test_prepare_dataset_uses_full_inventory_without_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(
                tmp,
                preprocess={"split_strategy": "object_balanced", "count_mode": "property", "target_val_fraction": 0.2, "split_seed": 11},
            )
            matched = self._write_inventory_rows(ctx, 24)
            annotation = {"type": "FeatureCollection", "features": [self._feature("scene_00.tif"), self._feature("scene_05.tif"), self._feature("scene_05.tif")]}
            with self._prepare_dataset_patches(ctx, annotation):
                report = run_prepare_dataset(ctx)
            audit = json.loads((ctx.store.run_dir / "prepare_dataset_input_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(matched), 24)
            self.assertEqual(report.counters["total_scenes"], 24)
            self.assertEqual(report.counters["upstream_inventory_matched_scenes"], 24)
            self.assertEqual(report.counters["selected_dataset_scenes"], 24)
            self.assertEqual(report.counters["excluded_dataset_scenes"], 0)
            self.assertEqual(audit["invariant_status"], "OK")
            self.assertFalse(audit["limit_applied"])

    def test_prepare_dataset_explicit_limit_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(
                tmp,
                preprocess={
                    "split_strategy": "object_balanced",
                    "count_mode": "property",
                    "max_dataset_scenes": 2,
                    "dataset_limit_reason": "unit mini dataset",
                },
            )
            self._write_inventory_rows(ctx, 24)
            annotation = {"type": "FeatureCollection", "features": [self._feature("scene_00.tif"), self._feature("scene_01.tif")]}
            with self._prepare_dataset_patches(ctx, annotation):
                report = run_prepare_dataset(ctx)
            audit = json.loads((ctx.store.run_dir / "prepare_dataset_input_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(report.counters["total_scenes"], 2)
            self.assertEqual(report.counters["upstream_inventory_matched_scenes"], 24)
            self.assertEqual(report.counters["selected_dataset_scenes"], 2)
            self.assertEqual(report.counters["excluded_dataset_scenes"], 22)
            self.assertTrue(report.counters["limit_applied"])
            self.assertEqual(audit["limit_source"], "pipeline_trace.preprocess.max_dataset_scenes")
            self.assertEqual(audit["invariant_status"], "OK with explicit limit")
            self.assertEqual(len(audit["excluded_scenes"]), 22)
            self.assertTrue(any("Dataset input was explicitly limited" in warning for warning in report.warnings))

    def test_prepare_dataset_mismatch_without_explicit_limit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(tmp, preprocess={"split_strategy": "object_balanced", "count_mode": "property"})
            matched = self._write_inventory_rows(ctx, 2)
            inventory = json.loads((ctx.store.run_dir / "inventory_scenes.json").read_text(encoding="utf-8"))
            inventory["matched_count"] = 24
            inventory["matched"] = matched
            write_json(ctx.store.run_dir / "inventory_scenes.json", inventory)
            with self.assertRaises(StageFailure) as raised:
                run_prepare_dataset(ctx)
            text = "\n".join(raised.exception.report.errors)
            self.assertIn("prepare_dataset input mismatch", text)
            self.assertIn("inventory_scenes.json path=", text)
            self.assertIn("dataset_manifest.json path=", text)
            audit = json.loads((ctx.store.run_dir / "prepare_dataset_input_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["invariant_status"].startswith("FAILED"))

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

    def test_inference_engine_pipeline_submits_and_polls_http(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = self._context(
                tmp,
                pseudolabel={"enabled": True, "source": "inference_engine", "run_on": "dataset_scenes", "max_scenes": 2},
            )
            self._write_inventory(ctx)
            self._write_inference_engine_compat_artifacts(ctx)
            calls: list[dict[str, object]] = []

            class FakeResponse:
                def __init__(self, payload: dict) -> None:
                    self.payload = json.dumps(payload).encode("utf-8")

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def read(self) -> bytes:
                    return self.payload

            states = [
                {"job_id": "job-unit", "status": "running", "metrics": {"tiles_done": 1}},
                {
                    "job_id": "job-unit",
                    "status": "success",
                    "metrics": {
                        "tiles_total": 4,
                        "tiles_done": 4,
                        "blocks_total": 2,
                        "blocks_done": 2,
                        "triton_batches": 2,
                        "streaming_overlap_sec": 1.5,
                    },
                    "artifacts": {"accepted_geojson": str(ctx.store.run_dir / "unit_stage.accepted.geojson")},
                },
            ]

            def fake_urlopen(request, timeout=30):
                method = request.get_method()
                url = request.full_url
                body = json.loads(request.data.decode("utf-8")) if getattr(request, "data", None) else None
                calls.append({"method": method, "url": url, "body": body, "timeout": timeout})
                if method == "POST" and url == "http://ie.local/api/v1/jobs":
                    return FakeResponse({"job_id": "job-unit", "status": "queued"})
                if method == "GET" and url == "http://ie.local/api/v1/jobs/job-unit":
                    return FakeResponse(states.pop(0))
                if method == "GET" and url == "http://ie.local/api/v1/jobs/job-unit/artifacts":
                    return FakeResponse({"job_id": "job-unit", "artifacts": {"accepted_geojson": "ok"}})
                raise AssertionError(f"unexpected request {method} {url}")

            with patch.dict("os.environ", {"INFERENCE_ENGINE_API_URL": "http://ie.local", "INFERENCE_ENGINE_PIPELINE_POLL_SEC": "0"}), \
                patch("urllib.request.urlopen", side_effect=fake_urlopen):
                report = run_inference_engine_pipeline(ctx)

            self.assertEqual(report.status, "success")
            self.assertTrue((ctx.store.run_dir / "inference_manifest.json").exists())
            self.assertTrue(any(call["method"] == "POST" and call["url"] == "http://ie.local/api/v1/jobs" for call in calls))
            self.assertTrue(any(call["method"] == "GET" and call["url"] == "http://ie.local/api/v1/jobs/job-unit" for call in calls))
            post_body = next(call["body"] for call in calls if call["method"] == "POST")
            self.assertEqual(post_body["source"], "inference_engine")
            self.assertEqual(len(post_body["scenes"]), 3)
            self.assertEqual(report.details["inference_engine_api_url"], "http://ie.local")
            self.assertEqual(report.details["inference_engine_job_id"], "job-unit")
            self.assertTrue(report.details["request_submitted_via_http"])
            self.assertEqual(report.counters["backend"], "inference_engine")
            self.assertEqual(report.counters["inference_engine_http_submitted"], 1)

    def test_inference_engine_payload_uses_shared_pipeline_run_path(self) -> None:
        mapped = _shared_run_dir_for_inference_engine(Path("/data/mlsystem/runs/unit_run"))
        self.assertEqual(mapped, Path("/data/mlsystem/runs/unit_run"))
        unchanged = _shared_run_dir_for_inference_engine(Path("/tmp/unit_run"))
        self.assertEqual(unchanged, Path("/tmp/unit_run"))

    def _context(self, tmp: str, *, preprocess: dict | None = None, pseudolabel: dict | None = None, smoke: bool = False) -> StageContext:
        preprocess = preprocess or {}
        pseudolabel = pseudolabel or {}
        conf = SimpleNamespace(
            schema_version=None,
            experiment_id="unit_stage",
            smoke=smoke,
            images_uri="s3://b/images/",
            layout_uri="s3://b/layouts/",
            scenes_file="scenes.txt",
            annotation_file="auto",
            preprocess=preprocess,
            pseudolabel=pseudolabel,
        )
        store = PipelineRunStore(Path(tmp), "manual__unit", {"experiment_id": "unit_stage"})
        raw_conf = {"experiment_id": "unit_stage", "preprocess": dict(preprocess), "pseudolabel": dict(pseudolabel)}
        return StageContext("unit_stage", "manual__unit", conf, raw_conf, Path(tmp), store, logging.getLogger("test"))

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
        self._write_inventory_payload(ctx, matched)

    def _write_inventory_rows(self, ctx: StageContext, count: int) -> list[dict]:
        matched = [
            {"entry": f"scene_{idx:02d}.tif", "name": f"scene_{idx:02d}.tif", "key": f"images/scene_{idx:02d}.tif", "score": 1.0}
            for idx in range(count)
        ]
        self._write_inventory_payload(ctx, matched)
        return matched

    def _write_inventory_payload(self, ctx: StageContext, matched: list[dict]) -> None:
        inventory = {
            "annotation_uri": "s3://b/layouts/ann.geojson",
            "matched_count": len(matched),
            "matched": matched,
            "available_images": matched,
        }
        write_json(ctx.store.run_dir / "inventory_scenes.json", inventory)
        write_json(ctx.store.run_dir / "scene_matching_report.json", {"matched_count": len(matched), "matched": matched})
        (ctx.store.run_dir / "matched_scenes.txt").write_text("\n".join(item["entry"] for item in matched) + "\n", encoding="utf-8")

    def _write_inference_engine_compat_artifacts(self, ctx: StageContext) -> None:
        write_json(ctx.store.run_dir / "coverage_report.json", {"scenes_processed": 2, "scenes_failed": 0, "total_predicted_windows": 4})
        write_json(ctx.store.run_dir / "pseudolabel_summary.json", {"metrics": {"accepted_objects": 1}})
        write_json(ctx.store.run_dir / "postprocess_summary.json", {"final_objects": 1})
        write_json(ctx.store.run_dir / "vectorization_summary.json", {"blocks_total": 2, "blocks_done": 2})
        write_json(ctx.store.run_dir / "pseudolabel_scene_results_manifest.json", {"scenes": ["scene_a.tif", "scene_b.tif"]})
        write_json(ctx.store.run_dir / "inference_timing_report.json", {"streaming_overlap_sec": 1.5})
        (ctx.store.run_dir / "pseudolabel_scenes.txt").write_text("scene_a.tif\nscene_b.tif\n", encoding="utf-8")
        (ctx.store.run_dir / "prediction_examples.html").write_text("<html></html>", encoding="utf-8")
        (ctx.store.run_dir / "accepted.geojson.gz").write_bytes(b"gz")
        (ctx.store.run_dir / "unit_stage.accepted.geojson").write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

    def _prepare_dataset_patches(self, ctx: StageContext, annotation: dict):
        return patch.multiple(
            "mlsystem.src.pipeline.stages.prepare_dataset",
            load_config=lambda: SimpleNamespace(storage=SimpleNamespace(heavy_backend="local", s3_bucket="b"), known_data_roots=[]),
            read_s3_text=lambda _cfg, _uri: json.dumps(annotation),
            raster_path_for_s3_key=lambda _cfg, key: str(ctx.store.run_dir / Path(key).name),
        )

    @staticmethod
    def _feature(scene_name: str) -> dict:
        return {
            "type": "Feature",
            "properties": {"image_name": scene_name},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        }


if __name__ == "__main__":
    unittest.main()
