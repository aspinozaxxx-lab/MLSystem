from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from InferenceEngine.src.inference_engine.api.schemas import JobRequest, ResourceConfig, SceneInput
from InferenceEngine.src.inference_engine.config.settings import InferenceEngineSettings
from InferenceEngine.src.inference_engine.planning.planner import build_job_plan
from InferenceEngine.src.inference_engine.storage.raster_paths import rasterio_path_for_uri
from InferenceEngine.src.inference_engine.storage.job_store import JobStore
from InferenceEngine.src.inference_engine.workers.backpressure import AdaptiveProducer
from InferenceEngine.src.inference_engine.workers.dependency_tracker import DependencyTracker
from InferenceEngine.src.inference_engine.workers.local_pipeline import run_job_local


class StreamingPipelineTests(unittest.TestCase):
    def test_dependency_tracker_publishes_block_ready_after_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = _request()
            plan = build_job_plan("job", request, Path(tmp))[0]
            tracker = DependencyTracker.from_scene_plan(plan)
            first_block_deps = list(plan.blocks[0].dependency_tile_ids)
            ready = []
            for tile_id in first_block_deps[:-1]:
                ready.extend(tracker.mark_tile_done(tile_id))
            self.assertEqual(ready, [])
            ready.extend(tracker.mark_tile_done(first_block_deps[-1]))
            self.assertIn(plan.blocks[0].block_id, ready)

    def test_streaming_overlap_and_boundary_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = JobStore(root / "jobs")
            request = _request()
            state = store.create(request.model_dump(), job_id="job")
            settings = InferenceEngineSettings(job_root=root / "jobs", spool_root=root / "spool", artifact_root=root / "artifacts", logs_root=root / "logs")
            final = run_job_local(state["job_id"], store=store, settings=settings)
            self.assertEqual(final["status"], "success")
            metrics = final["metrics"]
            self.assertLess(metrics["first_block_vectorized_at"], metrics["last_tile_inferred_at"])
            self.assertNotIn("plan.json", final["artifacts"])
            accepted = Path(final["artifacts"]["accepted_geojson"])
            payload = json.loads(accepted.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["features"]), 1)
            self.assertEqual(payload["crs"]["properties"]["name"], "EPSG:3857")
            self.assertFalse((root / "jobs" / "job" / "scenes").exists())
            self.assertFalse((root / "spool" / "job").exists())
            self.assertTrue((root / "jobs" / "job" / "cleanup.json").exists())

    def test_idempotent_duplicate_local_run_keeps_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = JobStore(root / "jobs")
            request = _request()
            store.create(request.model_dump(), job_id="job")
            settings = InferenceEngineSettings(job_root=root / "jobs", spool_root=root / "spool", artifact_root=root / "artifacts", logs_root=root / "logs")
            first = run_job_local("job", store=store, settings=settings)
            second = run_job_local("job", store=store, settings=settings)
            self.assertEqual(first["status"], "success")
            self.assertEqual(second["status"], "success")
            self.assertEqual(first["artifacts"]["accepted_geojson"], second["artifacts"]["accepted_geojson"])

    def test_failed_local_run_cleans_intermediates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = JobStore(root / "jobs")
            request = JobRequest(
                experiment_id="bad",
                scenes=[SceneInput(scene_id="bad_scene", name="bad_scene", path=str(root / "missing.tif"), width=16, height=16)],
                preprocess={"patch_size": 8, "stride": 8},
                pseudolabel={"threshold": 0.5, "core_size_px": 16, "halo_px": 2},
            )
            store.create(request.model_dump(), job_id="bad_job")
            settings = InferenceEngineSettings(job_root=root / "jobs", spool_root=root / "spool", artifact_root=root / "artifacts", logs_root=root / "logs")
            final = run_job_local("bad_job", store=store, settings=settings)
            self.assertEqual(final["status"], "failed")
            self.assertFalse((root / "jobs" / "bad_job" / "scenes").exists())
            self.assertFalse((root / "spool" / "bad_job").exists())
            self.assertTrue((root / "jobs" / "bad_job" / "cleanup.json").exists())

    def test_adaptive_backpressure_pauses_and_resumes(self) -> None:
        producer = AdaptiveProducer(ResourceConfig(triton_batch_size=2, batches_ahead=1, max_preprocess_queue=2, max_spool_bytes=100))
        self.assertTrue(producer.should_pause(infer_ready=5, infer_unacked=0, spool_bytes=0))
        self.assertEqual(producer.state.preprocess_pauses_total, 1)
        self.assertTrue(producer.should_resume(infer_ready=0, infer_unacked=0, spool_bytes=0))
        self.assertEqual(producer.state.preprocess_resumes_total, 1)

    def test_manifest_bucket_key_preferred_over_images_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "inference_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "scenes": [
                            {
                                "entry": "scene.tif",
                                "name": "scene.tif",
                                "bucket": "real-bucket",
                                "key": "real/prefix/scene.tif",
                                "width": 32,
                                "height": 32,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            request = JobRequest(experiment_id="manifest", inference_manifest=str(manifest), images_uri="s3://wrong-bucket/wrong-prefix/")
            plan = build_job_plan("job", request, root / "jobs")[0]
            self.assertEqual(plan.source["uri"], "s3://real-bucket/real/prefix/scene.tif")

    def test_inline_scene_bucket_key_preferred_over_images_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = JobRequest(
                experiment_id="inline",
                images_uri="s3://wrong-bucket/wrong-prefix/",
                scenes=[SceneInput(name="scene.tif", bucket="real-bucket", key="real/prefix/scene.tif", width=32, height=32)],
            )
            plan = build_job_plan("job", request, Path(tmp) / "jobs")[0]
            self.assertEqual(plan.source["image_uri"], "s3://real-bucket/real/prefix/scene.tif")

    def test_images_uri_prefix_is_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            request = JobRequest(
                experiment_id="inline",
                images_uri="s3://mlsystems/images/",
                scenes=[SceneInput(name="scene.tif", key="images/kanopus/scene.tif", width=32, height=32)],
            )
            plan = build_job_plan("job", request, Path(tmp) / "jobs")[0]
            self.assertEqual(plan.source["image_uri"], "s3://mlsystems/images/kanopus/scene.tif")

    def test_manifest_scene_infers_raster_transform_and_crs(self) -> None:
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_origin
        except Exception as exc:  # pragma: no cover - optional raster stack in unit envs.
            self.skipTest(f"rasterio/numpy unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "scene.tif"
            transform = from_origin(1000.0, 2000.0, 2.0, 3.0)
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                width=8,
                height=6,
                count=1,
                dtype="uint8",
                crs="EPSG:3857",
                transform=transform,
            ) as ds:
                ds.write(np.zeros((1, 6, 8), dtype="uint8"))

            manifest = root / "inference_manifest.json"
            manifest.write_text(json.dumps({"scenes": [{"entry": "scene.tif", "name": "scene.tif", "path": str(image_path)}]}), encoding="utf-8")
            request = JobRequest(experiment_id="manifest", inference_manifest=str(manifest))
            plan = build_job_plan("job", request, root / "jobs")[0]

            self.assertEqual(plan.width, 8)
            self.assertEqual(plan.height, 6)
            self.assertEqual(plan.crs, "EPSG:3857")
            self.assertEqual(plan.transform, tuple(float(v) for v in transform[:6]))

    def test_manifest_scene_infers_epsg4326_crs(self) -> None:
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_origin
        except Exception as exc:  # pragma: no cover - optional raster stack in unit envs.
            self.skipTest(f"rasterio/numpy unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "scene_4326.tif"
            transform = from_origin(46.0, 45.0, 0.00002, 0.00002)
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                width=8,
                height=6,
                count=1,
                dtype="uint8",
                crs="EPSG:4326",
                transform=transform,
            ) as ds:
                ds.write(np.zeros((1, 6, 8), dtype="uint8"))

            manifest = root / "inference_manifest.json"
            manifest.write_text(json.dumps({"scenes": [{"entry": "scene_4326.tif", "name": "scene_4326.tif", "path": str(image_path)}]}), encoding="utf-8")
            request = JobRequest(experiment_id="manifest", inference_manifest=str(manifest))
            plan = build_job_plan("job", request, root / "jobs")[0]

            self.assertEqual(plan.crs, "EPSG:4326")
            self.assertEqual(plan.transform, tuple(float(v) for v in transform[:6]))

    def test_epsg4326_synthetic_scene_vectorizes_to_metric_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = JobStore(root / "jobs")
            request = JobRequest(
                experiment_id="epsg4326",
                scenes=[
                    SceneInput(
                        scene_id="scene_4326",
                        name="scene_4326",
                        width=64,
                        height=16,
                        crs="EPSG:4326",
                        transform=[0.00002, 0, 46.0, 0, -0.00002, 45.0],
                        probability_rects=[[6, 2, 58, 14, 1.0]],
                    )
                ],
                preprocess={"patch_size": 8, "stride": 8},
                pseudolabel={"threshold": 0.5, "core_size_px": 16, "halo_px": 2, "local_min_area": 0, "final_min_area": 0, "merge_epsilon": 0},
                resource={"triton_batch_size": 1, "max_preprocess_queue": 4, "max_spool_bytes": 10_000_000},
            )
            store.create(request.model_dump(), job_id="job")
            settings = InferenceEngineSettings(job_root=root / "jobs", spool_root=root / "spool", artifact_root=root / "artifacts", logs_root=root / "logs")
            final = run_job_local("job", store=store, settings=settings)

            self.assertEqual(final["status"], "success")
            accepted = Path(final["artifacts"]["accepted_geojson"])
            payload = json.loads(accepted.read_text(encoding="utf-8"))
            self.assertEqual(payload["crs"]["properties"]["name"], "EPSG:3857")
            self.assertGreater(len(payload["features"]), 0)

    def test_s3_uri_uses_local_minio_mount_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_path = root / "minio" / "mlsystems" / "images" / "scene.tif"
            object_path.parent.mkdir(parents=True)
            object_path.touch()

            with patch.dict(os.environ, {"INFERENCE_ENGINE_LOCAL_S3_ROOT": str(root / "minio")}):
                self.assertEqual(rasterio_path_for_uri("s3://mlsystems/images/scene.tif"), str(object_path))

    def test_s3_uri_ignores_minio_erasure_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            object_dir = root / "minio" / "mlsystems" / "images" / "scene.tif"
            object_dir.mkdir(parents=True)
            (object_dir / "xl.meta").write_text("minio metadata", encoding="utf-8")

            with patch.dict(os.environ, {"INFERENCE_ENGINE_LOCAL_S3_ROOT": str(root / "minio")}):
                self.assertEqual(rasterio_path_for_uri("s3://mlsystems/images/scene.tif"), "/vsis3/mlsystems/images/scene.tif")

    def test_manifest_s3_scene_infers_metadata_from_local_minio_mount(self) -> None:
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_origin
        except Exception as exc:  # pragma: no cover - optional raster stack in unit envs.
            self.skipTest(f"rasterio/numpy unavailable: {exc}")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "minio" / "mlsystems" / "images" / "scene.tif"
            image_path.parent.mkdir(parents=True)
            transform = from_origin(10.0, 20.0, 1.5, 2.5)
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                width=9,
                height=7,
                count=1,
                dtype="uint8",
                crs="EPSG:3857",
                transform=transform,
            ) as ds:
                ds.write(np.zeros((1, 7, 9), dtype="uint8"))

            manifest = root / "inference_manifest.json"
            manifest.write_text(json.dumps({"scenes": [{"entry": "scene.tif", "name": "scene.tif", "key": "images/scene.tif"}]}), encoding="utf-8")
            request = JobRequest(experiment_id="manifest", inference_manifest=str(manifest), images_uri="s3://mlsystems/")
            with patch.dict(os.environ, {"INFERENCE_ENGINE_LOCAL_S3_ROOT": str(root / "minio")}):
                plan = build_job_plan("job", request, root / "jobs")[0]

            self.assertEqual(plan.source["uri"], "s3://mlsystems/images/scene.tif")
            self.assertEqual(plan.width, 9)
            self.assertEqual(plan.height, 7)
            self.assertEqual(plan.crs, "EPSG:3857")
            self.assertEqual(plan.transform, tuple(float(v) for v in transform[:6]))


def _request() -> JobRequest:
    return JobRequest(
        experiment_id="unit",
        scenes=[
            SceneInput(
                scene_id="scene_a",
                name="scene_a",
                width=64,
                height=16,
                transform=[1, 0, 0, 0, -1, 16],
                probability_rects=[[6, 2, 58, 14, 1.0]],
            )
        ],
        preprocess={"patch_size": 8, "stride": 8},
        pseudolabel={"threshold": 0.5, "core_size_px": 16, "halo_px": 2, "local_min_area": 0, "final_min_area": 0, "merge_epsilon": 0},
        resource={"triton_batch_size": 1, "max_preprocess_queue": 4, "max_spool_bytes": 10_000_000},
    )


if __name__ == "__main__":
    unittest.main()
