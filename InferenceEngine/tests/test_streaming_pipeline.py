from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from InferenceEngine.src.inference_engine.api.schemas import JobRequest, ResourceConfig, SceneInput
from InferenceEngine.src.inference_engine.config.settings import InferenceEngineSettings
from InferenceEngine.src.inference_engine.planning.planner import build_job_plan
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
            accepted = Path(final["artifacts"]["accepted_geojson"])
            payload = json.loads(accepted.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["features"]), 1)

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

    def test_adaptive_backpressure_pauses_and_resumes(self) -> None:
        producer = AdaptiveProducer(ResourceConfig(triton_batch_size=2, batches_ahead=1, max_preprocess_queue=2, max_spool_bytes=100))
        self.assertTrue(producer.should_pause(infer_ready=3, infer_unacked=0, spool_bytes=0))
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
