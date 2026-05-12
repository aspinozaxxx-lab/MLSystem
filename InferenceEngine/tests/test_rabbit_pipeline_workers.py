from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from InferenceEngine.src.inference_engine.api.schemas import JobRequest, SceneInput
from InferenceEngine.src.inference_engine.config.settings import InferenceEngineSettings
from InferenceEngine.src.inference_engine.queues.messages import QueueMessage, make_message
from InferenceEngine.src.inference_engine.storage.job_store import JobStore
from InferenceEngine.src.inference_engine.workers.rabbit_pipeline import ROLE_QUEUES, RabbitPipeline


class FakeClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, QueueMessage]] = []

    async def publish(self, queue: str, message: QueueMessage) -> None:
        self.published.append((queue, message))

    def pop(self, queue: str) -> QueueMessage:
        for idx, (name, message) in enumerate(self.published):
            if name == queue:
                del self.published[idx]
                return message
        raise AssertionError(f"No message for queue {queue}; published={[(name, msg.stage) for name, msg in self.published]}")

    def count(self, queue: str) -> int:
        return sum(1 for name, _message in self.published if name == queue)


class RabbitPipelineWorkerTests(unittest.TestCase):
    def test_role_queue_contracts_exist(self) -> None:
        self.assertEqual(ROLE_QUEUES["preprocess"], "ie.tile.preprocess")
        self.assertEqual(ROLE_QUEUES["triton"], "ie.tile.infer")
        self.assertEqual(ROLE_QUEUES["finalizer"], "ie.job.finalize")

    def test_stage_by_stage_synthetic_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, fake = _pipeline(Path(tmp))
            job_id = "job"
            pipeline.store.create(_request().model_dump(), job_id=job_id)

            asyncio.run(pipeline.handle_submit(make_message(job_id=job_id, stage="jobs.submit")))
            scene_plan_msg = fake.pop("ie.scene.plan")
            asyncio.run(pipeline.handle_scene_plan(scene_plan_msg))
            self.assertGreater(fake.count("ie.tile.preprocess"), 0)

            # Process all tile messages emitted by bounded producer.
            while True:
                try:
                    preprocess = fake.pop("ie.tile.preprocess")
                except AssertionError:
                    break
                asyncio.run(pipeline.handle_tile_preprocess(preprocess))
                infer = fake.pop("ie.tile.infer")
                asyncio.run(pipeline.handle_tile_infer_batch([infer]))
                done = fake.pop("ie.tile.done")
                asyncio.run(pipeline.handle_tile_done(done))
                while True:
                    try:
                        ready = fake.pop("ie.block.ready")
                    except AssertionError:
                        break
                    asyncio.run(pipeline.handle_block_ready(ready))
                    vectorize = fake.pop("ie.block.vectorize")
                    asyncio.run(pipeline.handle_block_vectorize(vectorize))
                    block_done = fake.pop("ie.block.done")
                    asyncio.run(pipeline.handle_block_done(block_done))

            scene_merge = fake.pop("ie.scene.merge")
            asyncio.run(pipeline.handle_scene_merge(scene_merge))
            scene_dir = Path(tmp) / "jobs" / job_id / "scenes" / "scene_a"
            self.assertTrue((scene_dir / "scene_a.accepted.geojson").exists())
            self.assertTrue((scene_dir / "plan.json").exists())
            self.assertFalse((scene_dir / "tiles").exists())
            self.assertFalse((scene_dir / "blocks").exists())
            finalize = fake.pop("ie.job.finalize")
            asyncio.run(pipeline.handle_job_finalize(finalize))
            state = pipeline.store.read(job_id)
            self.assertEqual(state["status"], "success")
            self.assertLess(state["metrics"]["first_block_vectorized_at"], state["metrics"]["last_tile_inferred_at"])
            self.assertIn("accepted_geojson", state["artifacts"])
            self.assertNotIn("scene_a.plan.json", state["artifacts"])
            self.assertFalse((Path(tmp) / "jobs" / job_id / "scenes").exists())
            self.assertFalse((Path(tmp) / "spool" / job_id).exists())
            self.assertTrue((Path(tmp) / "jobs" / job_id / "cleanup.json").exists())

    def test_duplicate_tile_done_and_block_done_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pipeline, fake = _pipeline(Path(tmp))
            job_id = "job"
            pipeline.store.create(_request().model_dump(), job_id=job_id)
            asyncio.run(pipeline.handle_submit(make_message(job_id=job_id, stage="jobs.submit")))
            scene_plan_msg = fake.pop("ie.scene.plan")
            asyncio.run(pipeline.handle_scene_plan(scene_plan_msg))
            preprocess = fake.pop("ie.tile.preprocess")
            asyncio.run(pipeline.handle_tile_preprocess(preprocess))
            infer = fake.pop("ie.tile.infer")
            asyncio.run(pipeline.handle_tile_infer_batch([infer]))
            done = fake.pop("ie.tile.done")
            asyncio.run(pipeline.handle_tile_done(done))
            first_metrics = dict(pipeline.store.read(job_id)["metrics"])
            asyncio.run(pipeline.handle_tile_done(done))
            second_metrics = dict(pipeline.store.read(job_id)["metrics"])
            self.assertEqual(first_metrics["tiles_done"], second_metrics["tiles_done"])


def _pipeline(root: Path) -> tuple[RabbitPipeline, FakeClient]:
    settings = InferenceEngineSettings(job_root=root / "jobs", spool_root=root / "spool", artifact_root=root / "artifacts", logs_root=root / "logs")
    settings.ensure_dirs()
    pipeline = RabbitPipeline(settings)
    pipeline.store = JobStore(settings.job_root)
    fake = FakeClient()
    pipeline.client = fake  # type: ignore[assignment]
    return pipeline, fake


def _request() -> JobRequest:
    return JobRequest(
        experiment_id="rabbit_unit",
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
        resource={"triton_batch_size": 1, "batches_ahead": 2, "max_preprocess_queue": 2, "max_spool_bytes": 10_000_000},
    )


if __name__ == "__main__":
    unittest.main()
