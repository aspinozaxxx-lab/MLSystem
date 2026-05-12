from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ..api.app import app as fastapi_app
from ..config.settings import InferenceEngineSettings
from ..storage.job_store import JobStore
from .local_pipeline import run_job_local
from .rabbit_pipeline import RabbitPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="InferenceEngine worker")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("api")
    for command in ("submit", "planner", "preprocess", "fused", "triton", "aggregator", "block", "merger", "finalizer", "all"):
        sub.add_parser(command)
    worker = sub.add_parser("worker")
    worker.add_argument("--role", default="planner", choices=["submit", "planner", "preprocess", "fused", "triton", "aggregator", "block", "merger", "finalizer", "all"])
    worker.add_argument("--concurrency", type=int, default=None)
    run_job = sub.add_parser("run-job")
    run_job.add_argument("job_id")
    args = parser.parse_args()
    settings = InferenceEngineSettings.from_env()
    settings.ensure_dirs()
    if args.command == "api":
        import uvicorn

        uvicorn.run(fastapi_app, host="0.0.0.0", port=8095)
    elif args.command == "worker":
        concurrency = args.concurrency or _role_concurrency(args.role, settings.default_worker_concurrency)
        asyncio.run(_run_role(settings, args.role, concurrency))
    elif args.command in {"submit", "planner", "preprocess", "fused", "triton", "aggregator", "block", "merger", "finalizer", "all"}:
        asyncio.run(_run_role(settings, args.command, _role_concurrency(args.command, settings.default_worker_concurrency)))
    elif args.command == "run-job":
        run_job_local(args.job_id, store=JobStore(settings.job_root), settings=settings)

async def _run_role(settings: InferenceEngineSettings, role: str, concurrency: int) -> None:
    concurrency = max(1, int(concurrency or 1))
    if concurrency == 1:
        await RabbitPipeline(settings).run_role(role)
        return
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix=f"ie-{role}") as executor:
        await asyncio.gather(
            *(
                loop.run_in_executor(executor, _run_role_sync, settings, role)
                for _ in range(concurrency)
            )
        )


def _run_role_sync(settings: InferenceEngineSettings, role: str) -> None:
    asyncio.run(RabbitPipeline(settings).run_role(role))


def _role_concurrency(role: str, default: int) -> int:
    import os

    normalized = role.replace("-", "_").upper()
    value = os.getenv(f"INFERENCE_ENGINE_{normalized}_CONCURRENCY")
    if value is None:
        value = os.getenv("INFERENCE_ENGINE_WORKER_CONCURRENCY")
    return int(value or default or 1)


if __name__ == "__main__":
    main()
