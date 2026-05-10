from __future__ import annotations

import argparse
import asyncio

from ..api.app import app as fastapi_app
from ..config.settings import InferenceEngineSettings
from ..storage.job_store import JobStore
from .local_pipeline import run_job_local
from .rabbit_pipeline import RabbitPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="InferenceEngine worker")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("api")
    for command in ("submit", "planner", "preprocess", "triton", "aggregator", "block", "merger", "finalizer", "all"):
        sub.add_parser(command)
    worker = sub.add_parser("worker")
    worker.add_argument("--role", default="planner", choices=["submit", "planner", "preprocess", "triton", "aggregator", "block", "merger", "finalizer", "all"])
    run_job = sub.add_parser("run-job")
    run_job.add_argument("job_id")
    args = parser.parse_args()
    settings = InferenceEngineSettings.from_env()
    settings.ensure_dirs()
    if args.command == "api":
        import uvicorn

        uvicorn.run(fastapi_app, host="0.0.0.0", port=8095)
    elif args.command == "worker":
        asyncio.run(RabbitPipeline(settings).run_role(args.role))
    elif args.command in {"submit", "planner", "preprocess", "triton", "aggregator", "block", "merger", "finalizer", "all"}:
        asyncio.run(RabbitPipeline(settings).run_role(args.command))
    elif args.command == "run-job":
        run_job_local(args.job_id, store=JobStore(settings.job_root), settings=settings)


if __name__ == "__main__":
    main()
