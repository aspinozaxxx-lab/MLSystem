from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import TrainPipelineRunStore, TrainPipelineRunner, load_trace_file


def main() -> int:
    parser = argparse.ArgumentParser(description="MLSystem pipeline runner CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--trace", required=True)
    run_parser.add_argument("--run-root", type=Path, default=None)

    status_parser = sub.add_parser("status")
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--run-root", type=Path, default=None)

    log_parser = sub.add_parser("log")
    log_parser.add_argument("--run-id", required=True)
    log_parser.add_argument("--run-root", type=Path, default=None)
    log_parser.add_argument("--tail", type=int, default=20000)

    args = parser.parse_args()
    store = TrainPipelineRunStore(args.run_root)
    runner = TrainPipelineRunner(store)
    if args.command == "run":
        run = runner.start_run(load_trace_file(args.trace), source="cli")
        print(json.dumps(run.model_dump(), ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "status":
        run = runner.refresh_run(args.run_id)
        print(json.dumps(run.model_dump(), ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0
    if args.command == "log":
        print(store.tail_log(args.run_id, max_chars=args.tail), end="")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
