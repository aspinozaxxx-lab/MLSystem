from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlsystem.src.pipeline_runner.config import load_trace_file
from mlsystem.src.pipeline_runner.run_store import PipelineRunStore
from mlsystem.src.pipeline_runner.runner import PipelineRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one MLSystem pipeline trace through the built-in runner.")
    parser.add_argument("--trace", required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=2.0)
    args = parser.parse_args()

    store = PipelineRunStore(args.run_root)
    runner = PipelineRunner(store)
    run = runner.start_run(load_trace_file(args.trace), source="debug-script")
    print(json.dumps(run.model_dump(), ensure_ascii=False, indent=2, sort_keys=True, default=str))
    if not args.wait:
        return 0
    while True:
        current = runner.refresh_run(run.run_id)
        print(f"{current.run_id}: state={current.state} progress={current.progress_percent}% stage={current.current_stage}", flush=True)
        if current.state in {"succeeded", "failed", "cancelled"}:
            return 0 if current.state == "succeeded" else 1
        try:
            time.sleep(max(0.1, args.poll_sec))
        except KeyboardInterrupt:
            current = runner.refresh_run(run.run_id)
            if current.state in {"succeeded", "failed", "cancelled"}:
                return 0 if current.state == "succeeded" else 1
            raise


if __name__ == "__main__":
    raise SystemExit(main())
