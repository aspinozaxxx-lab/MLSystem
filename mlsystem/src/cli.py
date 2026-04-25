from __future__ import annotations
import argparse, json, time
from pathlib import Path
from .codex_summary import build_codex_summary
from .job_executor import run_once
from .job_queue import enqueue, list_queue
from .mlflow_adapter import check_mlflow
from .pipeline_config import load_config, ensure_storage_layout
from .resource_manager import collect_status
from .s3_adapter import build_s3_layout_status, check_s3

def print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

def main() -> None:
    parser = argparse.ArgumentParser(description="MLSystem server MVP CLI")
    parser.add_argument("--config", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["status", "check-mlflow", "check-s3", "s3-layout", "queue-list", "run-once", "resources"]:
        sub.add_parser(name)
    p_forever = sub.add_parser("run-forever")
    p_forever.add_argument("--interval", type=float, default=10.0)
    p_enqueue = sub.add_parser("enqueue")
    p_enqueue.add_argument("job_yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    ensure_storage_layout(config)
    if args.command == "status":
        resources = collect_status(config, include_services=True)
        print_json({"resources": resources, "codex_summary": build_codex_summary(config, resources)})
    elif args.command == "check-mlflow":
        print_json(check_mlflow(config))
    elif args.command == "check-s3":
        print_json(check_s3(config, write_test=False))
    elif args.command == "s3-layout":
        print_json(build_s3_layout_status(config))
    elif args.command == "queue-list":
        print_json(list_queue(config))
    elif args.command == "enqueue":
        print_json(enqueue(config, Path(args.job_yaml)))
    elif args.command == "run-once":
        print_json(run_once(config))
    elif args.command == "run-forever":
        while True:
            print_json(run_once(config))
            time.sleep(max(args.interval, 1.0))
    elif args.command == "resources":
        print_json(collect_status(config, include_services=True))

if __name__ == "__main__":
    main()
