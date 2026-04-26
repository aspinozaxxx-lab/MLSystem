from __future__ import annotations
import argparse, json, time
from pathlib import Path
from .codex_summary import build_codex_summary
from .job_executor import run_once
from .job_queue import enqueue, list_queue
from .mlflow_adapter import check_mlflow, setup_deforest_experiment
from .pipeline_config import load_config, ensure_storage_layout
from .preprocess_inventory import build_preprocess_inventory, run_preprocess_forever
from .real_train import _find_layout_files, _list_s3_objects, _read_s3_text, build_scene_matching_report
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
    p_preprocess = sub.add_parser("preprocess-once")
    p_preprocess.add_argument("--dry-run", action="store_true")
    p_preprocess_forever = sub.add_parser("preprocess-forever")
    p_preprocess_forever.add_argument("--interval", type=float, default=300.0)
    p_forever = sub.add_parser("run-forever")
    p_forever.add_argument("--interval", type=float, default=10.0)
    p_enqueue = sub.add_parser("enqueue")
    p_enqueue.add_argument("job_yaml")
    p_mlflow_setup = sub.add_parser("mlflow-setup-experiment")
    p_mlflow_setup.add_argument("--experiment", default="mlsystem-deforest")
    p_match = sub.add_parser("match-scenes")
    p_match.add_argument("--class-name", default="deforest")
    p_match.add_argument("--images-uri", default=None)
    p_match.add_argument("--layout-uri", default=None)
    p_match.add_argument("--scenes-file", default="scenes.txt")
    p_match.add_argument("--annotation-file", default="auto")
    p_match.add_argument("--output", default=None)
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
    elif args.command == "mlflow-setup-experiment":
        print_json(setup_deforest_experiment(config, args.experiment))
    elif args.command == "match-scenes":
        images_uri = args.images_uri or config.s3_paths.images
        layout_uri = args.layout_uri or f"{config.s3_paths.layouts.rstrip('/')}/{args.class_name}/"
        images = _list_s3_objects(config, images_uri, suffixes=(".tif", ".tiff"))
        annotation_uri, scenes_uri = _find_layout_files(config, layout_uri, args.scenes_file, args.annotation_file)
        entries = [
            line.strip()
            for line in _read_s3_text(config, scenes_uri).splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        report = {
            **build_scene_matching_report(entries, images),
            "images_uri": images_uri,
            "layout_uri": layout_uri,
            "annotation_uri": annotation_uri,
            "scenes_uri": scenes_uri,
        }
        output = Path(args.output) if args.output else config.system_root / f"scene_matching_{args.class_name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print_json({**report, "output": str(output)})
    elif args.command == "run-once":
        print_json(run_once(config))
    elif args.command == "run-forever":
        while True:
            print_json(run_once(config))
            time.sleep(max(args.interval, 1.0))
    elif args.command == "resources":
        print_json(collect_status(config, include_services=True))
    elif args.command == "preprocess-once":
        print_json(build_preprocess_inventory(config, dry_run=args.dry_run))
    elif args.command == "preprocess-forever":
        run_preprocess_forever(config, interval=args.interval)

if __name__ == "__main__":
    main()
