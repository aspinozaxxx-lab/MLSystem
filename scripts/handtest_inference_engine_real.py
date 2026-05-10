from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InferenceEngine on a real inference manifest")
    parser.add_argument("--api", default="http://127.0.0.1:8095")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment-id", default="ie_real_2")
    parser.add_argument("--max-scenes", type=int, default=2)
    args = parser.parse_args()
    payload = {
        "experiment_id": args.experiment_id,
        "inference_manifest": str(Path(args.manifest).resolve()),
        "max_scenes": args.max_scenes,
        "model": {
            "mlflow_run_id": "a7838f91528a47e1931b685c2ea06686",
            "architecture": "segformer_b2",
            "triton_model_name": "segformer_b2",
        },
        "preprocess": {"patch_size": 1024, "stride": 768, "input_bands": [1, 2, 3, 4]},
        "pseudolabel": {"threshold": 0.5, "core_size_px": 4096, "halo_px": 512, "final_min_area": 0, "merge_epsilon": 1.0},
        "resource": {"triton_batch_size": 8, "batches_ahead": 4, "max_scenes_inflight": 1},
    }
    created = post_json(args.api + "/api/v1/jobs", payload)
    job_id = created["job_id"]
    while True:
        status = get_json(args.api + f"/api/v1/jobs/{job_id}")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        if status["status"] in {"success", "failed", "cancelled"}:
            break
        time.sleep(10)


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
