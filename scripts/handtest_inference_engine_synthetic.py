from __future__ import annotations

import argparse
import os
import json
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic InferenceEngine job")
    parser.add_argument("--api", default="http://127.0.0.1:8095")
    parser.add_argument("--token", default=os.getenv("INFERENCE_ENGINE_API_TOKEN"))
    args = parser.parse_args()
    payload = {
        "experiment_id": "ie_synthetic",
        "scenes": [
            {
                "scene_id": "synthetic_a",
                "name": "synthetic_a",
                "width": 512,
                "height": 256,
                "transform": [1, 0, 0, 0, -1, 256],
                "probability_rects": [[120, 40, 390, 210, 1.0]],
            }
        ],
        "preprocess": {"patch_size": 128, "stride": 128},
        "pseudolabel": {"threshold": 0.5, "core_size_px": 128, "halo_px": 16, "final_min_area": 0, "merge_epsilon": 0},
        "resource": {"triton_batch_size": 2, "batches_ahead": 2},
    }
    created = post_json(args.api + "/api/v1/jobs", payload, token=args.token)
    job_id = created["job_id"]
    while True:
        status = get_json(args.api + f"/api/v1/jobs/{job_id}", token=args.token)
        print(json.dumps(status, ensure_ascii=False, indent=2))
        if status["status"] in {"success", "failed", "cancelled"}:
            break
        time.sleep(2)


def post_json(url: str, payload: dict, *, token: str | None = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, *, token: str | None = None) -> dict:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
