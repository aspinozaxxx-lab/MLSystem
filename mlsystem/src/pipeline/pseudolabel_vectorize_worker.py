from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..storage.local_io import write_json
from .pseudolabel_pipeline import _vectorize_scene_result_row, vectorization_result_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Vectorize one pseudolabel scene result.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8-sig"))
    result = _vectorize_scene_result_row(
        (
            payload["row"],
            float(payload["threshold"]),
            float(payload.get("min_area_prefilter") or 0),
        )
    )
    write_json(Path(args.output), vectorization_result_to_dict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
