from __future__ import annotations

import argparse
import json
import pickle
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
    output_path = Path(args.output)
    if output_path.suffix == ".pkl":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(vectorization_result_to_dict(result), handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        write_json(output_path, vectorization_result_to_dict(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
