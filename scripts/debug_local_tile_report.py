from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlsystem.src.data.local_tile_report import LocalTileReportConfig, generate_local_tile_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Build image-only local tile sampling HTML reports for GeoTIFF scenes.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--positive-stride-factor", type=float, default=0.5)
    parser.add_argument("--hard-negative-stride-factor", type=float, default=0.5)
    parser.add_argument("--negative-stride-factor", type=float, default=1.0)
    parser.add_argument("--positive-repeat-factor", type=int, default=4)
    parser.add_argument("--hard-negative-repeat-factor", type=int, default=2)
    parser.add_argument("--negative-repeat-factor", type=int, default=1)
    parser.add_argument("--max-empty-tile-share", type=float, default=0.35)
    parser.add_argument("--max-overview-size", type=int, default=1600)
    parser.add_argument("--max-tile-examples", type=int, default=24)
    parser.add_argument("--augment-examples-per-tile", type=int, default=3)
    parser.add_argument(
        "--augmentation-mode",
        choices=("all", "individual", "production-groups", "random-training"),
        default="all",
        help="Which debug augmentation operations to render for the matrix.",
    )
    parser.add_argument(
        "--max-augmentation-tiles",
        type=int,
        default=8,
        help="How many selected tile previews receive the full augmentation matrix.",
    )
    parser.add_argument(
        "--augmentations",
        default=None,
        help="Optional comma-separated operation names, overriding --augmentation-mode.",
    )
    parser.add_argument("--augmentation-seed", type=int, default=42)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = LocalTileReportConfig(
        images_dir=Path(args.images_dir),
        tile_size=max(1, int(args.tile_size)),
        stride=max(1, int(args.stride)),
        positive_stride_factor=max(0.000001, float(args.positive_stride_factor)),
        hard_negative_stride_factor=max(0.000001, float(args.hard_negative_stride_factor)),
        negative_stride_factor=max(0.000001, float(args.negative_stride_factor)),
        positive_repeat_factor=max(1, int(args.positive_repeat_factor)),
        hard_negative_repeat_factor=max(1, int(args.hard_negative_repeat_factor)),
        negative_repeat_factor=max(1, int(args.negative_repeat_factor)),
        max_empty_tile_share=args.max_empty_tile_share,
        max_overview_size=max(1, int(args.max_overview_size)),
        max_tile_examples=max(0, int(args.max_tile_examples)),
        augment_examples_per_tile=max(0, int(args.augment_examples_per_tile)),
        max_augmentation_tiles=max(0, int(args.max_augmentation_tiles)),
        augmentation_mode=str(args.augmentation_mode),
        augmentations=args.augmentations,
        augmentation_seed=int(args.augmentation_seed),
        recursive=bool(args.recursive),
        seed=int(args.seed),
    )
    result = generate_local_tile_reports(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
