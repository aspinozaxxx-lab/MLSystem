from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlsystem.src.tile_preparation import TilePreparationConfig
from mlsystem.src.tile_preparation.report import generate_annotated_tile_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build annotated tile sampling report for one GeoTIFF and one GeoJSON.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--positive-stride-factor", type=float, default=0.5)
    parser.add_argument("--hard-negative-stride-factor", type=float, default=0.5)
    parser.add_argument("--negative-stride-factor", type=float, default=1.0)
    parser.add_argument("--positive-repeat-factor", type=int, default=4)
    parser.add_argument("--hard-negative-repeat-factor", type=int, default=2)
    parser.add_argument("--negative-repeat-factor", type=int, default=1)
    parser.add_argument("--max-empty-tile-share", type=float, default=0.35)
    parser.add_argument("--min-positive-pixels", type=int, default=1)
    parser.add_argument("--hard-negative-context-px", type=int, default=None)
    parser.add_argument("--annotation-crs", default="auto")
    parser.add_argument("--allow-inferred-annotation-crs", action="store_true")
    parser.add_argument("--max-overview-size", type=int, default=1600)
    parser.add_argument("--max-tile-examples", type=int, default=32)
    parser.add_argument("--max-augmentation-tiles", type=int, default=8)
    parser.add_argument(
        "--augmentation-mode",
        choices=("all", "individual", "production-groups", "random-training"),
        default="all",
    )
    parser.add_argument("--augmentation-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TilePreparationConfig(
        tile_size=args.tile_size,
        stride=args.stride,
        positive_stride_factor=args.positive_stride_factor,
        hard_negative_stride_factor=args.hard_negative_stride_factor,
        negative_stride_factor=args.negative_stride_factor,
        positive_repeat_factor=args.positive_repeat_factor,
        hard_negative_repeat_factor=args.hard_negative_repeat_factor,
        negative_repeat_factor=args.negative_repeat_factor,
        max_empty_tile_share=args.max_empty_tile_share,
        min_positive_pixels=args.min_positive_pixels,
        hard_negative_context_px=args.hard_negative_context_px,
        seed=args.seed,
        augmentation_mode=args.augmentation_mode,
        augmentation_seed=args.augmentation_seed,
        augmentations={
            "flips": True,
            "rot90": True,
            "brightness_contrast": True,
            "color_jitter": True,
            "gamma": True,
            "noise": True,
            "blur": True,
            "cutout": True,
            "coarse_dropout": True,
        },
    )
    result = generate_annotated_tile_report(
        args.input_dir,
        config=config,
        annotation_crs=args.annotation_crs,
        allow_inferred_annotation_crs=bool(args.allow_inferred_annotation_crs),
        max_overview_size=args.max_overview_size,
        max_tile_examples=args.max_tile_examples,
        max_augmentation_tiles=args.max_augmentation_tiles,
        augmentation_mode=args.augmentation_mode,
        augmentation_seed=args.augmentation_seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
