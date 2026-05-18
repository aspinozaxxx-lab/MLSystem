from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from mlsystem.src.tile_preparation import SceneInput
from mlsystem.src.tile_preparation._builder import bundle_to_jsonable
from mlsystem.src.tile_preparation.api import build_datasets, train_dataloader, val_dataloader
from mlsystem.src.tile_preparation.contracts import SceneInputContract, TileDatasetRequest


def _read_scene_list(images_root: Path, scene_list_path: Path) -> list[SceneInput]:
    root = Path(images_root)
    rows = Path(scene_list_path).read_text(encoding="utf-8").splitlines()
    scenes: list[SceneInput] = []
    for line in rows:
        value = line.strip().lstrip("\ufeff")
        if not value or value.startswith("#"):
            continue
        path = Path(value)
        image_path = path if path.is_absolute() else root / path
        if not image_path.is_file() or image_path.suffix.lower() not in {".tif", ".tiff"}:
            matches = sorted(root.glob(f"{value}.tif")) + sorted(root.glob(f"{value}.tiff"))
            if matches:
                image_path = matches[0]
        if not image_path.is_file() or image_path.suffix.lower() not in {".tif", ".tiff"}:
            raise FileNotFoundError(f"Scene list entry does not resolve to an image: {value}")
        scenes.append(SceneInput(image_path=image_path, scene_id=image_path.stem))
    if not scenes:
        raise ValueError(f"Scene list is empty: {scene_list_path}")
    return scenes


def run_train_smoke(
    *,
    images_root: Path,
    train_scene_list: Path | None = None,
    val_scene_list: Path | None = None,
    scene_list: Path | None = None,
    debug_split_first_n_train: int | None = None,
    annotation: Path,
    tile_size: int,
    stride: int,
    augmentation_level: int,
    batch_size: int,
    max_train_batches: int,
    max_val_batches: int,
    device: str,
    output_json: Path,
) -> dict[str, Any]:
    if train_scene_list and val_scene_list:
        train_scenes = _read_scene_list(images_root, train_scene_list)
        val_scenes = _read_scene_list(images_root, val_scene_list)
        split_source = "explicit_train_val_scene_lists"
    elif scene_list:
        scenes = _read_scene_list(images_root, scene_list)
        if len(scenes) == 1:
            train_scenes = scenes
            val_scenes = scenes
        else:
            train_count = int(debug_split_first_n_train or max(1, len(scenes) - 1))
            train_count = max(1, min(len(scenes) - 1, train_count))
            train_scenes = scenes[:train_count]
            val_scenes = scenes[train_count:]
        split_source = "debug_script_local_split"
    else:
        raise ValueError("provide either --train-scene-list and --val-scene-list, or debug-only --scene-list")

    bundle = build_datasets(
        TileDatasetRequest(
            train_scenes=[SceneInputContract(scene.image_path, scene.scene_id) for scene in train_scenes],
            val_scenes=[SceneInputContract(scene.image_path, scene.scene_id) for scene in val_scenes],
            annotation_path=annotation,
            tile_size=tile_size,
            stride=stride,
            augmentation_level=augmentation_level,
        )
    )
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    train_batches: list[dict[str, Any]] = []
    val_batches: list[dict[str, Any]] = []
    losses: list[float] = []
    device_obj = torch.device(device)
    try:
        for batch_index, (_indices, x, y) in enumerate(train_dataloader(bundle, batch_size=batch_size)):
            if batch_index >= max_train_batches:
                break
            x = x.to(device_obj).float()
            y = y.to(device_obj).float()
            if model is None:
                model = torch.nn.Conv2d(int(x.shape[1]), 1, kernel_size=1).to(device_obj)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            assert optimizer is not None
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            y_unique = sorted(float(item) for item in torch.unique(y.detach().cpu()).tolist())
            train_batches.append({"batch": batch_index, "x_shape": list(x.shape), "y_shape": list(y.shape), "y_values": y_unique, "loss": float(loss.detach().cpu())})
            losses.append(float(loss.detach().cpu()))

        for batch_index, (_indices, x, y) in enumerate(val_dataloader(bundle, batch_size=batch_size)):
            if batch_index >= max_val_batches:
                break
            y_unique = sorted(float(item) for item in torch.unique(y).tolist())
            val_batches.append({"batch": batch_index, "x_shape": list(x.shape), "y_shape": list(y.shape), "y_values": y_unique})
    finally:
        bundle.train_dataset.close()
        bundle.val_dataset.close()

    result = {
        "status": "ok",
        "entrypoint": "mlsystem.src.tile_preparation.api",
        "uses_fastapi": False,
        "split_source": split_source,
        "train_scenes": [scene.resolved_scene_id() for scene in train_scenes],
        "val_scenes": [scene.resolved_scene_id() for scene in val_scenes],
        "bundle": bundle_to_jsonable(bundle),
        "train_batches": train_batches,
        "val_batches": val_batches,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "backward_optimizer_step": bool(train_batches),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny local train smoke test through tile_preparation api.")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--train-scene-list")
    parser.add_argument("--val-scene-list")
    parser.add_argument("--scene-list", help="Debug-only fallback: split this local list inside the script, not in tile_preparation.")
    parser.add_argument("--debug-split-first-n-train", type=int)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--augmentation-level", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="outputs/debug_virtual_tile_sampling/train_smoke_result.json")
    args = parser.parse_args()

    result = run_train_smoke(
        images_root=Path(args.images_root),
        train_scene_list=Path(args.train_scene_list) if args.train_scene_list else None,
        val_scene_list=Path(args.val_scene_list) if args.val_scene_list else None,
        scene_list=Path(args.scene_list) if args.scene_list else None,
        debug_split_first_n_train=args.debug_split_first_n_train,
        annotation=Path(args.annotation),
        tile_size=args.tile_size,
        stride=args.stride,
        augmentation_level=args.augmentation_level,
        batch_size=args.batch_size,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        device=args.device,
        output_json=Path(args.output_json),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
