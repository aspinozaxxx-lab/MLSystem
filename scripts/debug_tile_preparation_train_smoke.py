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

from mlsystem.src.tile_preparation.facade import TilePreparationFacade, bundle_to_jsonable


def run_train_smoke(
    *,
    images_root: Path,
    scene_list: Path,
    annotation: Path,
    tile_size: int,
    augmentation_level: int,
    mosaic_mode: str,
    batch_size: int,
    max_train_batches: int,
    max_val_batches: int,
    device: str,
    output_json: Path,
) -> dict[str, Any]:
    bundle = TilePreparationFacade.from_scene_list(
        images_root=images_root,
        scene_list_path=scene_list,
        annotation_path=annotation,
        tile_size=tile_size,
        augmentation_level=augmentation_level,
        mosaic_mode=mosaic_mode,
        seed=42,
    )
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    train_batches: list[dict[str, Any]] = []
    val_batches: list[dict[str, Any]] = []
    losses: list[float] = []
    device_obj = torch.device(device)
    try:
        for batch_index, (_indices, x, y) in enumerate(TilePreparationFacade.train_dataloader(bundle, batch_size=batch_size)):
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

        for batch_index, (_indices, x, y) in enumerate(TilePreparationFacade.val_dataloader(bundle, batch_size=batch_size)):
            if batch_index >= max_val_batches:
                break
            y_unique = sorted(float(item) for item in torch.unique(y).tolist())
            val_batches.append({"batch": batch_index, "x_shape": list(x.shape), "y_shape": list(y.shape), "y_values": y_unique})
    finally:
        bundle.train_dataset.close()
        bundle.val_dataset.close()

    result = {
        "status": "ok",
        "entrypoint": "TilePreparationFacade",
        "uses_fastapi": False,
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
    parser = argparse.ArgumentParser(description="Run a tiny local train smoke test through tile_preparation facade.")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--augmentation-level", type=int, default=1)
    parser.add_argument("--mosaic-mode", choices=("off", "auto", "force"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, default=2)
    parser.add_argument("--max-val-batches", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="outputs/debug_virtual_tile_sampling/train_smoke_result.json")
    args = parser.parse_args()

    result = run_train_smoke(
        images_root=Path(args.images_root),
        scene_list=Path(args.scene_list),
        annotation=Path(args.annotation),
        tile_size=args.tile_size,
        augmentation_level=args.augmentation_level,
        mosaic_mode=args.mosaic_mode,
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
