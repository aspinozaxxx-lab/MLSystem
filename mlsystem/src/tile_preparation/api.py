from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import SceneInput
from .contracts import TileDatasetRequest


def build_datasets(request: TileDatasetRequest) -> Any:
    from ._builder import TilePreparationBuilder

    return TilePreparationBuilder.build_datasets(
        train_scenes=[SceneInput(Path(scene.path), scene.scene_id) for scene in request.train_scenes],
        val_scenes=[SceneInput(Path(scene.path), scene.scene_id) for scene in request.val_scenes],
        annotation_path=Path(request.annotation_path),
        tile_size=request.tile_size,
        stride=request.stride,
        augmentation_level=request.augmentation_level,
        mosaic_enabled=request.mosaic_enabled,
        normalization_mode=request.normalization_mode,
    )


def train_dataloader(bundle: Any, *, batch_size: int, **kwargs: Any) -> Any:
    from ._builder import TilePreparationBuilder

    return TilePreparationBuilder.train_dataloader(bundle, batch_size=batch_size, **kwargs)


def val_dataloader(bundle: Any, *, batch_size: int, **kwargs: Any) -> Any:
    from ._builder import TilePreparationBuilder

    return TilePreparationBuilder.val_dataloader(bundle, batch_size=batch_size, **kwargs)
