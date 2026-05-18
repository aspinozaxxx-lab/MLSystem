from __future__ import annotations

from .contracts import ModelSpec, TrainProgressSink, TrainRequest, TrainResult


def train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult:
    from ._trainer import run_training

    return run_training(request, progress_sink=progress_sink)


def list_supported_models() -> list[ModelSpec]:
    return [
        ModelSpec("tiny_unet_4ch", "unet", input_channels=4),
        ModelSpec("tiny_unet", "unet"),
        ModelSpec("unet_resnet18", "unet", requires_optional_dependency="segmentation_models_pytorch"),
        ModelSpec("unet_resnet34", "unet", requires_optional_dependency="segmentation_models_pytorch"),
        ModelSpec("unet_resnet50", "unet", requires_optional_dependency="segmentation_models_pytorch"),
        ModelSpec("segformer_b0", "segformer", requires_optional_dependency="segmentation_models_pytorch"),
        ModelSpec("segformer_b1", "segformer", requires_optional_dependency="segmentation_models_pytorch"),
        ModelSpec("segformer_b2", "segformer", requires_optional_dependency="segmentation_models_pytorch"),
        ModelSpec("segformer_b3", "segformer", requires_optional_dependency="segmentation_models_pytorch"),
        ModelSpec("deeplabv3plus_resnet34", "deeplabv3plus", requires_optional_dependency="segmentation_models_pytorch"),
        ModelSpec("deeplabv3plus_resnet50", "deeplabv3plus", requires_optional_dependency="segmentation_models_pytorch"),
    ]
