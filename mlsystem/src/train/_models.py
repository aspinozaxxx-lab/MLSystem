from __future__ import annotations

import torch

from .contracts import ModelSpec


class TinyUNet(torch.nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 8) -> None:
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base_channels, 3, padding=1),
            torch.nn.BatchNorm2d(base_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels, base_channels, 3, padding=1),
            torch.nn.BatchNorm2d(base_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.pool = torch.nn.MaxPool2d(2)
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            torch.nn.BatchNorm2d(base_channels * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            torch.nn.BatchNorm2d(base_channels * 2),
            torch.nn.ReLU(inplace=True),
        )
        self.up = torch.nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec = torch.nn.Sequential(
            torch.nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            torch.nn.BatchNorm2d(base_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base_channels, base_channels, 3, padding=1),
            torch.nn.BatchNorm2d(base_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.out = torch.nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        u = self.up(e2)
        if u.shape[-2:] != e1.shape[-2:]:
            u = torch.nn.functional.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(self.dec(torch.cat([e1, u], dim=1)))


def list_supported_model_specs() -> list[ModelSpec]:
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


def build_model(model_name: str, in_channels: int, out_channels: int, base_channels: int) -> torch.nn.Module:
    normalized = (model_name or "tiny_unet_4ch").strip().lower().replace("-", "_")
    if normalized in {"tiny", "tiny_unet", "tiny_unet_4ch", "unet_tiny"}:
        return TinyUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    encoders = {
        "unet_resnet18": ("Unet", "resnet18"),
        "unet_resnet34": ("Unet", "resnet34"),
        "unet_resnet50": ("Unet", "resnet50"),
        "segformer_b0": ("Segformer", "mit_b0"),
        "segformer_b1": ("Segformer", "mit_b1"),
        "segformer_b2": ("Segformer", "mit_b2"),
        "segformer_b3": ("Segformer", "mit_b3"),
        "deeplabv3plus_resnet34": ("DeepLabV3Plus", "resnet34"),
        "deeplabv3plus_resnet50": ("DeepLabV3Plus", "resnet50"),
        "deeplab_r34": ("DeepLabV3Plus", "resnet34"),
        "deeplab_r50": ("DeepLabV3Plus", "resnet50"),
    }
    if normalized not in encoders:
        supported = ", ".join(spec.name for spec in list_supported_model_specs())
        raise ValueError(f"Unsupported model_name={model_name}. Supported: {supported}")
    import segmentation_models_pytorch as smp

    class_name, encoder_name = encoders[normalized]
    model_cls = getattr(smp, class_name)
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=out_channels,
        activation=None,
    )


def set_batchnorm_eval(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def configure_dropout(model: torch.nn.Module, dropout_p: float | None) -> int:
    if dropout_p is None:
        return 0
    probability = float(dropout_p)
    if probability < 0.0 or probability >= 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {probability}")
    updated = 0
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.p = probability
            updated += 1
    return updated
