from __future__ import annotations

import torch

from .unet import TinyUNet


def build_model(model_name: str, in_channels: int, out_channels: int, base_channels: int) -> torch.nn.Module:
    name = (model_name or "tiny_unet_4ch").lower()
    if name in {"tiny_unet_4ch", "tiny_unet", "unet_tiny"}:
        return TinyUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name.startswith("segformer"):
        try:
            import segmentation_models_pytorch as smp

            encoder = "mit_b0"
            if "b1" in name:
                encoder = "mit_b1"
            elif "b2" in name:
                encoder = "mit_b2"
            return smp.Unet(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=in_channels,
                classes=out_channels,
            )
        except Exception:
            return TinyUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if name.startswith("deeplab"):
        try:
            import segmentation_models_pytorch as smp

            encoder = "resnet34" if "34" in name else "resnet18"
            return smp.DeepLabV3Plus(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=in_channels,
                classes=out_channels,
            )
        except Exception:
            return TinyUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    return TinyUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)


def set_batchnorm_eval(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
