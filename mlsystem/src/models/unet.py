from __future__ import annotations

import torch


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
