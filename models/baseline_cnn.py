from __future__ import annotations

import torch
from torch import nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout3d: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(p=dropout3d),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Baseline3DCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout3d: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock3D(1, 16, dropout3d=dropout3d),
            ConvBlock3D(16, 32, dropout3d=dropout3d),
            ConvBlock3D(32, 64, dropout3d=dropout3d),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
