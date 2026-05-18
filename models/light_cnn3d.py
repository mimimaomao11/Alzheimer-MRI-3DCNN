"""Lightweight 3D residual CNN for small-dataset MRI classification.

~470K parameters (vs DenseNet121's 11M) — designed for datasets under 200 subjects.

Architecture:
    Stem  : Conv(1→16, 7×7×7, stride=2) + BN + ReLU   → (16, 64³)
    Stage1: ResBlock(16→16) + DownConv(16→32, stride=2) → (32, 32³)
    Stage2: ResBlock(32→32) + DownConv(32→64, stride=2) → (64, 16³)
    Stage3: ResBlock(64→64) + DownConv(64→128,stride=2) → (128, 8³)
    Head  : GlobalAvgPool → [ClinicalMLP concat] → Dropout → Linear(128[+32], 2)

When num_clinical > 0, a small MLP maps clinical features → 32-dim and
concatenates with the image embedding before the final classifier.
num_clinical=0 (default) is identical to the original architecture so
old checkpoints load without modification.
"""
from __future__ import annotations

import torch
from torch import nn


class _ResBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class _DownConv3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightCNN3D(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 num_clinical: int = 0) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(_ResBlock3D(16, dropout=0.1), _DownConv3D(16, 32))
        self.stage2 = nn.Sequential(_ResBlock3D(32, dropout=0.1), _DownConv3D(32, 64))
        self.stage3 = nn.Sequential(_ResBlock3D(64, dropout=0.1), _DownConv3D(64, 128))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.num_clinical = num_clinical

        if num_clinical > 0:
            self.clinical_mlp = nn.Sequential(
                nn.Linear(num_clinical, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 32),
            )

        head_in = 128 + (32 if num_clinical > 0 else 0)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(head_in, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor,
                clinical: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)

        if self.num_clinical > 0 and clinical is not None:
            img_feat = x.flatten(1)                          # (B, 128)
            clin_feat = self.clinical_mlp(clinical)          # (B, 32)
            x = torch.cat([img_feat, clin_feat], dim=1)      # (B, 160)
            x = self.classifier[1](x)                        # Dropout
            return self.classifier[2](x)                     # Linear(160, 2)

        return self.classifier(x)


def build_light_cnn3d(num_classes: int = 2, dropout: float = 0.5,
                      num_clinical: int = 0) -> LightCNN3D:
    return LightCNN3D(num_classes=num_classes, dropout=dropout,
                      num_clinical=num_clinical)
