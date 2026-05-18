from __future__ import annotations

from monai.networks.nets import DenseNet121
from torch import nn


def build_densenet121(num_classes: int = 2, dropout_prob: float = 0.2) -> nn.Module:
    return DenseNet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        dropout_prob=dropout_prob,
    )


def build_densenet121_3ch(num_classes: int = 2, dropout_prob: float = 0.2) -> nn.Module:
    return DenseNet121(
        spatial_dims=3,
        in_channels=3,
        out_channels=num_classes,
        dropout_prob=dropout_prob,
    )
