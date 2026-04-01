# keisei/training/models/se_resnet.py
"""SE-ResNet architecture with KataGo-style global pooling bias."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .katago_base import KataGoBaseModel, KataGoOutput


@dataclass(frozen=True)
class SEResNetParams:
    num_blocks: int = 40
    channels: int = 256
    se_reduction: int = 16
    global_pool_channels: int = 128
    policy_channels: int = 32
    value_fc_size: int = 256
    score_fc_size: int = 128
    obs_channels: int = 50


class GlobalPoolBiasBlock(nn.Module):
    """SE-ResBlock with global pooling bias (KataGo-style).

    Architecture:
        conv1 -> BN -> ReLU -> add global_pool_bias(block_input) -> conv2 -> BN
        -> SE(scale + shift) -> residual add -> ReLU
    """

    def __init__(self, channels: int, se_reduction: int, global_pool_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # Global pooling bias: pools from block INPUT (mean + max + std -> bottleneck -> channels)
        # global_pool_channels controls the bottleneck width (spec: configurable, default 128)
        self.global_fc = nn.Sequential(
            nn.Linear(channels * 3, global_pool_channels),
            nn.ReLU(),
            nn.Linear(global_pool_channels, channels),
        )

        # SE: squeeze-and-excitation with scale + shift
        se_hidden = channels // se_reduction
        self.se_fc1 = nn.Linear(channels, se_hidden)
        self.se_fc2 = nn.Linear(se_hidden, channels * 2)  # scale + shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))

        # Global pool bias from block INPUT x (not post-conv1 out)
        g_mean = x.mean(dim=(-2, -1))  # (B, C)
        g_max = x.amax(dim=(-2, -1))  # (B, C)
        g_std = x.std(dim=(-2, -1), correction=0)  # (B, C) -- population std
        g = self.global_fc(torch.cat([g_mean, g_max, g_std], dim=-1))  # (B, C)
        out = out + g.unsqueeze(-1).unsqueeze(-1)  # broadcast over 9x9

        out = self.bn2(self.conv2(out))

        # SE attention: pool post-conv2 output
        se_input = out.mean(dim=(-2, -1))  # (B, C)
        se = F.relu(self.se_fc1(se_input))
        se = self.se_fc2(se)  # (B, 2C)
        scale, shift = se.chunk(2, dim=-1)  # each (B, C)
        out = out * torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1) + \
            shift.unsqueeze(-1).unsqueeze(-1)

        return F.relu(out + residual)
