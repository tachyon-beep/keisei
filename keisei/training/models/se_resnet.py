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

    def __post_init__(self) -> None:
        for field_name in (
            "num_blocks", "channels", "se_reduction", "global_pool_channels",
            "policy_channels", "value_fc_size", "score_fc_size", "obs_channels",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be >= 1, got {getattr(self, field_name)}")
        if self.channels // self.se_reduction < 1:
            raise ValueError(
                f"channels ({self.channels}) // se_reduction ({self.se_reduction}) "
                f"must be >= 1"
            )


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


def _global_pool(x: torch.Tensor) -> torch.Tensor:
    """Global pool: mean + max + std concatenated. Input (B, C, H, W) -> (B, 3C)."""
    g_mean = x.mean(dim=(-2, -1))
    g_max = x.amax(dim=(-2, -1))
    g_std = x.std(dim=(-2, -1), correction=0)  # population std
    return torch.cat([g_mean, g_max, g_std], dim=-1)


class SEResNetModel(KataGoBaseModel):
    """SE-ResNet with global pooling bias, 3-head output."""

    def __init__(self, params: SEResNetParams) -> None:
        super().__init__()  # KataGoBaseModel.__init__ — sets AMP defaults
        self.params = params
        ch = params.channels

        # Input conv
        self.input_conv = nn.Conv2d(params.obs_channels, ch, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(ch)

        # Residual tower
        self.blocks = nn.Sequential(*[
            GlobalPoolBiasBlock(ch, params.se_reduction, params.global_pool_channels)
            for _ in range(params.num_blocks)
        ])

        # Policy head: two conv layers -> (B, 139, 9, 9) -> permute to (B, 9, 9, 139)
        self.policy_conv1 = nn.Conv2d(ch, params.policy_channels, 1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(params.policy_channels)
        self.policy_conv2 = nn.Conv2d(params.policy_channels, self.SPATIAL_MOVE_TYPES, 1)

        # Value head: global pool -> FC -> 3 logits (W/D/L)
        self.value_fc1 = nn.Linear(ch * 3, params.value_fc_size)
        self.value_fc2 = nn.Linear(params.value_fc_size, 3)

        # Score head: global pool -> FC -> 1 scalar
        self.score_fc1 = nn.Linear(ch * 3, params.score_fc_size)
        self.score_fc2 = nn.Linear(params.score_fc_size, 1)

    def _forward_impl(self, obs: torch.Tensor) -> KataGoOutput:
        if obs.ndim != 4 or obs.shape[1] != self.params.obs_channels or obs.shape[2] != 9 or obs.shape[3] != 9:
            raise ValueError(
                f"Expected obs shape (batch, {self.params.obs_channels}, 9, 9), "
                f"got {tuple(obs.shape)}"
            )

        # Trunk
        x = F.relu(self.input_bn(self.input_conv(obs)))
        x = self.blocks(x)

        # Policy head
        p = F.relu(self.policy_bn1(self.policy_conv1(x)))
        p = self.policy_conv2(p)                    # (B, 139, 9, 9)
        p = p.permute(0, 2, 3, 1)                   # (B, 9, 9, 139)

        # Global pool shared by value and score heads (computed once)
        pool = _global_pool(x)                       # (B, 3C)

        # Value head
        v = F.relu(self.value_fc1(pool))
        v = self.value_fc2(v)                        # (B, 3) raw logits

        # Score head
        s = F.relu(self.score_fc1(pool))
        s = self.score_fc2(s)                        # (B, 1)

        return KataGoOutput(policy_logits=p, value_logits=v, score_lead=s)
