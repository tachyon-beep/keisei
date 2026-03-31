"""ResNet architecture for Shogi policy+value network."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseModel


@dataclass(frozen=True)
class ResNetParams:
    hidden_size: int
    num_layers: int


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class ResNetModel(BaseModel):
    def __init__(self, params: ResNetParams) -> None:
        super().__init__()
        ch = params.hidden_size

        self.input_conv = nn.Conv2d(self.OBS_CHANNELS, ch, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(ch)
        self.blocks = nn.Sequential(*[ResidualBlock(ch) for _ in range(params.num_layers)])

        policy_channels = 2
        self.policy_conv = nn.Conv2d(ch, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(
            policy_channels * self.BOARD_SIZE * self.BOARD_SIZE, self.ACTION_SPACE
        )

        value_channels = 1
        self.value_conv = nn.Conv2d(ch, value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * self.BOARD_SIZE * self.BOARD_SIZE, ch)
        self.value_fc2 = nn.Linear(ch, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.input_bn(self.input_conv(obs)))
        x = self.blocks(x)

        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
