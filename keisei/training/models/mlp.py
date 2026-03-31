"""MLP baseline architecture -- intentionally lacks spatial inductive bias."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseModel


@dataclass(frozen=True)
class MLPParams:
    hidden_sizes: list[int]


class MLPModel(BaseModel):
    def __init__(self, params: MLPParams) -> None:
        super().__init__()
        input_size = self.OBS_CHANNELS * self.BOARD_SIZE * self.BOARD_SIZE

        layers: list[nn.Module] = []
        prev_size = input_size
        for size in params.hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.ReLU())
            prev_size = size
        self.trunk = nn.Sequential(*layers)

        self.policy_fc = nn.Linear(prev_size, self.ACTION_SPACE)
        self.value_fc = nn.Linear(prev_size, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = obs.flatten(1)
        x = self.trunk(x)
        policy_logits = self.policy_fc(x)
        value = torch.tanh(self.value_fc(x))
        return policy_logits, value
