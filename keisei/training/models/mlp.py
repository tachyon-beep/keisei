"""MLP baseline architecture -- intentionally lacks spatial inductive bias."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .base import BaseModel


@dataclass(frozen=True)
class MLPParams:
    hidden_sizes: list[int]

    def __post_init__(self) -> None:
        if any(s <= 0 for s in self.hidden_sizes):
            raise ValueError(
                f"All hidden_sizes must be > 0, got {self.hidden_sizes}"
            )


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
        if obs.ndim != 4 or obs.shape[1] != self.OBS_CHANNELS or obs.shape[2] != self.BOARD_SIZE or obs.shape[3] != self.BOARD_SIZE:
            hint = ""
            if obs.ndim == 4 and obs.shape[-1] == self.OBS_CHANNELS:
                hint = " (input appears to be NHWC — expected NCHW)"
            raise ValueError(
                f"Expected obs shape (batch, {self.OBS_CHANNELS}, {self.BOARD_SIZE}, {self.BOARD_SIZE}), "
                f"got {tuple(obs.shape)}{hint}"
            )
        x = obs.flatten(1)
        x = self.trunk(x)
        policy_logits = self.policy_fc(x)
        value = torch.tanh(self.value_fc(x))
        return policy_logits, value
