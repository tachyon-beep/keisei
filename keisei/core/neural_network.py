"""
Minimal ActorCritic neural network for DRL Shogi Client (dummy forward pass).
"""

from typing import Tuple

import torch
from torch import nn

from keisei.constants import SHOGI_BOARD_SQUARES

from .base_actor_critic import BaseActorCriticModel


class ActorCritic(BaseActorCriticModel):
    """Actor-Critic neural network for Shogi RL agent (PPO-ready)."""

    def __init__(self, input_channels: int, num_actions_total: int):
        """Initialize the ActorCritic network with convolutional and linear layers."""
        super().__init__()
        self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.policy_head = nn.Linear(16 * SHOGI_BOARD_SQUARES, num_actions_total)
        self.value_head = nn.Linear(16 * SHOGI_BOARD_SQUARES, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns policy logits and value estimate."""
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value
