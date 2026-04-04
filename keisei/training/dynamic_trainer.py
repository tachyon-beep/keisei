"""DynamicTrainer — small PPO updates for Dynamic entries from league match data."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MatchRollout:
    """Replay data from a league match for Dynamic entry training.

    Step dimension is VARIABLE — depends on game length, not a fixed value.
    All tensors stored on CPU to avoid GPU memory pressure.
    """

    observations: torch.Tensor  # (steps, num_envs, obs_channels, 9, 9)
    actions: torch.Tensor  # (steps, num_envs)
    rewards: torch.Tensor  # (steps, num_envs)
    dones: torch.Tensor  # (steps, num_envs)
    legal_masks: torch.Tensor  # (steps, num_envs, action_space)
    perspective: torch.Tensor  # (steps, num_envs) — 0=player_A, 1=player_B
