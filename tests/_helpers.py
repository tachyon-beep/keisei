"""Shared test helpers for Keisei test suite.

Contains TinyModel and make_rollout used across multiple test files.
Import directly: ``from tests._helpers import TinyModel, make_rollout``
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from keisei.training.dynamic_trainer import MatchRollout

OBS_CHANNELS = 50
ACTION_SPACE = 11259


class TinyModel(nn.Module):
    """Minimal model satisfying DynamicTrainer's forward-pass contract."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(OBS_CHANNELS * 9 * 9, 128)
        self.policy_head = nn.Linear(128, ACTION_SPACE)
        self.value_head = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> SimpleNamespace:
        flat = x.reshape(x.shape[0], -1)
        h = torch.relu(self.fc(flat))
        return SimpleNamespace(
            policy_logits=self.policy_head(h).reshape(x.shape[0], 1, -1),
            value_logits=self.value_head(h),
        )


def make_rollout(
    steps: int = 10,
    num_envs: int = 1,
    side: int = 0,
    include_terminal: bool = True,
) -> MatchRollout:
    """Create a synthetic MatchRollout with valid data."""
    obs = torch.randn(steps, num_envs, OBS_CHANNELS, 9, 9)
    actions = torch.randint(0, ACTION_SPACE, (steps, num_envs))
    rewards = torch.zeros(steps, num_envs)
    dones = torch.zeros(steps, num_envs)
    legal_masks = torch.zeros(steps, num_envs, ACTION_SPACE, dtype=torch.bool)
    legal_masks[:, :, 0] = True
    for s in range(steps):
        for e in range(num_envs):
            legal_masks[s, e, actions[s, e]] = True
    perspective = torch.full((steps, num_envs), side, dtype=torch.long)

    if include_terminal:
        dones[-1, :] = 1.0
        rewards[-1, :] = 1.0

    return MatchRollout(
        observations=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        legal_masks=legal_masks,
        perspective=perspective,
    )
