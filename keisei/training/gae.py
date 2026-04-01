"""Generalized Advantage Estimation — shared utility for PPO variants."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute GAE advantages for a single environment's trajectory.

    Args:
        rewards: (T,) per-step rewards
        values: (T,) value estimates at each step
        dones: (T,) episode termination flags
        next_value: scalar value estimate for the state after the last step
        gamma: discount factor
        lam: GAE lambda (bias-variance tradeoff)

    Returns:
        (T,) advantage estimates
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    return advantages
