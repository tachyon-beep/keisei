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
    """Compute GAE advantages for one or many environments.

    Supports both 1D (single env) and 2D (batched) inputs:
      - 1D: rewards (T,), values (T,), dones (T,), next_value scalar
      - 2D: rewards (T, N), values (T, N), dones (T, N), next_value (N,)

    Args:
        rewards: per-step rewards
        values: value estimates at each step
        dones: episode termination flags
        next_value: value estimate(s) for the state after the last step
        gamma: discount factor
        lam: GAE lambda (bias-variance tradeoff)

    Returns:
        Advantage estimates, same shape as rewards
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)

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
