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


def compute_gae_padded(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_values: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute GAE for multiple envs with variable-length episodes via padding.

    All inputs are padded to (T_max, N). Positions beyond each env's length
    must have dones=1 so that GAE propagation is zeroed out for padding.

    Args:
        rewards: (T_max, N) padded rewards
        values: (T_max, N) padded value estimates
        dones: (T_max, N) termination flags — padding positions MUST be 1.0
        next_values: (N,) bootstrap values per env
        lengths: (N,) actual sequence length per env
        gamma: discount factor
        lam: GAE lambda

    Returns:
        (T_max, N) advantages — only [:lengths[i], i] are meaningful per env
    """
    T_max, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_values)

    # Build a (T_max, N) tensor of the correct next_val for each (t, env):
    # - at the env's actual last valid step (t == lengths[i]-1), use next_values[i]
    # - elsewhere within valid range, use values[t+1]
    # - padding positions have dones=1 so not_done=0 and the value doesn't matter
    next_vals = torch.zeros_like(rewards)
    next_vals[:-1] = values[1:]          # default: shift values by 1
    next_vals[-1] = next_values          # last row bootstrap (for T_max-length envs)

    # For envs shorter than T_max, override their last valid step with next_values[i]
    # lengths[i]-1 is the index of the last valid timestep for env i
    last_step_idx = (lengths - 1).clamp(min=0)  # (N,) indices, shape guard
    for i in range(N):
        t_last = last_step_idx[i].item()
        next_vals[t_last, i] = next_values[i]

    for t in reversed(range(T_max)):
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_vals[t] * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    return advantages
