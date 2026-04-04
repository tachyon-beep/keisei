"""Generalized Advantage Estimation — shared utility for PPO variants."""

from __future__ import annotations

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute GAE advantages for one or many environments.

    Supports both 1D (single env) and 2D (batched) inputs:
      - 1D: rewards (T,), values (T,), terminated (T,), next_value scalar
      - 2D: rewards (T, N), values (T, N), terminated (T, N), next_value (N,)

    Args:
        rewards: per-step rewards
        values: value estimates at each step
        terminated: True only for genuinely terminal episodes (not truncated).
            Truncated episodes bootstrap from V(s_next) instead of zeroing it.
        next_value: value estimate(s) for the state after the last step
        gamma: discount factor
        lam: GAE lambda (bias-variance tradeoff)

    Returns:
        Advantage estimates, same shape as rewards
    """
    T = rewards.shape[0]
    compute_dtype = values.dtype  # always float — safe for GAE math
    rewards = rewards.to(dtype=compute_dtype)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)

    for t in reversed(range(T)):
        if t == T - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        not_done = 1.0 - terminated[t].float()
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    return advantages


def compute_gae_padded(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    next_values: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute GAE for multiple envs with variable-length episodes via padding.

    All inputs are padded to (T_max, N). Positions beyond each env's length
    must have terminated=1 so that GAE propagation is zeroed out for padding.

    Args:
        rewards: (T_max, N) padded rewards
        values: (T_max, N) padded value estimates
        terminated: (T_max, N) termination flags — padding positions MUST be 1.0.
            True only for genuinely terminal episodes (not truncated).
            Truncated episodes bootstrap from V(s_next) instead of zeroing it.
        next_values: (N,) bootstrap values per env
        lengths: (N,) actual sequence length per env
        gamma: discount factor
        lam: GAE lambda

    Returns:
        (T_max, N) advantages — only [:lengths[i], i] are meaningful per env
    """
    T_max, N = rewards.shape
    compute_dtype = values.dtype
    rewards = rewards.to(dtype=compute_dtype)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_values)

    # Build a (T_max, N) tensor of the correct next_val for each (t, env):
    # - at the env's actual last valid step (t == lengths[i]-1), use next_values[i]
    # - elsewhere within valid range, use values[t+1]
    # - padding positions have terminated=1 so not_done=0 and the value doesn't matter
    next_vals = torch.zeros_like(rewards)
    next_vals[:-1] = values[1:]          # default: shift values by 1
    next_vals[-1] = next_values          # last row bootstrap (for T_max-length envs)

    # For envs shorter than T_max, override their last valid step with next_values[i]
    # lengths[i]-1 is the index of the last valid timestep for env i
    last_step_idx = (lengths - 1).clamp(min=0)  # (N,) indices, shape guard
    for i in range(N):
        t_last = int(last_step_idx[i].item())
        next_vals[t_last, i] = next_values[i]

    for t in reversed(range(T_max)):
        not_done = 1.0 - terminated[t].float()
        delta = rewards[t] + gamma * next_vals[t] * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        advantages[t] = last_gae

    return advantages


def compute_gae_gpu(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GPU GAE for structured (T, N) rollouts.

    Same recurrence as compute_gae(), but keeps all computation on the input
    device (typically CUDA). Only supports 2D (T, N) input where each column
    is a single environment's unbroken trajectory.

    The 1D flat case is NOT supported — the vectorized values[1:] trick would
    conflate transitions across environment boundaries after flattening. Use
    the original compute_gae() for 1D inputs. See spec hazard H3.

    Args:
        rewards: (T, N) per-step rewards
        values: (T, N) value estimates at each step
        terminated: (T, N) episode termination flags (1.0 = terminated). True only
            for genuinely terminal episodes (not truncated). Truncated episodes
            bootstrap from V(s_next) instead of zeroing it.
        next_value: (N,) bootstrap value for the state after the last step
        gamma: discount factor
        lam: GAE lambda (bias-variance tradeoff)

    Returns:
        (T, N) advantage estimates on the same device as inputs
    """
    if rewards.ndim != 2:
        raise ValueError(
            f"compute_gae_gpu only supports 2D (T, N) input, got shape {rewards.shape}"
        )

    T, N = rewards.shape
    compute_dtype = values.dtype
    rewards = rewards.to(dtype=compute_dtype)

    # Step 1: vectorized delta and decay (no Python loop)
    # next_values[t] = values[t+1] for t < T-1, next_value for t == T-1
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)
    not_done = 1.0 - terminated.float()
    delta = rewards + gamma * next_values * not_done - values
    decay = gamma * lam * not_done

    # Step 2: sequential backward scan — each step is a fused GPU kernel over N envs.
    # The Python loop has T iterations (~128), each launching ~2 CUDA kernels.
    # This is faster than CPU GAE because each step operates on N envs in parallel
    # without Python-level tensor ops or CPU-GPU round-trips.
    advantages = torch.empty_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=compute_dtype)
    for t in reversed(range(T)):
        last_gae = delta[t] + decay[t] * last_gae
        advantages[t] = last_gae

    return advantages
