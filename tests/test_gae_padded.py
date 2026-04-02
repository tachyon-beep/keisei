# tests/test_gae_padded.py
"""Tests for padded/batched GAE computation."""

from __future__ import annotations

import torch
from keisei.training.gae import compute_gae, compute_gae_padded


class TestComputeGaePadded:
    def test_padded_matches_per_env(self) -> None:
        """Padded batched GAE must match per-env sequential GAE."""
        torch.manual_seed(42)
        gamma, lam = 0.99, 0.95
        lengths = [5, 3, 7]
        max_T = max(lengths)
        N = len(lengths)

        rewards_list = [torch.randn(L) for L in lengths]
        values_list = [torch.randn(L) for L in lengths]
        dones_list = [torch.zeros(L) for L in lengths]
        next_values = torch.randn(N)

        # Per-env reference
        ref_advantages = []
        for i, L in enumerate(lengths):
            adv = compute_gae(rewards_list[i], values_list[i], dones_list[i],
                              next_values[i], gamma, lam)
            ref_advantages.append(adv)

        # Padded batched
        rewards_pad = torch.zeros(max_T, N)
        values_pad = torch.zeros(max_T, N)
        dones_pad = torch.ones(max_T, N)  # pad with done=1 to zero out GAE
        length_tensor = torch.tensor(lengths)

        for i, L in enumerate(lengths):
            rewards_pad[:L, i] = rewards_list[i]
            values_pad[:L, i] = values_list[i]
            dones_pad[:L, i] = dones_list[i]

        padded_adv = compute_gae_padded(
            rewards_pad, values_pad, dones_pad, next_values, length_tensor,
            gamma, lam,
        )

        for i, L in enumerate(lengths):
            torch.testing.assert_close(
                padded_adv[:L, i], ref_advantages[i],
                atol=1e-6, rtol=1e-5,
            )

    def test_equal_lengths_matches_standard(self) -> None:
        """When all envs have equal length, padded == standard 2D GAE."""
        torch.manual_seed(123)
        T, N = 8, 4
        gamma, lam = 0.99, 0.95
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        dones = torch.zeros(T, N)
        next_values = torch.randn(N)
        lengths = torch.full((N,), T)

        ref = compute_gae(rewards, values, dones, next_values, gamma, lam)
        padded = compute_gae_padded(rewards, values, dones, next_values, lengths, gamma, lam)

        torch.testing.assert_close(padded, ref, atol=1e-6, rtol=1e-5)
