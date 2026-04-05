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

    def test_single_step_env(self):
        """Length-1 environment should produce correct advantage for single timestep."""
        from keisei.training.gae import compute_gae, compute_gae_padded

        T_max, N = 5, 2
        lengths = torch.tensor([1, 5])
        rewards = torch.zeros(T_max, N)
        values = torch.zeros(T_max, N)
        dones = torch.ones(T_max, N)  # all done (padding)
        next_values = torch.tensor([0.5, 0.3])

        # Fill valid positions
        rewards[0, 0] = 1.0  # env 0: single step with reward 1.0
        values[0, 0] = 0.2
        dones[0, 0] = 0.0    # not done at step 0 for env 0

        for t in range(5):
            rewards[t, 1] = 0.1
            values[t, 1] = 0.1
            dones[t, 1] = 0.0

        gamma, lam = 0.99, 0.95

        result = compute_gae_padded(rewards, values, dones, next_values, lengths, gamma, lam)

        # Verify env 0 (length=1): reference GAE with single step
        ref_0 = compute_gae(
            rewards[:1, 0], values[:1, 0], dones[:1, 0],
            next_values[0], gamma, lam,
        )
        torch.testing.assert_close(result[0, 0], ref_0[0], atol=1e-6, rtol=1e-5)

        # Verify env 1 (length=5): reference GAE with full sequence
        ref_1 = compute_gae(
            rewards[:5, 1], values[:5, 1], dones[:5, 1],
            next_values[1], gamma, lam,
        )
        torch.testing.assert_close(result[:5, 1], ref_1, atol=1e-6, rtol=1e-5)

    def test_all_envs_full_length(self):
        """All envs at T_max length — no padding needed."""
        from keisei.training.gae import compute_gae, compute_gae_padded

        T_max, N = 4, 3
        lengths = torch.tensor([4, 4, 4])
        rewards = torch.randn(T_max, N)
        values = torch.randn(T_max, N)
        dones = torch.zeros(T_max, N)
        next_values = torch.randn(N)
        gamma, lam = 0.99, 0.95

        result = compute_gae_padded(rewards, values, dones, next_values, lengths, gamma, lam)

        for i in range(N):
            ref = compute_gae(rewards[:, i], values[:, i], dones[:, i],
                              next_values[i], gamma, lam)
            torch.testing.assert_close(result[:, i], ref, atol=1e-6, rtol=1e-5)
