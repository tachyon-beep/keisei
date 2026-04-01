"""Regression tests for compute_gae — extracted from ppo.py."""

import torch
import pytest

from keisei.training.gae import compute_gae


class TestComputeGAE:
    def test_single_step_no_done(self):
        rewards = torch.tensor([1.0])
        values = torch.tensor([0.5])
        dones = torch.tensor([False])
        next_value = torch.tensor(0.3)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.shape == (1,)
        # delta = 1.0 + 0.99 * 0.3 * 1.0 - 0.5 = 0.797
        assert abs(advantages[0].item() - 0.797) < 1e-3

    def test_episode_boundary_resets(self):
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.5, 0.5])
        dones = torch.tensor([True, False])
        next_value = torch.tensor(0.3)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.shape == (2,)
        # Step 0 is terminal: delta = 1.0 + 0 - 0.5 = 0.5
        assert abs(advantages[0].item() - 0.5) < 1e-3

    def test_multi_step_accumulation(self):
        """Verify GAE recursive accumulation over a non-terminal trajectory."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.shape == (3,)
        assert abs(advantages[2].item() - 2.5) < 1e-3
        assert abs(advantages[1].item() - 4.34625) < 1e-3
        assert abs(advantages[0].item() - 5.081) < 1e-2

    def test_output_dtype_and_device(self):
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert advantages.dtype == torch.float32
        assert advantages.shape == (3,)
