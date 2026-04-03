"""Regression tests for compute_gae — extracted from ppo.py."""

import torch
import pytest

from keisei.training.gae import compute_gae, compute_gae_gpu


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
        assert advantages.device == rewards.device


class TestGAEEdgeCases:
    """T9: Edge case parameters — undiscounted, TD(0), and Monte Carlo."""

    def test_gamma_one_undiscounted(self):
        """gamma=1 should give undiscounted advantages (no decay)."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.zeros(3)
        dones = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=1.0, lam=0.95)
        # With gamma=1, lam=0.95, values=0, next_value=0:
        # Step 2: delta=1, gae=1
        # Step 1: delta=1, gae=1+0.95*1=1.95
        # Step 0: delta=1, gae=1+0.95*1.95=2.8525
        assert abs(advantages[2].item() - 1.0) < 1e-5
        assert abs(advantages[1].item() - 1.95) < 1e-4
        assert abs(advantages[0].item() - 2.8525) < 1e-3

    def test_lam_zero_td0(self):
        """lam=0 should give pure TD(0) residuals (no GAE accumulation)."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([0.5, 0.5, 0.5])
        dones = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.0)
        # TD(0): advantage[t] = r_t + gamma * V(t+1) - V(t)
        assert abs(advantages[2].item() - (3.0 + 0.99 * 0.0 - 0.5)) < 1e-5
        assert abs(advantages[1].item() - (2.0 + 0.99 * 0.5 - 0.5)) < 1e-5
        assert abs(advantages[0].item() - (1.0 + 0.99 * 0.5 - 0.5)) < 1e-5

    def test_lam_one_monte_carlo(self):
        """lam=1 should give Monte Carlo-like advantages (full accumulation)."""
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.zeros(3)
        dones = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)
        advantages = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=1.0)
        # With lam=1, gamma=0.99, values=0:
        # Step 2: delta=1, gae=1
        # Step 1: delta=1, gae=1+0.99*1=1.99
        # Step 0: delta=1, gae=1+0.99*1.99=2.9701
        assert abs(advantages[2].item() - 1.0) < 1e-5
        assert abs(advantages[1].item() - 1.99) < 1e-4
        assert abs(advantages[0].item() - 2.9701) < 1e-3


class TestGAEBatched:
    """Test the vectorized (T, N) batched GAE path."""

    def test_batched_matches_per_env(self):
        """Batched (T, N) GAE should produce identical results to per-env calls."""
        T, N = 5, 3
        torch.manual_seed(42)
        rewards_2d = torch.randn(T, N)
        values_2d = torch.randn(T, N) * 0.5
        dones_2d = torch.zeros(T, N, dtype=torch.bool)
        dones_2d[2, 0] = True  # episode boundary in env 0
        dones_2d[3, 1] = True  # episode boundary in env 1
        next_values = torch.randn(N)

        # Batched call
        batched = compute_gae(rewards_2d, values_2d, dones_2d, next_values,
                              gamma=0.99, lam=0.95)

        # Per-env calls
        for env_i in range(N):
            per_env = compute_gae(rewards_2d[:, env_i], values_2d[:, env_i],
                                  dones_2d[:, env_i], next_values[env_i],
                                  gamma=0.99, lam=0.95)
            assert torch.allclose(batched[:, env_i], per_env, atol=1e-6), \
                f"Mismatch in env {env_i}"

    def test_batched_shape(self):
        """Batched output should be (T, N)."""
        T, N = 8, 4
        adv = compute_gae(
            torch.randn(T, N), torch.randn(T, N),
            torch.zeros(T, N, dtype=torch.bool), torch.randn(N),
            gamma=0.99, lam=0.95,
        )
        assert adv.shape == (T, N)


class TestComputeGAEGPU:
    """Test GPU GAE against CPU reference implementation.

    These tests run on CPU tensors to validate numerical equivalence.
    compute_gae_gpu() operates on whatever device its inputs are on —
    on CPU-only CI, this tests the algorithm; on CUDA machines, actual
    GPU execution is tested via the integration path in update().
    """

    def test_gpu_matches_cpu_basic(self):
        """GPU GAE matches CPU GAE within tolerance for basic 2D input."""
        T, N = 8, 4
        torch.manual_seed(42)
        rewards = torch.randn(T, N)
        values = torch.randn(T, N) * 0.5
        dones = torch.zeros(T, N)
        dones[3, 0] = 1.0  # episode boundary
        dones[5, 2] = 1.0
        next_value = torch.randn(N)

        cpu_result = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        gpu_result = compute_gae_gpu(
            rewards.clone(), values.clone(), dones.clone(), next_value.clone(),
            gamma=0.99, lam=0.95,
        )
        assert torch.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5), (
            f"Max diff: {(cpu_result - gpu_result).abs().max().item()}"
        )

    def test_gpu_matches_cpu_large(self):
        """GPU GAE matches CPU for realistic rollout dimensions."""
        T, N = 128, 64
        torch.manual_seed(123)
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        dones = (torch.rand(T, N) < 0.05).float()
        next_value = torch.randn(N)

        cpu_result = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        gpu_result = compute_gae_gpu(
            rewards.clone(), values.clone(), dones.clone(), next_value.clone(),
            gamma=0.99, lam=0.95,
        )
        assert torch.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5), (
            f"Max diff: {(cpu_result - gpu_result).abs().max().item()}"
        )

    def test_gpu_all_done(self):
        """All-done trajectory: every step is terminal, advantages = pure TD residuals."""
        T, N = 5, 3
        torch.manual_seed(7)
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        dones = torch.ones(T, N)
        next_value = torch.randn(N)

        cpu_result = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        gpu_result = compute_gae_gpu(
            rewards.clone(), values.clone(), dones.clone(), next_value.clone(),
            gamma=0.99, lam=0.95,
        )
        assert torch.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5)

    def test_gpu_single_timestep(self):
        """T=1 boundary case: values[1:] is empty, cat produces only bootstrap."""
        T, N = 1, 4
        torch.manual_seed(99)
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        dones = torch.zeros(T, N)
        next_value = torch.randn(N)

        cpu_result = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        gpu_result = compute_gae_gpu(
            rewards.clone(), values.clone(), dones.clone(), next_value.clone(),
            gamma=0.99, lam=0.95,
        )
        assert torch.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5)

    def test_gpu_single_env(self):
        """N=1 boundary case: single environment column."""
        T, N = 10, 1
        torch.manual_seed(55)
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        dones = torch.zeros(T, N)
        dones[5, 0] = 1.0
        next_value = torch.randn(N)

        cpu_result = compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        gpu_result = compute_gae_gpu(
            rewards.clone(), values.clone(), dones.clone(), next_value.clone(),
            gamma=0.99, lam=0.95,
        )
        assert torch.allclose(cpu_result, gpu_result, rtol=1e-5, atol=1e-5)

    def test_gpu_rejects_1d_input(self):
        """compute_gae_gpu must reject 1D input — only 2D (T, N) is supported.

        The 1D flat fallback conflates transitions across environment boundaries
        after flattening. See spec hazard H3.
        """
        rewards = torch.randn(10)
        values = torch.randn(10)
        dones = torch.zeros(10)
        next_value = torch.tensor(0.5)
        with pytest.raises(ValueError, match="only supports 2D"):
            compute_gae_gpu(rewards, values, dones, next_value, gamma=0.99, lam=0.95)

    def test_gpu_output_shape_and_dtype(self):
        """GPU GAE output has correct shape and dtype."""
        T, N = 10, 8
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        dones = torch.zeros(T, N)
        next_value = torch.randn(N)

        result = compute_gae_gpu(rewards, values, dones, next_value, gamma=0.99, lam=0.95)
        assert result.shape == (T, N)
        assert result.dtype == torch.float32
