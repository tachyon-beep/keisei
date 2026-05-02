"""Regression tests for compute_gae — extracted from ppo.py."""

import pytest
import torch

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


class TestGAETruncationBootstrap:
    """R1: Verify that truncated episodes bootstrap V(s_next) instead of zeroing."""

    def test_truncated_bootstraps_value(self):
        """With terminated-only signal, truncated positions should bootstrap."""
        rewards = torch.tensor([[0.0], [0.0], [1.0]])   # (T=3, N=1)
        values = torch.tensor([[0.5], [0.6], [0.7]])
        next_value = torch.tensor([0.8])

        # Old behavior: dones = terminated | truncated. Step 1 truncated -> dones=True
        old_dones = torch.tensor([[0.0], [1.0], [0.0]])
        adv_old = compute_gae_gpu(rewards, values, old_dones, next_value, gamma=0.99, lam=0.95)

        # New behavior: only truly terminated. Step 1 NOT terminated.
        terminated = torch.tensor([[0.0], [0.0], [0.0]])
        adv_new = compute_gae_gpu(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)

        # Old: adv_old[1,0] = -0.6 (done=True zeroed bootstrap, no gae carry-over from t=2)
        # New: adv_new[1,0] = delta[1] + gamma*lam*gae[2]
        #      delta[1] = 0 + 0.99 * 0.7 - 0.6 = 0.093
        #      gae[2]   = 1.0 + 0.99*0.8 - 0.7 = 1.092
        #      adv_new[1,0] ~= 0.093 + 0.99*0.95*1.092 ~= 1.120
        assert abs(adv_old[1, 0].item() - (-0.6)) < 0.01
        assert abs(adv_new[1, 0].item() - 1.120) < 0.05
        assert abs(adv_new[1, 0].item() - adv_old[1, 0].item()) > 0.5

    def test_truncation_vs_terminal_differ(self):
        """Passing dones (merged) vs terminated (split) must give different results."""
        rewards = torch.tensor([[0.0], [0.0]])
        values = torch.tensor([[0.5], [0.5]])
        next_value = torch.tensor([0.8])

        terminated = torch.tensor([[0.0], [0.0]])
        adv_terminated = compute_gae_gpu(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)

        dones_merged = torch.tensor([[1.0], [0.0]])
        adv_dones = compute_gae_gpu(rewards, values, dones_merged, next_value, gamma=0.99, lam=0.95)

        assert not torch.allclose(adv_terminated, adv_dones)

    def test_cpu_gae_truncation_bootstrap(self):
        """CPU compute_gae also bootstraps correctly for truncated positions."""
        rewards = torch.tensor([0.0, 0.0])
        values = torch.tensor([0.5, 0.5])
        next_value = torch.tensor(0.8)

        terminated = torch.tensor([0.0, 0.0])
        adv_terminated = compute_gae(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)

        dones_merged = torch.tensor([1.0, 0.0])
        adv_dones = compute_gae(rewards, values, dones_merged, next_value, gamma=0.99, lam=0.95)

        assert not torch.allclose(adv_terminated, adv_dones)

    def test_backward_compat_no_truncation(self):
        """When no truncation occurs, terminated == dones gives identical results."""
        rewards = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
        values = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.7, 0.2]])
        next_value = torch.tensor([0.1, 0.5])

        # Some positions terminated, NO truncation — so terminated == dones
        terminated = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        adv_terminated = compute_gae_gpu(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)

        # Simulate merged dones where dones has EXTRA done flags from truncation
        dones_with_truncation = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
        adv_dones = compute_gae_gpu(rewards, values, dones_with_truncation, next_value, gamma=0.99, lam=0.95)

        # When terminated == dones (no truncation), env 0 should match
        # But env 1 at step 1 has an extra done (truncation) that changes the result
        assert torch.allclose(adv_terminated[:, 0], adv_dones[:, 0])  # env 0: no truncation
        assert not torch.allclose(adv_terminated[:, 1], adv_dones[:, 1])  # env 1: truncation matters


class TestGAEPaddedTruncation:
    """R1: Truncation bootstrap for the padded GAE path."""

    def test_padded_truncated_bootstraps(self):
        from keisei.training.gae import compute_gae_padded
        rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
        values = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.0, 0.2]])
        terminated_pad = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        next_values = torch.tensor([0.3, 0.8])
        lengths = torch.tensor([2, 3])
        adv = compute_gae_padded(rewards, values, terminated_pad, next_values, lengths,
                                 gamma=0.99, lam=0.95)
        # Env 0, step 1: delta = 0.0 + 0.99 * 0.3 - 0.4 = -0.103
        assert abs(adv[1, 0].item() - (-0.103)) < 0.05


class TestGAEIntegerRewardDtype:
    """Regression: integer reward tensors must produce float advantages.

    GAE math is inherently fractional (gamma, lambda, value bootstraps).
    If the output buffer inherits an integer dtype from rewards, the
    fractional values are silently truncated on assignment.
    """

    def test_compute_gae_integer_rewards(self):
        """compute_gae with integer rewards must return float advantages.

        The dtype fix derives compute_dtype from values.dtype (always float
        in practice since values come from a neural network). This test
        validates that integer rewards are upcast to match.
        """
        rewards = torch.tensor([1, 2, 3], dtype=torch.long)
        values = torch.tensor([0.5, 0.5, 0.5])
        terminated = torch.zeros(3, dtype=torch.bool)
        next_value = torch.tensor(0.0)

        adv = compute_gae(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)
        assert adv.dtype.is_floating_point, f"Expected float dtype, got {adv.dtype}"

        # Compare against float reference
        ref = compute_gae(rewards.float(), values, terminated, next_value, gamma=0.99, lam=0.95)
        assert torch.allclose(adv, ref, atol=1e-5), (
            f"Integer rewards gave wrong values: {adv} vs {ref}"
        )

    def test_compute_gae_gpu_integer_rewards(self):
        """compute_gae_gpu with integer rewards must return float advantages."""
        rewards = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        values = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        terminated = torch.zeros(2, 2)
        next_value = torch.tensor([0.0, 0.0])

        adv = compute_gae_gpu(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)
        assert adv.dtype.is_floating_point, f"Expected float dtype, got {adv.dtype}"

        ref = compute_gae_gpu(rewards.float(), values, terminated, next_value, gamma=0.99, lam=0.95)
        assert torch.allclose(adv, ref, atol=1e-5)

    def test_compute_gae_padded_integer_rewards(self):
        """compute_gae_padded with integer rewards must return float advantages.

        Uses non-terminal trajectories so the GAE accumulation path (bootstrap +
        recursive last_gae carry) is exercised with fractional intermediate values.
        """
        from keisei.training.gae import compute_gae_padded

        rewards = torch.tensor([[1, 0], [2, 1], [0, 3]], dtype=torch.long)
        values = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.7, 0.2]])
        terminated = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        next_values = torch.tensor([0.0, 0.8])
        lengths = torch.tensor([3, 3])

        adv = compute_gae_padded(rewards, values, terminated, next_values, lengths,
                                 gamma=0.99, lam=0.95)
        assert adv.dtype.is_floating_point, f"Expected float dtype, got {adv.dtype}"

        ref = compute_gae_padded(rewards.float(), values, terminated, next_values, lengths,
                                 gamma=0.99, lam=0.95)
        assert torch.allclose(adv, ref, atol=1e-5)


class TestGAENoAutogradLeak:
    """Regression: GAE targets must be non-differentiable.

    The advantages/returns computed by GAE are training targets — they must
    not carry an autograd graph back into the value network or any prior
    forward pass. Production callers detach defensively, but the helpers
    themselves must enforce the contract so a future caller cannot leak
    critic gradients into the policy loss.
    """

    @staticmethod
    def _grad_inputs(T: int = 4, N: int = 2):
        """Build inputs where every float tensor requires grad."""
        rewards = torch.randn(T, N, requires_grad=True)
        values = torch.randn(T, N, requires_grad=True)
        terminated = torch.zeros(T, N)
        next_value = torch.randn(N, requires_grad=True)
        return rewards, values, terminated, next_value

    def test_compute_gae_does_not_require_grad(self):
        rewards, values, terminated, next_value = self._grad_inputs()
        adv = compute_gae(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)
        assert not adv.requires_grad
        assert adv.grad_fn is None

    def test_compute_gae_gpu_does_not_require_grad(self):
        rewards, values, terminated, next_value = self._grad_inputs()
        adv = compute_gae_gpu(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)
        assert not adv.requires_grad
        assert adv.grad_fn is None

    def test_compute_gae_padded_does_not_require_grad(self):
        from keisei.training.gae import compute_gae_padded

        T, N = 4, 2
        rewards = torch.randn(T, N, requires_grad=True)
        values = torch.randn(T, N, requires_grad=True)
        terminated = torch.zeros(T, N)
        next_values = torch.randn(N, requires_grad=True)
        lengths = torch.tensor([T, T])

        adv = compute_gae_padded(rewards, values, terminated, next_values, lengths,
                                 gamma=0.99, lam=0.95)
        assert not adv.requires_grad
        assert adv.grad_fn is None

    def test_compute_gae_padded_gpu_does_not_require_grad(self):
        from keisei.training.gae import compute_gae_padded_gpu

        T, N = 4, 2
        rewards = torch.randn(T, N, requires_grad=True)
        values = torch.randn(T, N, requires_grad=True)
        terminated = torch.zeros(T, N)
        next_values = torch.randn(N, requires_grad=True)
        lengths = torch.tensor([T, T])

        adv = compute_gae_padded_gpu(rewards, values, terminated, next_values, lengths,
                                     gamma=0.99, lam=0.95)
        assert not adv.requires_grad
        assert adv.grad_fn is None

    def test_inputs_remain_differentiable_after_call(self):
        """The helpers must not mutate the input tensors' grad state."""
        rewards, values, terminated, next_value = self._grad_inputs()
        _ = compute_gae(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)
        # Inputs are unchanged — caller can still backprop through them in
        # their own computation graph if it does so directly (not via GAE).
        assert values.requires_grad
        assert next_value.requires_grad
        assert rewards.requires_grad


class TestGAEPerCellNextValueOverride:
    """R: per-cell override of the bootstrap V used inside the GAE recurrence.

    Required for truncation handling where V(s_{t+1}) must come from the
    actual terminal observation, not from the post-auto-reset state. Also
    needed for two-player perspective handling where V(s_{t+1}) belongs
    to the opponent and must be sign-corrected.

    Contract: a NaN cell in `next_value_override` falls back to default
    behavior (values[t+1] for t<T-1, next_value at t=T-1). A finite cell
    replaces the bootstrap at that (t, n).
    """

    def test_compute_gae_override_replaces_values_shift(self):
        """Override at (t, n) replaces values[t+1, n] in the GAE delta."""
        T, N = 3, 1
        rewards = torch.tensor([[0.0], [0.0], [0.0]])
        values = torch.tensor([[0.5], [0.6], [0.7]])
        terminated = torch.zeros(T, N)
        next_value = torch.tensor([0.0])

        # Without override, step 0 bootstraps from values[1] = 0.6
        adv_default = compute_gae(rewards, values, terminated, next_value,
                                  gamma=1.0, lam=0.0)
        # delta[0] = 0 + 1.0 * 0.6 - 0.5 = 0.1
        assert abs(adv_default[0, 0].item() - 0.1) < 1e-5

        # With override at (0, 0) = 9.0, step 0 bootstraps from 9.0 instead
        override = torch.full((T, N), float("nan"))
        override[0, 0] = 9.0
        adv_override = compute_gae(rewards, values, terminated, next_value,
                                   gamma=1.0, lam=0.0,
                                   next_value_override=override)
        # delta[0] = 0 + 1.0 * 9.0 - 0.5 = 8.5
        assert abs(adv_override[0, 0].item() - 8.5) < 1e-5

    def test_compute_gae_gpu_override_replaces_values_shift(self):
        """compute_gae_gpu honours per-cell override the same way."""
        T, N = 3, 1
        rewards = torch.tensor([[0.0], [0.0], [0.0]])
        values = torch.tensor([[0.5], [0.6], [0.7]])
        terminated = torch.zeros(T, N)
        next_value = torch.tensor([0.0])

        adv_default = compute_gae_gpu(rewards, values, terminated, next_value,
                                      gamma=1.0, lam=0.0)
        assert abs(adv_default[0, 0].item() - 0.1) < 1e-5

        override = torch.full((T, N), float("nan"))
        override[0, 0] = 9.0
        adv_override = compute_gae_gpu(rewards, values, terminated, next_value,
                                       gamma=1.0, lam=0.0,
                                       next_value_override=override)
        assert abs(adv_override[0, 0].item() - 8.5) < 1e-5

    def test_override_replaces_last_step_bootstrap(self):
        """Override at t=T-1 replaces the last-step `next_value` parameter."""
        T, N = 2, 1
        rewards = torch.tensor([[0.0], [0.0]])
        values = torch.tensor([[0.5], [0.6]])
        terminated = torch.zeros(T, N)
        next_value = torch.tensor([0.0])

        adv_default = compute_gae(rewards, values, terminated, next_value,
                                  gamma=1.0, lam=0.0)
        # delta[1] = 0 + 1.0 * 0.0 - 0.6 = -0.6 (uses next_value)
        assert abs(adv_default[1, 0].item() - (-0.6)) < 1e-5

        override = torch.full((T, N), float("nan"))
        override[1, 0] = 5.0
        adv_override = compute_gae(rewards, values, terminated, next_value,
                                   gamma=1.0, lam=0.0,
                                   next_value_override=override)
        # delta[1] = 0 + 1.0 * 5.0 - 0.6 = 4.4
        assert abs(adv_override[1, 0].item() - 4.4) < 1e-5

    def test_override_partial_falls_back_for_nan(self):
        """Cells set to NaN keep default behavior; finite cells override."""
        T, N = 3, 2
        torch.manual_seed(0)
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        terminated = torch.zeros(T, N)
        next_value = torch.randn(N)

        baseline = compute_gae(rewards, values, terminated, next_value,
                               gamma=0.99, lam=0.95)

        # Override only env 0, step 1 — env 1 must match baseline
        override = torch.full((T, N), float("nan"))
        override[1, 0] = 42.0
        adv = compute_gae(rewards, values, terminated, next_value,
                          gamma=0.99, lam=0.95,
                          next_value_override=override)
        assert torch.allclose(adv[:, 1], baseline[:, 1], atol=1e-5), \
            "env 1 (no override) must match baseline"
        assert not torch.isclose(adv[1, 0], baseline[1, 0]), \
            "env 0 step 1 should differ — bootstrap was overridden"

    def test_override_none_is_unchanged_behavior(self):
        """Passing next_value_override=None must equal not passing it."""
        T, N = 4, 2
        torch.manual_seed(7)
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        terminated = torch.zeros(T, N)
        next_value = torch.randn(N)

        a = compute_gae(rewards, values, terminated, next_value,
                        gamma=0.99, lam=0.95)
        b = compute_gae(rewards, values, terminated, next_value,
                        gamma=0.99, lam=0.95, next_value_override=None)
        assert torch.allclose(a, b)

        a_gpu = compute_gae_gpu(rewards, values, terminated, next_value,
                                gamma=0.99, lam=0.95)
        b_gpu = compute_gae_gpu(rewards, values, terminated, next_value,
                                gamma=0.99, lam=0.95, next_value_override=None)
        assert torch.allclose(a_gpu, b_gpu)

    def test_override_preserves_no_grad_contract(self):
        """The override path must not leak gradients either."""
        T, N = 3, 2
        rewards = torch.randn(T, N, requires_grad=True)
        values = torch.randn(T, N, requires_grad=True)
        terminated = torch.zeros(T, N)
        next_value = torch.randn(N, requires_grad=True)
        override = torch.full((T, N), float("nan"))
        override[0, 0] = 1.5
        override.requires_grad_(True)

        adv = compute_gae(rewards, values, terminated, next_value,
                          gamma=0.99, lam=0.95,
                          next_value_override=override)
        assert not adv.requires_grad
        assert adv.grad_fn is None

        adv_gpu = compute_gae_gpu(rewards, values, terminated, next_value,
                                  gamma=0.99, lam=0.95,
                                  next_value_override=override)
        assert not adv_gpu.requires_grad
        assert adv_gpu.grad_fn is None


class TestGAEPaddedPerCellOverride:
    """R: per-cell next_value_override for compute_gae_padded(_gpu).

    Required for split-merge truncation handling: the per-env padded GAE
    must use V(terminal_observations) instead of values[t+1] (which is
    the next game's first learner-move state after auto-reset).

    Contract mirrors the (T, N) flat helpers: NaN cells fall back to the
    default next_vals (values[t+1] mid-sequence; next_values[i] at lengths[i]-1).
    Finite cells replace the bootstrap at that (t, env_index).
    """

    def test_padded_override_replaces_values_shift_mid_seq(self):
        from keisei.training.gae import compute_gae_padded
        T_max, N = 4, 1
        rewards = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        values = torch.tensor([[0.5], [0.6], [0.7], [0.8]])
        terminated = torch.zeros(T_max, N)
        next_values = torch.tensor([0.0])
        lengths = torch.tensor([4])

        adv_default = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=1.0, lam=0.0,
        )
        # delta[0] = 0 + 1.0 * values[1] - values[0] = 0.6 - 0.5 = 0.1
        assert abs(adv_default[0, 0].item() - 0.1) < 1e-5

        override = torch.full((T_max, N), float("nan"))
        override[0, 0] = 9.0
        adv_override = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=1.0, lam=0.0, next_value_override=override,
        )
        # delta[0] = 0 + 1.0 * 9.0 - 0.5 = 8.5
        assert abs(adv_override[0, 0].item() - 8.5) < 1e-5

    def test_padded_override_replaces_last_valid_step(self):
        """Override at (lengths[i]-1, i) must beat the next_values[i] stamp."""
        from keisei.training.gae import compute_gae_padded
        T_max, N = 4, 2
        rewards = torch.zeros(T_max, N)
        values = torch.tensor([[0.5, 0.5], [0.6, 0.6], [0.7, 0.0], [0.0, 0.0]])
        terminated = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        next_values = torch.tensor([0.0, 0.0])
        lengths = torch.tensor([3, 2])  # env 0: 3 valid, env 1: 2 valid

        adv_default = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=1.0, lam=0.0,
        )
        # env 0 step 2 (last valid): delta = 0 + 1.0 * next_values[0] - values[2,0]
        #                                 = 0 - 0.7 = -0.7
        assert abs(adv_default[2, 0].item() - (-0.7)) < 1e-5

        # Override env 0's last valid step with a finite bootstrap.
        override = torch.full((T_max, N), float("nan"))
        override[2, 0] = 4.0  # replaces next_values[0] at env 0's last step
        adv_override = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=1.0, lam=0.0, next_value_override=override,
        )
        # delta = 0 + 1.0 * 4.0 - 0.7 = 3.3
        assert abs(adv_override[2, 0].item() - 3.3) < 1e-5
        # env 1 untouched
        assert torch.allclose(adv_override[:, 1], adv_default[:, 1], atol=1e-5)

    def test_padded_override_partial_falls_back_for_nan(self):
        from keisei.training.gae import compute_gae_padded
        T_max, N = 3, 2
        torch.manual_seed(0)
        rewards = torch.randn(T_max, N)
        values = torch.randn(T_max, N)
        terminated = torch.zeros(T_max, N)
        next_values = torch.randn(N)
        lengths = torch.tensor([3, 3])

        baseline = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95,
        )
        override = torch.full((T_max, N), float("nan"))
        override[1, 0] = 42.0
        adv = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95, next_value_override=override,
        )
        assert torch.allclose(adv[:, 1], baseline[:, 1], atol=1e-5)
        assert not torch.isclose(adv[1, 0], baseline[1, 0])

    def test_padded_override_none_is_unchanged(self):
        from keisei.training.gae import compute_gae_padded
        T_max, N = 4, 2
        torch.manual_seed(11)
        rewards = torch.randn(T_max, N)
        values = torch.randn(T_max, N)
        terminated = torch.zeros(T_max, N)
        next_values = torch.randn(N)
        lengths = torch.tensor([4, 4])

        a = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95,
        )
        b = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95, next_value_override=None,
        )
        assert torch.allclose(a, b)

    def test_padded_gpu_override_matches_cpu(self):
        from keisei.training.gae import compute_gae_padded, compute_gae_padded_gpu
        T_max, N = 4, 2
        torch.manual_seed(3)
        rewards = torch.randn(T_max, N)
        values = torch.randn(T_max, N)
        terminated = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        next_values = torch.randn(N)
        lengths = torch.tensor([4, 3])
        override = torch.full((T_max, N), float("nan"))
        override[2, 1] = 1.25  # env 1 last valid step
        override[1, 0] = -0.5  # env 0 mid-sequence

        cpu = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=0.97, lam=0.9, next_value_override=override,
        )
        gpu = compute_gae_padded_gpu(
            rewards, values, terminated, next_values, lengths,
            gamma=0.97, lam=0.9, next_value_override=override,
        )
        assert torch.allclose(cpu, gpu, atol=1e-5)

    def test_padded_override_preserves_no_grad(self):
        from keisei.training.gae import compute_gae_padded, compute_gae_padded_gpu
        T_max, N = 3, 2
        rewards = torch.randn(T_max, N, requires_grad=True)
        values = torch.randn(T_max, N, requires_grad=True)
        terminated = torch.zeros(T_max, N)
        next_values = torch.randn(N, requires_grad=True)
        lengths = torch.tensor([3, 3])
        override = torch.full((T_max, N), float("nan"))
        override[0, 0] = 1.5
        override.requires_grad_(True)

        adv = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95, next_value_override=override,
        )
        assert not adv.requires_grad and adv.grad_fn is None
        adv_gpu = compute_gae_padded_gpu(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95, next_value_override=override,
        )
        assert not adv_gpu.requires_grad and adv_gpu.grad_fn is None

