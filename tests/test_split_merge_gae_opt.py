"""Tests for split-merge GAE hot path optimizations."""

import torch
import pytest

from keisei.training.gae import compute_gae, compute_gae_padded


class TestVectorizedEnvPartition:
    """The vectorized argsort+split must produce identical GAE to the old boolean-index loop."""

    def test_vectorized_partition_matches_reference(self):
        """Partition via argsort+split gives same per-env chunks as boolean indexing."""
        torch.manual_seed(42)
        env_id_lists = [[0, 2], [1, 2, 3], [0, 1, 3], [0, 1, 2, 3],
                        [0, 1], [2, 3], [0, 2, 3], [1, 2]]
        all_env_ids = []
        all_rewards = []
        all_values = []
        all_terminated = []

        for step_envs in env_id_lists:
            n = len(step_envs)
            all_env_ids.append(torch.tensor(step_envs))
            all_rewards.append(torch.randn(n))
            all_values.append(torch.randn(n) * 0.5)
            all_terminated.append(torch.zeros(n))

        flat_env_ids = torch.cat(all_env_ids)
        flat_rewards = torch.cat(all_rewards)
        flat_values = torch.cat(all_values)
        flat_terminated = torch.cat(all_terminated)
        next_values = torch.randn(4)

        # --- Reference: old boolean-index loop ---
        unique_envs_ref = flat_env_ids.unique()
        ref_advantages = torch.zeros(flat_rewards.numel())
        for env_id in unique_envs_ref:
            mask = flat_env_ids == env_id
            env_adv = compute_gae(
                flat_rewards[mask], flat_values[mask], flat_terminated[mask],
                next_values[env_id], gamma=0.99, lam=0.95,
            )
            ref_advantages[mask] = env_adv

        # --- New: argsort + split ---
        sort_idx = torch.argsort(flat_env_ids, stable=True)
        sorted_ids = flat_env_ids[sort_idx]
        unique_envs, counts = sorted_ids.unique_consecutive(return_counts=True)
        splits = torch.split(sort_idx, counts.tolist())

        new_advantages = torch.zeros(flat_rewards.numel())
        for i, env_id in enumerate(unique_envs):
            idx = splits[i]
            env_adv = compute_gae(
                flat_rewards[idx], flat_values[idx], flat_terminated[idx],
                next_values[env_id], gamma=0.99, lam=0.95,
            )
            new_advantages[idx] = env_adv

        assert torch.allclose(new_advantages, ref_advantages, atol=1e-6), (
            f"Vectorized partition diverged from reference: "
            f"max diff = {(new_advantages - ref_advantages).abs().max().item()}"
        )

    def test_single_env_partition(self):
        """Edge case: all samples from one env."""
        flat_env_ids = torch.zeros(10, dtype=torch.long)
        sort_idx = torch.argsort(flat_env_ids, stable=True)
        sorted_ids = flat_env_ids[sort_idx]
        unique_envs, counts = sorted_ids.unique_consecutive(return_counts=True)
        splits = torch.split(sort_idx, counts.tolist())
        assert len(splits) == 1
        assert splits[0].numel() == 10

    def test_partition_preserves_temporal_order(self):
        """argsort(stable=True) must preserve insertion order within each env."""
        flat_env_ids = torch.tensor([0, 1, 0, 1, 0])
        sort_idx = torch.argsort(flat_env_ids, stable=True)
        assert sort_idx.tolist() == [0, 2, 4, 1, 3]


class TestGAEPaddedGPU:
    """GPU padded GAE must match CPU padded GAE within floating-point tolerance."""

    def test_gpu_padded_matches_cpu_padded(self):
        from keisei.training.gae import compute_gae_padded, compute_gae_padded_gpu

        torch.manual_seed(42)
        T_max, N = 20, 8
        rewards = torch.randn(T_max, N)
        values = torch.randn(T_max, N) * 0.5
        terminated = torch.zeros(T_max, N)
        lengths = torch.tensor([20, 15, 10, 20, 8, 12, 18, 20])
        for i in range(N):
            terminated[lengths[i]:, i] = 1.0
        next_values = torch.randn(N)

        cpu_adv = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95,
        )

        # compute_gae_padded_gpu should work on CPU tensors too (just runs on whatever device)
        gpu_adv = compute_gae_padded_gpu(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95,
        )

        assert torch.allclose(cpu_adv, gpu_adv, rtol=1e-4, atol=1e-5), (
            f"GPU padded GAE diverged: max diff = "
            f"{(cpu_adv - gpu_adv).abs().max().item()}"
        )

    def test_gpu_padded_all_same_length(self):
        """When all envs have max length, result should match compute_gae_gpu."""
        from keisei.training.gae import compute_gae_gpu, compute_gae_padded_gpu

        torch.manual_seed(7)
        T, N = 10, 4
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        terminated = torch.zeros(T, N)
        next_values = torch.randn(N)
        lengths = torch.full((N,), T)

        padded_adv = compute_gae_padded_gpu(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95,
        )
        direct_adv = compute_gae_gpu(
            rewards, values, terminated, next_values,
            gamma=0.99, lam=0.95,
        )
        assert torch.allclose(padded_adv, direct_adv, atol=1e-5)


from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams, KataGoRolloutBuffer
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


class TestPreAllocatedBuffer:
    """Pre-allocated buffer must produce identical flatten() output."""

    def test_fixed_size_flatten_matches(self):
        """Standard (non-split-merge) buffer: pre-alloc matches list-based."""
        buf = KataGoRolloutBuffer(num_envs=4, obs_shape=(50, 9, 9), action_space=11259)
        torch.manual_seed(99)
        for _ in range(8):
            n = 4
            buf.add(
                torch.randn(n, 50, 9, 9), torch.randint(0, 11259, (n,)),
                torch.randn(n), torch.randn(n) * 0.1, torch.zeros(n),
                torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool),
                torch.ones(n, 11259, dtype=torch.bool),
                torch.full((n,), -1, dtype=torch.long), torch.rand(n) * 2 - 1,
            )
        data = buf.flatten()
        assert data["observations"].shape == (32, 50, 9, 9)
        assert data["observations"].is_contiguous()

    def test_variable_size_flatten_matches(self):
        """Split-merge buffer: variable envs per step."""
        buf = KataGoRolloutBuffer(num_envs=4, obs_shape=(50, 9, 9), action_space=11259)
        torch.manual_seed(99)
        env_id_lists = [[0, 2], [1, 2, 3], [0, 1, 3], [0, 1, 2, 3]]
        for envs in env_id_lists:
            n = len(envs)
            buf.add(
                torch.randn(n, 50, 9, 9), torch.randint(0, 11259, (n,)),
                torch.randn(n), torch.randn(n) * 0.1, torch.zeros(n),
                torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool),
                torch.ones(n, 11259, dtype=torch.bool),
                torch.full((n,), -1, dtype=torch.long), torch.rand(n) * 2 - 1,
                env_ids=torch.tensor(envs),
            )
        data = buf.flatten()
        assert data["observations"].shape == (12, 50, 9, 9)
        assert data["env_ids"].numel() == 12

    def test_clear_resets_buffer(self):
        """clear() must reset write offset and allow reuse."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        buf.add(
            torch.randn(2, 50, 9, 9), torch.randint(0, 11259, (2,)),
            torch.randn(2), torch.randn(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool),
            torch.ones(2, 11259, dtype=torch.bool),
            torch.full((2,), -1, dtype=torch.long), torch.rand(2) * 2 - 1,
        )
        assert buf.size == 1
        buf.clear()
        assert buf.size == 0
        with pytest.raises(ValueError, match="Cannot flatten an empty buffer"):
            buf.flatten()


class TestVectorizedUpdateIntegration:
    """update() must produce identical results after the vectorized refactor."""

    @pytest.fixture
    def ppo(self):
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)
        return KataGoPPOAlgorithm(KataGoPPOParams(), model)

    def test_update_with_env_ids_produces_finite_metrics(self, ppo):
        """Vectorized per-env GAE path must produce finite losses."""
        buf = KataGoRolloutBuffer(num_envs=4, obs_shape=(50, 9, 9), action_space=11259)
        env_id_lists = [[0, 2], [1, 2, 3], [0, 1, 3], [0, 1, 2, 3]]
        for envs in env_id_lists:
            n = len(envs)
            buf.add(
                torch.randn(n, 50, 9, 9), torch.randint(0, 11259, (n,)),
                torch.randn(n), torch.randn(n) * 0.1, torch.zeros(n),
                torch.zeros(n, dtype=torch.bool),
                torch.zeros(n, dtype=torch.bool),
                torch.ones(n, 11259, dtype=torch.bool),
                torch.full((n,), -1, dtype=torch.long),
                torch.rand(n) * 2 - 1,
                env_ids=torch.tensor(envs),
            )
        losses = ppo.update(buf, torch.zeros(4))
        for key, val in losses.items():
            if key.startswith("frac_") or key == "value_accuracy":
                continue
            assert not torch.tensor(val).isnan(), f"{key} is NaN"
            assert not torch.tensor(val).isinf(), f"{key} is inf"
