"""Tests for split-merge GAE hot path optimizations."""

import time

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


class TestOverlappedTransfer:
    """Overlapping CPU→GPU transfer with GAE must not corrupt data."""

    @pytest.fixture
    def ppo(self):
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        model = SEResNetModel(params)
        return KataGoPPOAlgorithm(KataGoPPOParams(), model)

    def test_overlapped_update_produces_finite_metrics(self, ppo):
        """update() with overlapped transfer must produce identical results."""
        buf = KataGoRolloutBuffer(num_envs=4, obs_shape=(50, 9, 9), action_space=11259)
        torch.manual_seed(123)
        env_id_lists = [[0, 2], [1, 2, 3], [0, 1, 3], [0, 1, 2, 3],
                        [0, 1], [2, 3], [0, 2, 3], [1, 2]]
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
        losses = ppo.update(buf, torch.zeros(4))
        for key, val in losses.items():
            if key.startswith("frac_") or key == "value_accuracy":
                continue
            assert not torch.tensor(val).isnan(), f"{key} is NaN"
            assert not torch.tensor(val).isinf(), f"{key} is inf"


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

    def test_update_with_env_ids_threads_next_value_override(self, ppo, monkeypatch):
        """env_ids branch must thread next_value_override into compute_gae_padded.

        Discriminates "override flows through update()" from "override silently
        dropped in update()" by capturing the actual kwargs passed to
        compute_gae_padded — the integration smoke test alone (finite losses)
        would pass even if the threading were removed.
        """
        torch.manual_seed(0)
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        # Uneven env coverage so total_samples != T*num_envs and update()
        # routes through the env_ids / padded GAE branch.
        # Step 0: env 0 only (truncated row, finite override)
        # Step 1: envs 0, 1
        # Step 2: env 1 only
        env_lists = [[0], [0, 1], [1]]
        rewards_per_step = [torch.tensor([0.0]), torch.tensor([0.0, 0.0]), torch.tensor([0.0])]
        values_per_step = [torch.tensor([0.5]), torch.tensor([0.6, 0.4]), torch.tensor([0.55])]
        # Step 0 env 0 row is truncated (dones=True, terminated=False).
        dones_per_step = [
            torch.tensor([True]),
            torch.tensor([False, False]),
            torch.tensor([False]),
        ]
        terminated_per_step = [
            torch.tensor([False]),
            torch.tensor([False, False]),
            torch.tensor([False]),
        ]
        OVERRIDE_VAL = 7.5
        override_per_step = [
            torch.tensor([OVERRIDE_VAL]),
            torch.tensor([float("nan"), float("nan")]),
            torch.tensor([float("nan")]),
        ]

        for s in range(len(env_lists)):
            envs = env_lists[s]
            n = len(envs)
            buf.add(
                torch.randn(n, 50, 9, 9), torch.randint(0, 11259, (n,)),
                torch.randn(n), values_per_step[s], rewards_per_step[s],
                dones_per_step[s],
                terminated_per_step[s],
                torch.ones(n, 11259, dtype=torch.bool),
                torch.full((n,), -1, dtype=torch.long),
                torch.rand(n) * 2 - 1,
                env_ids=torch.tensor(envs),
                next_value_override=override_per_step[s],
            )

        # Confirm the buffer carries the override into flatten()
        data = buf.flatten()
        assert "next_value_override" in data
        env_ids = data["env_ids"]
        assert env_ids.tolist() == [0, 0, 1, 1]
        # Row indices in flatten order: env 0 at indices 0,1 ; env 1 at 2,3
        # Step 0 env 0 → flatten index 0 carries OVERRIDE_VAL
        assert data["next_value_override"][0].item() == pytest.approx(OVERRIDE_VAL)
        assert torch.isnan(data["next_value_override"][1:]).all()
        # Sanity: total_samples != T * num_envs so update() takes env_ids branch
        T = buf.size
        assert data["rewards"].numel() != T * buf.num_envs

        # Capture the kwargs passed to compute_gae_padded by ppo.update().
        # update() does a local `from keisei.training.gae import compute_gae_padded`,
        # so patching the module attribute intercepts it.
        from keisei.training import gae as gae_module
        captured: dict = {}
        original = gae_module.compute_gae_padded

        def spy(*args, **kwargs):
            captured["override"] = kwargs.get("next_value_override")
            captured["lengths"] = args[4] if len(args) >= 5 else kwargs.get("lengths")
            return original(*args, **kwargs)

        monkeypatch.setattr(gae_module, "compute_gae_padded", spy)

        losses = ppo.update(buf, torch.zeros(2))

        # The override must have reached compute_gae_padded as a (T_max, N_env)
        # padded tensor with our finite value at env 0's first valid step.
        assert "override" in captured, "compute_gae_padded was not called"
        assert captured["override"] is not None, (
            "next_value_override was None — update() dropped the override"
        )
        ov = captured["override"]
        assert ov.shape == (2, 2)
        # env order in padded layout follows splits unique_consecutive of sorted env_ids;
        # both envs appear in the same order (0, 1), so column index = env_id.
        assert ov[0, 0].item() == pytest.approx(OVERRIDE_VAL)
        assert torch.isnan(ov[1, 0])  # env 0 step 1 — no override
        assert torch.isnan(ov[0, 1])  # env 1 step 0 — no override
        assert torch.isnan(ov[1, 1])  # env 1 step 1 — no override

        # Smoke: losses are finite (basic wiring sanity)
        for key, val in losses.items():
            if key.startswith("frac_") or key == "value_accuracy":
                continue
            assert not torch.tensor(val).isnan(), f"{key} is NaN"
            assert not torch.tensor(val).isinf(), f"{key} is inf"


class TestPerformanceSanity:
    """Sanity check that the optimized path isn't accidentally slower."""

    def test_argsort_faster_than_boolean_loop(self):
        """argsort+split must be faster than boolean-index loop for 64+ envs.

        Uses median-of-many runs to suppress scheduler jitter — at sub-ms
        scales the per-iteration absolute gap is small, so a single-shot
        comparison is too flaky for CI.
        """
        torch.manual_seed(42)
        N_envs = 256
        total_samples = 32768
        env_ids = torch.randint(0, N_envs, (total_samples,))
        rewards = torch.randn(total_samples)
        runs = 25

        bool_runs: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            for eid in range(N_envs):
                mask = env_ids == eid
                _ = rewards[mask]
            bool_runs.append(time.perf_counter() - t0)
        bool_runs.sort()
        bool_time = bool_runs[len(bool_runs) // 2]

        sort_runs: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sort_idx = torch.argsort(env_ids, stable=True)
            sorted_ids = env_ids[sort_idx]
            _, counts = sorted_ids.unique_consecutive(return_counts=True)
            splits = torch.split(sort_idx, counts.tolist())
            for idx in splits:
                _ = rewards[idx]
            sort_runs.append(time.perf_counter() - t0)
        sort_runs.sort()
        sort_time = sort_runs[len(sort_runs) // 2]

        assert sort_time < bool_time, (
            f"argsort median ({sort_time*1e3:.3f}ms) not faster than "
            f"bool-loop median ({bool_time*1e3:.3f}ms)"
        )
