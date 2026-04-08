# Split-Merge GAE Hot Path Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 10-20 second GPU stall at the start of each PPO update in split-merge (league) mode by vectorizing the per-env GAE path, running GAE on GPU, and overlapping computation with data transfer.

**Architecture:** The split-merge GAE path in `KataGoPPOAlgorithm.update()` currently uses an O(envs × samples) Python loop with boolean indexing to partition data by environment, then computes GAE on CPU. We replace this with: (1) a single `argsort` + `split` to partition in O(N log N), (2) GPU-accelerated padded GAE via the existing `compute_gae_gpu`, (3) pre-allocated contiguous buffer storage instead of list-of-tensors + `torch.cat`, and (4) overlapped CPU→GPU transfer of observations while GAE computes.

**Tech Stack:** PyTorch (tensors, CUDA streams), Python 3.13

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `keisei/training/katago_ppo.py:560-623` | Replace per-env loop with vectorized argsort+split, GPU GAE path |
| Modify | `keisei/training/katago_ppo.py:128-260` | Pre-allocated buffer with contiguous storage |
| Modify | `keisei/training/katago_ppo.py:627-639` | Overlap obs transfer with GAE via CUDA streams |
| Modify | `keisei/training/gae.py:54-110` | Add `compute_gae_padded_gpu` variant |
| Create | `tests/test_split_merge_gae_opt.py` | All tests for this optimization |

---

## Task 1: Vectorize the Per-Env Loop with argsort+split

**Files:**
- Create: `tests/test_split_merge_gae_opt.py`
- Modify: `keisei/training/katago_ppo.py:560-610`

This is the single highest-impact fix. The current code at `katago_ppo.py:576-582` runs a Python loop over every unique env ID, doing a full-buffer boolean comparison each iteration — O(envs × total_samples). We replace it with a single `argsort` + `unique_consecutive` + `split` that partitions all fields in O(N log N).

- [ ] **Step 1: Write failing test — vectorized partition matches boolean-index reference**

Create `tests/test_split_merge_gae_opt.py`:

```python
"""Tests for split-merge GAE hot path optimizations."""

import torch
import pytest

from keisei.training.gae import compute_gae, compute_gae_padded


class TestVectorizedEnvPartition:
    """The vectorized argsort+split must produce identical GAE to the old boolean-index loop."""

    def test_vectorized_partition_matches_reference(self):
        """Partition via argsort+split gives same per-env chunks as boolean indexing."""
        torch.manual_seed(42)
        # Simulate split-merge: 8 steps, variable envs per step (4 possible envs)
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
        # Env 0 indices: 0, 2, 4 (original order preserved)
        # Env 1 indices: 1, 3
        assert sort_idx.tolist() == [0, 2, 4, 1, 3]
```

- [ ] **Step 2: Run test to verify it passes (this is a self-contained reference test)**

Run: `uv run pytest tests/test_split_merge_gae_opt.py -v`

Expected: PASS (this test exercises the algorithm in isolation, not the production code)

- [ ] **Step 3: Write failing integration test — update() with env_ids uses vectorized path**

Append to `tests/test_split_merge_gae_opt.py`:

```python
from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams, KataGoRolloutBuffer
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


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
```

- [ ] **Step 4: Run integration test to verify it passes with current code**

Run: `uv run pytest tests/test_split_merge_gae_opt.py::TestVectorizedUpdateIntegration -v`

Expected: PASS (current code handles env_ids correctly, just slowly)

- [ ] **Step 5: Replace the boolean-index loop in `katago_ppo.py` with argsort+split**

In `keisei/training/katago_ppo.py`, replace lines 560-610 (the `elif "env_ids" in data:` block) with:

```python
        elif "env_ids" in data:
            # Per-env GAE for split-merge mode: partition by env via argsort
            # (O(N log N)) instead of per-env boolean indexing (O(envs × N)).
            from keisei.training.gae import compute_gae_padded

            env_ids = data["env_ids"]

            # Single sort partitions all envs; stable=True preserves temporal order.
            sort_idx = torch.argsort(env_ids, stable=True)
            sorted_ids = env_ids[sort_idx]
            unique_envs, counts = sorted_ids.unique_consecutive(return_counts=True)
            splits = torch.split(sort_idx, counts.tolist())

            N_env = len(unique_envs)
            env_lengths = counts.tolist()
            max_T = max(env_lengths)

            rewards_pad = torch.zeros(max_T, N_env)
            values_pad = torch.zeros(max_T, N_env)
            terminated_pad = torch.ones(max_T, N_env)  # padding = done to zero GAE
            nv = torch.zeros(N_env)

            if unique_envs.max() >= next_values_cpu.shape[0]:
                raise IndexError(
                    f"env_id {unique_envs.max().item()} >= next_values size "
                    f"{next_values_cpu.shape[0]}"
                )
            for i, idx in enumerate(splits):
                L = env_lengths[i]
                rewards_pad[:L, i] = data["rewards"][idx]
                values_pad[:L, i] = data["values"][idx]
                terminated_pad[:L, i] = data[gae_dones_key][idx]
                nv[i] = next_values_cpu[unique_envs[i]]

            lengths_t = torch.tensor(env_lengths)
            padded_adv = compute_gae_padded(
                rewards_pad, values_pad, terminated_pad, nv, lengths_t,
                gamma=self.params.gamma, lam=self.params.gae_lambda,
            )

            advantages = torch.zeros(total_samples)
            for i, idx in enumerate(splits):
                L = env_lengths[i]
                advantages[idx] = padded_adv[:L, i]
```

The key changes:
- Removed the inner `mask = env_ids == env_id` loop (was O(envs × N))
- `splits` gives us pre-computed index tensors per env from a single sort
- The fill loop uses integer indexing (`data["rewards"][idx]`) instead of boolean masks
- `env_masks` list is eliminated — we scatter back via `splits[i]` directly

- [ ] **Step 6: Run all per-env GAE tests**

Run: `uv run pytest tests/test_split_merge_gae_opt.py tests/test_katago_ppo.py::TestPerEnvGAE -v`

Expected: All PASS

- [ ] **Step 7: Run full PPO test suite for regressions**

Run: `uv run pytest tests/test_katago_ppo.py tests/test_gae.py tests/test_gae_batched.py -v`

Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add tests/test_split_merge_gae_opt.py keisei/training/katago_ppo.py
git commit -m "perf: vectorize split-merge env partition with argsort+split

Replace O(envs × N) boolean-index loop with O(N log N) argsort +
unique_consecutive + split. For 256 envs × 65K samples this reduces
the partition step from ~5s to <0.1s."
```

---

## Task 2: GPU GAE for Split-Merge Mode

**Files:**
- Modify: `keisei/training/gae.py:54-110` (add `compute_gae_padded_gpu`)
- Modify: `keisei/training/katago_ppo.py` (the `elif "env_ids"` block from Task 1)
- Modify: `tests/test_split_merge_gae_opt.py`

The padded GAE currently runs on CPU with a Python loop of ~512 iterations over `(max_T, N_env)` tensors. We add a GPU variant that pads on CPU (cheap) then runs the backward scan on GPU (fast — each iteration is a fused kernel over N envs). This reuses the pattern from `compute_gae_gpu` but accepts variable-length sequences via a padding mask.

- [ ] **Step 1: Write failing test — GPU padded GAE matches CPU padded GAE**

Append to `tests/test_split_merge_gae_opt.py`:

```python
class TestGAEPaddedGPU:
    """GPU padded GAE must match CPU padded GAE within floating-point tolerance."""

    def test_gpu_padded_matches_cpu_padded(self):
        from keisei.training.gae import compute_gae_padded, compute_gae_padded_gpu

        torch.manual_seed(42)
        T_max, N = 20, 8
        rewards = torch.randn(T_max, N)
        values = torch.randn(T_max, N) * 0.5
        terminated = torch.zeros(T_max, N)
        # Make some episodes shorter: set padding positions to terminated=1
        lengths = torch.tensor([20, 15, 10, 20, 8, 12, 18, 20])
        for i in range(N):
            terminated[lengths[i]:, i] = 1.0
        next_values = torch.randn(N)

        cpu_adv = compute_gae_padded(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95,
        )

        gpu_adv = compute_gae_padded_gpu(
            rewards.cuda(), values.cuda(), terminated.cuda(),
            next_values.cuda(), lengths, gamma=0.99, lam=0.95,
        )

        assert torch.allclose(cpu_adv, gpu_adv.cpu(), rtol=1e-4, atol=1e-5), (
            f"GPU padded GAE diverged: max diff = "
            f"{(cpu_adv - gpu_adv.cpu()).abs().max().item()}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gpu_padded_output_on_gpu(self):
        """Output tensor must be on the same device as input."""
        from keisei.training.gae import compute_gae_padded_gpu

        T_max, N = 5, 2
        rewards = torch.randn(T_max, N, device="cuda")
        values = torch.randn(T_max, N, device="cuda")
        terminated = torch.zeros(T_max, N, device="cuda")
        next_values = torch.randn(N, device="cuda")
        lengths = torch.tensor([5, 3])

        adv = compute_gae_padded_gpu(
            rewards, values, terminated, next_values, lengths,
            gamma=0.99, lam=0.95,
        )
        assert adv.device.type == "cuda"

    def test_gpu_padded_all_same_length(self):
        """When all envs have max length, result should match compute_gae_gpu."""
        from keisei.training.gae import compute_gae_gpu, compute_gae_padded_gpu

        torch.manual_seed(7)
        T, N = 10, 4
        rewards = torch.randn(T, N, device="cpu")
        values = torch.randn(T, N, device="cpu")
        terminated = torch.zeros(T, N, device="cpu")
        next_values = torch.randn(N, device="cpu")
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_merge_gae_opt.py::TestGAEPaddedGPU -v`

Expected: FAIL with `ImportError: cannot import name 'compute_gae_padded_gpu'`

- [ ] **Step 3: Implement `compute_gae_padded_gpu` in `gae.py`**

Add after `compute_gae_gpu` in `keisei/training/gae.py`:

```python
def compute_gae_padded_gpu(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    next_values: torch.Tensor,
    lengths: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GPU GAE for variable-length per-env sequences via padding.

    Same interface as compute_gae_padded() but runs the backward scan on the
    input device (typically CUDA). Padding positions must have terminated=1.0
    so that not_done=0 zeroes out GAE propagation through padding.

    The lengths tensor stays on CPU (used only for the next_vals override loop,
    which is O(N) scalar assignments — cheaper than a GPU kernel launch).

    Args:
        rewards: (T_max, N) padded rewards, on target device
        values: (T_max, N) padded value estimates, on target device
        terminated: (T_max, N) termination flags (padding=1.0), on target device
        next_values: (N,) bootstrap values per env, on target device
        lengths: (N,) actual sequence length per env (CPU tensor is fine)
        gamma: discount factor
        lam: GAE lambda

    Returns:
        (T_max, N) advantages on the same device as inputs
    """
    if rewards.ndim != 2:
        raise ValueError(
            f"compute_gae_padded_gpu only supports 2D (T_max, N) input, got shape {rewards.shape}"
        )

    T_max, N = rewards.shape
    compute_dtype = values.dtype
    rewards = rewards.to(dtype=compute_dtype)
    device = rewards.device

    # Build next_vals: shift values by 1, override last valid step per env
    next_vals = torch.zeros_like(rewards)
    next_vals[:-1] = values[1:]
    next_vals[-1] = next_values

    # Override each env's last valid step with its bootstrap value.
    # This is O(N) scalar writes — negligible vs the T_max backward scan.
    last_step_idx = (lengths - 1).clamp(min=0).cpu().numpy().astype(int)
    for i in range(N):
        next_vals[last_step_idx[i], i] = next_values[i]

    not_done = 1.0 - terminated.float()
    delta = rewards + gamma * next_vals * not_done - values
    decay = gamma * lam * not_done

    # Sequential backward scan — each step is a fused GPU kernel over N envs.
    advantages = torch.empty_like(rewards)
    last_gae = torch.zeros(N, device=device, dtype=compute_dtype)
    for t in reversed(range(T_max)):
        last_gae = delta[t] + decay[t] * last_gae
        advantages[t] = last_gae

    return advantages
```

- [ ] **Step 4: Run GPU padded GAE tests**

Run: `uv run pytest tests/test_split_merge_gae_opt.py::TestGAEPaddedGPU -v`

Expected: PASS (skip CUDA tests gracefully if no GPU in CI)

- [ ] **Step 5: Wire GPU padded GAE into the split-merge path**

In `keisei/training/katago_ppo.py`, update the `elif "env_ids" in data:` block. Replace the `compute_gae_padded` call and the preceding import with a GPU-aware branch:

Find this code (from Task 1):
```python
            lengths_t = torch.tensor(env_lengths)
            padded_adv = compute_gae_padded(
                rewards_pad, values_pad, terminated_pad, nv, lengths_t,
                gamma=self.params.gamma, lam=self.params.gae_lambda,
            )
```

Replace with:
```python
            lengths_t = torch.tensor(env_lengths)
            if device.type == "cuda":
                from keisei.training.gae import compute_gae_padded_gpu
                padded_adv = compute_gae_padded_gpu(
                    rewards_pad.to(device), values_pad.to(device),
                    terminated_pad.to(device), next_values.detach().float(),
                    lengths_t, gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).cpu()
            else:
                padded_adv = compute_gae_padded(
                    rewards_pad, values_pad, terminated_pad, nv, lengths_t,
                    gamma=self.params.gamma, lam=self.params.gae_lambda,
                )
```

Also update the import at the top of the block. The `from keisei.training.gae import compute_gae_padded` on line 563 is still needed for the CPU fallback. Keep it.

Note: we pass `next_values.detach().float()` (still on GPU from the bootstrap forward pass) instead of `nv` (CPU). The padded tensors are small (~512 × 256 × 4 bytes = 512 KB each) so the transfer is negligible.

- [ ] **Step 6: Run all split-merge and GAE tests**

Run: `uv run pytest tests/test_split_merge_gae_opt.py tests/test_katago_ppo.py::TestPerEnvGAE tests/test_gae.py tests/test_gae_batched.py -v`

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add keisei/training/gae.py keisei/training/katago_ppo.py tests/test_split_merge_gae_opt.py
git commit -m "perf: GPU padded GAE for split-merge mode

Add compute_gae_padded_gpu() — same backward scan as compute_gae_gpu
but handles variable-length per-env sequences via padding. Wire into
the split-merge update path so the ~512-step backward scan runs as
fused GPU kernels instead of CPU Python loops."
```

---

## Task 3: Pre-Allocated Contiguous Buffer

**Files:**
- Modify: `keisei/training/katago_ppo.py:128-260` (`KataGoRolloutBuffer`)
- Modify: `tests/test_split_merge_gae_opt.py`

The current buffer stores each timestep as a separate tensor in a Python list, then concatenates them all in `flatten()`. For 512 steps × 256 envs, the observations `torch.cat` alone allocates ~1 GB and copies 512 tensors. We replace this with a pre-allocated contiguous tensor that `add()` writes into directly, making `flatten()` a zero-copy slice.

The challenge: in split-merge mode, the number of envs per step varies. We handle this by allocating for the maximum possible (num_envs per step × max_steps) and tracking the actual write offset. `flatten()` returns a slice up to the offset.

- [ ] **Step 1: Write failing test — pre-allocated buffer flatten matches list-based**

Append to `tests/test_split_merge_gae_opt.py`:

```python
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
        # Observations should be contiguous and correct shape
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
        # Total samples: 2 + 3 + 3 + 4 = 12
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
```

- [ ] **Step 2: Run test to verify it passes with current implementation**

Run: `uv run pytest tests/test_split_merge_gae_opt.py::TestPreAllocatedBuffer -v`

Expected: PASS (testing current behavior before refactoring)

- [ ] **Step 3: Refactor `KataGoRolloutBuffer` to use pre-allocated contiguous storage**

Replace the `KataGoRolloutBuffer` class in `keisei/training/katago_ppo.py:128-260` with:

```python
class KataGoRolloutBuffer:
    def __init__(self, num_envs: int, obs_shape: tuple[int, ...], action_space: int) -> None:
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_space = action_space
        # Pre-allocate for up to _alloc_steps timesteps. Grows if needed.
        self._alloc_steps = 0
        self._alloc_samples = 0
        self._write_offset = 0  # sample-level offset (not step count)
        self._step_count = 0
        self._storage: dict[str, torch.Tensor] = {}
        self._has_env_ids = False

    def _ensure_capacity(self, n_samples: int) -> None:
        """Grow storage if needed to fit n_samples more entries."""
        needed = self._write_offset + n_samples
        if needed <= self._alloc_samples:
            return

        # Double capacity (minimum 512 * num_envs to avoid repeated reallocs)
        new_cap = max(needed * 2, 512 * self.num_envs)
        new_storage: dict[str, torch.Tensor] = {
            "observations": torch.empty(new_cap, *self.obs_shape),
            "actions": torch.empty(new_cap, dtype=torch.long),
            "log_probs": torch.empty(new_cap),
            "values": torch.empty(new_cap),
            "rewards": torch.empty(new_cap),
            "dones": torch.empty(new_cap, dtype=torch.bool),
            "terminated": torch.empty(new_cap, dtype=torch.bool),
            "legal_masks": torch.empty(new_cap, self.action_space, dtype=torch.bool),
            "value_categories": torch.empty(new_cap, dtype=torch.long),
            "score_targets": torch.empty(new_cap),
        }
        if self._has_env_ids:
            new_storage["env_ids"] = torch.empty(new_cap, dtype=torch.long)

        # Copy existing data
        if self._write_offset > 0:
            off = self._write_offset
            for key in new_storage:
                if key in self._storage:
                    new_storage[key][:off] = self._storage[key][:off]

        self._storage = new_storage
        self._alloc_samples = new_cap

    @property
    def size(self) -> int:
        return self._step_count

    def clear(self) -> None:
        self._write_offset = 0
        self._step_count = 0
        # Keep storage allocated for reuse — just reset the write head.
        # This avoids re-allocation every epoch.

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        terminated: torch.Tensor,
        legal_masks: torch.Tensor,
        value_categories: torch.Tensor,
        score_targets: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Add a timestep to the buffer.

        Args:
            score_targets: Pre-normalized score estimates in [-1, 1]. The caller
                (KataGoTrainingLoop) divides raw material difference by
                KataGoPPOParams.score_normalization before storing here.
                Raw scores can range from -200 to +200; without normalization,
                the MSE loss would dominate all other loss terms.
        """
        # Detach and move to CPU first — all validation below runs on CPU
        # tensors, avoiding implicit CUDA synchronization on the hot path.
        obs_cpu = obs.detach().cpu()
        actions_cpu = actions.detach().cpu()
        log_probs_cpu = log_probs.detach().cpu()
        values_cpu = values.detach().cpu()
        rewards_cpu = rewards.detach().cpu()
        dones_cpu = dones.detach().cpu()
        terminated_cpu = terminated.detach().cpu()

        # Guard: terminated must be a subset of dones
        if (terminated_cpu.bool() & ~dones_cpu.bool()).any():
            raise AssertionError(
                "terminated must be a subset of dones: every terminated position must also be done. "
                "Got terminated=True where dones=False — likely a call site passing the merged signal."
            )
        legal_masks_cpu = legal_masks.detach().cpu()
        value_cats_cpu = value_categories.detach().cpu()
        score_cpu = score_targets.detach().cpu()

        # Guard against invalid value categories
        valid_cats = {-1, 0, 1, 2}
        unique_cats = set(value_cats_cpu.unique().tolist())
        invalid = unique_cats - valid_cats
        if invalid:
            raise ValueError(
                f"value_categories contains invalid values {invalid}. "
                f"Expected only {{-1=ignore, 0=W, 1=D, 2=L}}."
            )

        # Guard: NaN score targets are no longer valid
        if score_cpu.isnan().any():
            raise ValueError(
                "score_targets contains NaN. With per-step material balance, "
                "all targets should be real-valued."
            )

        # Guard against unnormalized score targets
        abs_max = score_cpu.abs().max()
        if abs_max > 3.5:
            raise ValueError(
                f"score_targets appear unnormalized: max abs value = "
                f"{abs_max.item():.1f}. "
                f"Expected in [-1.7, +1.7] typical, theoretical max 2.58 (guard 3.5)."
            )

        n = obs_cpu.shape[0]

        # Track env_ids presence on first add
        if self._step_count == 0 and env_ids is not None:
            self._has_env_ids = True

        self._ensure_capacity(n)
        off = self._write_offset

        self._storage["observations"][off:off + n] = obs_cpu
        self._storage["actions"][off:off + n] = actions_cpu
        self._storage["log_probs"][off:off + n] = log_probs_cpu
        self._storage["values"][off:off + n] = values_cpu
        self._storage["rewards"][off:off + n] = rewards_cpu
        self._storage["dones"][off:off + n] = dones_cpu
        self._storage["terminated"][off:off + n] = terminated_cpu
        self._storage["legal_masks"][off:off + n] = legal_masks_cpu
        self._storage["value_categories"][off:off + n] = value_cats_cpu
        self._storage["score_targets"][off:off + n] = score_cpu
        if env_ids is not None:
            if "env_ids" not in self._storage:
                # Late init — first add didn't have env_ids but a later one does
                self._storage["env_ids"] = torch.empty(self._alloc_samples, dtype=torch.long)
            self._storage["env_ids"][off:off + n] = env_ids.detach().cpu()

        self._write_offset = off + n
        self._step_count += 1

    def flatten(self) -> dict[str, torch.Tensor]:
        if self._step_count == 0:
            raise ValueError(
                "Cannot flatten an empty buffer. Call add() at least once before flatten()."
            )
        off = self._write_offset
        result = {
            "observations": self._storage["observations"][:off].reshape(-1, *self.obs_shape),
            "actions": self._storage["actions"][:off].reshape(-1),
            "log_probs": self._storage["log_probs"][:off].reshape(-1),
            "values": self._storage["values"][:off].reshape(-1),
            "rewards": self._storage["rewards"][:off].reshape(-1),
            "dones": self._storage["dones"][:off].reshape(-1),
            "terminated": self._storage["terminated"][:off].reshape(-1),
            "legal_masks": self._storage["legal_masks"][:off].reshape(-1, self.action_space),
            "value_categories": self._storage["value_categories"][:off].reshape(-1),
            "score_targets": self._storage["score_targets"][:off].reshape(-1),
        }
        if self._has_env_ids and "env_ids" in self._storage:
            result["env_ids"] = self._storage["env_ids"][:off].reshape(-1)
        return result
```

Key changes:
- `add()` writes directly into pre-allocated contiguous tensors via slice assignment
- `flatten()` is a zero-copy slice (just adjusts tensor metadata, no data copy)
- `clear()` resets the write head without deallocating — buffer reuse across epochs
- `_ensure_capacity()` doubles capacity on growth (amortized O(1) per add)

- [ ] **Step 4: Run buffer tests**

Run: `uv run pytest tests/test_split_merge_gae_opt.py::TestPreAllocatedBuffer tests/test_katago_ppo.py::TestKataGoRolloutBuffer -v`

Expected: All PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `uv run pytest tests/test_katago_ppo.py tests/test_pytorch_training_gaps.py tests/test_torch_compile.py tests/test_amp.py tests/test_pytorch_amp_pipeline.py -v`

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_split_merge_gae_opt.py
git commit -m "perf: pre-allocated contiguous rollout buffer

Replace list-of-tensors + torch.cat with pre-allocated contiguous
storage. add() writes via slice assignment, flatten() is zero-copy.
Buffer is reused across epochs without reallocation.

For 512 steps × 256 envs, eliminates ~1 GB torch.cat allocation
that was taking 1-3s per epoch."
```

---

## Task 4: Overlap Observation Transfer with GAE Computation

**Files:**
- Modify: `keisei/training/katago_ppo.py:560-639` (the split-merge + transfer section)
- Modify: `tests/test_split_merge_gae_opt.py`

The ~1 GB observation tensor transfer to GPU (line 633) currently waits until GAE is complete. Since GAE runs on GPU after Task 2, we can start transferring observations on a separate CUDA stream *before* the GAE backward scan begins, so the PCIe transfer overlaps with GAE kernel execution.

For the non-split-merge (vectorized) path, a similar overlap is possible but less impactful since GPU GAE is already fast there. This task focuses on the split-merge path.

- [ ] **Step 1: Write test — overlapped transfer produces correct results**

Append to `tests/test_split_merge_gae_opt.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it passes (baseline)**

Run: `uv run pytest tests/test_split_merge_gae_opt.py::TestOverlappedTransfer -v`

Expected: PASS

- [ ] **Step 3: Implement overlapped transfer in the split-merge path**

In `keisei/training/katago_ppo.py`, restructure the code between the env partition (Task 1) and the GPU data load (line 629). The idea: start the observation transfer on a dedicated stream *before* GAE begins, so PCIe transfer and GAE computation overlap.

Find the block that currently reads (after Task 1 and Task 2 changes):

```python
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = min(self.params.batch_size, total_samples)

        amp_dtype, autocast_device = _amp_dtype_and_device(self.params.use_amp, device)

        # --- Optimization: move entire dataset to GPU once ---
        gpu_obs = data["observations"].to(device, non_blocking=True)
        gpu_actions = data["actions"].to(device, non_blocking=True)
        gpu_old_log_probs = data["log_probs"].to(device, non_blocking=True)
        gpu_advantages = advantages.to(device, non_blocking=True)
        gpu_legal_masks = data["legal_masks"].to(device, non_blocking=True)
        gpu_value_cats = data["value_categories"].to(device, non_blocking=True)
        gpu_score_targets = data["score_targets"].to(device, non_blocking=True)
```

Replace with:

```python
        batch_size = min(self.params.batch_size, total_samples)

        amp_dtype, autocast_device = _amp_dtype_and_device(self.params.use_amp, device)

        # --- Optimization: overlap CPU→GPU transfer with advantage normalization ---
        # Observations are the largest tensor (~1 GB for 65K × 50 × 9 × 9).
        # Start their transfer on a dedicated stream so it overlaps with the
        # advantage normalization and remaining small transfers on the default stream.
        # pin_memory() makes the async transfer truly non-blocking on CUDA.
        if device.type == "cuda":
            _transfer_stream = torch.cuda.Stream(device)
            with torch.cuda.stream(_transfer_stream):
                obs_pinned = data["observations"].pin_memory()
                gpu_obs = obs_pinned.to(device, non_blocking=True)
                lm_pinned = data["legal_masks"].pin_memory()
                gpu_legal_masks = lm_pinned.to(device, non_blocking=True)
        else:
            _transfer_stream = None
            gpu_obs = data["observations"].to(device)
            gpu_legal_masks = data["legal_masks"].to(device)

        # Advantage normalization runs on CPU while obs transfer proceeds on GPU
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Small tensors — transfer on default stream (fast, not worth a separate stream)
        gpu_actions = data["actions"].to(device, non_blocking=True)
        gpu_old_log_probs = data["log_probs"].to(device, non_blocking=True)
        gpu_advantages = advantages.to(device, non_blocking=True)
        gpu_value_cats = data["value_categories"].to(device, non_blocking=True)
        gpu_score_targets = data["score_targets"].to(device, non_blocking=True)

        # Wait for observation transfer to complete before training begins
        if _transfer_stream is not None:
            torch.cuda.current_stream(device).wait_stream(_transfer_stream)
```

Key points:
- `pin_memory()` enables truly async DMA transfer (without it, `non_blocking=True` may still block)
- Observations and legal_masks are the two largest tensors — transfer them on a separate stream
- Small tensors (actions, log_probs, advantages, etc.) transfer on the default stream — overhead of a separate stream would exceed the transfer time
- `wait_stream()` creates a dependency so the training loop doesn't read partial data

- [ ] **Step 4: Run all tests**

Run: `uv run pytest tests/test_split_merge_gae_opt.py tests/test_katago_ppo.py tests/test_gae.py tests/test_gae_batched.py tests/test_torch_compile.py tests/test_amp.py tests/test_pytorch_amp_pipeline.py -v`

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_split_merge_gae_opt.py
git commit -m "perf: overlap obs GPU transfer with advantage normalization

Start observation + legal_mask transfer on a dedicated CUDA stream
before advantage normalization. pin_memory() enables truly async DMA.
Small tensors transfer on the default stream. wait_stream() ensures
all data is resident before the first training kernel."
```

---

## Task 5: Final Integration Test and Cleanup

**Files:**
- Modify: `tests/test_split_merge_gae_opt.py` (add timing benchmark)
- Modify: `keisei/training/katago_ppo.py` (remove dead import if any)

- [ ] **Step 1: Add a timing sanity test that the optimized path is measurably faster**

Append to `tests/test_split_merge_gae_opt.py`:

```python
import time


class TestPerformanceSanity:
    """Sanity check that the optimized path isn't accidentally slower."""

    def test_argsort_faster_than_boolean_loop(self):
        """argsort+split must be faster than boolean-index loop for 64+ envs."""
        torch.manual_seed(42)
        N_envs = 64
        total_samples = 8192
        env_ids = torch.randint(0, N_envs, (total_samples,))
        rewards = torch.randn(total_samples)

        # Boolean-index loop (old path)
        start = time.perf_counter()
        for _ in range(3):
            for eid in range(N_envs):
                mask = env_ids == eid
                _ = rewards[mask]
        bool_time = (time.perf_counter() - start) / 3

        # argsort+split (new path)
        start = time.perf_counter()
        for _ in range(3):
            sort_idx = torch.argsort(env_ids, stable=True)
            sorted_ids = env_ids[sort_idx]
            _, counts = sorted_ids.unique_consecutive(return_counts=True)
            splits = torch.split(sort_idx, counts.tolist())
            for idx in splits:
                _ = rewards[idx]
        sort_time = (time.perf_counter() - start) / 3

        # argsort should be at least 2x faster for 64 envs
        assert sort_time < bool_time, (
            f"argsort ({sort_time:.4f}s) not faster than bool loop ({bool_time:.4f}s)"
        )
```

- [ ] **Step 2: Run the full optimization test suite**

Run: `uv run pytest tests/test_split_merge_gae_opt.py -v`

Expected: All PASS

- [ ] **Step 3: Run the complete PPO-related test suite one final time**

Run: `uv run pytest tests/test_katago_ppo.py tests/test_gae.py tests/test_gae_batched.py tests/test_torch_compile.py tests/test_amp.py tests/test_pytorch_amp_pipeline.py tests/test_pytorch_training_gaps.py tests/test_split_merge_gae_opt.py -v`

Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_split_merge_gae_opt.py
git commit -m "test: add performance sanity checks for split-merge GAE optimization"
```

---

## Expected Impact

| Fix | Target | Before | After (estimated) |
|-----|--------|--------|-------------------|
| Task 1: argsort+split | Per-env partition | 3-8s | < 0.1s |
| Task 2: GPU padded GAE | GAE backward scan | 1-2s | < 0.01s |
| Task 3: Pre-alloc buffer | buffer.flatten() | 1-3s | ~0s (zero-copy) |
| Task 4: Overlapped transfer | Obs GPU transfer | 1-2s (serial) | overlapped with GAE |
| **Total** | **GPU stall** | **10-20s** | **< 1s** |

## Risk Notes

- **Task 3 (pre-alloc buffer)** is the highest-risk change — it touches the most code and could introduce subtle bugs if the write offset tracking is wrong. The existing test suite for `KataGoRolloutBuffer` is comprehensive and will catch regressions.
- **Task 4 (overlapped transfer)** relies on `pin_memory()` which allocates page-locked host memory. On systems with limited RAM, this could cause allocation failures. The `if device.type == "cuda"` guard ensures CPU-only runs are unaffected.
- **Task 2 (GPU padded GAE)** passes `lengths` as a CPU tensor to avoid a GPU sync for the O(N) scalar loop. This is safe because the loop is negligible compared to the backward scan.
