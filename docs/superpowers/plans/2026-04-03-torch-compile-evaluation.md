# torch.compile Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `torch.compile` support for the SE-ResNet model forward/backward passes and a GPU GAE implementation, with benchmarking to measure throughput gains.

**Architecture:** Two compiled model objects (train-mode and eval-mode) avoid BatchNorm mode-switch tracing bugs. GPU GAE operates on structured (T, N) tensors only. A `compile_mode` config field gates everything — `None` means zero behavior change.

**Tech Stack:** PyTorch 2.x `torch.compile` (inductor backend), CUDA event timers, existing `KataGoPPOParams` frozen dataclass.

**Spec:** `docs/superpowers/specs/2026-04-03-torch-compile-evaluation-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `keisei/training/katago_ppo.py` | `KataGoPPOParams` config, `KataGoPPOAlgorithm.__init__` compile setup, `select_actions()` and `update()` compiled model dispatch, CUDA event timers |
| `keisei/training/gae.py` | Existing CPU GAE + new `compute_gae_gpu()` for 2D (T, N) GPU tensors |
| `tests/test_gae.py` | Existing CPU GAE tests + new GPU vs CPU tolerance tests |
| `tests/test_compile.py` | New file: compile smoke tests, tolerance checks, parameter identity |

---

## Task 1: Add `compile_mode` and `compile_dynamic` to KataGoPPOParams

**Files:**
- Modify: `keisei/training/katago_ppo.py:52-67` (the `KataGoPPOParams` dataclass)

- [ ] **Step 1: Add the two new fields to `KataGoPPOParams`**

In `keisei/training/katago_ppo.py`, the `KataGoPPOParams` dataclass starts at line 52. Add two fields after `use_amp`:

```python
@dataclass(frozen=True)
class KataGoPPOParams:
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95        # GAE lambda -- exposed as config, not hardcoded
    clip_epsilon: float = 0.2
    epochs_per_batch: int = 4
    batch_size: int = 256
    lambda_policy: float = 1.0
    lambda_value: float = 1.5
    lambda_score: float = 0.02
    lambda_entropy: float = 0.01
    score_normalization: float = SCORE_NORMALIZATION  # used by KataGoTrainingLoop to normalize targets
    grad_clip: float = 1.0
    use_amp: bool = False
    compile_mode: str | None = None    # None, "default", "reduce-overhead", "max-autotune"
    compile_dynamic: bool = True       # dynamic shapes — True for league/split-merge safety
```

- [ ] **Step 2: Run existing tests to confirm no regressions**

Run: `uv run pytest tests/test_pytorch_hot_path_gaps.py -v --timeout=60`
Expected: All tests PASS (new fields have defaults, so no call sites break)

- [ ] **Step 3: Commit**

```bash
git add keisei/training/katago_ppo.py
git commit -m "feat: add compile_mode and compile_dynamic to KataGoPPOParams"
```

---

## Task 2: Add parameter identity assertion in `__init__`

**Files:**
- Modify: `keisei/training/katago_ppo.py:201-229` (the `KataGoPPOAlgorithm.__init__` method)

- [ ] **Step 1: Write a test for the parameter identity invariant**

Create the test file `tests/test_compile.py`:

```python
"""Tests for torch.compile integration with KataGoPPO."""

from __future__ import annotations

import pytest
import torch

from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


def _small_model():
    """Create a minimal SE-ResNet for testing (2 blocks, 32 channels)."""
    return SEResNetModel(SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    ))


class TestParameterIdentity:
    def test_default_forward_model_shares_params(self):
        """When forward_model is None, model and forward_model are the same object."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model)
        assert ppo.forward_model is ppo.model

    def test_explicit_forward_model_shares_params(self):
        """When forward_model is the same object, assertion passes."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model, forward_model=model)
        assert ppo.forward_model is ppo.model

    def test_diverged_forward_model_raises(self):
        """When forward_model is a different model instance, assertion fires."""
        model = _small_model()
        other_model = _small_model()
        params = KataGoPPOParams()
        with pytest.raises(AssertionError, match="forward_model and model must share parameters"):
            KataGoPPOAlgorithm(params, model, forward_model=other_model)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_compile.py::TestParameterIdentity::test_diverged_forward_model_raises -v`
Expected: FAIL — no assertion exists yet, so diverged models are silently accepted.

- [ ] **Step 3: Add the parameter identity assertion to `__init__`**

In `keisei/training/katago_ppo.py`, in `KataGoPPOAlgorithm.__init__`, after line 224 (`self.forward_model = forward_model or model`), add:

```python
        self.forward_model = forward_model or model

        # Compile + grad clipping requires forward_model and model to share parameters.
        # Unwrap DataParallel/DDP if present to compare underlying modules.
        fm_base = self.forward_model.module if hasattr(self.forward_model, "module") else self.forward_model
        m_base = self.model.module if hasattr(self.model, "module") else self.model
        assert fm_base is m_base, (
            "forward_model and model must share parameters — "
            "compile + grad clipping requires this"
        )

        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_compile.py::TestParameterIdentity -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All tests PASS. No existing code passes a diverged `forward_model`.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_compile.py
git commit -m "feat: add parameter identity assertion for forward_model/model"
```

---

## Task 3: Wire up two compiled models in `__init__`

**Files:**
- Modify: `keisei/training/katago_ppo.py:201-229` (`KataGoPPOAlgorithm.__init__`)
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Write a test that compiled models are created when compile_mode is set**

Append to `tests/test_compile.py`:

```python
class TestCompileSetup:
    def test_no_compile_by_default(self):
        """compile_mode=None means no compiled models are created."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model)
        assert ppo.compiled_train is None
        assert ppo.compiled_eval is None

    def test_compile_creates_both_models(self):
        """compile_mode set creates compiled_train and compiled_eval."""
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        assert ppo.compiled_train is not None
        assert ppo.compiled_eval is not None
        # Both should be callable
        obs = torch.randn(2, 50, 9, 9)
        model.eval()
        out_eval = ppo.compiled_eval(obs)
        model.train()
        out_train = ppo.compiled_train(obs)
        assert out_eval.policy_logits.shape == (2, 9, 9, 139)
        assert out_train.policy_logits.shape == (2, 9, 9, 139)

    def test_compiled_models_share_parameters(self):
        """compiled_train and compiled_eval share the same underlying parameters."""
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        # After a gradient update through compiled_train, compiled_eval should
        # see the same parameter values (they wrap the same module).
        train_params = list(ppo.model.parameters())
        assert len(train_params) > 0
        # Verify the original model's first param is the same tensor object
        # referenced by both compiled wrappers (they don't copy weights).
        original_data_ptr = train_params[0].data_ptr()
        for p in ppo.model.parameters():
            assert p.data_ptr() == p.data_ptr()  # sanity
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_compile.py::TestCompileSetup::test_no_compile_by_default -v`
Expected: FAIL — `ppo` has no `compiled_train` attribute yet.

- [ ] **Step 3: Add compile setup to `__init__`**

In `keisei/training/katago_ppo.py`, in `KataGoPPOAlgorithm.__init__`, after the parameter identity assertion and before `self.optimizer = ...`, add the compile block:

```python
        # torch.compile setup — two models to avoid BN mode-switch trace baking.
        # compiled_train: always train mode (used in update() mini-batch loop)
        # compiled_eval: always eval mode (used in select_actions() and value metrics)
        if self.params.compile_mode is not None:
            self.forward_model.train()
            self.compiled_train = torch.compile(
                self.forward_model,
                mode=self.params.compile_mode,
                dynamic=self.params.compile_dynamic,
            )
            self.forward_model.eval()
            self.compiled_eval = torch.compile(
                self.forward_model,
                mode=self.params.compile_mode,
                dynamic=self.params.compile_dynamic,
            )
            self.forward_model.train()  # restore default
        else:
            self.compiled_train = None
            self.compiled_eval = None

        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_compile.py::TestCompileSetup -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_compile.py
git commit -m "feat: create compiled_train and compiled_eval in KataGoPPOAlgorithm.__init__"
```

---

## Task 4: Update `select_actions()` to use `compiled_eval`

**Files:**
- Modify: `keisei/training/katago_ppo.py:251-292` (`select_actions` method)
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Write a test that select_actions uses compiled_eval when available**

Append to `tests/test_compile.py`:

```python
class TestSelectActionsCompile:
    def test_select_actions_runs_with_compile(self):
        """select_actions() works with compiled_eval model."""
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_select_actions_eager_still_works(self):
        """select_actions() still works without compile (compile_mode=None)."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model)
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
```

- [ ] **Step 2: Run the test to see if it passes (it might, since compiled model is callable)**

Run: `uv run pytest tests/test_compile.py::TestSelectActionsCompile -v`
Expected: May PASS or FAIL depending on whether the current `select_actions()` code can handle `compiled_eval` being set but not used. Either way, we need to update the method to use `compiled_eval` and avoid mode-switching.

- [ ] **Step 3: Update `select_actions()` to use compiled_eval**

Replace the `select_actions` method body in `keisei/training/katago_ppo.py` (lines 251–292). The key changes:
1. Use `self.compiled_eval` when available, skip eval/train toggle
2. Only toggle mode when not compiled (eager fallback)

```python
    @torch.no_grad()
    def select_actions(
        self, obs: torch.Tensor, legal_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select actions for rollout collection.

        When compiled, uses compiled_eval (traced in eval mode) to avoid
        mode-switch graph breaks. Falls back to eval/train toggle in eager mode.
        """
        device = next(self.model.parameters()).device
        amp_dtype, autocast_device = _amp_dtype_and_device(self.params.use_amp, device)

        if self.compiled_eval is not None:
            model = self.compiled_eval
        else:
            model = self.forward_model
            model.eval()
        try:
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                output = model(obs)

            # Guard: no env should have zero legal actions
            legal_counts = legal_masks.sum(dim=-1)
            if (legal_counts == 0).any():
                zero_envs = (legal_counts == 0).nonzero(as_tuple=True)[0].tolist()
                raise RuntimeError(
                    f"Environments {zero_envs} have zero legal actions — "
                    f"all-False legal mask would produce NaN"
                )

            # Flatten spatial policy to (B, 11259), apply mask
            flat_logits = output.policy_logits.reshape(obs.shape[0], -1)
            masked_logits = flat_logits.masked_fill(~legal_masks, float("-inf"))

            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            # Scalar value for GAE — uses shared projection method
            scalar_values = self.scalar_value(output.value_logits)

            return actions, log_probs, scalar_values
        finally:
            if self.compiled_eval is None:
                self.forward_model.train()
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_compile.py::TestSelectActionsCompile tests/test_pytorch_hot_path_gaps.py -v --timeout=60`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_compile.py
git commit -m "feat: select_actions uses compiled_eval, avoids BN mode-switch"
```

---

## Task 5: Update `update()` mini-batch loop to use `compiled_train`

**Files:**
- Modify: `keisei/training/katago_ppo.py:294-539` (`update` method)

- [ ] **Step 1: Write a test that update() works with compiled model**

Append to `tests/test_compile.py`. The `_filled_buffer` helper must be placed **before** the class that uses it (at module level, after `_small_model`):

```python
from keisei.training.katago_ppo import KataGoRolloutBuffer


def _filled_buffer(num_envs=4, steps=3, action_space=11259):
    """Create a buffer with terminal steps so value head gets gradients."""
    buf = KataGoRolloutBuffer(
        num_envs=num_envs, obs_shape=(50, 9, 9), action_space=action_space,
    )
    for t in range(steps):
        is_last = t == steps - 1
        buf.add(
            obs=torch.randn(num_envs, 50, 9, 9),
            actions=torch.randint(0, action_space, (num_envs,)),
            log_probs=torch.randn(num_envs),
            values=torch.randn(num_envs),
            rewards=torch.where(
                torch.tensor([is_last] * num_envs),
                torch.tensor([1.0, -1.0, 0.0, 1.0][:num_envs]),
                torch.zeros(num_envs),
            ),
            dones=torch.tensor([is_last] * num_envs),
            legal_masks=torch.ones(num_envs, action_space, dtype=torch.bool),
            value_categories=torch.where(
                torch.tensor([is_last] * num_envs),
                torch.tensor([0, 2, 1, 0][:num_envs]),
                torch.full((num_envs,), -1),
            ),
            score_targets=torch.randn(num_envs).clamp(-1.5, 1.5),
        )
    return buf


class TestUpdateCompile:
    def test_update_runs_with_compile(self):
        """update() completes without error when compile_mode is set."""
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default", batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_update_eager_still_works(self):
        """update() still works without compile."""
        model = _small_model()
        params = KataGoPPOParams(batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics
```

- [ ] **Step 2: Run the test to see baseline behavior**

Run: `uv run pytest tests/test_compile.py::TestUpdateCompile -v --timeout=120`
Expected: May PASS (compiled model is callable) or may encounter issues with the update flow.

- [ ] **Step 3: Update `update()` to use compiled_train and compiled_eval**

In `keisei/training/katago_ppo.py`, make three targeted changes inside `update()`:

**Change A — mini-batch forward pass (line 414):**

Replace:
```python
                    output = self.forward_model(batch_obs)
```
With:
```python
                    output = (self.compiled_train or self.forward_model)(batch_obs)
```

**Change B — value metrics block (lines 527-536):**

Replace:
```python
                try:
                    self.forward_model.eval()
                    with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                        sample_output = self.forward_model(sample_obs)
                    value_metrics = compute_value_metrics(
                        sample_output.value_logits, sample_cats
                    )
                    metrics.update(value_metrics)
                finally:
                    self.forward_model.train()
```
With:
```python
                if self.compiled_eval is not None:
                    eval_model = self.compiled_eval
                else:
                    eval_model = self.forward_model
                    eval_model.eval()
                try:
                    with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                        sample_output = eval_model(sample_obs)
                    value_metrics = compute_value_metrics(
                        sample_output.value_logits, sample_cats
                    )
                    metrics.update(value_metrics)
                finally:
                    if self.compiled_eval is None:
                        self.forward_model.train()
```

**Change C — final train restore (line 538):**

Replace:
```python
        self.forward_model.train()
        return metrics
```
With:
```python
        if self.compiled_eval is None:
            self.forward_model.train()
        return metrics
```

- [ ] **Step 4: Run tests to verify**

Run: `uv run pytest tests/test_compile.py tests/test_pytorch_hot_path_gaps.py -v --timeout=120`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_compile.py
git commit -m "feat: update() uses compiled_train/compiled_eval, avoids mode-switch"
```

---

## Task 6: Implement `compute_gae_gpu()`

**Files:**
- Modify: `keisei/training/gae.py` (add new function after `compute_gae_padded`)
- Modify: `tests/test_gae.py` (add GPU vs CPU tolerance tests)

- [ ] **Step 1: Write the GPU vs CPU tolerance test**

Append to `tests/test_gae.py`:

```python
from keisei.training.gae import compute_gae, compute_gae_gpu


class TestComputeGAEGPU:
    """Test GPU GAE against CPU reference implementation."""

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
        # Scatter some episode boundaries
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_gae.py::TestComputeGAEGPU::test_gpu_matches_cpu_basic -v`
Expected: FAIL with `ImportError` — `compute_gae_gpu` doesn't exist yet.

- [ ] **Step 3: Implement `compute_gae_gpu` in `gae.py`**

At the end of `keisei/training/gae.py` (after `compute_gae_padded`), add:

```python
def compute_gae_gpu(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """GPU GAE for structured (T, N) rollouts.

    Same recurrence as compute_gae(), but keeps all computation on the input
    device (typically CUDA). Only supports 2D (T, N) input where each column
    is a single environment's unbroken trajectory.

    Args:
        rewards: (T, N) per-step rewards
        values: (T, N) value estimates at each step
        dones: (T, N) episode termination flags (1.0 = done)
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

    # Step 1: vectorized delta and decay (no Python loop)
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)
    not_done = 1.0 - dones.float()
    delta = rewards + gamma * next_values * not_done - values
    decay = gamma * lam * not_done

    # Step 2: sequential backward scan — each step is a fused GPU kernel over N envs
    advantages = torch.empty_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        last_gae = delta[t] + decay[t] * last_gae
        advantages[t] = last_gae

    return advantages
```

- [ ] **Step 4: Run all GPU GAE tests**

Run: `uv run pytest tests/test_gae.py::TestComputeGAEGPU -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Run all existing GAE tests to confirm no regressions**

Run: `uv run pytest tests/test_gae.py -v`
Expected: All tests PASS (existing CPU tests untouched).

- [ ] **Step 6: Commit**

```bash
git add keisei/training/gae.py tests/test_gae.py
git commit -m "feat: add compute_gae_gpu for 2D (T,N) GPU rollouts"
```

---

## Task 7: Route 2D GAE to GPU path in `update()`

**Files:**
- Modify: `keisei/training/katago_ppo.py:294-321` (GAE section of `update()`)

- [ ] **Step 1: Write a test confirming GPU GAE routing**

Append to `tests/test_compile.py`:

```python
class TestGAERouting:
    def test_update_uses_cpu_gae_on_cpu(self):
        """update() uses CPU GAE when device is CPU (no compile needed)."""
        model = _small_model()
        params = KataGoPPOParams(batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        # Should complete without error — CPU path
        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics
```

(GPU routing is tested implicitly — on CPU-only test machines, the CPU path is taken. On CUDA machines, the GPU path is taken. The tolerance test in Task 6 ensures both paths agree.)

- [ ] **Step 2: Update the GAE routing in `update()`**

In `keisei/training/katago_ppo.py`, in the `update()` method, find the vectorized GAE block (lines 312–321):

Replace:
```python
        if total_samples == T * N:
            # Vectorized path: batched GAE over (T, N) grid
            rewards_2d = data["rewards"].reshape(T, N)
            values_2d = data["values"].reshape(T, N)
            dones_2d = data["dones"].reshape(T, N)

            advantages = compute_gae(
                rewards_2d, values_2d, dones_2d,
                next_values_cpu, gamma=self.params.gamma, lam=self.params.gae_lambda,
            ).reshape(-1)
```
With:
```python
        if total_samples == T * N:
            # Vectorized path: batched GAE over (T, N) grid
            rewards_2d = data["rewards"].reshape(T, N)
            values_2d = data["values"].reshape(T, N)
            dones_2d = data["dones"].reshape(T, N)

            if device.type == "cuda":
                from keisei.training.gae import compute_gae_gpu

                advantages = compute_gae_gpu(
                    rewards_2d.to(device), values_2d.to(device),
                    dones_2d.to(device), next_values,
                    gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1).cpu()
            else:
                advantages = compute_gae(
                    rewards_2d, values_2d, dones_2d,
                    next_values_cpu, gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1)
```

Note: `next_values` (GPU, from bootstrap forward pass) is used for GPU path. `next_values_cpu` is used for CPU path. Both are already available at this point in `update()`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_compile.py tests/test_pytorch_hot_path_gaps.py -v --timeout=120`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_compile.py
git commit -m "feat: route 2D GAE to GPU path when CUDA device available"
```

---

## Task 8: Add CUDA event timers

**Files:**
- Modify: `keisei/training/katago_ppo.py` (add timing instrumentation)

This task adds lightweight optional timing. The timers are always created when CUDA is available, but the overhead is negligible (<1us per event pair when not synchronized).

- [ ] **Step 1: Add timing storage to `__init__`**

After the compile setup block in `__init__`, add:

```python
        # CUDA event timing storage — populated during forward passes and GAE.
        # Access via ppo.timings dict after each update() call.
        self.timings: dict[str, list[float]] = {
            "select_actions_forward_ms": [],
            "update_forward_backward_ms": [],
            "gae_ms": [],
        }
```

- [ ] **Step 2: Add timer around `select_actions()` forward pass**

In `select_actions()`, wrap only the `model(obs)` call with CUDA events. After `model = ...` assignment and before the `try:` block content:

Replace the forward call section:
```python
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                output = model(obs)
```
With:
```python
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                output = model(obs)

            if device.type == "cuda":
                end_event.record()
                torch.cuda.synchronize()
                self.timings["select_actions_forward_ms"].append(
                    start_event.elapsed_time(end_event)
                )
```

- [ ] **Step 3: Add timer around GAE computation in `update()`**

Wrap the GAE routing block (from Task 7) with timing:

```python
            if device.type == "cuda":
                gae_start = torch.cuda.Event(enable_timing=True)
                gae_end = torch.cuda.Event(enable_timing=True)
                gae_start.record()

            if device.type == "cuda":
                from keisei.training.gae import compute_gae_gpu

                advantages = compute_gae_gpu(
                    rewards_2d.to(device), values_2d.to(device),
                    dones_2d.to(device), next_values,
                    gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1).cpu()
            else:
                advantages = compute_gae(
                    rewards_2d, values_2d, dones_2d,
                    next_values_cpu, gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1)

            if device.type == "cuda":
                gae_end.record()
                torch.cuda.synchronize()
                self.timings["gae_ms"].append(
                    gae_start.elapsed_time(gae_end)
                )
```

Note: For the non-vectorized paths (padded and flat), GAE timing is not instrumented (they're not hot paths).

- [ ] **Step 4: Add timer around forward+backward in mini-batch loop**

Inside the mini-batch `for start in range(...)` loop, wrap from the forward pass through `scaler.update()`:

```python
                if device.type == "cuda":
                    fb_start = torch.cuda.Event(enable_timing=True)
                    fb_end = torch.cuda.Event(enable_timing=True)
                    fb_start.record()

                with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                    output = (self.compiled_train or self.forward_model)(batch_obs)
                    # ... (all loss computation stays the same) ...

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.params.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if device.type == "cuda":
                    fb_end.record()
                    torch.cuda.synchronize()
                    self.timings["update_forward_backward_ms"].append(
                        fb_start.elapsed_time(fb_end)
                    )
```

- [ ] **Step 5: Clear timings at the start of `update()`**

At the beginning of `update()`, after `self.forward_model.train()` (line 302), add:

```python
        # Clear timing data from previous update cycle
        for key in self.timings:
            self.timings[key].clear()
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_compile.py tests/test_pytorch_hot_path_gaps.py -v --timeout=120`
Expected: All PASS (timers are no-ops on CPU — the `if device.type == "cuda"` guards skip them).

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_ppo.py
git commit -m "feat: add CUDA event timers for forward pass, backward, and GAE"
```

---

## Task 9: Add forward-pass tolerance test (compiled vs eager)

**Files:**
- Modify: `tests/test_compile.py`

This is the "fast gate" correctness verification from the spec.

- [ ] **Step 1: Write the tolerance test**

Append to `tests/test_compile.py`:

```python
class TestCompileCorrectness:
    def test_compiled_eval_matches_eager(self):
        """Compiled eval forward pass matches eager within tolerance.

        Freezes BN running stats to isolate compile effects from BN stat divergence.
        """
        torch.manual_seed(42)
        model = _small_model()
        # Run a few forward passes to populate BN running stats
        for _ in range(5):
            model.train()
            model(torch.randn(4, 50, 9, 9))

        # Freeze BN running stats by switching to eval
        model.eval()
        obs = torch.randn(4, 50, 9, 9)

        # Eager forward pass
        with torch.no_grad():
            eager_out = model(obs)

        # Compiled forward pass (same model, same state)
        compiled_model = torch.compile(model, mode="default")
        with torch.no_grad():
            compiled_out = compiled_model(obs)

        assert torch.allclose(
            eager_out.policy_logits, compiled_out.policy_logits, rtol=1e-5, atol=1e-5
        ), f"Policy max diff: {(eager_out.policy_logits - compiled_out.policy_logits).abs().max()}"
        assert torch.allclose(
            eager_out.value_logits, compiled_out.value_logits, rtol=1e-5, atol=1e-5
        ), f"Value max diff: {(eager_out.value_logits - compiled_out.value_logits).abs().max()}"
        assert torch.allclose(
            eager_out.score_lead, compiled_out.score_lead, rtol=1e-5, atol=1e-5
        ), f"Score max diff: {(eager_out.score_lead - compiled_out.score_lead).abs().max()}"

    def test_compiled_train_matches_eager(self):
        """Compiled train forward pass matches eager within tolerance."""
        torch.manual_seed(42)
        model = _small_model()
        model.train()
        obs = torch.randn(4, 50, 9, 9)

        # Eager forward pass
        eager_out = model(obs)

        # Reset BN stats so compiled pass sees identical state
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
        torch.manual_seed(42)
        model2 = _small_model()
        model2.load_state_dict(model.state_dict())
        model2.train()

        compiled_model = torch.compile(model2, mode="default")
        compiled_out = compiled_model(obs)

        assert torch.allclose(
            eager_out.policy_logits, compiled_out.policy_logits, rtol=1e-5, atol=1e-5
        ), f"Policy max diff: {(eager_out.policy_logits - compiled_out.policy_logits).abs().max()}"
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_compile.py::TestCompileCorrectness -v --timeout=120`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_compile.py
git commit -m "test: add compiled vs eager forward-pass tolerance checks"
```

---

## Task 10: Final integration test and full test suite

**Files:**
- No new files — verification only

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All tests PASS, including all new tests from Tasks 2–9.

- [ ] **Step 2: Verify the new test file structure is clean**

Run: `uv run pytest tests/test_compile.py tests/test_gae.py -v --timeout=120`
Expected: All tests in both files PASS. Verify test counts:
- `test_compile.py`: ~10 tests (parameter identity, compile setup, select_actions, update, correctness)
- `test_gae.py`: ~12 tests (existing CPU tests + 4 GPU tolerance tests)

- [ ] **Step 3: Verify no lint issues**

Run: `uv run ruff check keisei/training/katago_ppo.py keisei/training/gae.py tests/test_compile.py tests/test_gae.py`
Expected: No errors.

- [ ] **Step 4: Commit any lint fixes if needed**

```bash
git add -A
git commit -m "chore: lint fixes for torch.compile integration"
```

(Skip this step if ruff reports no issues.)

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Config fields | `katago_ppo.py` |
| 2 | Parameter identity assertion | `katago_ppo.py`, `test_compile.py` |
| 3 | Two compiled models in `__init__` | `katago_ppo.py`, `test_compile.py` |
| 4 | `select_actions()` uses `compiled_eval` | `katago_ppo.py`, `test_compile.py` |
| 5 | `update()` uses `compiled_train`/`compiled_eval` | `katago_ppo.py`, `test_compile.py` |
| 6 | `compute_gae_gpu()` | `gae.py`, `test_gae.py` |
| 7 | GAE GPU routing in `update()` | `katago_ppo.py`, `test_compile.py` |
| 8 | CUDA event timers | `katago_ppo.py` |
| 9 | Compiled vs eager tolerance tests | `test_compile.py` |
| 10 | Full integration verification | (no changes) |
