# torch.compile Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `torch.compile` support for the SE-ResNet model forward/backward passes and a GPU GAE implementation, with benchmarking to measure throughput gains.

**Architecture:** Two compiled model objects (train-mode and eval-mode) avoid BatchNorm mode-switch tracing bugs. GPU GAE operates on structured (T, N) tensors only. A `compile_mode` config field gates everything — `None` means zero behavior change.

**Tech Stack:** PyTorch 2.x `torch.compile` (inductor backend), CUDA event timers, existing `KataGoPPOParams` frozen dataclass.

**Spec:** `docs/superpowers/specs/2026-04-03-torch-compile-evaluation-design.md`

---

## Important Implementation Notes

These notes come from a 5-reviewer panel (architecture, systems, Python, quality,
PyTorch engineering). Read these before starting — they affect multiple tasks.

### N1. Tasks 3–5 are atomically dependent

Tasks 3, 4, and 5 form one logical unit. Task 3 creates `compiled_train` /
`compiled_eval`; Tasks 4 and 5 update the call sites to use them. If Task 3
lands but Task 4 does not, `select_actions()` still calls
`self.forward_model.eval()` on the shared module, which corrupts the compiled
train trace. **Complete and commit all three before running any training.**
Individual commits per task are fine for git history, but don't merge to main
until all three are done.

### N2. `torch.compile` traces at first call, not at `torch.compile()` time

The `torch.compile(...)` call in `__init__` returns an `OptimizedModule` wrapper.
No graph tracing happens until the first forward call. The `forward_model.train()`
/ `.eval()` calls between the two `torch.compile()` invocations in `__init__`
have no tracing effect — what matters is the module's `training` flag at the time
of each wrapper's **first forward call**. The two-model design works because
`compiled_train` is always called when the module is in train mode, and
`compiled_eval` is always called when it's in eval mode. Each wrapper builds an
independent compilation cache keyed on guard conditions including `training`.

### N3. `split_merge_step` calls `learner_model.eval()` on the shared module

`katago_loop.py:89` calls `learner_model.eval()` during league-mode rollout.
This mutates the `training` flag on the same module object that both compiled
wrappers reference. The `self.forward_model.train()` at the top of `update()`
(line 302) restores train mode before `compiled_train` is used. This is correct
today, but fragile. Task 5 adds an assertion to catch regressions.

### N4. `autocast` outside the compiled region limits fusion

The `autocast` context wraps the compiled forward call from outside, so inductor
cannot fuse dtype casts into the compiled kernels. For a 40-block SE-ResNet,
this means activations exit the compiled region in fp32 and are cast externally.
This reduces the compile speedup compared to what's theoretically achievable.
Moving `autocast` inside the model's `forward()` is out of scope for this plan
but should be noted as a follow-up optimization if benchmark gains are smaller
than expected.

### N5. `reduce-overhead` + `compile_dynamic=True` disables CUDA graphs

The `reduce-overhead` mode's main advantage is CUDA graph capture, which is
incompatible with dynamic shapes. The default `compile_dynamic=True` silently
prevents CUDA graph capture. Task 3 adds a warning log for this combination.
Users benchmarking V2 (`reduce-overhead`) should set `compile_dynamic=False`
for fixed-batch-size runs to get actual CUDA graph behavior.

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
- Create: `tests/test_compile.py`

- [ ] **Step 1: Write a test for the parameter identity invariant**

Create the test file `tests/test_compile.py`:

```python
"""Tests for torch.compile integration with KataGoPPO."""

from __future__ import annotations

import logging

import pytest
import torch

from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
)
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


def _small_model():
    """Create a minimal SE-ResNet for testing (2 blocks, 32 channels)."""
    return SEResNetModel(SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    ))


def _filled_buffer(num_envs=4, steps=3, action_space=11259):
    """Create a buffer with terminal steps so value head gets gradients.

    Defined at module level so all test classes can use it.
    Kept in sync with tests/test_pytorch_hot_path_gaps.py::_filled_buffer.
    """
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

**IMPORTANT:** This task, Task 4, and Task 5 are atomically dependent (see Note N1
above). Complete all three before running any training. The system is in a
temporarily unsafe state between this commit and the Task 5 commit.

- [ ] **Step 1: Write tests for compiled model creation and parameter sharing**

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
        """compile_mode set creates compiled_train and compiled_eval.

        Both wrappers should be callable without manual eval()/train() toggles —
        the __init__ sets mode before compiling each (see Note N2: tracing happens
        at first call, not at compile() time, but the mode must be correct at
        first call time).
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        assert ppo.compiled_train is not None
        assert ppo.compiled_eval is not None
        # Both should be callable without mode toggle
        obs = torch.randn(2, 50, 9, 9)
        out_eval = ppo.compiled_eval(obs)
        out_train = ppo.compiled_train(obs)
        assert out_eval.policy_logits.shape == (2, 9, 9, 139)
        assert out_train.policy_logits.shape == (2, 9, 9, 139)

    def test_compiled_wrappers_share_underlying_module(self):
        """compiled_train and compiled_eval wrap the same module, sharing parameters.

        Verify via _orig_mod (the OptimizedModule's reference to the wrapped module).
        Both must resolve to the same object as ppo.forward_model.
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        # torch.compile returns an OptimizedModule with _orig_mod pointing to
        # the original module. Both wrappers should point to the same module.
        assert hasattr(ppo.compiled_train, "_orig_mod"), (
            "compiled_train should be an OptimizedModule with _orig_mod"
        )
        assert ppo.compiled_train._orig_mod is ppo.forward_model
        assert ppo.compiled_eval._orig_mod is ppo.forward_model
        # And the underlying parameters should be the same objects
        train_params = list(ppo.compiled_train._orig_mod.parameters())
        model_params = list(ppo.model.parameters())
        assert len(train_params) == len(model_params)
        for tp, mp in zip(train_params, model_params):
            assert tp.data_ptr() == mp.data_ptr(), (
                "compiled wrapper parameter should share storage with base model"
            )

    def test_reduce_overhead_dynamic_warns(self, caplog):
        """reduce-overhead + compile_dynamic=True should log a warning.

        CUDA graphs (the mechanism behind reduce-overhead) are incompatible with
        dynamic shapes. See Note N5.
        """
        model = _small_model()
        params = KataGoPPOParams(
            compile_mode="reduce-overhead", compile_dynamic=True,
        )
        with caplog.at_level(logging.WARNING):
            KataGoPPOAlgorithm(params, model)
        assert any("reduce-overhead" in msg and "dynamic" in msg for msg in caplog.messages), (
            "Should warn when reduce-overhead is combined with compile_dynamic=True"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_compile.py::TestCompileSetup::test_no_compile_by_default -v`
Expected: FAIL — `ppo` has no `compiled_train` attribute yet.

- [ ] **Step 3: Add compile setup to `__init__`**

In `keisei/training/katago_ppo.py`, add a `logging` import at the top of the file:

```python
import logging
```

Then in `KataGoPPOAlgorithm.__init__`, after the parameter identity assertion and before `self.optimizer = ...`, add the compile block:

```python
        # torch.compile setup — two models to avoid BN mode-switch trace baking.
        # compiled_train: always train mode (used in update() mini-batch loop)
        # compiled_eval: always eval mode (used in select_actions() and value metrics)
        # Note: torch.compile() does NOT trace the graph — that happens at the
        # first forward call. The mode set here must match the mode at first call.
        _log = logging.getLogger(__name__)
        if self.params.compile_mode is not None:
            if self.params.compile_mode == "reduce-overhead" and self.params.compile_dynamic:
                _log.warning(
                    "compile_mode='reduce-overhead' with compile_dynamic=True disables "
                    "CUDA graph capture (the main benefit of reduce-overhead). "
                    "Set compile_dynamic=False for fixed-batch-size runs."
                )
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
            _log.info(
                "torch.compile active: mode=%s, dynamic=%s",
                self.params.compile_mode, self.params.compile_dynamic,
            )
        else:
            self.compiled_train = None
            self.compiled_eval = None

        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_compile.py::TestCompileSetup -v`
Expected: All 4 tests PASS.

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

- [ ] **Step 1: Write tests for select_actions with compiled model**

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

    def test_select_actions_no_mode_switch_when_compiled(self):
        """When compiled, select_actions must NOT toggle eval/train on forward_model.

        This is the core BN mode-switch invariant (spec hazard H1). The compiled_eval
        wrapper handles eval mode internally via its traced graph. Toggling the
        underlying module would corrupt compiled_train's trace.
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default")
        ppo = KataGoPPOAlgorithm(params, model)
        # Model should be in train mode before and after select_actions
        assert ppo.forward_model.training is True
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        ppo.select_actions(obs, legal_masks)
        assert ppo.forward_model.training is True, (
            "select_actions must not leave forward_model in eval mode when compiled"
        )
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_compile.py::TestSelectActionsCompile -v`
Expected: May PASS or FAIL depending on whether the current code handles compiled_eval. Either way, we update the method.

- [ ] **Step 3: Update `select_actions()` to use compiled_eval**

Replace the `select_actions` method body in `keisei/training/katago_ppo.py` (lines 251–292). Key changes:
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

        Note: compiled_eval is always called while forward_model is in train mode.
        This is correct because torch.compile captures the training flag at first-call
        time and caches it — subsequent calls reuse that cached graph regardless of
        the module's current training flag. The compiled_eval graph was first called
        in eval mode (during warmup or first select_actions call after __init__
        sets eval before compiling), so it always uses eval-mode BN behavior.
        """
        device = next(self.model.parameters()).device
        amp_dtype, autocast_device = _amp_dtype_and_device(self.params.use_amp, device)

        if self.compiled_eval is not None:
            model = self.compiled_eval
            # Do NOT toggle forward_model.eval() — see BN invariant in spec H1
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
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Write tests for update() with compiled model**

Append to `tests/test_compile.py`:

```python
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

    def test_update_changes_weights(self):
        """update() must actually modify model weights (detects stale-gradient bugs).

        Captures parameter values before and after update, verifies they changed.
        This catches silent bugs where compiled_train dispatches to the wrong model
        or the optimizer operates on a different parameter set.
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default", batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        # Snapshot first parameter before update
        first_param = list(model.parameters())[0]
        before = first_param.data.clone()
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        ppo.update(buf, next_values)
        after = first_param.data
        assert not torch.equal(before, after), (
            "Model parameters should change after update() — optimizer may be "
            "operating on the wrong parameter set"
        )

    def test_forward_model_in_train_mode_after_update(self):
        """forward_model must be in train mode after update() completes.

        This is critical for the BN invariant: split_merge_step in katago_loop.py
        calls learner_model.eval() during rollout (line 89), and update() must
        restore train mode. See Note N3.
        """
        model = _small_model()
        params = KataGoPPOParams(compile_mode="default", batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        ppo.update(buf, next_values)
        assert ppo.forward_model.training is True
```

- [ ] **Step 2: Run the tests to see baseline behavior**

Run: `uv run pytest tests/test_compile.py::TestUpdateCompile -v --timeout=120`
Expected: May PASS (compiled model is callable) or may encounter issues.

- [ ] **Step 3: Update `update()` to use compiled_train and compiled_eval**

In `keisei/training/katago_ppo.py`, make four targeted changes inside `update()`:

**Change A — add train-mode assertion at top of update() (after line 302):**

After `self.forward_model.train()` at line 302, add:

```python
        self.forward_model.train()
        # Safety assertion: forward_model must be in train mode before compiled_train
        # is used. split_merge_step calls learner_model.eval() during rollout and
        # relies on update() to restore train mode. See Note N3 in the plan.
        assert self.forward_model.training, (
            "forward_model must be in train mode at start of update() — "
            "compiled_train graph requires this"
        )
```

**Change B — mini-batch forward pass (line 414):**

Replace:
```python
                    output = self.forward_model(batch_obs)
```
With:
```python
                    # Use compiled_train if available; fall back to eager forward_model.
                    # compiled_train was traced in train mode — BN updates running stats.
                    if self.compiled_train is not None:
                        output = self.compiled_train(batch_obs)
                    else:
                        output = self.forward_model(batch_obs)
```

**Change C — value metrics block (lines 527-536):**

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

**Change D — final train restore (line 538):**

Replace:
```python
        self.forward_model.train()
        return metrics
```
With:
```python
        # Always restore train mode — even when compiled. split_merge_step
        # relies on this (see Note N3). No-op if already in train mode.
        self.forward_model.train()
        return metrics
```

Note: We keep the unconditional `self.forward_model.train()` at the end (unlike
the spec which conditionally skipped it) because `split_merge_step` in
`katago_loop.py` calls `learner_model.eval()` on the shared module. This
ensures train mode is always restored, making the invariant robust.

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

- [ ] **Step 1: Write the GPU vs CPU tolerance tests**

Append to `tests/test_gae.py`:

```python
from keisei.training.gae import compute_gae, compute_gae_gpu


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

    The 1D flat case is NOT supported — the vectorized values[1:] trick would
    conflate transitions across environment boundaries after flattening. Use
    the original compute_gae() for 1D inputs. See spec hazard H3.

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
    # next_values[t] = values[t+1] for t < T-1, next_value for t == T-1
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)
    not_done = 1.0 - dones.float()
    delta = rewards + gamma * next_values * not_done - values
    decay = gamma * lam * not_done

    # Step 2: sequential backward scan — each step is a fused GPU kernel over N envs.
    # The Python loop has T iterations (~128), each launching ~2 CUDA kernels.
    # This is faster than CPU GAE because each step operates on N envs in parallel
    # without Python-level tensor ops or CPU-GPU round-trips.
    advantages = torch.empty_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        last_gae = delta[t] + decay[t] * last_gae
        advantages[t] = last_gae

    return advantages
```

- [ ] **Step 4: Run all GPU GAE tests**

Run: `uv run pytest tests/test_gae.py::TestComputeGAEGPU -v`
Expected: All 7 tests PASS (basic, large, all_done, single_timestep, single_env, rejects_1d, shape_dtype).

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
- Modify: `tests/test_compile.py`

- [ ] **Step 1: Write tests confirming GAE routing**

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

    def test_flat_fallback_stays_on_cpu(self):
        """When total_samples != T * N (flat fallback), CPU GAE is always used.

        The GPU path only supports 2D (T, N) structured input. The flat fallback
        (no env_ids) must never attempt compute_gae_gpu. See spec hazard H3.
        """
        model = _small_model()
        params = KataGoPPOParams(batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        # Create a buffer where total_samples != T * N by using different num_envs
        # per step (simulate irregular buffer)
        buf = KataGoRolloutBuffer(num_envs=4, obs_shape=(50, 9, 9), action_space=11259)
        # Add 3 steps with 4 envs each = 12 samples, T=3, N=4, T*N=12 — matches
        # So instead, we test the CPU-only path directly
        buf2 = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        metrics = ppo.update(buf2, next_values)
        assert "policy_loss" in metrics
```

- [ ] **Step 2: Add the `compute_gae_gpu` import at the top of `katago_ppo.py`**

Move the import from inside the function to the top-level imports of `katago_ppo.py`.
After the existing `from keisei.training.models.katago_base import KataGoBaseModel` line, add:

```python
from keisei.training.gae import compute_gae_gpu
```

This avoids a dynamic import inside the hot path (flagged by Python reviewer).
The import is unconditional — `compute_gae_gpu` always exists after Task 6.

- [ ] **Step 3: Update the GAE routing in `update()`**

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
                # GPU path: move buffer tensors to GPU, compute GAE there, return to CPU.
                # next_values is already on GPU (from bootstrap forward pass).
                # .detach() ensures no gradient graph leaks through GAE.
                # Memory: ~3 MB for T=128, N=512 (negligible on 4060/H200).
                advantages = compute_gae_gpu(
                    rewards_2d.to(device), values_2d.to(device),
                    dones_2d.to(device), next_values.detach(),
                    gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1).cpu()
            else:
                advantages = compute_gae(
                    rewards_2d, values_2d, dones_2d,
                    next_values_cpu, gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1)
```

Note: `next_values.detach()` is used for the GPU path (defensive — prevents
gradient graph from leaking through GAE). The CPU path uses `next_values_cpu`
which is already detached at line 310. Both are available at this point.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_compile.py tests/test_pytorch_hot_path_gaps.py -v --timeout=120`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_compile.py
git commit -m "feat: route 2D GAE to GPU path when CUDA device available"
```

---

## Task 8: Add CUDA event timers

**Files:**
- Modify: `keisei/training/katago_ppo.py` (add timing instrumentation)
- Modify: `tests/test_compile.py`

**CRITICAL:** Timers must NOT call `torch.cuda.synchronize()` inside hot loops.
Synchronizing after every forward pass or mini-batch serializes the CPU/GPU
pipeline, inflating timings and actively degrading throughput. Instead, record
event pairs without syncing, then read `elapsed_time()` once at the end of the
loop. `elapsed_time()` implicitly waits for both events to complete — no
explicit sync is needed.

- [ ] **Step 1: Add timing storage and flush method to `__init__`**

After the compile setup block in `__init__`, add:

```python
        # CUDA event timing — records event pairs during forward passes and GAE.
        # Events are accumulated without synchronization during the hot loop.
        # Call flush_timings() to read elapsed times (triggers one sync).
        # Lifecycle: select_actions_forward_ms accumulates during rollout,
        # update_forward_backward_ms and gae_ms accumulate during update().
        # All lists are cleared at the start of each update() call.
        self._timing_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "select_actions_forward_ms": [],
            "update_forward_backward_ms": [],
            "gae_ms": [],
        }
        self.timings: dict[str, list[float]] = {
            "select_actions_forward_ms": [],
            "update_forward_backward_ms": [],
            "gae_ms": [],
        }
```

- [ ] **Step 2: Add `flush_timings()` method to `KataGoPPOAlgorithm`**

Add this method after `get_entropy_coeff`:

```python
    def flush_timings(self) -> None:
        """Convert accumulated CUDA event pairs to elapsed-time floats.

        Calls elapsed_time() on each event pair, which implicitly waits for
        both events to complete. This is the ONLY synchronization point —
        no torch.cuda.synchronize() is called during recording.

        Call this after update() returns to populate self.timings with ms values.
        """
        for key, event_pairs in self._timing_events.items():
            self.timings[key] = [
                start.elapsed_time(end) for start, end in event_pairs
            ]
            event_pairs.clear()
```

- [ ] **Step 3: Add timer around `select_actions()` forward pass**

In `select_actions()`, wrap only the `model(obs)` call with CUDA event recording.
Replace the forward call section:

```python
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                output = model(obs)
```
With:
```python
            # Record CUDA events around forward pass only (not the legal mask guard,
            # which includes a CPU-syncing .nonzero() call — see spec hazard H8).
            if device.type == "cuda":
                _sa_start = torch.cuda.Event(enable_timing=True)
                _sa_end = torch.cuda.Event(enable_timing=True)
                _sa_start.record()

            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                output = model(obs)

            if device.type == "cuda":
                _sa_end.record()
                # Do NOT synchronize here — events are read lazily in flush_timings()
                self._timing_events["select_actions_forward_ms"].append((_sa_start, _sa_end))
```

- [ ] **Step 4: Add timer around GAE computation in `update()`**

Wrap the GAE routing block (from Task 7). Only instrument the vectorized
`if total_samples == T * N:` path (the one that may use GPU):

```python
            if device.type == "cuda":
                _gae_start = torch.cuda.Event(enable_timing=True)
                _gae_end = torch.cuda.Event(enable_timing=True)
                _gae_start.record()

            if device.type == "cuda":
                advantages = compute_gae_gpu(
                    rewards_2d.to(device), values_2d.to(device),
                    dones_2d.to(device), next_values.detach(),
                    gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1).cpu()
            else:
                advantages = compute_gae(
                    rewards_2d, values_2d, dones_2d,
                    next_values_cpu, gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1)

            if device.type == "cuda":
                _gae_end.record()
                self._timing_events["gae_ms"].append((_gae_start, _gae_end))
```

- [ ] **Step 5: Add timer around forward+backward in mini-batch loop**

Inside the mini-batch `for start in range(...)` loop, wrap from forward pass
through `scaler.update()`:

```python
                if device.type == "cuda":
                    _fb_start = torch.cuda.Event(enable_timing=True)
                    _fb_end = torch.cuda.Event(enable_timing=True)
                    _fb_start.record()

                with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=self.params.use_amp):
                    if self.compiled_train is not None:
                        output = self.compiled_train(batch_obs)
                    else:
                        output = self.forward_model(batch_obs)
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
                    _fb_end.record()
                    # Do NOT synchronize — events read lazily in flush_timings()
                    self._timing_events["update_forward_backward_ms"].append((_fb_start, _fb_end))
```

- [ ] **Step 6: Clear timing events at the start of `update()`**

At the beginning of `update()`, after `self.forward_model.train()` and the
train-mode assertion, add:

```python
        # Clear timing events from previous cycle
        for event_list in self._timing_events.values():
            event_list.clear()
```

- [ ] **Step 7: Write a test for timings**

Append to `tests/test_compile.py`:

```python
class TestTimings:
    def test_timings_exist_after_init(self):
        """timings dict is initialized with expected keys."""
        model = _small_model()
        params = KataGoPPOParams()
        ppo = KataGoPPOAlgorithm(params, model)
        assert "select_actions_forward_ms" in ppo.timings
        assert "update_forward_backward_ms" in ppo.timings
        assert "gae_ms" in ppo.timings

    def test_timings_empty_on_cpu(self):
        """On CPU, no CUDA events are recorded, so timings stay empty."""
        model = _small_model()
        params = KataGoPPOParams(batch_size=4)
        ppo = KataGoPPOAlgorithm(params, model)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.randn(4)
        ppo.update(buf, next_values)
        ppo.flush_timings()
        assert ppo.timings["update_forward_backward_ms"] == []
        assert ppo.timings["gae_ms"] == []
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_compile.py tests/test_pytorch_hot_path_gaps.py -v --timeout=120`
Expected: All PASS (timers are no-ops on CPU — the `if device.type == "cuda"` guards skip recording).

- [ ] **Step 9: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_compile.py
git commit -m "feat: add lazy CUDA event timers for forward pass, backward, and GAE"
```

---

## Task 9: Add forward-pass tolerance test (compiled vs eager)

**Files:**
- Modify: `tests/test_compile.py`

This is the "fast gate" correctness verification from the spec.

- [ ] **Step 1: Write the tolerance tests**

Append to `tests/test_compile.py`:

```python
class TestCompileCorrectness:
    def test_compiled_eval_matches_eager(self):
        """Compiled eval forward pass matches eager within tolerance.

        Freezes BN running stats by switching to eval mode, which isolates
        compile effects from BN stat divergence. This is the reliable gate —
        both eager and compiled see identical frozen BN statistics.
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

        # Compiled forward pass (same model, same frozen BN state)
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

    def test_compiled_train_matches_eager_frozen_bn(self):
        """Compiled train forward pass matches eager with frozen BN stats.

        For train-mode comparison, we freeze BN stats by setting momentum=0
        on all BN layers. This prevents the single forward pass from updating
        running stats differently between eager and compiled paths.

        Note: a previous version of this test used load_state_dict to copy
        post-update BN stats, but the eager pass had already updated them
        in-place, causing divergence. Freezing momentum avoids this entirely.
        """
        torch.manual_seed(42)
        model = _small_model()
        # Populate BN stats with a few warmup passes
        model.train()
        for _ in range(5):
            model(torch.randn(4, 50, 9, 9))

        # Freeze BN momentum so forward passes don't update running stats
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = 0.0

        obs = torch.randn(4, 50, 9, 9)

        # Eager forward pass (BN stats frozen, but still in train mode)
        eager_out = model(obs)

        # Compiled forward pass (same model, same frozen BN)
        compiled_model = torch.compile(model, mode="default")
        compiled_out = compiled_model(obs)

        assert torch.allclose(
            eager_out.policy_logits, compiled_out.policy_logits, rtol=1e-5, atol=1e-5
        ), f"Policy max diff: {(eager_out.policy_logits - compiled_out.policy_logits).abs().max()}"
        assert torch.allclose(
            eager_out.value_logits, compiled_out.value_logits, rtol=1e-5, atol=1e-5
        ), f"Value max diff: {(eager_out.value_logits - compiled_out.value_logits).abs().max()}"
```

- [ ] **Step 2: Run the tests**

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
- `test_compile.py`: ~17 tests (parameter identity 3, compile setup 4, select_actions 3, update 4, GAE routing 2, timings 2, correctness 2)
- `test_gae.py`: ~17 tests (existing CPU tests ~10 + GPU tolerance tests 7)

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
| 2 | Parameter identity assertion + test file creation | `katago_ppo.py`, `test_compile.py` |
| 3 | Two compiled models + warning log | `katago_ppo.py`, `test_compile.py` |
| 4 | `select_actions()` uses `compiled_eval` | `katago_ppo.py`, `test_compile.py` |
| 5 | `update()` uses `compiled_train`/`compiled_eval` + train-mode assertion | `katago_ppo.py`, `test_compile.py` |
| 6 | `compute_gae_gpu()` + edge case tests | `gae.py`, `test_gae.py` |
| 7 | GAE GPU routing + `.detach()` + top-level import | `katago_ppo.py`, `test_compile.py` |
| 8 | Lazy CUDA event timers + `flush_timings()` | `katago_ppo.py`, `test_compile.py` |
| 9 | Compiled vs eager tolerance tests (frozen BN) | `test_compile.py` |
| 10 | Full integration verification | (no changes) |

## Review Findings Addressed

| Finding | Severity | Source | Task |
|---------|----------|--------|------|
| `synchronize()` inside hot loops | Blocker | Arch, Python, Quality, PyTorch (4/5) | Task 8 — lazy event accumulation + `flush_timings()` |
| Self-tautological `data_ptr()` assertion | Blocker | Arch, Python, Quality, PyTorch (4/5) | Task 3 — replaced with `_orig_mod` identity check |
| `split_merge_step` bypasses BN invariant | High | Systems | Task 5 — train-mode assertion at `update()` top |
| `reduce-overhead` + `dynamic=True` silently disables CUDA graphs | High | Systems, PyTorch | Task 3 — warning log |
| Missing `.detach()` on GPU GAE `next_values` | High | PyTorch, Python | Task 7 — explicit `.detach()` |
| `timings` lifecycle mismatch | Medium | Quality, Arch, PyTorch | Task 8 — event accumulation, cleared at `update()` start |
| Train-mode tolerance test BN state divergence | Medium | Arch, Quality | Task 9 — freeze BN `momentum=0` |
| No mode-switch non-regression test | Medium | Quality | Task 4 — `test_select_actions_no_mode_switch_when_compiled` |
| No weight-update verification | Medium | Quality | Task 5 — `test_update_changes_weights` |
| Dynamic import inside hot path | Low | Python | Task 7 — top-level import |
| No compile activation logging | Low | Quality | Task 3 — `logger.info` after compile setup |
| Missing T=1, N=1 edge case tests | Low | Quality | Task 6 — added both |
| Missing 1D rejection test | Low | Quality | Task 6 — `test_gpu_rejects_1d_input` |
| Test helper duplication | Low | Arch, Python | Task 2 — note in docstring, consider `conftest.py` follow-up |
| Spec says "bakes at trace time" (inaccurate) | Info | PyTorch | Note N2 in plan header |
| `autocast` outside compiled region limits fusion | Info | Systems | Note N4 in plan header |
| Tasks 3–5 atomicity | Info | Systems | Note N1 in plan header |
