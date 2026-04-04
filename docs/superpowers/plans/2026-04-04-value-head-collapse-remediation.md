# Value Head Collapse Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the PPO training collapse caused by truncation bootstrap bug and value head supervision starvation.

**Architecture:** Five remediations (R1-R5) deployed in two phases. Phase 1: R1 (correctness fix: separate terminated from truncated in GAE) + R4 (LR scheduler monitor fix). Phase 2: R2 (score blend into GAE) + R3 (lambda_score increase) + R5 (entropy annealing) via config only. All code changes are in the Python training loop — the Rust engine is untouched.

**Tech Stack:** Python 3.13, PyTorch, uv for package management. Run tests with `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-04-04-value-head-collapse-remediation-design.md`

---

## File Map

**Production files modified:**
- `keisei/training/gae.py` — R1: rename `dones` -> `terminated` in all 3 GAE functions
- `keisei/training/katago_ppo.py` — R1/R2/R5: buffer schema, params, select_actions, GAE call sites
- `keisei/training/katago_loop.py` — R1/R2/R4: PendingTransitions, buffer.add sites, value_adapter plumbing, LR monitor, observability
- `keisei/training/value_adapter.py` — R2: `scalar_value_blended` method, `score_blend_alpha` param
- `keisei-500k-league.toml` — R3/R5: production config values

**Test files modified/created:**
- `tests/test_gae.py` — R1: truncation bootstrap tests
- `tests/test_katago_ppo.py` — R1: buffer terminated field tests
- `tests/test_value_adapter.py` — R2: blend tests
- `tests/test_entropy_annealing.py` — R5: new file
- `tests/test_katago_loop.py` — R1: value_cats fix test

---

### Task 1: R5 — Entropy Annealing (simplest, standalone)

**Files:**
- Modify: `keisei/training/katago_ppo.py:54-70` (KataGoPPOParams) and `keisei/training/katago_ppo.py:309-317` (get_entropy_coeff)
- Create: `tests/test_entropy_annealing.py`

- [ ] **Step 1: Write failing tests for entropy annealing**

Create `tests/test_entropy_annealing.py`:

```python
"""Tests for smooth entropy annealing (R5)."""

import pytest
from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams
from unittest.mock import MagicMock


def _make_algo(lambda_entropy=0.01, entropy_decay_epochs=0, warmup_epochs=10, warmup_entropy=0.05):
    """Create a KataGoPPOAlgorithm with a mock model for entropy testing."""
    params = KataGoPPOParams(lambda_entropy=lambda_entropy, entropy_decay_epochs=entropy_decay_epochs)
    model = MagicMock()
    model.parameters.return_value = iter([])
    algo = KataGoPPOAlgorithm(
        params, model, warmup_epochs=warmup_epochs, warmup_entropy=warmup_entropy,
    )
    return algo


class TestEntropyAnnealing:
    def test_decay_zero_matches_step_behavior(self):
        """decay_epochs=0 should immediately return lambda_entropy after warmup."""
        algo = _make_algo(entropy_decay_epochs=0, warmup_epochs=10)
        assert algo.get_entropy_coeff(9) == 0.05   # still in warmup
        assert algo.get_entropy_coeff(10) == 0.01   # step to base
        assert algo.get_entropy_coeff(100) == 0.01

    def test_linear_decay_at_warmup_boundary(self):
        """First epoch after warmup should still return warmup_entropy."""
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(10) == 0.05

    def test_linear_decay_midpoint(self):
        """Midpoint of decay should return average of warmup and base."""
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10,
                         warmup_entropy=0.05, lambda_entropy=0.01)
        result = algo.get_entropy_coeff(110)  # epoch 10 + 100 = midpoint
        expected = 0.05 + 0.5 * (0.01 - 0.05)  # = 0.03
        assert abs(result - expected) < 1e-9

    def test_linear_decay_one_before_end(self):
        """One epoch before decay ends: not yet at base."""
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        result = algo.get_entropy_coeff(209)  # elapsed = 199
        assert result > 0.01  # not yet at base
        assert result < 0.05  # past warmup

    def test_linear_decay_at_end(self):
        """At decay_epochs: should return exactly lambda_entropy."""
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(210) == 0.01

    def test_linear_decay_past_end(self):
        """Past decay_epochs: should return lambda_entropy."""
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(500) == 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_entropy_annealing.py -v`
Expected: FAIL — `KataGoPPOParams` does not have `entropy_decay_epochs` field.

- [ ] **Step 3: Add `entropy_decay_epochs` to KataGoPPOParams**

In `keisei/training/katago_ppo.py`, in the `KataGoPPOParams` dataclass (after line 70):

```python
# Add after compile_dynamic field:
    entropy_decay_epochs: int = 0  # 0 = instant transition (backward compat); >0 = linear decay over N epochs
```

- [ ] **Step 4: Implement smooth `get_entropy_coeff()`**

In `keisei/training/katago_ppo.py`, replace the `get_entropy_coeff` method (lines 309-317):

```python
    def get_entropy_coeff(self, epoch: int) -> float:
        """Return the entropy coefficient for the current epoch.

        During warmup: elevated entropy to soften overconfident SL policy.
        After warmup: linear decay from warmup_entropy to lambda_entropy
        over entropy_decay_epochs (0 = instant transition).
        """
        if epoch < self.warmup_epochs:
            return self.warmup_entropy
        decay_epochs = self.params.entropy_decay_epochs
        if decay_epochs <= 0:
            return self.params.lambda_entropy
        elapsed = epoch - self.warmup_epochs
        if elapsed >= decay_epochs:
            return self.params.lambda_entropy
        t = elapsed / decay_epochs
        return self.warmup_entropy + t * (self.params.lambda_entropy - self.warmup_entropy)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_entropy_annealing.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Run full test suite for regression**

Run: `uv run pytest tests/ -x -q`
Expected: No regressions.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_entropy_annealing.py
git commit -m "feat(R5): smooth entropy annealing with linear decay

Add entropy_decay_epochs param for gradual entropy coefficient transition
from warmup_entropy to lambda_entropy. Default 0 preserves existing
step-function behavior."
```

---

### Task 2: R2 — Score Blend in Value Adapter (standalone, no buffer changes needed)

**Files:**
- Modify: `keisei/training/value_adapter.py`
- Modify: `keisei/training/katago_ppo.py:54-70` (KataGoPPOParams)
- Modify: `tests/test_value_adapter.py`

- [ ] **Step 1: Write failing tests for `scalar_value_blended`**

Append to `tests/test_value_adapter.py`:

```python
class TestScalarValueBlended:
    def test_alpha_zero_matches_wdl_only(self):
        """alpha=0 should return identical result to scalar_value_from_output."""
        adapter = MultiHeadValueAdapter(score_blend_alpha=0.0)
        value_logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        score_lead = torch.tensor([[0.5], [-0.3]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        wdl_only = adapter.scalar_value_from_output(value_logits)
        assert torch.allclose(blended, wdl_only)

    def test_alpha_one_uses_score_only(self):
        """alpha=1 should use only score_lead (clamped)."""
        adapter = MultiHeadValueAdapter(score_blend_alpha=1.0)
        value_logits = torch.tensor([[2.0, 0.0, 0.0]])
        score_lead = torch.tensor([[0.7]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        assert torch.allclose(blended, torch.tensor([0.7]))

    def test_alpha_half_is_arithmetic_mean(self):
        """alpha=0.5 should return average of WDL value and score."""
        adapter = MultiHeadValueAdapter(score_blend_alpha=0.5)
        # W/D/L logits = [10, 0, 0] => P(W)~1, P(L)~0 => WDL value ~ 1.0
        value_logits = torch.tensor([[10.0, 0.0, 0.0]])
        score_lead = torch.tensor([[0.0]])  # score = 0.0
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        wdl_value = adapter.scalar_value_from_output(value_logits)
        expected = 0.5 * wdl_value + 0.5 * 0.0
        assert torch.allclose(blended, expected, atol=1e-5)

    def test_extreme_score_clamped(self):
        """Score values outside [-1, 1] should be clamped."""
        adapter = MultiHeadValueAdapter(score_blend_alpha=1.0)
        value_logits = torch.tensor([[0.0, 0.0, 0.0]])
        score_lead = torch.tensor([[5.0]])  # extreme
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        assert blended.item() == pytest.approx(1.0)

    def test_negative_extreme_score_clamped(self):
        """Negative extreme scores clamped to -1."""
        adapter = MultiHeadValueAdapter(score_blend_alpha=1.0)
        value_logits = torch.tensor([[0.0, 0.0, 0.0]])
        score_lead = torch.tensor([[-5.0]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        assert blended.item() == pytest.approx(-1.0)

    def test_scalar_adapter_inherits_default(self):
        """ScalarValueAdapter should inherit the default (ignore score_lead)."""
        adapter = ScalarValueAdapter()
        value_logits = torch.tensor([[0.5], [-0.3]])
        score_lead = torch.tensor([[0.9], [0.1]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        expected = adapter.scalar_value_from_output(value_logits)
        assert torch.allclose(blended, expected)

    def test_get_value_adapter_passes_alpha(self):
        """get_value_adapter should pass score_blend_alpha through."""
        adapter = get_value_adapter("multi_head", score_blend_alpha=0.3)
        assert adapter.score_blend_alpha == 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_value_adapter.py::TestScalarValueBlended -v`
Expected: FAIL — `score_blend_alpha` not a valid param.

- [ ] **Step 3: Add `score_blend_alpha` to KataGoPPOParams**

In `keisei/training/katago_ppo.py`, in `KataGoPPOParams` (after the `entropy_decay_epochs` field added in Task 1):

```python
    score_blend_alpha: float = 0.0   # 0.0 = pure WDL; >0 blends score_lead into GAE value
    use_terminated_for_gae: bool = True  # R1 feature flag (also needed later, add now)
```

- [ ] **Step 4: Implement `scalar_value_blended` on value adapters**

In `keisei/training/value_adapter.py`:

On `ValueHeadAdapter` ABC (after `compute_value_loss`):

```python
    def scalar_value_blended(
        self, value_logits: torch.Tensor, score_lead: torch.Tensor,
    ) -> torch.Tensor:
        """Blend W/D/L value with score prediction for GAE.

        Default: ignore score_lead, use W/D/L only. Override in subclasses
        that have a score head and want blending.

        Args:
            value_logits: (batch, 3) W/D/L logits or (batch, 1) scalar value.
            score_lead: (batch, 1) score prediction. Ignored in default impl.
        """
        return self.scalar_value_from_output(value_logits)
```

On `MultiHeadValueAdapter.__init__`, add `score_blend_alpha`:

```python
    def __init__(self, lambda_value: float = 1.5, lambda_score: float = 0.02,
                 score_blend_alpha: float = 0.0) -> None:
        self.lambda_value = lambda_value
        self.lambda_score = lambda_score
        self.score_blend_alpha = score_blend_alpha
```

On `MultiHeadValueAdapter`, add the override:

```python
    def scalar_value_blended(
        self, value_logits: torch.Tensor, score_lead: torch.Tensor,
    ) -> torch.Tensor:
        """Blend W/D/L value with score prediction for GAE.

        Args:
            value_logits: (batch, 3) W/D/L logits.
            score_lead: (batch, 1) normalized material balance. Squeezed to
                (batch,) and clamped to [-1, 1].
        """
        wdl_value = self.scalar_value_from_output(value_logits)
        score_value = score_lead.squeeze(-1).clamp(-1, 1)
        alpha = self.score_blend_alpha
        if alpha == 0.0:
            return wdl_value
        return (1 - alpha) * wdl_value + alpha * score_value
```

On `get_value_adapter()`, add `score_blend_alpha` parameter:

```python
def get_value_adapter(
    model_contract: str,
    lambda_value: float = 1.5,
    lambda_score: float = 0.02,
    score_blend_alpha: float = 0.0,
) -> ValueHeadAdapter:
    if model_contract == "scalar":
        return ScalarValueAdapter()
    elif model_contract == "multi_head":
        return MultiHeadValueAdapter(
            lambda_value=lambda_value, lambda_score=lambda_score,
            score_blend_alpha=score_blend_alpha,
        )
    else:
        raise ValueError(f"Unknown model contract: {model_contract}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_value_adapter.py -v`
Expected: All tests PASS (new and existing).

- [ ] **Step 6: Run full test suite for regression**

Run: `uv run pytest tests/ -x -q`
Expected: No regressions.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/value_adapter.py keisei/training/katago_ppo.py tests/test_value_adapter.py
git commit -m "feat(R2): add scalar_value_blended to value adapters

Concrete default on ABC ignores score_lead. MultiHeadValueAdapter
overrides with configurable alpha blend of P(W)-P(L) and score_lead.
score_blend_alpha=0.0 preserves existing behavior."
```

---

### Task 3: R1 Part A — GAE functions: rename `dones` to `terminated`

**Files:**
- Modify: `keisei/training/gae.py`
- Modify: `tests/test_gae.py`

- [ ] **Step 1: Write failing tests for GAE truncation bootstrap**

Append to `tests/test_gae.py`:

```python
class TestGAETruncationBootstrap:
    """R1: Verify that truncated episodes bootstrap V(s_next) instead of zeroing."""

    def test_truncated_bootstraps_value(self):
        """With terminated (not dones), truncated positions should bootstrap."""
        # 3 steps, 1 env. Step 1 is truncated (not terminated).
        rewards = torch.tensor([[0.0], [0.0], [1.0]])   # (T=3, N=1)
        values = torch.tensor([[0.5], [0.6], [0.7]])
        next_value = torch.tensor([0.8])

        # Old behavior: dones = terminated | truncated. Step 1 truncated -> dones=True
        old_dones = torch.tensor([[0.0], [1.0], [0.0]])
        adv_old = compute_gae_gpu(rewards, values, old_dones, next_value, gamma=0.99, lam=0.95)

        # New behavior: only truly terminated. Step 1 NOT terminated.
        terminated = torch.tensor([[0.0], [0.0], [0.0]])
        adv_new = compute_gae_gpu(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)

        # The advantage at step 1 should differ: old zeros V(s_next), new bootstraps.
        # Old: delta[1] = 0 + 0.99 * values[2] * 0 - 0.6 = -0.6 (bootstrap zeroed)
        # New: delta[1] = 0 + 0.99 * values[2] * 1 - 0.6 = 0.99*0.7 - 0.6 = 0.093
        assert abs(adv_old[1, 0].item() - (-0.6)) < 0.1  # approximately -0.6
        assert adv_new[1, 0].item() > 0  # positive because bootstrap works

    def test_backward_compat_no_truncation(self):
        """When no truncation, terminated == dones, results are bit-exact."""
        rewards = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
        values = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.7, 0.2]])
        terminated = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        next_value = torch.tensor([0.1, 0.5])

        adv_terminated = compute_gae_gpu(rewards, values, terminated, next_value,
                                         gamma=0.99, lam=0.95)
        adv_dones = compute_gae_gpu(rewards, values, terminated, next_value,
                                    gamma=0.99, lam=0.95)

        assert torch.allclose(adv_terminated, adv_dones)


class TestGAEPaddedTruncation:
    """R1: Truncation bootstrap for the padded GAE path (split-merge mode)."""

    def test_padded_truncated_bootstraps(self):
        from keisei.training.gae import compute_gae_padded
        # 2 envs, max_T=3. Env 0 has length 2 (step 1 truncated). Env 1 has length 3.
        rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
        values = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.0, 0.2]])
        # Env 0: step 1 truncated (not terminated). Padding at step 2.
        # Env 1: no truncation/termination.
        terminated = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])  # padding for env 0 at t=2
        # For padded path, padding positions must have dones=1.
        # We pass terminated for the padded positions too.
        terminated_pad = terminated.clone()
        terminated_pad[2, 0] = 1.0  # padding = "done" to zero GAE beyond valid range
        next_values = torch.tensor([0.3, 0.8])
        lengths = torch.tensor([2, 3])

        adv = compute_gae_padded(rewards, values, terminated_pad, next_values, lengths,
                                 gamma=0.99, lam=0.95)
        # Env 0, step 1 (last valid): should bootstrap from next_values[0]=0.3
        # delta = 0.0 + 0.99 * 0.3 - 0.4 = -0.103 (not zeroed)
        assert adv[1, 0].item() != 0.0  # would be ~0 if bootstrap was zeroed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_gae.py::TestGAETruncationBootstrap -v`
Expected: PASS (GAE already works with `terminated` parameter — it's just named `dones`). These tests should actually PASS already since we're passing the correct tensor. The key validation is that the OLD behavior (passing merged dones) gives different results. Let's verify.

Run: `uv run pytest tests/test_gae.py -v`
Expected: All PASS (these are documenting expected behavior).

- [ ] **Step 3: Rename `dones` to `terminated` in all GAE functions**

In `keisei/training/gae.py`, make these changes:

`compute_gae` (line 8): Change parameter `dones` to `terminated`, update docstring and line 43.
`compute_gae_padded` (line 51): Change parameter `dones` to `terminated`, update docstring and line 97.
`compute_gae_gpu` (line 105): Change parameter `dones` to `terminated`, update docstring and line 144.

For each function, change `not_done = 1.0 - dones[t].float()` to `not_done = 1.0 - terminated[t].float()` (or the non-indexed equivalent for `compute_gae_gpu`).

Update all docstrings from `dones: episode termination flags` to:
```
terminated: True only for genuinely terminal episodes (not truncated).
    Truncated episodes bootstrap from V(s_next) instead of zeroing it.
```

- [ ] **Step 4: Update existing test imports/calls**

In `tests/test_gae.py`, update all existing tests that pass `dones=` keyword to pass `terminated=` instead. For positional calls, the rename is transparent. Search for `dones` in the test file and update keyword arguments.

- [ ] **Step 5: Run all GAE tests**

Run: `uv run pytest tests/test_gae.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/gae.py tests/test_gae.py
git commit -m "refactor(R1): rename dones->terminated in GAE functions

Semantic rename to clarify that GAE's not_done mask should only reflect
truly terminal episodes, not truncated ones. Truncated episodes now
bootstrap V(s_next) instead of zeroing it."
```

---

### Task 4: R1 Part B — Buffer schema: add `terminated` field

**Files:**
- Modify: `keisei/training/katago_ppo.py:82-202` (KataGoRolloutBuffer)
- Modify: `tests/test_katago_ppo.py`

- [ ] **Step 1: Write failing test for buffer terminated field**

Add to `tests/test_katago_ppo.py` (find the buffer test class, or add a new one):

```python
class TestBufferTerminatedField:
    def test_add_stores_terminated(self):
        """Buffer should store terminated separately from dones."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(4, 9, 9), action_space=100)
        obs = torch.zeros(2, 4, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        rewards = torch.zeros(2)
        dones = torch.tensor([1.0, 1.0])      # both "done" (terminated | truncated)
        terminated = torch.tensor([1.0, 0.0])  # only env 0 truly terminated
        legal_masks = torch.zeros(2, 100, dtype=torch.bool)
        legal_masks[:, 0] = True  # at least one legal action
        value_cats = torch.tensor([-1, -1], dtype=torch.long)
        score_targets = torch.tensor([0.1, -0.1])

        buf.add(obs, actions, log_probs, values, rewards, dones,
                terminated, legal_masks, value_cats, score_targets)

        data = buf.flatten()
        assert "terminated" in data
        assert data["terminated"].shape == (2,)
        assert data["terminated"][0].item() == 1.0
        assert data["terminated"][1].item() == 0.0
        # dones should still be present
        assert "dones" in data
        assert data["dones"][0].item() == 1.0
        assert data["dones"][1].item() == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_ppo.py::TestBufferTerminatedField -v`
Expected: FAIL — `buffer.add()` does not accept `terminated` parameter.

- [ ] **Step 3: Add `terminated` to buffer**

In `keisei/training/katago_ppo.py`:

In `clear()` (line 89), add after `self.dones`:
```python
        self.terminated: list[torch.Tensor] = []
```

In `add()` (line 105), change the signature — add `terminated: torch.Tensor` after `dones`:
```python
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
```

After `dones_cpu = dones.detach().cpu()` (line 134), add:
```python
        terminated_cpu = terminated.detach().cpu()
```

After `self.dones.append(dones_cpu)` (line 175), add:
```python
        self.terminated.append(terminated_cpu)
```

In `flatten()` (line 182), add to the result dict after `"dones"`:
```python
            "terminated": torch.cat(self.terminated, dim=0).reshape(-1),
```

- [ ] **Step 4: Fix all existing `buffer.add()` calls in tests**

Search test files for `buf.add(` or `buffer.add(` calls and add the `terminated` argument. For existing tests where `terminated == dones`, pass `dones` as both arguments. This is a mechanical change — every existing call that passes `dones` now also passes `terminated=dones` (or the equivalent positional arg).

Run: `uv run pytest tests/test_katago_ppo.py -v` to find which tests break and fix them one by one.

- [ ] **Step 5: Run tests to verify buffer test passes**

Run: `uv run pytest tests/test_katago_ppo.py::TestBufferTerminatedField -v`
Expected: PASS.

- [ ] **Step 6: Run full test suite to find remaining call sites**

Run: `uv run pytest tests/ -x -q`
Expected: Some tests may fail if they call `buffer.add()` without the new `terminated` param. Fix each by adding `terminated=dones` (same tensor) as a positional arg after `dones`.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_ppo.py tests/
git commit -m "feat(R1): add terminated field to KataGoRolloutBuffer

Buffer now stores terminated separately from dones (terminated | truncated).
GAE will use terminated for bootstrap gating. Existing tests pass
terminated=dones for backward compatibility."
```

---

### Task 5: R1 Part C — PendingTransitions.finalize() threading

**Files:**
- Modify: `keisei/training/katago_loop.py:198-235` (PendingTransitions.finalize)
- Modify: `tests/test_katago_loop.py` (or a test file that covers PendingTransitions)

- [ ] **Step 1: Write failing test for finalize with three populations**

Add to the appropriate test file:

```python
class TestPendingTransitionsTerminated:
    """R1: finalize() must return terminated separately from dones."""

    def test_finalize_three_populations(self):
        from keisei.training.katago_loop import PendingTransitions
        device = torch.device("cpu")
        pt = PendingTransitions(num_envs=3, obs_shape=(4, 9, 9), action_space=100, device=device)

        # Create pending transitions for all 3 envs
        env_mask = torch.ones(3, dtype=torch.bool, device=device)
        pt.create(
            env_mask,
            obs=torch.zeros(3, 4, 9, 9, device=device),
            actions=torch.zeros(3, dtype=torch.long, device=device),
            log_probs=torch.zeros(3, device=device),
            values=torch.zeros(3, device=device),
            legal_masks=torch.zeros(3, 100, dtype=torch.bool, device=device),
            rewards=torch.zeros(3, device=device),
            score_targets=torch.zeros(3, device=device),
        )

        # Population A: env 0 = terminated (game ended)
        # Population B: env 1 = truncated (hit max_ply)
        # Population C: env 2 = neither (epoch-end flush)
        dones = torch.tensor([1.0, 1.0, 0.0], device=device)           # A=done, B=done, C=not
        terminated = torch.tensor([1.0, 0.0, 0.0], device=device)      # A=terminated, B=not, C=not

        finalize_mask = torch.ones(3, dtype=torch.bool, device=device)
        result = pt.finalize(finalize_mask, dones, terminated)

        assert result is not None
        assert "dones" in result
        assert "terminated" in result

        # Population A: terminated game
        assert result["dones"][0].item() == 1.0
        assert result["terminated"][0].item() == 1.0

        # Population B: truncated game
        assert result["dones"][1].item() == 1.0
        assert result["terminated"][1].item() == 0.0

        # Population C: epoch-end flush
        assert result["dones"][2].item() == 0.0
        assert result["terminated"][2].item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestPendingTransitionsTerminated -v` (or wherever you placed it)
Expected: FAIL — `finalize()` does not accept `terminated` parameter.

- [ ] **Step 3: Update `finalize()` signature and result dict**

In `keisei/training/katago_loop.py`, modify `PendingTransitions.finalize()`:

Change signature (line 198):
```python
    def finalize(
        self,
        finalize_mask: torch.Tensor,
        dones: torch.Tensor,
        terminated: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
```

In the result dict (line 219), add `"terminated"`:
```python
        result = {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "log_probs": self.log_probs[indices],
            "values": self.values[indices],
            "rewards": self.rewards[indices],
            "dones": dones[indices].float(),
            "terminated": terminated[indices].float(),
            "legal_masks": self.legal_masks[indices],
            "score_targets": self.score_targets[indices],
            "env_ids": indices,
        }
```

- [ ] **Step 4: Fix all `finalize()` call sites**

In `keisei/training/katago_loop.py`, update every call to `pending.finalize()`:

Line 833 (split-merge finalize):
```python
finalized = pending.finalize(finalize_mask, dones, terminated)
```

Line 872 (immediate terminal):
```python
imm_finalized = pending.finalize(imm_terminal, dones, terminated)
```

Line 951 (epoch-end flush):
```python
remaining_terminated = torch.zeros(self.num_envs, device=self.device)
remaining = pending.finalize(remaining_mask, remaining_dones, remaining_terminated)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_katago_loop.py::TestPendingTransitionsTerminated -v`
Expected: PASS.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: Some tests calling `pending.finalize()` may need the extra arg. Fix by passing `terminated=dones` for existing tests.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_loop.py tests/
git commit -m "feat(R1): thread terminated through PendingTransitions.finalize()

finalize() now returns both dones and terminated in its result dict.
Three call sites updated: split-merge finalize, immediate terminal,
and epoch-end flush."
```

---

### Task 6: R1 Part D — Fix `_compute_value_cats` call sites + `buffer.add()` sites + `terminal_mask`

**Files:**
- Modify: `keisei/training/katago_loop.py` (lines 797, 833-845, 870-885, 901, 925-936, 957-964)

- [ ] **Step 1: Write failing test for `_compute_value_cats` with truncation**

Add to tests:

```python
class TestComputeValueCatsTerminated:
    """R1: _compute_value_cats must use terminated, not dones."""

    def test_truncated_game_gets_ignore_not_draw(self):
        from keisei.training.katago_loop import _compute_value_cats
        device = torch.device("cpu")
        rewards = torch.tensor([0.0, 1.0, 0.0])
        # Env 0: truncated (dones=True, terminated=False, reward=0)
        # Env 1: won (dones=True, terminated=True, reward=1)
        # Env 2: still playing (dones=False, terminated=False)
        terminated = torch.tensor([False, True, False])

        cats = _compute_value_cats(rewards, terminated, device)

        assert cats[0].item() == -1  # truncated = IGNORE, not draw
        assert cats[1].item() == 0   # win
        assert cats[2].item() == -1  # still playing = ignore
```

- [ ] **Step 2: Run test to verify it passes**

This test should already PASS — we're passing `terminated` as the `dones_bool` arg. The function itself doesn't change; the fix is at the call sites. This test documents the correct behavior.

Run: `uv run pytest tests/test_katago_loop.py::TestComputeValueCatsTerminated -v`
Expected: PASS.

- [ ] **Step 3: Fix all `_compute_value_cats` call sites**

In `keisei/training/katago_loop.py`:

Line 836-837 (split-merge finalize):
```python
# Before:
f_value_cats = _compute_value_cats(
    finalized["rewards"], finalized["dones"].bool(), self.device,
)
# After:
f_value_cats = _compute_value_cats(
    finalized["rewards"], finalized["terminated"].bool(), self.device,
)
```

Line 874-875 (immediate terminal):
```python
# Before:
imm_value_cats = _compute_value_cats(
    imm_finalized["rewards"], imm_finalized["dones"].bool(), self.device,
)
# After:
imm_value_cats = _compute_value_cats(
    imm_finalized["rewards"], imm_finalized["terminated"].bool(), self.device,
)
```

Line 925 (non-split-merge — this is the `terminal_mask` path):
```python
# Before:
terminal_mask = dones.bool()
...
value_cats = _compute_value_cats(rewards, terminal_mask, self.device)
# After:
terminal_mask = terminated.bool()
...
value_cats = _compute_value_cats(rewards, terminal_mask, self.device)
```

- [ ] **Step 4: Fix `terminal_mask` for win/loss/draw counting**

Line 797 (split-merge):
```python
# Before:
terminal_mask = dones.bool()
# After:
terminal_mask = terminated.bool()
```

Line 901 (non-split-merge):
```python
# Before:
terminal_mask = dones.bool()
# After:
terminal_mask = terminated.bool()
```

- [ ] **Step 5: Fix all `buffer.add()` call sites to pass `terminated`**

Line 839-846 (split-merge finalize):
```python
self.buffer.add(
    finalized["obs"], finalized["actions"],
    finalized["log_probs"], finalized["values"],
    finalized["rewards"], finalized["dones"],
    finalized["terminated"],
    finalized["legal_masks"], f_value_cats,
    finalized["score_targets"],
    env_ids=finalized["env_ids"],
)
```

Line 878-885 (immediate terminal):
```python
self.buffer.add(
    imm_finalized["obs"], imm_finalized["actions"],
    imm_finalized["log_probs"], imm_finalized["values"],
    imm_finalized["rewards"], imm_finalized["dones"],
    imm_finalized["terminated"],
    imm_finalized["legal_masks"], imm_value_cats,
    imm_finalized["score_targets"],
    env_ids=imm_finalized["env_ids"],
)
```

Line 934-937 (non-split-merge):
```python
self.buffer.add(
    obs, actions, log_probs, values, rewards, dones,
    terminated,
    legal_masks, value_cats, score_targets,
)
```

Line 957-964 (epoch-end flush):
```python
self.buffer.add(
    remaining["obs"], remaining["actions"],
    remaining["log_probs"], remaining["values"],
    remaining["rewards"], remaining["dones"],
    remaining["terminated"],
    remaining["legal_masks"], remaining_value_cats,
    remaining["score_targets"],
    env_ids=remaining["env_ids"],
)
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: Fix any remaining call-site mismatches. The training loop tests that mock VecEnv will need `terminated` in their step results.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_loop.py tests/
git commit -m "feat(R1): fix _compute_value_cats and buffer.add call sites

All _compute_value_cats calls now use terminated (not dones), preventing
truncated games from being mislabeled as draws. All buffer.add() calls
pass terminated as a separate argument. terminal_mask for win/loss/draw
counting uses terminated."
```

---

### Task 7: R1 Part E — GAE call sites in `update()` with feature flag

**Files:**
- Modify: `keisei/training/katago_ppo.py:442-522` (update method, all 3 GAE paths)

- [ ] **Step 1: Update GAE call sites in `update()` with feature flag**

In `keisei/training/katago_ppo.py`, in the `update()` method:

After `data = buffer.flatten()` (line 433), add:
```python
        gae_dones_key = "terminated" if self.params.use_terminated_for_gae else "dones"
```

Path 1 — vectorized (line 446):
```python
# Before:
            dones_2d = data["dones"].reshape(T, N)
# After:
            terminated_2d = data[gae_dones_key].reshape(T, N)
```

Update the `compute_gae_gpu` call (line 458):
```python
                advantages = compute_gae_gpu(
                    rewards_2d.to(device), values_2d.to(device),
                    terminated_2d.to(device), next_values.detach(),
                    gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1).cpu()
```

And the CPU `compute_gae` call (line 464):
```python
                advantages = compute_gae(
                    rewards_2d, values_2d, terminated_2d,
                    next_values_cpu, gamma=self.params.gamma, lam=self.params.gae_lambda,
                ).reshape(-1)
```

Path 2 — split-merge padded (line 488-501):
```python
# Rename env_dones -> env_terminated throughout:
            env_terminated = []
            ...
            for env_id in unique_envs:
                mask = env_ids == env_id
                ...
                env_terminated.append(data[gae_dones_key][mask])
                ...

            terminated_pad = torch.ones(max_T, N_env)  # padding = terminated to zero GAE
            ...
            for j, length in enumerate(env_lengths):
                ...
                terminated_pad[:length, j] = env_terminated[j]
```

Pass `terminated_pad` to `compute_gae_padded`.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add keisei/training/katago_ppo.py
git commit -m "feat(R1): use terminated (not dones) for GAE in update()

Feature-flagged via use_terminated_for_gae param. All three GAE paths
(vectorized, padded, CPU) now use the terminated signal for bootstrap
gating. Truncated episodes correctly bootstrap V(s_next)."
```

---

### Task 8: R2 Part B — Plumb `value_adapter` through training loop

**Files:**
- Modify: `keisei/training/katago_loop.py` (init, split_merge_step call, bootstrap, update call)
- Modify: `keisei/training/katago_ppo.py:345-404` (select_actions)

- [ ] **Step 1: Store `value_adapter` on KataGoTrainingLoop**

In `keisei/training/katago_loop.py`, in `__init__()`, after `self.ppo = KataGoPPOAlgorithm(...)` (around line 445):

```python
        from keisei.training.value_adapter import get_value_adapter
        self.value_adapter = get_value_adapter(
            model_contract="multi_head",
            lambda_value=ppo_params.lambda_value,
            lambda_score=ppo_params.lambda_score,
            score_blend_alpha=ppo_params.score_blend_alpha,
        )
```

- [ ] **Step 2: Pass `value_adapter` to `split_merge_step()`**

At line 773, add `value_adapter=self.value_adapter`:
```python
                    sm_result = split_merge_step(
                        obs=obs, legal_masks=legal_masks,
                        current_players=current_players,
                        learner_model=self.model,
                        opponent_model=self._current_opponent,
                        learner_side=learner_side,
                        value_adapter=self.value_adapter,
                    )
```

Inside `split_merge_step()` at line 284, change to blended:
```python
            learner_values = value_adapter.scalar_value_blended(l_output.value_logits, l_output.score_lead)
```

- [ ] **Step 3: Add `value_adapter` param to `select_actions()`**

In `keisei/training/katago_ppo.py`, modify `select_actions` (line 345):

```python
    def select_actions(
        self, obs: torch.Tensor, legal_masks: torch.Tensor,
        value_adapter: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

At line 401-402, change the value computation:
```python
            # Scalar value for GAE — uses blended projection if adapter available
            if value_adapter is not None:
                scalar_values = value_adapter.scalar_value_blended(output.value_logits, output.score_lead)
            else:
                scalar_values = self.scalar_value(output.value_logits)
```

- [ ] **Step 4: Pass `value_adapter` to `select_actions()` in non-split-merge path**

At line 888 in `katago_loop.py`:
```python
# Before:
                    actions, log_probs, values = self.ppo.select_actions(obs, legal_masks)
# After:
                    actions, log_probs, values = self.ppo.select_actions(
                        obs, legal_masks, value_adapter=self.value_adapter,
                    )
```

- [ ] **Step 5: Use blended value for bootstrap**

At line 974:
```python
# Before:
                next_values = KataGoPPOAlgorithm.scalar_value(output.value_logits)
# After:
                next_values = self.value_adapter.scalar_value_blended(
                    output.value_logits, output.score_lead,
                )
```

- [ ] **Step 6: Pass `value_adapter` to `ppo.update()`**

At line 995:
```python
            losses = self.ppo.update(
                self.buffer, next_values,
                value_adapter=self.value_adapter,
                heartbeat_fn=self._maybe_update_heartbeat,
            )
```

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS. Tests that construct `KataGoTrainingLoop` may need mocking adjustments.

- [ ] **Step 8: Commit**

```bash
git add keisei/training/katago_loop.py keisei/training/katago_ppo.py
git commit -m "feat(R2): plumb value_adapter through training loop

Store value_adapter on KataGoTrainingLoop. Pass to split_merge_step,
select_actions, bootstrap, and update. All value projections now use
scalar_value_blended (no-op at alpha=0.0)."
```

---

### Task 9: R4 — Fix LR Scheduler Monitor

**Files:**
- Modify: `keisei/training/katago_loop.py:1015-1032`

- [ ] **Step 1: Fix the monitor key**

At line 1016:
```python
# Before:
                monitor_value = losses.get("value_loss")
                if monitor_value is not None:
# After:
                monitor_value = losses.get("policy_loss")
                if monitor_value is None:
                    raise RuntimeError(
                        "LR scheduler expects 'policy_loss' in losses dict but it was absent. "
                        "Check that ppo.update() returns 'policy_loss'."
                    )
```

Update the log message at line 1031:
```python
                        logger.info("LR reduced: %.6f -> %.6f (monitor=policy_loss, value=%.4f)",
                                    old_lr, new_lr, monitor_value)
```

- [ ] **Step 2: Run test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "fix(R4): LR scheduler monitors policy_loss instead of value_loss

value_loss trends to 0 during collapse (W/D/L head starvation), which
the scheduler misinterprets as improvement. policy_loss is a more
reliable training health signal."
```

---

### Task 10: Observability — Logging additions

**Files:**
- Modify: `keisei/training/katago_loop.py` (rollout loop, post-GAE, bootstrap)

- [ ] **Step 1: Add terminated/truncated counting in rollout loop**

Near the top of the epoch loop, initialize counters:
```python
            terminated_count = 0
            truncated_count = 0
```

Where `terminated` and `truncated` tensors are available (around line 791 for split-merge, line 900 for non-split-merge), accumulate:
```python
                    terminated_count += terminated.bool().sum().item()
                    truncated_count += (truncated.bool() & ~terminated.bool()).sum().item()
```

After the rollout loop, log:
```python
            total_transitions = self.buffer.size * self.num_envs if not pending else self.buffer.size
            logger.info(
                "Epoch %d: %d terminated, %d truncated (bootstrapped)",
                epoch_i, terminated_count, truncated_count,
            )
```

- [ ] **Step 2: Log entropy coefficient every epoch**

After the `self.ppo.current_entropy_coeff = ...` line (around line 986):
```python
            logger.info("Epoch %d: entropy_coeff=%.4f", epoch_i, self.ppo.current_entropy_coeff)
```

- [ ] **Step 3: Run test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "feat: add observability logging for terminated/truncated counts

Log per-epoch counts of terminated vs truncated transitions, and
entropy coefficient at every epoch. Enables monitoring R1 effectiveness."
```

---

### Task 11: Config Updates

**Files:**
- Modify: `keisei-500k-league.toml`

- [ ] **Step 1: Update production config**

In `keisei-500k-league.toml`, in `[training.algorithm_params]`:

```toml
# Phase 1 (R1 + R4)
use_terminated_for_gae = true

# Phase 2 (R2 + R3 + R5) — uncomment when ready
# score_blend_alpha = 0.3
# lambda_score = 0.1
# entropy_decay_epochs = 200
```

Add warmup section if not present:
```toml
[training.algorithm_params.rl_warmup]
epochs = 50
entropy_bonus = 0.05
```

- [ ] **Step 2: Commit**

```bash
git add keisei-500k-league.toml
git commit -m "config: Phase 1 settings for value head collapse fix

Enable use_terminated_for_gae. Phase 2 params (score_blend_alpha,
lambda_score, entropy_decay_epochs) commented out for staged deploy."
```

---

### Task 12: Final Regression Test

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: All tests PASS with no regressions.

- [ ] **Step 2: Verify no stale `dones` references in GAE path**

```bash
grep -n "data\[.dones.\]" keisei/training/katago_ppo.py
```

Expected: Only in the `gae_dones_key` fallback path (feature flag off), nowhere else in the GAE computation.

```bash
grep -n 'finalized\["dones"\]\.bool()' keisei/training/katago_loop.py
```

Expected: Zero matches. All `_compute_value_cats` calls should use `finalized["terminated"].bool()`.

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A && git commit -m "fix: address final regression test findings"
```

Only if there were fixes. Otherwise skip.
