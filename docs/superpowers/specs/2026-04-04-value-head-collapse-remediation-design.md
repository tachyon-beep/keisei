# Value Head Collapse Remediation

**Date**: 2026-04-04
**Status**: Reviewed (incorporated feedback from Python, QA, and Systems reviews)
**Problem**: Training collapses to degenerate strategies after ~350-400 epochs due to value head supervision starvation and a truncation bootstrap correctness bug.

## Background

### Observed Symptoms (from `keisei-500k-league.db`)

| Phase | Entropy | Draw% | Trunc% | Grad Norm | Value Loss | Ep Length |
|-------|---------|-------|--------|-----------|------------|-----------|
| Early (0-199) | 3.76 | 12.4% | 24.4% | 16.5 | 0.346 | 350 |
| Peak (200-399) | 3.48 | 17.0% | 17.3% | 13.9 | 0.241 | 308 |
| Collapse (400-599) | 3.09 | 6.7% | 45.7% | 4.9 | 0.095 | 370 |
| Late (600+) | 3.14 | 8.3% | 34.4% | 6.3 | 0.137 | 356 |

Sharp transition at epoch 368-410: value_loss drops to 0.0, grad_norm crashes to 1.6, truncation spikes to 80%.

### Root Causes

1. **Truncation bootstrap bug**: `katago_loop.py:791` merges `dones = terminated | truncated`. GAE zeros V(s_next) for truncated games, treating them as terminal. With 45% truncation, nearly half the training data has mathematically zero advantage signal.

2. **W/D/L supervision starvation**: The value head (W/D/L cross-entropy) only trains on terminal states (~0.15% of data at collapse). The GAE scalar value `P(W) - P(L)` depends entirely on this head, creating a death spiral: poor values -> poor advantages -> poor policy -> longer games -> fewer terminals -> less supervision.

3. **Dense signal ignored by GAE**: The score head (material balance MSE) trains on 100% of data but is unused for advantage estimation and weighted at 0.02 (vs 1.5 for W/D/L).

4. **LR scheduler monitors collapsed signal**: `ReduceLROnPlateau` monitors `value_loss`, which trends to 0 during collapse — interpreted as improvement rather than failure.

5. **Hard entropy cutoff**: `get_entropy_coeff()` jumps from `warmup_entropy` to `lambda_entropy` at a single epoch boundary.

### System Archetype

This collapse matches the **"Success to the Successful"** reinforcing archetype. The W/D/L head (losing competitor) gets less signal over time while the score head (winning competitor, trained on 100% of data) is structurally locked out of the advantage computation. R2 breaks this archetype by giving the score head a voice.

## Deployment Strategy

**Deploy in two phases, not all at once.**

Rationale: R1 alone may be sufficient to break the death spiral. If we deploy all five simultaneously and training succeeds, we cannot attribute which intervention was critical. If it fails, we cannot isolate which caused the problem. R2-R5 are all config-gated and can be enabled without code changes.

**Phase 1**: R1 (code: truncation bootstrap fix) + R4 (code: LR scheduler monitor fix).
- Run 50-100 epochs from checkpoint.
- **Why R4 must accompany R1**: Once R1 is active, value_loss will increase (more valid gradient signal). If R4 is not deployed, the scheduler will interpret this increase as degradation and cut the learning rate.
- **Success criteria**: `value_loss > 0.05` at every epoch, `advantages.std() > 0.01`, truncated-position advantages have non-zero mean.

**Phase 2**: Enable R2 + R3 + R5 via config changes only (no code deployment).
- Set `score_blend_alpha = 0.3`, `lambda_score = 0.1`, `entropy_decay_epochs = 200`.
- Consider starting with `score_blend_alpha = 0.1` and increasing if the score head's predictions are well-calibrated.

## Remediations

### R1: Separate `terminated` from `truncated` in GAE

**Priority**: 1 (correctness bug)
**Risk**: HIGH (touches buffer schema, GAE, PendingTransitions, and 3+ buffer.add() call sites; silent-failure mode if any site is missed)
**Rollback**: Config flag `use_terminated_for_gae` (see below)

#### Problem

`katago_loop.py:791`:
```python
dones = terminated | truncated
```

This merged `dones` flows to `buffer.add()`, `_compute_value_cats()`, and all three GAE functions. GAE uses `not_done = 1.0 - dones.float()` to gate the bootstrap value V(s_next). For truncated games (not truly over), V(s_next) should NOT be zeroed — the game continues, we just stopped collecting.

Additionally, `_compute_value_cats()` uses `dones` to assign W/D/L labels. Truncated games have reward=0, so they get labeled as **draws** (category 1). This feeds wrong targets to the W/D/L head — a truncated game is not a draw, it is an unfinished game. This must also be fixed.

#### Feature Flag

Add `use_terminated_for_gae: bool = True` to `KataGoPPOParams`. This gates R1 at the GAE call sites:

```python
# In update(), choose which done signal to use for GAE:
gae_dones_key = "terminated" if self.params.use_terminated_for_gae else "dones"
```

Setting `use_terminated_for_gae = false` in config reverts R1 without code changes. This is the only remediation that cannot otherwise be disabled by config.

#### Changes (exhaustive list)

**`KataGoPPOParams`** (`katago_ppo.py:50-70`):
- Add field: `use_terminated_for_gae: bool = True`

**`KataGoRolloutBuffer`** (`katago_ppo.py:82-98`):
- Add `self.terminated: list[torch.Tensor] = []` to `clear()` (line 89).
- Existing `self.dones` is retained — it stores `terminated | truncated` and is used ONLY for backward-compat scenarios when `use_terminated_for_gae=False`.

**`buffer.add()`** (`katago_ppo.py:105`):
- Add `terminated: torch.Tensor` parameter after `dones`.
- Store `terminated_cpu = terminated.detach().cpu()` in `self.terminated`.
- Updated signature:
  ```python
  def add(self, obs, actions, log_probs, values, rewards, dones,
          terminated, legal_masks, value_categories, score_targets,
          env_ids=None):
  ```

**`buffer.flatten()`** (`katago_ppo.py:182`):
- Add to result dict: `"terminated": torch.cat(self.terminated, dim=0).reshape(-1),`

**`PendingTransitions.finalize()`** (`katago_loop.py:198-235`):
- Change signature from `finalize(self, finalize_mask, dones)` to:
  ```python
  def finalize(self, finalize_mask: torch.Tensor, dones: torch.Tensor,
               terminated: torch.Tensor) -> dict[str, torch.Tensor] | None:
  ```
- Add to result dict at line 225:
  ```python
  "dones": dones[indices].float(),
  "terminated": terminated[indices].float(),
  ```

**`_compute_value_cats()`** (`katago_loop.py:64-79`):
- No code change to the function itself — but ALL CALL SITES must pass `terminated` instead of `dones` as the `dones_bool` argument:
  - Line 837: `_compute_value_cats(finalized["rewards"], finalized["terminated"].bool(), self.device)`
  - Line 875: `_compute_value_cats(imm_finalized["rewards"], imm_finalized["terminated"].bool(), self.device)`
  - Line 925 (non-split-merge): `_compute_value_cats(rewards, terminated.bool(), self.device)` (was `terminal_mask` which came from `dones.bool()`)
  - Line 953-954 (epoch-end flush): Already uses `torch.full(..., -1, ...)` — no change needed (these are non-terminal by definition).

**`katago_loop.py` — call site: split-merge finalize (line 833)**:
```python
# Before:
finalized = pending.finalize(finalize_mask, dones)
# After:
finalized = pending.finalize(finalize_mask, dones, terminated)
```

**`katago_loop.py` — call site: immediate terminal finalize (line 872)**:
```python
# Before:
imm_finalized = pending.finalize(imm_terminal, dones)
# After:
imm_finalized = pending.finalize(imm_terminal, dones, terminated)
```

**`katago_loop.py` — call site: epoch-end flush (line 951)**:
```python
# Before:
remaining_dones = torch.zeros(self.num_envs, device=self.device)
remaining = pending.finalize(remaining_mask, remaining_dones)
# After:
remaining_dones = torch.zeros(self.num_envs, device=self.device)
remaining_terminated = torch.zeros(self.num_envs, device=self.device)
remaining = pending.finalize(remaining_mask, remaining_dones, remaining_terminated)
```

**`katago_loop.py` — all `buffer.add()` call sites** (3 in split-merge, 1 in non-split-merge):
- Line 839: Add `finalized["terminated"]` as the new `terminated` argument.
- Line 878: Add `imm_finalized["terminated"]` as the new `terminated` argument.
- Line 934 (non-split-merge): Add `terminated` tensor directly.
- Line 957 (epoch-end flush): Add `remaining["terminated"]` as the new `terminated` argument.

**`katago_loop.py` — terminal_mask fix (line 797 and 901)**:
- Line 797: `terminal_mask = terminated.bool()` (was `dones.bool()`). Win/loss/draw counting should only count genuinely finished games, not truncations.
- Line 901 (non-split-merge equivalent): Same change.

**GAE functions** (`gae.py`):
- All three functions (`compute_gae`, `compute_gae_padded`, `compute_gae_gpu`):
  - Rename `dones` parameter to `terminated`.
  - Line 43/97/144: `not_done = 1.0 - terminated.float()` (same code, clearer name).
  - Update docstrings: `terminated: True only for genuinely terminal episodes (not truncated). Truncated episodes bootstrap from V(s_next) instead of zeroing it.`

**GAE call sites in `katago_ppo.py:update()`** (3 paths):

Path 1 — vectorized (lines 442-467):
```python
# Before:
dones_2d = data["dones"].reshape(T, N)
# After:
gae_dones_key = "terminated" if self.params.use_terminated_for_gae else "dones"
terminated_2d = data[gae_dones_key].reshape(T, N)
```
Then pass `terminated_2d` (not `dones_2d`) to `compute_gae_gpu` and `compute_gae`.

Path 2 — split-merge padded (lines 472-522):
```python
# Before (line 492):
env_dones.append(data["dones"][mask])
# After:
env_terminated.append(data[gae_dones_key][mask])
```
Rename `env_dones` -> `env_terminated`, `dones_pad` -> `terminated_pad` throughout this block.

Path 3 — fallback 1D (if it exists): same pattern.

#### Edge Cases

- When `truncated=False` and `terminated=True`: identical to current behavior.
- When `truncated=True` and `terminated=False`: V(s_next) now bootstraps instead of being zeroed. This is the fix.
- Epoch-end flush: transitions flushed with `dones=False, terminated=False` — unchanged behavior.
- `use_terminated_for_gae=False`: entire R1 deactivates; buffer still stores `terminated` but GAE reads `dones`.

### R2: Blend `score_pred` into GAE Scalar Value

**Priority**: 2 (addresses starvation death spiral)
**Risk**: Medium (new method + plumbing `value_adapter` to new scope + 4 call sites)
**Rollback**: Set `score_blend_alpha = 0.0` in config (no-op by design)

#### Problem

`scalar_value_from_output()` returns `P(W) - P(L)` from a head trained on <1% of data. The well-supervised score head is completely ignored for advantage estimation.

#### Changes

**`ValueHeadAdapter` ABC** (`value_adapter.py:16`):
- Add `scalar_value_blended` as a **concrete default method** (NOT abstract). Models without a score head should not need to override:
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

**`ScalarValueAdapter`** (`value_adapter.py:37`):
- No change needed — inherits the default.

**`MultiHeadValueAdapter`** (`value_adapter.py:56`):
```python
def __init__(self, lambda_value=1.5, lambda_score=0.02, score_blend_alpha=0.0):
    self.lambda_value = lambda_value
    self.lambda_score = lambda_score
    self.score_blend_alpha = score_blend_alpha

def scalar_value_blended(
    self, value_logits: torch.Tensor, score_lead: torch.Tensor,
) -> torch.Tensor:
    """Blend W/D/L value with score prediction for GAE.

    Args:
        value_logits: (batch, 3) W/D/L logits.
        score_lead: (batch, 1) normalized material balance prediction.
            Squeezed to (batch,) and clamped to [-1, 1].
    """
    wdl_value = self.scalar_value_from_output(value_logits)  # P(W) - P(L), range [-1, 1]
    score_value = score_lead.squeeze(-1).clamp(-1, 1)  # (batch, 1) -> (batch,)
    alpha = self.score_blend_alpha
    if alpha == 0.0:
        return wdl_value
    return (1 - alpha) * wdl_value + alpha * score_value
```

**`KataGoPPOParams`** (`katago_ppo.py:50-70`):
- Add field: `score_blend_alpha: float = 0.0` (backward compatible default).

**`get_value_adapter()`** (`value_adapter.py:99`):
- Add `score_blend_alpha` parameter, pass through to `MultiHeadValueAdapter`.

**Plumbing: Store `value_adapter` on `KataGoTrainingLoop`**

Currently `value_adapter` is only a parameter of `split_merge_step()` and `ppo.update()`. It is NOT stored as an instance attribute on `KataGoTrainingLoop`, which means R2 cannot access it at the bootstrap site (line 974). Fix:

In `KataGoTrainingLoop.__init__()`, after constructing `self.ppo`:
```python
from keisei.training.value_adapter import get_value_adapter
self.value_adapter = get_value_adapter(
    model_contract="multi_head",
    lambda_value=ppo_params.lambda_value,
    lambda_score=ppo_params.lambda_score,
    score_blend_alpha=ppo_params.score_blend_alpha,
)
```

**Call sites** (4 locations, all must use `self.value_adapter`):

1. **`split_merge_step()` call** (`katago_loop.py:773`):
   ```python
   # Before:
   sm_result = split_merge_step(obs=obs, ..., learner_side=learner_side)
   # After:
   sm_result = split_merge_step(obs=obs, ..., learner_side=learner_side,
                                value_adapter=self.value_adapter)
   ```
   Inside `split_merge_step()` at line 283-284, the existing code already handles `value_adapter is not None`. Change line 284:
   ```python
   # Before:
   learner_values = value_adapter.scalar_value_from_output(l_output.value_logits)
   # After:
   learner_values = value_adapter.scalar_value_blended(l_output.value_logits, l_output.score_lead)
   ```

2. **`ppo.select_actions()`** (`katago_ppo.py:402`):
   ```python
   # Before:
   scalar_values = self.scalar_value(output.value_logits)
   # After (add value_adapter parameter to select_actions, or use self.value_adapter):
   ```
   The cleanest approach: add `value_adapter` parameter to `select_actions()`:
   ```python
   def select_actions(self, obs, legal_masks, value_adapter=None):
       ...
       if value_adapter is not None:
           scalar_values = value_adapter.scalar_value_blended(output.value_logits, output.score_lead)
       else:
           scalar_values = self.scalar_value(output.value_logits)
   ```
   And update the call at line 888: `self.ppo.select_actions(obs, legal_masks, value_adapter=self.value_adapter)`

3. **Bootstrap** (`katago_loop.py:974`):
   ```python
   # Before:
   next_values = KataGoPPOAlgorithm.scalar_value(output.value_logits)
   # After:
   next_values = self.value_adapter.scalar_value_blended(output.value_logits, output.score_lead)
   ```

4. **`ppo.update()`** (`katago_ppo.py:413`):
   Already accepts `value_adapter` parameter. Pass it from the training loop:
   ```python
   # Line 995:
   losses = self.ppo.update(self.buffer, next_values,
                            value_adapter=self.value_adapter, ...)
   ```

**Config** (`keisei-500k-league.toml`):
```toml
score_blend_alpha = 0.3
```

#### Design Rationale

- `alpha=0.0` preserves existing behavior for configs that don't set it.
- `alpha=0.3` gives the score head meaningful influence without overwhelming the W/D/L signal. Consider starting at 0.1 if score head calibration is uncertain.
- The `.clamp(-1, 1)` on score_lead prevents extreme material advantages from dominating the blend.
- Both `P(W)-P(L)` and clamped `score_lead` live in `[-1, 1]`, making the linear blend well-scaled.

### R3: Increase `lambda_score` to 0.1

**Priority**: 3 (config change, complements R2)
**Risk**: Low
**Rollback**: Set `lambda_score = 0.02` in config

#### Problem

At `lambda_score=0.02`, the score head contributes 75x less gradient than the W/D/L head to the shared backbone. The score head trains on 100% of data but barely influences learned features.

#### Changes

**Keep Python defaults at 0.02. Change only the TOML config.**

Rationale: 24 test assertions and `keisei/sl/trainer.py` hardcode `lambda_score=0.02`. Changing Python defaults would silently break all of them. The safe approach is to change only the production config and leave code defaults unchanged. Tests that construct `MultiHeadValueAdapter()` without arguments will continue to use 0.02 — they are testing the adapter's math, not production hyperparameters.

- `keisei-500k-league.toml` line 45: `lambda_score = 0.1`
- All other TOML configs that should use the new default: update explicitly.
- Python defaults (`KataGoPPOParams`, `MultiHeadValueAdapter.__init__`, `get_value_adapter()`): **unchanged at 0.02**.

At 0.1, the effective ratio is 15:1 (value:score) when both have targets. This gives the score head real influence on the shared backbone features while keeping W/D/L dominant for terminal states.

#### Existing Task

This closes `keisei-1d50558dda`: "lambda_score=0.02 may be too small now that score loss is dense."

### R4: Fix LR Scheduler Monitor Signal

**Priority**: 4 (one-line fix, prevents scheduler masking collapse)
**Risk**: Low-medium (interaction with distributed all_reduce)
**Rollback**: Change back to `"value_loss"` in config or code

#### Problem

`katago_loop.py:1016`:
```python
monitor_value = losses.get("value_loss")
```

When the W/D/L head starves, `value_loss -> 0`. The `ReduceLROnPlateau` scheduler (mode="min") interprets this as improvement, never triggering LR reduction when it should.

#### Changes

```python
# Before (line 1016):
monitor_value = losses.get("value_loss")
# After:
monitor_value = losses.get("policy_loss")
if monitor_value is None:
    raise RuntimeError(
        "LR scheduler expects 'policy_loss' in losses dict but it was absent. "
        "Check that ppo.update() returns 'policy_loss'."
    )
```

The defensive assertion prevents silent scheduler failure if loss keys are ever renamed.

**Distributed safety**: The `dist.all_reduce` at line 1024 operates on whichever `monitor_value` is set. Changing the key is safe because all ranks compute `policy_loss` identically (same forward/backward pass).

### R5: Smooth Entropy Annealing

**Priority**: 5 (prevents future brittleness)
**Risk**: Low
**Rollback**: Set `entropy_decay_epochs = 0` in config (instant transition, matching current behavior)

#### Problem

`katago_ppo.py:309-317`:
```python
def get_entropy_coeff(self, epoch: int) -> float:
    if epoch < self.warmup_epochs:
        return self.warmup_entropy
    return self.params.lambda_entropy
```

Hard step from 0.05 to 0.01 at a single epoch boundary.

#### Changes

**`KataGoPPOParams`**:
- Add `entropy_decay_epochs: int = 0` (backward compatible: 0 means instant transition).

**`get_entropy_coeff()`**:
```python
def get_entropy_coeff(self, epoch: int) -> float:
    if epoch < self.warmup_epochs:
        return self.warmup_entropy
    decay_epochs = self.params.entropy_decay_epochs
    if decay_epochs <= 0:
        return self.params.lambda_entropy
    elapsed = epoch - self.warmup_epochs
    if elapsed >= decay_epochs:
        return self.params.lambda_entropy
    # Linear interpolation: warmup_entropy -> lambda_entropy over decay_epochs
    t = elapsed / decay_epochs
    return self.warmup_entropy + t * (self.params.lambda_entropy - self.warmup_entropy)
```

Note: `self.warmup_entropy` is an instance attribute on `KataGoPPOAlgorithm` (set in `__init__` at line 306), while `self.params.entropy_decay_epochs` lives on the `KataGoPPOParams` dataclass. This asymmetry already exists in the codebase — do NOT move `warmup_entropy` to `KataGoPPOParams` (it has a different config path through `rl_warmup_config`).

**Config** (`keisei-500k-league.toml`):
```toml
[training.algorithm_params.rl_warmup]
epochs = 50
entropy_bonus = 0.05

[training.algorithm_params]
entropy_decay_epochs = 200
```

## Observability

**These logging/metrics additions are required before deploying Phase 1.**

### New Log Lines

1. **Truncation bootstrap count** (per epoch, in `katago_loop.py` after rollout):
   ```python
   logger.info(
       "Epoch %d: %d terminated, %d truncated (bootstrapped), %d total transitions",
       epoch_i, terminated_count, truncated_count, total_transitions,
   )
   ```
   Derive `terminated_count` and `truncated_count` by accumulating during the rollout loop.

2. **Blended vs unblended value divergence** (per epoch, when `score_blend_alpha > 0`):
   ```python
   # After bootstrap computation:
   wdl_only = self.value_adapter.scalar_value_from_output(output.value_logits)
   blended = self.value_adapter.scalar_value_blended(output.value_logits, output.score_lead)
   blend_divergence = (blended - wdl_only).abs().mean().item()
   logger.info("Epoch %d: value blend divergence=%.4f", epoch_i, blend_divergence)
   ```

3. **Entropy coefficient** (every epoch, not just epoch 0 and warmup boundary):
   ```python
   logger.info("Epoch %d: entropy_coeff=%.4f", epoch_i, self.ppo.current_entropy_coeff)
   ```

4. **LR scheduler monitor source** (update existing log at line 1031):
   ```python
   logger.info("LR reduced: %.6f -> %.6f (monitor=policy_loss, value=%.4f)", ...)
   ```

### New DB Metrics

Add columns to the `metrics` table (or log them for manual analysis):

- `terminated_count`: Number of genuinely terminal transitions this epoch.
- `truncated_count`: Number of truncated (bootstrapped) transitions.
- `advantage_mean`: Mean advantage across all transitions.
- `advantage_std`: Std of advantages (collapse signal: std -> 0).

These can be computed cheaply after GAE and before `ppo.update()`.

## Testing Strategy

### Unit Tests — Required

1. **GAE with truncation** (`tests/test_gae.py`):
   - Create a scenario where `terminated[t] = False` but `truncated[t] = True` (i.e., old `dones[t] = True`).
   - Call `compute_gae_gpu` with `terminated` (R1 behavior): verify advantage at position t is non-zero (bootstraps from V(s_next)).
   - Call `compute_gae_gpu` with `dones = terminated | truncated` (old behavior): verify advantage at position t is near-zero (V(s_next) zeroed).
   - This test proves R1 changes the outcome.

2. **GAE backward compatibility** (`tests/test_gae.py`):
   - When `truncated` is all-False, results from `compute_gae_gpu(terminated=...)` exactly match results from the old `compute_gae_gpu(dones=...)`. Bit-exact.

3. **PendingTransitions.finalize() with three populations** (`tests/test_pending_transitions.py` or existing test file):
   - Population A: `terminated=True, truncated=False` (game ended). Verify `result["terminated"]` is 1.0, `result["dones"]` is 1.0.
   - Population B: `terminated=False, truncated=True` (hit max_ply). Verify `result["terminated"]` is 0.0, `result["dones"]` is 1.0.
   - Population C: `terminated=False, truncated=False` (epoch-end flush). Verify `result["terminated"]` is 0.0, `result["dones"]` is 0.0.

4. **`_compute_value_cats` uses terminated, not dones** (`tests/test_value_cats.py`):
   - Create rewards where `dones[i] = True` (from truncation) but `terminated[i] = False`, with `reward[i] = 0`.
   - With old behavior (`dones`): position i gets category 1 (draw). **Wrong.**
   - With new behavior (`terminated`): position i gets category -1 (ignore). **Correct.**

5. **`scalar_value_blended`** (`tests/test_value_adapter.py`):
   - alpha=0.0: output exactly matches `scalar_value_from_output`.
   - alpha=1.0: output exactly matches `score_lead.squeeze(-1).clamp(-1, 1)`.
   - alpha=0.5: output is arithmetic mean of the two.
   - Extreme score values (e.g., 5.0) are clamped to 1.0 before blending.
   - `ScalarValueAdapter.scalar_value_blended` ignores `score_lead` (inherits default).

6. **Entropy annealing** (`tests/test_entropy_annealing.py`):
   - `decay_epochs=0`: returns `lambda_entropy` immediately after warmup (matches current behavior).
   - `decay_epochs=200, epoch=warmup`: returns `warmup_entropy` (start of decay).
   - `decay_epochs=200, epoch=warmup+100`: returns midpoint between warmup_entropy and lambda_entropy.
   - `decay_epochs=200, epoch=warmup+199`: returns value very close to but not equal to lambda_entropy (boundary test).
   - `decay_epochs=200, epoch=warmup+200`: returns exactly `lambda_entropy`.

7. **Buffer `terminated` field** (`tests/test_katago_ppo.py`):
   - `buffer.add()` with `terminated` stores correctly.
   - `buffer.flatten()` returns `"terminated"` key.
   - `terminated` values survive the CPU detach round-trip.

8. **LR scheduler monitor key** (`tests/test_lr_scheduler.py`):
   - Mock `losses = {"policy_loss": 0.5, "value_loss": 0.0}`.
   - Verify scheduler steps on 0.5 (policy_loss), not 0.0 (value_loss).

9. **Split-merge padded GAE path** (`tests/test_gae.py`):
   - `compute_gae_padded` with truncation: same test pattern as test 1, but using the padded path.

### Integration Test

Run 5-epoch training with `max_ply=128`, 64 envs. Assert:
- `value_loss > 0` at every epoch (even with >50% truncation).
- `advantages.std() > 0.01` (advantages are not degenerate).
- `grad_norm > 1.0` (policy is actually updating).

### Regression

- All existing GAE tests pass (they use no truncation, so `terminated == dones`).
- All existing training loop tests pass with the new buffer schema.
- All tests that hardcode `lambda_score=0.02` are unchanged (R3 only changes TOML).

## Config Changes Summary

### Phase 1 (code deployment)
```toml
[training.algorithm_params]
use_terminated_for_gae = true  # new, feature flag for R1
```

### Phase 2 (config-only changes)
```toml
[training.algorithm_params]
lambda_score = 0.1          # was 0.02 (R3)
score_blend_alpha = 0.3     # new (R2), consider starting at 0.1
entropy_decay_epochs = 200  # new (R5)

[training.algorithm_params.rl_warmup]
epochs = 50                 # explicit (R5)
entropy_bonus = 0.05        # explicit (R5)
```

## Files Modified

| File | Remediations | Nature of Change |
|------|-------------|------------------|
| `keisei/training/gae.py` | R1 | Rename `dones` -> `terminated` in all 3 GAE functions + docstrings |
| `keisei/training/katago_ppo.py` | R1, R2, R5 | Buffer: add `terminated` field. `select_actions`: add `value_adapter` param. `update`: feature-flag GAE dones key. Rename `dones_2d`->`terminated_2d`. Add `score_blend_alpha`, `use_terminated_for_gae`, `entropy_decay_epochs` to params. |
| `keisei/training/katago_loop.py` | R1, R2, R4 | Thread `terminated` through PendingTransitions, all buffer.add() sites, all `_compute_value_cats` sites. Store `self.value_adapter`. Pass adapter to `split_merge_step`, `select_actions`, bootstrap, `update`. Fix LR monitor. Fix `terminal_mask` to use `terminated`. Add observability logging. |
| `keisei/training/value_adapter.py` | R2 | Add concrete `scalar_value_blended` to ABC. Override in `MultiHeadValueAdapter`. Add `score_blend_alpha` to constructor and `get_value_adapter()`. |
| `keisei-500k-league.toml` | R2, R3, R5 | Config values |
| `tests/test_gae.py` | R1 | New truncation bootstrap tests (vectorized + padded paths) |
| `tests/test_katago_ppo.py` | R1, R2 | Buffer terminated field, blend tests |
| `tests/test_value_adapter.py` | R2 | `scalar_value_blended` tests |
| `tests/test_entropy_annealing.py` | R5 | New file: annealing interpolation tests |
| `tests/test_pending_transitions.py` | R1 | Three-population finalize test |

## Known Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `terminated` vs `dones` confusion at a call site | Medium | High (silent wrong advantages) | Feature flag, comprehensive unit tests, code review |
| Score head injects noise via R2 blend | Medium | Medium | Start alpha at 0.1, monitor `corr(score, wdl)` |
| Adam momentum staleness delays recovery | Medium | Low | Consider optimizer state reset at Phase 1 deploy |
| Opponent pool contains degenerate snapshots | Low | Medium | Purge low-Elo entries before resuming |
| `_compute_value_cats` labels truncated as draws | High if missed | High | Explicit call-site audit in this spec + dedicated test |

## Non-Goals

- Changing the network architecture (adding heads, changing backbone).
- Modifying the Rust engine or VecEnv (the bug is entirely in the Python training loop).
- Adding intrinsic motivation or curiosity bonuses (future work).
- Changing the action space or observation encoding.
- Changing Python defaults for `lambda_score` (only TOML config changes).
- Coupling entropy schedule to value head health (too complex for this remediation).
