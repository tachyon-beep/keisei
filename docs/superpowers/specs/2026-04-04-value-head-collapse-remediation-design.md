# Value Head Collapse Remediation

**Date**: 2026-04-04
**Status**: Draft
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

## Remediations

### R1: Separate `terminated` from `truncated` in GAE

**Priority**: 1 (correctness bug)
**Risk**: Medium (touches buffer schema and GAE, many call sites)

#### Problem

`katago_loop.py:791`:
```python
dones = terminated | truncated
```

This merged `dones` flows to `buffer.add()` and then to all three GAE functions. GAE uses `not_done = 1.0 - dones.float()` to gate the bootstrap value V(s_next). For truncated games (not truly over), V(s_next) should NOT be zeroed — the game continues, we just stopped collecting.

#### Changes

**`KataGoRolloutBuffer`** (`katago_ppo.py:82-98`):
- Add `self.terminated: list[torch.Tensor] = []` to `clear()`.
- Existing `self.dones` is retained for backward compatibility in `_compute_value_cats`.

**`buffer.add()`** (`katago_ppo.py:105`):
- Add `terminated: torch.Tensor` parameter after `dones`.
- Store `terminated_cpu` in `self.terminated`.

**`buffer.flatten()`**:
- Include `"terminated"` key in the returned dict.

**`katago_loop.py`** — all `buffer.add()` call sites:
- **Line 839** (split-merge finalize): Pass `finalized["terminated"]`. Requires `PendingTransition.finalize()` to return `terminated` separately from `dones`.
- **Line 934** (non-split-merge): Pass `terminated` tensor directly (already available from `step_result.terminated`).
- **Line 957** (epoch-end flush): Pass `remaining_terminated = torch.zeros(...)` (these are not terminated).

**`PendingTransition`** (`katago_loop.py:131-250`):
- `finalize()` must thread `terminated` (not `dones`) through. Currently `finalize(mask, dones)` receives the merged `dones`. Change to `finalize(mask, terminated, truncated)` or `finalize(mask, dones, terminated)` so finalized dict includes both.

**GAE functions** (`gae.py`):
- All three functions (`compute_gae`, `compute_gae_padded`, `compute_gae_gpu`): Rename `dones` parameter to `terminated`. Update `not_done = 1.0 - terminated.float()`.
- This is a pure rename — the semantics change is that truncated episodes are now "not done" for bootstrapping purposes.

**GAE call sites** (`katago_ppo.py:442-510`):
- Line 446: `dones_2d = data["terminated"].reshape(T, N)` (was `data["dones"]`)
- Line 460: Pass `dones_2d` (now terminated) to `compute_gae_gpu`
- Line 493: Same for padded path

**Bootstrap** (`katago_loop.py:970-974`):
- No change needed — bootstrap already uses the fresh observation after auto-reset.

#### Edge Cases

- When `truncated=False` and `terminated=True`: identical to current behavior.
- When `truncated=True` and `terminated=False`: V(s_next) now bootstraps instead of being zeroed. This is the fix.
- Epoch-end flush: transitions flushed with `dones=False, terminated=False` — unchanged behavior.

### R2: Blend `score_pred` into GAE Scalar Value

**Priority**: 2 (addresses starvation death spiral)
**Risk**: Low-medium (new method + 4 call sites)

#### Problem

`scalar_value_from_output()` returns `P(W) - P(L)` from a head trained on <1% of data. The well-supervised score head is completely ignored for advantage estimation.

#### Changes

**`MultiHeadValueAdapter`** (`value_adapter.py:56`):
```python
def __init__(self, lambda_value=1.5, lambda_score=0.1, score_blend_alpha=0.0):
    self.lambda_value = lambda_value
    self.lambda_score = lambda_score
    self.score_blend_alpha = score_blend_alpha

def scalar_value_blended(
    self, value_logits: torch.Tensor, score_lead: torch.Tensor,
) -> torch.Tensor:
    """Blend W/D/L value with score prediction for GAE."""
    wdl_value = self.scalar_value_from_output(value_logits)  # P(W) - P(L), range [-1, 1]
    score_value = score_lead.squeeze(-1).clamp(-1, 1)
    alpha = self.score_blend_alpha
    if alpha == 0.0:
        return wdl_value
    return (1 - alpha) * wdl_value + alpha * score_value
```

**`ScalarValueAdapter`** (`value_adapter.py:37`):
```python
def scalar_value_blended(
    self, value_logits: torch.Tensor, score_lead: torch.Tensor,
) -> torch.Tensor:
    return self.scalar_value_from_output(value_logits)  # no-op, ignores score
```

**`ValueHeadAdapter` ABC** (`value_adapter.py:16`):
- Add `scalar_value_blended` as abstract method.

**Call sites** (4 locations):
1. `katago_loop.py:284` (split-merge learner values):
   ```python
   learner_values = value_adapter.scalar_value_blended(l_output.value_logits, l_output.score_lead)
   ```
2. `katago_ppo.py:342` (non-split-merge select_actions): Same pattern.
3. `katago_loop.py:974` (bootstrap):
   ```python
   next_values = value_adapter.scalar_value_blended(output.value_logits, output.score_lead)
   ```
   (Replace the direct `KataGoPPOAlgorithm.scalar_value()` call.)
4. Any remaining value computation for pending transition flush.

**`KataGoPPOParams`** (`katago_ppo.py`):
- Add `score_blend_alpha: float = 0.0` (backward compatible default).

**`get_value_adapter()`** (`value_adapter.py:99`):
- Pass `score_blend_alpha` through to `MultiHeadValueAdapter`.

**Config** (`keisei-500k-league.toml`):
```toml
score_blend_alpha = 0.3
```

#### Design Rationale

- `alpha=0.0` preserves existing behavior for configs that don't set it.
- `alpha=0.3` gives the score head meaningful influence without overwhelming the W/D/L signal.
- The `.clamp(-1, 1)` on score_lead prevents extreme material advantages from dominating the blend.
- Both `P(W)-P(L)` and clamped `score_lead` live in `[-1, 1]`, making the linear blend well-scaled.

### R3: Increase `lambda_score` to 0.1

**Priority**: 3 (config change, complements R2)
**Risk**: Low

#### Problem

At `lambda_score=0.02`, the score head contributes 75x less gradient than the W/D/L head to the shared backbone. The score head trains on 100% of data but barely influences learned features.

#### Changes

- `keisei-500k-league.toml` line 45: `lambda_score = 0.1`
- `MultiHeadValueAdapter.__init__` default: `lambda_score=0.1` (was 0.02)
- `get_value_adapter()` default: `lambda_score=0.1`
- `KataGoPPOParams.lambda_score` default: `0.1`

At 0.1, the effective ratio is 15:1 (value:score) when both have targets. This gives the score head real influence on the shared backbone features while keeping W/D/L dominant for terminal states.

#### Existing Task

This closes `keisei-1d50558dda`: "lambda_score=0.02 may be too small now that score loss is dense."

### R4: Fix LR Scheduler Monitor Signal

**Priority**: 4 (one-line fix, prevents scheduler masking collapse)
**Risk**: Low

#### Problem

`katago_loop.py:1016`:
```python
monitor_value = losses.get("value_loss")
```

When the W/D/L head starves, `value_loss -> 0`. The `ReduceLROnPlateau` scheduler (mode="min") interprets this as improvement, never triggering LR reduction when it should, or failing to signal that training is collapsing.

#### Change

```python
# Before:
monitor_value = losses.get("value_loss")
# After:
monitor_value = losses.get("policy_loss")
```

Policy loss reflects actual policy learning quality. It increases when the policy is struggling (clipped ratio hitting bounds) and decreases when learning is productive. This is a more reliable health signal.

### R5: Smooth Entropy Annealing

**Priority**: 5 (prevents future brittleness)
**Risk**: Low

#### Problem

`katago_ppo.py:309-317`:
```python
def get_entropy_coeff(self, epoch: int) -> float:
    if epoch < self.warmup_epochs:
        return self.warmup_entropy
    return self.params.lambda_entropy
```

Hard step from 0.05 to 0.01 at a single epoch boundary. This sudden exploration reduction can trigger policy brittleness.

#### Changes

**`KataGoPPOParams`**:
- Add `entropy_decay_epochs: int = 0` (backward compatible: 0 means instant transition, matching current behavior).

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
    t = elapsed / decay_epochs
    return self.warmup_entropy + t * (self.params.lambda_entropy - self.warmup_entropy)
```

**Config** (`keisei-500k-league.toml`):
```toml
rl_warmup_epochs = 50
warmup_entropy = 0.05
entropy_decay_epochs = 200
```

This linearly decays entropy coefficient from 0.05 to 0.01 over 200 epochs after warmup, giving the policy time to develop diverse strategies before exploration pressure decreases.

## Testing Strategy

### Unit Tests

1. **GAE with truncation**: Test `compute_gae_gpu` where `terminated != (terminated | truncated)`. Verify that truncated positions bootstrap from V(s_next) while terminated positions get V(s_next)=0.
2. **GAE backward compatibility**: When `truncated` is all-False, results match existing behavior exactly.
3. **`scalar_value_blended`**: Test alpha=0.0 matches `scalar_value_from_output`. Test alpha=1.0 uses pure score. Test clamping of extreme score values.
4. **Entropy annealing**: Test linear interpolation at warmup boundary, midpoint, and completion. Test `decay_epochs=0` matches step behavior.
5. **Buffer `terminated` field**: Test that buffer stores and flattens `terminated` correctly alongside `dones`.

### Integration Test

- Run 5-epoch training with `max_ply=128`, 64 envs. Assert `value_loss > 0` even when truncation exceeds 50%. This directly tests that R1 prevents the starvation spiral.

### Regression

- All existing GAE tests pass (they use no truncation, so `terminated == dones`).
- All existing training loop tests pass with the new buffer schema (new field has a sensible default path).

## Config Changes Summary

```toml
[training.algorithm_params]
lambda_score = 0.1          # was 0.02 (R3)
score_blend_alpha = 0.3     # new (R2)
rl_warmup_epochs = 50       # explicit (R5)
warmup_entropy = 0.05       # explicit (R5)
entropy_decay_epochs = 200   # new (R5)
```

## Files Modified

| File | Remediations | Nature of Change |
|------|-------------|------------------|
| `keisei/training/gae.py` | R1 | Rename `dones` -> `terminated` in all GAE functions |
| `keisei/training/katago_ppo.py` | R1, R2, R5 | Buffer schema, GAE call sites, entropy annealing, blend alpha param |
| `keisei/training/katago_loop.py` | R1, R2, R4 | Thread terminated through buffer, blend at bootstrap, fix LR monitor |
| `keisei/training/value_adapter.py` | R2, R3 | Add `scalar_value_blended`, update defaults |
| `keisei-500k-league.toml` | R2, R3, R5 | Config values |
| `tests/test_gae.py` | R1 | New truncation bootstrap tests |
| `tests/test_value_adapter.py` | R2, R3 | New blend tests |
| `tests/test_entropy.py` | R5 | Annealing interpolation tests |

## Non-Goals

- Changing the network architecture (adding heads, changing backbone).
- Modifying the Rust engine or VecEnv (the bug is entirely in the Python training loop).
- Adding intrinsic motivation or curiosity bonuses (future work).
- Changing the action space or observation encoding.
