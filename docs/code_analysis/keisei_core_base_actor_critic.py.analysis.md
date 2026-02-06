# Analysis: keisei/core/base_actor_critic.py

**Lines:** 184
**Role:** Abstract base class that implements the shared `get_action_and_value` and `evaluate_actions` methods for all actor-critic models. This is the most critical file in the interface layer -- all action selection, probability computation, and value estimation during both training and inference flows through these two methods.
**Key dependencies:** Imports `torch`, `torch.nn`, `torch.nn.functional`, `log_error_to_stderr` from unified_logger, `ActorCriticProtocol`. Imported by `neural_network.py` (ActorCritic), `resnet_tower.py` (ActorCriticResTower). Indirectly consumed by PPOAgent via the protocol.
**Analysis depth:** FULL

## Summary

This file contains the core inference logic and has several significant issues. The most critical is the inconsistency between models in how `forward()` returns value shape -- `ActorCritic` returns `(batch, 1)` while `ActorCriticResTower` returns `(batch,)` after squeezing -- and the base class tries to paper over this with conditional squeezing that creates a batch-size-dependent bug. There are also issues with the NaN fallback strategy that could mask serious training problems, and a silent `pass` on the "all actions masked" case. Confidence is HIGH.

## Critical Findings

### [82-87] Silent pass when all legal actions are masked -- produces NaN outputs

**What:** When `legal_mask` is provided but has no `True` values (no legal moves), the code enters the `if not torch.any(legal_mask)` branch and executes `pass`. It then proceeds to compute `F.softmax(masked_logits, dim=-1)` where all logits are `-inf`, producing a tensor of NaN values.

**Why it matters:** The NaN fallback on lines 93-101 catches this case and replaces NaN probabilities with a uniform distribution over *all* 13,527 actions. This means the agent will select an *illegal* action uniformly at random when there are no legal moves, and the log probability will be computed as if that action is valid. The caller (PPOAgent) will then attempt to map this to a Shogi move, potentially causing the game engine to reject it or enter an invalid state.

The comment on line 87 says "PPOAgent should handle no legal moves" but the PPOAgent's error handling on line 163-172 of ppo_agent.py only logs a warning and lets execution continue. There is no circuit-breaker.

**Evidence:**
```python
if not torch.any(legal_mask):
    pass  # Let it proceed, PPOAgent should handle no legal moves.
probs = F.softmax(masked_logits, dim=-1)
```
Lines 93-101 then:
```python
if torch.isnan(probs).any():
    log_error_to_stderr(...)
    probs = torch.ones_like(policy_logits) / policy_logits.shape[-1]
```

### [82] Batch-dimension bug in all-masked check

**What:** The check `if not torch.any(legal_mask)` checks whether *any* element in the entire legal_mask tensor is True. For batched inputs (batch_size > 1), this only triggers when *all* samples in the batch have *all* actions masked. If only some samples in the batch have all actions masked, the check passes (returns False), the NaN will still occur in those rows, and the NaN fallback on line 93 will then replace *all* rows that happen to have NaN, applying the uniform distribution to rows that had legitimate masked distributions.

**Why it matters:** In batched `get_action_and_value` calls, if one sample in the batch has no legal moves but others do, the all-masked check is skipped (correctly -- it is batch-unaware), but then the NaN check on line 93 does `torch.isnan(probs).any()` which checks the *entire* batch tensor. The fallback on line 101 replaces `probs` for the *entire* batch with uniform distributions: `probs = torch.ones_like(policy_logits) / policy_logits.shape[-1]`. This destroys the valid probability distributions for all other samples in the batch.

**Evidence:**
```python
# Line 93-101: NaN check is batch-wide
if torch.isnan(probs).any():
    # This replaces ALL probs, not just the NaN rows
    probs = torch.ones_like(policy_logits) / policy_logits.shape[-1]
```

Note that `evaluate_actions` (line 168-174) handles this correctly with per-row NaN replacement:
```python
nan_rows = torch.isnan(probs).any(dim=1)
probs[nan_rows] = torch.ones_like(probs[nan_rows]) / policy_logits.shape[-1]
```

So `get_action_and_value` has a batch-wide NaN fallback but `evaluate_actions` has a per-row NaN fallback. This inconsistency means `get_action_and_value` is broken for batch sizes > 1 when any single sample has all actions masked.

### [69-74] Legal mask broadcasting only handles batch_size=1

**What:** The unsqueeze logic on lines 69-74 only adjusts the legal_mask shape when `legal_mask.ndim == 1` and `policy_logits.ndim == 2` and `policy_logits.shape[0] == 1`. This means if a caller passes a 1D legal_mask with a batch size > 1, the `torch.where` on line 76 will either broadcast incorrectly (applying the same mask to all batch samples) or raise a runtime error.

**Why it matters:** In practice, `get_action_and_value` is called with batch_size=1 during action selection (via PPOAgent.select_action), so this code path works. But the method signature and class position as a shared base class implies it should handle arbitrary batch sizes correctly. The `evaluate_actions` method (line 147-155) has a comment acknowledging this limitation but does not implement any shape adjustment, which is actually correct because it expects callers to provide correctly-shaped masks. The inconsistency between the two methods is confusing.

**Evidence:**
```python
if (
    legal_mask.ndim == 1
    and policy_logits.ndim == 2
    and policy_logits.shape[0] == 1
):
    legal_mask = legal_mask.unsqueeze(0)  # Adapt for batch size 1
```

## Warnings

### [8] Unused import: sys

**What:** `import sys` on line 8 is imported but never used in this module. The `log_error_to_stderr` function from unified_logger handles stderr output internally.

**Why it matters:** Dead import. Minor code hygiene issue. Should be removed.

**Evidence:** Line 8: `import sys` -- grep for `sys.` in this file returns no results.

### [76-79] torch.tensor scalar creation inside torch.where is potentially inefficient

**What:** On line 79, `torch.tensor(float("-inf"), device=policy_logits.device)` creates a new scalar tensor on every call to `get_action_and_value` and `evaluate_actions`. Same pattern appears on line 160.

**Why it matters:** While PyTorch is efficient at handling small tensor creation, this allocates a new tensor object on every forward pass. During training with thousands of steps, this creates unnecessary garbage collection pressure. A module-level constant or a cached tensor would be more efficient.

**Evidence:**
```python
torch.tensor(float("-inf"), device=policy_logits.device)
```
This appears twice: lines 79 and 160.

### [88-90] Softmax computed from probabilities, not log-probabilities

**What:** The code computes `probs = F.softmax(masked_logits, dim=-1)` and then constructs `torch.distributions.Categorical(probs=probs)`. The Categorical distribution re-normalizes and computes log_probs internally. Using `F.log_softmax` + `Categorical(logits=...)` would be more numerically stable and avoid a redundant normalization step.

**Why it matters:** `F.softmax` followed by `log_prob` involves computing `softmax` and then `log(softmax(...))` internally. Using `log_softmax` directly is more numerically stable because it avoids the intermediate exponentiation and re-logarithm. For the 13,527-dimensional action space, this could introduce small numerical errors that accumulate over long training runs.

**Evidence:**
```python
probs = F.softmax(masked_logits, dim=-1)
# ...
dist = torch.distributions.Categorical(probs=probs)
```
vs the more stable:
```python
log_probs = F.log_softmax(masked_logits, dim=-1)
dist = torch.distributions.Categorical(logits=masked_logits)  # or log_probs
```

### [112-116] Value squeezing is model-dependent and fragile

**What:** Lines 113-114 check `if value.dim() > 1 and value.shape[-1] == 1` and squeeze the last dimension. This is needed because `ActorCritic.forward()` returns value with shape `(batch, 1)` while `ActorCriticResTower.forward()` returns value with shape `(batch,)` (it squeezes in `forward()`). The same pattern is on lines 181-182.

**Why it matters:** The `forward()` return shape is not specified in the protocol. This means the base class must handle both shapes, which it does, but it creates a fragile implicit contract: subclasses can return value in any shape and the base class will try to normalize it. If a subclass returns value with shape `(batch, 2)` (e.g., due to a bug in the value head), the squeeze would not trigger and the downstream code would get a 2D value where it expects 1D, causing silent shape mismatches in the PPO loss computation.

**Evidence:**
- `ActorCritic.forward()` (neural_network.py line 28): returns `value` with shape `(batch, 1)` (from `nn.Linear(16 * 9 * 9, 1)`)
- `ActorCriticResTower.forward()` (resnet_tower.py line 83): returns `value.squeeze(-1)` with shape `(batch,)`
- Base class squeezing (lines 113-114): `if value.dim() > 1 and value.shape[-1] == 1: value = value.squeeze(-1)`

### [21] Multiple inheritance with Protocol is unusual

**What:** `BaseActorCriticModel(nn.Module, ActorCriticProtocol, ABC)` inherits from a Protocol class. While this works in Python, it is semantically unusual -- Protocols are meant for structural typing, not for inheritance. The ABC already provides the abstract method mechanism.

**Why it matters:** Including `ActorCriticProtocol` in the MRO means that if the Protocol ever gains default implementations or properties, they could unexpectedly affect the base class behavior. In practice, since Protocol methods are all stubs (returning `...`), this is not currently dangerous, but it muddies the design intent. The Protocol should be used for type-checking external implementations, while the ABC should be used for the inheritance hierarchy.

**Evidence:** Line 21: `class BaseActorCriticModel(nn.Module, ActorCriticProtocol, ABC):`

## Observations

### [97-100] Error logging uses stderr only, not the unified logger

**What:** The NaN fallback uses `log_error_to_stderr` which writes directly to stderr. In a production training run with Rich console UI, these messages may not be visible in the main display or log files.

### [64] forward() called explicitly instead of using __call__

**What:** Line 64 calls `self.forward(obs)` directly rather than `self(obs)`. When using `nn.Module`, the recommended pattern is `self(obs)` which triggers hooks (forward hooks, backward hooks). Calling `self.forward()` directly bypasses these hooks.

**Why it matters:** If anyone adds forward hooks to the model (e.g., for debugging, profiling, or feature extraction), they will not fire during `get_action_and_value` or `evaluate_actions`. This is also the case on line 139.

**Evidence:** Lines 64 and 139: `policy_logits, value = self.forward(obs)`

### [118-184] evaluate_actions has better NaN handling than get_action_and_value

The per-row NaN replacement in `evaluate_actions` (lines 173-174) is correct, while the batch-wide replacement in `get_action_and_value` (line 101) is not. This inconsistency suggests the evaluate_actions code was updated later to fix a known issue, but the corresponding fix was not applied to get_action_and_value.

## Verdict

**Status:** CRITICAL
**Recommended action:**
1. **Immediately** fix the batch-wide NaN fallback in `get_action_and_value` (line 101) to use per-row replacement matching the `evaluate_actions` pattern.
2. **Immediately** address the silent `pass` on all-masked case -- at minimum, log a warning; ideally, raise an explicit error or return a sentinel value that the caller can detect.
3. Standardize the value shape contract in the Protocol (should `forward()` return squeezed or unsqueezed values?).
4. Switch from `F.softmax` + `Categorical(probs=...)` to `Categorical(logits=masked_logits)` for better numerical stability.
5. Remove unused `import sys`.
6. Use `self(obs)` instead of `self.forward(obs)` to respect nn.Module hooks.
**Confidence:** HIGH -- All findings verified by reading source code of the implementation, its consumers, and its subclasses.
