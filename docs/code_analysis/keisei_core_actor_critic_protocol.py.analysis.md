# Analysis: keisei/core/actor_critic_protocol.py

**Lines:** 89
**Role:** Defines `ActorCriticProtocol`, the structural typing interface (Python Protocol) that all neural network models must satisfy to be used with the PPO training system. This is the contract layer between the PPO agent and any model architecture.
**Key dependencies:** Imports `torch`, `torch.nn`. Imported by `base_actor_critic.py` (implements it), `ppo_agent.py` (type-annotates model parameter), `model_manager.py`, `trainer.py`, `setup_manager.py`, `performance_benchmarker.py`, `compilation_validator.py`, `self_play_worker.py`, `training/models/__init__.py`.
**Analysis depth:** FULL

## Summary

This file defines the central interface contract. It has one critical finding: the documented return type of `evaluate_actions` specifies a return order of `(log_probs, values, entropy)` that disagrees with the actual implementation in `BaseActorCriticModel`, which returns `(log_probs, entropy, value)`. The PPO agent consumes the implementation's order, meaning the protocol's docstring is misleading. Any implementer following the documented contract (as `DynamicActorCritic` in the evaluation subsystem does) will produce silently swapped entropy and value outputs, causing corrupt training signals. Confidence is HIGH.

## Critical Findings

### [47-64] evaluate_actions return order documented incorrectly -- silent data corruption risk

**What:** The docstring on lines 60-63 documents the return tuple as `(log_probs, values, entropy)`:
```python
Returns:
    Tuple of (log_probs, values, entropy)
```
However, the canonical implementation in `BaseActorCriticModel.evaluate_actions()` (line 184 of `base_actor_critic.py`) actually returns `(log_probs, entropy, value)` -- with entropy and value swapped relative to the docstring. The PPO agent (line 333 of `ppo_agent.py`) unpacks as `new_log_probs, entropy, new_values`, matching the implementation, not the protocol docstring.

**Why it matters:** This is a silent data corruption vector. Any independent implementation of `ActorCriticProtocol` that follows the docstring will return `(log_probs, values, entropy)`, causing the PPO agent to treat the value estimate as entropy and the entropy as the value. This will:
1. Corrupt the value loss (MSE against returns computed using entropy instead of value)
2. Corrupt the entropy bonus (using value estimate instead of entropy)
3. Training will appear to "work" (no crashes, no NaNs necessarily) but the agent will learn a garbage policy

This is not a theoretical risk -- it is already happening. The `DynamicActorCritic` class in `keisei/evaluation/core/model_manager.py` line 428 returns `probs.log_prob(actions), value.squeeze(-1), probs.entropy()` -- following the protocol docstring, putting value second and entropy third. This means evaluation using `DynamicActorCritic` with `evaluate_actions` will produce incorrect results.

**Evidence:**
- Protocol docstring (line 62): `Tuple of (log_probs, values, entropy)`
- Base implementation return (base_actor_critic.py line 184): `return log_probs, entropy, value`
- PPO agent consumption (ppo_agent.py line 333): `new_log_probs, entropy, new_values = self.model.evaluate_actions(...)`
- DynamicActorCritic (evaluation/core/model_manager.py line 428): `return probs.log_prob(actions), value.squeeze(-1), probs.entropy()`

### [28-45] get_action_and_value return type is underspecified

**What:** The protocol specifies `get_action_and_value` returns `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` documented as `(action, log_prob, value)`. The tuple type annotation provides no protection against returning the wrong tensors in the wrong positions. Python's structural typing with Protocol means there is no runtime verification that the three returned tensors semantically represent what the docstring claims.

**Why it matters:** Combined with the evaluate_actions issue above, this establishes a pattern: the protocol defines shape contracts but not semantic contracts. Any new model implementation relies entirely on getting the docstrings right, and as shown above, the docstrings are wrong for evaluate_actions. With get_action_and_value, the risk is lower because all current implementations inherit from BaseActorCriticModel, but if someone implements the Protocol directly (which is the stated purpose of having a Protocol), they would have only documentation to guide them.

**Evidence:** The type annotation `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` tells the type checker nothing about which tensor is which.

## Warnings

### [67-89] Protocol includes PyTorch Module methods without full signatures

**What:** The protocol declares `train`, `eval`, `parameters`, `state_dict`, `load_state_dict`, and `to` as required methods. While this correctly captures the PPO agent's needs, the signatures are simplified compared to the actual `nn.Module` signatures. For example:
- `train` returns `Any` instead of `Self` (nn.Module returns self)
- `eval` returns `Any` instead of `Self`
- `state_dict` uses `*args, **kwargs` which loses type information
- `to` uses `*args, **kwargs` which loses type information

**Why it matters:** These simplified signatures mean mypy cannot verify that callers are using the return values correctly. For instance, `model.to(device)` could in theory return a different type than the model, and the type checker would not catch it. In practice this works fine because all implementations inherit from `nn.Module`, but it weakens the type safety that the Protocol is supposed to provide.

**Evidence:** Compare protocol's `def to(self, *args, **kwargs) -> Any` with nn.Module's actual typed overloads.

### [7] Unused import: Dict

**What:** `Dict` is imported from typing but used only in `state_dict` and `load_state_dict` signatures. While technically used, the import of `Iterator` and `Optional` alongside it suggests these were added piecemeal. Not a correctness issue, but worth noting for housekeeping.

**Evidence:** Line 7: `from typing import Any, Dict, Iterator, Optional, Protocol, Tuple` -- all are used, this is actually fine. No issue here on second review.

## Observations

### [13] Protocol class is not runtime-checkable

The protocol class does not use `@runtime_checkable` decorator. This means `isinstance(model, ActorCriticProtocol)` will fail at runtime. This is acceptable for type-checking purposes only, but it means defensive runtime checks are not possible without explicit type checks against BaseActorCriticModel or nn.Module.

### [6] pylint suppression comment

Line 5 has `# pylint: disable=unnecessary-ellipsis` which is needed because Protocol methods use `...` as their body. This is standard practice and appropriate.

### [16] forward method takes only single tensor input

The `forward` method signature only accepts a single `x: torch.Tensor`. This means the protocol cannot accommodate architectures that need additional inputs (e.g., attention masks, positional encodings, or auxiliary information). This is acceptable for the current Shogi use case but limits extensibility.

## Verdict

**Status:** CRITICAL
**Recommended action:**
1. **Immediately** fix the `evaluate_actions` docstring to match the actual return order `(log_probs, entropy, value)`, OR change the base implementation to match the docstring. Given that the PPO agent and tests already consume `(log_probs, entropy, value)`, the docstring should be fixed.
2. **Immediately** audit and fix `DynamicActorCritic` in `evaluation/core/model_manager.py` which follows the incorrect docstring and returns values in the wrong order.
3. Consider adding a comment or named tuple to make the return order unambiguous.
**Confidence:** HIGH -- The return order mismatch is verified by reading the source of all three files (protocol, implementation, consumer). The DynamicActorCritic discrepancy is confirmed by reading line 428 of evaluation/core/model_manager.py.
