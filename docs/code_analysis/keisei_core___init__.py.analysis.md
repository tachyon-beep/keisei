# Analysis: keisei/core/__init__.py

**Lines:** 6
**Role:** Package initializer for the core module. Re-exports `BaseActorCriticModel` and `ActorCritic` for convenient imports by downstream consumers.
**Key dependencies:** Imports from `.base_actor_critic` and `.neural_network`. Imported by tests, evaluation subsystem, parallel training, and utility modules.
**Analysis depth:** FULL

## Summary

This is a minimal package initializer that re-exports two symbols. It is functionally correct for its purpose but has a notable omission: `ActorCriticProtocol` -- the central interface of this package -- is not re-exported, forcing all consumers to import it directly from the submodule. This is a design inconsistency rather than a bug. Confidence is HIGH.

## Warnings

### [3-4] Incomplete public API surface -- ActorCriticProtocol not exported

**What:** The `__all__` list exports `BaseActorCriticModel` and `ActorCritic` but does not export `ActorCriticProtocol`, which is the primary interface type that downstream consumers (PPOAgent, ModelManager, Trainer, setup_manager, performance_benchmarker, compilation_validator, self_play_worker) depend on.

**Why it matters:** Every consumer of the protocol must use the verbose import path `from keisei.core.actor_critic_protocol import ActorCriticProtocol` instead of the expected `from keisei.core import ActorCriticProtocol`. This creates an implicit dependency on the internal module structure. If `actor_critic_protocol.py` is ever renamed or reorganized, all 10+ import sites break rather than just the `__init__.py` re-export.

**Evidence:** Grep shows these direct submodule imports across the codebase:
- `keisei/core/ppo_agent.py` line 15
- `keisei/training/model_manager.py` line 24
- `keisei/training/trainer.py` line 13
- `keisei/training/models/__init__.py` line 1
- `keisei/training/setup_manager.py` line 12
- `keisei/utils/performance_benchmarker.py` line 23
- `keisei/utils/compilation_validator.py` line 17
- `keisei/training/parallel/self_play_worker.py` line 17

## Observations

### [6] __all__ is well-defined

The `__all__` list is explicit and matches the actual imports. This is good practice for controlling the public API surface. The only issue is its incompleteness as noted above.

### [3-4] No re-export of other core module symbols

Other important symbols like `PPOAgent`, `ExperienceBuffer`, `SchedulerFactory` are not re-exported either. This suggests a deliberate design choice to keep this `__init__.py` minimal, but it creates an inconsistency: some core symbols are convenient imports and others are not.

## Verdict

**Status:** NEEDS_ATTENTION
**Recommended action:** Add `ActorCriticProtocol` to the `__init__.py` exports. Consider whether other commonly-imported symbols (e.g., `PPOAgent`, `ExperienceBuffer`) should also be re-exported for API consistency. This is a low-effort improvement that reduces coupling to internal module layout.
**Confidence:** HIGH -- The file is trivial; the finding is about API design consistency, not correctness.
