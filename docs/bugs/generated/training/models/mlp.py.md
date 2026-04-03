## Summary

No concrete bug found in /home/john/keisei/keisei/training/models/mlp.py

## Severity

- Severity: trivial
- Priority: P3

## Evidence

- [keisei/training/models/mlp.py](/home/john/keisei/keisei/training/models/mlp.py):21-40
  - Input size is derived from `BaseModel` constants (`46*9*9`), trunk/head dimensions are internally consistent, and forward returns `(policy_logits, value)` with value bounded by `tanh`.
- [keisei/training/models/base.py](/home/john/keisei/keisei/training/models/base.py):15-23
  - Base contract for scalar models is `(batch, 46, 9, 9) -> policy (batch, 13527), value (batch, 1)`, matching `MLPModel`.
- [keisei/training/model_registry.py](/home/john/keisei/keisei/training/model_registry.py):24-27
  - Registry maps `mlp` to scalar contract with `obs_channels=46`, consistent with `MLPModel` construction.
- [tests/test_models.py](/home/john/keisei/tests/test_models.py):89-114 and [tests/test_model_gaps.py](/home/john/keisei/tests/test_model_gaps.py):51-86
  - Existing tests cover MLP forward shapes, bounded value output, LayerNorm presence, gradient flow, and empty-hidden-layer edge case.
- Cross-file issues observed (e.g., evaluation/demonstrator using `observation_mode="katago"` + `action_mode="spatial"` at [keisei/training/evaluate.py](/home/john/keisei/keisei/training/evaluate.py):103-104 and [keisei/training/demonstrator.py](/home/john/keisei/keisei/training/demonstrator.py):154-157) are integration-level and not primarily fixable inside `mlp.py`.

## Root Cause Hypothesis

No bug identified

## Suggested Fix

No change recommended in `/home/john/keisei/keisei/training/models/mlp.py` based on current evidence.
