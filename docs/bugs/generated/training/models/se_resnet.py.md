## Summary

`SEResNetParams` accepts invalid structural hyperparameters, and `GlobalPoolBiasBlock` then performs unchecked dimension math (`channels // se_reduction`) that can trigger divide-by-zero or invalid `nn.Linear` layer construction at runtime.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- Target file uses unchecked division and layer sizes:
  - [se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py#L51): `se_hidden = channels // se_reduction`
  - [se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py#L52): `self.se_fc1 = nn.Linear(channels, se_hidden)`
  - [se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py#L45): `nn.Linear(channels * 3, global_pool_channels)`
- No validation exists in the params dataclass:
  - [se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py#L16)
- Upstream model param validation only checks dataclass field names/types, not numeric bounds:
  - [model_registry.py](/home/john/keisei/keisei/training/model_registry.py#L41)

## Root Cause Hypothesis

The model assumes config values are sane but never enforces bounds. If `se_reduction <= 0`, construction can fail with division-by-zero; if `channels < se_reduction`, `se_hidden` becomes `0` and `nn.Linear(..., 0)` is invalid; if `global_pool_channels <= 0`, `global_fc` construction fails. This is a config-boundary validation gap in the target file.

## Suggested Fix

Add a `__post_init__` in `SEResNetParams` (or equivalent checks in `SEResNetModel.__init__`) to enforce:
- `num_blocks >= 1`
- `channels >= 1`
- `se_reduction >= 1`
- `channels // se_reduction >= 1`
- `global_pool_channels >= 1`
- `policy_channels >= 1`
- `value_fc_size >= 1`
- `score_fc_size >= 1`
- `obs_channels >= 1`

Raise a clear `ValueError` with the offending field/value before layer construction.
---
## Summary

`SEResNetModel._forward_impl` validates only channel count, not board geometry, so non-`9x9` inputs can propagate and then break downstream PPO masking/action-shape assumptions.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- Target file checks only channels:
  - [se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py#L120)
- Policy head output shape depends on runtime `H,W` (after `permute`), with comments assuming `9x9`:
  - [se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py#L132)
  - [se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py#L133)
- Downstream PPO assumes flattened policy size `11259 = 9*9*139` and masks against fixed legal-mask width:
  - [katago_ppo.py](/home/john/keisei/keisei/training/katago_ppo.py#L392)
  - [katago_ppo.py](/home/john/keisei/keisei/training/katago_ppo.py#L394)
- Training loop itself hardcodes spatial action-space expectation `11259`:
  - [katago_loop.py](/home/john/keisei/keisei/training/katago_loop.py#L291)

## Root Cause Hypothesis

The model contract is effectively `(B, C, 9, 9)`, but `_forward_impl` only guards `C`. If an integration path supplies wrong board dimensions (e.g., preprocessing/env regression), the model emits logits with `H*W*139 != 11259`, causing late shape failures in PPO instead of immediate, local validation failure.

## Suggested Fix

In `SEResNetModel._forward_impl`, add explicit input-shape checks before running convs, e.g.:
- `obs.ndim == 4`
- `obs.shape[2] == 9 and obs.shape[3] == 9`

Raise a clear `ValueError` describing expected `(batch, obs_channels, 9, 9)` and actual shape.
