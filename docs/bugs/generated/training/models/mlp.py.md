## Summary

`MLPModel.forward()` does not validate observation layout, so NHWC input `(B, 9, 9, 50)` is silently accepted and flattened as if it were NCHW, producing incorrect policy/value outputs without raising an error.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/training/models/mlp.py:35`](/home/john/keisei/keisei/training/models/mlp.py:35) to [`:37`](/home/john/keisei/keisei/training/models/mlp.py:37): `obs` is directly flattened via `obs.flatten(1)` with no dimensional/layout checks.
- [`/home/john/keisei/keisei/training/models/base.py:15`](/home/john/keisei/keisei/training/models/base.py:15): contract explicitly states input must be `(batch, 50, 9, 9)`.
- [`/home/john/keisei/keisei/training/models/transformer.py:56`](/home/john/keisei/keisei/training/models/transformer.py:56) to [`:63`](/home/john/keisei/keisei/training/models/transformer.py:63): transformer model already guards this exact issue (including NHWC hint), showing expected behavior elsewhere.

## Root Cause Hypothesis

The MLP path relies on flattening, which removes spatial/channel structure and therefore cannot distinguish valid NCHW from invalid NHWC when total element count matches; this bypasses contract enforcement and can silently corrupt training/evaluation if upstream layout regresses.

## Suggested Fix

Add explicit shape/layout validation in `MLPModel.forward()` (same guard pattern as `TransformerModel.forward()`), raising `ValueError` unless `obs.shape == (B, 50, 9, 9)`, with an NHWC hint when `obs.shape[-1] == 50`.
---
## Summary

`MLPModel` accepts `hidden_sizes` entries of `0`, which creates zero-width linear layers and can collapse the network into near-constant/bias-only outputs instead of a feature-dependent policy/value function.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/training/models/mlp.py:25`](/home/john/keisei/keisei/training/models/mlp.py:25) to [`:29`](/home/john/keisei/keisei/training/models/mlp.py:29): hidden sizes are used without positivity checks.
- [`/home/john/keisei/keisei/training/models/mlp.py:32`](/home/john/keisei/keisei/training/models/mlp.py:32) to [`:33`](/home/john/keisei/keisei/training/models/mlp.py:33): downstream heads are built from `prev_size` even if it becomes `0`.
- [`/home/john/keisei/keisei/training/model_registry.py:51`](/home/john/keisei/keisei/training/model_registry.py:51) to [`:71`](/home/john/keisei/keisei/training/model_registry.py:71): semantic validation exists for transformer/se_resnet but none for `mlp` hidden sizes.
- Runtime confirmation in this environment: `torch.nn.Linear(10, 0)` and `torch.nn.Linear(0, 5)` are accepted by PyTorch (warning only), and `Linear(0, 5)` outputs zeros plus bias, enabling degenerate behavior instead of hard failure.

## Root Cause Hypothesis

`MLPParams.hidden_sizes` is treated as structurally valid if type-correct, but not semantically validated (`> 0`), so invalid architectural widths pass through model construction and fail silently in optimization quality rather than with an explicit config error.

## Suggested Fix

In `MLPModel.__init__`, validate `hidden_sizes` entries before layer creation (for example: `if any(s <= 0 for s in params.hidden_sizes): raise ValueError(...)`). Optionally also enforce this in model-registry semantic validation for `architecture == "mlp"` to fail earlier at config parsing time.
