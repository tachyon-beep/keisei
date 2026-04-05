## Summary

`ResNetParams` in `/home/john/keisei/keisei/training/models/resnet.py` accepts invalid widths (for example `hidden_size=0`), allowing model construction to succeed but causing a runtime crash on first forward pass.

## Severity

- Severity: major
- Priority: P1

## Evidence

- [`/home/john/keisei/keisei/training/models/resnet.py:13`](/home/john/keisei/keisei/training/models/resnet.py:13) defines `ResNetParams` with no semantic validation (`hidden_size`, `num_layers` unconstrained).
- [`/home/john/keisei/keisei/training/models/resnet.py:39`](/home/john/keisei/keisei/training/models/resnet.py:39) uses `hidden_size` directly as `Conv2d` out_channels.
- [`/home/john/keisei/keisei/training/model_registry.py:43`](/home/john/keisei/keisei/training/model_registry.py:43) through [`:73`](/home/john/keisei/keisei/training/model_registry.py:73) performs semantic checks for `transformer` and `se_resnet`, but not `resnet`.
- Reproduced via runtime path:
  - `validate_model_params("resnet", {"hidden_size": 0, "num_layers": 1})` returns `ResNetParams(hidden_size=0, num_layers=1)`.
  - `build_model("resnet", ...)` succeeds, but `forward` fails with: `RuntimeError ... expected weight to be at least 1 at dimension 0, but got [0, 50, 3, 3]`.

## Root Cause Hypothesis

The model relies on `hidden_size` being positive, but neither `ResNetParams` nor `ResNetModel.__init__` enforces that invariant. Invalid config values pass through registry validation and fail later inside PyTorch ops.

## Suggested Fix

Add parameter validation in `ResNetParams.__post_init__` in `resnet.py`:
- Require `hidden_size >= 1`.
- Require `num_layers >= 0` (or `>= 1` if zero-layer towers should be disallowed by design).
- Raise `ValueError` with explicit messages.
---
## Summary

`ResNetModel.forward` in `/home/john/keisei/keisei/training/models/resnet.py` does not validate observation geometry, so malformed tensors fail later with opaque matmul/conv errors instead of a clear boundary error.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- [`/home/john/keisei/keisei/training/models/resnet.py:56`](/home/john/keisei/keisei/training/models/resnet.py:56) through [`:68`](/home/john/keisei/keisei/training/models/resnet.py:68): `forward` has no shape checks before convolutions/linear layers.
- For wrong board size `(B,50,8,8)`, failure occurs later at linear layer (`mat1 and mat2 shapes cannot be multiplied`) rather than at input boundary.
- Other architectures in this repo explicitly validate input shape early:
  - [`/home/john/keisei/keisei/training/models/transformer.py:56`](/home/john/keisei/keisei/training/models/transformer.py:56)
  - [`/home/john/keisei/keisei/training/models/se_resnet.py:133`](/home/john/keisei/keisei/training/models/se_resnet.py:133)

## Root Cause Hypothesis

`ResNetModel` assumes upstream data pipeline always emits `(batch, 50, 9, 9)` tensors and omits defensive validation. When this contract is violated, errors surface deep in tensor ops and are harder to diagnose.

## Suggested Fix

At the start of `ResNetModel.forward`, add an explicit shape guard:
- Validate `obs.ndim == 4`, channel count `== OBS_CHANNELS`, and board dims `== BOARD_SIZE`.
- Raise `ValueError` with expected vs actual shape and optional NHWC hint (matching transformer style).
