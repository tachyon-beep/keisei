## Summary

`TransformerModel.forward()` silently accepts incorrectly ordered `NHWC` observations and produces valid-shaped but semantically wrong outputs instead of failing fast on the contract-required `NCHW` layout.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Contract requires `obs` shape `(batch, 46, 9, 9)` in [`/home/john/keisei/keisei/training/models/base.py:15`](file:///home/john/keisei/keisei/training/models/base.py:15).
- Transformer forward currently does:
  - `x = obs.permute(0, 2, 3, 1).reshape(batch, 81, self.OBS_CHANNELS)` in [`/home/john/keisei/keisei/training/models/transformer.py:57`](file:///home/john/keisei/keisei/training/models/transformer.py:57).
- Because of that `permute(...).reshape(...)` pattern, a wrong-layout tensor `(B, 9, 9, 46)` is still accepted (same element count), giving silent misinterpretation rather than an error.
- Runtime repro (executed locally): both `x1.shape=(2,46,9,9)` and `x2.shape=(2,9,9,46)` produced output shapes `(2,13527)` and `(2,1)` with different values, confirming silent wrong behavior.

## Root Cause Hypothesis

The model relies on a fixed permutation+reshape path without validating input dimensionality/layout. For `NHWC` input, dimensions are rearranged into a compatible flattened shape, so PyTorch does not raise, and training/inference can proceed on corrupted feature ordering.

## Suggested Fix

Add strict input validation at the top of `forward()` in `transformer.py`:

- Require `obs.ndim == 4`.
- Require `obs.shape[1] == self.OBS_CHANNELS`, `obs.shape[2] == self.BOARD_SIZE`, `obs.shape[3] == self.BOARD_SIZE`.
- Raise a clear `ValueError` with expected vs actual shape if violated (and optionally hint if `obs.shape[-1] == OBS_CHANNELS`, indicating probable `NHWC` input).

Also replace hardcoded `81` with `self.BOARD_SIZE * self.BOARD_SIZE` for consistency and safer future maintenance.
