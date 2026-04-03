## Summary

`compute_gae`, `compute_gae_padded`, and `compute_gae_gpu` silently return integer-truncated advantages when `rewards` is an integer tensor because outputs are allocated with `zeros_like/empty_like(rewards)`.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target file allocations use `rewards` dtype for advantage buffers:
  - `/home/john/keisei/keisei/training/gae.py:34` `advantages = torch.zeros_like(rewards)`
  - `/home/john/keisei/keisei/training/gae.py:78` `advantages = torch.zeros_like(rewards)`
  - `/home/john/keisei/keisei/training/gae.py:152` `advantages = torch.empty_like(rewards)`
- These lines force integer output if `rewards.dtype` is integer, even though GAE math is fractional (`gamma`, `lam`, bootstrap value).
- Repro (executed in repo):
  - `compute_gae(torch.tensor([1,2,3], dtype=torch.long), ...)` returned `tensor([5, 4, 2])` with `torch.int64` (fractional parts lost).
  - `compute_gae_padded(... rewards dtype=torch.long ...)` returned `torch.int64`.
  - `compute_gae_gpu(... rewards dtype=torch.long ...)` returned `torch.int64`.
- Integration plausibility:
  - rewards originate from env numpy arrays without explicit float cast:
    - `/home/john/keisei/keisei/training/katago_loop.py:484`
    - `/home/john/keisei/keisei/training/katago_loop.py:529`
  - so integer reward tensors are a realistic upstream input if env emits integral rewards.

## Root Cause Hypothesis

The implementation assumes reward tensors are already floating-point, but no dtype normalization is enforced. Because output buffers are created from `rewards` dtype, any integer reward input causes silent truncation on assignment of floating GAE values, corrupting training targets without raising an error.

## Suggested Fix

In `gae.py`, normalize computation to a floating dtype and allocate outputs using that dtype (not `rewards.dtype`).

Concrete change pattern for all three functions:
- Derive `compute_dtype` from floating inputs (e.g., `values.dtype`/`next_value.dtype`), fallback to `torch.float32` if needed.
- Cast `rewards`, `values`, `next_value`/`next_values`, and `not_done` to `compute_dtype`.
- Allocate:
  - `advantages = torch.zeros(rewards.shape, device=rewards.device, dtype=compute_dtype)` (or `empty` for GPU path)
  - `last_gae = torch.zeros(..., device=..., dtype=compute_dtype)`

Also add a regression test in `tests/test_gae.py` (and padded/GPU suites) asserting integer `rewards` still produce floating advantages matching float-reference results.
