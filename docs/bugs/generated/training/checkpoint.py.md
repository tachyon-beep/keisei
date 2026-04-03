## Summary

Distributed resume restores identical RNG state on every rank, collapsing per-rank stochasticity because checkpoint RNG is saved only from rank 0 and then loaded by all ranks.

## Severity

- Severity: major
- Priority: P1

## Evidence

- In checkpoint save, only one RNG snapshot is persisted (from the process that writes the file):
  - [`/home/john/keisei/keisei/training/checkpoint.py:63`](file:///home/john/keisei/keisei/training/checkpoint.py:63) to [`/home/john/keisei/keisei/training/checkpoint.py:72`](file:///home/john/keisei/keisei/training/checkpoint.py:72)
  - Saves single `python`, `numpy`, `torch_cpu`, and single `torch_cuda` state.
- On resume, all ranks load that same checkpoint and unconditionally apply those RNG states:
  - [`/home/john/keisei/keisei/training/katago_loop.py:382`](file:///home/john/keisei/keisei/training/katago_loop.py:382) to [`/home/john/keisei/keisei/training/katago_loop.py:402`](file:///home/john/keisei/keisei/training/katago_loop.py:402)
  - [`/home/john/keisei/keisei/training/checkpoint.py:142`](file:///home/john/keisei/keisei/training/checkpoint.py:142) to [`/home/john/keisei/keisei/training/checkpoint.py:160`](file:///home/john/keisei/keisei/training/checkpoint.py:160)
- The project’s own distributed seeding contract expects different RNG streams per rank (`base_seed + rank`):
  - [`/home/john/keisei/keisei/training/distributed.py:140`](file:///home/john/keisei/keisei/training/distributed.py:140) to [`/home/john/keisei/keisei/training/distributed.py:147`](file:///home/john/keisei/keisei/training/distributed.py:147)
  - [`/home/john/keisei/keisei/training/katago_loop.py:830`](file:///home/john/keisei/keisei/training/katago_loop.py:830) to [`/home/john/keisei/keisei/training/katago_loop.py:832`](file:///home/john/keisei/keisei/training/katago_loop.py:832)
- Stochastic action sampling uses torch RNG in rollout:
  - [`/home/john/keisei/keisei/training/katago_loop.py:104`](file:///home/john/keisei/keisei/training/katago_loop.py:104) to [`/home/john/keisei/keisei/training/katago_loop.py:106`](file:///home/john/keisei/keisei/training/katago_loop.py:106)
  - [`/home/john/keisei/keisei/training/katago_loop.py:127`](file:///home/john/keisei/keisei/training/katago_loop.py:127) to [`/home/john/keisei/keisei/training/katago_loop.py:128`](file:///home/john/keisei/keisei/training/katago_loop.py:128)

## Root Cause Hypothesis

`load_checkpoint()` treats RNG restoration as single-process state restoration, but in DDP resume the same checkpoint is loaded on every rank. Because only rank 0 wrote RNG state, all ranks get rank-0 RNG streams after resume, violating intended per-rank randomness and reducing rollout diversity.

## Suggested Fix

In `checkpoint.py`, gate RNG restoration for distributed resumes so rank streams are not clobbered by a single-rank snapshot.

Concrete direction:
- Add a parameter like `restore_rng: bool = True` (or derive from `current_world_size`).
- In `load_checkpoint()`, skip restoring `python`/`numpy`/`torch_cpu`/`torch_cuda` when `current_world_size > 1` (log an explicit warning).
- Keep current behavior for single-rank runs (`current_world_size == 1`).

This keeps reproducible single-process resume while preventing incorrect cross-rank RNG coupling in DDP.
