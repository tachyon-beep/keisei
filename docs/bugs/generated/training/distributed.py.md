## Summary

`setup_distributed()` only sets the CUDA current device for `nccl`, so in CUDA distributed runs that use `gloo`, rank-local CUDA context is wrong and `seed_all_ranks()` seeds the wrong GPU RNG stream.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target file: [`distributed.py`](/home/john/keisei/keisei/training/distributed.py:122)
- In `setup_distributed()`, current CUDA device is set only inside the `backend == "nccl"` branch:
```python
if backend == "nccl":
    torch.cuda.set_device(ctx.local_rank)
dist.init_process_group(backend=backend)
```
- Target file: [`distributed.py`](/home/john/keisei/keisei/training/distributed.py:154)
- In `seed_all_ranks()`, CUDA seeding uses current device:
```python
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
```
- Integration path: [`katago_loop.py`](/home/john/keisei/keisei/training/katago_loop.py:1764) then [`katago_loop.py`](/home/john/keisei/keisei/training/katago_loop.py:1778)
  - `setup_distributed(dist_ctx)` is called before `seed_all_ranks(...)`.
  - If backend is CUDA+`gloo`, current device is never pinned to `local_rank`, so rank>0 may seed/default to GPU 0.

## Root Cause Hypothesis

The code assumes CUDA device pinning is only needed for `nccl`, but CUDA default-device semantics still matter under `gloo` when tensors/models live on `cuda:{local_rank}`. This mismatch is triggered when `torch.cuda.is_available()` is true, distributed mode is active, and backend is `gloo` (explicit override). Then rank-local RNG/device-dependent operations can target the wrong GPU.

## Suggested Fix

In `/home/john/keisei/keisei/training/distributed.py`, set rank-local CUDA device for any distributed CUDA run, not just `nccl`, before process-group init. Also harden seeding by using all-device seeding.

Concrete change:
- In `setup_distributed()`:
  - Move `torch.cuda.set_device(ctx.local_rank)` out of the `backend == "nccl"` conditional.
  - Guard with `if torch.cuda.is_available():`.
- In `seed_all_ranks()`:
  - Replace `torch.cuda.manual_seed(seed)` with `torch.cuda.manual_seed_all(seed)` (or at minimum rely on guaranteed `set_device` + keep current behavior).
