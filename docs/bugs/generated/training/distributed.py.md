## Summary

`setup_distributed()` defaults to `backend="nccl"` even when CUDA is unavailable, so CPU-only `torchrun` execution can fail during process-group initialization instead of selecting a valid backend (`gloo`) or failing with a clear precondition error.

## Severity

- Severity: major
- Priority: P1

## Evidence

- Target file hardcodes NCCL default and still initializes with that backend when CUDA is absent:
  - [`/home/john/keisei/keisei/training/distributed.py:97`](/home/john/keisei/keisei/training/distributed.py:97) defines `setup_distributed(..., backend: str = "nccl")`
  - [`/home/john/keisei/keisei/training/distributed.py:112`](/home/john/keisei/keisei/training/distributed.py:112) only gates `torch.cuda.set_device(...)` on CUDA availability
  - [`/home/john/keisei/keisei/training/distributed.py:114`](/home/john/keisei/keisei/training/distributed.py:114) still calls `dist.init_process_group(backend=backend)` with `"nccl"`
- Main training path calls `setup_distributed(dist_ctx)` without overriding backend:
  - [`/home/john/keisei/keisei/training/katago_loop.py:818`](/home/john/keisei/keisei/training/katago_loop.py:818)
- The file docstring itself states `"gloo" for CPU-only (tests)`, but current default path does not auto-select it:
  - [`/home/john/keisei/keisei/training/distributed.py:105`](/home/john/keisei/keisei/training/distributed.py:105)

## Root Cause Hypothesis

Backend selection is static (`"nccl"` default) rather than device-aware. When `RANK`/`WORLD_SIZE` indicate distributed mode but CUDA is not available (CPU CI, CPU test host, misconfigured GPU node), initialization attempts NCCL anyway and fails in `init_process_group`.

## Suggested Fix

In `setup_distributed`, resolve backend dynamically before `init_process_group`, for example:

- If caller did not explicitly set a backend, choose `"nccl"` when `torch.cuda.is_available()` else `"gloo"`.
- If backend is explicitly `"nccl"` and CUDA is unavailable, raise a clear `RuntimeError` before init with actionable guidance.
- Keep existing `torch.cuda.set_device` only for CUDA/NCCL paths.

Concrete change belongs in [`/home/john/keisei/keisei/training/distributed.py`](/home/john/keisei/keisei/training/distributed.py).
