## Summary

`SLTrainer` selects AMP dtype using global CUDA availability instead of the model’s actual device, which can choose `float16` for CPU autocast and cause runtime failures or unstable mixed-precision behavior on CPU builds.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `/home/john/keisei/keisei/sl/trainer.py:81-90`:
  - AMP dtype is picked by `torch.cuda.is_available()`/`torch.cuda.is_bf16_supported()`, not `self.device.type`.
  - On CPU training, fallback is `torch.float16`.
- `/home/john/keisei/keisei/sl/trainer.py:128-132`:
  - That dtype is used in `autocast(device_type="cpu", dtype=self._amp_dtype, enabled=...)` when model is on CPU.
- `/home/john/keisei/keisei/training/katago_ppo.py:23-29`:
  - PPO path already treats CPU as `bfloat16` specifically (`# CPU autocast only supports bfloat16`), showing inconsistent and safer logic elsewhere in-repo.

## Root Cause Hypothesis

AMP selection logic in `SLTrainer` was copied/implemented with CUDA-centric checks and did not branch on the actual training device. This is triggered when `use_amp=True` and training is on CPU (especially on environments where CPU float16 autocast is unsupported or numerically fragile).

## Suggested Fix

In `SLTrainer.__init__`, derive AMP dtype from `self.device.type` first, matching PPO logic:
- If `not use_amp`: placeholder dtype.
- If `self.device.type == "cpu"`: use `torch.bfloat16`.
- Else on CUDA: use `bfloat16` if supported, otherwise `float16`.

Also gate scaler to CUDA explicitly (`GradScaler(enabled=config.use_amp and self.device.type == "cuda")`) to avoid CPU AMP scaler mismatch.
---
## Summary

`SLConfig` does not validate `num_workers`, so negative values are accepted until `DataLoader` construction fails at runtime for non-empty datasets.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- `/home/john/keisei/keisei/sl/trainer.py:24` defines `num_workers`.
- `/home/john/keisei/keisei/sl/trainer.py:32-40` validates other fields but not `num_workers`.
- `/home/john/keisei/keisei/sl/trainer.py:105` passes `config.num_workers` directly into `DataLoader` when data exists.

## Root Cause Hypothesis

Validation coverage in `SLConfig.__post_init__` is incomplete. Invalid worker settings are not rejected at config boundary and instead fail deeper in the PyTorch dataloader path, producing later and less actionable errors.

## Suggested Fix

Add explicit `num_workers` validation in `SLConfig.__post_init__`:
- Require integer type.
- Require `num_workers >= 0`.
- Raise `ValueError` with a direct message (for example, `num_workers must be >= 0`).
