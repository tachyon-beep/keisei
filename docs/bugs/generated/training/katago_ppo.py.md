## Summary

`KataGoPPOAlgorithm` initializes `GradScaler` with `enabled=params.use_amp` unconditionally, which triggers incorrect CUDA-only scaler activation attempts in non-CUDA runs (for example CPU AMP), producing warnings and inconsistent AMP behavior.

## Severity

- Severity: minor
- Priority: P3

## Evidence

- Target file uses unconditional scaler enable:
  - `/home/john/keisei/keisei/training/katago_ppo.py:367`
  - `self.scaler = GradScaler(enabled=params.use_amp)`
- Same codebase already handles this correctly in SL path:
  - `/home/john/keisei/keisei/sl/trainer.py:77`
  - `self.scaler = GradScaler(enabled=config.use_amp and self.device.type == "cuda")`
- Runtime behavior confirms non-CUDA warning/disable path:
  - `torch.amp.GradScaler(enabled=True)` on non-CUDA emits: “GradScaler is enabled, but CUDA is not available. Disabling.”

## Root Cause Hypothesis

`katago_ppo.py` enables gradient scaling based only on config (`use_amp`) and not actual device type. `GradScaler` is CUDA-oriented in this usage, so CPU AMP configurations hit a warning path and silent scaler disable, indicating mismatched AMP/scaler device assumptions.

## Suggested Fix

In `/home/john/keisei/keisei/training/katago_ppo.py`, gate scaler enablement by device type (matching the SL trainer pattern), for example:

- Compute `device = next(model.parameters()).device` (already done earlier).
- Replace line 367 with:
  - `self.scaler = GradScaler(enabled=params.use_amp and device.type == "cuda")`

Optional hardening:
- If `params.use_amp` is true on non-CUDA, log a one-time info message that AMP runs without grad scaling on that device.
