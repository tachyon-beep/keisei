## Summary

`SLTrainer` hard-codes AMP autocast device type to `"cpu"` for every non-CUDA device, which can make `train_epoch()` crash (or misconfigure AMP) when the model is on devices like `mps`.

## Severity

- Severity: major
- Priority: P2

## Evidence

- [`/home/john/keisei/keisei/sl/trainer.py:86`](/home/john/keisei/keisei/sl/trainer.py:86) to [`/home/john/keisei/keisei/sl/trainer.py:92`](/home/john/keisei/keisei/sl/trainer.py:92):
```python
elif self.device.type == "cpu":
    self._amp_dtype = torch.bfloat16
elif torch.cuda.is_bf16_supported():
    self._amp_dtype = torch.bfloat16
else:
    self._amp_dtype = torch.float16
self._amp_device_type = "cuda" if self.device.type == "cuda" else "cpu"
```
- [`/home/john/keisei/keisei/sl/trainer.py:130`](/home/john/keisei/keisei/sl/trainer.py:130) to [`/home/john/keisei/keisei/sl/trainer.py:134`](/home/john/keisei/keisei/sl/trainer.py:134):
```python
with autocast(
    device_type=self._amp_device_type,
    dtype=self._amp_dtype,
    enabled=self.config.use_amp,
):
```
- [`/home/john/keisei/keisei/training/models/katago_base.py:71`](/home/john/keisei/keisei/training/models/katago_base.py:71) to [`/home/john/keisei/keisei/training/models/katago_base.py:74`](/home/john/keisei/keisei/training/models/katago_base.py:74) confirms the same `(device_type, dtype)` is used inside model forward.
- If `self.device.type == "mps"` and `use_amp=True`, trainer sets `device_type="cpu"` and often `dtype=torch.float16`; CPU autocast does not support float16, so this is a runtime failure path controlled by logic in `trainer.py`.

## Root Cause Hypothesis

The AMP setup assumes only two device classes (`cuda` and `cpu`) and maps all other device types to `cpu`. That fallback is invalid for non-CPU accelerators and can pair CPU autocast with unsupported dtype choices, causing failures when AMP is enabled.

## Suggested Fix

In `SLTrainer.__init__`, derive AMP settings by actual `self.device.type` instead of binary CUDA/CPU branching. For example:

- Use `self._amp_device_type = self.device.type`.
- Choose dtype per device type (`cpu -> bfloat16`, `cuda -> bf16 if supported else fp16`, `mps -> float16` if supported).
- If a device type is unsupported for AMP autocast, disable AMP with a warning (`self.config.use_amp=False` behaviorally) rather than forcing `cpu` autocast.

This keeps AMP configuration internally consistent with the model/device used in `train_epoch()`.
