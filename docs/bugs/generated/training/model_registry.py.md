## Summary

`validate_model_params()` accepts Transformer hyperparameters where `d_model % nhead != 0`, so invalid configs pass registry validation and then fail later inside PyTorch with a low-level `AssertionError`.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- [keisei/training/model_registry.py](/home/john/keisei/keisei/training/model_registry.py):43-49 only instantiates the dataclass and catches `TypeError`; it performs no semantic validation.
- [keisei/training/model_registry.py](/home/john/keisei/keisei/training/model_registry.py):52-56 calls `validate_model_params()` then constructs the model.
- [keisei/training/models/transformer.py](/home/john/keisei/keisei/training/models/transformer.py):30-33 passes `d_model`/`nhead` to `nn.TransformerEncoderLayer`, which requires divisibility.
- Repro from this audit: `build_model("transformer", {"d_model": 32, "nhead": 5, "num_layers": 1})` raised `AssertionError: embed_dim must be divisible by num_heads` (while `validate_model_params(...)` accepted it).

## Root Cause Hypothesis

The registry treats dataclass construction as full validation, but dataclasses do not enforce cross-field invariants like `d_model % nhead == 0`. This allows invalid config through the registry boundary and fails later in model construction.

## Suggested Fix

In `validate_model_params()` (target file), add architecture-specific semantic checks after dataclass creation, e.g. for `transformer`:
- `nhead > 0`
- `d_model > 0`
- `d_model % nhead == 0`
Raise a clear `ValueError` with architecture/field details before `build_model()` instantiates the module.
---
## Summary

`validate_model_params()` allows `se_resnet` settings where `se_reduction > channels`, which creates zero-width SE layers and silently degrades the model instead of rejecting invalid params.

## Severity

- Severity: minor
- Priority: P2 (downgraded: minimal evidence)
## Evidence

- [keisei/training/model_registry.py](/home/john/keisei/keisei/training/model_registry.py):43-49 has no bounds/invariant checks for `se_resnet` params.
- [keisei/training/models/se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py):51 computes `se_hidden = channels // se_reduction`.
- [keisei/training/models/se_resnet.py](/home/john/keisei/keisei/training/models/se_resnet.py):52-53 builds `nn.Linear(channels, se_hidden)` and `nn.Linear(se_hidden, channels * 2)`.
- Repro from this audit:
  - `validate_model_params("se_resnet", {"channels": 8, "se_reduction": 16})` succeeded.
  - `build_model("se_resnet", {"channels": 8, "se_reduction": 16})` succeeded but emitted `UserWarning: Initializing zero-element tensors is a no-op`.

## Root Cause Hypothesis

Registry validation currently checks only constructor signature compatibility, not value-domain constraints. For `channels < se_reduction`, integer division yields `se_hidden == 0`, creating degenerate SE MLP layers that do not provide intended capacity.

## Suggested Fix

In `validate_model_params()` (target file), add `se_resnet` semantic checks such as:
- `channels > 0`
- `se_reduction > 0`
- `channels // se_reduction >= 1` (or equivalently `se_reduction <= channels`)
- Optional: positive checks for `num_blocks`, `policy_channels`, and FC sizes.
Raise `ValueError` with actionable messages before model construction.
