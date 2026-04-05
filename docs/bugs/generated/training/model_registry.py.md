## Summary

`validate_model_params()` accepts `transformer.num_layers=0`, so an invalid config passes registry validation and crashes later at first forward pass with `IndexError`.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `keisei/training/model_registry.py:52-61` validates only `nhead` and `d_model` for `"transformer"`; no `num_layers` check exists.
- `keisei/training/model_registry.py:76-80` builds models immediately from validated params, so bad configs propagate.
- `keisei/training/models/transformer.py:37-40` constructs `nn.TransformerEncoder(..., num_layers=params.num_layers)`.
- `keisei/training/models/transformer.py:75` calls `self.encoder(x)`; with `num_layers=0`, PyTorch raises at runtime (`IndexError: index 0 is out of range`).
- Integration path: `keisei/training/katago_loop.py:433` calls `build_model(...)` during loop init, so this misconfig survives startup and fails only once forward executes.

## Root Cause Hypothesis

Registry-level semantic validation for transformer params is incomplete: it checks divisibility/positivity of `d_model`/`nhead` but omits `num_layers >= 1`, allowing an invalid architecture config through the boundary.

## Suggested Fix

In `/home/john/keisei/keisei/training/model_registry.py`, extend the transformer validation block:

- Add `if validated.num_layers <= 0: raise ValueError("transformer: num_layers must be > 0 ...")`.

This keeps failure at config/model-build boundary with a clear error instead of a late PyTorch internal exception.
---
## Summary

`validate_model_params()` accepts degenerate width settings (for example `resnet.hidden_size=0` and `mlp.hidden_sizes` containing `0`), causing either runtime tensor-shape failure or silent constant-output models.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `keisei/training/model_registry.py:51-72` contains semantic checks only for `"transformer"` and `"se_resnet"`; no semantic checks exist for `"resnet"` or `"mlp"`.
- `keisei/training/models/resnet.py:39` uses `hidden_size` directly as Conv2d out-channels; with `hidden_size=0`, first forward fails (`RuntimeError ... expected weight to be at least 1 ... got [0, 50, 3, 3]`).
- `keisei/training/models/mlp.py:25-33` uses each entry in `hidden_sizes` as linear layer width; if a width is `0`, the final heads become bias-only (`in_features=0`), producing identical outputs for different inputs (silent training corruption).
- Integration path: `keisei/training/katago_loop.py:433` builds model from config via registry, so this bad config is not rejected early.

## Root Cause Hypothesis

`model_registry.validate_model_params()` is treated as the config boundary validator but does not enforce positive layer widths/depth for all registered architectures, allowing invalid-but-constructible param sets that fail later or behave pathologically.

## Suggested Fix

In `/home/john/keisei/keisei/training/model_registry.py`, add architecture-specific checks:

- For `"resnet"`: `hidden_size > 0`, `num_layers >= 1`.
- For `"mlp"`: `hidden_sizes` non-empty and all entries `> 0`.

Raise explicit `ValueError` messages per field so invalid configs fail immediately with actionable errors.
