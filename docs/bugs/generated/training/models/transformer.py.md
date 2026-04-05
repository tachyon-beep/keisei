## Summary

`TransformerModel` accepts `TransformerParams.num_layers=0` (or negative), then crashes at runtime on the first forward pass with `IndexError` instead of failing fast with config validation.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `/home/john/keisei/keisei/training/models/transformer.py:13-17` defines `TransformerParams` with no semantic validation (`num_layers` can be `0`).
- `/home/john/keisei/keisei/training/models/transformer.py:37-40` passes `params.num_layers` directly into `nn.TransformerEncoder(...)`.
- `/home/john/keisei/keisei/training/models/transformer.py:75` calls `self.encoder(x)`; with `num_layers=0`, this path raises `IndexError: index 0 is out of range`.
- Verified by direct repro in this repo:
  - Construct `TransformerModel(TransformerParams(d_model=32, nhead=4, num_layers=0))`
  - Run forward on `torch.randn(2, 50, 9, 9)`
  - Observed exception: `IndexError index 0 is out of range`.

## Root Cause Hypothesis

Parameter boundary checks are incomplete in the target file: `TransformerParams`/`TransformerModel.__init__` do not enforce `num_layers >= 1`, so invalid architecture configs are instantiated and only fail later inside PyTorch internals during forward execution.

## Suggested Fix

In `/home/john/keisei/keisei/training/models/transformer.py`, add explicit validation in the target file (preferably `TransformerParams.__post_init__`, alternatively `TransformerModel.__init__`), e.g. reject:

- `num_layers <= 0`
- `d_model <= 0`
- `nhead <= 0`
- `d_model % nhead != 0`

Raise clear `ValueError`s so bad configs fail immediately with actionable messages.
