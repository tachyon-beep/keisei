## Summary

Transformer checkpoints saved before `_row_idx`/`_col_idx` were introduced will fail to load because those constant index buffers are persisted in `state_dict`, creating a backward-compatibility break in checkpoint restore paths.

## Severity

- Severity: major
- Priority: P2

## Evidence

- Target file persists new buffers as part of model state:
  - `/home/john/keisei/keisei/training/models/transformer.py:64`
  - `/home/john/keisei/keisei/training/models/transformer.py:65`
  - Code: `self.register_buffer("_row_idx", torch.arange(self.BOARD_SIZE))` and `self.register_buffer("_col_idx", torch.arange(self.BOARD_SIZE))`
- Multiple load paths use strict `load_state_dict` semantics (default strict=True or explicit strict=True), so missing/new keys are load errors:
  - `/home/john/keisei/keisei/training/checkpoint.py:114`
  - `/home/john/keisei/keisei/training/evaluate.py:90`
  - `/home/john/keisei/keisei/training/evaluate.py:97`
  - `/home/john/keisei/keisei/training/opponent_store.py:731` (explicit `strict=True`)

## Root Cause Hypothesis

The positional index tensors are constants used only to index embeddings, but they are currently registered as persistent buffers. That means they become required checkpoint keys. Any older Transformer checkpoint that predates these buffers will not contain `_row_idx`/`_col_idx`; strict load then raises missing-key errors.

## Suggested Fix

In `transformer.py`, register these index buffers as non-persistent so they still move with `.to(device)` but are excluded from checkpoint state:

- Change:
  - `self.register_buffer("_row_idx", torch.arange(self.BOARD_SIZE))`
  - `self.register_buffer("_col_idx", torch.arange(self.BOARD_SIZE))`
- To:
  - `self.register_buffer("_row_idx", torch.arange(self.BOARD_SIZE, dtype=torch.long), persistent=False)`
  - `self.register_buffer("_col_idx", torch.arange(self.BOARD_SIZE, dtype=torch.long), persistent=False)`

This keeps runtime behavior identical while restoring backward compatibility for older checkpoints.
