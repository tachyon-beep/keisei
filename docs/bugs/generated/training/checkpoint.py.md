## Summary

`save_checkpoint()` writes directly to the final `.pt` path via `torch.save(...)` without an atomic temp-file swap, so process crash/interruption can leave a partially written checkpoint that later fails resume.

## Severity

- Severity: major
- Priority: P2

## Evidence

- `/home/john/keisei/keisei/training/checkpoint.py:79` writes directly to final path:
```python
torch.save(data, path)
```
- No `.tmp` write + `rename`/`replace` pattern exists in this file (`save_checkpoint` only does `mkdir` then direct save).
- Resume path relies on this file being valid and will hard-fail on corruption at load:
  - `/home/john/keisei/keisei/training/checkpoint.py:95` (`torch.load(...)`)
  - `/home/john/keisei/tests/test_checkpoint.py:77-84` explicitly confirms corrupted checkpoint raises.
- Main training loop periodically writes checkpoints, so this path is on the critical training/resume path:
  - `/home/john/keisei/keisei/training/katago_loop.py:1621-1628`

## Root Cause Hypothesis

Checkpoint persistence in `save_checkpoint()` uses a non-atomic single-file write. If the process is killed, OOMs, or host crashes mid-write, the final checkpoint path can contain truncated/invalid bytes. On next startup, `load_checkpoint()` attempts to deserialize that file and fails. Trigger condition: interruption during `torch.save(...)` to the target checkpoint file.

## Suggested Fix

In `save_checkpoint()` (target file), switch to atomic write semantics:

1. Write to a sibling temp file (for example `path.with_suffix(path.suffix + ".tmp")`).
2. `torch.save(data, tmp_path)`.
3. Optionally flush/fsync temp file handle/directory for stronger durability.
4. Atomically replace final file (`tmp_path.replace(path)` or `os.replace`).
5. Best-effort cleanup of stale temp on exceptions.

This keeps either the old valid checkpoint or the new valid checkpoint, avoiding half-written files at the final path.
