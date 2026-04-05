## Summary

`DemonstratorRunner` can be configured with `device="cuda"` even when CUDA is unavailable, causing repeated per-game failures and effectively disabling demonstrator gameplay by default on CPU-only hosts.

## Severity

- Severity: major
- Priority: P1

## Evidence

- `keisei/training/demonstrator.py:66` stores `self.device = device` without validating availability.
- `keisei/training/demonstrator.py:70-71` only gates CUDA stream creation, not CUDA model/tensor usage.
- `keisei/training/demonstrator.py:168-169` always calls `self.store.load_opponent(..., device=self.device)`.
- `keisei/training/opponent_store.py:732-733` does `model = model.to(device)`; with `device="cuda"` and no CUDA, this raises.
- `keisei/config.py:477` sets `DemonstratorConfig.device` default to `"cuda"`.

## Root Cause Hypothesis

Device selection is treated as a raw string in `DemonstratorRunner` and never normalized/fallback-checked against runtime CUDA availability. On CPU-only machines (or misconfigured CUDA), model loading fails every game attempt, so demonstrator slots never run.

## Suggested Fix

In `DemonstratorRunner.__init__` (`keisei/training/demonstrator.py`), normalize and validate the requested device once:

- If requested device starts with `"cuda"` and `torch.cuda.is_available()` is false, log a warning and fallback to `"cpu"`.
- Optionally validate CUDA index (e.g., `"cuda:1"` when only one GPU exists) and fallback/log.
- Use the resolved device for both model loads and tensor `.to(...)`.
---
## Summary

`stop()` is not promptly honored during active games because the loop uses `time.sleep(self.move_delay)`, which blocks thread shutdown for up to one full move interval.

## Severity

- Severity: minor
- Priority: P2

## Evidence

- `keisei/training/demonstrator.py:227` uses `time.sleep(self.move_delay)` inside the per-move loop.
- `keisei/training/demonstrator.py:202` checks `not self._stop_event.is_set()` only at loop boundaries, so a stop request during sleep waits until sleep completes.
- With default-like low speeds (e.g., `moves_per_minute=1`), `move_delay` is 60s (`keisei/training/demonstrator.py:65`), making shutdown latency large.

## Root Cause Hypothesis

The sleep primitive is non-interruptible with respect to the thread’s stop event. The thread cannot react until sleep returns, so `stop()` responsiveness is tied to `move_delay`.

## Suggested Fix

Replace the blocking sleep with event-based waiting in `keisei/training/demonstrator.py`:

- Change `time.sleep(self.move_delay)` to `if self._stop_event.wait(timeout=self.move_delay): break`.
- This preserves pacing but allows immediate exit when `stop()` is called.
