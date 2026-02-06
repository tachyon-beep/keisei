# Code Analysis: `keisei/training/training_loop_manager.py`

## 1. Purpose & Role

`TrainingLoopManager` encapsulates the main training iteration logic, separated from the `Trainer` class for modularity. It handles epoch-based experience collection (both sequential and parallel modes), step execution, episode lifecycle management, display updates, and SPS (steps-per-second) calculation. It is instantiated by `Trainer` and receives the full `Trainer` instance as its primary dependency.

## 2. Interface Contracts

### Constructor (`__init__`, lines 39-74)
- **Input**: `trainer: Trainer` -- full trainer instance providing access to all managers and components.
- **Side effects**: Copies references to `config`, `agent`, `buffer`, `step_manager`, `display`, `callbacks` from trainer. Optionally initializes `ParallelManager` if `config.parallel.enabled`.
- **Invariant**: `episode_state` is None until `set_initial_episode_state()` is called.

### `set_initial_episode_state(initial_episode_state)` (line 76)
- Must be called before `run()`. Sets the initial game observation state.

### `run()` (lines 80-183)
- Main synchronous training loop. Iterates epochs until `global_timestep >= total_timesteps`.
- Each epoch collects `steps_per_epoch` experiences, then triggers PPO update and callbacks.
- Raises `KeyboardInterrupt` on user interruption, or `RuntimeError/ValueError/AttributeError` on training errors.

### `run_async()` (lines 226-323)
- Async variant of `run()` for native async callback execution.
- Nearly identical to `run()` but awaits async callbacks directly instead of using `asyncio.run()`.

## 3. Correctness Analysis

### Stale References in Constructor (lines 45-50)
- Lines 45-50 copy `trainer.agent`, `trainer.experience_buffer`, `trainer.step_manager`, `trainer.display`, `trainer.callbacks` as direct references. These are captured at construction time.
- However, `self.step_manager` is used throughout this class (lines 463, 512-513, 519-520, 522, 533-534) while `self.trainer.step_manager` is used at line 558. The two references point to the same object at construction, but if `trainer.step_manager` were ever reassigned after `TrainingLoopManager` construction, `self.step_manager` would be stale.
- Similarly, `self.display` (line 49) is captured before `trainer.display` is fully set up -- `trainer.display` is assigned at line 183 of `trainer.py`, but `TrainingLoopManager` is constructed at line 187 of `trainer.py`, so the reference is valid at construction time.
- `self.agent` (line 46) and `self.buffer` (line 47) are captured but `self.agent` is used at lines 372-375 while `self.trainer.agent` is used at lines 102, 107, 247. This inconsistency means some code paths would miss a reassignment of `trainer.agent` while others would pick it up. In practice, these are never reassigned, but the inconsistent access pattern is fragile.

### Epoch Loop Structure (lines 119-168, `run()`)
- The while loop at line 120-123 checks `global_timestep < total_timesteps`.
- After `_run_epoch()` (line 126), lines 128-135 re-check the condition and break if met. This is a defensive double-check that is correct.
- Lines 137-149: PPO update is performed using `self.episode_state.current_obs`. If `episode_state` is None or `current_obs` is None, the update is skipped with a warning. This is correct since an episode might end exactly at the epoch boundary, leaving no valid observation.
- Lines 142 and 149: `set_processing(False)` is called in both branches of the if/else. If `perform_ppo_update` raises an exception, `set_processing(False)` is never called, leaving the metrics manager in a stale "processing" state. This is not wrapped in a try/finally.

### Sequential Epoch (`_run_epoch_sequential`, lines 440-454)
- Simple loop: process step, handle episode, increment counter, update display.
- No check against `total_timesteps` within this loop -- the step count is bounded only by `steps_per_epoch`. The timestep check happens inside `_process_step_and_handle_episode` (line 504-508), which returns `False` to break the outer loop.

### Parallel Epoch (`_run_epoch_parallel`, lines 345-438)
- Lines 361-366: Loop bounded by `steps_per_epoch`, `total_timesteps`, and `max_collection_attempts` (50). The triple guard prevents infinite loops.
- Lines 384-411: Collects experiences from workers. On successful collection, increments timestep by the batch amount (line 392-393) and updates SPS counter (line 396).
- Lines 414-431: Catches a broad set of exceptions during parallel collection. On max attempts, falls back to sequential mode (line 431). This fallback is correct and graceful.
- Line 405: Display update every 100 steps is hardcoded rather than configurable.

### Step Processing (`_process_step_and_handle_episode`, lines 500-544)
- Lines 504-508: Checks timestep limit before processing.
- Lines 510-517: If `episode_state` is None, resets and returns True. If reset fails (returns None), raises RuntimeError (line 516). This is correct.
- Lines 528-536: If step fails, logs warning and resets episode. Does not increment timestep on failed steps, which is correct -- failed steps should not count.
- Line 542: `increment_timestep()` is called after a successful step. This is called once per step in sequential mode, matching the expected behavior.

### Episode End Handling (`_handle_successful_step`, lines 456-498)
- Lines 477-485: Calls `step_manager.handle_episode_end()`, which returns a new episode state and winner color string.
- Lines 487-491: Converts winner color string to `Color` enum. The logic handles "black", "white", and anything else (mapped to None for draw). This is correct.
- Line 492: `update_episode_stats(winner_color_enum)` updates win/loss/draw counters.

### Async Loop (`run_async`, lines 226-323)
- This is a near-complete copy of `run()` with two differences: (1) async callbacks are awaited directly at line 301, and (2) the error message references "run_async" at line 317.
- The duplication between `run()` and `run_async()` is substantial (~100 lines). Any bug fix in one must be manually propagated to the other, creating a maintenance risk.

### Async Callback Bridge (`_run_async_callbacks_sync`, lines 185-224)
- Lines 194-207: Checks for an existing event loop. If one exists, skips async callbacks with a warning. If no loop exists (RuntimeError), proceeds to create one.
- Line 216: Uses `asyncio.run()` which creates a new event loop, runs the coroutine, and tears down the loop. This is correct but creates and destroys an event loop on every callback invocation, which has overhead.
- Lines 218-224: Broad `except Exception` catch returns None on any failure. This prevents async callback errors from crashing training, which is appropriate for optional callbacks.

### Display Update Throttling (`_update_display_if_needed`, lines 597-649)
- Line 605-607: Uses `getattr` to read `config.training.rich_display_update_interval_seconds` with a default of 0.2. This `getattr` pattern suggests the field may not exist on all config versions.
- Lines 617-618: Uses `setdefault` for pending updates, meaning existing values are not overwritten. This is intentional -- earlier updates within the same display cycle take precedence.
- Lines 638-643: WebUI update is called in parallel with display update. The WebUI `update_progress` call happens synchronously in the training loop, which could block if the WebSocket connection is slow. However, this risk is mitigated by the WebUI manager's internal design (which typically queues messages).
- Lines 647-648: SPS counters are reset only when the display is actually updated (inside the time check). This means SPS is calculated over the actual display interval, not a fixed window. This is correct for a smoothed SPS metric.

### `_log_episode_metrics` (lines 546-595)
- Lines 565-582: Computes win/draw rates by dividing wins/draws by total games. Division-by-zero is guarded by `if total_games > 0`.
- Line 550: `ep_metrics_str` is computed but used only as a progress update value (line 586). The variable `turns_count = ep_len` (line 551) is an alias with no additional logic, suggesting it was intended for a more nuanced metric (e.g., player-turns vs. total plies).

## 4. Robustness & Error Handling

### Processing State Leak (lines 138-149)
- If `perform_ppo_update` raises an exception, `set_processing(False)` at line 142 is skipped. The exception propagates to the outer try/except in `run()` (line 176), but `set_processing(False)` is never called. This leaves the metrics manager in a "processing" state permanently. The outer error handler does not reset this flag.

### Parallel Fallback (lines 426-431)
- When parallel collection fails `max_collection_attempts` times, the code falls back to sequential mode. However, the parallel manager is not stopped or cleaned up -- its workers may still be running. The variable `self.parallel_manager` is set to `None` at line 114 only in the startup path, not in the fallback path at line 431. The fallback at line 431 calls `_run_epoch_sequential` and returns, but `self.parallel_manager` remains set, meaning subsequent epochs will still attempt parallel collection.

### Redundant step_manager None Checks
- Lines 463-464, 512-513, 519-520, 533-534 all check `if self.step_manager is None: raise RuntimeError(...)`. The step_manager is set in the constructor (line 48) and never modified. These are defensive checks that add noise but no real protection since `step_manager` being None would indicate a construction failure that should have been caught earlier.

### Exception Scope in `run()` (lines 176-183)
- Catches `RuntimeError, ValueError, AttributeError`. Notably missing: `TypeError`, `IndexError`, `KeyError`. If the step manager or agent produces one of these, the error will propagate uncaught from the `try` block. The `finally` in `Trainer.run_training_loop()` (line 463-464) provides a safety net for finalization, but the error message from `TrainingLoopManager` will not be logged.

## 5. Performance & Scalability

### SPS Calculation
- Lines 610-614: SPS is computed as `steps / time_delta`. The counter `steps_since_last_time_for_sps` resets at display intervals (line 648), providing a rolling average. This is efficient and accurate.

### Display Update Throttling
- Line 609: Display updates are gated by a time-based interval (default 0.2s). This prevents the Rich console from being overwhelmed by frequent updates, which is important for training runs at high steps-per-second.

### Parallel Collection Polling
- Line 409: When no experiences are collected, the code sleeps for 10ms. This is a simple polling approach. For high-throughput parallel collection, a condition variable or event-based notification would be more efficient, but 10ms polling is acceptable for the expected throughput range.

### `asyncio.run()` Per Callback Invocation (line 216)
- Creates and destroys an event loop on every async callback execution. For callbacks that execute rarely (e.g., periodic evaluation), this is negligible. If async callbacks execute every epoch, the overhead of loop creation/teardown accumulates.

## 6. Security & Safety

### No Direct Security Concerns
- The module does not handle user input, file I/O, or network operations directly (those are delegated to managers).
- WebUI display updates (lines 638-643) send training state to WebSocket clients. The data includes training metrics but not model weights or configuration secrets.

## 7. Maintainability

### Code Duplication Between `run()` and `run_async()`
- Lines 80-183 (`run`) and 226-323 (`run_async`) are approximately 80% identical. The only material differences are:
  - `run_async` is `async def` (line 226)
  - `run_async` awaits async callbacks directly (line 301) instead of using `_run_async_callbacks_sync`
  - Different error messages reference "run_async" vs "run"
- This duplication is a significant maintenance risk. Any fix or feature added to one method must be manually propagated to the other.

### Mixed Access Patterns
- Some methods use `self.agent` (from constructor copy) while others use `self.trainer.agent`. Same for `self.step_manager` vs `self.trainer.step_manager` (line 558). This inconsistency makes it unclear which reference is authoritative.

### Constants
- `STEP_MANAGER_NOT_AVAILABLE_MSG` (line 15) is defined as a module-level constant, used in 4 locations. This is good practice for consistent error messages.
- `max_collection_attempts = 50` (line 355) and display update `100 steps` (line 405) are hardcoded magic numbers.

### File Length
- At 694 lines, this is the largest file in the package. The complexity is spread across 15 methods, with the longest being `_run_epoch_parallel` at ~93 lines. Individual method complexity is reasonable.

## 8. Verdict

**NEEDS_ATTENTION**

Key concerns:
1. **Processing state leak** (lines 138-149): If `perform_ppo_update` raises, `set_processing(False)` is never called, leaving metrics manager in a stale state.
2. **Code duplication** between `run()` and `run_async()` (~100 lines). A bug fix in one will likely be missed in the other.
3. **Stale reference risk**: Constructor copies (`self.agent`, `self.buffer`, `self.step_manager`) vs direct trainer access (`self.trainer.agent`) are used inconsistently. While reassignment never happens in practice, this is structurally fragile.
4. **Parallel fallback does not clean up workers**: When parallel collection fails and falls back to sequential (line 431), the parallel manager's workers are not stopped, and `self.parallel_manager` is not set to None, meaning subsequent epochs will retry parallel mode.
5. **`asyncio.run()` overhead**: Creating a new event loop per callback invocation (line 216) adds unnecessary overhead if async callbacks are frequent.
