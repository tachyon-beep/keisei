# Code Analysis: `keisei/evaluation/core/background_tournament.py`

**Lines:** 538 (including blanks and docstrings)
**Last read:** 2026-02-07

---

## 1. Purpose & Role

This module implements `BackgroundTournamentManager`, which provides asynchronous tournament execution that can run in the background without blocking the training loop. It wraps a `TournamentEvaluator` with progress tracking, persistence, lifecycle management (start/cancel/shutdown), and optional real-time progress callbacks. It also defines `TournamentStatus` (an enum for lifecycle states) and `TournamentProgress` (a dataclass for tracking per-tournament metrics).

---

## 2. Interface Contracts

### `TournamentStatus(Enum)`
Six states: `CREATED`, `RUNNING`, `PAUSED`, `COMPLETED`, `FAILED`, `CANCELLED`.

### `TournamentProgress` (dataclass)
Tracks tournament state including: ID, status, game counts (total/completed/failed), round info, timing, performance metrics (games/second, avg duration), results list, and standings dict. Properties:
- `completion_percentage` -- `(completed / total) * 100` (safe for `total_games == 0`).
- `is_active` -- True if `RUNNING` or `PAUSED`.
- `is_complete` -- True if `COMPLETED`, `FAILED`, or `CANCELLED`.

### `BackgroundTournamentManager`

| Method | Contract |
|---|---|
| `__init__(max_concurrent_tournaments, progress_callback, result_storage_dir)` | Creates semaphore, shutdown event, storage directory. |
| `start_tournament(config, agent_info, opponents, name, priority)` | Creates tournament task, returns tournament ID string. |
| `get_tournament_progress(id)` | Returns `TournamentProgress` or `None`. |
| `list_active_tournaments()` | Returns list of tournaments where `is_active` is True. |
| `list_all_tournaments()` | Returns all tracked tournaments. |
| `cancel_tournament(id)` | Cancels the asyncio task, updates status. Returns bool success. |
| `shutdown()` | Sets shutdown event, cancels all tournaments, gathers tasks. |

Internal methods:
- `_execute_tournament(...)` -- main execution loop inside semaphore.
- `_execute_with_progress_tracking(...)` -- iterates opponents, calls evaluator internals.
- `_update_progress(...)` -- thread-safe progress update with callback invocation.
- `_save_tournament_results(...)` -- JSON persistence to disk.
- `_cleanup_tournament(...)` -- removes tasks and locks from tracking dicts.

---

## 3. Correctness Analysis

**Lines 29-35: Module-level lazy import of `TournamentEvaluator`.** The module sets `TournamentEvaluator = None` and tries to import `_TournamentEvaluator` from `..strategies.tournament`. If the import fails, `TournamentEvaluator` stays `None`, and `_execute_tournament` will raise `ImportError` at line 232. This is a correct guard for optional dependencies, but the error message is generic ("TournamentEvaluator not available") and does not include the original import error.

**Lines 162-167: `num_games_per_opponent` calculation.** Uses `getattr(tournament_config, "num_games_per_opponent", None)` with a fallback to `2`. The `tournament_config` parameter is untyped (`Any` by convention), so if it has the attribute set to `0`, the calculation produces `total_games = 0`. This is a valid edge case but might confuse the progress tracker (`completion_percentage` returns `0.0` for `total_games == 0`).

**Line 224: `await asyncio.sleep(0.15)`.** A 150ms sleep is explicitly added inside `_execute_tournament` "to ensure tournament is in RUNNING state for tests." This is a test accommodation baked into production code. It delays every tournament start by 150ms regardless of context.

**Lines 300: `games_per_opponent` extraction.** `getattr(context.configuration, "num_games_per_opponent", 2)` is used, but line 162-166 in `start_tournament` does the same extraction. If `tournament_config` and `context.configuration` differ (they are different objects -- `tournament_config` is the argument, `context.configuration` is set at line 242), the game counts could be inconsistent.

**Lines 313: `await evaluator._play_games_against_opponent(...)`.** This calls a **private** method on `TournamentEvaluator`. This tightly couples `BackgroundTournamentManager` to the internal implementation of `TournamentEvaluator`. If that private method is renamed or its signature changes, this code breaks silently (at runtime only).

**Lines 359: `evaluator._calculate_tournament_standings(...)`.** Same concern -- calling a private method directly. Two private API dependencies on the strategy implementation.

**Lines 190-200: Task done callback.** The `task_done_callback` captures `tournament_id` and `progress` variables from the enclosing scope. However, `progress` (line 197) is re-fetched from `self._active_tournaments[tournament_id]`, which is correct. But if `_cleanup_tournament` has already removed the tournament from the dict (line 472-473 removes the task, lines 485-486 remove the lock), the tournament ID may still be in `_active_tournaments` since `_cleanup_tournament` does not remove it. The `_active_tournaments` entry is never cleaned up, which is an intentional design choice for history access via `list_all_tournaments()`, but means memory grows unboundedly over the manager's lifetime.

**Lines 469-486: `_cleanup_tournament`.** Removes the task and the lock, but does NOT remove the tournament from `_active_tournaments`. This is intentional (to preserve history), but `_tournament_locks` is removed, so subsequent `_update_progress` calls for that tournament ID would raise `KeyError` at line 381 when trying `async with self._tournament_locks[tournament_id]`.

**Lines 506-520: `cancel_tournament`.** Calls `task.cancel()` and then `_update_progress`. If the cancellation triggers `_execute_tournament`'s `CancelledError` handler (lines 269-273), that handler also calls `_update_progress` with `CANCELLED`. This could result in a race: `cancel_tournament` updates to `CANCELLED` at line 514-515, then the task's `CancelledError` handler also updates to `CANCELLED` at line 270-271. The double-update is benign (same status) but the lock access in the `CancelledError` handler may fail if `_cleanup_tournament` (called in `finally` at line 285) has already deleted the lock.

**Lines 533-535: `shutdown` gathers tasks.** `asyncio.gather(*self._tournament_tasks.values(), return_exceptions=True)` is called after `cancel_tournament` has already been called on each. By the time `gather` runs, some tasks may have been removed from `_tournament_tasks` by `_cleanup_tournament`. The `list(self._tournament_tasks.keys())` at line 528 takes a snapshot, but the actual gathering at line 533 reads `.values()` again, which may have changed.

---

## 4. Robustness & Error Handling

**Lines 269-273: `CancelledError` handling.** Correctly re-raises after updating status, which is the proper pattern for asyncio cancellation.

**Lines 276-281: General exception handling in `_execute_tournament`.** Catches all exceptions, logs with traceback (`exc_info=True`), and updates status to `FAILED`. Does NOT re-raise, so the task completes silently on failure. This is appropriate for background tasks.

**Lines 390-397: Progress callback error handling.** Wraps both sync and async callbacks in try/except, logging errors without propagating. Correctly uses `asyncio.iscoroutinefunction` to distinguish callback types.

**Lines 399-467: `_save_tournament_results` robustness.** Very defensive serialization with multiple try/except blocks for `context.to_dict()` and `summary_stats`. Uses `default=str` in `json.dump` as a fallback for non-serializable objects. Initializes `result_data = {}` before the outer try to avoid `UnboundLocalError` in the except handler at line 467.

**Lines 474-483: `_cleanup_tournament` RuntimeError handling.** Catches `RuntimeError` for "Event loop is closed" specifically, which handles cleanup during test teardown. Other `RuntimeError` variants are re-raised.

---

## 5. Performance & Scalability

**Semaphore-based concurrency (line 126-127, 217):** `asyncio.Semaphore(max_concurrent_tournaments)` bounds how many tournaments execute simultaneously. Default is 2. This is appropriate for resource management.

**Line 224: 150ms sleep per tournament start.** A fixed delay added for test reliability. In production with frequent tournament starts, this adds unnecessary latency.

**`_active_tournaments` is never pruned (see correctness note).** Over the lifetime of the manager, all tournament history accumulates in memory. For long-running training processes that trigger many evaluations, this could grow to contain thousands of entries with full `GameResult` lists (line 254: `progress.results = result.games`).

**JSON serialization at lines 457-459:** Each completed tournament writes results to disk synchronously (within an async context but using synchronous `open()/json.dump()`). For large tournaments, this blocks the event loop. Should ideally use `aiofiles` or run in an executor.

---

## 6. Security & Safety

**Line 118: Default result storage directory is `./tournament_results`.** Created with `mkdir(parents=True, exist_ok=True)`. If the current working directory is sensitive, tournament results (which contain agent names, checkpoint paths, and game data) are written to a predictable location. This is low risk since the system is not network-facing.

**Line 457: File write uses `open(result_file, "w")`.** The filename includes the tournament ID, which may contain user-provided `tournament_name` (line 159: `f"{tournament_name}_{tournament_id[:8]}"`). If `tournament_name` contains path separators or special characters, the file could be written to an unexpected location. The Path API used here would handle most edge cases, but no explicit sanitization is performed.

---

## 7. Maintainability

**Tight coupling to `TournamentEvaluator` internals.** Lines 313 and 359 call private methods `_play_games_against_opponent` and `_calculate_tournament_standings`. This creates a fragile dependency on implementation details of another module.

**`tournament_config` is untyped** (lines 137, 207, 234). It is passed as a generic parameter with no type annotation, relying on duck typing for `num_games_per_opponent` attribute access.

**`TournamentStatus.PAUSED` is defined** but never set by any code path in this module. No method exists to pause a tournament; `is_active` returns True for PAUSED status, but there is no way to reach this state.

**Unused imports:** `Set` (line 18), `defaultdict` (line 13), and `Union` (line 18) are imported but never used.

**Line 224 sleep comment:** The hardcoded `asyncio.sleep(0.15)` is explicitly documented as test infrastructure, but there is no mechanism to disable it in production.

---

## 8. Verdict

**NEEDS_ATTENTION**

The module provides a functional background tournament system with good error handling and defensive serialization. The primary concerns are:

1. **Private method coupling** -- `_play_games_against_opponent` and `_calculate_tournament_standings` are called on `TournamentEvaluator`, creating a brittle cross-module dependency.
2. **Lock deletion race** -- `_cleanup_tournament` deletes locks from `_tournament_locks` while `_update_progress` may still attempt to acquire them, risking `KeyError` during cancellation sequences.
3. **Unbounded memory growth** -- `_active_tournaments` and `progress.results` are never pruned, accumulating all tournament history including full game result lists.
4. **Hardcoded 150ms sleep** -- test infrastructure leaking into production code.
5. **Synchronous file I/O** in async context (`json.dump` at line 458) blocks the event loop.
