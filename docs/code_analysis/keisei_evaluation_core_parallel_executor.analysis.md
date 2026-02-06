# Code Analysis: `keisei/evaluation/core/parallel_executor.py`

**Lines:** 402 (including blanks and docstrings)
**Last read:** 2026-02-07

---

## 1. Purpose & Role

This module provides two executor classes (`ParallelGameExecutor` and `BatchGameExecutor`) for running multiple evaluation games concurrently, along with a `ParallelGameTask` data class for task representation and a `create_parallel_game_tasks` factory function. `ParallelGameExecutor` uses a `ThreadPoolExecutor` for thread-based parallelism with timeout control, while `BatchGameExecutor` partitions work into batches and delegates each batch to a `ParallelGameExecutor`.

---

## 2. Interface Contracts

### `ParallelGameTask`
A plain data class holding: `task_id`, `agent_info`, `opponent_info`, `context`, `game_executor` (a callable), `metadata`, timing fields (`start_time`, `end_time`), and `result`/`error` fields. Populated during execution.

### `ParallelGameExecutor`

| Method | Contract |
|---|---|
| `__init__(max_concurrent_games, max_memory_usage_mb, timeout_per_game_seconds)` | Stores configuration; no executor created yet. |
| `__enter__` / `__exit__` | Context manager that creates/shuts down `ThreadPoolExecutor`. |
| `execute_games_parallel(tasks, progress_callback)` | **Async** method. Submits all tasks to thread pool, collects results. Returns `(List[GameResult], List[str])`. Raises `RuntimeError` if not used as context manager. |
| `get_execution_stats()` | Returns dict with task counts, success rate, average duration. |

Internal:
- `_execute_single_game(task)` -- Synchronous; runs in thread pool. Handles asyncio event loop detection.
- `_start_memory_monitoring()` / `_stop_memory_monitoring()` -- Stubs; set a boolean flag only.

### `BatchGameExecutor`

| Method | Contract |
|---|---|
| `__init__(batch_size, max_concurrent_games, timeout_per_batch_seconds)` | Stores config. Note: `timeout_per_batch_seconds` is accepted but never used. |
| `execute_games_in_batches(tasks, progress_callback)` | **Async**. Splits tasks into batches, delegates each batch to a new `ParallelGameExecutor`. Returns `(List[GameResult], List[str])`. |

### `create_parallel_game_tasks` (module-level function)
Creates `ParallelGameTask` objects for all opponents and games. Alternates sente/gote assignment. Returns `List[ParallelGameTask]`.

---

## 3. Correctness Analysis

**Lines 156-221: `_execute_single_game` -- asyncio/thread interaction.** This method runs in a `ThreadPoolExecutor` thread (submitted at line 108) but the `game_executor` callback is expected to be an async function (it is `await`-ed at lines 177-189 and 198-211). The code attempts to detect whether an event loop is running:

- **Lines 174-183 (in-memory path):** Calls `asyncio.get_running_loop()` -- if it succeeds, uses `run_coroutine_threadsafe` to schedule the coroutine on that loop and waits for the result with a 300-second timeout. If `RuntimeError` is raised (no running loop), falls back to `asyncio.run()`.
- **Lines 194-211 (regular path):** Same pattern.

**Critical issue:** `asyncio.get_running_loop()` raises `RuntimeError` when there is no running loop in the current thread. Since this code runs in a `ThreadPoolExecutor` thread, there is typically no event loop running in that thread. However, the code at line 175 catches `RuntimeError` at line 184 and falls back to `asyncio.run()`. The problem is that `asyncio.run()` creates a NEW event loop in the thread pool thread. If the `game_executor` coroutine accesses any asyncio primitives (locks, events, queues) that are bound to the MAIN event loop, this will produce `RuntimeError` ("attached to a different loop") or silent deadlocks. The `run_coroutine_threadsafe` path (line 177-183) is the correct approach when a loop exists, but it requires the loop reference to be the correct one -- `asyncio.get_running_loop()` gets the loop of the CURRENT thread, which in a thread pool thread is typically None (triggering the fallback).

**In practice:** If `execute_games_parallel` is called from an async context (as its `async def` signature suggests), the calling thread has a running event loop. But `_execute_single_game` runs in a different thread (the pool thread), where `get_running_loop()` will raise `RuntimeError`. So the code will almost always fall back to `asyncio.run()` (line 186/208), creating isolated event loops per game. This works if the game executors are self-contained, but is fragile if they share any asyncio state with the main loop.

**Lines 114-117: Global timeout for `as_completed`.** The timeout is `self.timeout_per_game_seconds * len(tasks)`. This is a total timeout for ALL games, not per-game. If there are 100 tasks with 300s per game, the total timeout is 30,000 seconds (8.3 hours). This provides a very loose upper bound. If a single game hangs, it blocks `as_completed` iteration until the global timeout expires, preventing progress callbacks for subsequent completed games.

**Lines 91-92: Context manager enforcement.** `execute_games_parallel` checks `if not self._executor` and raises `RuntimeError`. This is correct but note that `BatchGameExecutor.execute_games_in_batches` at lines 314-317 properly uses `with ParallelGameExecutor(...) as executor:`.

**Lines 243-247: `get_execution_stats` average duration calculation.** The sum includes a guard `if task.start_time and task.end_time` inside the generator, but divides by `len(self.completed_tasks)` (line 248) which includes ALL completed tasks, even those without timing data. If some tasks have `start_time=None` or `end_time=None`, the average will be underestimated. In practice, `_execute_single_game` always sets both times (line 158 and line 224), so this is a theoretical issue.

**Lines 366-401: `create_parallel_game_tasks` -- sente alternation logic.** Line 370: `agent_plays_sente = game_idx < (games_per_opponent + 1) // 2`. For `games_per_opponent=2`, this gives sente for game 0 (0 < 1 = True) and gote for game 1 (1 < 1 = False). For `games_per_opponent=3`, sente for games 0,1 (0 < 2, 1 < 2) and gote for game 2. For odd numbers, the agent gets one extra sente game, which is a slight asymmetry but is standard practice.

---

## 4. Robustness & Error Handling

**Lines 104-153: `execute_games_parallel` try/finally.** Memory monitoring is always stopped in the finally block. Individual task failures are caught at line 133 and appended to `error_messages`. Task cleanup (removing from `active_tasks`) happens in a `finally` block at line 140-142. This is robust.

**Lines 216-221: `_execute_single_game` outer exception handler.** Catches all exceptions, sets `task.error`, logs with traceback, returns `None`. The caller at line 122-131 handles `None` results by adding to `error_messages`. This double-layer error handling is thorough.

**Lines 91-92: Pre-condition check.** Raises `RuntimeError` if executor is not initialized. Clear error message.

**No timeout per individual game in thread pool.** While `timeout_per_game_seconds` is accepted in the constructor (line 56), individual game execution in `_execute_single_game` uses hardcoded 300-second timeouts (lines 183, 204) for `run_coroutine_threadsafe`, and no timeout at all for `asyncio.run()` (lines 186-190, 206-211). A game that hangs in `asyncio.run()` will block the thread indefinitely, consuming one of the limited thread pool slots.

---

## 5. Performance & Scalability

**Thread-based parallelism.** `ThreadPoolExecutor` with `max_concurrent_games` workers (default 4). Since Shogi game evaluation likely involves PyTorch inference (CPU-bound with the GIL, or GPU-bound), thread-based parallelism may not provide true CPU parallelism due to the GIL. However, if the game executor involves I/O (network, disk) or if PyTorch releases the GIL during tensor operations, threads can be effective.

**Lines 226-236: Memory monitoring is a no-op.** `_start_memory_monitoring` and `_stop_memory_monitoring` only set a boolean flag. The `max_memory_usage_mb` parameter (line 55) is accepted but never enforced. This is dead configuration.

**Lines 272-273: `BatchGameExecutor` uses `timeout_per_batch_seconds` in constructor** but never references it during execution (lines 278-342). The per-game timeout is hardcoded to 300 seconds at line 316.

**Lines 314-317: New `ParallelGameExecutor` per batch.** Each batch creates a new `ThreadPoolExecutor`, which means thread pools are created and destroyed for every batch. For many small batches, this adds overhead.

**Lines 106-109: All tasks submitted eagerly to thread pool.** For a large task list, all futures are created at once. The `ThreadPoolExecutor` internally queues them, so only `max_workers` run concurrently. This is standard and acceptable.

---

## 6. Security & Safety

**Line 39: `game_executor` is a callable stored on the task.** It is invoked at lines 178, 187, 199, 208 without any sandboxing. If a malicious callable were injected, it would run with full process privileges in a thread pool thread. This is within the expected trust boundary since game executors are constructed by the evaluation framework itself.

**No file system access.** This module does not read or write files directly.

---

## 7. Maintainability

**Duplicated asyncio event loop detection logic.** Lines 173-190 (in-memory path) and lines 193-211 (regular path) are nearly identical code blocks. The only difference is which coroutine is awaited. This duplication increases maintenance burden.

**Dead configuration parameters:**
- `max_memory_usage_mb` (line 55) -- accepted but never used.
- `timeout_per_batch_seconds` (line 273) -- accepted but never used.
- `_memory_monitor_active` (line 65) -- set but never read.

**Hardcoded timeouts.** 300-second timeouts appear at lines 183 and 204, independent of the configurable `timeout_per_game_seconds`. This means the constructor parameter is partially ignored.

**Unused imports:** `uuid` (line 9) is imported at the module level and used only in `create_parallel_game_tasks` (line 384). `Union` (line 11) is imported but never used. `Tuple` is imported and used.

**`execute_games_parallel` is `async def` but uses synchronous `ThreadPoolExecutor`.** The method submits work to threads and then uses synchronous `as_completed` to iterate results. This means the event loop is blocked during the `for future in as_completed(...)` loop (lines 114-146). If this method is `await`-ed from an asyncio event loop, it will block the entire loop for the duration of all game executions. This defeats the purpose of being an async method.

---

## 8. Verdict

**NEEDS_ATTENTION**

The module provides functional parallel game execution but has several significant concerns:

1. **Blocking `async def`** -- `execute_games_parallel` is declared `async` but blocks the event loop with synchronous `as_completed` iteration and synchronous `ThreadPoolExecutor` submission. This is an architectural mismatch that could cause the calling event loop to freeze.
2. **`asyncio.run()` in thread pool threads** -- The fallback at lines 186/208 creates new event loops in pool threads, which will fail if game executors share asyncio primitives with the main loop.
3. **Dead configuration** -- `max_memory_usage_mb` and `timeout_per_batch_seconds` are accepted but never enforced, creating a false sense of configuration control.
4. **Hardcoded 300s timeout** -- overrides the configurable `timeout_per_game_seconds` in the critical code path.
5. **Duplicated event loop detection** -- identical logic blocks at lines 173-190 and 193-211.

The blocking-async mismatch (item 1) is the most significant issue as it can cause the entire training loop's event loop to stall during evaluation if called from an async context.
