# Code Analysis: keisei/utils/profiling.py

## 1. Purpose & Role

This module provides lightweight development profiling utilities for timing operations, counting events, and collecting cProfile statistics. It offers a `PerformanceMonitor` class for aggregating timing data across operations, several decorator-based profiling tools, and a `memory_usage_mb` function. A global singleton (`perf_monitor`) is provided for convenient cross-module usage without explicit dependency injection.

## 2. Interface Contracts

### Classes

- **`_FastTimerContext`** (lines 22-41): Internal context manager implementing a low-overhead timer. Uses `time.perf_counter()` and directly appends to the monitor's `timings` dict. Not intended for external use (name-mangled with underscore prefix).

- **`PerformanceMonitor`** (lines 44-153): Main profiling class.
  - `time_operation(operation_name)` -> context manager (line 51): Returns a `_FastTimerContext` for timing a code block.
  - `increment_counter(counter_name, value=1)` (lines 80-84): Increments a named counter.
  - `get_stats()` -> `Dict[str, Any]` (lines 86-102): Returns aggregated statistics (avg, min, max, total, count per operation, plus all counters).
  - `reset()` (lines 104-107): Clears all collected data.
  - `print_summary()` (lines 109-152): Prints a formatted summary to stdout/stderr.

### Module-Level Functions

- **`profile_function(func)`** -> decorated callable (lines 159-168): Decorator that times function execution using the global `perf_monitor`.
- **`profile_code_block(description)`** -> context manager (lines 171-175): Context manager for timing arbitrary code blocks.
- **`run_profiler(func, *args, **kwargs)`** -> `tuple(result, profile_stats_string)` (lines 178-201): Runs a function under `cProfile` and returns the cumulative-sorted stats as a string.
- **`profile_training_step(step_func)`** -> decorated callable (lines 204-224): Decorator for training step functions; times execution and increments a step counter.
- **`profile_game_operation(operation_name)`** -> decorator factory (lines 227-248): Parameterized decorator for game operations; times execution and increments a per-operation counter.
- **`memory_usage_mb()`** -> `float` (lines 251-275): Returns current process memory usage in MB using `psutil` with a `tracemalloc` fallback.

### Global State

- **`perf_monitor`** (line 156): Module-level singleton `PerformanceMonitor` instance used by all decorators and convenience functions.

### Dependencies

- `cProfile`, `functools`, `io`, `logging`, `pstats`, `time`, `contextlib`
- `keisei.utils.unified_logger.log_info_to_stderr`
- Optional: `psutil` (with `tracemalloc` fallback)

## 3. Correctness Analysis

### `_FastTimerContext` vs `_timer_context` Discrepancy (lines 22-41 vs 59-78)

Two timer implementations exist: `_FastTimerContext` (class-based, used by `time_operation`) and `_timer_context` (generator-based contextmanager, unused). The `_timer_context` method includes debug logging (lines 75-78) that `_FastTimerContext` does not. The `_timer_context` method is dead code -- it is defined on the class but `time_operation` on line 53 calls `_timer_context_fast` instead. This means debug logging for individual timing operations is silently disabled.

### `print_summary` Uses `print()` Directly (lines 126-152)

The `print_summary` method uses `print()` for most output (lines 126-134, 149-150, 152) while using `log_info_to_stderr` only for the header (line 113). This is inconsistent with the project's stated practice of using the unified logger. The `print()` calls will go to stdout while the header goes to stderr.

### `get_stats` Key Collision Risk (lines 86-102)

The `get_stats` method creates keys like `"{operation}_avg"`, `"{operation}_count"` etc. for timing data, then merges counter data on line 100 with `stats.update(self.counters)`. If a counter name matches a timing-derived key (e.g., a counter named `"my_op_avg"`), it would overwrite the timing statistic. This is unlikely in practice but is a latent collision.

### `print_summary` Operation Name Extraction (line 123)

The operation name extraction `op_name = key.rsplit("_", 1)[0]` splits on the last underscore. This works correctly for suffixes like `_avg`, `_min`, `_max`, `_total`, `_count`. However, if an operation name itself contains underscores (e.g., `"training_step_forward"`), the rsplit correctly preserves the full operation name because it splits from the right on only the last underscore.

### `memory_usage_mb` Tracemalloc Side Effect (lines 266-275)

When `psutil` is not available, the fallback calls `tracemalloc.start()` if tracing is not active (line 270). This has a permanent side effect on the process -- `tracemalloc` remains active after the call, adding overhead to all subsequent memory allocations. The function does not stop tracing after measurement.

### `memory_usage_mb` Minimum Return Value (line 275)

The `max(1.0, ...)` on line 275 ensures at least 1MB is reported. This means zero or near-zero memory usage (theoretically possible for very small processes) would be reported as 1MB. The comment explains this guards against the case where `tracemalloc` returns 0 immediately after starting.

## 4. Robustness & Error Handling

**Strengths:**
- `_FastTimerContext.__exit__` does not suppress exceptions (returns `None`/falsy), so errors in timed code blocks propagate correctly.
- `run_profiler` (lines 185-201) uses `try/finally` to ensure the profiler is disabled even if the function raises.
- `memory_usage_mb` handles the `psutil` import failure gracefully with a tracemalloc fallback.

**Weaknesses:**
- `memory_usage_mb` does not handle the case where `tracemalloc` itself is unavailable (though it is a stdlib module, so this is theoretical).
- No error handling in `print_summary` -- if the stats dict is malformed (e.g., a timing list is empty despite being present), `min()` or `max()` would raise on empty sequences. However, the guard on line 92 (`if times:`) in `get_stats` prevents empty lists from generating stats entries.

## 5. Performance & Scalability

- `_FastTimerContext` is designed for minimal overhead: it uses direct attribute access and list append without logging on the hot path. This is appropriate for timing operations that run thousands of times during training.
- The unused `_timer_context` includes a `logger.isEnabledFor(logging.DEBUG)` guard (line 75) that avoids string formatting overhead when debug is disabled.
- Timing data is stored as unbounded lists (line 41: `timings[self.operation_name].append(duration)`). For long training runs with millions of steps, this could consume significant memory. There is no windowing, sampling, or automatic aggregation.
- `get_stats` computes `sum(times)` and `len(times)` on every call, which is O(n) in the number of recorded timings. For very large timing lists, this could become noticeable.

### Memory Growth Concern

For a training run of 1 million timesteps with 5 timed operations per step, this would accumulate 5 million float entries (approximately 40MB of list overhead). The `reset()` method exists but must be called explicitly.

## 6. Security & Safety

- No file I/O, network access, or subprocess execution.
- The `os.getpid()` call on line 263 is used safely within `psutil.Process()`.
- The global `perf_monitor` singleton is not thread-safe. Concurrent access from multiple threads could produce corrupted timing lists. However, Python's GIL provides some protection for simple list.append operations.

## 7. Maintainability

**Strengths:**
- Clean separation between the core `PerformanceMonitor` class and the convenience decorators/functions.
- The `_FastTimerContext` class provides a well-optimized timing mechanism.
- The `example_usage` function (lines 279-315) serves as inline documentation, though it uses `time.sleep()` which makes it unsuitable for automated testing.
- The `__main__` guard (lines 318-321) is present but the example is commented out to avoid import delays.

**Weaknesses:**
- Dead code: `_timer_context` method (lines 59-78) is never called.
- Inconsistent output: `print_summary` mixes `print()` and `log_info_to_stderr`.
- The global singleton pattern makes testing difficult since state persists between tests unless `reset()` is called.
- No type annotations on `_FastTimerContext.__init__` parameters (lines 25-28).
- The unbounded timing list growth is not documented as a caveat.

## 8. Verdict

**NEEDS_ATTENTION**

The core timing mechanism is well-designed with the fast-path `_FastTimerContext` class. The module provides a useful set of profiling tools for development. The main concerns are: (1) unbounded memory growth from accumulated timing lists during long training runs, (2) the `tracemalloc.start()` side effect in `memory_usage_mb`, (3) dead code in `_timer_context`, and (4) inconsistent use of `print()` vs the unified logger. None are critical for correctness, but the memory growth issue could become problematic in production training scenarios.
