"""
Development profiling helpers for performance monitoring.

This module provides utilities for profiling and monitoring performance
during development and debugging.
"""

import cProfile
import functools
import io
import logging
import pstats
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict

from keisei.utils.unified_logger import log_info_to_stderr

logger = logging.getLogger(__name__)


class _FastTimerContext:
    """Fast timer context with minimal overhead."""

    def __init__(self, monitor, operation_name):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time

        # Fast path: pre-initialize if needed and append
        timings = self.monitor.timings
        if self.operation_name not in timings:
            timings[self.operation_name] = []
        timings[self.operation_name].append(duration)


class PerformanceMonitor:
    """Simple performance monitoring for development."""

    def __init__(self):
        self.timings: Dict[str, list] = {}
        self.counters: Dict[str, int] = {}

    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return self._timer_context_fast(operation_name)

    def _timer_context_fast(self, operation_name: str):
        """Fast timer context manager with minimal overhead."""
        return _FastTimerContext(self, operation_name)

    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a named counter."""
        if counter_name not in self.counters:
            self.counters[counter_name] = 0
        self.counters[counter_name] += value

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}

        # Timing statistics
        for operation, times in self.timings.items():
            if times:
                stats[f"{operation}_avg"] = sum(times) / len(times)
                stats[f"{operation}_min"] = min(times)
                stats[f"{operation}_max"] = max(times)
                stats[f"{operation}_total"] = sum(times)
                stats[f"{operation}_count"] = len(times)

        # Counter statistics
        stats.update(self.counters)

        return stats

    def reset(self):
        """Reset all collected statistics."""
        self.timings.clear()
        self.counters.clear()

    def print_summary(self):
        """Print a summary of performance statistics."""
        stats = self.get_stats()

        log_info_to_stderr("Profiling", "\n=== Performance Summary ===")

        # Group by operation type
        timing_ops = set()
        for key in stats:
            if any(
                key.endswith(suffix)
                for suffix in ["_avg", "_min", "_max", "_total", "_count"]
            ):
                op_name = key.rsplit("_", 1)[0]
                timing_ops.add(op_name)

        if timing_ops:
            print("\nTiming Operations:")
            for op in sorted(timing_ops):
                if f"{op}_count" in stats:
                    print(f"  {op}:")
                    print(f"    Count: {stats[f'{op}_count']}")
                    print(f"    Average: {stats[f'{op}_avg']:.4f}s")
                    print(f"    Min: {stats[f'{op}_min']:.4f}s")
                    print(f"    Max: {stats[f'{op}_max']:.4f}s")
                    print(f"    Total: {stats[f'{op}_total']:.4f}s")

        # Show counters
        counters = {
            k: v
            for k, v in stats.items()
            if k
            not in {
                f"{op}_{suffix}"
                for op in timing_ops
                for suffix in ["avg", "min", "max", "total", "count"]
            }
        }
        if counters:
            print("\nCounters:")
            for name, value in sorted(counters.items()):
                print(f"  {name}: {value}")

        print("=" * 27)


# Global performance monitor instance
perf_monitor = PerformanceMonitor()


def profile_function(func: Callable) -> Callable:
    """Decorator to profile a function's execution time."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        operation_name = f"{func.__module__}.{func.__name__}"
        with perf_monitor.time_operation(operation_name):
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def profile_code_block(description: str):
    """Context manager to profile a code block."""
    with perf_monitor.time_operation(description):
        yield


def run_profiler(func: Callable, *args, **kwargs) -> tuple:
    """
    Run a function with cProfile and return results.

    Returns:
        tuple: (function_result, profile_stats_string)
    """
    profiler = cProfile.Profile()

    # Run the function with profiling
    profiler.enable()
    try:
        result = func(*args, **kwargs)
    finally:
        profiler.disable()

    # Get profile statistics
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats("cumulative")
    ps.print_stats()

    return result, s.getvalue()


def profile_training_step(step_func: Callable):
    """
    Decorator specifically for profiling training steps.

    This is designed to be used with training loop functions to
    monitor performance during model training.
    """

    @functools.wraps(step_func)
    def wrapper(*args, **kwargs):
        step_name = f"training_step_{step_func.__name__}"

        with perf_monitor.time_operation(step_name):
            result = step_func(*args, **kwargs)

        # Increment step counter
        perf_monitor.increment_counter("training_steps_completed")

        return result

    return wrapper


def profile_game_operation(operation_name: str):
    """
    Decorator for profiling game operations (moves, evaluations, etc.).

    Args:
        operation_name: Name to use for the operation in profiling results
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with perf_monitor.time_operation(operation_name):
                result = func(*args, **kwargs)

            # Increment operation counter
            perf_monitor.increment_counter(f"{operation_name}_count")

            return result

        return wrapper

    return decorator


def memory_usage_mb() -> float:
    """
    Get current memory usage in MB.

    Note: Requires psutil package for more accurate measurements.
    This is a simple version using basic Python tools.
    """
    try:
        import os

        import psutil  # type: ignore[import]

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # Fallback to tracemalloc if psutil not available
        import tracemalloc

        if not tracemalloc.is_tracing():
            tracemalloc.start()

        current, peak = tracemalloc.get_traced_memory()
        # Use peak memory if current is 0 (which can happen immediately after starting)
        memory_bytes = current if current > 0 else peak
        return max(1.0, memory_bytes / 1024 / 1024)  # Ensure at least 1MB is reported


# Example usage functions for documentation
def example_usage():
    """
    Example usage of profiling utilities.

    This function demonstrates how to use the profiling tools
    during development.
    """

    # Using the performance monitor
    with perf_monitor.time_operation("example_operation"):
        time.sleep(0.1)  # Simulate work

    # Using decorators
    @profile_function
    def example_function():
        time.sleep(0.05)
        return "result"

    @profile_game_operation("move_generation")
    def generate_moves():
        time.sleep(0.02)
        return ["move1", "move2"]

    # Run examples
    example_function()
    generate_moves()

    # Print summary
    perf_monitor.print_summary()

    # Run with cProfile
    def heavy_computation():
        return sum(i**2 for i in range(10000))

    _, profile_output = run_profiler(heavy_computation)
    print("\nProfile output:")
    print(profile_output[:500] + "..." if len(profile_output) > 500 else profile_output)


if __name__ == "__main__":
    # Run example if script is executed directly
    print("Profiling utilities module loaded successfully")
    # example_usage()  # Disabled to prevent import delays
