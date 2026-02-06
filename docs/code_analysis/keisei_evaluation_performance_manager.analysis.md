# Code Analysis: keisei/evaluation/performance_manager.py

**Lines:** 315 (314 non-blank)
**Module:** `keisei.evaluation.performance_manager`

---

## 1. Purpose & Role

This module provides performance safeguards, resource monitoring, and SLA enforcement for the evaluation subsystem. It contains five classes: `PerformanceMetrics` (a data container), `ResourceMonitor` (system resource sampling via `psutil`), `EvaluationPerformanceSLA` (threshold-based SLA validation), `EvaluationPerformanceManager` (the main orchestrator combining concurrency limits, timeouts, and resource enforcement), and `PerformanceGuard` (an async context manager for evaluation scoping). It also defines two custom exception types (`EvaluationTimeoutError`, `EvaluationResourceError`) and a `MockEvaluator` adapter class.

This module is imported and instantiated by `core_manager.py` (line 24-25, line 55-58).

---

## 2. Interface Contracts

### `PerformanceMetrics` (dataclass, lines 16-36)
- Fields: `evaluation_latency_ms`, `memory_overhead_mb`, `gpu_utilization_percent` (Optional[float]), `cpu_utilization_percent`, `start_time`, `end_time`.
- `to_dict()`: Returns serializable dict with ISO-format timestamps.

### `ResourceMonitor` (lines 39-86)
- `start_monitoring()`: Records baseline memory and CPU.
- `get_memory_usage()`: Returns RSS in bytes.
- `get_cpu_percent()`: Returns CPU usage percentage.
- `get_gpu_utilization()`: Returns GPU utilization percentage or `None` if unavailable.
- `get_memory_overhead()`: Returns memory delta in MB since monitoring started.

### `EvaluationPerformanceSLA` (lines 89-128)
- `SLA_METRICS`: Class-level dict with thresholds (5000ms latency, 500MB memory, 5% training impact, 80% GPU).
- `validate_performance_sla(metrics)`: Returns `True` if all metrics within thresholds. Skips `None` values.
- `log_performance_metrics(metrics)`: Logs and validates.

### `EvaluationPerformanceManager` (lines 130-253)
- `__init__(max_concurrent, timeout_seconds)`: Creates `asyncio.Semaphore` and `EvaluationPerformanceSLA`.
- `run_evaluation_with_safeguards(evaluator, agent_info, context)`: Primary method. Enforces concurrency, timeout, and resource limits.
- `run_evaluation_with_monitoring(evaluation_func, *args, **kwargs)`: Alternative entry point using `MockEvaluator` wrapper.
- `enable_enforcement()` / `disable_enforcement()`: Toggle resource limit enforcement.

### `MockEvaluator` (lines 255-270)
- Wraps a callable to match the `evaluate(agent_info, context)` interface.

### `PerformanceGuard` (lines 285-314)
- Async context manager that starts monitoring on enter and logs metrics on exit.

### Custom exceptions (lines 273-282)
- `EvaluationTimeoutError`: Raised when evaluation exceeds timeout.
- `EvaluationResourceError`: Raised when memory limits are exceeded.

---

## 3. Correctness Analysis

### Bug: `MockEvaluator.evaluate` has incompatible signature (lines 261-270)

The `MockEvaluator` is used by `run_evaluation_with_monitoring` at line 241:
```python
return await self.run_evaluation_with_safeguards(
    MockEvaluator(evaluation_func), args, kwargs
)
```

Here, `args` is a tuple and `kwargs` is a dict. `run_evaluation_with_safeguards` then calls:
```python
result = await evaluator.evaluate(agent_info, context)
```
where `agent_info=args` (a tuple) and `context=kwargs` (a dict).

Inside `MockEvaluator.evaluate` (line 261):
```python
async def evaluate(self, args, kwargs):
    if asyncio.iscoroutinefunction(self.evaluation_func):
        return await self.evaluation_func(*args, **kwargs)
```

The parameter names `args` and `kwargs` shadow the method's intent. The method expects to receive the original tuple and dict, and it destructures them. However, `run_evaluation_with_safeguards` passes `agent_info` and `context` as the two positional arguments. So `args` receives the original `args` tuple and `kwargs` receives the original `kwargs` dict, and they are correctly destructured with `*args, **kwargs`.

While this works mechanically, the method signature `evaluate(self, args, kwargs)` does not match the `BaseEvaluator.evaluate(self, agent_info, context)` protocol. If any code checks for this protocol via `isinstance` or type checking, it would fail. This is a correctness concern if the interface is formalized.

### Bug: `run_in_executor` argument passing (lines 268-269)
```python
return await loop.run_in_executor(
    None, self.evaluation_func, *args, **kwargs
)
```
`loop.run_in_executor` takes `(executor, func, *args)` -- it does NOT support `**kwargs`. Passing `**kwargs` here would raise a `TypeError`. This means `run_evaluation_with_monitoring` is broken for synchronous functions that require keyword arguments.

### Bug: `asyncio.get_event_loop()` deprecation (line 267)
```python
loop = asyncio.get_event_loop()
```
In Python 3.10+, `asyncio.get_event_loop()` emits a `DeprecationWarning` when there is no running event loop. Since this code is already inside an async method (called from `run_evaluation_with_safeguards` which is async), `asyncio.get_running_loop()` should be used instead.

### Concern: Resource enforcement happens AFTER evaluation completes (lines 215-216)
```python
# CRITICAL FIX: Actually enforce resource limits
self.enforce_resource_limits(metrics)
```
The `enforce_resource_limits` method raises `EvaluationResourceError` if memory exceeds the limit. However, this check happens **after** the evaluation has already completed (line 192 `result = await asyncio.wait_for(...)`). The resource limits are reactive, not preventive. If an evaluation allocates 2GB of memory, the code runs to completion and then raises an error. The memory has already been consumed and potentially caused OOM issues before the check triggers.

### Concern: SLA validated twice (lines 168, 219)
In `run_evaluation_with_safeguards`, `enforce_resource_limits` calls `self.sla_monitor.validate_performance_sla(metrics.to_dict())` at line 168, and then line 219 calls `self.sla_monitor.log_performance_metrics(metrics)` which internally calls `validate_performance_sla` again (line 124). The SLA is validated twice per evaluation, producing duplicate log messages for violations.

### Concern: `initial_memory` and `initial_cpu` captured but unused (lines 186-187)
```python
initial_memory = self.resource_monitor.get_memory_usage()
initial_cpu = self.resource_monitor.get_cpu_percent()
```
These local variables are captured but never used. The memory overhead is obtained from `self.resource_monitor.get_memory_overhead()` at line 198, which uses the state set by `start_monitoring()` at line 182.

### Concern: CPU percent measurement timing (line 187, 199)
`psutil`'s `cpu_percent()` returns the CPU usage since the last call to `cpu_percent()`. The first call after `start_monitoring()` (line 50) returns the baseline. The call at line 199 returns usage since line 187. However, `start_monitoring()` at line 182 also calls `cpu_percent()` (line 50), so the call at line 187 measures the tiny interval between lines 182 and 187, which will be near-zero and meaningless.

### Memory overhead threshold duplicated (lines 138, 222)
`self.max_memory_mb = 500` at line 138, and the hardcoded check at line 222 `if memory_overhead > 500`. These should reference the same value but are independently defined.

---

## 4. Robustness & Error Handling

- **Timeout enforcement** (line 190-192): Uses `asyncio.wait_for` with configurable timeout. Catches `asyncio.TimeoutError` and raises custom `EvaluationTimeoutError`.
- **Generic exception handling** (lines 234-236): Re-raises non-timeout exceptions after logging.
- **GPU monitoring graceful fallback** (lines 60-79): Tries `GPUtil`, then `pynvml`, then returns `None`. This is robust against missing GPU monitoring libraries.
- **`PerformanceGuard.__aexit__`** (lines 298-314): Logs exception type if one occurred, then logs final metrics. Returns `False` to not suppress exceptions.

**Gap:** `ResourceMonitor.__init__` (line 43) calls `psutil.Process()`. If `psutil` is not installed, this raises `ImportError` at module import time (line 6). Unlike the GPU monitoring libraries, `psutil` is a hard dependency with no fallback.

---

## 5. Performance & Scalability

- **Semaphore for concurrency** (line 134): `asyncio.Semaphore(max_concurrent)` properly limits concurrent evaluations. Default is 4.
- **Lazy ResourceMonitor initialization** (lines 141-146): The `resource_monitor` property creates the monitor on first access, avoiding overhead when performance management is disabled.
- **psutil overhead:** Each `cpu_percent()` and `memory_info()` call involves system calls. For frequent evaluations, this adds measurable overhead. However, evaluations are typically infrequent (e.g., every 50K timesteps per the default config).
- **GPU utilization check** (lines 60-79): The `GPUtil.getGPUs()` and `pynvml` calls can be slow (10-100ms). This happens once per evaluation after completion.

---

## 6. Security & Safety

- **`psutil.Process()` access** (line 43): Accesses process information. No elevated privileges needed for self-monitoring.
- **`pynvml.nvmlInit()` (line 73)**: Initializes NVIDIA management library but does not shut it down (`nvmlShutdown()`). Over many evaluations, this could leak NVML handles, though Python's garbage collector typically handles this.
- **Custom exceptions** are clean subclasses of `Exception` with no security concerns.

---

## 7. Maintainability

- **Clear class responsibilities:** Each class has a focused purpose. `ResourceMonitor` handles sampling, `EvaluationPerformanceSLA` handles thresholds, `EvaluationPerformanceManager` orchestrates.
- **`MockEvaluator` is confusingly named** (line 255): It is not a mock in the testing sense -- it is an adapter that wraps an arbitrary callable. The name suggests test-only usage when it is actually part of the production API (`run_evaluation_with_monitoring`).
- **SLA thresholds are class-level constants** (lines 92-97): Easy to find and modify, but not configurable at runtime. No integration with the config schema.
- **Unused local variables** (lines 186-187, 197): `initial_memory`, `initial_cpu`, `final_memory` are computed but not used for anything. They add noise to the code.
- **`PerformanceGuard` class** (lines 285-314) is defined but never used anywhere in the codebase (confirmed by grep -- `core_manager.py` uses `run_evaluation_with_safeguards` directly). This is dead code.

---

## 8. Verdict

**NEEDS_ATTENTION**

Primary concerns:
1. `MockEvaluator.evaluate` uses `loop.run_in_executor(None, func, *args, **kwargs)` which does not support `**kwargs`, making `run_evaluation_with_monitoring` broken for sync functions with keyword arguments.
2. Resource enforcement is purely reactive (post-evaluation), not preventive. Memory limits are checked after the evaluation has already consumed memory.
3. SLA validation is performed twice per evaluation, producing duplicate log output.
4. Multiple unused local variables and the unused `PerformanceGuard` class indicate incomplete implementation or leftover code from refactoring.
5. Hardcoded memory threshold at line 222 is duplicated from `self.max_memory_mb` at line 138.
