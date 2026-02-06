# Code Analysis: `keisei/training/parallel/parallel_manager.py`

## 1. Purpose & Role

`ParallelManager` is the top-level coordinator for the parallel experience collection system. It manages the lifecycle of worker processes (`SelfPlayWorker`), orchestrates model synchronization timing, collects experiences from workers into the main `ExperienceBuffer`, and provides health-checking and statistics. It is designed to be used as a context manager (lines 339-345).

## 2. Interface Contracts

### Constructor (`__init__`, lines 31-79)
- **Inputs**: `env_config: Dict`, `model_config: Dict`, `parallel_config: Dict`, `device: str = "cuda"`
- **Expects** `parallel_config` to contain keys: `num_workers`, `batch_size`, `enabled`, `max_queue_size`, `timeout_seconds`, `sync_interval`, `compression_enabled`, `worker_seed_offset`.
- **Creates**: `WorkerCommunicator` and `ModelSynchronizer` instances.
- **State**: `workers: List[SelfPlayWorker]`, `worker_stats: Dict[int, Dict]`, counters, `is_running: bool`.

### `start_workers` (lines 81-124)
- **Inputs**: `initial_model: nn.Module`
- **Returns**: `bool` -- True if all workers started.
- **Contract**: Creates `SelfPlayWorker` instances, calls `worker.start()`, then sends initial model weights. On failure, calls `stop_workers()`.

### `collect_experiences` (lines 126-173)
- **Inputs**: `experience_buffer: ExperienceBuffer`
- **Returns**: `int` -- number of experiences collected.
- **Contract**: Drains worker queues via communicator, calls `experience_buffer.add_from_worker_batch()` for each batch, updates stats.

### `sync_model_if_needed` (lines 175-192)
- **Inputs**: `model: nn.Module`, `current_step: int`
- **Returns**: `bool` -- True if sync occurred.

### `stop_workers` (lines 223-253)
- **Contract**: Sends stop command, joins workers with 5s timeout, force-terminates lingerers with 2s final join, cleans up communicator.

### `is_healthy` (lines 311-337)
- **Returns**: `bool`
- **Contract**: Checks that all workers are alive and at least one worker produced data in the last 30 seconds.

## 3. Correctness Analysis

- **Worker PID access before ready (line 112)**: After `worker.start()` (line 109), `worker.pid` is accessed on line 112 for logging. The `pid` attribute is set by the `start()` call and is available immediately after, so this is correct.

- **`collect_experiences` relies on `add_from_worker_batch` (line 148)**: The batch data passed is `batch_data["experiences"]`, which is the batched tensor dict from `SelfPlayWorker._experiences_to_batch`. This matches the expected input format of `ExperienceBuffer.add_from_worker_batch(worker_data: Dict[str, torch.Tensor])`. The tensor keys (`obs`, `actions`, `rewards`, `log_probs`, `values`, `dones`, `legal_masks`) match what `add_from_worker_batch` expects. This contract is correctly maintained.

- **`_sync_model_to_workers` bypasses `ModelSynchronizer.prepare_model_for_sync`** (lines 194-221): As noted in the `model_sync.py` analysis, this method directly calls `self.communicator.send_model_weights(model.state_dict(), ...)` rather than using `self.model_sync.prepare_model_for_sync()`. The `ModelSynchronizer` is used only for interval tracking via `should_sync` and `mark_sync_completed`. This is functionally correct but renders half of `ModelSynchronizer`'s API dead.

- **`_calculate_collection_rate` (lines 278-292)**: Calculates steps/second based on `last_batch_size` from workers active in the last 60 seconds. The formula divides total recent steps by 60.0 seconds. This is an approximation -- it sums the most recent batch sizes (not all batches in the last minute) and divides by 60. If a worker sends one batch of 100 experiences at second 59, the rate would be 100/60 = 1.67 steps/sec, which is misleading. The metric is labeled "collection_rate" and serves monitoring purposes, so approximate values are acceptable.

- **`is_healthy` false positive on startup (lines 311-337)**: After `start_workers`, `worker_stats` is empty (no batches collected yet). `is_healthy` checks `recent_activity` via `any(...)` over `self.worker_stats.values()`, which returns `False` for an empty dict. This means `is_healthy` returns `False` immediately after startup, before any worker has sent its first batch. This could trigger false health alarms during the initial model sync period.

- **`stop_workers` double-stop safety (lines 223-253)**: The method checks `if not self.workers` at line 225, preventing operations on an empty worker list. After stopping, `self.workers.clear()` ensures re-entry is safe. However, `communicator.cleanup()` (line 245) sends another stop command internally (via `send_control_command("stop")`), resulting in two stop commands being sent -- one from `stop_workers` directly (line 232) and one from `cleanup` (line 224 of communication.py). This is harmless but redundant.

## 4. Robustness & Error Handling

- **Worker startup failure (lines 120-124)**: If any worker fails to start, `stop_workers()` is called to clean up already-started workers. This is correct cleanup behavior.
- **Silent experience collection failures (lines 170-173)**: Errors during collection are caught and logged, returning 0. The caller has no way to distinguish "no experiences available" from "collection failed."
- **Force termination (lines 239-242)**: Workers that don't join within 5 seconds are terminated. After `terminate()`, a 2-second join is attempted. If the worker still doesn't terminate (e.g., stuck in a system call), it becomes a zombie process. There is no `kill()` fallback.
- **No worker restart**: If a worker dies, `is_healthy` detects it but there is no mechanism to restart failed workers. The system degrades with fewer workers.

## 5. Performance & Scalability

- **Sequential worker creation (lines 97-111)**: Workers are created and started in a loop. For a small number of workers (e.g., 4-8), this is fine. The startup cost is dominated by model initialization in each worker process.
- **Sequential experience collection**: `collect_experiences` drains queues sequentially. This is adequate since queue reads are O(queue_depth) with no blocking.
- **Model sync cost**: Every sync transfers the full model state dict through each worker's queue, requiring N separate pickle operations. For large models with many workers, this could be a bottleneck.
- **Context manager support** (lines 339-345): Ensures cleanup on scope exit, preventing resource leaks.

## 6. Security & Safety

- **Config dict access**: `parallel_config` keys are accessed with direct indexing (e.g., `parallel_config["num_workers"]` on line 53) rather than `.get()`. Missing keys will raise `KeyError` at construction time, which is fail-fast behavior but could produce unclear error messages.
- **No resource limits**: There is no upper bound on `num_workers`. Creating too many worker processes could exhaust system resources.
- **Process termination**: `worker.terminate()` sends SIGTERM on Unix. Worker processes should handle this gracefully, but the worker's `_cleanup_worker` in the `finally` block may not execute if the process is terminated externally.

## 7. Maintainability

- **Clean structure**: The class follows a clear lifecycle pattern (init -> start -> collect/sync loop -> stop) with well-separated methods.
- **Good logging**: Operations are logged at appropriate levels throughout.
- **Dict-based config**: Using raw dicts for configuration means key names are stringly-typed and there's no schema validation. This contrasts with the Pydantic-based config system used elsewhere in the project.
- **Context manager**: The `__enter__`/`__exit__` pattern is a good practice for resource management.
- **Health check limitations**: `is_healthy` has the startup false-positive issue noted above, and the 30-second activity threshold is hardcoded.

## 8. Verdict

**NEEDS_ATTENTION** -- The module is structurally sound and handles the happy path well. Key concerns are: (1) `is_healthy` returns false immediately after startup before workers produce data, (2) no worker restart mechanism means the system degrades permanently if a worker dies, (3) zombie process risk if `terminate()` fails to stop a worker, and (4) dict-based config bypasses the project's Pydantic validation system.
