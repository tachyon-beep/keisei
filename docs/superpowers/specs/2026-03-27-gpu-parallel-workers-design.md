# GPU Parallel Self-Play Workers

**Status**: Approved
**Date**: 2026-03-27

## Problem

Sequential training at ~7 it/s means each game takes 1-8 minutes per episode (100-500 moves). To observe training at human-readable speed (~1 move/2 seconds per game), we need many concurrent games. CPU inference is infeasible (9.2s/step for a 9-block/256-channel ResNet), but GPU inference is only 2ms/step — the bottleneck is the Python game engine (~140ms/step).

## Solution

Run 24 self-play worker processes (12 per GPU), each with its own `ShogiGame` instance and CUDA model copy. Workers collect experience independently and stream it to the main process via queues. The main process runs PPO updates on GPU 0.

## Hardware

- 2x NVIDIA RTX 4060 Ti 16GB
- 24 CPU cores
- Model size: ~300MB VRAM per copy
- 12 copies per GPU = ~3.6GB, well within 16GB

## Architecture

```
GPU 0 (cuda:0)                    GPU 1 (cuda:1)
├── Main process (PPO updates)    ├── Worker 12 (game + model)
├── Worker 0  (game + model)      ├── Worker 13 (game + model)
├── Worker 1  (game + model)      ├── ...
├── ...                           └── Worker 23 (game + model)
└── Worker 11 (game + model)

            ↓ experience queues ↓
        ┌───────────────────────────┐
        │   Main Process (GPU 0)    │
        │   - Collect from queues   │
        │   - Fill ExperienceBuffer │
        │   - PPO update            │
        │   - Broadcast weights     │
        └───────────────────────────┘
            ↑ model weight queues ↑
```

## Data Flow

1. Each worker runs its own game loop: `game.step()` → `model(obs)` → `experience` → `queue.put()`
2. Main process calls `parallel_manager.collect_experiences(buffer)` — non-blocking reads from all 24 queues
3. When buffer reaches `steps_per_epoch` (2048), main runs PPO update on GPU 0
4. After PPO update, `ModelSynchronizer` broadcasts updated weights to all workers via model queues
5. Workers receive weights (CPU numpy), load to their assigned GPU, continue playing

## Changes

### 1. ParallelConfig (config_schema.py)

Add one field:

```python
worker_device_map: str = Field(
    "auto",
    description=(
        "Device assignment for parallel workers. "
        "'auto' distributes round-robin across available GPUs. "
        "'cuda:0' puts all workers on GPU 0. "
        "'cpu' forces CPU inference (not recommended for large models)."
    ),
)
```

Update defaults:

```python
num_workers: int = Field(24, ge=1, description="Number of parallel workers")
```

### 2. SelfPlayWorker (self_play_worker.py)

Current code in `run()` creates the model on CPU:

```python
self.device = torch.device("cpu")
```

Change to read device from config:

```python
self.device = torch.device(self.parallel_config.get("worker_device", "cpu"))
```

The `worker_device` key is set per-worker by `ParallelManager` before spawning (see below). No other changes — the worker already handles device placement for observations and model inference.

### 3. ParallelManager (parallel_manager.py)

In `start_workers()`, compute per-worker device assignment:

```python
def _assign_worker_device(self, worker_id: int) -> str:
    device_map = self.parallel_config.get("worker_device_map", "auto")
    if device_map == "auto":
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            return "cpu"
        return f"cuda:{worker_id % gpu_count}"
    return device_map  # e.g. "cuda:0" or "cpu"
```

Before creating each worker, inject the device into its config copy:

```python
worker_config = dict(self.parallel_config)
worker_config["worker_device"] = self._assign_worker_device(worker_id)
worker = SelfPlayWorker(..., parallel_config=worker_config, ...)
```

### 4. WorkerCommunicator (communication.py)

No changes. Experience data is serialized as numpy arrays (CPU). Model weights are serialized as CPU numpy by `ModelSynchronizer`. The queues are device-agnostic.

### 5. ModelSynchronizer (model_sync.py)

No changes. `prepare_model_for_sync()` already calls `.cpu().numpy()` on all tensors. `restore_model_from_sync()` uses `torch.from_numpy().to(device)` — the worker's device is already respected.

### 6. default_config.yaml

```yaml
parallel:
  enabled: false
  num_workers: 24
  worker_device_map: "auto"
  batch_size: 64
  sync_interval: 200
  # ... rest unchanged
```

## Existing Infrastructure Used As-Is

- **Spawn context**: Already fixed — `WorkerCommunicator` creates queues with `mp.get_context("spawn")`, `SelfPlayWorker` extends `spawn.Process`
- **Experience buffer**: `add_from_worker_batch()` already handles worker data
- **Model sync**: Compression, decompression, and weight broadcast already work
- **Queue management**: `max_queue_size=1000` with timeout handles backpressure during PPO updates
- **Worker health checks**: `ParallelManager` already verifies workers are alive after spawn

## What We're NOT Doing

- No batched/centralized inference server — each worker runs independently
- No shared model instances — each worker has its own copy
- No double buffering — workers block briefly if queue fills during PPO
- No mixed precision for workers — FP32 only (mixed precision PPO is a separate bug: keisei-640f840cdf)
- No DDP — that's Phase 5

## Expected Performance

- 24 workers × ~7 steps/sec per worker = ~168 steps/sec aggregate
- PPO epoch (2048 steps) fills in ~12 seconds instead of ~4.5 minutes
- At 168 steps/sec across 24 games: ~7 moves/sec total, ~0.3 moves/sec per game (~3 seconds between moves per game)
- PPO update takes ~14 seconds, queue capacity (1000) absorbs backpressure

## Testing

1. Unit: `ParallelManager._assign_worker_device()` returns correct devices for various GPU counts
2. Integration: Start 4 workers (2 per GPU), verify all alive after 10 seconds, collect experiences
3. End-to-end: `python train.py train --override parallel.enabled=true --override parallel.num_workers=4` runs for 2+ epochs without error
