# GPU Parallel Self-Play Workers

**Status**: Approved
**Date**: 2026-03-27

## Problem

Sequential training at ~7 it/s means each game takes 1-8 minutes per episode (100-500 moves). To observe training at human-readable speed (~1 move/2 seconds per game), we need many concurrent games. CPU inference is infeasible (9.2s/step for a 9-block/256-channel ResNet), but GPU inference is only 2ms/step — the bottleneck is the Python game engine (~140ms/step).

## Solution

Run 16 self-play worker processes (8 per GPU), each with its own `ShogiGame` instance and CUDA model copy. Workers collect experience independently and stream it to the main process via queues. The main process runs PPO updates on GPU 0.

Worker count is conservative (8/GPU) because each spawned process creates a full CUDA context (~300-500MB overhead beyond model weights). Actual VRAM per worker should be measured at startup; the `max_workers_per_gpu` config field enforces the cap.

## Hardware

- 2x NVIDIA RTX 4060 Ti 16GB
- 24 CPU cores
- Model size: ~300MB VRAM per copy + ~400MB CUDA context per process
- 8 copies per GPU = ~5.6GB, leaving ~10GB headroom

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

Add two fields:

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
max_workers_per_gpu: int = Field(
    8,
    ge=1,
    le=32,
    description="Maximum workers per GPU. Each worker creates a full CUDA context (~400MB overhead).",
)
```

Keep `num_workers` schema default at 4 (conservative for unknown hardware). Set 16 in `default_config.yaml` for documented 2-GPU setup.

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
  num_workers: 16
  worker_device_map: "auto"
  max_workers_per_gpu: 8
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

## Design Notes

- **torch.compile**: Workers use non-compiled models (inference-only, compile overhead not justified). The main process may use compiled models; `state_dict()` keys are compatible between compiled and non-compiled models in current PyTorch.
- **Worker restart**: If a worker dies during training (CUDA OOM, corrupted queue), `ParallelManager` should detect it via `is_alive()` checks during `collect_experiences()`, log a warning, and respawn the worker with a fresh CUDA context. This prevents silent throughput degradation over long runs.
- **CUDA_VISIBLE_DEVICES**: Each worker should set `os.environ["CUDA_VISIBLE_DEVICES"]` to its assigned GPU index before any torch import in `run()`. This prevents accidental cross-GPU allocations.

## What We're NOT Doing

- No batched/centralized inference server — each worker runs independently
- No shared model instances — each worker has its own copy
- No shared-memory weight broadcast — queue-per-worker is sufficient at 16 workers; shared memory is a future optimization if broadcast latency becomes a bottleneck
- No double buffering — workers block briefly if queue fills during PPO
- No mixed precision for workers — FP32 only (mixed precision PPO is a separate bug: keisei-640f840cdf)
- No DDP — that's Phase 5

## Expected Performance

- 16 workers × ~7 steps/sec per worker = ~112 steps/sec aggregate
- PPO epoch (2048 steps) fills in ~18 seconds instead of ~4.5 minutes
- At 112 steps/sec across 16 games: ~7 moves/sec total, ~0.4 moves/sec per game (~2.5 seconds between moves per game)
- PPO update takes ~14 seconds, queue capacity (1000) absorbs backpressure
- Can scale to 24+ workers if VRAM measurement confirms headroom

## Testing

1. Unit: `ParallelManager._assign_worker_device()` returns correct devices for various GPU counts
2. Integration: Start 4 workers (2 per GPU), verify all alive after 10 seconds, collect experiences
3. End-to-end: `python train.py train --override parallel.enabled=true --override parallel.num_workers=4` runs for 2+ epochs without error
