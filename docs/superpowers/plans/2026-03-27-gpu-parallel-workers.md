# GPU Parallel Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable parallel self-play workers on GPU instead of CPU, with automatic round-robin device assignment across available GPUs.

**Architecture:** Add `worker_device_map` and `max_workers_per_gpu` to `ParallelConfig`. `ParallelManager` assigns devices per worker. `SelfPlayWorker` reads the device from its config and places its model there. Existing queue/sync infrastructure is unchanged.

**Tech Stack:** Python 3.13, PyTorch 2.11, multiprocessing (spawn context), Pydantic config

**Spec:** `docs/superpowers/specs/2026-03-27-gpu-parallel-workers-design.md`

---

## File Structure

### Modified Files

| File | Change |
|------|--------|
| `keisei/config_schema.py:545-565` | Add `worker_device_map`, `max_workers_per_gpu` to `ParallelConfig` |
| `keisei/training/parallel/parallel_manager.py:81-138` | Add `_assign_worker_device()`, inject device into worker config, add worker restart |
| `keisei/training/parallel/self_play_worker.py:75-80` | Read device from config instead of hardcoding CPU |
| `default_config.yaml:436-465` | Add new fields, update `num_workers` to 16 |

### New Files

| File | Responsibility |
|------|---------------|
| `tests/unit/test_parallel_device_assignment.py` | Unit tests for device assignment and config validation |

---

## Task 1: Config Fields

**Files:**
- Modify: `keisei/config_schema.py:545-565`
- Modify: `default_config.yaml:436-465`
- Test: `tests/unit/test_parallel_device_assignment.py` (new)

- [ ] **Step 1: Write failing tests for new config fields**

Create `tests/unit/test_parallel_device_assignment.py`:

```python
"""Unit tests for GPU parallel worker device assignment."""

import pytest

pytestmark = pytest.mark.unit


class TestParallelConfigFields:
    """New config fields for GPU worker device assignment."""

    def test_worker_device_map_default(self):
        from keisei.config_schema import ParallelConfig

        cfg = ParallelConfig()
        assert cfg.worker_device_map == "auto"

    def test_max_workers_per_gpu_default(self):
        from keisei.config_schema import ParallelConfig

        cfg = ParallelConfig()
        assert cfg.max_workers_per_gpu == 8

    def test_max_workers_per_gpu_validation(self):
        from pydantic import ValidationError

        from keisei.config_schema import ParallelConfig

        with pytest.raises(ValidationError):
            ParallelConfig(max_workers_per_gpu=0)
        with pytest.raises(ValidationError):
            ParallelConfig(max_workers_per_gpu=33)

    def test_worker_device_map_accepts_explicit_device(self):
        from keisei.config_schema import ParallelConfig

        cfg = ParallelConfig(worker_device_map="cuda:1")
        assert cfg.worker_device_map == "cuda:1"

    def test_worker_device_map_accepts_cpu(self):
        from keisei.config_schema import ParallelConfig

        cfg = ParallelConfig(worker_device_map="cpu")
        assert cfg.worker_device_map == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_parallel_device_assignment.py -v`
Expected: FAIL — `worker_device_map` not a field on `ParallelConfig`

- [ ] **Step 3: Add fields to ParallelConfig**

In `keisei/config_schema.py`, add after `worker_seed_offset` (line ~565):

```python
    worker_device_map: str = Field(
        "auto",
        description=(
            "Device assignment for parallel workers. "
            "'auto' distributes round-robin across available GPUs. "
            "'cuda:0' puts all workers on one GPU. "
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

- [ ] **Step 4: Update default_config.yaml**

In `default_config.yaml`, replace the parallel section (lines 436-465):

```yaml
parallel:
  # Enable parallel experience collection with multiple worker processes
  enabled: false

  # Number of parallel worker processes for experience collection
  # For 2x GPU setup: 16 (8 per GPU). Schema default is 4 (conservative).
  num_workers: 16

  # Device assignment for parallel workers
  # "auto" distributes round-robin across GPUs. "cuda:0" for single GPU. "cpu" for CPU-only.
  worker_device_map: "auto"

  # Maximum workers per GPU (each creates a ~400MB CUDA context)
  max_workers_per_gpu: 8

  # Batch size for experience transmission from workers to main process
  batch_size: 64

  # Steps between model weight synchronization across workers
  sync_interval: 200

  # Enable compression for model weight transmission
  compression_enabled: true

  # Timeout for worker communication operations (seconds)
  timeout_seconds: 10.0

  # Maximum size of experience queues
  max_queue_size: 1000

  # Random seed offset for worker processes
  worker_seed_offset: 1000
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_parallel_device_assignment.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/config_schema.py default_config.yaml tests/unit/test_parallel_device_assignment.py
git commit -m "feat(parallel): add worker_device_map and max_workers_per_gpu config fields"
```

---

## Task 2: Device Assignment Logic

**Files:**
- Modify: `keisei/training/parallel/parallel_manager.py:81-138`
- Test: `tests/unit/test_parallel_device_assignment.py`

- [ ] **Step 1: Write failing tests for device assignment**

Add to `tests/unit/test_parallel_device_assignment.py`:

```python
from unittest.mock import patch


class TestDeviceAssignment:
    """ParallelManager._assign_worker_device() distributes workers across GPUs."""

    def _make_manager(self, **overrides):
        from keisei.training.parallel.parallel_manager import ParallelManager

        defaults = {
            "num_workers": 4,
            "enabled": True,
            "batch_size": 8,
            "sync_interval": 50,
            "compression_enabled": False,
            "timeout_seconds": 5,
            "max_queue_size": 100,
            "worker_seed_offset": 1000,
            "worker_device_map": "auto",
            "max_workers_per_gpu": 8,
        }
        defaults.update(overrides)
        return ParallelManager(
            env_config={"seed": 42, "device": "cpu", "input_channels": 46,
                        "num_actions_total": 13527, "max_moves_per_game": 100,
                        "observation_mode": "core46"},
            model_config={"model_type": "resnet", "tower_depth": 2,
                          "tower_width": 32, "se_ratio": 0.0,
                          "input_features": "core46"},
            parallel_config=defaults,
            device="cuda",
        )

    @patch("torch.cuda.device_count", return_value=2)
    def test_auto_round_robin_two_gpus(self, mock_dc):
        pm = self._make_manager()
        assert pm._assign_worker_device(0) == "cuda:0"
        assert pm._assign_worker_device(1) == "cuda:1"
        assert pm._assign_worker_device(2) == "cuda:0"
        assert pm._assign_worker_device(3) == "cuda:1"

    @patch("torch.cuda.device_count", return_value=1)
    def test_auto_single_gpu(self, mock_dc):
        pm = self._make_manager()
        assert pm._assign_worker_device(0) == "cuda:0"
        assert pm._assign_worker_device(1) == "cuda:0"

    @patch("torch.cuda.device_count", return_value=0)
    def test_auto_no_gpu_falls_back_to_cpu(self, mock_dc):
        pm = self._make_manager()
        assert pm._assign_worker_device(0) == "cpu"

    def test_explicit_device(self):
        pm = self._make_manager(worker_device_map="cuda:1")
        assert pm._assign_worker_device(0) == "cuda:1"
        assert pm._assign_worker_device(5) == "cuda:1"

    def test_cpu_device(self):
        pm = self._make_manager(worker_device_map="cpu")
        assert pm._assign_worker_device(0) == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_parallel_device_assignment.py::TestDeviceAssignment -v`
Expected: FAIL — `_assign_worker_device` method doesn't exist

- [ ] **Step 3: Implement `_assign_worker_device` in ParallelManager**

In `keisei/training/parallel/parallel_manager.py`, add method after `__init__`:

```python
    def _assign_worker_device(self, worker_id: int) -> str:
        """Assign a CUDA device to a worker based on the device map config.

        With ``"auto"``, workers are distributed round-robin across available
        GPUs.  Falls back to CPU when no GPU is detected.
        """
        device_map = self.parallel_config.get("worker_device_map", "auto")
        if device_map == "auto":
            gpu_count = torch.cuda.device_count()
            if gpu_count == 0:
                return "cpu"
            return f"cuda:{worker_id % gpu_count}"
        return device_map
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_parallel_device_assignment.py::TestDeviceAssignment -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/parallel/parallel_manager.py tests/unit/test_parallel_device_assignment.py
git commit -m "feat(parallel): add _assign_worker_device with round-robin GPU distribution"
```

---

## Task 3: Wire Device into Worker Spawn

**Files:**
- Modify: `keisei/training/parallel/parallel_manager.py:95-107` (`start_workers`)
- Modify: `keisei/training/parallel/self_play_worker.py:75-80` (`__init__` and `run`)
- Test: `tests/unit/test_parallel_device_assignment.py`

- [ ] **Step 1: Write failing test for worker device injection**

Add to `tests/unit/test_parallel_device_assignment.py`:

```python
class TestWorkerDeviceInjection:
    """ParallelManager injects per-worker device into config before spawn."""

    @patch("torch.cuda.device_count", return_value=2)
    def test_worker_receives_assigned_device(self, mock_dc):
        from keisei.training.parallel.parallel_manager import ParallelManager

        defaults = {
            "num_workers": 2, "enabled": True, "batch_size": 8,
            "sync_interval": 50, "compression_enabled": False,
            "timeout_seconds": 5, "max_queue_size": 100,
            "worker_seed_offset": 1000, "worker_device_map": "auto",
            "max_workers_per_gpu": 8,
        }
        pm = ParallelManager(
            env_config={"seed": 42, "device": "cpu", "input_channels": 46,
                        "num_actions_total": 13527, "max_moves_per_game": 100,
                        "observation_mode": "core46"},
            model_config={"model_type": "resnet", "tower_depth": 2,
                          "tower_width": 32, "se_ratio": 0.0,
                          "input_features": "core46"},
            parallel_config=defaults,
            device="cuda",
        )
        # Build the config that would be passed to worker 0 and worker 1
        cfg0 = dict(defaults)
        cfg0["worker_device"] = pm._assign_worker_device(0)
        cfg1 = dict(defaults)
        cfg1["worker_device"] = pm._assign_worker_device(1)

        assert cfg0["worker_device"] == "cuda:0"
        assert cfg1["worker_device"] == "cuda:1"
```

- [ ] **Step 2: Update `start_workers` to inject device**

In `keisei/training/parallel/parallel_manager.py`, modify the worker creation loop in `start_workers()`. Replace:

```python
            for worker_id in range(self.num_workers):
                worker = SelfPlayWorker(
                    worker_id=worker_id,
                    env_config=self.env_config,
                    model_config=self.model_config,
                    parallel_config=self.parallel_config,
```

With:

```python
            for worker_id in range(self.num_workers):
                # Inject per-worker device assignment
                worker_config = dict(self.parallel_config)
                worker_config["worker_device"] = self._assign_worker_device(worker_id)

                worker = SelfPlayWorker(
                    worker_id=worker_id,
                    env_config=self.env_config,
                    model_config=self.model_config,
                    parallel_config=worker_config,
```

- [ ] **Step 3: Update SelfPlayWorker to read device from config**

In `keisei/training/parallel/self_play_worker.py`, change line 80 from:

```python
        self.device = torch.device("cpu")  # Workers use CPU
```

To:

```python
        self.device = torch.device(
            self.parallel_config.get("worker_device", "cpu")
        )
```

- [ ] **Step 4: Run all parallel tests**

Run: `pytest tests/unit/test_parallel_device_assignment.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/parallel/parallel_manager.py keisei/training/parallel/self_play_worker.py tests/unit/test_parallel_device_assignment.py
git commit -m "feat(parallel): inject GPU device into workers at spawn time"
```

---

## Task 4: Worker Restart on Death

**Files:**
- Modify: `keisei/training/parallel/parallel_manager.py:140-184` (`collect_experiences`)
- Test: `tests/unit/test_parallel_device_assignment.py`

- [ ] **Step 1: Write failing test for worker restart**

Add to `tests/unit/test_parallel_device_assignment.py`:

```python
from unittest.mock import MagicMock


class TestWorkerRestart:
    """Dead workers are detected and respawned during collection."""

    def test_restart_dead_workers_called(self):
        from keisei.training.parallel.parallel_manager import ParallelManager

        defaults = {
            "num_workers": 2, "enabled": True, "batch_size": 8,
            "sync_interval": 50, "compression_enabled": False,
            "timeout_seconds": 5, "max_queue_size": 100,
            "worker_seed_offset": 1000, "worker_device_map": "cpu",
            "max_workers_per_gpu": 8,
        }
        pm = ParallelManager(
            env_config={"seed": 42, "device": "cpu", "input_channels": 46,
                        "num_actions_total": 13527, "max_moves_per_game": 100,
                        "observation_mode": "core46"},
            model_config={"model_type": "resnet", "tower_depth": 2,
                          "tower_width": 32, "se_ratio": 0.0,
                          "input_features": "core46"},
            parallel_config=defaults,
            device="cpu",
        )
        pm.is_running = True

        # Simulate two workers, one dead
        alive_worker = MagicMock()
        alive_worker.is_alive.return_value = True
        alive_worker.worker_id = 0
        dead_worker = MagicMock()
        dead_worker.is_alive.return_value = False
        dead_worker.exitcode = 1
        dead_worker.worker_id = 1
        pm.workers = [alive_worker, dead_worker]

        # _restart_dead_workers should exist and be callable
        assert hasattr(pm, "_restart_dead_workers")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_parallel_device_assignment.py::TestWorkerRestart -v`
Expected: FAIL — `_restart_dead_workers` not found

- [ ] **Step 3: Implement `_restart_dead_workers`**

In `keisei/training/parallel/parallel_manager.py`, add after `collect_experiences`:

```python
    def _restart_dead_workers(self) -> int:
        """Detect and respawn any workers that have died.

        Returns the number of workers restarted.
        """
        restarted = 0
        for i, worker in enumerate(self.workers):
            if not worker.is_alive():
                logger.warning(
                    "Worker %d died (exit code %s), respawning",
                    worker.worker_id,
                    worker.exitcode,
                )
                worker_config = dict(self.parallel_config)
                worker_config["worker_device"] = self._assign_worker_device(
                    worker.worker_id
                )
                new_worker = SelfPlayWorker(
                    worker_id=worker.worker_id,
                    env_config=self.env_config,
                    model_config=self.model_config,
                    parallel_config=worker_config,
                    experience_queue=self.communicator.experience_queues[
                        worker.worker_id
                    ],
                    model_queue=self.communicator.model_queues[worker.worker_id],
                    control_queue=self.communicator.control_queues[worker.worker_id],
                    seed_offset=self.parallel_config["worker_seed_offset"],
                )
                new_worker.start()
                self.workers[i] = new_worker
                restarted += 1
                logger.info(
                    "Worker %d respawned (PID: %d)",
                    new_worker.worker_id,
                    new_worker.pid,
                )
        return restarted
```

Then call it at the start of `collect_experiences`, after the `if not self.enabled` guard:

```python
    def collect_experiences(self, experience_buffer: ExperienceBuffer) -> int:
        if not self.enabled or not self.is_running:
            return 0

        # Respawn any dead workers before collecting
        self._restart_dead_workers()

        experiences_collected = 0
        # ... rest unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_parallel_device_assignment.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing parallel tests for regressions**

Run: `pytest tests/unit/test_communication.py tests/unit/test_model_sync.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/training/parallel/parallel_manager.py tests/unit/test_parallel_device_assignment.py
git commit -m "feat(parallel): auto-restart dead workers during experience collection"
```

---

## Task 5: Integration Test

**Files:**
- Test: `tests/unit/test_parallel_device_assignment.py`

This task validates the full start → collect → stop lifecycle with GPU workers. It requires CUDA, so it's marked with the `integration` marker but kept in the unit test file for convenience.

- [ ] **Step 1: Write integration test**

Add to `tests/unit/test_parallel_device_assignment.py`:

```python
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
class TestGPUWorkerIntegration:
    """End-to-end: start GPU workers, verify alive, stop."""

    def test_start_and_stop_gpu_workers(self):
        import torch
        from keisei.training.parallel.parallel_manager import ParallelManager
        from keisei.training.models.resnet_tower import ActorCriticResTower

        gpu_count = torch.cuda.device_count()
        workers_per_gpu = 2
        num_workers = workers_per_gpu * gpu_count

        config = {
            "num_workers": num_workers, "enabled": True, "batch_size": 8,
            "sync_interval": 50, "compression_enabled": True,
            "timeout_seconds": 10, "max_queue_size": 100,
            "worker_seed_offset": 1000, "worker_device_map": "auto",
            "max_workers_per_gpu": 8,
        }
        pm = ParallelManager(
            env_config={"seed": 42, "device": "cpu", "input_channels": 46,
                        "num_actions_total": 13527, "max_moves_per_game": 100,
                        "observation_mode": "core46"},
            model_config={"model_type": "resnet", "tower_depth": 2,
                          "tower_width": 32, "se_ratio": 0.0,
                          "input_features": "core46"},
            parallel_config=config,
            device="cuda",
        )
        model = ActorCriticResTower(
            input_channels=46, num_actions_total=13527,
            tower_depth=2, tower_width=32, se_ratio=0.0,
        )

        ok = pm.start_workers(model)
        assert ok, "Failed to start GPU workers"

        import time
        time.sleep(10)

        alive = sum(1 for w in pm.workers if w.is_alive())
        assert alive == num_workers, f"Only {alive}/{num_workers} workers alive"

        pm.stop_workers()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/unit/test_parallel_device_assignment.py::TestGPUWorkerIntegration -v -s`
Expected: PASS — workers start on GPUs, survive 10 seconds, stop cleanly

- [ ] **Step 3: Run full unit test suite for regressions**

Run: `pytest tests/unit/ -q`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_parallel_device_assignment.py
git commit -m "test(parallel): add GPU worker integration test"
```

---

## Task 6: End-to-End Training Test

No new code — just verify the full pipeline works.

- [ ] **Step 1: Run short parallel training**

```bash
timeout 120 uv run python train.py train \
  --override env.device=cuda \
  --override training.total_timesteps=10000 \
  --override training.steps_per_epoch=512 \
  --override training.mixed_precision=false \
  --override parallel.enabled=true \
  --override parallel.num_workers=4 \
  --override parallel.worker_device_map=auto \
  --override webui.enabled=false \
  --override wandb.enabled=false \
  --override logging.run_name=parallel-e2e-test
```

Expected: Runs for 2+ PPO epochs without error. Check `models/parallel-e2e-test/training_log.txt` for PPO update lines.

- [ ] **Step 2: Verify workers used GPU**

Check training log for worker device assignments (logged by ParallelManager).

- [ ] **Step 3: Commit any fixes needed**

If issues arise, fix and commit before proceeding.

- [ ] **Step 4: Final commit with updated CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "feat(parallel): GPU parallel self-play workers"
```
