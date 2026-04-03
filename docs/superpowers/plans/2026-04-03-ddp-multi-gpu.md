# PyTorch DDP Multi-GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable distributed training across multiple GPUs using PyTorch DistributedDataParallel (DDP), where each rank runs its own vector environment and collects independent rollouts, then synchronizes gradients during PPO updates.

**Architecture:** Each rank is an independent training process with its own `VecEnv`, rollout buffer, and copy of the DDP-wrapped model. Gradients are automatically averaged by DDP during `loss.backward()`. Rank 0 is responsible for checkpointing, metrics logging, DB writes, and game snapshots. All ranks participate in rollout collection and PPO updates. State that must be synchronized across ranks (LR scheduler, checkpoint resume) uses explicit `dist.all_reduce` or `dist.broadcast_object_list` calls.

**Tech Stack:** PyTorch DDP (`torch.distributed`, `torch.nn.parallel.DistributedDataParallel`), NCCL backend (GPU), gloo backend (CPU/tests), `torchrun` launcher

**Filigree issue:** keisei-8f9a29c247

---

## Critical Invariants

These invariants must be maintained throughout implementation. Violating any of them causes silent training corruption or deadlocks.

1. **All ranks must have identical model weights at all times.** DDP averages gradients, so if weights diverge, gradients are computed from inconsistent parameter values.
2. **All ranks must call `loss.backward()` the same number of times per epoch.** A mismatch causes an NCCL allreduce deadlock (one rank waits forever for the other).
3. **LR scheduler state must be identical across ranks.** Different LRs cause different update magnitudes despite averaged gradients, silently degrading training.
4. **On resume, ALL ranks must load the checkpoint** — not just rank 0. DDP does not re-broadcast weights after `__init__()`.
5. **Per-rank RNG seeds must differ** so rollouts are independent. But numpy/Python/torch RNG all need seeding, not just torch.
6. **League mode (split-merge) is incompatible with DDP** because buffer sizes can differ across ranks, causing a deadlock. The guard in Task 8 prevents this.

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `keisei/training/distributed.py` | DDP setup/teardown, rank helpers, distributed utility functions |
| Modify | `keisei/config.py:17-25` | Add `DistributedConfig` dataclass and wire into `AppConfig` |
| Modify | `keisei/training/katago_loop.py:155-310` | DDP init in constructor, rank-gated I/O, per-rank VecEnv |
| Modify | `keisei/training/katago_loop.py:376-635` | Rank-gated logging, checkpointing, metrics, LR sync |
| Modify | `keisei/training/katago_loop.py:716-738` | Switch entry point to support `torchrun` launch |
| Modify | `keisei/training/katago_ppo.py:205-308` | Accept DDP-wrapped forward_model (already supports this — no changes needed) |
| Modify | `keisei/training/checkpoint.py:42-136` | Per-rank CUDA RNG state, world_size metadata, world_size mismatch guard |
| Create | `tests/unit/test_distributed.py` | Unit tests for distributed utilities |
| Create | `tests/integration/test_ddp_training.py` | Integration test: 2-rank training on CPU |
| Modify | `pyproject.toml` | Register `slow` pytest marker |

---

### Task 1: Create `keisei/training/distributed.py` — DDP utilities

**Files:**
- Create: `keisei/training/distributed.py`
- Test: `tests/unit/test_distributed.py`

- [ ] **Step 1: Write failing tests for distributed utilities**

```python
# tests/unit/test_distributed.py
"""Unit tests for distributed training utilities."""

import os
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

from keisei.training.distributed import (
    DistributedContext,
    get_distributed_context,
    setup_distributed,
    cleanup_distributed,
    seed_all_ranks,
    broadcast_string,
)


class TestDistributedContext:
    def test_single_gpu_context(self):
        """Non-distributed mode returns rank=0, world_size=1."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
            assert ctx.rank == 0
            assert ctx.local_rank == 0
            assert ctx.world_size == 1
            assert ctx.is_distributed is False
            assert ctx.is_main is True
            assert ctx.device == torch.device("cpu")

    def test_multi_gpu_context_rank0(self):
        ctx = DistributedContext(rank=0, local_rank=0, world_size=4, is_distributed=True)
        assert ctx.is_main is True

    def test_multi_gpu_context_rank1(self):
        ctx = DistributedContext(rank=1, local_rank=1, world_size=4, is_distributed=True)
        assert ctx.is_main is False

    @patch("torch.cuda.is_available", return_value=False)
    def test_device_cpu_when_no_cuda(self, _mock):
        """Device is CPU when CUDA is unavailable."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        assert ctx.device == torch.device("cpu")

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cuda_when_not_distributed(self, _mock):
        """Non-distributed with CUDA returns cuda:0."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        assert ctx.device == torch.device("cuda")

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_cuda_distributed_uses_local_rank(self, _mock):
        """Distributed mode returns cuda:{local_rank}."""
        ctx = DistributedContext(rank=3, local_rank=2, world_size=4, is_distributed=True)
        assert ctx.device == torch.device("cuda:2")

    def test_device_is_stable(self):
        """Device property returns the same object on repeated access."""
        with patch("torch.cuda.is_available", return_value=False):
            ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
            assert ctx.device is ctx.device


class TestGetDistributedContext:
    def test_returns_single_gpu_when_no_env(self):
        """Without torchrun env vars, returns non-distributed context."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
        with patch.dict(os.environ, env, clear=True):
            ctx = get_distributed_context()
            assert ctx.is_distributed is False
            assert ctx.world_size == 1

    def test_returns_distributed_when_env_set(self):
        """With torchrun env vars, returns distributed context."""
        env = {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
        with patch.dict(os.environ, env):
            ctx = get_distributed_context()
            assert ctx.is_distributed is True
            assert ctx.rank == 1
            assert ctx.world_size == 4

    def test_raises_on_partial_env(self):
        """RANK set but LOCAL_RANK missing raises RuntimeError."""
        env = {"RANK": "0"}
        with patch.dict(os.environ, env, clear=False):
            # Remove LOCAL_RANK and WORLD_SIZE if they exist
            os.environ.pop("LOCAL_RANK", None)
            os.environ.pop("WORLD_SIZE", None)
            with pytest.raises(RuntimeError, match="LOCAL_RANK"):
                get_distributed_context()


class TestSeedAllRanks:
    def test_seeds_all_rngs(self):
        """seed_all_ranks sets torch, numpy, and Python RNG."""
        seed_all_ranks(123)
        # Verify reproducibility
        a = torch.randn(3)
        b = np.random.rand(3)
        c = random.random()

        seed_all_ranks(123)
        assert torch.equal(torch.randn(3), a)
        assert np.array_equal(np.random.rand(3), b)
        assert random.random() == c
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_distributed.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'keisei.training.distributed'`

- [ ] **Step 3: Implement distributed utilities**

```python
# keisei/training/distributed.py
"""Distributed training utilities for PyTorch DDP.

Usage:
    # In main():
    ctx = get_distributed_context()
    setup_distributed(ctx)
    try:
        # ... training ...
    finally:
        cleanup_distributed(ctx)

    # Launch with:
    torchrun --nproc_per_node=4 -m keisei.training.katago_loop config.toml
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _resolve_device(is_distributed: bool, local_rank: int) -> torch.device:
    """Compute the device once at construction time."""
    if is_distributed and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Immutable snapshot of the distributed training environment.

    Created once at process startup and threaded through all components.
    The `device` field is resolved at construction time (not lazily) to
    avoid inconsistent results across the DDP setup boundary.
    """

    rank: int
    local_rank: int
    world_size: int
    is_distributed: bool
    device: torch.device = field(init=False)

    def __post_init__(self) -> None:
        # frozen=True requires object.__setattr__ for post-init assignment
        object.__setattr__(
            self, "device",
            _resolve_device(self.is_distributed, self.local_rank),
        )

    @property
    def is_main(self) -> bool:
        """True if this is rank 0 or non-distributed."""
        return self.rank == 0


def _require_env(key: str) -> str:
    """Read a required torchrun environment variable."""
    val = os.environ.get(key)
    if val is None:
        raise RuntimeError(
            f"torchrun env var {key!r} is missing. "
            f"Ensure RANK, LOCAL_RANK, and WORLD_SIZE are all set. "
            f"Launch with: torchrun --nproc_per_node=N your_script.py"
        )
    return val


def get_distributed_context() -> DistributedContext:
    """Detect distributed environment from torchrun env vars.

    Returns a non-distributed context if RANK env var is absent (single-GPU mode).
    Raises RuntimeError if RANK is set but LOCAL_RANK or WORLD_SIZE are missing.
    """
    rank_str = os.environ.get("RANK")
    if rank_str is None:
        return DistributedContext(
            rank=0, local_rank=0, world_size=1, is_distributed=False,
        )
    return DistributedContext(
        rank=int(rank_str),
        local_rank=int(_require_env("LOCAL_RANK")),
        world_size=int(_require_env("WORLD_SIZE")),
        is_distributed=True,
    )


def setup_distributed(ctx: DistributedContext, backend: str = "nccl") -> None:
    """Initialize the process group for DDP.

    Must be called before any CUDA operations or DDP wrapping.
    No-op if ctx.is_distributed is False.

    Args:
        ctx: The distributed context from get_distributed_context().
        backend: Communication backend. "nccl" for GPU training,
                 "gloo" for CPU-only (tests). Default: "nccl".
    """
    if not ctx.is_distributed:
        return

    try:
        if backend == "nccl" and torch.cuda.is_available():
            torch.cuda.set_device(ctx.local_rank)
        dist.init_process_group(backend=backend)
        logger.info(
            "DDP initialized: rank=%d, local_rank=%d, world_size=%d, backend=%s",
            ctx.rank, ctx.local_rank, ctx.world_size, backend,
        )
    except Exception:
        logger.error(
            "DDP init failed: rank=%d, local_rank=%d, world_size=%d, "
            "MASTER_ADDR=%s, MASTER_PORT=%s",
            ctx.rank, ctx.local_rank, ctx.world_size,
            os.environ.get("MASTER_ADDR", "<unset>"),
            os.environ.get("MASTER_PORT", "<unset>"),
        )
        raise


def cleanup_distributed(ctx: DistributedContext) -> None:
    """Destroy the process group. No-op if not distributed."""
    if ctx.is_distributed and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("DDP process group destroyed: rank=%d", ctx.rank)


def seed_all_ranks(seed: int) -> None:
    """Set torch, numpy, and Python RNG seeds for reproducibility.

    Call with `base_seed + rank` to get different-but-reproducible
    rollouts per rank.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def broadcast_string(value: str | None, src: int = 0) -> str:
    """Broadcast a string from src rank to all ranks.

    Used to share the checkpoint path from rank 0 (which reads the DB)
    to all other ranks (which need to load the same checkpoint).

    Args:
        value: The string to broadcast. Only needs to be set on src rank.
        src: Source rank (default 0).

    Returns:
        The broadcast string on all ranks.
    """
    if not dist.is_initialized():
        assert value is not None
        return value

    obj_list = [value]
    dist.broadcast_object_list(obj_list, src=src)
    result = obj_list[0]
    assert result is not None, "broadcast_string returned None"
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_distributed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/distributed.py tests/unit/test_distributed.py
git commit -m "feat(ddp): add distributed training utilities module"
```

---

### Task 2: Add `DistributedConfig` to config system and register pytest marker

**Files:**
- Modify: `keisei/config.py:69-75` (AppConfig) and `keisei/config.py:78-155` (load_config)
- Modify: `pyproject.toml` (register `slow` marker)
- Test: `tests/test_katago_config.py` (add test cases)

- [ ] **Step 1: Write failing test for DistributedConfig**

Add to `tests/test_katago_config.py`:

```python
from keisei.config import DistributedConfig


class TestDistributedConfig:
    def test_defaults(self):
        cfg = DistributedConfig()
        assert cfg.sync_batchnorm is True
        assert cfg.find_unused_parameters is False
        assert cfg.gradient_as_bucket_view is True

    def test_custom_values(self):
        cfg = DistributedConfig(sync_batchnorm=False)
        assert cfg.sync_batchnorm is False

    def test_rejects_unknown_keys(self):
        """Typos in config keys should fail loudly."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            DistributedConfig(sycn_batchnorm=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_config.py::TestDistributedConfig -v`
Expected: FAIL with `ImportError: cannot import name 'DistributedConfig'`

- [ ] **Step 3: Add DistributedConfig dataclass and wire into AppConfig**

In `keisei/config.py`, add after `DemonstratorConfig` (after line 66):

```python
@dataclass(frozen=True)
class DistributedConfig:
    """Configuration for PyTorch DDP multi-GPU training.

    DDP activation is determined by torchrun environment variables (RANK,
    LOCAL_RANK, WORLD_SIZE), not by this config. This config controls DDP
    behavior only when DDP is active.
    """

    sync_batchnorm: bool = True
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
```

Note: The original plan had an `enabled: bool` field, but this was removed because DDP
activation is driven entirely by the presence of torchrun env vars. A config flag would
create a false impression of control — a user could set `enabled=true` without torchrun
and silently get no DDP. Removing it eliminates the trap.

Modify `AppConfig` (line 69-75) to add the field:

```python
@dataclass(frozen=True)
class AppConfig:
    training: TrainingConfig
    display: DisplayConfig
    model: ModelConfig
    league: LeagueConfig | None = None
    demonstrator: DemonstratorConfig | None = None
    distributed: DistributedConfig = DistributedConfig()
```

In `load_config()`, before the final `return` (around line 150), add:

```python
    dist_raw = raw.get("distributed", {})
    # Validate field names explicitly like other config sections. This catches
    # typos that would otherwise produce a confusing TypeError from __init__.
    valid_dist_fields = {"sync_batchnorm", "find_unused_parameters", "gradient_as_bucket_view"}
    unknown = set(dist_raw.keys()) - valid_dist_fields
    if unknown:
        raise ValueError(
            f"Unknown [distributed] config keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid_dist_fields)}"
        )
    dist_config = DistributedConfig(**dist_raw)
```

And update the return statement:

```python
    return AppConfig(
        training=training, display=display, model=model,
        league=league_config, demonstrator=demo_config,
        distributed=dist_config,
    )
```

- [ ] **Step 4: Register `slow` pytest marker in pyproject.toml**

Add to `[tool.pytest.ini_options]` in `pyproject.toml`:

```toml
markers = [
    "slow: marks tests as slow-running (deselect with '-m \"not slow\"')",
]
```

If a `[tool.pytest.ini_options]` section already exists, add the `markers` key to it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_config.py -v`
Expected: PASS (both new and existing tests)

- [ ] **Step 6: Commit**

```bash
git add keisei/config.py pyproject.toml tests/test_katago_config.py
git commit -m "feat(ddp): add DistributedConfig to config system, register slow marker"
```

---

### Task 3: Modify `KataGoTrainingLoop.__init__` for DDP

**Files:**
- Modify: `keisei/training/katago_loop.py:155-310`
- Modify: `tests/test_katago_loop.py` (add `_make_config()` helper and DDP init tests)

This is the core task. The constructor needs to:
1. Accept a `DistributedContext` parameter
2. Use `ctx.device` instead of auto-detecting device
3. Wrap model in DDP after moving to device
4. Pass the DDP-wrapped model as `forward_model` to PPO
5. Create per-rank VecEnv (each rank gets its own environments)
6. Optionally convert BatchNorm to SyncBatchNorm

**Important ordering invariant:** The order must be:
1. `model.to(device)` — move model to GPU
2. `SyncBatchNorm.convert_sync_batchnorm()` — convert BN layers (must be before DDP wrapping)
3. `DDP(model, ...)` — wrap in DDP (registers gradient hooks)
4. `KataGoPPOAlgorithm(base_model, forward_model=ddp_model)` — PPO gets wrapped model
5. `_check_resume()` — checkpoint loading (must load on ALL ranks, see Task 5)

Steps 2 and 3 are DDP-only. This ordering ensures DDP wraps the SyncBN model,
and `configure_amp()` in `KataGoPPOAlgorithm.__init__` is called on the unwrapped
base model (which is correct — DDP's `forward` delegates to `module.forward`).

**Note on torch.compile + DDP:** `KataGoPPOAlgorithm.__init__` calls `torch.compile(self.forward_model)` where `forward_model` is the DDP-wrapped model. This is the correct order per PyTorch docs — compile wraps the already-DDP-wrapped module. The compiled graph will include NCCL gradient hooks. When `compile_mode is not None` and `sync_batchnorm=True`, integration tests must spawn all expected ranks or the embedded NCCL calls will deadlock.

**Note on nested autocast:** The PPO `update()` method wraps the forward pass in an outer `autocast` context (line 568 of katago_ppo.py). The model's `forward()` method also has an inner `autocast` (katago_base.py:71-75). With identical `device_type` and `dtype`, the inner autocast is a no-op. Do not change either autocast site independently — they must agree on dtype.

- [ ] **Step 1: Add `_make_config()` test helper and write failing tests**

The existing test file uses a `@pytest.fixture` called `katago_config` (line 82 of `tests/test_katago_loop.py`). New DDP test classes need config without a fixture, so add a free-function helper. Add this BEFORE the `katago_config` fixture (after `_make_mock_katago_vecenv`):

```python
def _make_config(tmp_path: Path | None = None) -> AppConfig:
    """Create a minimal AppConfig for testing.

    Uses a temp directory for checkpoint_dir and db_path. If tmp_path is
    None, uses /tmp with a unique suffix.
    """
    import tempfile

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    return AppConfig(
        training=TrainingConfig(
            num_games=2,
            max_ply=50,
            algorithm="katago_ppo",
            checkpoint_interval=5,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            algorithm_params={
                "learning_rate": 2e-4,
                "gamma": 0.99,
                "lambda_policy": 1.0,
                "lambda_value": 1.5,
                "lambda_score": 0.02,
                "lambda_entropy": 0.01,
                "score_normalization": 76.0,
                "grad_clip": 1.0,
            },
        ),
        display=DisplayConfig(
            moves_per_minute=0,
            db_path=str(tmp_path / "test.db"),
        ),
        model=ModelConfig(
            display_name="Test-KataGo",
            architecture="se_resnet",
            params={
                "num_blocks": 2,
                "channels": 32,
                "se_reduction": 8,
                "global_pool_channels": 16,
                "policy_channels": 8,
                "value_fc_size": 32,
                "score_fc_size": 16,
                "obs_channels": 50,
            },
        ),
    )
```

Also add `from pathlib import Path` to the imports at the top of the file if not already present.

Then add the import and test class:

```python
from keisei.training.distributed import DistributedContext


class TestDDPInit:
    def test_training_loop_accepts_dist_context(self):
        """KataGoTrainingLoop accepts a DistributedContext."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
        assert loop.dist_ctx is ctx
        assert loop.dist_ctx.is_main is True

    def test_non_distributed_backward_compatible(self):
        """Omitting dist_ctx gives a non-distributed context."""
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        assert loop.dist_ctx.is_distributed is False
        assert loop.dist_ctx.world_size == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestDDPInit -v`
Expected: FAIL with `TypeError: KataGoTrainingLoop.__init__() got an unexpected keyword argument 'dist_ctx'`

- [ ] **Step 3: Modify KataGoTrainingLoop.__init__ to accept and use DistributedContext**

In `keisei/training/katago_loop.py`, add imports at top:

```python
from keisei.training.distributed import DistributedContext, get_distributed_context, broadcast_string
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
```

Modify the constructor signature (line 156):

```python
def __init__(
    self, config: AppConfig, vecenv: Any = None,
    resume_mode: str = "rl",
    dist_ctx: DistributedContext | None = None,
) -> None:
```

Early in `__init__`, after `self.config = config` (line 160), replace the device detection block (line 165):

```python
        self.dist_ctx = dist_ctx or get_distributed_context()
        self.device = self.dist_ctx.device
```

Replace the existing multi-GPU logging block (lines 181-189) with DDP wrapping:

```python
        if self.dist_ctx.is_distributed:
            if config.distributed.sync_batchnorm:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
                logger.info("Converted BatchNorm layers to SyncBatchNorm")
            self.model = DDP(
                self.model,
                device_ids=[self.dist_ctx.local_rank] if self.device.type == "cuda" else None,
                output_device=self.dist_ctx.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=config.distributed.find_unused_parameters,
                gradient_as_bucket_view=config.distributed.gradient_as_bucket_view,
            )
            logger.info(
                "DDP wrapping complete: rank=%d, world_size=%d",
                self.dist_ctx.rank, self.dist_ctx.world_size,
            )
        else:
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                logger.info(
                    "Found %d GPUs; DataParallel skipped for KataGo (use DDP instead)",
                    gpu_count,
                )
```

Update the param_count logging (lines 191-199) to use `_base_model`:

```python
        param_count = sum(p.numel() for p in self._base_model.parameters())
        logger.info(
            "Model: %s (%s), params: %d, device: %s, world_size: %d",
            config.model.display_name,
            config.model.architecture,
            param_count,
            self.device,
            self.dist_ctx.world_size,
        )
```

The PPO init (line 219-222) already uses `self._base_model` and `forward_model=self.model`, which is exactly right for DDP — `self._base_model` unwraps the DDP wrapper, and `self.model` (the DDP wrapper) is the forward model. No changes needed here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: PASS (both new and all existing tests)

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat(ddp): wire DistributedContext into KataGoTrainingLoop init"
```

---

### Task 4: Rank-gate I/O operations and synchronize LR scheduler

**Files:**
- Modify: `keisei/training/katago_loop.py:376-635` (the `run()` method)
- Modify: `keisei/training/katago_loop.py:674-713` (heartbeat and snapshot methods)

**Operations that should ONLY run on rank 0:**
- `write_metrics()` (line 598)
- `update_training_progress()` (line 606)
- Console logging of per-epoch losses (line 610-614)
- `save_checkpoint()` (line 619-625) — with `dist.barrier()` before and after
- `_maybe_write_snapshots()` (line 491)
- `_maybe_update_heartbeat()` (line 492, 512) — rank-gated to avoid DB writes inside NCCL collective windows
- DB writes for Elo tracking (lines 548-568)
- League pool snapshots (lines 578-585)

**Operations that ALL ranks must do:**
- Rollout collection (each rank has its own VecEnv)
- PPO update (DDP synchronizes gradients automatically)
- LR scheduler step — but with **synchronized monitor value** (see below)
- Buffer operations

**Critical: LR scheduler synchronization.** `ReduceLROnPlateau.step(value_loss)` must receive the same `value_loss` on all ranks. Otherwise LR diverges across ranks after the first plateau trigger, causing different update magnitudes despite averaged gradients. Fix: all-reduce the monitor value before calling `scheduler.step()`.

- [ ] **Step 1: Write failing test for rank-gated behavior**

Add to `tests/test_katago_loop.py`:

```python
class TestRankGating:
    def test_non_main_rank_skips_checkpoint(self):
        """Non-main rank should not write checkpoints."""
        ctx = DistributedContext(rank=1, local_rank=1, world_size=2, is_distributed=True)
        config = _make_config()
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, checkpoint_interval=1),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.init_db"):
            loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)

        with patch("keisei.training.katago_loop.save_checkpoint") as mock_save:
            loop.run(num_epochs=1, steps_per_epoch=2)
            mock_save.assert_not_called()

    def test_main_rank_writes_checkpoint(self):
        """Main rank should write checkpoints normally."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        config = _make_config()
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, checkpoint_interval=1),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)

        with patch("keisei.training.katago_loop.save_checkpoint") as mock_save:
            loop.run(num_epochs=1, steps_per_epoch=2)
            assert mock_save.call_count >= 1

    def test_non_main_rank_skips_metrics(self):
        """Non-main rank should not write metrics to DB."""
        ctx = DistributedContext(rank=1, local_rank=1, world_size=2, is_distributed=True)
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.init_db"):
            loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)

        with patch("keisei.training.katago_loop.write_metrics") as mock_write:
            loop.run(num_epochs=1, steps_per_epoch=2)
            mock_write.assert_not_called()
```

Note: `test_non_main_rank_skips_checkpoint` patches `init_db` because Task 5 will gate DB init (the tests are designed to work in sequence).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestRankGating -v`
Expected: FAIL (non-main rank still writes checkpoint/metrics)

- [ ] **Step 3: Add rank gates and LR synchronization to run() method**

In `keisei/training/katago_loop.py`, wrap the metrics/logging block (lines 587-614) with a rank check:

```python
            if self.dist_ctx.is_main:
                # Metrics and logging (rank 0 only — SQLite is single-writer)
                ep_completed = getattr(self.vecenv, "episodes_completed", 0)
                metrics = {
                    "epoch": epoch_i, "step": self.global_step,
                    "policy_loss": losses["policy_loss"],
                    "value_loss": losses["value_loss"],
                    "entropy": losses["entropy"],
                    "gradient_norm": losses["gradient_norm"],
                    "episodes_completed": ep_completed,
                }
                try:
                    write_metrics(self.db_path, metrics)
                except Exception:
                    logger.exception("Failed to write metrics for epoch %d — continuing", epoch_i)

                if hasattr(self.vecenv, "reset_stats"):
                    self.vecenv.reset_stats()

                try:
                    update_training_progress(self.db_path, epoch_i, self.global_step)
                except Exception:
                    logger.exception("Failed to update training progress — continuing")

                logger.info(
                    "Epoch %d | step %d | policy=%.4f value=%.4f score=%.4f entropy=%.4f",
                    epoch_i, self.global_step, losses["policy_loss"],
                    losses["value_loss"], losses["score_loss"], losses["entropy"],
                )
```

**Synchronize LR scheduler across ranks** (replace existing lines 521-535). The monitor value must be all-reduced before stepping the scheduler so all ranks see the same value and maintain identical LR state:

```python
            # LR scheduler logic — ALL ranks must participate with the SAME
            # monitor value to keep LR state synchronized across ranks.
            if epoch_i == self.ppo.warmup_epochs and self.lr_scheduler is not None:
                self.lr_scheduler.best = self.lr_scheduler.mode_worse
                self.lr_scheduler.num_bad_epochs = 0
                if self.dist_ctx.is_main:
                    logger.info("LR scheduler fully reset at warmup boundary (epoch %d)", epoch_i)

            if self.lr_scheduler is not None:
                monitor_value = losses.get("value_loss")
                if monitor_value is not None:
                    # Synchronize monitor value across ranks so all schedulers
                    # step identically and maintain the same LR state.
                    if self.dist_ctx.is_distributed:
                        monitor_tensor = torch.tensor(
                            monitor_value, device=self.device,
                        )
                        dist.all_reduce(monitor_tensor, op=dist.ReduceOp.AVG)
                        monitor_value = monitor_tensor.item()

                    old_lr = self.ppo.optimizer.param_groups[0]["lr"]
                    self.lr_scheduler.step(monitor_value)
                    new_lr = self.ppo.optimizer.param_groups[0]["lr"]
                    if new_lr != old_lr and self.dist_ctx.is_main:
                        logger.info("LR reduced: %.6f -> %.6f (value_loss=%.4f)",
                                    old_lr, new_lr, monitor_value)
```

Wrap Elo tracking and counter materialization (lines 537-568). Non-main ranks still flush GPU tensors to avoid accumulating unflushed data:

```python
            # Materialise GPU counters → CPU once per epoch (all ranks, to release GPU memory)
            win_count = win_acc.item()
            loss_count = loss_acc.item()
            draw_count = draw_acc.item()

            if self.dist_ctx.is_main:
                # Elo tracking (league mode, rank 0 only)
                total_games = win_count + loss_count + draw_count
                if (self.pool is not None and self._current_opponent_entry is not None
                        and total_games > 0):
                    # ... rest of Elo block unchanged
```

**Add `dist.barrier()` around checkpoint saves** (replace lines 616-634):

```python
            if (epoch_i + 1) % self.config.training.checkpoint_interval == 0:
                # Barrier ensures all ranks finish the PPO update before rank 0
                # writes the checkpoint. Without this, a crash during the write
                # leaves a partial checkpoint while other ranks have moved on.
                if self.dist_ctx.is_distributed:
                    dist.barrier()

                if self.dist_ctx.is_main:
                    ckpt_path = Path(self.config.training.checkpoint_dir) / f"epoch_{epoch_i:05d}.pt"
                    try:
                        save_checkpoint(
                            ckpt_path, self._base_model, self.ppo.optimizer,
                            epoch_i + 1, self.global_step,
                            architecture=self.config.model.architecture,
                            scheduler=self.lr_scheduler,
                            grad_scaler=self.ppo.scaler,
                            world_size=self.dist_ctx.world_size,
                        )
                        logger.info("Checkpoint saved: %s", ckpt_path)
                    except Exception:
                        logger.exception("Failed to save checkpoint %s — continuing", ckpt_path)
                    try:
                        update_training_progress(
                            self.db_path, epoch_i + 1, self.global_step, str(ckpt_path),
                        )
                    except Exception:
                        logger.exception("Failed to record checkpoint path in DB — continuing")

                # Barrier after save — all ranks proceed to next epoch together
                if self.dist_ctx.is_distributed:
                    dist.barrier()
```

Wrap seat rotation and pool snapshot (lines 570-585):

```python
            if self.dist_ctx.is_main:
                # Seat rotation and pool snapshots (rank 0 only — shared state)
                rotating_this_epoch = (
                    self.config.league is not None and self.pool is not None
                    and (epoch_i + 1) % self.config.league.epochs_per_seat == 0
                )
                if rotating_this_epoch:
                    self._rotate_seat(epoch_i)

                if (self.pool is not None and self.config.league is not None
                        and (epoch_i + 1) % self.config.league.snapshot_interval == 0
                        and not rotating_this_epoch):
                    self.pool.add_snapshot(
                        self._base_model, self.config.model.architecture,
                        dict(self.config.model.params), epoch=epoch_i + 1,
                    )
```

In `_maybe_update_heartbeat` (line 674), add rank gate:

```python
    def _maybe_update_heartbeat(self) -> None:
        if not self.dist_ctx.is_main:
            return
        now = time.monotonic()
        # ... rest unchanged
```

In `_maybe_write_snapshots` (line 682), add rank gate:

```python
    def _maybe_write_snapshots(self) -> None:
        if not self.dist_ctx.is_main:
            return
        if self.moves_per_minute <= 0:
            return
        # ... rest unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat(ddp): rank-gate I/O, synchronize LR scheduler, barrier around checkpoints"
```

---

### Task 5: Rank-gate DB init and fix `_check_resume` for all-rank checkpoint loading

**Files:**
- Modify: `keisei/training/katago_loop.py:155-375`

**Critical correctness fix:** On resume, ALL ranks must load the checkpoint — not just rank 0. DDP does NOT re-broadcast weights after `DDP.__init__()`. If rank 0 loads a checkpoint and ranks 1..N don't, they train from diverged weights. The gradient allreduce averages nonsensical gradients and training is silently corrupted.

**Approach:** Rank 0 reads the checkpoint path from the DB, then broadcasts the path string to all ranks via `broadcast_string()`. All ranks then independently call `load_checkpoint()`. Only rank 0 writes the "fresh start" training state.

- [ ] **Step 1: Write failing test for DB init gating**

Add to `tests/test_katago_loop.py`:

```python
class TestDDPDBInit:
    def test_non_main_rank_skips_db_init(self):
        """Non-main rank should not call init_db."""
        ctx = DistributedContext(rank=1, local_rank=1, world_size=2, is_distributed=True)
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.init_db") as mock_init:
            loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
            mock_init.assert_not_called()

    def test_main_rank_calls_db_init(self):
        """Main rank should call init_db normally."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.init_db") as mock_init:
            loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
            mock_init.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestDDPDBInit -v`
Expected: FAIL (non-main rank still calls init_db)

- [ ] **Step 3: Add rank gates to __init__ and fix _check_resume**

In `__init__` (line 163), wrap `init_db`:

```python
        if self.dist_ctx.is_main:
            init_db(self.db_path)
```

Replace `_check_resume` (lines 317-374) with a DDP-safe version. The key change: rank 0 reads the DB to find the checkpoint path, then ALL ranks load the checkpoint:

```python
    def _check_resume(self) -> None:
        # NOTE: When resuming from an SL checkpoint into RL training, the SL
        # optimizer state is intentionally discarded. See original comment for details.

        # Rank 0 reads the DB to find checkpoint path; non-main ranks get None.
        checkpoint_path_str: str | None = None
        current_epoch: int = 0
        if self.dist_ctx.is_main:
            state = read_training_state(self.db_path)
            if state is not None and state.get("checkpoint_path"):
                cp = Path(state["checkpoint_path"])
                if cp.exists():
                    checkpoint_path_str = str(cp)
                    current_epoch = state.get("current_epoch", 0)

        # Broadcast checkpoint path to all ranks so everyone loads the same checkpoint.
        # broadcast_string is a no-op in non-distributed mode.
        if self.dist_ctx.is_distributed:
            # Use broadcast_object_list: rank 0 sends the path (or None), all ranks receive.
            obj_list = [checkpoint_path_str]
            dist.broadcast_object_list(obj_list, src=0)
            checkpoint_path_str = obj_list[0]

            # Also broadcast epoch/step so non-main ranks know where to resume
            meta_list: list[object] = [current_epoch]
            dist.broadcast_object_list(meta_list, src=0)
            current_epoch = meta_list[0]  # type: ignore[assignment]

        # ALL ranks load the checkpoint (critical for DDP weight consistency)
        if checkpoint_path_str is not None:
            checkpoint_path = Path(checkpoint_path_str)
            skip_opt = self._resume_mode == "sl"
            logger.warning(
                "[rank %d] Resuming from checkpoint: %s (skip_optimizer=%s)",
                self.dist_ctx.rank, checkpoint_path, skip_opt,
            )
            meta = load_checkpoint(
                checkpoint_path,
                self._base_model,
                self.ppo.optimizer,
                expected_architecture=self.config.model.architecture,
                scheduler=self.lr_scheduler,
                grad_scaler=self.ppo.scaler,
                skip_optimizer=skip_opt,
            )
            if skip_opt:
                self.epoch = 0
                self.global_step = 0
            else:
                self.epoch = meta["epoch"]
                self.global_step = meta["step"]
            return

        # Fresh start — only rank 0 writes training state to DB
        if self.dist_ctx.is_main:
            write_training_state(
                self.db_path,
                {
                    "config_json": json.dumps(
                        {
                            "training": {
                                "num_games": self.config.training.num_games,
                                "algorithm": self.config.training.algorithm,
                            },
                            "model": {"architecture": self.config.model.architecture},
                        }
                    ),
                    "display_name": self.config.model.display_name,
                    "model_arch": self.config.model.architecture,
                    "algorithm_name": self.config.training.algorithm,
                    "started_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                },
            )
        else:
            logger.info(
                "[rank %d] Non-main rank: skipping DB write for fresh training state",
                self.dist_ctx.rank,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat(ddp): all-rank checkpoint resume via broadcast, rank-gate DB init"
```

---

### Task 6: Update checkpoint save/load for DDP — per-rank RNG and world_size guard

**Files:**
- Modify: `keisei/training/checkpoint.py:42-72` (save_checkpoint)
- Modify: `keisei/training/checkpoint.py:75-136` (load_checkpoint)
- Test: `tests/test_katago_checkpoint.py`

Three changes:
1. Add `world_size` metadata to checkpoints
2. Replace `get_rng_state_all()` / `set_rng_state_all()` with per-device variants — the "all" variants save/restore ALL visible GPUs which crashes when resuming on a different GPU count
3. Add world_size mismatch warning on load

- [ ] **Step 1: Write failing tests**

Add to `tests/test_katago_checkpoint.py`:

```python
class TestDDPCheckpoint:
    def test_save_records_world_size(self, tmp_path):
        """Checkpoint records world_size for DDP awareness."""
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "ckpt.pt"

        save_checkpoint(path, model, optimizer, epoch=1, step=100, world_size=4)

        ckpt = torch.load(path, weights_only=True)
        assert ckpt["world_size"] == 4

    def test_save_defaults_world_size_1(self, tmp_path):
        """Without world_size arg, defaults to 1."""
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "ckpt.pt"

        save_checkpoint(path, model, optimizer, epoch=1, step=100)

        ckpt = torch.load(path, weights_only=True)
        assert ckpt["world_size"] == 1

    def test_load_warns_on_world_size_mismatch(self, tmp_path, caplog):
        """Loading a checkpoint with different world_size logs a warning."""
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.Adam(model.parameters())
        path = tmp_path / "ckpt.pt"

        save_checkpoint(path, model, optimizer, epoch=1, step=100, world_size=4)
        load_checkpoint(path, model, optimizer, current_world_size=2)
        assert "world_size mismatch" in caplog.text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_checkpoint.py::TestDDPCheckpoint -v`
Expected: FAIL

- [ ] **Step 3: Update save_checkpoint and load_checkpoint**

In `keisei/training/checkpoint.py`, modify `save_checkpoint` signature (line 42):

```python
def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    architecture: str | None = None,
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
    world_size: int = 1,
) -> None:
```

In the data dict (after line 57), add `world_size` and replace the CUDA RNG state:

```python
        "world_size": world_size,
```

Replace the CUDA RNG state section (lines 64-65):

```python
    # Save per-device CUDA RNG state (not get_rng_state_all, which saves ALL
    # visible GPUs and crashes on resume if GPU count changes).
    if torch.cuda.is_available():
        data["rng_states"]["torch_cuda"] = torch.cuda.get_rng_state()
```

Add `import logging` at the top of checkpoint.py and create a logger:

```python
import logging
logger = logging.getLogger(__name__)
```

Modify `load_checkpoint` signature (line 75):

```python
def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_architecture: str | None = None,
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
    skip_optimizer: bool = False,
    current_world_size: int = 1,
) -> dict[str, Any]:
```

After loading the checkpoint (after line 95), add world_size mismatch warning:

```python
    ckpt_world_size = checkpoint.get("world_size", 1)
    if ckpt_world_size != current_world_size:
        logger.warning(
            "World_size mismatch: checkpoint was saved with world_size=%d "
            "but current world_size=%d. torch.compile may recompile; "
            "CUDA RNG state from checkpoint will not match.",
            ckpt_world_size, current_world_size,
        )
```

Replace the CUDA RNG restore (lines 133-134):

```python
        # Restore per-device CUDA RNG state. Handles both old checkpoints
        # (that saved get_rng_state_all() as a list) and new checkpoints
        # (that save get_rng_state() as a single tensor).
        if "torch_cuda" in rng and torch.cuda.is_available():
            cuda_state = rng["torch_cuda"]
            if isinstance(cuda_state, list):
                # Legacy checkpoint from get_rng_state_all — use first element
                if len(cuda_state) > 0:
                    torch.cuda.set_rng_state(cuda_state[0])
            else:
                torch.cuda.set_rng_state(cuda_state)
```

Update the checkpoint save call in `katago_loop.py` (line 619) to pass both `world_size` and update the load call in `_check_resume` to pass `current_world_size`:

In `save_checkpoint` call:
```python
                        world_size=self.dist_ctx.world_size,
```

In `load_checkpoint` call in `_check_resume`:
```python
            meta = load_checkpoint(
                checkpoint_path,
                self._base_model,
                self.ppo.optimizer,
                expected_architecture=self.config.model.architecture,
                scheduler=self.lr_scheduler,
                grad_scaler=self.ppo.scaler,
                skip_optimizer=skip_opt,
                current_world_size=self.dist_ctx.world_size,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_checkpoint.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/checkpoint.py keisei/training/katago_loop.py tests/test_katago_checkpoint.py
git commit -m "feat(ddp): per-rank CUDA RNG state, world_size metadata and mismatch guard"
```

---

### Task 7: Update `main()` entry point for torchrun

**Files:**
- Modify: `keisei/training/katago_loop.py:716-738`

The entry point needs to:
1. Detect distributed environment via `get_distributed_context()`
2. Call `setup_distributed()` before any model/CUDA operations
3. Call `cleanup_distributed()` on exit
4. Set ALL random seeds per-rank (torch, numpy, Python stdlib) for reproducible but different rollouts

- [ ] **Step 1: Write failing test for entry point DDP awareness**

Add to `tests/test_katago_loop.py`:

```python
from keisei.training.distributed import setup_distributed, cleanup_distributed


class TestMainEntryPoint:
    def test_main_calls_setup_and_cleanup(self):
        """main() should call setup_distributed and cleanup_distributed."""
        with patch("keisei.training.katago_loop.get_distributed_context") as mock_ctx, \
             patch("keisei.training.katago_loop.setup_distributed") as mock_setup, \
             patch("keisei.training.katago_loop.cleanup_distributed") as mock_cleanup, \
             patch("keisei.training.katago_loop.seed_all_ranks") as mock_seed, \
             patch("keisei.training.katago_loop.load_config"), \
             patch("keisei.training.katago_loop.KataGoTrainingLoop") as mock_loop, \
             patch("sys.argv", ["prog", "dummy.toml", "--epochs", "1"]):
            mock_ctx.return_value = DistributedContext(
                rank=0, local_rank=0, world_size=1, is_distributed=False,
            )
            mock_loop.return_value.run = MagicMock()

            from keisei.training.katago_loop import main
            main()

            mock_setup.assert_called_once()
            mock_cleanup.assert_called_once()
            mock_seed.assert_called_once_with(42)  # base_seed + rank 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestMainEntryPoint -v`
Expected: FAIL

- [ ] **Step 3: Update main() function**

Replace `main()` in `keisei/training/katago_loop.py` (lines 716-738). Add imports for `setup_distributed`, `cleanup_distributed`, and `seed_all_ranks`:

```python
from keisei.training.distributed import (
    DistributedContext, get_distributed_context,
    broadcast_string, setup_distributed, cleanup_distributed, seed_all_ranks,
)
```

```python
def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dist_ctx = get_distributed_context()
    setup_distributed(dist_ctx)

    try:
        parser = argparse.ArgumentParser(description="Keisei training")
        parser.add_argument("config", type=Path, help="Path to TOML config file")
        parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
        parser.add_argument("--steps-per-epoch", type=int, default=None,
                            help="Steps per epoch (default: max_ply from config)")
        parser.add_argument("--seed", type=int, default=42,
                            help="Base random seed (each rank adds its rank index)")
        args = parser.parse_args()

        # Set all RNG seeds: base_seed + rank for different-but-reproducible rollouts.
        # Seeds torch, numpy, and Python stdlib RNG.
        seed_all_ranks(args.seed + dist_ctx.rank)

        from keisei.config import load_config
        config = load_config(args.config)
        steps = args.steps_per_epoch or config.training.max_ply
        loop = KataGoTrainingLoop(config, dist_ctx=dist_ctx)
        loop.run(num_epochs=args.epochs, steps_per_epoch=steps)
    finally:
        cleanup_distributed(dist_ctx)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat(ddp): update main() for torchrun with full RNG seeding"
```

---

### Task 8: League mode compatibility guard

**Files:**
- Modify: `keisei/training/katago_loop.py:281-310` (league setup in __init__)

League mode (opponent pool, Elo tracking, seat rotation, split-merge) is incompatible with DDP for two reasons:
1. **Shared state:** Opponent pool, Elo tracking, and SQLite DB assume single-process access.
2. **Buffer size deadlock:** In split-merge mode, the learner/opponent split varies per step. Different ranks may collect different numbers of learner transitions, resulting in different `total_samples`. The PPO `update()` does `torch.randperm(total_samples)` to generate mini-batches. If `total_samples` differs across ranks, the number of `loss.backward()` calls differs, causing an NCCL allreduce deadlock.

- [ ] **Step 1: Write failing test for league+DDP guard**

Add to `tests/test_katago_loop.py`:

```python
class TestDDPLeagueGuard:
    def test_league_with_ddp_raises(self):
        """League mode is not yet supported with DDP."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=2, is_distributed=True)
        config = _make_config()
        config = dataclasses.replace(config, league=LeagueConfig())
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with pytest.raises(ValueError, match="League mode.*not.*supported.*DDP"):
            KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)

    def test_league_without_ddp_ok(self):
        """League mode works fine without DDP."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        config = _make_config()
        config = dataclasses.replace(config, league=LeagueConfig())
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
        assert loop.pool is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestDDPLeagueGuard -v`
Expected: FAIL (no guard exists yet)

- [ ] **Step 3: Add the guard**

In `keisei/training/katago_loop.py`, before the league setup block (line 288), add:

```python
        if config.league is not None and self.dist_ctx.is_distributed:
            raise ValueError(
                "League mode is not yet supported with DDP. "
                "League mode uses split-merge rollout collection where buffer sizes "
                "can differ across ranks, causing NCCL allreduce deadlocks. "
                "Run without torchrun or remove [league] config."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat(ddp): guard against league mode with DDP (buffer size deadlock risk)"
```

---

### Task 9: Integration test — 2-rank training on CPU backend

**Files:**
- Create: `tests/integration/test_ddp_training.py`

This test spawns 2 processes using `torch.multiprocessing.spawn` with the `gloo` CPU backend to verify end-to-end DDP training works. It runs for 1 epoch with 4 steps.

**Important:** The test calls `setup_distributed(ctx, backend="gloo")` rather than `dist.init_process_group` directly, so it exercises the actual production code path.

**Important:** `compile_mode` is left as None (no torch.compile) because compiled graphs with SyncBN embed NCCL collectives that would deadlock on gloo.

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_ddp_training.py
"""Integration test: DDP training with 2 ranks on CPU (gloo backend)."""

import os
import socket
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from keisei.config import (
    AppConfig, DisplayConfig, DistributedConfig, ModelConfig, TrainingConfig,
)
from keisei.training.distributed import (
    DistributedContext, setup_distributed, cleanup_distributed, seed_all_ranks,
)
from keisei.training.katago_loop import KataGoTrainingLoop


def _find_free_port() -> int:
    """Find a free TCP port for MASTER_PORT to avoid collisions in CI."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_mock_vecenv(num_envs: int = 2, seed: int = 42) -> MagicMock:
    """Minimal mock VecEnv for DDP integration test.

    Each call to step() returns a fresh result with new random observations.
    """
    rng = np.random.default_rng(seed)
    mock = MagicMock()
    mock.observation_channels = 50
    mock.action_space_size = 11259
    mock.episodes_completed = 0
    mock.mean_episode_length = 0.0
    mock.truncation_rate = 0.0

    def make_result():
        result = MagicMock()
        result.observations = rng.standard_normal((num_envs, 50, 9, 9)).astype(np.float32)
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        result.rewards = np.zeros(num_envs, dtype=np.float32)
        result.terminated = np.zeros(num_envs, dtype=bool)
        result.truncated = np.zeros(num_envs, dtype=bool)
        result.current_players = np.zeros(num_envs, dtype=np.uint8)
        result.step_metadata = MagicMock()
        result.step_metadata.material_balance = np.zeros(num_envs, dtype=np.int32)
        return result

    # Each call to reset/step produces a fresh result
    mock.reset.side_effect = lambda: make_result()
    mock.step.side_effect = lambda actions: make_result()
    mock.reset_stats = MagicMock()
    return mock


def _worker(rank: int, world_size: int, tmp_dir: str, port: int) -> None:
    """Worker function for each DDP rank."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    ctx = DistributedContext(
        rank=rank, local_rank=rank, world_size=world_size, is_distributed=True,
    )

    # Use the actual setup_distributed function with gloo backend
    setup_distributed(ctx, backend="gloo")

    try:
        # Seed per rank (mirrors what main() does)
        seed_all_ranks(42 + rank)

        config = AppConfig(
            training=TrainingConfig(
                num_games=2, max_ply=500, algorithm="katago_ppo",
                checkpoint_interval=1, checkpoint_dir=str(Path(tmp_dir) / "ckpt"),
                algorithm_params={
                    "learning_rate": 1e-3, "epochs_per_batch": 1, "batch_size": 4,
                },
                use_amp=False,
            ),
            display=DisplayConfig(
                moves_per_minute=0,
                db_path=str(Path(tmp_dir) / f"test_rank{rank}.db"),
            ),
            model=ModelConfig(
                display_name="TestDDP", architecture="se_resnet",
                params={"num_blocks": 2, "channels": 32, "obs_channels": 50},
            ),
            distributed=DistributedConfig(sync_batchnorm=False),
        )

        # Each rank gets its own VecEnv with a different seed
        mock_env = _make_mock_vecenv(num_envs=2, seed=42 + rank)
        loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
        loop.run(num_epochs=1, steps_per_epoch=4)

        # Verify: all ranks should have the same model weights after training
        # (DDP guarantees synchronized gradients)
        state = {k: v.cpu() for k, v in loop._base_model.state_dict().items()}
        torch.save(state, Path(tmp_dir) / f"weights_rank{rank}.pt")

    finally:
        cleanup_distributed(ctx)


@pytest.mark.slow
def test_ddp_two_ranks():
    """End-to-end: 2-rank DDP training produces synchronized weights."""
    port = _find_free_port()
    with tempfile.TemporaryDirectory() as tmp_dir:
        mp.spawn(_worker, args=(2, tmp_dir, port), nprocs=2, join=True)

        # Load weights from both ranks and verify they're identical
        w0 = torch.load(Path(tmp_dir) / "weights_rank0.pt", weights_only=True)
        w1 = torch.load(Path(tmp_dir) / "weights_rank1.pt", weights_only=True)

        for key in w0:
            torch.testing.assert_close(
                w0[key], w1[key],
                msg=f"Weight mismatch on parameter '{key}' between rank 0 and rank 1",
            )

        # Verify rank 0 wrote a checkpoint
        ckpt_dir = Path(tmp_dir) / "ckpt"
        ckpts = list(ckpt_dir.glob("epoch_*.pt"))
        assert len(ckpts) >= 1, "Rank 0 should have saved at least one checkpoint"

        # Verify checkpoint contains world_size
        ckpt = torch.load(ckpts[0], weights_only=True)
        assert ckpt["world_size"] == 2
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/integration/test_ddp_training.py -v -m slow --timeout=120`
Expected: PASS (2 ranks train and produce synchronized weights)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_ddp_training.py
git commit -m "test(ddp): add 2-rank integration test with gloo backend"
```

---

### Task 10: Documentation and TOML config example

**Files:**
- Create: `configs/ddp_example.toml` (example config for DDP training)

- [ ] **Step 1: Check if configs directory exists**

Run: `ls configs/ 2>/dev/null || mkdir -p configs/`

- [ ] **Step 2: Create example TOML config**

```toml
# configs/ddp_example.toml
# Example config for multi-GPU DDP training.
#
# Launch with torchrun (preferred — portable across environments):
#   torchrun --nproc_per_node=<NUM_GPUS> -m keisei.training.katago_loop configs/ddp_example.toml
#
# DDP activation is automatic when launched via torchrun (sets RANK, LOCAL_RANK,
# WORLD_SIZE env vars). No "enabled" flag is needed in the config. Running this
# config with plain `python` instead of `torchrun` trains on a single GPU.
#
# Each rank gets its own VecEnv instances. Total parallel environments =
# num_games × num_gpus. With num_games=32 and 4 GPUs, you get 128 total envs.
#
# global_step logged by rank 0 counts rank-0's steps only. Total environment
# steps per epoch = steps_per_epoch × world_size.

[training]
num_games = 32          # per-rank (total envs = num_games * num_gpus)
max_ply = 500
algorithm = "katago_ppo"
checkpoint_interval = 50
checkpoint_dir = "checkpoints/ddp"
use_amp = true

[training.algorithm_params]
learning_rate = 2e-4
batch_size = 256        # per-rank mini-batch size
epochs_per_batch = 4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
lambda_entropy = 0.01

[display]
moves_per_minute = 30
db_path = "keisei.db"

[model]
display_name = "KataGo-DDP"
architecture = "se_resnet"

[model.params]
num_blocks = 20
channels = 128
obs_channels = 50

# [distributed] section controls DDP behavior WHEN launched via torchrun.
# Omit this section entirely to use defaults (sync_batchnorm=true, etc.).
[distributed]
sync_batchnorm = true               # synchronized BN across GPUs (recommended for small per-GPU batches)
find_unused_parameters = false       # set true only if model has conditionally-unused parameters
gradient_as_bucket_view = true       # memory optimization for gradient communication

# NOTE: League mode ([league] section) is NOT supported with DDP.
# Split-merge rollout collection can produce unequal buffer sizes across ranks,
# causing NCCL allreduce deadlocks. Use single-GPU for league training.
```

- [ ] **Step 3: Commit**

```bash
git add configs/ddp_example.toml
git commit -m "docs(ddp): add example TOML config for multi-GPU training"
```

---

### Task 11: Run full test suite and verify no regressions

**Files:** None (verification only)

- [ ] **Step 1: Run all existing tests**

Run: `uv run pytest -x -v`
Expected: All existing tests PASS (no regressions from DDP changes)

- [ ] **Step 2: Run type check if available**

Run: `uv run mypy keisei/training/distributed.py keisei/config.py keisei/training/katago_loop.py keisei/training/checkpoint.py --ignore-missing-imports`
Expected: No type errors

- [ ] **Step 3: Run the DDP integration test specifically**

Run: `uv run pytest tests/integration/test_ddp_training.py -v -m slow --timeout=120`
Expected: PASS

- [ ] **Step 4: Commit any fixes**

If any tests needed fixes, commit them:

```bash
git add -u
git commit -m "fix(ddp): address test regressions from DDP integration"
```

---

## Design Decisions

### Why independent rollouts per rank (not centralized collection)?

Each rank runs its own `VecEnv` and collects its own rollouts. This is simpler and more scalable than having rank 0 collect all data and broadcast:
- **Linear scaling:** 4 GPUs = 4x the experience collected per epoch
- **No communication bottleneck** during rollout phase (only gradient sync during update)
- **Each rank is self-contained** during collection — no distributed coordination needed
- **PPO is on-policy** — each rank's rollouts are immediately consumed and discarded

The PPO clip ratio is computed per-rank (each rank's `batch_old_log_probs` come from its own rollout). This is correct — it's equivalent to having a larger rollout buffer. The gradient averaging across ranks averages PPO updates from different game trajectories, which increases effective sample diversity. Do NOT "fix" this by synchronizing `torch.randperm` across ranks.

### Why guard against league mode?

League mode involves:
- Shared opponent pool with snapshots on disk
- Elo tracking in a shared SQLite database
- Seat rotation that resets the optimizer
- **Split-merge rollout collection** where buffer sizes can differ across ranks

The buffer size issue is the primary technical blocker: in split-merge mode, the learner/opponent split varies per step. Different ranks may collect different numbers of learner transitions, giving different `total_samples`. The PPO `update()` iterates `range(0, total_samples, batch_size)`, producing different numbers of `loss.backward()` calls across ranks. DDP's NCCL allreduce requires all ranks to call `backward()` the same number of times — a mismatch causes a permanent hang.

### Why not use DistributedSampler?

PPO doesn't have a static dataset. The rollout buffer is filled fresh each epoch. With independent VecEnvs per rank, each rank already has unique data. The mini-batch sampling in `update()` uses `torch.randperm` which produces different permutations per rank (different seeds), but this is fine — DDP averages gradients across ranks regardless of data ordering.

### SyncBatchNorm trade-off

The SE-ResNet model uses BatchNorm. With DDP, each rank computes BN statistics on its own mini-batch, which can cause divergence with small per-GPU batch sizes. `SyncBatchNorm` synchronizes statistics across ranks at the cost of one allreduce per BN layer per forward pass. We make it configurable (default: on) and let the user disable it if their per-GPU batch size is large enough (>= 32).

### Why all ranks must load checkpoint on resume

DDP's `__init__()` broadcasts parameters from rank 0 to all ranks, but this only happens once during wrapping. After `_check_resume` loads a checkpoint on rank 0 (mutating the base model in-place), DDP does NOT re-broadcast. Other ranks keep their post-`DDP.__init__()` weights. To ensure all ranks have identical weights, optimizer state, LR scheduler state, and epoch counters, all ranks must independently call `load_checkpoint`.

### LR scheduler synchronization

`ReduceLROnPlateau.step(value_loss)` accumulates internal state (`num_bad_epochs`, `best`). If each rank feeds its own local `value_loss`, the internal state diverges. After the first rank triggers a reduction, ranks have different learning rates. Since Adam multiplies the averaged gradient by the per-rank LR, update magnitudes become inconsistent across ranks. The fix is a single `dist.all_reduce(monitor_value, op=AVG)` before stepping.

### Per-device CUDA RNG state (not get_rng_state_all)

`torch.cuda.get_rng_state_all()` returns states for ALL visible GPUs. In DDP with torchrun, each rank may see only its assigned GPU (1 element) or all GPUs (N elements) depending on `CUDA_VISIBLE_DEVICES`. On resume, `set_rng_state_all()` requires the same list length — a mismatch crashes. Using the single-device `get_rng_state()` / `set_rng_state()` is correct and portable.

### global_step counting

`global_step` is incremented per-step on each rank independently. Rank 0's logged/checkpointed `global_step` counts rank-0's steps only. Total environment steps per epoch = `steps_per_epoch * world_size`. This is a known limitation documented in the example config. A future improvement could log `global_step * world_size` as total env steps.

### No `enabled` field in DistributedConfig

DDP activation is determined entirely by torchrun environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`). An `enabled: bool` config flag would create a trap: a user could set `enabled=true` without `torchrun` and silently get no DDP, or launch via `torchrun` with `enabled=false` and still get DDP. Removing the flag eliminates the ambiguity.
