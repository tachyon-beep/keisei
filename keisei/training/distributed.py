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
