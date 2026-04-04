"""Model checkpointing: save and load model + optimizer + training state."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _numpy_rng_to_safe(state: tuple) -> dict[str, Any]:
    """Convert numpy RNG state to torch-safe types for weights_only loading.

    np.random.get_state() returns ('MT19937', ndarray, int, int, float).
    The ndarray trips torch.load(weights_only=True), so we convert it
    to a torch.IntTensor.
    """
    name, keys, pos, has_gauss, cached_gaussian = state
    return {
        "name": name,
        "keys": torch.from_numpy(keys.copy()),
        "pos": pos,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }


def _safe_to_numpy_rng(d: dict[str, Any]) -> tuple:
    """Inverse of _numpy_rng_to_safe."""
    return (
        d["name"],
        d["keys"].numpy(),
        d["pos"],
        d["has_gauss"],
        d["cached_gaussian"],
    )


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
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "world_size": world_size,
        "rng_states": {
            "python": random.getstate(),
            "numpy": _numpy_rng_to_safe(np.random.get_state()),
            "torch_cpu": torch.random.get_rng_state(),
        },
    }
    # Save per-device CUDA RNG state (not get_rng_state_all, which saves ALL
    # visible GPUs and crashes on resume if GPU count changes).
    if torch.cuda.is_available():
        data["rng_states"]["torch_cuda"] = torch.cuda.get_rng_state()
    if architecture is not None:
        data["architecture"] = architecture
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    if grad_scaler is not None:
        data["grad_scaler_state_dict"] = grad_scaler.state_dict()
    torch.save(data, path)


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
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if expected_architecture is not None:
        ckpt_arch = checkpoint.get("architecture")
        if ckpt_arch is not None and ckpt_arch != expected_architecture:
            raise ValueError(
                f"Checkpoint architecture mismatch: "
                f"expected '{expected_architecture}', got '{ckpt_arch}'"
            )

    ckpt_world_size = checkpoint.get("world_size", 1)
    if ckpt_world_size != current_world_size:
        logger.warning(
            "World_size mismatch: checkpoint was saved with world_size=%d "
            "but current world_size=%d. torch.compile may recompile; "
            "CUDA RNG state from checkpoint will not match.",
            ckpt_world_size, current_world_size,
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if not skip_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Move optimizer state (Adam moment buffers) to match model device.
        # torch.load(map_location="cpu") puts all tensors on CPU, but the model
        # may already be on CUDA. Without this, optimizer.step() crashes with a
        # device mismatch error.
        device = next(model.parameters()).device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # Restore LR scheduler and GradScaler state only when loading full
    # training state.  When skip_optimizer=True (SL→RL transition), the SL
    # scheduler's best/patience tracking and the SL scaler's loss-scale
    # value would poison the fresh RL training state.
    if not skip_optimizer:
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if grad_scaler is not None and "grad_scaler_state_dict" in checkpoint:
            grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])

    # Restore RNG states for reproducible resume (backward-compatible:
    # old checkpoints without rng_states are silently skipped).
    # In distributed training, the checkpoint contains only rank-0's RNG state.
    # Restoring it to all ranks would collapse per-rank stochasticity (each rank
    # should have a different RNG stream seeded with base_seed + rank).  Skip
    # RNG restoration when world_size > 1 so ranks keep their independently
    # seeded streams from setup_distributed / seed_all_ranks.
    rng = checkpoint.get("rng_states")
    if rng is not None and current_world_size <= 1:
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(_safe_to_numpy_rng(rng["numpy"]))
        if "torch_cpu" in rng:
            torch.random.set_rng_state(rng["torch_cpu"])
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
    elif rng is not None:
        logger.info(
            "Skipping RNG state restoration: world_size=%d > 1 — "
            "per-rank RNG streams from seed_all_ranks will be used instead",
            current_world_size,
        )

    return {"epoch": checkpoint["epoch"], "step": checkpoint["step"]}
