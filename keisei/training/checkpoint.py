"""Model checkpointing: save and load model + optimizer + training state."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "rng_states": {
            "python": random.getstate(),
            "numpy": _numpy_rng_to_safe(np.random.get_state()),
            "torch_cpu": torch.random.get_rng_state(),
        },
    }
    if torch.cuda.is_available():
        data["rng_states"]["torch_cuda"] = torch.cuda.get_rng_state_all()
    if architecture is not None:
        data["architecture"] = architecture
    if scheduler is not None:
        data["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(data, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_architecture: str | None = None,
    scheduler: Any | None = None,
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

    model.load_state_dict(checkpoint["model_state_dict"])
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

    # Restore LR scheduler state if present in checkpoint and caller provided one.
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG states for reproducible resume (backward-compatible:
    # old checkpoints without rng_states are silently skipped).
    rng = checkpoint.get("rng_states")
    if rng is not None:
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(_safe_to_numpy_rng(rng["numpy"]))
        if "torch_cpu" in rng:
            torch.random.set_rng_state(rng["torch_cpu"])
        if "torch_cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return {"epoch": checkpoint["epoch"], "step": checkpoint["step"]}
