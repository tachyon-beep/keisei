"""Model checkpointing: save and load model + optimizer + training state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    architecture: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if architecture is not None:
        data["architecture"] = architecture
    torch.save(data, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_architecture: str | None = None,
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

    return {"epoch": checkpoint["epoch"], "step": checkpoint["step"]}
