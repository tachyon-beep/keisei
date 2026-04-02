# tests/test_amp.py
"""Tests for AMP mixed precision support."""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from keisei.training.checkpoint import save_checkpoint, load_checkpoint


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TestGradScalerCheckpoint:
    def test_scaler_state_round_trip(self, tmp_path: Path) -> None:
        """GradScaler state survives save → load cycle."""
        model = _TinyModel()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler()

        # Simulate a few steps so scaler has non-default state
        scaler._scale = torch.tensor(32768.0)
        scaler._growth_tracker = torch.tensor(5)

        save_checkpoint(
            tmp_path / "ckpt.pt", model, optimizer,
            epoch=3, step=100, grad_scaler=scaler,
        )

        model2 = _TinyModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        scaler2 = torch.amp.GradScaler()

        load_checkpoint(
            tmp_path / "ckpt.pt", model2, optimizer2, grad_scaler=scaler2,
        )

        assert scaler2.get_scale() == 32768.0
        # _growth_tracker is lazily initialized on CUDA; check the init value
        # which is set by load_state_dict on both CPU and CUDA environments.
        assert scaler2._init_growth_tracker == 5

    def test_load_checkpoint_without_scaler_state(self, tmp_path: Path) -> None:
        """Old checkpoints without scaler state load without error."""
        model = _TinyModel()
        optimizer = torch.optim.Adam(model.parameters())

        # Save without scaler
        save_checkpoint(
            tmp_path / "ckpt.pt", model, optimizer, epoch=1, step=0,
        )

        model2 = _TinyModel()
        optimizer2 = torch.optim.Adam(model2.parameters())
        scaler2 = torch.amp.GradScaler()
        original_scale = scaler2.get_scale()

        # Load with scaler — should not crash, scaler keeps defaults
        load_checkpoint(
            tmp_path / "ckpt.pt", model2, optimizer2, grad_scaler=scaler2,
        )

        assert scaler2.get_scale() == original_scale
