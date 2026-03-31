"""Gap-analysis tests for checkpoint.py: optimizer state restoration."""

from __future__ import annotations

from pathlib import Path

import torch

from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.models.resnet import ResNetModel, ResNetParams


# ===================================================================
# H4 — Optimizer state (momentum buffers, step counts) restored after load
# ===================================================================


class TestOptimizerStateRestoration:
    """load_checkpoint must restore Adam's internal state (momentum buffers,
    step counters), not just model weights."""

    def test_optimizer_state_dict_restored(self, tmp_path: Path) -> None:
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Run a training step to populate Adam's momentum buffers
        obs = torch.randn(2, 46, 9, 9)
        policy, value = model(obs)
        loss = policy.sum() + value.sum()
        loss.backward()
        optimizer.step()

        # Now optimizer.state should be non-empty
        assert len(optimizer.state) > 0, "Adam state should be populated after step"

        # Capture the step count of the first param group's first param
        first_param = list(optimizer.state.keys())[0]
        original_step = optimizer.state[first_param]["step"].item()
        assert original_step >= 1

        # Save checkpoint
        path = tmp_path / "ckpt.pt"
        save_checkpoint(path, model, optimizer, epoch=5, step=100)

        # Create fresh model + optimizer
        model2 = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        assert len(optimizer2.state) == 0, "Fresh optimizer should have no state"

        # Load checkpoint
        meta = load_checkpoint(path, model2, optimizer2)
        assert meta["epoch"] == 5
        assert meta["step"] == 100

        # Verify optimizer state is restored
        assert len(optimizer2.state) > 0, (
            "Optimizer state should be restored after load_checkpoint"
        )
        first_param2 = list(optimizer2.state.keys())[0]
        restored_step = optimizer2.state[first_param2]["step"].item()
        assert restored_step == original_step, (
            f"Step count mismatch: expected {original_step}, got {restored_step}"
        )

    def test_momentum_buffers_match(self, tmp_path: Path) -> None:
        """Verify that exp_avg (first moment) buffers are numerically restored."""
        model = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Generate momentum
        for _ in range(3):
            obs = torch.randn(2, 46, 9, 9)
            policy, value = model(obs)
            loss = policy.sum() + value.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Capture first param's exp_avg
        first_key = list(optimizer.state.keys())[0]
        original_exp_avg = optimizer.state[first_key]["exp_avg"].clone()

        path = tmp_path / "ckpt_momentum.pt"
        save_checkpoint(path, model, optimizer, epoch=3, step=30)

        model2 = ResNetModel(ResNetParams(hidden_size=16, num_layers=1))
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        load_checkpoint(path, model2, optimizer2)

        first_key2 = list(optimizer2.state.keys())[0]
        restored_exp_avg = optimizer2.state[first_key2]["exp_avg"]

        assert torch.allclose(original_exp_avg, restored_exp_avg, atol=1e-7), (
            "exp_avg momentum buffer not faithfully restored"
        )
