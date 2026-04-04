# tests/test_amp.py
"""Tests for AMP mixed precision support."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams, KataGoRolloutBuffer
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


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



def _make_ppo(use_amp: bool = False) -> KataGoPPOAlgorithm:
    """Create a minimal PPO algorithm with a tiny SEResNet model."""
    params = KataGoPPOParams(use_amp=use_amp, batch_size=4, epochs_per_batch=1)
    model_params = SEResNetParams(
        num_blocks=1, channels=16, se_reduction=4,
        global_pool_channels=8, policy_channels=4,
        value_fc_size=16, score_fc_size=8, obs_channels=50,
    )
    model = SEResNetModel(model_params)
    return KataGoPPOAlgorithm(params, model)


def _fill_buffer(ppo: KataGoPPOAlgorithm, num_envs: int = 2, steps: int = 4) -> KataGoRolloutBuffer:
    """Fill a rollout buffer with random data."""
    obs_shape = (50, 9, 9)
    action_space = 81 * 139
    buf = KataGoRolloutBuffer(num_envs, obs_shape, action_space)

    for _ in range(steps):
        obs = torch.randn(num_envs, *obs_shape)
        actions = torch.randint(0, action_space, (num_envs,))
        log_probs = torch.randn(num_envs)
        values = torch.randn(num_envs)
        rewards = torch.randn(num_envs)
        dones = torch.zeros(num_envs)
        legal_masks = torch.ones(num_envs, action_space, dtype=torch.bool)
        value_cats = torch.randint(0, 3, (num_envs,))
        score_targets = torch.randn(num_envs).clamp(-1.5, 1.5)
        buf.add(obs, actions, log_probs, values, rewards, dones, dones, legal_masks,
                value_categories=value_cats, score_targets=score_targets)

    return buf


class TestPPOAmp:
    def test_update_with_amp_produces_finite_loss(self) -> None:
        """PPO update with use_amp=True runs without error and produces finite metrics."""
        ppo = _make_ppo(use_amp=True)
        buf = _fill_buffer(ppo)
        next_values = torch.randn(2)

        metrics = ppo.update(buf, next_values)

        assert all(
            torch.isfinite(torch.tensor(v)) for v in metrics.values() if isinstance(v, float)
        ), f"Non-finite metrics: {metrics}"

    def test_update_without_amp_still_works(self) -> None:
        """use_amp=False (default) doesn't break anything."""
        ppo = _make_ppo(use_amp=False)
        buf = _fill_buffer(ppo)
        next_values = torch.randn(2)

        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics

    def test_amp_on_cpu_uses_no_op_autocast(self) -> None:
        """AMP on CPU should not crash — autocast('cpu') is a valid no-op."""
        ppo = _make_ppo(use_amp=True)
        buf = _fill_buffer(ppo)
        next_values = torch.randn(2)

        metrics = ppo.update(buf, next_values)
        assert "policy_loss" in metrics


class TestSelectActionsAmp:
    def test_select_actions_with_amp(self) -> None:
        """select_actions with use_amp=True produces valid actions."""
        ppo = _make_ppo(use_amp=True)
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.ones(2, 81 * 139, dtype=torch.bool)

        actions, log_probs, values = ppo.select_actions(obs, legal_masks)

        assert actions.shape == (2,)
        assert log_probs.shape == (2,)
        assert values.shape == (2,)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(values).all()

    def test_select_actions_without_amp(self) -> None:
        """select_actions with use_amp=False (default) still works correctly."""
        ppo = _make_ppo(use_amp=False)
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.ones(2, 81 * 139, dtype=torch.bool)

        actions, log_probs, values = ppo.select_actions(obs, legal_masks)

        assert actions.shape == (2,)
        assert log_probs.shape == (2,)
        assert values.shape == (2,)
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(values).all()


class TestAmpIntegration:
    def test_ppo_checkpoint_round_trip_with_amp(self, tmp_path: Path) -> None:
        """Full cycle: PPO update with AMP → save → load → update again."""
        ppo = _make_ppo(use_amp=True)
        buf = _fill_buffer(ppo)
        next_values = torch.randn(2)

        metrics1 = ppo.update(buf, next_values)
        assert "policy_loss" in metrics1

        # Save checkpoint
        save_checkpoint(
            tmp_path / "ckpt.pt", ppo.model, ppo.optimizer,
            epoch=1, step=10, grad_scaler=ppo.scaler,
        )

        # Create fresh PPO and load
        ppo2 = _make_ppo(use_amp=True)
        load_checkpoint(
            tmp_path / "ckpt.pt", ppo2.model, ppo2.optimizer,
            grad_scaler=ppo2.scaler,
        )

        # Verify scaler state transferred
        assert ppo2.scaler.get_scale() == ppo.scaler.get_scale()

        # Second update should work
        buf2 = _fill_buffer(ppo2)
        metrics2 = ppo2.update(buf2, next_values)
        assert "policy_loss" in metrics2
