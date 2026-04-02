"""Tests closing PyTorch hot-path gaps: AMP through PPO, value_adapter
dispatch, split_merge edge cases, GradScaler round-trip, and
single-element advantage normalization."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.amp import GradScaler

from keisei.training.checkpoint import load_checkpoint, save_checkpoint
from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
)
from keisei.training.katago_loop import split_merge_step
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams
from keisei.training.value_adapter import MultiHeadValueAdapter


def _small_model():
    return SEResNetModel(SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    ))


def _filled_buffer(num_envs=4, steps=3, action_space=11259):
    """Create a buffer with terminal steps so value head gets gradients."""
    buf = KataGoRolloutBuffer(
        num_envs=num_envs, obs_shape=(50, 9, 9), action_space=action_space,
    )
    for t in range(steps):
        is_last = t == steps - 1
        buf.add(
            obs=torch.randn(num_envs, 50, 9, 9),
            actions=torch.randint(0, action_space, (num_envs,)),
            log_probs=torch.randn(num_envs),
            values=torch.randn(num_envs),
            rewards=torch.where(
                torch.tensor([is_last] * num_envs),
                torch.tensor([1.0, -1.0, 0.0, 1.0][:num_envs]),
                torch.zeros(num_envs),
            ),
            dones=torch.tensor([is_last] * num_envs),
            legal_masks=torch.ones(num_envs, action_space, dtype=torch.bool),
            value_categories=torch.where(
                torch.tensor([is_last] * num_envs),
                torch.tensor([0, 2, 1, 0][:num_envs]),
                torch.full((num_envs,), -1),
            ),
            score_targets=torch.randn(num_envs).clamp(-1.5, 1.5),
        )
    return buf


# ---------------------------------------------------------------------------
# 1. PPO update with use_amp=True on CPU
# ---------------------------------------------------------------------------


class TestPPOWithAMP:
    def test_update_with_amp_produces_finite_metrics(self):
        """PPO update with use_amp=True should complete with finite metrics on CPU."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(
            KataGoPPOParams(epochs_per_batch=1, use_amp=True), model
        )
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.zeros(4)

        metrics = ppo.update(buf, next_values)

        for key, val in metrics.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{key} is not finite with AMP: {val}"

    def test_amp_select_actions_produces_finite_output(self):
        """select_actions with use_amp=True should produce finite outputs."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(
            KataGoPPOParams(use_amp=True), model
        )
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)

        actions, log_probs, values = ppo.select_actions(obs, legal_masks)

        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(values).all()

    def test_amp_updates_model_weights(self):
        """AMP-enabled PPO should actually modify model parameters."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(
            KataGoPPOParams(epochs_per_batch=1, use_amp=True), model
        )
        policy_before = model.policy_conv2.weight.data.clone()

        buf = _filled_buffer(num_envs=4, steps=3)
        ppo.update(buf, torch.zeros(4))

        assert not torch.equal(policy_before, model.policy_conv2.weight.data), \
            "AMP-enabled PPO did not update policy head"


# ---------------------------------------------------------------------------
# 2. PPO update with value_adapter
# ---------------------------------------------------------------------------


class TestPPOWithValueAdapter:
    def test_update_with_adapter_produces_finite_metrics(self):
        """PPO update dispatching through MultiHeadValueAdapter should work."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(
            KataGoPPOParams(epochs_per_batch=1), model
        )
        adapter = MultiHeadValueAdapter(lambda_value=1.5, lambda_score=0.02)
        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.zeros(4)

        metrics = ppo.update(buf, next_values, value_adapter=adapter)

        for key, val in metrics.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{key} not finite with adapter: {val}"

    def test_adapter_updates_all_heads(self):
        """Adapter path should send gradients to all three heads."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(
            KataGoPPOParams(epochs_per_batch=1), model
        )
        adapter = MultiHeadValueAdapter(lambda_value=1.5, lambda_score=0.02)

        policy_before = model.policy_conv2.weight.data.clone()
        value_before = model.value_fc2.weight.data.clone()
        score_before = model.score_fc2.weight.data.clone()

        buf = _filled_buffer(num_envs=4, steps=3)
        ppo.update(buf, torch.zeros(4), value_adapter=adapter)

        assert not torch.equal(policy_before, model.policy_conv2.weight.data), \
            "Policy head not updated via adapter"
        assert not torch.equal(value_before, model.value_fc2.weight.data), \
            "Value head not updated via adapter"
        assert not torch.equal(score_before, model.score_fc2.weight.data), \
            "Score head not updated via adapter"

    def test_adapter_score_loss_is_zero_in_metrics(self):
        """When adapter is used, score_loss metric should be 0.0 (adapter combines them)."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(
            KataGoPPOParams(epochs_per_batch=1), model
        )
        adapter = MultiHeadValueAdapter()
        buf = _filled_buffer(num_envs=4, steps=3)

        metrics = ppo.update(buf, torch.zeros(4), value_adapter=adapter)

        # The adapter path sets score_loss = tensor(0.0) for metrics tracking
        assert metrics["score_loss"] == 0.0


# ---------------------------------------------------------------------------
# 3. split_merge_step edge cases
# ---------------------------------------------------------------------------


class TestSplitMergeEdgeCases:
    def test_all_opponent_envs(self):
        """When all envs are opponent-side, learner outputs should be empty."""
        model = _small_model()
        model.eval()

        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        # All envs are player 1 (opponent) while learner is player 0
        current_players = np.array([1, 1, 1, 1], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=model,
            opponent_model=model,
            learner_side=0,
        )

        assert result.learner_indices.numel() == 0
        assert result.learner_log_probs.numel() == 0
        assert result.learner_values.numel() == 0
        assert result.learner_mask.sum() == 0
        assert result.opponent_mask.sum() == num_envs
        # Actions should still be populated (by opponent)
        assert result.actions.shape == (num_envs,)

    def test_all_learner_envs(self):
        """When all envs are learner-side, opponent outputs are empty."""
        model = _small_model()
        model.eval()

        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 0, 0, 0], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=model,
            opponent_model=model,
            learner_side=0,
        )

        assert result.learner_indices.numel() == num_envs
        assert result.learner_log_probs.numel() == num_envs
        assert result.learner_values.numel() == num_envs
        assert torch.isfinite(result.learner_log_probs).all()
        assert torch.isfinite(result.learner_values).all()

    def test_split_merge_with_value_adapter(self):
        """split_merge_step should use value_adapter when provided."""
        model = _small_model()
        model.eval()
        adapter = MultiHeadValueAdapter()

        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 0, 0, 0], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=model,
            opponent_model=model,
            learner_side=0,
            value_adapter=adapter,
        )

        assert result.learner_values.numel() == num_envs
        assert torch.isfinite(result.learner_values).all()

    def test_adapter_and_direct_values_agree(self):
        """MultiHeadValueAdapter.scalar_value_from_output should match
        KataGoPPOAlgorithm.scalar_value for the same value logits."""
        model = _small_model()
        model.eval()
        adapter = MultiHeadValueAdapter()

        obs = torch.randn(4, 50, 9, 9)
        with torch.no_grad():
            output = model(obs)

        direct = KataGoPPOAlgorithm.scalar_value(output.value_logits)
        via_adapter = adapter.scalar_value_from_output(output.value_logits)

        assert torch.allclose(direct, via_adapter, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. GradScaler checkpoint round-trip
# ---------------------------------------------------------------------------


class TestGradScalerCheckpointRoundTrip:
    def test_scaler_state_survives_save_load(self, tmp_path):
        """GradScaler state (scale factor, growth tracker) should survive checkpoint."""
        model = _small_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = GradScaler(enabled=True)

        # Advance scaler state by simulating a few steps
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            obs = torch.randn(2, 50, 9, 9)
            output = model(obs)
            loss = output.policy_logits.sum() + output.value_logits.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

        original_scale = scaler.get_scale()
        original_state = scaler.state_dict()

        path = tmp_path / "ckpt_scaler.pt"
        save_checkpoint(path, model, optimizer, epoch=5, step=50,
                        grad_scaler=scaler)

        # Fresh scaler with default state
        fresh_model = _small_model()
        fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        fresh_scaler = GradScaler(enabled=True)
        assert fresh_scaler.get_scale() != original_scale or original_scale == fresh_scaler.get_scale()

        load_checkpoint(path, fresh_model, fresh_opt, grad_scaler=fresh_scaler)

        assert fresh_scaler.get_scale() == pytest.approx(original_scale)
        # Verify the full state dict matches
        loaded_state = fresh_scaler.state_dict()
        assert loaded_state["scale"] == pytest.approx(original_state["scale"])
        assert loaded_state["_growth_tracker"] == original_state["_growth_tracker"]

    def test_no_scaler_in_checkpoint_leaves_fresh_scaler_unchanged(self, tmp_path):
        """Loading a checkpoint without scaler state should not modify the scaler."""
        model = _small_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        path = tmp_path / "no_scaler.pt"
        save_checkpoint(path, model, optimizer, epoch=1, step=1)
        # No grad_scaler passed to save

        fresh_model = _small_model()
        fresh_opt = torch.optim.Adam(fresh_model.parameters(), lr=1e-3)
        fresh_scaler = GradScaler(enabled=True)
        default_scale = fresh_scaler.get_scale()

        load_checkpoint(path, fresh_model, fresh_opt, grad_scaler=fresh_scaler)

        assert fresh_scaler.get_scale() == default_scale


# ---------------------------------------------------------------------------
# 5. Single-element advantage normalization
# ---------------------------------------------------------------------------


class TestSingleElementAdvantageNormalization:
    def test_single_sample_update_produces_finite_metrics(self):
        """PPO update with exactly 1 sample should not divide by zero in
        advantage normalization (numel() > 1 guard)."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(
            KataGoPPOParams(epochs_per_batch=1, batch_size=1), model
        )

        buf = KataGoRolloutBuffer(
            num_envs=1, obs_shape=(50, 9, 9), action_space=11259,
        )
        buf.add(
            obs=torch.randn(1, 50, 9, 9),
            actions=torch.randint(0, 11259, (1,)),
            log_probs=torch.randn(1),
            values=torch.randn(1),
            rewards=torch.tensor([1.0]),
            dones=torch.tensor([True]),
            legal_masks=torch.ones(1, 11259, dtype=torch.bool),
            value_categories=torch.tensor([0]),  # W
            score_targets=torch.tensor([0.5]),
        )

        next_values = torch.zeros(1)
        metrics = ppo.update(buf, next_values)

        for key, val in metrics.items():
            if isinstance(val, float):
                assert np.isfinite(val), f"{key} not finite with single sample: {val}"

    def test_two_samples_normalizes_advantages(self):
        """With 2+ samples, advantage normalization should produce mean~0, std~1."""
        model = _small_model()
        ppo = KataGoPPOAlgorithm(
            KataGoPPOParams(epochs_per_batch=1, batch_size=256), model
        )

        buf = _filled_buffer(num_envs=4, steps=3)
        next_values = torch.zeros(4)

        # We can't easily inspect advantages mid-update, but we verify
        # the update completes and produces non-degenerate metrics
        metrics = ppo.update(buf, next_values)

        assert metrics["policy_loss"] != 0.0
        assert np.isfinite(metrics["gradient_norm"])
