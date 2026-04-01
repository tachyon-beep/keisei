# tests/test_katago_ppo.py
"""Tests for the KataGo multi-head PPO algorithm."""

import pytest
import torch

from keisei.training.katago_ppo import (
    KataGoPPOAlgorithm,
    KataGoPPOParams,
    KataGoRolloutBuffer,
    compute_value_metrics,
)
from keisei.training.ppo import compute_gae
from keisei.training.algorithm_registry import validate_algorithm_params, VALID_ALGORITHMS
from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams


class TestKataGoPPOParams:
    def test_defaults(self):
        params = KataGoPPOParams()
        assert params.learning_rate == 2e-4
        assert params.gamma == 0.99
        assert params.gae_lambda == 0.95
        assert params.lambda_value == 1.5
        assert params.lambda_score == 0.02
        assert params.lambda_entropy == 0.01
        assert params.score_normalization == 76.0
        assert params.grad_clip == 1.0

    def test_custom_params(self):
        params = KataGoPPOParams(learning_rate=1e-3, gamma=1.0)
        assert params.learning_rate == 1e-3
        assert params.gamma == 1.0


class TestKataGoRolloutBuffer:
    def test_add_and_size(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.tensor([100, 200])
        log_probs = torch.tensor([0.5, 0.6])
        values = torch.tensor([0.1, 0.2])
        rewards = torch.tensor([0.0, 0.0])
        dones = torch.tensor([False, False])
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        value_cats = torch.tensor([0, 2])  # W, L
        score_targets = torch.tensor([0.5, -0.3])

        buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks,
                value_cats, score_targets)
        assert buf.size == 1

    def test_flatten(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(3):
            buf.add(
                torch.randn(2, 50, 9, 9),
                torch.randint(0, 11259, (2,)),
                torch.randn(2),
                torch.randn(2),
                torch.randn(2),
                torch.zeros(2, dtype=torch.bool),
                torch.ones(2, 11259, dtype=torch.bool),
                torch.randint(0, 3, (2,)),
                torch.rand(2) * 2 - 1,  # uniform in [-1, 1] to stay within guard
            )
        data = buf.flatten()
        assert data["observations"].shape == (6, 50, 9, 9)
        assert data["actions"].shape == (6,)
        assert data["value_categories"].shape == (6,)
        assert data["score_targets"].shape == (6,)

    def test_clear(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        buf.add(
            torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
            torch.zeros(2), torch.zeros(2), torch.zeros(2),
            torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
            torch.zeros(2, dtype=torch.long), torch.zeros(2),
        )
        assert buf.size == 1
        buf.clear()
        assert buf.size == 0


class TestAlgorithmRegistry:
    def test_katago_ppo_in_valid_algorithms(self):
        assert "katago_ppo" in VALID_ALGORITHMS

    def test_validate_katago_ppo_params(self):
        validated = validate_algorithm_params("katago_ppo", {"learning_rate": 1e-3})
        assert isinstance(validated, KataGoPPOParams)
        assert validated.learning_rate == 1e-3


@pytest.fixture
def small_model():
    params = SEResNetParams(
        num_blocks=2, channels=32, se_reduction=8,
        global_pool_channels=16, policy_channels=8,
        value_fc_size=32, score_fc_size=16, obs_channels=50,
    )
    return SEResNetModel(params)


@pytest.fixture
def ppo(small_model):
    params = KataGoPPOParams()
    return KataGoPPOAlgorithm(params, small_model)


class TestKataGoPPOActionSelection:
    def test_select_actions_shapes(self, ppo):
        obs = torch.randn(4, 50, 9, 9)
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        actions, log_probs, values = ppo.select_actions(obs, legal_masks)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)
        assert values.shape == (4,)

    def test_select_actions_values_bounded(self, ppo):
        """Scalar value P(W) - P(L) should be in [-1, 1]."""
        obs = torch.randn(8, 50, 9, 9)
        legal_masks = torch.ones(8, 11259, dtype=torch.bool)
        _, _, values = ppo.select_actions(obs, legal_masks)
        assert values.min() >= -1.0
        assert values.max() <= 1.0

    def test_select_actions_all_illegal_raises(self, ppo):
        """All-False legal mask should raise, not produce NaN."""
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.zeros(2, 11259, dtype=torch.bool)  # all illegal
        with pytest.raises(RuntimeError, match="zero legal actions"):
            ppo.select_actions(obs, legal_masks)

    def test_select_actions_respects_mask(self, ppo):
        """Actions should only be sampled from legal positions (20 trials)."""
        obs = torch.randn(2, 50, 9, 9)
        legal_masks = torch.zeros(2, 11259, dtype=torch.bool)
        legal_masks[:, 0] = True
        legal_masks[:, 1000] = True
        for _ in range(20):
            actions, _, _ = ppo.select_actions(obs, legal_masks)
            for a in actions.tolist():
                assert a in (0, 1000), f"Action {a} should be 0 or 1000"


class TestValueMetrics:
    def test_compute_value_metrics(self):
        """Verify the metrics helper computes frac_predicted_win/draw/loss correctly."""
        # 4 predictions: 2 predict W, 1 predicts D, 1 predicts L
        value_logits = torch.tensor([
            [2.0, 0.0, 0.0],  # predicts W
            [0.0, 2.0, 0.0],  # predicts D
            [0.0, 0.0, 2.0],  # predicts L
            [1.5, 0.0, 0.0],  # predicts W
        ])
        value_targets = torch.tensor([0, 1, 2, 0])  # W, D, L, W

        metrics = compute_value_metrics(value_logits, value_targets)
        assert metrics["value_accuracy"] == 1.0  # all correct
        assert metrics["frac_predicted_win"] == 0.5
        assert metrics["frac_predicted_draw"] == 0.25
        assert metrics["frac_predicted_loss"] == 0.25


class TestKataGoPPOUpdate:
    def test_update_returns_metrics(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.rand(2) * 2 - 1,
            )
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert "score_loss" in losses
        assert "entropy" in losses
        assert "gradient_norm" in losses

    def test_update_loss_values_are_finite(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.randn(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), torch.rand(2) * 2 - 1,
            )
        next_values = torch.zeros(2)
        losses = ppo.update(buf, next_values)
        for key, val in losses.items():
            assert not torch.tensor(val).isnan(), f"{key} is NaN"
            assert not torch.tensor(val).isinf(), f"{key} is inf"

    def test_update_clears_buffer(self, ppo):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.zeros(2, dtype=torch.long), torch.zeros(2),
            )
        ppo.update(buf, torch.zeros(2))
        assert buf.size == 0
