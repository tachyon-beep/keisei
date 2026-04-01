# tests/test_katago_ppo.py
"""Tests for the KataGo multi-head PPO algorithm."""

import pytest
import torch

from keisei.training.katago_ppo import KataGoPPOParams, KataGoRolloutBuffer
from keisei.training.algorithm_registry import validate_algorithm_params, VALID_ALGORITHMS


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
