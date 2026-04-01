# tests/test_katago_ppo.py
"""Tests for the KataGo multi-head PPO algorithm."""

import pytest

from keisei.training.katago_ppo import KataGoPPOParams
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


class TestAlgorithmRegistry:
    def test_katago_ppo_in_valid_algorithms(self):
        assert "katago_ppo" in VALID_ALGORITHMS

    def test_validate_katago_ppo_params(self):
        validated = validate_algorithm_params("katago_ppo", {"learning_rate": 1e-3})
        assert isinstance(validated, KataGoPPOParams)
        assert validated.learning_rate == 1e-3
