"""Tests for smooth entropy annealing (R5)."""

import torch.nn as nn

from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams


def _make_algo(lambda_entropy=0.01, entropy_decay_epochs=0, warmup_epochs=10, warmup_entropy=0.05):
    params = KataGoPPOParams(lambda_entropy=lambda_entropy, entropy_decay_epochs=entropy_decay_epochs)
    # Minimal real model — just needs parameters() to satisfy Adam init.
    model = nn.Linear(1, 1)
    return KataGoPPOAlgorithm(params, model, warmup_epochs=warmup_epochs, warmup_entropy=warmup_entropy)  # type: ignore[arg-type]


class TestEntropyAnnealing:
    def test_decay_zero_matches_step_behavior(self):
        algo = _make_algo(entropy_decay_epochs=0, warmup_epochs=10)
        assert algo.get_entropy_coeff(9) == 0.05
        assert algo.get_entropy_coeff(10) == 0.01
        assert algo.get_entropy_coeff(100) == 0.01

    def test_linear_decay_at_warmup_boundary(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(10) == 0.05

    def test_linear_decay_midpoint(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10,
                         warmup_entropy=0.05, lambda_entropy=0.01)
        result = algo.get_entropy_coeff(110)
        expected = 0.05 + 0.5 * (0.01 - 0.05)
        assert abs(result - expected) < 1e-9

    def test_linear_decay_one_before_end(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        result = algo.get_entropy_coeff(209)
        assert result > 0.01
        assert result < 0.05

    def test_linear_decay_at_end(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(210) == 0.01

    def test_linear_decay_past_end(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(500) == 0.01
