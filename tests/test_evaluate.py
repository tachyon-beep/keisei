"""Tests for the keisei-evaluate head-to-head CLI."""

from unittest.mock import patch

import pytest

from keisei.training.evaluate import EvalResult, run_evaluation


class TestEvalResult:
    def test_win_rate(self):
        result = EvalResult(wins=60, losses=30, draws=10)
        assert result.total_games == 100
        assert abs(result.win_rate - 0.65) < 1e-6  # (60 + 5) / 100

    def test_elo_delta_positive(self):
        result = EvalResult(wins=60, losses=30, draws=10)
        assert result.elo_delta() > 0

    def test_elo_delta_negative(self):
        result = EvalResult(wins=30, losses=60, draws=10)
        assert result.elo_delta() < 0

    def test_elo_delta_perfect(self):
        result = EvalResult(wins=100, losses=0, draws=0)
        assert result.elo_delta() == float("inf")

    def test_elo_delta_zero_wins(self):
        result = EvalResult(wins=0, losses=100, draws=0)
        assert result.elo_delta() == float("-inf")

    def test_confidence_interval(self):
        result = EvalResult(wins=200, losses=150, draws=50)
        low, high = result.win_rate_ci(confidence=0.95)
        assert low < result.win_rate
        assert high > result.win_rate
        assert high - low < 0.15  # 400 games -> CI < +/-7.5%

    def test_empty_result(self):
        result = EvalResult(wins=0, losses=0, draws=0)
        assert result.total_games == 0
        assert result.win_rate == 0.0
        low, high = result.win_rate_ci()
        assert low == 0.0
        assert high == 1.0


class TestRunEvaluation:
    def test_returns_eval_result(self):
        with patch("keisei.training.evaluate._play_evaluation_games") as mock_play:
            mock_play.return_value = EvalResult(wins=5, losses=3, draws=2)
            result = run_evaluation(
                checkpoint_a="/fake/a.pt", arch_a="resnet",
                checkpoint_b="/fake/b.pt", arch_b="se_resnet",
                games=10, max_ply=100,
            )
            assert result.total_games == 10
            assert result.wins == 5
