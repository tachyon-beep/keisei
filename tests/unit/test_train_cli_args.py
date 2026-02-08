"""Tests for CLI argument parsing in train.py evaluation subcommand."""

import argparse

from keisei.training.train import add_evaluation_arguments


def _parse_eval_args(*args: str) -> argparse.Namespace:
    """Parse evaluation arguments, simulating the evaluate subcommand."""
    parser = argparse.ArgumentParser()
    add_evaluation_arguments(parser)
    return parser.parse_args(list(args))


class TestEvalArgDefaults:
    """Verify that evaluation CLI defaults are None, not hardcoded values."""

    def test_strategy_default_is_none(self):
        """--strategy defaults to None when not explicitly passed."""
        args = _parse_eval_args("--agent_checkpoint", "dummy.pt")
        assert args.strategy is None

    def test_num_games_default_is_none(self):
        """--num_games defaults to None when not explicitly passed."""
        args = _parse_eval_args("--agent_checkpoint", "dummy.pt")
        assert args.num_games is None

    def test_opponent_type_default_is_none(self):
        """--opponent_type defaults to None when not explicitly passed."""
        args = _parse_eval_args("--agent_checkpoint", "dummy.pt")
        assert args.opponent_type is None


class TestEvalArgExplicit:
    """Verify that explicitly passed CLI args are captured correctly."""

    def test_explicit_strategy(self):
        """--strategy captures the value when passed explicitly."""
        args = _parse_eval_args(
            "--agent_checkpoint", "dummy.pt", "--strategy", "tournament"
        )
        assert args.strategy == "tournament"

    def test_explicit_num_games(self):
        """--num_games captures the value when passed explicitly."""
        args = _parse_eval_args(
            "--agent_checkpoint", "dummy.pt", "--num_games", "50"
        )
        assert args.num_games == 50

    def test_explicit_opponent_type(self):
        """--opponent_type captures the value when passed explicitly."""
        args = _parse_eval_args(
            "--agent_checkpoint", "dummy.pt", "--opponent_type", "heuristic"
        )
        assert args.opponent_type == "heuristic"
