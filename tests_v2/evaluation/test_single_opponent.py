"""
Behavior tests for SingleOpponentEvaluator.

These tests exercise the actual evaluation pipeline: real PPOAgent instances
play real Shogi games against real opponents. Only filesystem/wandb boundaries
are mocked via tmp_path.
"""

import asyncio

import pytest

from keisei.config_schema import EvaluationConfig
from keisei.evaluation.core.evaluation_context import AgentInfo, OpponentInfo
from keisei.evaluation.strategies.single_opponent import SingleOpponentEvaluator


# ---------------------------------------------------------------------------
# Helper to run async evaluate() in sync tests
# ---------------------------------------------------------------------------

def run_eval(evaluator, agent_info, context=None):
    """Run the async evaluate method synchronously."""
    return asyncio.run(evaluator.evaluate(agent_info, context))


def run_eval_step(evaluator, agent_info, opponent_info, context):
    """Run the async evaluate_step method synchronously."""
    return asyncio.run(evaluator.evaluate_step(agent_info, opponent_info, context))


# ---------------------------------------------------------------------------
# Tests: basic evaluation with N games
# ---------------------------------------------------------------------------


class TestSingleOpponentBasicEvaluation:
    """Tests that verify full evaluation runs produce correct result structure."""

    def test_two_games_produces_two_results(
        self, eval_config_single, agent_info
    ):
        """Running evaluation with num_games=2 produces exactly 2 game results."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games == 2
        assert len(result.games) == 2

    def test_win_loss_draw_sum_equals_total(
        self, eval_config_single, agent_info
    ):
        """Wins + losses + draws always equals total games played."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        result = run_eval(evaluator, agent_info)

        stats = result.summary_stats
        assert stats.agent_wins + stats.opponent_wins + stats.draws == stats.total_games

    def test_win_rate_is_consistent(self, eval_config_single, agent_info):
        """Win rate equals agent_wins / total_games."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        result = run_eval(evaluator, agent_info)

        stats = result.summary_stats
        if stats.total_games > 0:
            expected_wr = stats.agent_wins / stats.total_games
            assert abs(stats.win_rate - expected_wr) < 1e-9

    def test_game_results_have_positive_move_counts(
        self, eval_config_single, agent_info
    ):
        """Each game result records a non-negative move count."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        result = run_eval(evaluator, agent_info)

        for game in result.games:
            assert game.moves_count >= 0

    def test_game_results_have_positive_duration(
        self, eval_config_single, agent_info
    ):
        """Each game result records a non-negative duration."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        result = run_eval(evaluator, agent_info)

        for game in result.games:
            assert game.duration_seconds >= 0.0

    def test_game_results_have_valid_winner_codes(
        self, eval_config_single, agent_info
    ):
        """Winner is 0 (agent), 1 (opponent), or None (draw)."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        result = run_eval(evaluator, agent_info)

        for game in result.games:
            assert game.winner in (0, 1, None)


# ---------------------------------------------------------------------------
# Tests: single game step
# ---------------------------------------------------------------------------


class TestSingleOpponentEvaluateStep:
    """Tests for the evaluate_step method (single game execution)."""

    def test_single_game_completes(
        self, eval_config_single, agent_info, opponent_info_random
    ):
        """A single evaluate_step returns a GameResult with a game_id."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        context = evaluator.setup_context(agent_info)
        game_result = run_eval_step(
            evaluator, agent_info, opponent_info_random, context
        )

        assert game_result.game_id is not None
        assert game_result.game_id.startswith("single_")

    def test_single_game_has_agent_and_opponent_info(
        self, eval_config_single, agent_info, opponent_info_random
    ):
        """GameResult carries correct agent_info and opponent_info."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        context = evaluator.setup_context(agent_info)
        game_result = run_eval_step(
            evaluator, agent_info, opponent_info_random, context
        )

        assert game_result.agent_info.name == agent_info.name
        assert game_result.opponent_info.name == opponent_info_random.name


# ---------------------------------------------------------------------------
# Tests: opponent types
# ---------------------------------------------------------------------------


class TestSingleOpponentTypes:
    """Tests that various opponent types work with the evaluator."""

    def test_random_opponent_works(self, agent_info, tmp_path):
        """Evaluation against 'random' opponent completes successfully."""
        config = EvaluationConfig(
            num_games=1,
            max_moves_per_game=50,
            strategy="single_opponent",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_single_opponent(
            opponent_name="random",
            play_as_both_colors=False,
        )
        evaluator = SingleOpponentEvaluator(config)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games == 1
        assert len(result.errors) == 0

    def test_heuristic_opponent_works(self, agent_info, tmp_path):
        """Evaluation against 'heuristic' opponent completes successfully."""
        config = EvaluationConfig(
            num_games=1,
            max_moves_per_game=50,
            strategy="single_opponent",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_single_opponent(
            opponent_name="heuristic",
            play_as_both_colors=False,
        )
        evaluator = SingleOpponentEvaluator(config)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games == 1
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Tests: color balancing
# ---------------------------------------------------------------------------


class TestSingleOpponentColorBalance:
    """Tests for the play_as_both_colors / color balance feature."""

    def test_color_balance_splits_games(self, agent_info, tmp_path):
        """With play_as_both_colors=True and 4 games, agent plays 2 as each color."""
        config = EvaluationConfig(
            num_games=4,
            max_moves_per_game=50,
            strategy="single_opponent",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_single_opponent(
            opponent_name="random",
            play_as_both_colors=True,
        )
        evaluator = SingleOpponentEvaluator(config)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games == 4

        # Check analytics has color-specific data
        analytics = result.analytics_data
        if "first_player_games" in analytics and "second_player_games" in analytics:
            assert analytics["first_player_games"] + analytics["second_player_games"] == 4

    def test_no_color_balance_all_agent_first(self, agent_info, tmp_path):
        """With play_as_both_colors=False, all games have agent as first player."""
        config = EvaluationConfig(
            num_games=2,
            max_moves_per_game=50,
            strategy="single_opponent",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_single_opponent(
            opponent_name="random",
            play_as_both_colors=False,
        )
        evaluator = SingleOpponentEvaluator(config)
        dist = evaluator._calculate_game_distribution()

        assert dist["agent_first"] == 2
        assert dist["agent_second"] == 0


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestSingleOpponentEdgeCases:
    """Edge case tests for the single opponent evaluator."""

    def test_one_game_evaluation(self, agent_info, tmp_path):
        """Evaluation with num_games=1 works correctly."""
        config = EvaluationConfig(
            num_games=1,
            max_moves_per_game=50,
            strategy="single_opponent",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_single_opponent(
            opponent_name="random",
            play_as_both_colors=False,
        )
        evaluator = SingleOpponentEvaluator(config)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games == 1
        assert len(result.games) == 1

    def test_game_distribution_odd_number(self, tmp_path):
        """Odd number of games distributes correctly between colors."""
        config = EvaluationConfig(
            num_games=3,
            max_moves_per_game=50,
            strategy="single_opponent",
            save_path=str(tmp_path / "eval"),
        )
        config.configure_for_single_opponent(
            opponent_name="random",
            play_as_both_colors=True,
        )
        evaluator = SingleOpponentEvaluator(config)
        dist = evaluator._calculate_game_distribution()

        assert dist["agent_first"] + dist["agent_second"] == 3
        # agent_first gets the remainder
        assert dist["agent_first"] == 2
        assert dist["agent_second"] == 1

    def test_no_errors_in_clean_run(self, eval_config_single, agent_info):
        """A clean evaluation run produces no errors."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        result = run_eval(evaluator, agent_info)

        assert len(result.errors) == 0

    def test_summary_stats_total_moves_is_sum(
        self, eval_config_single, agent_info
    ):
        """SummaryStats.total_moves equals sum of all game move counts."""
        evaluator = SingleOpponentEvaluator(eval_config_single)
        result = run_eval(evaluator, agent_info)

        expected_total = sum(g.moves_count for g in result.games)
        assert result.summary_stats.total_moves == expected_total
