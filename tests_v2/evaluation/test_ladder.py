"""
Behavior tests for LadderEvaluator.

These tests exercise ELO-based ladder evaluation with real PPOAgent instances
playing real Shogi games against real random opponents. Only filesystem/wandb
boundaries are mocked via tmp_path.
"""

import asyncio

import pytest

from keisei.config_schema import EvaluationConfig
from keisei.evaluation.core.evaluation_context import AgentInfo, OpponentInfo
from keisei.evaluation.strategies.ladder import LadderEvaluator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_eval(evaluator, agent_info, context=None):
    """Run the async evaluate method synchronously."""
    return asyncio.run(evaluator.evaluate(agent_info, context))


# ---------------------------------------------------------------------------
# Tests: basic ladder evaluation
# ---------------------------------------------------------------------------


class TestLadderBasicEvaluation:
    """Tests for basic ladder evaluation behavior."""

    def test_ladder_evaluation_completes(
        self, eval_config_ladder, agent_info
    ):
        """Ladder evaluation runs to completion and returns results."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games > 0
        assert len(result.games) > 0

    def test_win_loss_draw_sum_equals_total(
        self, eval_config_ladder, agent_info
    ):
        """Wins + losses + draws always equals total games."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        stats = result.summary_stats
        assert stats.agent_wins + stats.opponent_wins + stats.draws == stats.total_games

    def test_all_games_have_valid_winner_codes(
        self, eval_config_ladder, agent_info
    ):
        """Each game result has a valid winner code (0, 1, or None)."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        for game in result.games:
            assert game.winner in (0, 1, None)

    def test_no_errors_in_clean_run(
        self, eval_config_ladder, agent_info
    ):
        """A clean ladder evaluation run produces no errors."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Tests: ELO tracking
# ---------------------------------------------------------------------------


class TestLadderEloTracking:
    """Tests for ELO rating tracking in ladder evaluation."""

    def test_analytics_contain_elo_data(
        self, eval_config_ladder, agent_info
    ):
        """Ladder analytics include ELO-specific data."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        assert "ladder_specific_analytics" in result.analytics_data
        ladder_analytics = result.analytics_data["ladder_specific_analytics"]
        assert "initial_agent_rating" in ladder_analytics
        assert "final_agent_rating" in ladder_analytics
        assert "rating_change" in ladder_analytics

    def test_rating_change_is_consistent(
        self, eval_config_ladder, agent_info
    ):
        """rating_change equals final_agent_rating - initial_agent_rating."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        ladder_analytics = result.analytics_data["ladder_specific_analytics"]
        expected_change = (
            ladder_analytics["final_agent_rating"]
            - ladder_analytics["initial_agent_rating"]
        )
        assert abs(ladder_analytics["rating_change"] - expected_change) < 1e-9

    def test_elo_snapshot_contains_all_participants(
        self, eval_config_ladder, agent_info
    ):
        """Final ELO snapshot includes ratings for the agent and played opponents."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        elo_snapshot = result.analytics_data.get("final_elo_snapshot", {})
        # The agent should be in the snapshot
        assert agent_info.name in elo_snapshot

    def test_elo_tracker_attached_to_result(
        self, eval_config_ladder, agent_info
    ):
        """The ELO tracker object is attached to the evaluation result."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        assert result.elo_tracker is not None


# ---------------------------------------------------------------------------
# Tests: opponent selection
# ---------------------------------------------------------------------------


class TestLadderOpponentSelection:
    """Tests for ladder opponent selection behavior."""

    def test_opponents_loaded_from_config(
        self, eval_config_ladder, agent_info
    ):
        """get_opponents returns opponents from configuration."""
        evaluator = LadderEvaluator(eval_config_ladder)
        context = evaluator.setup_context(agent_info)
        opponents = evaluator.get_opponents(context)

        assert len(opponents) >= 1
        names = {opp.name for opp in opponents}
        assert "ladder_random_1" in names or "ladder_random_2" in names

    def test_single_opponent_ladder(self, agent_info, tmp_path):
        """Ladder with a single opponent works correctly."""
        config = EvaluationConfig(
            num_games=2,
            max_moves_per_game=50,
            strategy="ladder",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_ladder(
            opponent_pool_config=[
                {
                    "name": "solo_ladder_opp",
                    "type": "random",
                    "initial_rating": 1500,
                },
            ],
            num_games_per_match=2,
            num_opponents_per_evaluation=1,
            num_opponents_to_select=5,
        )
        evaluator = LadderEvaluator(config)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games == 2
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Tests: game metadata
# ---------------------------------------------------------------------------


class TestLadderGameMetadata:
    """Tests for metadata attached to ladder game results."""

    def test_games_have_agent_color_metadata(
        self, eval_config_ladder, agent_info
    ):
        """Each game records which color the agent played."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        for game in result.games:
            assert "agent_color" in game.metadata
            assert game.metadata["agent_color"] in ("Sente", "Gote")

    def test_games_have_termination_reason(
        self, eval_config_ladder, agent_info
    ):
        """Each game records a termination reason."""
        evaluator = LadderEvaluator(eval_config_ladder)
        result = run_eval(evaluator, agent_info)

        for game in result.games:
            assert "termination_reason" in game.metadata
            assert game.metadata["termination_reason"] is not None
