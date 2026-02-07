"""
Behavior tests for TournamentEvaluator.

These tests exercise round-robin tournament evaluation with real PPOAgent
instances playing real Shogi games against real random/heuristic opponents.
Only filesystem/wandb boundaries are mocked via tmp_path.
"""

import asyncio

import pytest

from keisei.config_schema import EvaluationConfig
from keisei.evaluation.core.evaluation_context import AgentInfo, OpponentInfo
from keisei.evaluation.core.evaluation_result import SummaryStats
from keisei.evaluation.strategies.tournament import TournamentEvaluator


# ---------------------------------------------------------------------------
# Helper to run async evaluate() in sync tests
# ---------------------------------------------------------------------------

def run_eval(evaluator, agent_info, context=None):
    """Run the async evaluate method synchronously."""
    return asyncio.run(evaluator.evaluate(agent_info, context))


# ---------------------------------------------------------------------------
# Tests: basic tournament evaluation
# ---------------------------------------------------------------------------


class TestTournamentBasicEvaluation:
    """Tests for basic tournament evaluation behavior."""

    def test_tournament_with_two_opponents_produces_correct_game_count(
        self, eval_config_tournament, agent_info
    ):
        """Tournament with 2 opponents and 2 games each produces 4 total games."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games == 4
        assert len(result.games) == 4

    def test_win_loss_draw_sum_equals_total(
        self, eval_config_tournament, agent_info
    ):
        """Wins + losses + draws always equals total games."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        stats = result.summary_stats
        assert stats.agent_wins + stats.opponent_wins + stats.draws == stats.total_games

    def test_tournament_results_have_game_ids(
        self, eval_config_tournament, agent_info
    ):
        """All games in the tournament have valid game IDs."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        for game in result.games:
            assert game.game_id is not None
            assert len(game.game_id) > 0

    def test_tournament_no_errors_in_clean_run(
        self, eval_config_tournament, agent_info
    ):
        """A clean tournament run produces no errors."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        assert len(result.errors) == 0

    def test_each_game_has_valid_winner(
        self, eval_config_tournament, agent_info
    ):
        """Each game result has a valid winner code (0, 1, or None)."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        for game in result.games:
            assert game.winner in (0, 1, None)


# ---------------------------------------------------------------------------
# Tests: tournament standings / analytics
# ---------------------------------------------------------------------------


class TestTournamentStandings:
    """Tests for tournament standings and per-opponent analytics."""

    def test_standings_contain_overall_stats(
        self, eval_config_tournament, agent_info
    ):
        """Tournament analytics include overall tournament stats."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        assert "tournament_specific_analytics" in result.analytics_data
        standings = result.analytics_data["tournament_specific_analytics"]
        assert "overall_tournament_stats" in standings

    def test_overall_stats_match_summary(
        self, eval_config_tournament, agent_info
    ):
        """Overall standings total games matches summary stats total games."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        standings = result.analytics_data["tournament_specific_analytics"]
        overall = standings["overall_tournament_stats"]
        assert overall["total_games"] == result.summary_stats.total_games

    def test_per_opponent_results_exist(
        self, eval_config_tournament, agent_info
    ):
        """Tournament standings include per-opponent results."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        standings = result.analytics_data["tournament_specific_analytics"]
        assert "per_opponent_results" in standings
        per_opp = standings["per_opponent_results"]
        assert len(per_opp) >= 1

    def test_per_opponent_games_sum_to_total(
        self, eval_config_tournament, agent_info
    ):
        """Sum of per-opponent played counts equals total games."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        standings = result.analytics_data["tournament_specific_analytics"]
        per_opp = standings["per_opponent_results"]
        total_played = sum(opp_stats["played"] for opp_stats in per_opp.values())
        assert total_played == result.summary_stats.total_games

    def test_per_opponent_wld_sum_to_played(
        self, eval_config_tournament, agent_info
    ):
        """For each opponent, wins + losses + draws = played."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        result = run_eval(evaluator, agent_info)

        standings = result.analytics_data["tournament_specific_analytics"]
        per_opp = standings["per_opponent_results"]
        for opp_name, opp_stats in per_opp.items():
            total = opp_stats["wins"] + opp_stats["losses"] + opp_stats["draws"]
            assert total == opp_stats["played"], (
                f"Opponent {opp_name}: W+L+D={total} != played={opp_stats['played']}"
            )


# ---------------------------------------------------------------------------
# Tests: opponent configuration
# ---------------------------------------------------------------------------


class TestTournamentOpponentConfig:
    """Tests for tournament opponent configuration and loading."""

    def test_get_opponents_returns_configured_opponents(
        self, eval_config_tournament, agent_info
    ):
        """get_opponents returns the opponents from configuration."""
        evaluator = TournamentEvaluator(eval_config_tournament)
        context = evaluator.setup_context(agent_info)
        opponents = evaluator.get_opponents(context)

        assert len(opponents) == 2
        names = {opp.name for opp in opponents}
        assert "random_opp_1" in names
        assert "random_opp_2" in names

    def test_default_opponent_when_no_pool_configured(
        self, agent_info, tmp_path
    ):
        """With empty pool config, get_opponents provides a default random opponent."""
        config = EvaluationConfig(
            num_games=2,
            max_moves_per_game=50,
            strategy="tournament",
            save_path=str(tmp_path / "eval"),
        )
        config.configure_for_tournament(opponent_pool_config=[])
        evaluator = TournamentEvaluator(config)
        context = evaluator.setup_context(agent_info)
        opponents = evaluator.get_opponents(context)

        assert len(opponents) == 1
        assert opponents[0].type == "random"

    def test_single_opponent_tournament(self, agent_info, tmp_path):
        """Tournament with a single opponent works correctly."""
        config = EvaluationConfig(
            num_games=2,
            max_moves_per_game=50,
            strategy="tournament",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_tournament(
            opponent_pool_config=[
                {"name": "solo_random", "type": "random", "metadata": {}},
            ],
            num_games_per_opponent=2,
        )
        evaluator = TournamentEvaluator(config)
        result = run_eval(evaluator, agent_info)

        assert result.summary_stats.total_games == 2
        assert len(result.errors) == 0


# ---------------------------------------------------------------------------
# Tests: game distribution across opponents
# ---------------------------------------------------------------------------


class TestTournamentGameDistribution:
    """Tests for how games are distributed among opponents."""

    def test_even_distribution_with_no_per_opponent_count(
        self, agent_info, tmp_path
    ):
        """Without num_games_per_opponent, games are split evenly across opponents."""
        config = EvaluationConfig(
            num_games=6,
            max_moves_per_game=50,
            strategy="tournament",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_tournament(
            opponent_pool_config=[
                {"name": "opp_a", "type": "random", "metadata": {}},
                {"name": "opp_b", "type": "random", "metadata": {}},
                {"name": "opp_c", "type": "random", "metadata": {}},
            ],
            num_games_per_opponent=None,
        )
        evaluator = TournamentEvaluator(config)
        result = run_eval(evaluator, agent_info)

        # 6 total games divided among 3 opponents = 2 each
        assert result.summary_stats.total_games == 6

    def test_fixed_games_per_opponent(self, agent_info, tmp_path):
        """With num_games_per_opponent set, each opponent gets exactly that many games."""
        config = EvaluationConfig(
            num_games=10,
            max_moves_per_game=50,
            strategy="tournament",
            save_path=str(tmp_path / "eval"),
            random_seed=42,
        )
        config.configure_for_tournament(
            opponent_pool_config=[
                {"name": "opp_x", "type": "random", "metadata": {}},
                {"name": "opp_y", "type": "random", "metadata": {}},
            ],
            num_games_per_opponent=3,
        )
        evaluator = TournamentEvaluator(config)
        result = run_eval(evaluator, agent_info)

        # 2 opponents * 3 games each = 6 total
        assert result.summary_stats.total_games == 6
