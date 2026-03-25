"""
Performance Analyzer for Keisei Shogi DRL Evaluation System.

This module provides a class to perform detailed analysis of evaluation results,
extracting metrics beyond basic win/loss/draw rates.
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.evaluation_context import AgentInfo, OpponentInfo  # Added OpponentInfo
from ..core.evaluation_result import EvaluationResult, GameResult, SummaryStats


class PerformanceAnalyzer:
    """
    Analyzes a collection of game results to extract detailed performance metrics.
    """

    def __init__(self, results: EvaluationResult):
        """
        Initialize the analyzer with evaluation results.

        Args:
            results: The EvaluationResult object containing game data.
        """
        self.results: EvaluationResult = results
        self.games: List[GameResult] = results.games
        self.summary_stats: SummaryStats = results.summary_stats

    def calculate_win_loss_draw_streaks(self) -> Dict[str, Any]:
        """
        Calculates win, loss, and draw streaks.

        Returns:
            A dictionary containing:
                - "max_win_streak": Maximum consecutive wins.
                - "max_loss_streak": Maximum consecutive losses.
                - "max_draw_streak": Maximum consecutive draws.
                - "current_win_streak": Current ongoing win streak.
                - "current_loss_streak": Current ongoing loss streak.
                - "current_draw_streak": Current ongoing draw streak.
        """
        if not self.games:
            return {
                "max_win_streak": 0,
                "max_loss_streak": 0,
                "max_draw_streak": 0,
                "current_win_streak": 0,
                "current_loss_streak": 0,
                "current_draw_streak": 0,
            }

        streaks = {"win": [], "loss": [], "draw": []}
        current_streaks = {"win": 0, "loss": 0, "draw": 0}
        last_outcome = None

        for game in self.games:
            outcome = None
            if game.is_agent_win:
                outcome = "win"
            elif game.is_opponent_win:
                outcome = "loss"
            elif game.is_draw:
                outcome = "draw"

            if outcome == last_outcome:
                current_streaks[outcome] += 1
            else:
                if last_outcome:  # Store the completed streak
                    streaks[last_outcome].append(current_streaks[last_outcome])
                # Reset all current streaks
                current_streaks = {"win": 0, "loss": 0, "draw": 0}
                if outcome:  # Start new streak
                    current_streaks[outcome] = 1
            last_outcome = outcome

        # Store the final streaks
        if last_outcome:
            streaks[last_outcome].append(current_streaks[last_outcome])

        return {
            "max_win_streak": max(streaks["win"]) if streaks["win"] else 0,
            "max_loss_streak": max(streaks["loss"]) if streaks["loss"] else 0,
            "max_draw_streak": max(streaks["draw"]) if streaks["draw"] else 0,
            "current_win_streak": (
                current_streaks["win"] if last_outcome == "win" else 0
            ),
            "current_loss_streak": (
                current_streaks["loss"] if last_outcome == "loss" else 0
            ),
            "current_draw_streak": (
                current_streaks["draw"] if last_outcome == "draw" else 0
            ),
        }

    def analyze_game_length_distribution(
        self, bins: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyzes the distribution of game lengths (number of moves).

        Args:
            bins: Optional list of integers defining the edges of bins for histogram.
                  If None, default bins will be used (e.g., [0, 50, 100, 150, 200, 300, 500]).

        Returns:
            A dictionary containing:
                - "mean_game_length": Average game length.
                - "median_game_length": Median game length.
                - "std_dev_game_length": Standard deviation of game length.
                - "min_game_length": Minimum game length.
                - "max_game_length": Maximum game length.
                - "histogram": A list of tuples (bin_range_str, count) for the histogram.
        """
        if not self.games:
            return {
                "mean_game_length": 0,
                "median_game_length": 0,
                "std_dev_game_length": 0,
                "min_game_length": 0,
                "max_game_length": 0,
                "histogram": [],
            }

        game_lengths = [game.moves_count for game in self.games]

        if not bins:
            bins = [
                0,
                25,
                50,
                75,
                100,
                125,
                150,
                175,
                200,
                250,
                300,
                400,
                500,
            ]  # Default bins

        hist, bin_edges = np.histogram(game_lengths, bins=bins)

        histogram_data = []
        for i in range(len(hist)):
            bin_range_str = f"{bin_edges[i]}-{bin_edges[i+1]}"
            histogram_data.append({"range": bin_range_str, "count": int(hist[i])})

        return {
            "mean_game_length": float(np.mean(game_lengths)) if game_lengths else 0,
            "median_game_length": float(np.median(game_lengths)) if game_lengths else 0,
            "std_dev_game_length": float(np.std(game_lengths)) if game_lengths else 0,
            "min_game_length": min(game_lengths) if game_lengths else 0,
            "max_game_length": max(game_lengths) if game_lengths else 0,
            "histogram": histogram_data,
        }

    def analyze_termination_reasons(self) -> Dict[str, int]:
        """
        Counts the occurrences of different game termination reasons.

        Returns:
            A dictionary where keys are termination reasons and values are their counts.
        """
        if not self.games:
            return {}

        termination_reasons = Counter()
        for game in self.games:
            reason = game.metadata.get("termination_reason", "Unknown")
            # Standardize common reasons if necessary, e.g., checkmate, resignation, timeout
            termination_reasons[reason] += 1
        return dict(termination_reasons)

    def get_performance_by_color(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates agent's performance (win/loss/draw rates) when playing as Sente (Black)
        and Gote (White).

        Returns:
            A dictionary with keys "sente" and "gote", each containing:
                - "wins": Number of wins.
                - "losses": Number of losses.
                - "draws": Number of draws.
                - "total_games": Total games played with that color.
                - "win_rate": Win rate with that color.
        """
        sente_games = [
            g for g in self.games if g.metadata.get("agent_color") == "sente"
        ]  # Assuming metadata stores this
        gote_games = [g for g in self.games if g.metadata.get("agent_color") == "gote"]

        def calculate_color_stats(
            games_list: List[GameResult], color_name: str
        ) -> Dict[str, Any]:
            if not games_list:
                return {
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "total_games": 0,
                    "win_rate": 0.0,
                    "loss_rate": 0.0,
                    "draw_rate": 0.0,
                }

            wins = sum(1 for g in games_list if g.is_agent_win)
            losses = sum(1 for g in games_list if g.is_opponent_win)
            draws = sum(1 for g in games_list if g.is_draw)
            total = len(games_list)

            return {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "total_games": total,
                "win_rate": wins / total if total > 0 else 0.0,
                "loss_rate": losses / total if total > 0 else 0.0,
                "draw_rate": draws / total if total > 0 else 0.0,
            }

        return {
            "sente": calculate_color_stats(sente_games, "sente"),
            "gote": calculate_color_stats(gote_games, "gote"),
        }

    def get_performance_vs_opponent_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes performance against different types of opponents.
        More relevant for TournamentEvaluator or LadderEvaluator.

        Returns:
            A dictionary where keys are opponent types (e.g., "ppo_agent", "heuristic_v1")
            and values are dictionaries of performance stats (wins, losses, draws, win_rate)
            against that opponent type.
        """
        performance_by_opponent: Dict[str, Dict[str, Any]] = {}
        games_by_opponent_type: Dict[str, List[GameResult]] = {}

        for game in self.games:
            opponent_type = (
                game.opponent_info.type
            )  # Assuming OpponentInfo has a 'type'
            if opponent_type not in games_by_opponent_type:
                games_by_opponent_type[opponent_type] = []
            games_by_opponent_type[opponent_type].append(game)

        for opp_type, opp_games in games_by_opponent_type.items():
            if not opp_games:
                continue

            wins = sum(1 for g in opp_games if g.is_agent_win)
            losses = sum(1 for g in opp_games if g.is_opponent_win)
            draws = sum(1 for g in opp_games if g.is_draw)
            total = len(opp_games)

            performance_by_opponent[opp_type] = {
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "total_games": total,
                "win_rate": wins / total if total > 0 else 0.0,
                "loss_rate": losses / total if total > 0 else 0.0,
                "draw_rate": draws / total if total > 0 else 0.0,
            }
        return performance_by_opponent

    def run_all_analyses(self) -> Dict[str, Any]:
        """
        Runs all implemented analysis methods and returns a consolidated dictionary.

        Returns:
            A dictionary where keys are analysis names (e.g., "streaks", "game_length_distribution")
            and values are the results of those analyses.
        """
        analyses: Dict[str, Any] = {}

        analyses["summary_stats"] = self.summary_stats.to_dict()  # Include base summary
        analyses["streaks"] = self.calculate_win_loss_draw_streaks()
        analyses["game_length_distribution"] = self.analyze_game_length_distribution()
        analyses["termination_reasons"] = self.analyze_termination_reasons()
        analyses["performance_by_color"] = self.get_performance_by_color()

        # This might be empty for SingleOpponentEvaluator but useful for others
        performance_vs_opp_types = self.get_performance_vs_opponent_types()
        if performance_vs_opp_types:  # Only add if there's data
            analyses["performance_vs_opponent_types"] = performance_vs_opp_types

        return analyses
