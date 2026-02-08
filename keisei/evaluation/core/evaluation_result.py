"""
Evaluation result data structures.

This module defines the result structures that capture the outcomes of evaluation
sessions, including individual game results and comprehensive evaluation summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..analytics.elo_tracker import EloTracker
    from ..analytics.performance_analyzer import PerformanceAnalyzer
    from keisei.config_schema import EvaluationConfig  # Added for type hint
    from .evaluation_context import AgentInfo, EvaluationContext, OpponentInfo


@dataclass
class GameResult:
    """Result of a single game evaluation."""

    game_id: str
    winner: Optional[int]  # 0=agent, 1=opponent, None=draw
    moves_count: int
    duration_seconds: float
    agent_info: AgentInfo
    opponent_info: OpponentInfo
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_agent_win(self) -> bool:
        """True if the agent won this game."""
        return self.winner == 0

    @property
    def is_opponent_win(self) -> bool:
        """True if the opponent won this game."""
        return self.winner == 1

    @property
    def is_draw(self) -> bool:
        """True if the game was a draw (not an error)."""
        return self.winner is None and "error" not in self.metadata

    @property
    def is_error(self) -> bool:
        """True if the game ended due to an error."""
        return "error" in self.metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert GameResult to a dictionary for serialization."""
        return {
            "game_id": self.game_id,
            "winner": self.winner,
            "moves_count": self.moves_count,
            "duration_seconds": self.duration_seconds,
            "agent_info": self.agent_info.to_dict(),  # Uses AgentInfo.to_dict()
            "opponent_info": self.opponent_info.to_dict(),  # Uses OpponentInfo.to_dict()
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameResult":
        """Create GameResult from dictionary."""
        from .evaluation_context import AgentInfo, OpponentInfo  # Local import

        agent_info_data = data.get("agent_info", {})
        opponent_info_data = data.get("opponent_info", {})

        agent_info = AgentInfo.from_dict(agent_info_data)
        opponent_info = OpponentInfo.from_dict(opponent_info_data)

        return cls(
            game_id=data.get("game_id", "unknown_game"),
            winner=data.get("winner"),
            moves_count=data.get("moves_count", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            agent_info=agent_info,
            opponent_info=opponent_info,
            metadata=data.get("metadata", {}),
        )


@dataclass
class SummaryStats:
    """Summary statistics for an evaluation session."""

    total_games: int
    agent_wins: int
    opponent_wins: int
    draws: int
    win_rate: float
    loss_rate: float
    draw_rate: float
    avg_game_length: float
    total_moves: int
    avg_duration_seconds: float

    @classmethod
    def from_games(cls, games: List[GameResult]) -> SummaryStats:
        """Calculate summary statistics from a list of game results."""
        total_games = len(games)
        if total_games == 0:
            return cls(
                total_games=0,
                agent_wins=0,
                opponent_wins=0,
                draws=0,
                win_rate=0.0,
                loss_rate=0.0,
                draw_rate=0.0,
                avg_game_length=0.0,
                total_moves=0,
                avg_duration_seconds=0.0,
            )

        # Exclude error games from statistics
        valid_games = [g for g in games if not g.is_error]
        if not valid_games:
            return cls(
                total_games=0,
                agent_wins=0,
                opponent_wins=0,
                draws=0,
                win_rate=0.0,
                loss_rate=0.0,
                draw_rate=0.0,
                avg_game_length=0.0,
                total_moves=0,
                avg_duration_seconds=0.0,
            )
        total_games = len(valid_games)
        agent_wins = sum(1 for g in valid_games if g.is_agent_win)
        opponent_wins = sum(1 for g in valid_games if g.is_opponent_win)
        draws = sum(1 for g in valid_games if g.is_draw)
        total_moves = sum(g.moves_count for g in valid_games)
        total_duration = sum(g.duration_seconds for g in valid_games)

        return cls(
            total_games=total_games,
            agent_wins=agent_wins,
            opponent_wins=opponent_wins,
            draws=draws,
            win_rate=agent_wins / total_games,
            loss_rate=opponent_wins / total_games,
            draw_rate=draws / total_games,
            avg_game_length=total_moves / total_games,
            total_moves=total_moves,
            avg_duration_seconds=total_duration / total_games,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_games": self.total_games,
            "agent_wins": self.agent_wins,
            "opponent_wins": self.opponent_wins,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "draw_rate": self.draw_rate,
            "avg_game_length": self.avg_game_length,
            "total_moves": self.total_moves,
            "avg_duration_seconds": self.avg_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SummaryStats":
        """Create summary stats from dictionary."""
        return cls(
            total_games=data["total_games"],
            agent_wins=data["agent_wins"],
            opponent_wins=data["opponent_wins"],
            draws=data["draws"],
            win_rate=data["win_rate"],
            loss_rate=data["loss_rate"],
            draw_rate=data["draw_rate"],
            avg_game_length=data["avg_game_length"],
            total_moves=data["total_moves"],
            avg_duration_seconds=data["avg_duration_seconds"],
        )


@dataclass
class EvaluationResult:
    """Comprehensive evaluation results."""

    context: EvaluationContext
    games: List[GameResult]
    summary_stats: SummaryStats
    analytics_data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    elo_tracker: Optional["EloTracker"] = field(default=None, repr=False)

    _analyzer: Optional["PerformanceAnalyzer"] = field(
        default=None, init=False, repr=False
    )
    _stats_calculated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Ensure summary stats are calculated and prepare analyzer."""
        if not self._stats_calculated:
            if self.games and (
                not self.summary_stats
                or self.summary_stats.total_games != len(self.games)
            ):
                self.summary_stats = SummaryStats.from_games(self.games)
            self._stats_calculated = True

        if not self._analyzer and self.games:
            from ..analytics.performance_analyzer import (  # Local import
                PerformanceAnalyzer,
            )

            self._analyzer = PerformanceAnalyzer(self)

    def calculate_analytics(self, force_recalculate: bool = False) -> Dict[str, Any]:
        """
        Calculates (or retrieves cached) detailed performance analytics.
        The results are stored in `self.analytics_data`.

        Args:
            force_recalculate: If True, re-runs all analyses even if already computed.

        Returns:
            A dictionary containing various analytics metrics.
        """
        if not self.games:
            self.analytics_data = {}
            return self.analytics_data

        if not self._analyzer:
            from ..analytics.performance_analyzer import (  # Local import
                PerformanceAnalyzer,
            )

            self._analyzer = PerformanceAnalyzer(self)

        if force_recalculate or not self.analytics_data:
            if self._analyzer:
                self.analytics_data = self._analyzer.run_all_analyses()
            else:
                self.analytics_data = {}

        return self.analytics_data

    def update_elo_ratings(
        self, agent_id: str, opponent_id: str, score_agent: float
    ) -> None:
        """Updates Elo ratings using the attached EloTracker."""
        if self.elo_tracker:
            self.elo_tracker.update_rating(agent_id, opponent_id, score_agent)
        # else: Consider logging a warning

    def get_elo_snapshot(self) -> Optional[Dict[str, float]]:
        """Returns a snapshot of the current Elo ratings from the tracker."""
        if self.elo_tracker:
            return self.elo_tracker.get_all_ratings()
        return None

    def generate_report(self, report_type: str = "text") -> Any:  # Removed **kwargs
        """
        Generates a report of the evaluation results.

        Args:
            report_type: The type of report to generate ("text", "json", "md").

        Returns:
            The generated report (string for text/md, dict for json).
        """
        if not self.analytics_data and self.games:
            self.calculate_analytics()

        from ..analytics.report_generator import ReportGenerator  # Local import

        report_gen = ReportGenerator(
            evaluation_result=self,
            performance_analytics=self.analytics_data,
            elo_snapshot=self.get_elo_snapshot(),
        )

        if report_type == "text":
            return report_gen.generate_text_summary()
        elif report_type == "json":
            return report_gen.generate_json_report()
        elif report_type == "md":
            return report_gen.generate_markdown_report()
        else:
            raise ValueError(f"Unsupported report type: {report_type}")

    def save_report(
        self,
        base_filename: str,
        directory: str = ".",
        formats: Optional[List[str]] = None,
    ) -> List[str]:
        """Saves the report in specified formats."""
        if not self.analytics_data and self.games:
            self.calculate_analytics()

        from ..analytics.report_generator import ReportGenerator  # Local import

        report_gen = ReportGenerator(
            evaluation_result=self,
            performance_analytics=self.analytics_data,
            elo_snapshot=self.get_elo_snapshot(),
        )
        return report_gen.save_report(base_filename, directory, formats)

    def to_dict(self) -> Dict[str, Any]:
        """Convert EvaluationResult to a dictionary for serialization."""
        if not self.analytics_data and self.games:  # Ensure analytics are calculated
            self.calculate_analytics()

        return {
            "context": self.context.to_dict(),
            "games": [game.to_dict() for game in self.games],
            "summary_stats": self.summary_stats.to_dict(),
            "analytics_data": self.analytics_data,
            "errors": self.errors,
            "elo_snapshot": self.get_elo_snapshot(),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        config: Optional["EvaluationConfig"] = None,
        elo_tracker: Optional["EloTracker"] = None,
    ) -> "EvaluationResult":
        """
        Reconstructs an EvaluationResult object from a dictionary.
        """
        from keisei.config_schema import EvaluationConfig
        from .evaluation_context import EvaluationContext

        eval_config_data = data.get("context", {}).get("configuration")
        current_eval_config: Optional[EvaluationConfig] = (
            config  # Use passed config first
        )

        if (
            not current_eval_config
            and eval_config_data
            and isinstance(eval_config_data, dict)
        ):
            try:
                current_eval_config = EvaluationConfig.model_validate(eval_config_data)
            except Exception:
                current_eval_config = EvaluationConfig()

        if not current_eval_config:
            # This case should ideally be handled by the caller providing a valid config
            # or ensuring 'configuration' in context data is complete.
            # For robustness, we might create a default EvaluationConfig if absolutely necessary,
            # but it might lead to incorrect behavior if the original config had specific params.
            # Consider raising an error or logging a significant warning.
            # For now, let's assume EvaluationContext.from_dict can handle config=None if it must.
            pass  # current_eval_config remains None, EvaluationContext.from_dict must handle it

        context = EvaluationContext.from_dict(data["context"], config=current_eval_config)  # type: ignore

        games_data = data.get("games", [])
        games = [GameResult.from_dict(g_data) for g_data in games_data]

        summary_stats_data = data.get("summary_stats")
        if not summary_stats_data:
            # If summary_stats are missing, try to recalculate from games, or create default
            summary_stats = (
                SummaryStats.from_games(games)
                if games
                else SummaryStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            )
        else:
            summary_stats = SummaryStats.from_dict(summary_stats_data)

        instance = cls(
            context=context,
            games=games,
            summary_stats=summary_stats,
            elo_tracker=elo_tracker,
        )

        instance.analytics_data = data.get("analytics_data", {})
        instance.errors = data.get("errors", [])

        if not instance.elo_tracker and data.get("elo_snapshot"):
            from ..analytics.elo_tracker import EloTracker  # Local import

            instance.elo_tracker = EloTracker(initial_ratings=data["elo_snapshot"])

        return instance


def create_game_result(
    game_id: str,
    winner: Optional[int],
    moves_count: int,
    duration_seconds: float,
    agent_info: "AgentInfo",
    opponent_info: "OpponentInfo",
    metadata: Optional[Dict[str, Any]] = None,
) -> GameResult:
    """Convenience helper to build a GameResult."""
    return GameResult(
        game_id=game_id,
        winner=winner,
        moves_count=moves_count,
        duration_seconds=duration_seconds,
        agent_info=agent_info,
        opponent_info=opponent_info,
        metadata=metadata or {},
    )
