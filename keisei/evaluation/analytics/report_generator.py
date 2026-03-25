"""
Report Generator for Keisei Shogi DRL Evaluation System.

This module provides classes and functions to generate human-readable and machine-parseable
reports from evaluation results and analytics data.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..core.evaluation_context import EvaluationContext
from ..core.evaluation_result import EvaluationResult, GameResult, SummaryStats
from .elo_tracker import EloTracker  # Assuming this is available
from .performance_analyzer import PerformanceAnalyzer  # Assuming this is available


class ReportGenerator:
    """
    Generates various report formats from evaluation data.
    """

    def __init__(
        self,
        evaluation_result: EvaluationResult,
        performance_analytics: Optional[Dict[str, Any]] = None,
        elo_snapshot: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the report generator.

        Args:
            evaluation_result: The core EvaluationResult object.
            performance_analytics: Optional pre-computed analytics from PerformanceAnalyzer.
            elo_snapshot: Optional snapshot of Elo ratings from EloTracker.
        """
        self.result: EvaluationResult = evaluation_result
        self.context: EvaluationContext = evaluation_result.context
        self.summary: SummaryStats = evaluation_result.summary_stats
        self.games: List[GameResult] = evaluation_result.games

        self.analytics: Dict[str, Any] = (
            performance_analytics if performance_analytics is not None else {}
        )
        if (
            not self.analytics
            and hasattr(self.result, "analytics")
            and self.result.analytics
        ):
            self.analytics = (
                self.result.analytics
            )  # Use analytics from result if available

        self.elo_snapshot: Optional[Dict[str, float]] = elo_snapshot
        if (
            not self.elo_snapshot
            and hasattr(self.result, "elo_snapshot")
            and self.result.elo_snapshot
        ):
            self.elo_snapshot = self.result.elo_snapshot

    def _format_header(self) -> str:
        """
        Creates a standard header for text-based reports.
        """
        header = f"Keisei Shogi Evaluation Report\n"
        header += f"================================\n"
        header += f"Session ID: {self.context.session_id}\n"
        header += f"Timestamp: {self.context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"Agent: {self.context.agent_info.name} (Version: {self.context.agent_info.version or 'N/A'})\n"
        header += f"Strategy: {self.context.configuration.strategy.value}\n"
        header += f"Total Games: {self.summary.total_games}\n"
        header += f"--------------------------------\n"
        return header

    def generate_text_summary(self) -> str:
        """
        Generates a concise text summary of the evaluation.
        """
        report_parts = [self._format_header()]

        report_parts.append("Overall Performance:")
        report_parts.append(f"  Win Rate: {self.summary.win_rate:.2%}")
        report_parts.append(f"  Wins: {self.summary.agent_wins}")
        report_parts.append(f"  Losses: {self.summary.opponent_wins}")
        report_parts.append(f"  Draws: {self.summary.draws}")
        report_parts.append(
            f"  Avg Game Length (moves): {self.summary.avg_game_length:.1f}"
        )
        report_parts.append(f"--------------------------------")

        if self.analytics:
            report_parts.append("Detailed Analytics:")
            if "streaks" in self.analytics:
                s = self.analytics["streaks"]
                report_parts.append(f"  Max Win Streak: {s.get('max_win_streak', 0)}")
                report_parts.append(f"  Max Loss Streak: {s.get('max_loss_streak', 0)}")

            if "performance_by_color" in self.analytics:
                pbc = self.analytics["performance_by_color"]
                if pbc.get("sente") and pbc["sente"]["total_games"] > 0:
                    sente = pbc["sente"]
                    report_parts.append(
                        f"  As Sente (Black): {sente['wins']}W-{sente['losses']}L-{sente['draws']}D (WR: {sente['win_rate']:.2%})"
                    )
                if pbc.get("gote") and pbc["gote"]["total_games"] > 0:
                    gote = pbc["gote"]
                    report_parts.append(
                        f"  As Gote (White): {gote['wins']}W-{gote['losses']}L-{gote['draws']}D (WR: {gote['win_rate']:.2%})"
                    )

            if "termination_reasons" in self.analytics:
                tr = self.analytics["termination_reasons"]
                report_parts.append(f"  Termination Reasons: {json.dumps(tr)}")
            report_parts.append(f"--------------------------------")

        if self.elo_snapshot:
            report_parts.append("Elo Ratings Snapshot:")
            agent_elo = self.elo_snapshot.get(self.context.agent_info.name)
            if agent_elo is not None:
                report_parts.append(
                    f"  Agent ({self.context.agent_info.name}): {agent_elo:.1f}"
                )
            # Could list other relevant Elo ratings if available
            report_parts.append(f"--------------------------------")

        if self.result.errors:
            report_parts.append("Errors Encountered:")
            for err in self.result.errors:
                report_parts.append(f"  - {err}")
            report_parts.append(f"--------------------------------")

        return "\n".join(report_parts)

    def generate_json_report(self) -> Dict[str, Any]:
        """
        Generates a comprehensive JSON report.
        """
        report_data: Dict[str, Any] = {
            "evaluation_context": self.context.to_dict(),
            "summary_stats": self.summary.to_dict(),
            "performance_analytics": self.analytics,  # Already a dict
            "elo_snapshot": self.elo_snapshot,  # Already a dict or None
            "errors": self.result.errors,
        }

        # Optionally include detailed game logs if configured (can be large)
        # if self.context.configuration.save_games_in_report: # Example config flag
        # report_data["game_results"] = [game.to_dict() for game in self.games]
        # Assuming GameResult has a to_dict() method

        return report_data

    def generate_markdown_report(self) -> str:
        """
        Generates a Markdown formatted report.
        """
        md_parts = []

        md_parts.append(f"# Keisei Shogi Evaluation Report")
        md_parts.append(f"**Session ID:** `{self.context.session_id}`")
        md_parts.append(
            f"**Timestamp:** {self.context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        md_parts.append(
            f"**Agent:** `{self.context.agent_info.name}` (Version: `{self.context.agent_info.version or 'N/A'}`)"
        )
        md_parts.append(f"**Strategy:** `{self.context.configuration.strategy.value}`")
        md_parts.append(f"**Total Games:** {self.summary.total_games}")
        md_parts.append("***")

        md_parts.append(f"## Overall Performance")
        md_parts.append(f"- **Win Rate:** {self.summary.win_rate:.2%}")
        md_parts.append(f"- **Wins:** {self.summary.agent_wins}")
        md_parts.append(f"- **Losses:** {self.summary.opponent_wins}")
        md_parts.append(f"- **Draws:** {self.summary.draws}")
        md_parts.append(
            f"- **Avg Game Length (moves):** {self.summary.avg_game_length:.1f}"
        )
        md_parts.append("***")

        if self.analytics:
            md_parts.append(f"## Detailed Analytics")
            if "streaks" in self.analytics:
                s = self.analytics["streaks"]
                md_parts.append(f"### Streaks")
                md_parts.append(f"- Max Win Streak: {s.get('max_win_streak', 0)}")
                md_parts.append(f"- Max Loss Streak: {s.get('max_loss_streak', 0)}")

            if "performance_by_color" in self.analytics:
                pbc = self.analytics["performance_by_color"]
                md_parts.append(f"### Performance by Color")
                if pbc.get("sente") and pbc["sente"]["total_games"] > 0:
                    sente = pbc["sente"]
                    md_parts.append(
                        f"- **As Sente (Black):** {sente['wins']}W - {sente['losses']}L - {sente['draws']}D (Win Rate: {sente['win_rate']:.2%})"
                    )
                if pbc.get("gote") and pbc["gote"]["total_games"] > 0:
                    gote = pbc["gote"]
                    md_parts.append(
                        f"- **As Gote (White):** {gote['wins']}W - {gote['losses']}L - {gote['draws']}D (Win Rate: {gote['win_rate']:.2%})"
                    )

            if "game_length_distribution" in self.analytics:
                gld = self.analytics["game_length_distribution"]
                md_parts.append(f"### Game Length Distribution")
                md_parts.append(f"- Mean: {gld.get('mean_game_length', 0):.1f} moves")
                md_parts.append(
                    f"- Median: {gld.get('median_game_length', 0):.0f} moves"
                )
                md_parts.append(
                    f"- Min/Max: {gld.get('min_game_length', 0)} / {gld.get('max_game_length', 0)} moves"
                )
                # Could add histogram table if desired

            if "termination_reasons" in self.analytics:
                tr = self.analytics["termination_reasons"]
                md_parts.append(f"### Termination Reasons")
                md_parts.append("```json")
                md_parts.append(json.dumps(tr, indent=2))
                md_parts.append("```")
            md_parts.append("***")

        if self.elo_snapshot:
            md_parts.append(f"## Elo Ratings Snapshot")
            agent_elo = self.elo_snapshot.get(self.context.agent_info.name)
            if agent_elo is not None:
                md_parts.append(
                    f"- **Agent ({self.context.agent_info.name}):** {agent_elo:.1f}"
                )
            # Could list other relevant Elo ratings as a table
            md_parts.append("***")

        if self.result.errors:
            md_parts.append(f"## Errors Encountered")
            for err in self.result.errors:
                md_parts.append(f"- `{err}`")
            md_parts.append("***")

        return "\n".join(md_parts)

    def save_report(
        self,
        base_filename: str,
        directory: str = ".",
        formats: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Saves the report in specified formats to a directory.

        Args:
            base_filename: The base name for the report files (e.g., "evaluation_run_XYZ").
            directory: The directory to save reports in. Defaults to current directory.
            formats: A list of formats to save (e.g., ["txt", "json", "md"]).
                     Defaults to ["txt", "json"].

        Returns:
            A list of paths to the saved report files.
        """
        if formats is None:
            formats = ["txt", "json"]

        os.makedirs(directory, exist_ok=True)
        saved_files: List[str] = []

        if "txt" in formats:
            content = self.generate_text_summary()
            filepath = os.path.join(directory, f"{base_filename}_summary.txt")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            saved_files.append(filepath)

        if "json" in formats:
            content_dict = self.generate_json_report()
            filepath = os.path.join(directory, f"{base_filename}_report.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(content_dict, f, indent=2)
            saved_files.append(filepath)

        if "md" in formats:
            content = self.generate_markdown_report()
            filepath = os.path.join(directory, f"{base_filename}_report.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            saved_files.append(filepath)

        return saved_files
