"""
Enhanced Opponent Management for Keisei Evaluation System.

Provides adaptive opponent selection strategies, dynamic difficulty balancing,
and historical performance tracking for optimized training experiences.
"""

import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..core import GameResult, OpponentInfo

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Opponent selection strategies."""

    RANDOM = "random"
    ELO_BASED = "elo_based"
    ADAPTIVE_DIFFICULTY = "adaptive_difficulty"
    CURRICULUM_LEARNING = "curriculum_learning"
    DIVERSITY_MAXIMIZING = "diversity_maximizing"


@dataclass
class OpponentPerformanceData:
    """Performance data for a specific opponent."""

    opponent_name: str
    total_games: int = 0
    wins_against: int = 0
    losses_against: int = 0
    draws_against: int = 0
    last_played: Optional[datetime] = None
    average_game_length: float = 0.0
    elo_rating: float = 1200.0
    difficulty_level: float = 0.5  # 0-1 scale

    # Learning curve data
    recent_win_rates: List[float] = field(default_factory=list)
    performance_trend: str = "stable"  # "improving", "declining", "stable"

    # Metadata
    opponent_type: str = "unknown"
    tags: Set[str] = field(default_factory=set)

    @property
    def win_rate_against(self) -> float:
        """Calculate win rate against this opponent."""
        if self.total_games == 0:
            return 0.0
        return self.wins_against / self.total_games

    @property
    def games_since_last_played(self) -> int:
        """Number of games since last played (approximation)."""
        if self.last_played is None:
            return float("inf")
        days_since = (datetime.now() - self.last_played).days
        return max(0, days_since * 10)  # Rough approximation

    def update_with_result(self, game_result: GameResult) -> None:
        """Update performance data with new game result."""
        self.total_games += 1
        self.last_played = datetime.now()

        if game_result.is_agent_win:
            self.wins_against += 1
        elif game_result.is_opponent_win:
            self.losses_against += 1
        else:
            self.draws_against += 1

        # Update average game length
        if self.total_games == 1:
            self.average_game_length = game_result.moves_count
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_game_length = (
                alpha * game_result.moves_count + (1 - alpha) * self.average_game_length
            )

        # Update recent win rates (keep last 10 games)
        self.recent_win_rates.append(1.0 if game_result.is_agent_win else 0.0)
        if len(self.recent_win_rates) > 10:
            self.recent_win_rates.pop(0)

        # Update performance trend
        self._update_performance_trend()

    def _update_performance_trend(self) -> None:
        """Update performance trend based on recent games."""
        if len(self.recent_win_rates) < 5:
            self.performance_trend = "stable"
            return

        # Simple trend detection using linear regression
        x = list(range(len(self.recent_win_rates)))
        y = self.recent_win_rates

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope > 0.05:
            self.performance_trend = "improving"
        elif slope < -0.05:
            self.performance_trend = "declining"
        else:
            self.performance_trend = "stable"


class EnhancedOpponentManager:
    """
    Enhanced opponent management with adaptive selection strategies.

    Features:
    - Multiple selection strategies
    - Adaptive difficulty balancing
    - Historical performance tracking
    - Curriculum learning support
    - Diversity-aware opponent selection
    """

    def __init__(
        self,
        opponent_data_file: Optional[Path] = None,
        target_win_rate: float = 0.55,
        difficulty_adaptation_rate: float = 0.1,
        curriculum_progression_threshold: float = 0.65,
    ):
        self.opponent_data_file = opponent_data_file or Path(
            "opponent_performance_data.json"
        )
        self.target_win_rate = target_win_rate
        self.difficulty_adaptation_rate = difficulty_adaptation_rate
        self.curriculum_progression_threshold = curriculum_progression_threshold

        # Performance tracking
        self.opponent_data: Dict[str, OpponentPerformanceData] = {}
        self.available_opponents: List[OpponentInfo] = []

        # Strategy state
        self.current_strategy = SelectionStrategy.ADAPTIVE_DIFFICULTY
        self.curriculum_level = 1
        self.max_curriculum_level = 5

        # Load existing data
        self._load_opponent_data()

        logger.info(
            f"EnhancedOpponentManager initialized with {len(self.opponent_data)} tracked opponents"
        )

    def register_opponents(self, opponents: List[OpponentInfo]) -> None:
        """Register available opponents for selection."""
        self.available_opponents = opponents

        # Initialize performance data for new opponents
        for opponent in opponents:
            if opponent.name not in self.opponent_data:
                self.opponent_data[opponent.name] = OpponentPerformanceData(
                    opponent_name=opponent.name,
                    opponent_type=opponent.type,
                    difficulty_level=self._estimate_initial_difficulty(opponent),
                )

        logger.info(f"Registered {len(opponents)} opponents")

    def select_opponent(
        self,
        strategy: Optional[SelectionStrategy] = None,
        agent_current_win_rate: Optional[float] = None,
        exclude_recent: bool = True,
        diversity_factor: float = 0.3,
    ) -> Optional[OpponentInfo]:
        """
        Select opponent based on specified strategy.

        Args:
            strategy: Selection strategy to use
            agent_current_win_rate: Current agent win rate for adaptive selection
            exclude_recent: Whether to avoid recently played opponents
            diversity_factor: Factor for diversity-aware selection (0-1)
        """
        if not self.available_opponents:
            return None

        selection_strategy = strategy or self.current_strategy

        # Filter out recently played opponents if requested
        candidates = self.available_opponents
        if exclude_recent:
            candidates = self._filter_recent_opponents(candidates)

        if not candidates:
            candidates = self.available_opponents  # Fallback to all opponents

        # Apply selection strategy
        if selection_strategy == SelectionStrategy.RANDOM:
            return random.choice(candidates)
        elif selection_strategy == SelectionStrategy.ELO_BASED:
            return self._select_by_elo(candidates, agent_current_win_rate)
        elif selection_strategy == SelectionStrategy.ADAPTIVE_DIFFICULTY:
            return self._select_adaptive_difficulty(candidates, agent_current_win_rate)
        elif selection_strategy == SelectionStrategy.CURRICULUM_LEARNING:
            return self._select_curriculum_learning(candidates, agent_current_win_rate)
        elif selection_strategy == SelectionStrategy.DIVERSITY_MAXIMIZING:
            return self._select_diversity_maximizing(candidates, diversity_factor)
        else:
            logger.warning(f"Unknown strategy {selection_strategy}, using random")
            return random.choice(candidates)

    def update_performance(self, game_result: GameResult) -> None:
        """Update opponent performance data with game result."""
        opponent_name = game_result.opponent_info.name

        if opponent_name not in self.opponent_data:
            self.opponent_data[opponent_name] = OpponentPerformanceData(
                opponent_name=opponent_name,
                opponent_type=game_result.opponent_info.type,
            )

        self.opponent_data[opponent_name].update_with_result(game_result)

        # Adaptive difficulty adjustment
        self._adjust_difficulty_levels(game_result)

        # Curriculum progression check
        self._check_curriculum_progression()

        # Periodic save
        if random.random() < 0.1:  # Save 10% of the time
            self._save_opponent_data()

    def get_opponent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive opponent statistics."""
        if not self.opponent_data:
            return {"total_opponents": 0}

        total_games = sum(data.total_games for data in self.opponent_data.values())
        total_opponents = len(self.opponent_data)

        # Win rate distribution
        win_rates = [
            data.win_rate_against
            for data in self.opponent_data.values()
            if data.total_games > 0
        ]
        avg_win_rate = np.mean(win_rates) if win_rates else 0.0

        # Difficulty distribution
        difficulty_levels = [
            data.difficulty_level for data in self.opponent_data.values()
        ]
        avg_difficulty = np.mean(difficulty_levels) if difficulty_levels else 0.5

        # Most and least played
        most_played = max(
            self.opponent_data.values(), key=lambda x: x.total_games, default=None
        )
        least_played = min(
            [data for data in self.opponent_data.values() if data.total_games > 0],
            key=lambda x: x.total_games,
            default=None,
        )

        return {
            "total_opponents": total_opponents,
            "total_games": total_games,
            "average_win_rate": avg_win_rate,
            "average_difficulty": avg_difficulty,
            "curriculum_level": self.curriculum_level,
            "current_strategy": self.current_strategy.value,
            "most_played_opponent": most_played.opponent_name if most_played else None,
            "least_played_opponent": (
                least_played.opponent_name if least_played else None
            ),
            "opponents_by_difficulty": self._group_opponents_by_difficulty(),
        }

    def _select_by_elo(
        self, candidates: List[OpponentInfo], agent_win_rate: Optional[float]
    ) -> OpponentInfo:
        """Select opponent based on ELO ratings."""
        # Estimate agent ELO from win rate
        if agent_win_rate is not None:
            agent_elo = self._win_rate_to_elo(agent_win_rate)
        else:
            agent_elo = 1200.0  # Default ELO

        # Calculate ELO-based probabilities
        opponent_scores = []
        for opponent in candidates:
            data = self.opponent_data.get(opponent.name)
            if data:
                opponent_elo = data.elo_rating
                # Prefer opponents with similar ELO (±200 points ideal)
                elo_diff = abs(agent_elo - opponent_elo)
                score = 1.0 / (1.0 + elo_diff / 200.0)
                opponent_scores.append(score)
            else:
                opponent_scores.append(0.5)  # Default score for unknown opponents

        # Weighted random selection
        return self._weighted_random_choice(candidates, opponent_scores)

    def _select_adaptive_difficulty(
        self, candidates: List[OpponentInfo], agent_win_rate: Optional[float]
    ) -> OpponentInfo:
        """Select opponent with adaptive difficulty balancing."""
        if agent_win_rate is None:
            return random.choice(candidates)

        # Target difficulty based on current performance
        if agent_win_rate > self.target_win_rate + 0.1:
            # Agent performing well, increase difficulty
            target_difficulty = min(1.0, agent_win_rate + 0.2)
        elif agent_win_rate < self.target_win_rate - 0.1:
            # Agent struggling, decrease difficulty
            target_difficulty = max(0.1, agent_win_rate - 0.1)
        else:
            # Performance near target, maintain current difficulty
            target_difficulty = 0.5

        # Find opponents closest to target difficulty
        opponent_scores = []
        for opponent in candidates:
            data = self.opponent_data.get(opponent.name)
            if data:
                difficulty_diff = abs(data.difficulty_level - target_difficulty)
                score = 1.0 / (1.0 + difficulty_diff * 5)  # Sharper preference

                # Bonus for less recently played opponents
                recency_bonus = 1.0 - 1.0 / (1.0 + data.games_since_last_played / 10.0)
                score *= 1.0 + recency_bonus * 0.3

                opponent_scores.append(score)
            else:
                opponent_scores.append(0.3)  # Lower score for unknown opponents

        return self._weighted_random_choice(candidates, opponent_scores)

    def _select_curriculum_learning(
        self, candidates: List[OpponentInfo], agent_win_rate: Optional[float]
    ) -> OpponentInfo:
        """Select opponent based on curriculum learning progression."""
        # Map curriculum level to difficulty range
        level_difficulty_ranges = {
            1: (0.0, 0.3),  # Easy opponents
            2: (0.2, 0.5),  # Easy-medium opponents
            3: (0.4, 0.7),  # Medium opponents
            4: (0.6, 0.9),  # Medium-hard opponents
            5: (0.8, 1.0),  # Hard opponents
        }

        min_diff, max_diff = level_difficulty_ranges.get(
            self.curriculum_level, (0.0, 1.0)
        )

        # Filter candidates by curriculum level
        suitable_candidates = []
        for opponent in candidates:
            data = self.opponent_data.get(opponent.name)
            if data and min_diff <= data.difficulty_level <= max_diff:
                suitable_candidates.append(opponent)

        if not suitable_candidates:
            suitable_candidates = candidates  # Fallback

        # Within curriculum level, use adaptive difficulty
        return self._select_adaptive_difficulty(suitable_candidates, agent_win_rate)

    def _select_diversity_maximizing(
        self, candidates: List[OpponentInfo], diversity_factor: float
    ) -> OpponentInfo:
        """Select opponent to maximize diversity of experience."""
        opponent_scores = []

        for opponent in candidates:
            data = self.opponent_data.get(opponent.name)
            if data:
                # Score based on how infrequently this opponent has been played
                play_frequency = data.total_games / max(
                    1, sum(d.total_games for d in self.opponent_data.values())
                )
                diversity_score = 1.0 - play_frequency

                # Factor in opponent type diversity
                type_frequency = sum(
                    1
                    for d in self.opponent_data.values()
                    if d.opponent_type == data.opponent_type and d.total_games > 0
                ) / max(1, len(self.opponent_data))

                type_diversity_score = 1.0 - type_frequency

                # Combine scores
                total_score = (
                    diversity_score * diversity_factor
                    + type_diversity_score * (1 - diversity_factor)
                )
                opponent_scores.append(total_score)
            else:
                # High score for never-played opponents
                opponent_scores.append(1.0)

        return self._weighted_random_choice(candidates, opponent_scores)

    def _filter_recent_opponents(
        self, candidates: List[OpponentInfo], recent_threshold: int = 5
    ) -> List[OpponentInfo]:
        """Filter out recently played opponents."""
        filtered = []
        for opponent in candidates:
            data = self.opponent_data.get(opponent.name)
            if not data or data.games_since_last_played > recent_threshold:
                filtered.append(opponent)
        return filtered or candidates  # Return all if no non-recent opponents

    def _adjust_difficulty_levels(self, game_result: GameResult) -> None:
        """Adjust opponent difficulty levels based on game outcomes."""
        opponent_name = game_result.opponent_info.name
        data = self.opponent_data.get(opponent_name)

        if not data:
            return

        # Adjust difficulty based on result
        if game_result.is_agent_win:
            # Agent won, this opponent might be too easy
            data.difficulty_level = min(
                1.0, data.difficulty_level + self.difficulty_adaptation_rate
            )
        elif game_result.is_opponent_win:
            # Agent lost, this opponent might be too hard
            data.difficulty_level = max(
                0.0, data.difficulty_level - self.difficulty_adaptation_rate
            )
        # Draws don't change difficulty

    def _check_curriculum_progression(self) -> None:
        """Check if agent is ready to progress in curriculum."""
        if self.curriculum_level >= self.max_curriculum_level:
            return

        # Calculate recent overall win rate
        recent_games = []
        for data in self.opponent_data.values():
            recent_games.extend(data.recent_win_rates)

        if len(recent_games) < 20:  # Need sufficient data
            return

        recent_win_rate = np.mean(recent_games[-20:])  # Last 20 games

        if recent_win_rate >= self.curriculum_progression_threshold:
            self.curriculum_level += 1
            logger.info(
                f"Curriculum progression! Advanced to level {self.curriculum_level}"
            )

    def _estimate_initial_difficulty(self, opponent: OpponentInfo) -> float:
        """Estimate initial difficulty for new opponent."""
        # Simple heuristic based on opponent type
        type_difficulties = {
            "random": 0.1,
            "heuristic": 0.3,
            "easy_ai": 0.4,
            "medium_ai": 0.6,
            "hard_ai": 0.8,
            "ppo_agent": 0.7,
            "expert": 0.9,
        }

        base_difficulty = type_difficulties.get(opponent.type.lower(), 0.5)

        # Add some randomness
        return max(0.0, min(1.0, base_difficulty + random.uniform(-0.1, 0.1)))

    def _win_rate_to_elo(self, win_rate: float) -> float:
        """Convert win rate to approximate ELO rating."""
        if win_rate <= 0:
            return 800.0
        elif win_rate >= 1:
            return 2000.0
        else:
            # Simple conversion: 50% win rate = 1200 ELO
            return 1200.0 + 400.0 * math.log(win_rate / (1 - win_rate)) / math.log(10)

    def _weighted_random_choice(
        self, choices: List[OpponentInfo], weights: List[float]
    ) -> OpponentInfo:
        """Select random choice based on weights."""
        if not choices:
            raise ValueError("No choices available")

        if len(weights) != len(choices):
            return random.choice(choices)

        total_weight = sum(weights)
        if total_weight <= 0:
            return random.choice(choices)

        r = random.uniform(0, total_weight)
        cumulative = 0

        for choice, weight in zip(choices, weights):
            cumulative += weight
            if r <= cumulative:
                return choice

        return choices[-1]  # Fallback

    def _group_opponents_by_difficulty(self) -> Dict[str, List[str]]:
        """Group opponents by difficulty level."""
        groups = {"easy": [], "medium": [], "hard": []}

        for data in self.opponent_data.values():
            if data.difficulty_level < 0.4:
                groups["easy"].append(data.opponent_name)
            elif data.difficulty_level < 0.7:
                groups["medium"].append(data.opponent_name)
            else:
                groups["hard"].append(data.opponent_name)

        return groups

    def _load_opponent_data(self) -> None:
        """Load opponent performance data from file."""
        if not self.opponent_data_file.exists():
            return

        try:
            with open(self.opponent_data_file, "r") as f:
                data = json.load(f)

            for name, info in data.items():
                perf_data = OpponentPerformanceData(opponent_name=name)
                perf_data.total_games = info.get("total_games", 0)
                perf_data.wins_against = info.get("wins_against", 0)
                perf_data.losses_against = info.get("losses_against", 0)
                perf_data.draws_against = info.get("draws_against", 0)
                perf_data.average_game_length = info.get("average_game_length", 0.0)
                perf_data.elo_rating = info.get("elo_rating", 1200.0)
                perf_data.difficulty_level = info.get("difficulty_level", 0.5)
                perf_data.recent_win_rates = info.get("recent_win_rates", [])
                perf_data.performance_trend = info.get("performance_trend", "stable")
                perf_data.opponent_type = info.get("opponent_type", "unknown")
                perf_data.tags = set(info.get("tags", []))

                if info.get("last_played"):
                    perf_data.last_played = datetime.fromisoformat(info["last_played"])

                self.opponent_data[name] = perf_data

            logger.info(f"Loaded data for {len(self.opponent_data)} opponents")

        except Exception as e:
            logger.error(f"Failed to load opponent data: {e}")

    def _save_opponent_data(self) -> None:
        """Save opponent performance data to file."""
        try:
            data = {}
            for name, perf_data in self.opponent_data.items():
                data[name] = {
                    "total_games": perf_data.total_games,
                    "wins_against": perf_data.wins_against,
                    "losses_against": perf_data.losses_against,
                    "draws_against": perf_data.draws_against,
                    "average_game_length": perf_data.average_game_length,
                    "elo_rating": perf_data.elo_rating,
                    "difficulty_level": perf_data.difficulty_level,
                    "recent_win_rates": perf_data.recent_win_rates,
                    "performance_trend": perf_data.performance_trend,
                    "opponent_type": perf_data.opponent_type,
                    "tags": list(perf_data.tags),
                    "last_played": (
                        perf_data.last_played.isoformat()
                        if perf_data.last_played
                        else None
                    ),
                }

            with open(self.opponent_data_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved opponent data to {self.opponent_data_file}")

        except Exception as e:
            logger.error(f"Failed to save opponent data: {e}")
