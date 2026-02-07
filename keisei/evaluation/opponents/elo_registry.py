"""
Simple Elo rating system for opponent management.

This is a simplified replacement for the legacy EloRegistry without backward compatibility.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EloRegistry:
    """Simple Elo rating registry for opponents."""

    def __init__(
        self, file_path: Path, initial_rating: float = 1500.0, k_factor: float = 32.0
    ):
        """
        Initialize the Elo registry.

        Args:
            file_path: Path to save/load ratings
            initial_rating: Starting rating for new players
            k_factor: K-factor for Elo calculations
        """
        self.file_path = file_path
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.load()

    def load(self) -> None:
        """Load ratings from file if it exists."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ratings = {
                        k: float(v) for k, v in data.get("ratings", {}).items()
                    }
                logger.info(f"Loaded {len(self.ratings)} ratings from {self.file_path}")
            except Exception as e:
                logger.warning(f"Failed to load ratings from {self.file_path}: {e}")
                self.ratings = {}

    def save(self) -> None:
        """Save ratings to file."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "ratings": self.ratings,
                "metadata": {
                    "initial_rating": self.initial_rating,
                    "k_factor": self.k_factor,
                },
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.ratings)} ratings to {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to save ratings to {self.file_path}: {e}")

    def get_rating(self, player_id: str) -> float:
        """Get rating for a player, creating if new."""
        if player_id not in self.ratings:
            self.ratings[player_id] = self.initial_rating
        return self.ratings[player_id]

    def set_rating(self, player_id: str, rating: float) -> None:
        """Set rating for a player."""
        self.ratings[player_id] = rating

    def update_ratings(
        self, player1_id: str, player2_id: str, results: List[str]
    ) -> None:
        """
        Update ratings based on match results.

        Args:
            player1_id: First player identifier
            player2_id: Second player identifier
            results: List of game results ('agent_win', 'opponent_win', 'draw')
        """
        if not results:
            return

        rating1 = self.get_rating(player1_id)
        rating2 = self.get_rating(player2_id)

        # Calculate total score for player1
        score1 = 0.0
        for result in results:
            if result == "agent_win":
                score1 += 1.0
            elif result == "draw":
                score1 += 0.5
            # opponent_win adds 0.0

        score2 = len(results) - score1

        # Expected score per game
        expected1 = 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
        expected2 = 1.0 - expected1

        # Update ratings: K * (actual_total - N * expected_per_game)
        n = len(results)
        new_rating1 = rating1 + self.k_factor * (score1 - n * expected1)
        new_rating2 = rating2 + self.k_factor * (score2 - n * expected2)

        self.set_rating(player1_id, new_rating1)
        self.set_rating(player2_id, new_rating2)

        logger.debug(
            f"Updated ratings: {player1_id}: {rating1:.1f} -> {new_rating1:.1f}, "
            f"{player2_id}: {rating2:.1f} -> {new_rating2:.1f}"
        )

    def get_all_ratings(self) -> Dict[str, float]:
        """Get all current ratings."""
        return self.ratings.copy()

    def get_top_players(self, limit: int = 10) -> List[tuple[str, float]]:
        """Get top players by rating."""
        sorted_players = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_players[:limit]
