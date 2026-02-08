"""Elo rating calculation utility for training progress."""

from __future__ import annotations

from typing import Dict, List, Optional

from keisei.shogi.shogi_core_definitions import Color


class EloRatingSystem:
    """Maintain Elo ratings for black and white players."""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0) -> None:
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.black_rating = initial_rating
        self.white_rating = initial_rating
        self.rating_history: List[Dict[str, float]] = []

    @staticmethod
    def _expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(self, winner_color: Optional[Color]) -> Dict[str, float]:
        expected_black = self._expected_score(self.black_rating, self.white_rating)
        expected_white = 1.0 - expected_black

        if winner_color == Color.BLACK:
            actual_black, actual_white = 1.0, 0.0
        elif winner_color == Color.WHITE:
            actual_black, actual_white = 0.0, 1.0
        else:
            actual_black = actual_white = 0.5

        new_black = self.black_rating + self.k_factor * (actual_black - expected_black)
        new_white = self.white_rating + self.k_factor * (actual_white - expected_white)

        self.rating_history.append(
            {
                "black_rating": new_black,
                "white_rating": new_white,
                "difference": new_black - new_white,
            }
        )

        self.black_rating = new_black
        self.white_rating = new_white

        return {
            "black_rating": self.black_rating,
            "white_rating": self.white_rating,
            "rating_difference": self.black_rating - self.white_rating,
        }

    def to_dict(self) -> Dict[str, object]:
        """Serialize Elo state for checkpoint saving."""
        return {
            "black_rating": self.black_rating,
            "white_rating": self.white_rating,
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "rating_history": self.rating_history,
        }

    def restore_from_dict(self, data: Dict[str, object]) -> None:
        """Restore Elo state from checkpoint data."""
        self.black_rating = float(data.get("black_rating", self.initial_rating))
        self.white_rating = float(data.get("white_rating", self.initial_rating))
        history = data.get("rating_history")
        if isinstance(history, list):
            self.rating_history = history

    def get_strength_assessment(self) -> str:
        diff = abs(self.black_rating - self.white_rating)
        if diff < 50:
            return "Balanced"
        if diff < 100:
            return "Slight advantage"
        if diff < 200:
            return "Clear advantage"
        if diff < 400:
            return "Strong advantage"
        return "Overwhelming advantage"
