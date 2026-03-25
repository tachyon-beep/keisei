"""
Elo Tracker for Keisei Shogi DRL Evaluation System.

This module provides a class to manage and update Elo ratings for agents and opponents
based on game outcomes.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_K_FACTOR = 32
DEFAULT_INITIAL_RATING = 1500


class EloTracker:
    """
    Manages Elo ratings for multiple entities (agents/opponents).
    Ratings are stored in a dictionary mapping entity ID to its rating.
    """

    def __init__(
        self,
        initial_ratings: Optional[Dict[str, float]] = None,
        default_k_factor: float = DEFAULT_K_FACTOR,
        default_initial_rating: float = DEFAULT_INITIAL_RATING,
    ):
        """
        Initialize the EloTracker.

        Args:
            initial_ratings: A dictionary of initial ratings {entity_id: rating}.
            default_k_factor: The default K-factor to use for rating updates.
            default_initial_rating: The default rating for new entities.
        """
        self.ratings: Dict[str, float] = (
            initial_ratings.copy() if initial_ratings else {}
        )
        self.default_k_factor: float = default_k_factor
        self.default_initial_rating: float = default_initial_rating
        self.history: List[Dict[str, Any]] = []  # To store rating changes

    def get_rating(self, entity_id: str) -> float:
        """
        Get the current rating of an entity.
        If the entity is not tracked, it will be added with the default initial rating.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            The Elo rating of the entity.
        """
        if entity_id not in self.ratings:
            logger.info(
                f"Entity '{entity_id}' not found. Initializing with default rating {self.default_initial_rating}."
            )
            self.ratings[entity_id] = self.default_initial_rating
        return self.ratings[entity_id]

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate the expected score of player A against player B.
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))
        """
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def update_rating(
        self,
        entity_id_a: str,
        entity_id_b: str,
        score_a: float,
        k_factor_a: Optional[float] = None,
        k_factor_b: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Update the Elo ratings of two entities based on a game outcome.

        Args:
            entity_id_a: ID of the first entity.
            entity_id_b: ID of the second entity.
            score_a: Score of entity A (1 for win, 0.5 for draw, 0 for loss).
            k_factor_a: K-factor for entity A. Uses default if None.
            k_factor_b: K-factor for entity B. Uses default if None.

        Returns:
            A tuple containing the new ratings (new_rating_a, new_rating_b).
        """
        rating_a = self.get_rating(entity_id_a)  # Ensures entity A exists
        rating_b = self.get_rating(entity_id_b)  # Ensures entity B exists

        k_a = k_factor_a if k_factor_a is not None else self.default_k_factor
        k_b = k_factor_b if k_factor_b is not None else self.default_k_factor

        expected_a = self._expected_score(rating_a, rating_b)
        expected_b = self._expected_score(rating_b, rating_a)  # == 1 - expected_a

        score_b = 1.0 - score_a  # If A wins, B loses; if A draws, B draws

        new_rating_a = rating_a + k_a * (score_a - expected_a)
        new_rating_b = rating_b + k_b * (score_b - expected_b)

        self.ratings[entity_id_a] = new_rating_a
        self.ratings[entity_id_b] = new_rating_b

        self.history.append(
            {
                "entity_a": entity_id_a,
                "old_rating_a": rating_a,
                "new_rating_a": new_rating_a,
                "entity_b": entity_id_b,
                "old_rating_b": rating_b,
                "new_rating_b": new_rating_b,
                "score_a": score_a,
                "expected_a": expected_a,
                "k_factor_a": k_a,
                "k_factor_b": k_b,
            }
        )

        logger.debug(
            f"Elo update: {entity_id_a} ({rating_a:.1f} -> {new_rating_a:.1f}), "
            f"{entity_id_b} ({rating_b:.1f} -> {new_rating_b:.1f}) | Score A: {score_a}"
        )

        return new_rating_a, new_rating_b

    def add_entity(self, entity_id: str, rating: Optional[float] = None) -> None:
        """
        Add a new entity to the tracker.

        Args:
            entity_id: The unique identifier of the entity.
            rating: The initial rating. Uses default if None.
        """
        if entity_id in self.ratings:
            logger.warning(f"Entity '{entity_id}' already exists. Not overwriting.")
        else:
            self.ratings[entity_id] = (
                rating if rating is not None else self.default_initial_rating
            )
            logger.info(
                f"Added entity '{entity_id}' with rating {self.ratings[entity_id]}."
            )

    def get_all_ratings(self) -> Dict[str, float]:
        """
        Get all current ratings.

        Returns:
            A dictionary of all entity IDs and their ratings.
        """
        return self.ratings.copy()

    def get_rating_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of rating changes.

        Returns:
            A list of dictionaries, each representing a rating update event.
        """
        return self.history.copy()

    def get_leaderboard(self, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get a leaderboard of entities sorted by rating.

        Args:
            top_n: If specified, return only the top N entities.

        Returns:
            A list of (entity_id, rating) tuples, sorted by rating in descending order.
        """
        sorted_ratings = sorted(
            self.ratings.items(), key=lambda item: item[1], reverse=True
        )
        if top_n is not None:
            return sorted_ratings[:top_n]
        return sorted_ratings

    def reset_ratings(self, initial_ratings: Optional[Dict[str, float]] = None) -> None:
        """
        Resets all ratings to initial values or an empty state.

        Args:
            initial_ratings: Optional dictionary of ratings to reset to.
        """
        self.ratings = initial_ratings.copy() if initial_ratings else {}
        self.history = []
        logger.info("Elo ratings have been reset.")

