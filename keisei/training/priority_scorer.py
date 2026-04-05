"""PriorityScorer -- ranks matchups by informativeness for scheduling."""

from __future__ import annotations

from collections import Counter, deque

from keisei.config import PriorityScorerConfig
from keisei.training.opponent_store import OpponentEntry, Role


class PriorityScorer:
    """Computes priority scores for candidate pairings.

    Higher scores = more informative matchups that should be played first.
    Maintains in-memory state for pair game counts and round-level repeat tracking.
    """

    def __init__(self, config: PriorityScorerConfig) -> None:
        self.config = config
        self._pair_games: Counter[tuple[int, int]] = Counter()
        self._round_history: deque[set[tuple[int, int]]] = deque(
            maxlen=config.repeat_window_rounds,
        )
        self._current_round_pairs: set[tuple[int, int]] = set()

    def _pair_key(self, id_a: int, id_b: int) -> tuple[int, int]:
        """Canonical (smaller, larger) key for a pair."""
        return (min(id_a, id_b), max(id_a, id_b))

    def record_result(self, id_a: int, id_b: int) -> None:
        """Record that a game was played between these entries."""
        self._pair_games[self._pair_key(id_a, id_b)] += 1

    def record_round_result(self, id_a: int, id_b: int) -> None:
        """Record that this pair was matched in the current round."""
        self._current_round_pairs.add(self._pair_key(id_a, id_b))

    def advance_round(self) -> None:
        """Advance the sliding window: push current round, drop oldest if full."""
        self._round_history.append(self._current_round_pairs)
        self._current_round_pairs = set()

    def _under_sample_bonus(self, id_a: int, id_b: int) -> float:
        count = self._pair_games[self._pair_key(id_a, id_b)]
        return 1.0 / max(1, count)

    def _uncertainty_bonus(self, a: OpponentEntry, b: OpponentEntry) -> float:
        return 1.0 if abs(a.elo_rating - b.elo_rating) < 100 else 0.0

    def _has_recent_fixed(self, a: OpponentEntry, b: OpponentEntry) -> float:
        return 1.0 if a.role == Role.RECENT_FIXED or b.role == Role.RECENT_FIXED else 0.0

    def _lineage_diversity(self, a: OpponentEntry, b: OpponentEntry) -> float:
        if a.lineage_group is None or b.lineage_group is None:
            return 1.0
        return 0.0 if a.lineage_group == b.lineage_group else 1.0

    def _repeat_count(self, id_a: int, id_b: int) -> float:
        key = self._pair_key(id_a, id_b)
        return sum(1 for round_pairs in self._round_history if key in round_pairs)

    def _lineage_closeness(self, a: OpponentEntry, b: OpponentEntry) -> float:
        if a.parent_entry_id == b.id or b.parent_entry_id == a.id:
            return 1.0
        if (
            a.lineage_group is not None
            and b.lineage_group is not None
            and a.lineage_group == b.lineage_group
        ):
            return 0.5
        return 0.0

    def score(self, a: OpponentEntry, b: OpponentEntry) -> float:
        """Compute priority score for a pairing. Higher = more informative."""
        c = self.config
        return (
            c.under_sample_weight * self._under_sample_bonus(a.id, b.id)
            + c.uncertainty_weight * self._uncertainty_bonus(a, b)
            + c.recent_fixed_bonus * self._has_recent_fixed(a, b)
            + c.diversity_weight * self._lineage_diversity(a, b)
            + c.repeat_penalty * self._repeat_count(a.id, b.id)
            + c.lineage_penalty * self._lineage_closeness(a, b)
        )

    def score_round(
        self,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Score all pairings and return sorted by priority (highest first)."""
        scored = [(self.score(a, b), a, b) for a, b in pairings]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(a, b) for _, a, b in scored]
