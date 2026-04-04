"""RoleEloTracker: per-context Elo updates for role-specific ratings."""

from __future__ import annotations

import logging

from keisei.config import RoleEloConfig
from keisei.training.opponent_store import (
    EloColumn,
    OpponentEntry,
    OpponentStore,
    Role,
    compute_elo_update,
)

logger = logging.getLogger(__name__)


class RoleEloTracker:
    """Computes and stores role-specific Elo updates.

    Wraps the existing compute_elo_update function with role-awareness.
    The composite elo_rating is NOT modified by this tracker — it is
    managed independently by the existing training loop and tournament.
    """

    def __init__(self, store: OpponentStore, config: RoleEloConfig) -> None:
        self.store = store
        self.config = config

    def update_from_result(
        self,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
        result_score: float,
        match_context: str,
    ) -> None:
        """Compute role-specific Elo deltas and write to DB atomically.

        Args:
            entry_a: First participant.
            entry_b: Second participant.
            result_score: 1.0 = A wins, 0.5 = draw, 0.0 = A loses.
            match_context: One of 'frontier', 'dynamic', 'recent',
                'historical', or 'cross_dynamic_recent'.
        """
        column_a, column_b, k = self._resolve_context(
            entry_a, entry_b, match_context
        )

        elo_a = self._get_entry_role_elo(entry_a, column_a)
        elo_b = self._get_entry_role_elo(entry_b, column_b)

        new_a, new_b = compute_elo_update(elo_a, elo_b, result_score, k=k)

        with self.store.transaction():
            self.store.update_role_elo(entry_a.id, column_a, new_a)
            self.store.update_role_elo(entry_b.id, column_b, new_b)

    def get_role_elos(self, entry_id: int) -> dict[str, float]:
        """Returns dict of {role: elo_value} for an entry."""
        entry = self.store._get_entry(entry_id)
        if entry is None:
            return {}
        return {
            EloColumn.FRONTIER: entry.elo_frontier,
            EloColumn.DYNAMIC: entry.elo_dynamic,
            EloColumn.RECENT: entry.elo_recent,
            EloColumn.HISTORICAL: entry.elo_historical,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_context(
        self,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
        match_context: str,
    ) -> tuple[EloColumn, EloColumn, float]:
        """Determine which Elo columns to update and the K-factor."""
        if match_context == "historical":
            return EloColumn.HISTORICAL, EloColumn.HISTORICAL, self.config.historical_k

        if match_context == "frontier":
            return EloColumn.FRONTIER, EloColumn.FRONTIER, self.config.frontier_k

        if match_context == "dynamic":
            return EloColumn.DYNAMIC, EloColumn.DYNAMIC, self.config.dynamic_k

        if match_context == "recent":
            return EloColumn.RECENT, EloColumn.RECENT, self.config.recent_k

        if match_context == "cross_dynamic_recent":
            # Dynamic vs Recent Fixed: different columns for each
            if entry_a.role == Role.DYNAMIC:
                return EloColumn.DYNAMIC, EloColumn.RECENT, self.config.dynamic_k
            else:
                return EloColumn.RECENT, EloColumn.DYNAMIC, self.config.dynamic_k

        raise ValueError(f"Unknown match context: {match_context!r}")

    @staticmethod
    def _get_entry_role_elo(entry: OpponentEntry, column: EloColumn) -> float:
        """Read the role-specific Elo from an entry."""
        return getattr(entry, column.value)

    @staticmethod
    def determine_match_context(entry_a: OpponentEntry, entry_b: OpponentEntry) -> str:
        """Infer the match context from participant roles."""
        roles = {entry_a.role, entry_b.role}

        if Role.FRONTIER_STATIC in roles:
            return "frontier"

        if roles == {Role.DYNAMIC}:
            return "dynamic"

        if roles == {Role.RECENT_FIXED}:
            return "recent"

        if Role.DYNAMIC in roles and Role.RECENT_FIXED in roles:
            return "cross_dynamic_recent"

        # Fallback: treat as dynamic context
        return "dynamic"
