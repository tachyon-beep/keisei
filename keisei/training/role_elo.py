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

        For the 'historical' context, only entry_a (the learner) is updated.
        Entry_b (the historical anchor) keeps its frozen Elo.
        """
        column_a, column_b, k = self._resolve_context(
            entry_a, entry_b, match_context
        )

        elo_a = self._get_entry_role_elo(entry_a, column_a)
        # When column_b is None (historical anchor), read its elo_historical
        # for accurate Elo math but don't persist the update.
        read_column_b = column_b if column_b is not None else EloColumn.HISTORICAL
        elo_b = self._get_entry_role_elo(entry_b, read_column_b)

        new_a, new_b = compute_elo_update(elo_a, elo_b, result_score, k=k)

        with self.store.transaction():
            self.store.update_role_elo(entry_a.id, column_a, new_a)
            if column_b is not None:
                self.store.update_role_elo(entry_b.id, column_b, new_b)

    def k_for_context(self, match_context: str) -> float:
        """Return the K-factor for a given match context.

        Used by the tournament to apply context-appropriate K-factors to the
        composite elo_rating, not just the role-specific columns.
        """
        context_k = {
            "frontier": self.config.frontier_k,
            "dynamic": self.config.dynamic_k,
            "recent": self.config.recent_k,
            "historical": self.config.historical_k,
            "cross_dynamic_recent": self.config.dynamic_k,
        }
        return context_k.get(match_context, self.config.frontier_k)

    def get_role_elos(self, entry_id: int) -> dict[EloColumn, float]:
        """Returns dict of {EloColumn: elo_value} for an entry."""
        entry = self.store.get_entry(entry_id)
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
    ) -> tuple[EloColumn, EloColumn | None, float]:
        """Determine which Elo columns to update and the K-factor.

        Returns None for column_b when the second participant is a frozen
        anchor (historical benchmarks) whose Elo should not drift.
        """
        if match_context == "historical":
            # entry_a's elo_historical tracks "how well this entry performs against
            # historical benchmarks" — distinct from elo_dynamic/elo_frontier.
            # entry_b is a frozen anchor — read its elo_historical for the
            # computation, but column_b=None signals "don't write back".
            return EloColumn.HISTORICAL, None, self.config.historical_k

        if match_context == "frontier":
            return EloColumn.FRONTIER, EloColumn.FRONTIER, self.config.frontier_k

        if match_context == "dynamic":
            return EloColumn.DYNAMIC, EloColumn.DYNAMIC, self.config.dynamic_k

        if match_context == "recent":
            return EloColumn.RECENT, EloColumn.RECENT, self.config.recent_k

        if match_context == "cross_dynamic_recent":
            # Dynamic vs Recent Fixed: different columns for each, K tracks entry_a's role
            if entry_a.role == Role.DYNAMIC:
                return EloColumn.DYNAMIC, EloColumn.RECENT, self.config.dynamic_k
            else:
                return EloColumn.RECENT, EloColumn.DYNAMIC, self.config.recent_k

        raise ValueError(f"Unknown match context: {match_context!r}")

    @staticmethod
    def _get_entry_role_elo(entry: OpponentEntry, column: EloColumn) -> float:
        """Read the role-specific Elo from an entry."""
        return getattr(entry, column.value)

    @staticmethod
    def determine_match_context(entry_a: OpponentEntry, entry_b: OpponentEntry) -> str:
        """Infer the match context from participant roles."""
        roles = {entry_a.role, entry_b.role}

        # Frontier context takes priority: any match involving a Frontier entry
        # updates elo_frontier for BOTH participants.  This is intentional —
        # Dynamic entries accumulate a meaningful elo_frontier that tracks how
        # well they perform against the Frontier benchmark tier.
        if Role.FRONTIER_STATIC in roles:
            return "frontier"

        if roles == {Role.DYNAMIC}:
            return "dynamic"

        if roles == {Role.RECENT_FIXED}:
            return "recent"

        if Role.DYNAMIC in roles and Role.RECENT_FIXED in roles:
            return "cross_dynamic_recent"

        # NOTE: "historical" context is NOT inferred here — there is no
        # Role.HISTORICAL enum value.  Historical gauntlet matches pass
        # "historical" as an explicit match_context to update_from_result(),
        # bypassing this method entirely.  This is by design: historical
        # library entries reuse existing roles (they are tagged references,
        # not a separate tier).

        # Fallback: treat as dynamic context (e.g. UNASSIGNED vs UNASSIGNED)
        logger.warning(
            "Unrecognised role combination %s vs %s — falling back to 'dynamic' context",
            entry_a.role,
            entry_b.role,
        )
        return "dynamic"
