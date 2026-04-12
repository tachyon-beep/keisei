"""Training-side tournament dispatcher.

Generates round-robin pairings from the league pool and enqueues them
into the ``tournament_pairing_queue`` table for worker processes to
claim.  Lives inside the training process; called once per epoch
boundary with an adaptive queue-depth gate (only enqueues when the
queue has room).  Owns ``MatchScheduler`` and ``PriorityScorer`` —
workers never touch these.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from keisei.db import _connect
from keisei.training.tournament_queue import enqueue_pairings, get_round_status

if TYPE_CHECKING:
    from keisei.training.match_scheduler import MatchScheduler
    from keisei.training.opponent_store import OpponentStore
    from keisei.training.priority_scorer import PriorityScorer

logger = logging.getLogger(__name__)


class TournamentDispatcher:
    """Generates pairings and enqueues them for sidecar workers."""

    def __init__(
        self,
        *,
        store: OpponentStore,
        scheduler: MatchScheduler,
        games_per_match: int = 3,
        min_pool_size: int = 3,
        priority_scorer: PriorityScorer | None = None,
    ) -> None:
        self.store = store
        self.scheduler = scheduler
        self.games_per_match = games_per_match
        self.min_pool_size = min_pool_size
        self.priority_scorer = priority_scorer
        self._round_counter = 0

    def _next_round_id(self) -> int:
        """Monotonically increasing round ID derived from the DB."""
        conn = _connect(self.store.db_path)
        try:
            row = conn.execute(
                "SELECT COALESCE(MAX(round_id), 0) + 1 AS next_id "
                "FROM tournament_pairing_queue"
            ).fetchone()
            return int(row["next_id"])
        finally:
            conn.close()

    def enqueue_round(self, *, epoch: int) -> int | None:
        """Generate pairings for a full round and enqueue them.

        Returns the round_id on success, or None if the pool is too
        small to generate any pairings.
        """
        entries = self.store.list_entries()
        if len(entries) < self.min_pool_size:
            logger.debug(
                "Pool has %d entries (< %d min) — skipping round",
                len(entries), self.min_pool_size,
            )
            return None

        pairings = self.scheduler.generate_round(entries)
        if not pairings:
            return None

        round_id = self._next_round_id()

        priorities: list[float] | None = None
        if self.priority_scorer is not None:
            priorities = [
                self.priority_scorer.score(a, b)
                for a, b in pairings
            ]

        pairing_tuples = [
            (a.id, b.id, self.games_per_match) for a, b in pairings
        ]

        enqueue_pairings(
            self.store.db_path,
            round_id=round_id,
            epoch=epoch,
            pairings=pairing_tuples,
            priorities=priorities,
        )

        logger.info(
            "Dispatched round %d: %d pairings from %d entries at epoch %d",
            round_id, len(pairings), len(entries), epoch,
        )
        return round_id

    def check_round_completion(self, round_id: int) -> bool:
        """Check if all pairings in a round are finished.

        When complete, advances the PriorityScorer round if present.
        Returns True if the round is done.
        """
        status = get_round_status(self.store.db_path, round_id)
        pending = status.get("pending", 0)
        playing = status.get("playing", 0)
        if pending > 0 or playing > 0:
            return False
        if self.priority_scorer is not None:
            self.priority_scorer.advance_round()
        return True
