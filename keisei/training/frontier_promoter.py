"""FrontierPromoter — evaluates Dynamic entries for Frontier Static promotion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from keisei.config import FrontierStaticConfig
    from keisei.training.opponent_store import OpponentEntry

logger = logging.getLogger(__name__)


class FrontierPromoter:
    """Evaluates Dynamic entries for promotion to Frontier Static tier.

    Promotion criteria (ALL must be met):
    1. games_played >= min_games_for_promotion
    2. Entry is in top-K by Elo among Dynamic entries
    3. Entry has held top-K position for >= streak_epochs consecutive epochs
    4. Entry Elo >= weakest Frontier Elo + promotion_margin_elo
    5. Lineage overlap with existing Frontier entries < max_lineage_overlap

    Special case: if Frontier tier is empty, only criterion 1 (min_games) is
    required — criteria 2-5 are bypassed to seed the tier with a calibrated entry.

    Streak tracking is IN-MEMORY ONLY — intentionally lost on restart
    (conservative: delays promotion slightly after restart).
    """

    def __init__(self, config: FrontierStaticConfig) -> None:
        self.config = config
        # Maps entry_id -> epoch when entry first entered top-K.  Streak length
        # is computed as (current_epoch - first_seen_epoch), NOT stored directly.
        self._topk_streaks: dict[int, int] = {}

    def evaluate(
        self,
        dynamic_entries: list[OpponentEntry],
        frontier_entries: list[OpponentEntry],
        epoch: int,
    ) -> OpponentEntry | None:
        """Find the best promotion candidate, or None."""
        # Sort by elo_frontier descending.  Dynamic entries accumulate elo_frontier
        # from matches against Frontier opponents (via RoleEloTracker "frontier"
        # context), so this column reflects how well each Dynamic entry performs
        # against the benchmark tier — the metric most relevant for promotion.
        sorted_dynamics = sorted(
            dynamic_entries, key=lambda e: e.elo_frontier, reverse=True
        )

        # Identify top-K
        topk_entries = sorted_dynamics[: self.config.topk]
        topk_ids = {e.id for e in topk_entries}

        # Update streak tracking
        # Add new top-K entries
        for entry in topk_entries:
            if entry.id not in self._topk_streaks:
                self._topk_streaks[entry.id] = epoch

        # Remove entries that dropped out of top-K
        dropped = [eid for eid in self._topk_streaks if eid not in topk_ids]
        for eid in dropped:
            del self._topk_streaks[eid]

        # Check each top-K entry (highest Elo first)
        for entry in topk_entries:
            if self.should_promote(entry, frontier_entries, epoch):
                return entry

        return None

    def should_promote(
        self,
        candidate: OpponentEntry,
        frontier_entries: list[OpponentEntry],
        epoch: int,
    ) -> bool:
        """Check if a candidate meets all promotion criteria."""
        # 1. Games played threshold (always required, even when seeding)
        if candidate.games_played < self.config.min_games_for_promotion:
            return False

        # Special case: empty Frontier — promote to seed once calibrated.
        # Criteria 2-5 (top-K, streak, elo margin, lineage) are bypassed because
        # there is no benchmark tier to measure consistency against.  The
        # min_games check above ensures the entry has enough calibration matches.
        if not frontier_entries:
            return True

        # 2. Must be in top-K (caller ensures this, but verify)
        if candidate.id not in self._topk_streaks:
            return False

        # 3. Streak duration (idempotent: repeated calls at the same epoch
        #    produce the same result — no extra streak progress is granted)
        first_seen = self._topk_streaks[candidate.id]
        if epoch - first_seen < self.config.streak_epochs:
            return False

        # 4. Elo margin above weakest Frontier
        weakest_frontier_elo = min(e.elo_frontier for e in frontier_entries)
        if candidate.elo_frontier < weakest_frontier_elo + self.config.promotion_margin_elo:
            return False

        # 5. Lineage overlap — same_lineage_count is the number of EXISTING
        #    Frontier entries sharing the candidate's lineage (excluding the
        #    candidate itself).  Blocking when count >= max_lineage_overlap means
        #    the total after promotion would exceed the limit.  E.g. with
        #    max_lineage_overlap=2: 1 existing same-lineage entry passes (total
        #    becomes 2), 2 existing same-lineage entries blocks (total would be 3).
        same_lineage_count = sum(
            1
            for e in frontier_entries
            if e.lineage_group
            and candidate.lineage_group
            and e.lineage_group == candidate.lineage_group
        )
        if same_lineage_count >= self.config.max_lineage_overlap:
            return False

        return True
