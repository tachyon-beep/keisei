"""Tier managers: Frontier, RecentFixed, Dynamic."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Callable

from keisei.config import DynamicConfig, FrontierStaticConfig, RecentFixedConfig
from keisei.training.opponent_store import (
    OpponentEntry,
    OpponentStore,
    Role,
)

if TYPE_CHECKING:
    import torch

    from keisei.training.frontier_promoter import FrontierPromoter

logger = logging.getLogger(__name__)


class ReviewOutcome(StrEnum):
    PROMOTE = "promote"
    RETIRE = "retire"
    DELAY = "delay"


# ---------------------------------------------------------------------------
# FrontierManager
# ---------------------------------------------------------------------------


class FrontierManager:
    """Manages the Frontier Static tier -- Elo-spread anchors, reviewed infrequently."""

    def __init__(
        self,
        store: OpponentStore,
        config: FrontierStaticConfig,
        promoter: FrontierPromoter | None = None,
    ) -> None:
        self._store = store
        self._config = config
        self._promoter = promoter

    def get_active(self) -> list[OpponentEntry]:
        """Return all active FRONTIER_STATIC entries."""
        return self._store.list_by_role(Role.FRONTIER_STATIC)

    def select_initial(
        self, entries: list[OpponentEntry], count: int
    ) -> list[OpponentEntry]:
        """Select *count* entries spanning the Elo range via evenly-spaced indices.

        Sorts by Elo ascending, then picks at quintile-spread indices.
        If fewer entries than *count*, returns all of them.
        """
        if count < 1:
            return []
        if len(entries) <= count:
            logger.info(
                "Frontier select_initial: only %d entries available (requested %d)",
                len(entries),
                count,
            )
            return list(entries)

        sorted_entries = sorted(entries, key=lambda e: e.elo_rating)
        n = len(sorted_entries)
        if count == 1:
            # Single slot: pick the median entry
            indices = [n // 2]
        else:
            # Evenly spaced indices across [0, n-1]
            indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
        selected = [sorted_entries[i] for i in indices]
        logger.info(
            "Frontier select_initial: picked %d from %d entries, Elo range %.0f-%.0f",
            len(selected),
            n,
            selected[0].elo_rating,
            selected[-1].elo_rating,
        )
        return selected

    def review(self, epoch: int) -> None:
        """Review frontier entries for promotion from Dynamic tier.

        When no promoter is configured (backward compat), this is a no-op.
        Otherwise, evaluates Dynamic entries via the promoter, and if a
        candidate qualifies, atomically clones it into Frontier and retires
        the weakest/stalest Frontier entry if at capacity.

        Only one retirement per review() call — by design, to prevent
        wholesale turnover of the Frontier tier in a single epoch.  If the
        tier is over capacity, successive review() calls will drain it by
        one per review window.

        Note: methods like list_by_role() acquire OpponentStore._lock inside
        a transaction() that already holds it.  This is safe because the
        store uses a threading.RLock (reentrant) specifically for this pattern.
        """
        if self._promoter is None:
            return  # No-op when promoter not provided (backward compat)

        dynamic_entries = self._store.list_by_role(Role.DYNAMIC)
        frontier_entries = self._store.list_by_role(Role.FRONTIER_STATIC)

        candidate = self._promoter.evaluate(dynamic_entries, frontier_entries, epoch)
        if candidate is None:
            return

        with self._store.transaction():
            # Re-fetch inside transaction for accurate slot count.
            # Retire BEFORE clone so that if retirement fails and the
            # transaction rolls back, no orphaned checkpoint file is left
            # from the clone's file copy (which is non-transactional).
            frontier_entries = self._store.list_by_role(Role.FRONTIER_STATIC)
            retired_id = None
            if len(frontier_entries) >= self._config.slots:
                retired_id = self._retire_weakest_or_stalest(
                    frontier_entries, epoch
                )
            new_entry = self._store.clone_entry(
                candidate.id,
                Role.FRONTIER_STATIC,
                reason=(
                    f"promoted from Dynamic entry {candidate.id} "
                    f"({candidate.display_name}) at epoch {epoch}"
                ),
            )

        logger.info(
            "Frontier promotion: Dynamic entry %d (Elo=%.1f) promoted as entry %d, retired=%s",
            candidate.id,
            candidate.elo_rating,
            new_entry.id,
            retired_id if retired_id is not None else "none",
        )

    def _retire_weakest_or_stalest(
        self, frontier_entries: list[OpponentEntry], epoch: int
    ) -> int | None:
        """Retire the weakest or stalest Frontier entry. Returns retired entry ID."""
        if not frontier_entries:
            return None

        # Filter to entries past min_tenure.  When min_tenure_epochs=0, all
        # entries are eligible (created_epoch <= epoch is always true).
        eligible = [
            e
            for e in frontier_entries
            if e.created_epoch + self._config.min_tenure_epochs <= epoch
        ]

        if not eligible:
            # Fallback: if no entries are past tenure, retire the oldest anyway.
            # This guarantees a retirement always occurs when called at capacity,
            # so the caller (review()) can safely clone before checking capacity
            # without risk of exceeding the slot limit.
            eligible = sorted(frontier_entries, key=lambda e: e.created_epoch)
            target = eligible[0]
        else:
            # Retire lowest elo_frontier; tie-break by oldest created_epoch.
            # Must match the metric used by FrontierPromoter (elo_frontier)
            # so that the weakest entry by promotion criteria is the one retired.
            target = min(eligible, key=lambda e: (e.elo_frontier, e.created_epoch))

        self._store.retire_entry(
            target.id, reason=f"replaced by promotion at epoch {epoch}"
        )
        return target.id

    def is_due_for_review(self, epoch: int) -> bool:
        """Check whether a frontier review should run at this epoch."""
        return epoch > 0 and epoch % self._config.review_interval_epochs == 0


# ---------------------------------------------------------------------------
# RecentFixedManager
# ---------------------------------------------------------------------------


class RecentFixedManager:
    """Manages the Recent Fixed tier -- admits learner snapshots, reviews for promotion."""

    def __init__(self, store: OpponentStore, config: RecentFixedConfig) -> None:
        self._store = store
        self._config = config
        self._weakest_elo_fn: Callable[[], float | None] | None = None

    def set_weakest_elo_fn(self, fn: Callable[[], float | None]) -> None:
        """Set the callback used to query Dynamic tier's weakest eligible Elo."""
        self._weakest_elo_fn = fn

    def admit(
        self,
        model: torch.nn.Module,
        arch: str,
        params: dict[str, Any],
        epoch: int,
    ) -> OpponentEntry:
        """Admit a new learner snapshot into the Recent Fixed tier."""
        entry = self._store.add_entry(
            model, arch, params, epoch=epoch, role=Role.RECENT_FIXED
        )
        logger.info(
            "RecentFixed admit: id=%d epoch=%d (count now %d)",
            entry.id,
            epoch,
            self.count(),
        )
        return entry

    def count(self) -> int:
        """Number of active RECENT_FIXED entries."""
        return len(self._store.list_by_role(Role.RECENT_FIXED))

    def get_unique_opponent_count(self, entry_id: int) -> int:
        """Count distinct opponents this entry has faced (in either seat)."""
        return self._store.count_unique_opponents(entry_id)

    def review_oldest(
        self, total_active_count: int | None = None,
    ) -> tuple[ReviewOutcome, OpponentEntry]:
        """Review the oldest active Recent Fixed entry for promotion/retirement.

        Args:
            total_active_count: Total active entries across all tiers.  When
                provided, ``min_unique_opponents`` is scaled down so that
                promotion remains achievable with a small pool.

        Decision logic:
        1. If games >= min_games AND unique_opponents >= min_unique AND Elo qualifies -> PROMOTE
        2. If under-calibrated AND soft overflow budget remains -> DELAY
        3. Otherwise -> RETIRE
        """
        entries = self._store.list_by_role(Role.RECENT_FIXED)
        if not entries:
            raise ValueError("review_oldest called with no Recent Fixed entries")
        # Oldest first (list_by_role orders by created_epoch ASC)
        oldest = entries[0]

        games_played = oldest.games_played
        unique_opponents = self.get_unique_opponent_count(oldest.id)

        games_ok = games_played >= self._config.min_games_for_review
        # Scale min_unique_opponents so promotion is achievable with a small
        # pool.  An entry can face at most (pool_size - 1) unique opponents.
        effective_min_opponents = self._config.min_unique_opponents
        if total_active_count is not None:
            effective_min_opponents = min(
                effective_min_opponents,
                max(1, total_active_count - 1),
            )
        opponents_ok = unique_opponents >= effective_min_opponents

        # Elo check: pass if Dynamic tier is empty (weakest_elo_fn returns None)
        floor_elo: float | None = None
        if self._weakest_elo_fn is not None:
            floor_elo = self._weakest_elo_fn()
        if floor_elo is None:
            elo_qualified = True
        else:
            elo_qualified = (
                oldest.elo_rating >= floor_elo - self._config.promotion_margin_elo
            )

        if games_ok and opponents_ok and elo_qualified:
            logger.info(
                "RecentFixed review: PROMOTE id=%d (games=%d, opponents=%d, elo=%.1f)",
                oldest.id,
                games_played,
                unique_opponents,
                oldest.elo_rating,
            )
            return ReviewOutcome.PROMOTE, oldest

        # Check soft overflow budget: count - slots tells us overflow used.
        # When negative (under capacity), delay is always appropriate — no
        # pressure to evict when slots are available.
        # Delay if under-calibrated (insufficient games OR insufficient
        # unique opponents) and the soft-overflow budget is not exceeded.
        # Concurrency note: review_oldest is called from the main training
        # thread; the tournament thread reads but doesn't modify Recent Fixed
        # entries, so count() is effectively single-threaded here.
        overflow_used = self.count() - self._config.slots
        can_delay = overflow_used <= self._config.soft_overflow and (
            not games_ok or not opponents_ok
        )

        if can_delay:
            logger.info(
                "RecentFixed review: DELAY id=%d (games=%d/%d, opponents=%d/%d, overflow %d/%d)",
                oldest.id,
                games_played,
                self._config.min_games_for_review,
                unique_opponents,
                self._config.min_unique_opponents,
                overflow_used,
                self._config.soft_overflow,
            )
            return ReviewOutcome.DELAY, oldest

        logger.info(
            "RecentFixed review: RETIRE id=%d (games=%d, opponents=%d, elo=%.1f)",
            oldest.id,
            games_played,
            unique_opponents,
            oldest.elo_rating,
        )
        return ReviewOutcome.RETIRE, oldest


# ---------------------------------------------------------------------------
# DynamicManager
# ---------------------------------------------------------------------------


class DynamicManager:
    """Manages the Dynamic tier -- receives promoted entries, evicts weakest."""

    def __init__(self, store: OpponentStore, config: DynamicConfig) -> None:
        self._store = store
        self._config = config

    def count(self) -> int:
        """Number of active DYNAMIC entries."""
        return len(self._store.list_by_role(Role.DYNAMIC))

    def is_full(self) -> bool:
        """Whether the Dynamic tier has reached its slot limit."""
        return self.count() >= self._config.slots

    def admit(self, source_entry: OpponentEntry) -> OpponentEntry | None:
        """Clone *source_entry* into the Dynamic tier.

        If full, evicts the weakest eligible entry first.
        Returns None if full and no entry can be evicted.

        Note: is_full() and evict_weakest() acquire the store lock inside
        this transaction — safe due to RLock (see FrontierManager.review).
        """
        with self._store.transaction():
            if self.is_full():
                evicted = self.evict_weakest()
                if evicted is None:
                    logger.warning(
                        "DynamicManager.admit: full and all entries protected, "
                        "cannot admit source id=%d",
                        source_entry.id,
                    )
                    return None

            entry = self._store.clone_entry(
                source_entry.id, Role.DYNAMIC, "promoted from recent_fixed"
            )
            # Set protection
            self._store.set_protection(entry.id, self._config.protection_matches)
            logger.info(
                "Dynamic admit: id=%d (from id=%d), protection=%d",
                entry.id,
                source_entry.id,
                self._config.protection_matches,
            )
            # Re-fetch to get updated protection_remaining
            refreshed = self._store.get_entry(entry.id)
            assert refreshed is not None
            return refreshed

    def evict_weakest(self, disabled_entry_ids: set[int] | None = None) -> OpponentEntry | None:
        """Evict the lowest-Elo eligible entry (not protected, enough games).

        Disabled entries (passed via *disabled_entry_ids*) are always eligible
        for eviction regardless of protection or games played.

        Returns the evicted entry, or None if all entries are protected.
        """
        entries = self._store.list_by_role(Role.DYNAMIC)
        disabled = disabled_entry_ids or set()
        eligible = [
            e
            for e in entries
            if (e.protection_remaining == 0
                and e.games_played >= self._config.min_games_before_eviction)
            or e.id in disabled
        ]
        if not eligible:
            logger.info("Dynamic evict_weakest: no eligible entries")
            return None

        weakest = min(eligible, key=lambda e: e.elo_dynamic)
        self._store.retire_entry(weakest.id, "evicted: weakest Elo in dynamic tier")
        logger.info(
            "Dynamic evict_weakest: retired id=%d (elo_dynamic=%.1f)",
            weakest.id,
            weakest.elo_dynamic,
        )
        return weakest

    def get_trainable(self, disabled_entries: set[int] | None = None) -> list[OpponentEntry]:
        """Return Dynamic entries eligible for training updates.

        Note: protected entries ARE included — protection prevents eviction,
        not training.  Newly admitted Dynamic entries should receive updates
        immediately so they begin diverging from their frozen source.
        """
        if not self._config.training_enabled:
            return []
        disabled = disabled_entries or set()
        return [e for e in self._store.list_by_role(Role.DYNAMIC)
                if e.id not in disabled]

    def weakest_elo(self) -> float | None:
        """Return the Elo of the weakest eligible entry, or None if all protected."""
        entries = self._store.list_by_role(Role.DYNAMIC)
        eligible = [
            e
            for e in entries
            if e.protection_remaining == 0
            and e.games_played >= self._config.min_games_before_eviction
        ]
        if not eligible:
            return None
        return min(e.elo_rating for e in eligible)

    def weakest_dynamic_elo(self) -> float | None:
        """Return the elo_dynamic of the weakest eligible entry, or None."""
        entries = self._store.list_by_role(Role.DYNAMIC)
        eligible = [
            e
            for e in entries
            if e.protection_remaining == 0
            and e.games_played >= self._config.min_games_before_eviction
        ]
        if not eligible:
            return None
        return min(e.elo_dynamic for e in eligible)
