"""TieredPool -- orchestrator for the tiered opponent league."""

from __future__ import annotations

import logging

from keisei.config import LeagueConfig
from keisei.training.dynamic_trainer import DynamicTrainer
from keisei.training.frontier_promoter import FrontierPromoter
from keisei.training.historical_gauntlet import HistoricalGauntlet
from keisei.training.historical_library import HistoricalLibrary, HistoricalSlot
from keisei.training.match_scheduler import MatchScheduler
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role
from keisei.training.role_elo import RoleEloTracker
from keisei.training.tier_managers import (
    DynamicManager,
    FrontierManager,
    RecentFixedManager,
    ReviewOutcome,
)

logger = logging.getLogger(__name__)


class TieredPool:
    """High-level orchestrator tying together the three tier managers."""

    def __init__(
        self,
        store: OpponentStore,
        config: LeagueConfig,
        learner_lr: float = 0.0,
    ) -> None:
        self.store = store
        self.config = config

        promoter = FrontierPromoter(config.frontier)
        self.frontier_manager = FrontierManager(store, config.frontier, promoter=promoter)
        self.recent_manager = RecentFixedManager(store, config.recent)
        self.dynamic_manager = DynamicManager(store, config.dynamic)
        self.recent_manager.set_weakest_elo_fn(self.dynamic_manager.weakest_elo)
        self.historical_library = HistoricalLibrary(store, config.history)
        self.historical_library.refresh(0)
        self.role_elo_tracker = RoleEloTracker(store, config.elo)
        self.scheduler = MatchScheduler(config.scheduler)
        self.gauntlet: HistoricalGauntlet | None = None
        if config.gauntlet.enabled:
            self.gauntlet = HistoricalGauntlet(
                store=store,
                role_elo_tracker=self.role_elo_tracker,
                config=config.gauntlet,
            )

        if config.dynamic.training_enabled:
            self.dynamic_trainer: DynamicTrainer | None = DynamicTrainer(
                store=self.store,
                config=config.dynamic,
                learner_lr=learner_lr,
            )
            logger.warning(
                "Dynamic training enabled — ensure tournament_device differs from "
                "learner device to avoid GPU memory contention"
            )
        else:
            self.dynamic_trainer = None

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def _total_capacity(self) -> int:
        """Total slots across all tiers."""
        return (
            self.config.frontier.slots
            + self.config.recent.slots
            + self.config.dynamic.slots
        )

    def _total_active(self) -> int:
        """Count of all active entries across all tiers."""
        return len(self.store.list_entries())

    def has_spare_capacity(self) -> bool:
        """True when total active entries are below total tier capacity."""
        return self._total_active() < self._total_capacity()

    def _frontier_promotion_candidate_ids(self) -> frozenset[int]:
        """Return IDs of Dynamic entries currently in top-K for Frontier promotion.

        These entries are protected from eviction per §7.2.
        """
        promoter = self.frontier_manager._promoter
        if promoter is None:
            return frozenset()
        return frozenset(promoter._topk_streaks.keys())

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot_learner(
        self, model: object, arch: str, params: dict[str, object], epoch: int
    ) -> OpponentEntry:
        """Take a learner snapshot and admit it to the Recent Fixed tier.

        If the tier overflows, the oldest entry is reviewed for promotion
        to Dynamic or retirement.  Retirement is suppressed while the total
        pool has spare capacity — entries stay in overflow until slots fill.
        """
        entry = self.recent_manager.admit(model, arch, params, epoch)

        # May fire repeatedly for the same entry when Dynamic is full and all
        # protected (PROMOTE succeeds but admit returns None → entry stays in
        # overflow).  Self-correcting: protection eventually expires or overflow
        # budget is exceeded, triggering RETIRE.
        if self.recent_manager.count() > self.config.recent.slots:
            total_active = self._total_active()
            outcome, oldest = self.recent_manager.review_oldest(
                total_active_count=total_active,
            )
            if outcome is ReviewOutcome.PROMOTE:
                # Outer transaction wraps the full promote-and-retire sequence.
                # admit() opens a nested transaction internally, but
                # OpponentStore.transaction() supports nesting — only the
                # outermost level commits, so the whole operation is atomic.
                with self.store.transaction():
                    # §7.2: exclude current Frontier promotion candidates from eviction
                    promo_ids = self._frontier_promotion_candidate_ids()
                    clone = self.dynamic_manager.admit(oldest, promotion_candidate_ids=promo_ids)
                    if clone is not None:
                        self.store.retire_entry(oldest.id, "promoted to dynamic")
                    # else: Dynamic full and all protected — keep in Recent Fixed overflow
            elif outcome is ReviewOutcome.RETIRE:
                if total_active < self._total_capacity():
                    logger.info(
                        "Skipping retirement of id=%d: pool has spare capacity "
                        "(%d/%d slots used)",
                        oldest.id, total_active, self._total_capacity(),
                    )
                else:
                    self.store.retire_entry(oldest.id, "did not qualify for dynamic")
            # DELAY: do nothing, let it sit in overflow

        return entry

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def entries_by_role(self) -> dict[Role, list[OpponentEntry]]:
        """Return active entries grouped by role."""
        return {
            Role.FRONTIER_STATIC: self.store.list_by_role(Role.FRONTIER_STATIC),
            Role.RECENT_FIXED: self.store.list_by_role(Role.RECENT_FIXED),
            Role.DYNAMIC: self.store.list_by_role(Role.DYNAMIC),
        }

    def list_all_active(self) -> list[OpponentEntry]:
        """Return all active entries across all tiers."""
        return self.store.list_entries()

    def sample_opponent_for_learner(self) -> OpponentEntry:
        """Sample an opponent for the learner using §11 tier ratios.

        Uses the MatchScheduler's configured learner_dynamic/frontier/recent
        ratios (default 50/30/20) to pick from the appropriate tier.
        """
        return self.scheduler.sample_for_learner(self.entries_by_role())

    # ------------------------------------------------------------------
    # Epoch hooks
    # ------------------------------------------------------------------

    def on_epoch_end(self, epoch: int) -> None:
        """Run end-of-epoch maintenance (frontier review, historical refresh, etc.)."""
        if self.frontier_manager.is_due_for_review(epoch):
            self.frontier_manager.review(epoch)
        if self.historical_library.is_due_for_refresh(epoch):
            self.historical_library.refresh(epoch)

    def is_gauntlet_due(self, epoch: int) -> bool:
        """Check whether a historical gauntlet should run at this epoch."""
        if self.gauntlet is None:
            return False
        return self.gauntlet.is_due(epoch)

    def get_historical_slots(self) -> list[HistoricalSlot]:
        """Delegate to the historical library."""
        return self.historical_library.get_slots()

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap_from_flat_pool(self) -> None:
        """One-time migration: assign roles to UNASSIGNED entries.

        Allocation order:
        1. Recent Fixed -- most recent N entries by epoch
        2. Frontier Static -- Elo-spread quintile selection from remaining
        3. Dynamic -- everything left, sorted by Elo descending
        """
        with self.store.transaction():
            if self.store.is_bootstrapped():
                logger.info("Bootstrap already complete, skipping")
                return

            entries = [
                e for e in self.store.list_entries() if e.role is Role.UNASSIGNED
            ]
            if not entries:
                self.store.set_bootstrapped()
                return

            n = len(entries)

            # --- Recent Fixed: most recent N by epoch ---
            n_recent = max(1, round(n * 0.25)) if n >= 3 else (1 if n >= 2 else 0)
            entries_by_epoch = sorted(
                entries, key=lambda e: e.created_epoch, reverse=True
            )
            recent_selected = entries_by_epoch[:n_recent]
            recent_ids = {e.id for e in recent_selected}
            for e in recent_selected:
                self.store.update_role(
                    e.id, Role.RECENT_FIXED, "bootstrap: recent fixed"
                )

            # --- Frontier Static: quintile Elo spread from remaining ---
            # n_frontier is computed from total n (not len(remaining_after_recent))
            # so each tier gets a fixed ~25% share of the original pool.  If
            # n_frontier > len(remaining_after_recent), select_initial() safely
            # returns all available entries.  n_dynamic below uses actual selected
            # counts, so no entries are double-assigned or lost.
            remaining_after_recent = [e for e in entries if e.id not in recent_ids]
            n_frontier = (
                max(1, round(n * 0.25)) if n >= 3 else (1 if n >= 1 else 0)
            )
            frontier_selected = self.frontier_manager.select_initial(
                remaining_after_recent, count=n_frontier
            )
            frontier_ids = {e.id for e in frontier_selected}
            for e in frontier_selected:
                self.store.update_role(
                    e.id, Role.FRONTIER_STATIC, "bootstrap: frontier static"
                )

            # --- Dynamic: rest, sorted by Elo descending ---
            assigned_ids = recent_ids | frontier_ids
            n_dynamic = n - len(recent_selected) - len(frontier_selected)
            dynamic_candidates = sorted(
                [e for e in entries if e.id not in assigned_ids],
                key=lambda e: e.elo_rating,
                reverse=True,
            )
            for e in dynamic_candidates[:n_dynamic]:
                self.store.update_role(e.id, Role.DYNAMIC, "bootstrap: dynamic")

            # Retire any remainder (shouldn't happen, but defensive)
            dynamic_ids = {e.id for e in dynamic_candidates[:n_dynamic]}
            for e in entries:
                if e.id not in assigned_ids and e.id not in dynamic_ids:
                    self.store.retire_entry(
                        e.id, "bootstrap: excess entry retired"
                    )

            self.store.set_bootstrapped()

        logger.info(
            "Bootstrap complete: %d recent, %d frontier, %d dynamic",
            len(recent_selected),
            len(frontier_selected),
            min(n_dynamic, len(dynamic_candidates)),
        )
