"""HistoricalLibrary: 5-slot milestone manager for long-range regression detection."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from keisei.config import HistoricalLibraryConfig
from keisei.training.opponent_store import EntryStatus, OpponentStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalSlot:
    """A single slot in the historical library."""

    slot_index: int
    target_epoch: int
    entry_id: int | None
    actual_epoch: int | None
    selection_mode: str  # 'log_spaced' or 'fallback'
    selected_at: str | None = None
    display_name: str | None = None
    checkpoint_path: str | None = None


class HistoricalLibrary:
    """Manages 5 log-spaced milestone slots for historical regression detection.

    Does not own models or run matches — only selects which archived
    checkpoints fill the slots.
    """

    def __init__(self, store: OpponentStore, config: HistoricalLibraryConfig) -> None:
        self.store = store
        self.config = config

    def is_due_for_refresh(self, epoch: int) -> bool:
        """True if epoch aligns with the refresh interval."""
        if epoch < self.config.min_epoch_for_selection:
            return False
        return epoch % self.config.refresh_interval_epochs == 0

    def refresh(self, current_epoch: int) -> None:
        """Recompute log-spaced targets and snap to nearest archived checkpoints."""
        targets = self._compute_targets(current_epoch, num_slots=self.config.slots)
        candidates = self._get_candidates()

        if not candidates:
            for i in range(self.config.slots):
                self.store.upsert_historical_slot(
                    slot_index=i,
                    target_epoch=targets[i],
                    entry_id=None,
                    actual_epoch=None,
                    selection_mode="fallback",
                )
            logger.warning(
                "Historical library refresh: no candidates at epoch %d", current_epoch
            )
            return

        # Two-pass assignment (§6.4):
        # Pass 1: fill slots where a candidate is within the 50% proximity
        #         threshold — these are good log-spaced matches.
        # Pass 2: backfill any empty slots with the closest remaining
        #         candidate, so all slots are always populated.
        neighbor_dists = self._neighbor_distances(targets)
        used_ids: set[int] = set()
        enough_candidates = len(candidates) >= self.config.slots
        # slot_assignments[i] = (entry, mode) or None
        slot_assignments: list[tuple[object, str] | None] = [None] * len(targets)

        # Pass 1: assign within-threshold candidates
        for i, target in enumerate(targets):
            best = self._snap_to_nearest(target, candidates, used_ids)
            if best is None:
                continue
            distance = abs(best.created_epoch - target)
            threshold = neighbor_dists[i] * 0.5
            if threshold == 0 or distance > threshold:
                continue  # zero spacing or beyond threshold — leave for pass 2
            used_ids.add(best.id)
            mode = "log_spaced" if enough_candidates else "fallback"
            slot_assignments[i] = (best, mode)

        # Pass 2: backfill empty slots with closest unused candidate
        for i, target in enumerate(targets):
            if slot_assignments[i] is not None:
                continue
            best = self._snap_to_nearest(target, candidates, used_ids)
            if best is None:
                continue
            used_ids.add(best.id)
            slot_assignments[i] = (best, "fallback")

        # Snapshot current slots so we can detect re-pointing (§13.4).
        old_slots = {
            s["slot_index"]: s["entry_id"]
            for s in self.store.get_historical_slots()
        }

        # Write all slots to DB
        for i, target in enumerate(targets):
            assignment = slot_assignments[i]
            if assignment is None:
                new_entry_id = None
                self.store.upsert_historical_slot(
                    slot_index=i,
                    target_epoch=target,
                    entry_id=None,
                    actual_epoch=None,
                    selection_mode="fallback",
                )
            else:
                entry, mode = assignment
                new_entry_id = entry.id
                self.store.upsert_historical_slot(
                    slot_index=i,
                    target_epoch=target,
                    entry_id=entry.id,
                    actual_epoch=entry.created_epoch,
                    selection_mode=mode,
                )
            # Log transition when slot's entry_id changed (§13.4).
            old_entry_id = old_slots.get(i)
            if new_entry_id != old_entry_id and (
                new_entry_id is not None or old_entry_id is not None
            ):
                log_id = new_entry_id if new_entry_id is not None else old_entry_id
                self.store.log_transition(
                    entry_id=log_id,
                    from_role=None,
                    to_role=None,
                    from_status=None,
                    to_status=None,
                    reason=f"historical_slot_repointed slot={i} old={old_entry_id} new={new_entry_id}",
                )

        filled = sum(1 for a in slot_assignments if a is not None)
        logger.info(
            "Historical library refresh: epoch=%d, filled=%d/%d",
            current_epoch, filled, self.config.slots,
        )

    def get_slots(self) -> list[HistoricalSlot]:
        """Returns list of HistoricalSlot dataclasses for configured slots only.

        Filters out stale DB rows from prior configs with more slots.
        """
        raw_slots = self.store.get_historical_slots()
        result: list[HistoricalSlot] = []
        for row in raw_slots:
            if row["slot_index"] >= self.config.slots:
                continue
            result.append(HistoricalSlot(
                slot_index=row["slot_index"],
                target_epoch=row["target_epoch"],
                entry_id=row["entry_id"],
                actual_epoch=row["actual_epoch"],
                selection_mode=row["selection_mode"],
                selected_at=row.get("selected_at"),
                display_name=row.get("display_name"),
                checkpoint_path=row.get("checkpoint_path"),
            ))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_targets(current_epoch: int, num_slots: int = 5) -> list[int]:
        """Compute log-spaced target epochs from 1 to current_epoch.

        The first target is always epoch 1 (the earliest training point) and
        the last target is current_epoch, with intermediate targets log-spaced
        between them.  This is intentional: the library wants coverage from the
        very beginning of training to now.
        """
        if num_slots == 1:
            # Single slot: just return the current epoch (clamped to 1 so we
            # always target an actual training checkpoint, not epoch 0).
            return [max(current_epoch, 1)]
        e = max(current_epoch, 2)
        return [
            round(math.exp(math.log(e) * i / (num_slots - 1)))
            for i in range(num_slots)
        ]

    def _get_candidates(self) -> list:
        """Get all entries that could serve as historical milestones.

        Prefers retired/archived entries (stable), but includes active
        entries if not enough stable ones exist.
        """
        entries = self.store.list_all_entries()

        # Sort: prefer retired/archived (stable) over active.
        # All entries are OpponentEntry instances from list_all_entries() and
        # always have a .status attribute — no getattr guard needed.
        def stability_key(e: object) -> int:
            if e.status in (EntryStatus.RETIRED, EntryStatus.ARCHIVED):
                return 0
            return 1

        entries.sort(key=lambda e: (stability_key(e), e.created_epoch))
        return entries

    @staticmethod
    def _snap_to_nearest(
        target: int, candidates: list, used_ids: set[int]
    ) -> object | None:
        """Find the candidate closest to target that hasn't been used yet."""
        best = None
        best_dist = float("inf")
        for c in candidates:
            if c.id in used_ids:
                continue
            dist = abs(c.created_epoch - target)
            # Prefer stable entries on ties
            stability = 0 if c.status in (EntryStatus.RETIRED, EntryStatus.ARCHIVED) else 1
            if dist < best_dist or (dist == best_dist and stability == 0):
                best = c
                best_dist = dist
        return best

    @staticmethod
    def _neighbor_distances(targets: list[int]) -> list[float]:
        """Compute distance to nearest neighbor for each target (for 50% threshold).

        Precondition: targets must be sorted in ascending order (as produced
        by _compute_targets).
        """
        assert all(targets[i] <= targets[i + 1] for i in range(len(targets) - 1)), (
            f"targets must be sorted ascending, got {targets}"
        )
        n = len(targets)
        dists: list[float] = []
        for i in range(n):
            if n == 1:
                dists.append(float("inf"))
            elif i == 0:
                dists.append(float(targets[1] - targets[0]))
            elif i == n - 1:
                dists.append(float(targets[-1] - targets[-2]))
            else:
                dists.append(float(min(targets[i] - targets[i - 1], targets[i + 1] - targets[i])))
        return dists
