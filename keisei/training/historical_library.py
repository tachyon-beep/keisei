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

        # Compute neighbor distances for the 50% proximity threshold
        neighbor_dists = self._neighbor_distances(targets)
        used_ids: set[int] = set()

        for i, target in enumerate(targets):
            best = self._snap_to_nearest(target, candidates, used_ids)
            if best is None:
                self.store.upsert_historical_slot(
                    slot_index=i,
                    target_epoch=target,
                    entry_id=None,
                    actual_epoch=None,
                    selection_mode="fallback",
                )
                continue

            distance = abs(best.created_epoch - target)
            threshold = neighbor_dists[i] * 0.5
            if threshold > 0 and distance > threshold:
                # Too far from target — leave slot empty
                self.store.upsert_historical_slot(
                    slot_index=i,
                    target_epoch=target,
                    entry_id=None,
                    actual_epoch=None,
                    selection_mode="log_spaced",
                )
            else:
                used_ids.add(best.id)
                self.store.upsert_historical_slot(
                    slot_index=i,
                    target_epoch=target,
                    entry_id=best.id,
                    actual_epoch=best.created_epoch,
                    selection_mode="log_spaced" if len(candidates) >= self.config.slots else "fallback",
                )

        filled = sum(
            1 for s in self.get_slots() if s.entry_id is not None
        )
        logger.info(
            "Historical library refresh: epoch=%d, filled=%d/%d",
            current_epoch, filled, self.config.slots,
        )

    def get_slots(self) -> list[HistoricalSlot]:
        """Returns list of HistoricalSlot dataclasses for all slots."""
        raw_slots = self.store.get_historical_slots()
        result: list[HistoricalSlot] = []
        for row in raw_slots:
            result.append(HistoricalSlot(
                slot_index=row["slot_index"],
                target_epoch=row["target_epoch"],
                entry_id=row["entry_id"],
                actual_epoch=row["actual_epoch"],
                selection_mode=row["selection_mode"],
                display_name=row.get("display_name"),
                checkpoint_path=row.get("checkpoint_path"),
            ))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_targets(current_epoch: int, num_slots: int = 5) -> list[int]:
        """Compute log-spaced target epochs."""
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

        # Sort: prefer retired/archived (stable) over active
        def stability_key(e: object) -> int:
            status = getattr(e, "status", None)
            if status in (EntryStatus.RETIRED, EntryStatus.ARCHIVED):
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
        """Compute distance to nearest neighbor for each target (for 50% threshold)."""
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
