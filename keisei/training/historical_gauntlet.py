"""HistoricalGauntlet: periodic benchmark runner for regression detection."""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch

from keisei.config import GauntletConfig
from keisei.training.historical_library import HistoricalSlot
from keisei.training.match_utils import play_match, release_models
from keisei.training.opponent_store import OpponentEntry, OpponentStore
from keisei.training.role_elo import RoleEloTracker

logger = logging.getLogger(__name__)


class HistoricalGauntlet:
    """Periodic benchmark runner: learner vs historical library entries.

    Runs synchronously on the tournament thread after each round-robin
    round when is_due(epoch) returns True.
    """

    def __init__(
        self,
        store: OpponentStore,
        role_elo_tracker: RoleEloTracker,
        config: GauntletConfig,
        *,
        stop_event: threading.Event | None = None,
        device: str = "cpu",
        num_envs: int = 64,
        max_ply: int = 512,
    ) -> None:
        self.store = store
        self.role_elo_tracker = role_elo_tracker
        self.config = config
        # Fallback Event is never-set by default — gauntlet runs to completion
        # unless the caller shares a real stop Event via constructor or set_stop_event().
        self._stop_event = stop_event or threading.Event()
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.max_ply = max_ply

    def set_stop_event(self, event: threading.Event) -> None:
        """Set the stop event (called by tournament to share its event)."""
        self._stop_event = event

    def is_due(self, epoch: int) -> bool:
        """True if epoch aligns with gauntlet interval.

        interval_epochs is guaranteed >= 1 by GauntletConfig.__post_init__.
        """
        if not self.config.enabled:
            return False
        if epoch < 1:
            return False
        return epoch % self.config.interval_epochs == 0

    def run_gauntlet(
        self,
        epoch: int,
        learner_entry: OpponentEntry,
        historical_slots: list[HistoricalSlot],
        vecenv: Any | None = None,
    ) -> None:
        """Play learner vs each non-empty historical slot, record results.

        Args:
            epoch: Current training epoch.
            learner_entry: The learner's current league entry.
            historical_slots: List of HistoricalSlot from the library.
            vecenv: Optional pre-created VecEnv (reused from tournament).
        """
        filled_slots = [s for s in historical_slots if s.entry_id is not None]
        if not filled_slots:
            logger.warning(
                "Gauntlet skipped: all historical slots empty at epoch %d", epoch
            )
            return

        logger.info(
            "Gauntlet started: epoch=%d, filled_slots=%d/%d",
            epoch, len(filled_slots), len(historical_slots),
        )

        # Lazy-import and create VecEnv if not provided.
        # VecEnv is a Rust/PyO3 struct — no explicit close() needed; Rust's
        # Drop trait releases OS resources when the Python reference is collected.
        if vecenv is None:
            from shogi_gym import VecEnv
            vecenv = VecEnv(
                num_envs=self.num_envs,
                max_ply=self.max_ply,
                observation_mode="katago",
                action_mode="spatial",
            )

        try:
            learner_model = self.store.load_opponent(learner_entry, device=str(self.device))
        except Exception:
            logger.exception(
                "Gauntlet aborted: failed to load learner model (entry %d)", learner_entry.id
            )
            return

        slots_played = 0
        total_games = 0

        try:
            for slot in filled_slots:
                if self._stop_event.is_set():
                    break

                assert slot.entry_id is not None
                hist_entry = self.store.get_entry(slot.entry_id)
                if hist_entry is None:
                    logger.warning(
                        "Gauntlet slot %d: entry %d not found, skipping",
                        slot.slot_index, slot.entry_id,
                    )
                    continue

                try:
                    hist_model = self.store.load_opponent(hist_entry, device=str(self.device))
                except Exception:
                    logger.warning(
                        "Gauntlet slot %d: failed to load model for entry %d (epoch %d), skipping",
                        slot.slot_index, slot.entry_id, slot.actual_epoch or 0,
                        exc_info=True,
                    )
                    continue

                try:
                    wins, losses, draws = play_match(
                        vecenv, learner_model, hist_model,
                        device=self.device, num_envs=self.num_envs,
                        max_ply=self.max_ply, games_target=self.config.games_per_matchup,
                        stop_event=self._stop_event,
                    )
                except Exception:
                    logger.exception(
                        "Gauntlet slot %d: match failed against entry %d",
                        slot.slot_index, slot.entry_id,
                    )
                    continue
                finally:
                    release_models(hist_model, device_type=self.device.type)

                game_count = wins + losses + draws
                if game_count == 0:
                    continue

                # Re-read learner for accurate elo_before (and for the Elo computation,
                # which reads Elo from the entry object's attributes).
                current_learner = self.store.get_entry(learner_entry.id)
                if current_learner is None:
                    logger.warning(
                        "Gauntlet slot %d: learner entry %d vanished mid-gauntlet, skipping",
                        slot.slot_index, learner_entry.id,
                    )
                    continue
                elo_before = current_learner.elo_historical

                # Update role Elo via RoleEloTracker (atomic two-entry update)
                result_score = (wins + 0.5 * draws) / game_count
                self.role_elo_tracker.update_from_result(
                    current_learner, hist_entry, result_score, "historical",
                )

                # Re-read to get elo_after (if entry vanished between update and
                # re-read, use elo_before as conservative fallback — the DB write
                # already happened, we just can't read the result)
                updated_learner = self.store.get_entry(learner_entry.id)
                elo_after = updated_learner.elo_historical if updated_learner else elo_before

                # Record gauntlet result
                self.store.record_gauntlet_result(
                    epoch=epoch,
                    entry_id=learner_entry.id,
                    historical_slot=slot.slot_index,
                    historical_entry_id=slot.entry_id,
                    wins=wins,
                    losses=losses,
                    draws=draws,
                    elo_before=elo_before,
                    elo_after=elo_after,
                )

                slots_played += 1
                total_games += game_count

                logger.info(
                    "Gauntlet slot %d: wins=%d losses=%d draws=%d (vs epoch %d)",
                    slot.slot_index, wins, losses, draws, slot.actual_epoch or 0,
                )

        finally:
            release_models(learner_model, device_type=self.device.type)

        logger.info(
            "Gauntlet complete: epoch=%d, slots_played=%d, total_games=%d",
            epoch, slots_played, total_games,
        )
