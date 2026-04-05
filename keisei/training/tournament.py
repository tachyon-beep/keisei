"""Background round-robin tournament for Elo calibration.

Runs inference-only matches between pairs of league entries to produce
accurate Elo ratings across the full pool — not just learner-vs-opponent.
Models are loaded frozen (no gradients, no training). Results are written
to the same league_results / elo_history tables used by the main loop.

Usage:
    tournament = LeagueTournament(store, scheduler, device="cuda:1")
    tournament.start()   # background thread
    ...
    tournament.stop()    # graceful shutdown
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import torch

from keisei.training.concurrent_matches import ConcurrentMatchPool
from keisei.training.dynamic_trainer import DynamicTrainer

if TYPE_CHECKING:
    from keisei.training.dynamic_trainer import MatchRollout
from keisei.training.historical_gauntlet import HistoricalGauntlet
from keisei.training.historical_library import HistoricalLibrary
from keisei.training.match_scheduler import MatchScheduler, is_training_match
from keisei.training.match_utils import play_match, release_models
from keisei.training.opponent_store import EntryStatus, OpponentEntry, OpponentStore, Role, compute_elo_update
from keisei.training.role_elo import RoleEloTracker

logger = logging.getLogger(__name__)


def majority_wins_result(a_wins: int, b_wins: int, draws: int) -> float:
    """Compute match result for Elo using majority-wins scoring.

    Works for any game count. Returns 1.0 if A won more games,
    0.0 if B won more, 0.5 if tied (including all-draws).
    """
    if a_wins > b_wins:
        return 1.0
    elif b_wins > a_wins:
        return 0.0
    return 0.5


class LeagueTournament:
    """Background thread that runs round-robin matches between pool entries."""

    def __init__(
        self,
        store: OpponentStore,
        scheduler: MatchScheduler,
        *,
        device: str = "cuda:1",
        num_envs: int = 64,
        max_ply: int = 512,
        games_per_match: int = 3,
        k_factor: float = 16.0,
        pause_seconds: float = 5.0,
        min_pool_size: int = 3,
        learner_entry_id: int | None = None,
        historical_library: HistoricalLibrary | None = None,
        gauntlet: HistoricalGauntlet | None = None,
        dynamic_trainer: DynamicTrainer | None = None,
        concurrent_pool: ConcurrentMatchPool | None = None,
        role_elo_tracker: RoleEloTracker | None = None,
    ) -> None:
        """
        Args:
            store: OpponentStore managing pool entries and DB access.
            scheduler: MatchScheduler for generating pairings.
            device: Torch device for inference (should be separate from trainer).
            num_envs: Number of concurrent environments per match.
            max_ply: Maximum ply per game before truncation.
            games_per_match: Total games to play per matchup (split across envs).
            k_factor: Elo K-factor. Lower than main loop (16 vs 32) since
                      these are calibration matches, not primary evaluations.
            pause_seconds: Sleep between matches to avoid starving the main loop.
            min_pool_size: Don't run matches until pool has at least this many entries.
            learner_entry_id: Entry ID of the learner (needed for gauntlet).
            historical_library: The milestone library instance (Phase 2).
            gauntlet: The gauntlet runner instance (Phase 2).
            dynamic_trainer: DynamicTrainer for updating Dynamic entries (Phase 3).
            role_elo_tracker: RoleEloTracker for per-context Elo updates (Phase 4).
        """
        self.store = store
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.max_ply = max_ply
        self.games_per_match = games_per_match
        self.k_factor = k_factor
        self.pause_seconds = pause_seconds
        self.min_pool_size = min_pool_size
        self._learner_entry_id = learner_entry_id
        self._learner_lock = threading.Lock()
        self.historical_library = historical_library
        self.gauntlet = gauntlet
        self.dynamic_trainer = dynamic_trainer
        self.concurrent_pool = concurrent_pool
        self.role_elo_tracker = role_elo_tracker

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def learner_entry_id(self) -> int | None:
        with self._learner_lock:
            return self._learner_entry_id

    @learner_entry_id.setter
    def learner_entry_id(self, value: int | None) -> None:
        with self._learner_lock:
            self._learner_entry_id = value

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        """Start the tournament background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Tournament thread already running")
            return
        self._stop_event.clear()
        if self.gauntlet is not None:
            self.gauntlet.set_stop_event(self._stop_event)
        self._thread = threading.Thread(
            target=self._run_loop, name="league-tournament", daemon=True,
        )
        self._thread.start()
        logger.info(
            "Tournament started: device=%s, num_envs=%d, games_per_match=%d",
            self.device, self.num_envs, self.games_per_match,
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Signal the tournament thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Tournament thread did not stop within %.0fs", timeout)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Main loop ────────────────────────────────────────────

    def _run_loop(self) -> None:
        """Main tournament loop (runs in background thread)."""
        logger.info("Tournament thread started")

        # Lazy-import VecEnv so the module can be imported without shogi_gym
        from shogi_gym import VecEnv

        # Local num_envs intentionally overrides self.num_envs when
        # concurrent pool is configured (pool manages its own env count).
        num_envs = (
            self.concurrent_pool.config.total_envs
            if self.concurrent_pool is not None
            else self.num_envs
        )
        vecenv = VecEnv(
            num_envs=num_envs,
            max_ply=self.max_ply,
            observation_mode="katago",  # type: ignore[call-arg]
            action_mode="spatial",
        )

        round_number = 0

        try:
            while not self._stop_event.is_set():
                # Total entry count is sufficient: generate_round pairs ALL
                # entries regardless of role.  Role diversity is irrelevant for
                # round-robin — any entry can play any other.
                entries = self.store.list_entries()
                if len(entries) < self.min_pool_size:
                    self._stop_event.wait(self.pause_seconds * 2)
                    continue

                # Wait for training to reach epoch 5 before starting rounds
                # (ensures a few entries exist and initial instability settles).
                epoch = self._current_epoch()
                if epoch < 5:
                    self._stop_event.wait(self.pause_seconds)
                    continue

                pairings = self.scheduler.generate_round(entries)
                if not pairings:
                    self._stop_event.wait(self.pause_seconds)
                    continue

                round_number += 1
                logger.info(
                    "Tournament round %d (epoch %d): %d entries, %d pairings",
                    round_number, epoch, len(entries), len(pairings),
                )

                if self.concurrent_pool is not None:
                    self._run_concurrent_round(vecenv, pairings, epoch)
                else:
                    any_played = False
                    for entry_a, entry_b in pairings:
                        if self._stop_event.is_set():
                            break
                        try:
                            games_played = self._run_one_match(
                                vecenv, entry_a, entry_b,
                                epoch=epoch, num_envs=num_envs,
                            )
                        except Exception:
                            logger.exception(
                                "Match failed: %s vs %s",
                                entry_a.display_name, entry_b.display_name,
                            )
                            continue
                        if games_played > 0 and self.scheduler.priority_scorer is not None:
                            any_played = True
                            self.scheduler.priority_scorer.record_round_result(
                                entry_a.id, entry_b.id,
                            )
                            self.scheduler.priority_scorer.record_result(
                                entry_a.id, entry_b.id,
                            )
                        self._stop_event.wait(self.pause_seconds)

                    # Advance round even if stop_event fired mid-round: a partial
                    # round is still a round for scorer state purposes.  Not
                    # advancing would freeze repeat-penalty decay, and since
                    # stop is typically for shutdown, the next session starts fresh.
                    if any_played and self.scheduler.priority_scorer is not None:
                        self.scheduler.priority_scorer.advance_round()

                logger.info("Tournament round %d complete", round_number)

                # Run historical gauntlet if due (Phase 2)
                if (
                    self.gauntlet
                    and self.learner_entry_id is not None
                    and self.historical_library
                    and self.gauntlet.is_due(epoch)
                    and not self._stop_event.is_set()
                ):
                    try:
                        self.historical_library.refresh(epoch)
                        slots = self.historical_library.get_slots()
                        learner_entry = self.store.get_entry(self.learner_entry_id)
                        if learner_entry and slots:
                            self.gauntlet.run_gauntlet(
                                epoch, learner_entry, slots, vecenv=vecenv,
                            )
                    except Exception:
                        logger.exception("Gauntlet failed at epoch %d", epoch)
        except Exception:
            logger.exception("Tournament thread crashed")
        finally:
            logger.info("Tournament thread stopped")

    # ── Trainability ──────────────────────────────────────────

    def _is_trainable_match(
        self, entry_a: OpponentEntry, entry_b: OpponentEntry,
    ) -> bool:
        """D-vs-D or D-vs-RF produces training data (§8.2 / §10.1)."""
        return is_training_match(entry_a, entry_b)

    # ── Concurrent round ───────────────────────────────────

    def _run_concurrent_round(
        self,
        vecenv: object,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
        epoch: int,
    ) -> None:
        """Run a round using the ConcurrentMatchPool."""
        assert self.concurrent_pool is not None

        # Closures capture self.device by reference — safe because device is
        # set once in __init__ and never reassigned.
        def _load_fn(entry: OpponentEntry) -> object:
            return self.store.load_opponent(entry, device=str(self.device))

        def _release_fn(model_a: object, model_b: object) -> None:
            release_models(model_a, model_b, device_type=self.device.type)

        results = self.concurrent_pool.run_round(
            vecenv,
            pairings,
            load_fn=_load_fn,
            release_fn=_release_fn,
            device=self.device,
            games_per_match=self.games_per_match,
            max_ply=self.max_ply,
            stop_event=self._stop_event,
            trainable_fn=self._is_trainable_match if self.dynamic_trainer else None,
        )
        for result in results:
            total = result.a_wins + result.b_wins + result.draws
            if total == 0:
                continue
            try:
                current_a = self.store.get_entry(result.entry_a.id)
                current_b = self.store.get_entry(result.entry_b.id)
                if current_a is None or current_b is None:
                    continue
                # Skip results for entries retired mid-round (race with tier manager)
                if current_a.status != EntryStatus.ACTIVE or current_b.status != EntryStatus.ACTIVE:
                    logger.info(
                        "  Skipping result %s vs %s — entry retired mid-round",
                        result.entry_a.display_name, result.entry_b.display_name,
                    )
                    continue
                result_score = majority_wins_result(result.a_wins, result.b_wins, result.draws)
                context = RoleEloTracker.determine_match_context(
                    current_a, current_b,
                )
                k = (
                    self.role_elo_tracker.k_for_context(context)
                    if self.role_elo_tracker
                    else self.k_factor
                )
                new_a_elo, new_b_elo = compute_elo_update(
                    current_a.elo_rating, current_b.elo_rating,
                    result=result_score, k=k,
                )
                is_train = is_training_match(current_a, current_b)
                # §9.1/13.3: store role-specific Elo (not composite) in match
                # records so analytics see the context-appropriate view.
                if self.role_elo_tracker:
                    col_a, col_b = self.role_elo_tracker.columns_for_context(
                        current_a, current_b, context,
                    )
                    elo_before_a = getattr(current_a, col_a.value)
                    elo_before_b = getattr(current_b, (col_b or col_a).value)
                    role_new_a, role_new_b = compute_elo_update(
                        elo_before_a, elo_before_b, result=result_score, k=k,
                    )
                else:
                    elo_before_a = current_a.elo_rating
                    elo_before_b = current_b.elo_rating
                    role_new_a = new_a_elo
                    role_new_b = new_b_elo
                self.store.record_result(
                    epoch=epoch,
                    entry_a_id=result.entry_a.id,
                    entry_b_id=result.entry_b.id,
                    wins_a=result.a_wins,
                    wins_b=result.b_wins,
                    draws=result.draws,
                    match_type="train" if is_train else "calibration",
                    role_a=current_a.role,
                    role_b=current_b.role,
                    elo_before_a=elo_before_a,
                    elo_after_a=role_new_a,
                    elo_before_b=elo_before_b,
                    elo_after_b=role_new_b,
                    training_updates_a=current_a.update_count,
                    training_updates_b=current_b.update_count,
                )
                self.store.update_elo(result.entry_a.id, new_a_elo, epoch=epoch)
                self.store.update_elo(result.entry_b.id, new_b_elo, epoch=epoch)
                if self.role_elo_tracker:
                    self.role_elo_tracker.update_from_result(
                        current_a, current_b, result_score, context,
                    )
                logger.info(
                    "  %s vs %s — %dW %dL %dD",
                    result.entry_a.display_name, result.entry_b.display_name,
                    result.a_wins, result.b_wins, result.draws,
                )
                if self.dynamic_trainer and result.rollout is not None:
                    for i, entry in enumerate([current_a, current_b]):
                        if entry.role == Role.DYNAMIC:
                            self.dynamic_trainer.record_match(
                                entry.id, result.rollout, side=i,
                            )
                            if (
                                self.dynamic_trainer.should_update(entry.id)
                                and not self.dynamic_trainer.is_rate_limited()
                                and not self.dynamic_trainer.is_gpu_backpressured(str(self.device))
                            ):
                                self.dynamic_trainer.update(
                                    entry, device=str(self.device),
                                )
            except Exception:
                logger.exception(
                    "Failed to process result: %s vs %s",
                    result.entry_a.display_name, result.entry_b.display_name,
                )

        # Update priority scorer state after concurrent round
        if self.scheduler.priority_scorer is not None:
            any_played = False
            for result in results:
                total = result.a_wins + result.b_wins + result.draws
                if total == 0:
                    continue
                any_played = True
                # Record once per match (not per game) to match
                # the sequential path — per-game recording inflates
                # _pair_games count and collapses _under_sample_bonus.
                self.scheduler.priority_scorer.record_result(
                    result.entry_a.id, result.entry_b.id,
                )
                self.scheduler.priority_scorer.record_round_result(
                    result.entry_a.id, result.entry_b.id,
                )
            if any_played:
                self.scheduler.priority_scorer.advance_round()

    # ── Per-match logic ─────────────────────────────────────

    def _run_one_match(
        self,
        vecenv: object,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
        epoch: int,
        num_envs: int | None = None,
    ) -> int:
        """Play a single match, update Elo, and trigger training if applicable.

        Returns the total number of games actually played.
        """
        is_trainable = (
            self.dynamic_trainer is not None
            and self._is_trainable_match(entry_a, entry_b)
        )

        if is_trainable:
            result = self._play_match(
                vecenv, entry_a, entry_b, collect_rollout=True,
                num_envs=num_envs,
            )
            wins_a, wins_b, draws, rollout = result
        else:
            wins_a, wins_b, draws = self._play_match(
                vecenv, entry_a, entry_b, num_envs=num_envs,
            )
            rollout = None

        total = wins_a + wins_b + draws
        if total > 0:
            current_a = self.store.get_entry(entry_a.id)
            current_b = self.store.get_entry(entry_b.id)
            if current_a is None or current_b is None:
                return total  # entry retired mid-round
            if current_a.status != EntryStatus.ACTIVE or current_b.status != EntryStatus.ACTIVE:
                logger.info(
                    "  Skipping %s vs %s — entry retired mid-round",
                    entry_a.display_name, entry_b.display_name,
                )
                return total
            result_score = majority_wins_result(wins_a, wins_b, draws)
            context = RoleEloTracker.determine_match_context(
                current_a, current_b,
            )
            k = (
                self.role_elo_tracker.k_for_context(context)
                if self.role_elo_tracker
                else self.k_factor
            )
            new_a_elo, new_b_elo = compute_elo_update(
                current_a.elo_rating, current_b.elo_rating,
                result=result_score, k=k,
            )
            # §9.1/13.3: store role-specific Elo in match records
            if self.role_elo_tracker:
                col_a, col_b = self.role_elo_tracker.columns_for_context(
                    current_a, current_b, context,
                )
                elo_before_a = getattr(current_a, col_a.value)
                elo_before_b = getattr(current_b, (col_b or col_a).value)
                role_new_a, role_new_b = compute_elo_update(
                    elo_before_a, elo_before_b, result=result_score, k=k,
                )
            else:
                elo_before_a = current_a.elo_rating
                elo_before_b = current_b.elo_rating
                role_new_a = new_a_elo
                role_new_b = new_b_elo
            self.store.record_result(
                epoch=epoch, entry_a_id=entry_a.id, entry_b_id=entry_b.id,
                wins_a=wins_a, wins_b=wins_b, draws=draws,
                match_type="train" if is_trainable else "calibration",
                role_a=current_a.role,
                role_b=current_b.role,
                elo_before_a=elo_before_a,
                elo_after_a=role_new_a,
                elo_before_b=elo_before_b,
                elo_after_b=role_new_b,
                training_updates_a=current_a.update_count,
                training_updates_b=current_b.update_count,
            )
            self.store.update_elo(entry_a.id, new_a_elo, epoch=epoch)
            self.store.update_elo(entry_b.id, new_b_elo, epoch=epoch)
            if self.role_elo_tracker:
                self.role_elo_tracker.update_from_result(
                    current_a, current_b, result_score, context,
                )
            logger.info(
                "  %s vs %s — %dW %dL %dD",
                entry_a.display_name, entry_b.display_name,
                wins_a, wins_b, draws,
            )

            # Training trigger (after Elo update) — use re-fetched entries
            # so role changes between pairing and completion are respected.
            if is_trainable and rollout is not None:
                for i, entry in enumerate([current_a, current_b]):
                    if entry.role == Role.DYNAMIC:
                        self.dynamic_trainer.record_match(entry.id, rollout, side=i)
                        if (
                            self.dynamic_trainer.should_update(entry.id)
                            and not self.dynamic_trainer.is_rate_limited()
                            and not self.dynamic_trainer.is_gpu_backpressured(str(self.device))
                        ):
                            self.dynamic_trainer.update(
                                entry, device=str(self.device),
                            )

        return total

    # ── Helpers ──────────────────────────────────────────────

    def _current_epoch(self) -> int:
        """Read the current training epoch from the DB."""
        return self.store.get_current_epoch()

    # ── Match execution ──────────────────────────────────────

    def _play_match(
        self,
        vecenv: object,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
        collect_rollout: bool = False,
        num_envs: int | None = None,
    ) -> tuple[int, int, int] | tuple[int, int, int, MatchRollout]:
        """Play a set of games between two frozen models.

        Returns (a_wins, b_wins, draws), or (a_wins, b_wins, draws, MatchRollout)
        when collect_rollout=True.
        """
        model_a = self.store.load_opponent(entry_a, device=str(self.device))
        model_b = self.store.load_opponent(entry_b, device=str(self.device))

        try:
            return play_match(
                vecenv, model_a, model_b,
                device=self.device, num_envs=num_envs or self.num_envs,
                max_ply=self.max_ply, games_target=self.games_per_match,
                stop_event=self._stop_event,
                collect_rollout=collect_rollout,
            )
        finally:
            release_models(model_a, model_b, device_type=self.device.type)
