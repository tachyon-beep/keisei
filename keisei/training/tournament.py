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
from keisei.training.match_scheduler import MatchScheduler
from keisei.training.match_utils import play_match, release_models
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role, compute_elo_update
from keisei.training.role_elo import RoleEloTracker

logger = logging.getLogger(__name__)


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
        games_per_match: int = 64,
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

        last_epoch = -1

        try:
            while not self._stop_event.is_set():
                entries = self.store.list_entries()
                if len(entries) < self.min_pool_size:
                    self._stop_event.wait(self.pause_seconds * 2)
                    continue

                epoch = self._current_epoch()
                if epoch < 5 or epoch == last_epoch:
                    self._stop_event.wait(self.pause_seconds)
                    continue

                pairings = self.scheduler.generate_round(entries)
                if not pairings:
                    self._stop_event.wait(self.pause_seconds)
                    continue

                last_epoch = epoch
                logger.info("Tournament round E%d: %d pairings", epoch, len(pairings))

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
                            for _ in range(games_played):
                                self.scheduler.priority_scorer.record_result(
                                    entry_a.id, entry_b.id,
                                )
                        self._stop_event.wait(self.pause_seconds)

                    if any_played and self.scheduler.priority_scorer is not None:
                        self.scheduler.priority_scorer.advance_round()

                logger.info("Tournament round E%d complete", epoch)

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
        """D-vs-D or D-vs-RF produces training data. D-vs-FS and Historical do not."""
        trainable_roles = {Role.DYNAMIC, Role.RECENT_FIXED}
        return (
            entry_a.role in trainable_roles
            and entry_b.role in trainable_roles
            and (entry_a.role == Role.DYNAMIC or entry_b.role == Role.DYNAMIC)
        )

    # ── Concurrent round ───────────────────────────────────

    def _run_concurrent_round(
        self,
        vecenv: object,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
        epoch: int,
    ) -> None:
        """Run a round using the ConcurrentMatchPool."""
        assert self.concurrent_pool is not None

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
            current_a = self.store.get_entry(result.entry_a.id)
            current_b = self.store.get_entry(result.entry_b.id)
            if current_a is None or current_b is None:
                continue
            result_score = (result.a_wins + 0.5 * result.draws) / total
            new_a_elo, new_b_elo = compute_elo_update(
                current_a.elo_rating, current_b.elo_rating,
                result=result_score, k=self.k_factor,
            )
            context = RoleEloTracker.determine_match_context(
                current_a, current_b,
            )
            self.store.record_result(
                epoch=epoch,
                learner_id=result.entry_a.id,
                opponent_id=result.entry_b.id,
                wins=result.a_wins,
                losses=result.b_wins,
                draws=result.draws,
                elo_delta_a=round(new_a_elo - current_a.elo_rating, 1),
                elo_delta_b=round(new_b_elo - current_b.elo_rating, 1),
                match_context=context,
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
                        ):
                            self.dynamic_trainer.update(
                                entry, device=str(self.device),
                            )

        # Update priority scorer state after concurrent round
        if self.scheduler.priority_scorer is not None:
            any_played = False
            for result in results:
                total = result.a_wins + result.b_wins + result.draws
                if total == 0:
                    continue
                any_played = True
                for _ in range(total):
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
            result_score = (wins_a + 0.5 * draws) / total
            new_a_elo, new_b_elo = compute_elo_update(
                current_a.elo_rating, current_b.elo_rating,
                result=result_score, k=self.k_factor,
            )
            context = RoleEloTracker.determine_match_context(
                current_a, current_b,
            )
            self.store.record_result(
                epoch=epoch, learner_id=entry_a.id, opponent_id=entry_b.id,
                wins=wins_a, losses=wins_b, draws=draws,
                elo_delta_a=round(new_a_elo - current_a.elo_rating, 1),
                elo_delta_b=round(new_b_elo - current_b.elo_rating, 1),
                match_context=context,
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
