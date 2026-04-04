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
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from keisei.training.match_scheduler import MatchScheduler
from keisei.training.opponent_store import OpponentEntry, OpponentStore, compute_elo_update

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

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Lifecycle ────────────────────────────────────────────

    def start(self) -> None:
        """Start the tournament background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Tournament thread already running")
            return
        self._stop_event.clear()
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

        vecenv = VecEnv(
            num_envs=self.num_envs,
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

                last_epoch = epoch
                pairings = self.scheduler.generate_round(entries)
                if not pairings:
                    self._stop_event.wait(self.pause_seconds)
                    continue

                logger.info("Tournament round E%d: %d pairings", epoch, len(pairings))

                for entry_a, entry_b in pairings:
                    if self._stop_event.is_set():
                        break
                    try:
                        wins_a, wins_b, draws = self._play_match(
                            vecenv, entry_a, entry_b,
                        )
                    except Exception:
                        logger.exception(
                            "Match failed: %s vs %s",
                            entry_a.display_name, entry_b.display_name,
                        )
                        continue

                    total = wins_a + wins_b + draws
                    if total > 0:
                        result_score = (wins_a + 0.5 * draws) / total
                        new_a_elo, new_b_elo = compute_elo_update(
                            entry_a.elo_rating, entry_b.elo_rating,
                            result=result_score, k=self.k_factor,
                        )
                        self.store.record_result(
                            epoch=epoch, learner_id=entry_a.id, opponent_id=entry_b.id,
                            wins=wins_a, losses=wins_b, draws=draws,
                            elo_delta_a=round(new_a_elo - entry_a.elo_rating, 1),
                            elo_delta_b=round(new_b_elo - entry_b.elo_rating, 1),
                        )
                        self.store.update_elo(entry_a.id, new_a_elo, epoch=epoch)
                        self.store.update_elo(entry_b.id, new_b_elo, epoch=epoch)
                        logger.info(
                            "  %s vs %s — %dW %dL %dD",
                            entry_a.display_name, entry_b.display_name,
                            wins_a, wins_b, draws,
                        )

                    self._stop_event.wait(self.pause_seconds)

                logger.info("Tournament round E%d complete", epoch)
        except Exception:
            logger.exception("Tournament thread crashed")
        finally:
            logger.info("Tournament thread stopped")

    # ── Helpers ──────────────────────────────────────────────

    def _current_epoch(self) -> int:
        """Read the current training epoch from the DB via the store's connection."""
        try:
            row = self.store._conn.execute(
                "SELECT current_epoch FROM training_state WHERE id = 1"
            ).fetchone()
            return row["current_epoch"] if row else 0
        except Exception:
            return 0

    # ── Match execution ──────────────────────────────────────

    def _play_match(
        self,
        vecenv: Any,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
    ) -> tuple[int, int, int]:
        """Play a set of games between two frozen models. Returns (a_wins, b_wins, draws)."""
        model_a = self._load_model(entry_a)
        model_b = self._load_model(entry_b)

        total_a_wins = 0
        total_b_wins = 0
        total_draws = 0

        # Run enough batches to hit games_per_match total completed games
        games_remaining = self.games_per_match
        while games_remaining > 0 and not self._stop_event.is_set():
            a_wins, b_wins, draws = self._play_batch(vecenv, model_a, model_b)
            total_a_wins += a_wins
            total_b_wins += b_wins
            total_draws += draws
            games_remaining -= (a_wins + b_wins + draws)

        # Clean up GPU memory
        del model_a, model_b
        torch.cuda.empty_cache()

        return total_a_wins, total_b_wins, total_draws

    def _play_batch(
        self,
        vecenv: Any,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
    ) -> tuple[int, int, int]:
        """Play one batch of games (num_envs concurrent). Returns (a_wins, b_wins, draws)."""
        reset_result = vecenv.reset()
        obs = torch.from_numpy(np.asarray(reset_result.observations)).to(self.device)
        legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(self.device)
        current_players = np.zeros(self.num_envs, dtype=np.uint8)

        a_wins = 0
        b_wins = 0
        draws = 0

        for _ply in range(self.max_ply):
            if self._stop_event.is_set():
                break

            # Player A = side 0 (black), Player B = side 1 (white)
            player_a_mask = torch.tensor(current_players == 0, device=self.device)
            player_b_mask = ~player_a_mask

            actions = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

            # Model A forward
            a_indices = player_a_mask.nonzero(as_tuple=True)[0]
            if a_indices.numel() > 0:
                with torch.no_grad():
                    a_out = model_a(obs[a_indices])
                a_logits = a_out.policy_logits.reshape(a_indices.numel(), -1)
                a_masked = a_logits.masked_fill(~legal_masks[a_indices], float("-inf"))
                a_probs = F.softmax(a_masked, dim=-1)
                actions[a_indices] = torch.distributions.Categorical(a_probs).sample()

            # Model B forward
            b_indices = player_b_mask.nonzero(as_tuple=True)[0]
            if b_indices.numel() > 0:
                with torch.no_grad():
                    b_out = model_b(obs[b_indices])
                b_logits = b_out.policy_logits.reshape(b_indices.numel(), -1)
                b_masked = b_logits.masked_fill(~legal_masks[b_indices], float("-inf"))
                b_probs = F.softmax(b_masked, dim=-1)
                actions[b_indices] = torch.distributions.Categorical(b_probs).sample()

            step_result = vecenv.step(actions.cpu().numpy())
            obs = torch.from_numpy(np.asarray(step_result.observations)).to(self.device)
            legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(self.device)
            current_players = np.asarray(step_result.current_players, dtype=np.uint8)
            rewards = np.asarray(step_result.rewards)
            terminated = np.asarray(step_result.terminated)
            truncated = np.asarray(step_result.truncated)

            # Count results from finished games
            for i in range(self.num_envs):
                if terminated[i] or truncated[i]:
                    r = rewards[i]
                    if r > 0:
                        a_wins += 1  # black (A) wins
                    elif r < 0:
                        b_wins += 1  # white (B) wins
                    else:
                        draws += 1

            # If all envs have finished at least once, we have enough data
            if a_wins + b_wins + draws >= self.num_envs:
                break

        return a_wins, b_wins, draws

    def _load_model(self, entry: OpponentEntry) -> torch.nn.Module:
        """Load a frozen model for inference."""
        return self.store.load_opponent(entry, device=str(self.device))
