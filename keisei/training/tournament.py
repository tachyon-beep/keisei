"""Background round-robin tournament for Elo calibration.

Runs inference-only matches between pairs of league entries to produce
accurate Elo ratings across the full pool — not just learner-vs-opponent.
Models are loaded frozen (no gradients, no training). Results are written
to the same league_results / elo_history tables used by the main loop.

Usage:
    tournament = LeagueTournament(db_path, league_dir, device="cuda:1")
    tournament.start()   # background thread
    ...
    tournament.stop()    # graceful shutdown
"""

from __future__ import annotations

import itertools
import logging
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from keisei.training.league import OpponentEntry, compute_elo_update
from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


class LeagueTournament:
    """Background thread that runs round-robin matches between pool entries."""

    def __init__(
        self,
        db_path: str,
        league_dir: str,
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
            db_path: Path to the shared SQLite database.
            league_dir: Directory containing league checkpoint files.
            device: Torch device for inference (should be separate from trainer).
            num_envs: Number of concurrent environments per match.
            max_ply: Maximum ply per game before truncation.
            games_per_match: Total games to play per matchup (split across envs).
            k_factor: Elo K-factor. Lower than main loop (16 vs 32) since
                      these are calibration matches, not primary evaluations.
            pause_seconds: Sleep between matches to avoid starving the main loop.
            min_pool_size: Don't run matches until pool has at least this many entries.
        """
        self.db_path = db_path
        self.league_dir = Path(league_dir)
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
            observation_mode="katago",
            action_mode="spatial",
        )

        conn = self._open_connection()

        try:
            while not self._stop_event.is_set():
                entries = self._load_entries(conn)
                if len(entries) < self.min_pool_size:
                    self._stop_event.wait(self.pause_seconds * 2)
                    continue

                pair = self._pick_pair(conn, entries)
                if pair is None:
                    self._stop_event.wait(self.pause_seconds)
                    continue

                entry_a, entry_b = pair
                try:
                    wins_a, wins_b, draws = self._play_match(
                        vecenv, entry_a, entry_b,
                    )
                except Exception:
                    logger.exception(
                        "Match failed: %s vs %s", entry_a.display_name, entry_b.display_name,
                    )
                    self._stop_event.wait(self.pause_seconds)
                    continue

                total = wins_a + wins_b + draws
                if total > 0:
                    epoch = self._current_epoch(conn)
                    self._record_result(conn, entry_a, entry_b, wins_a, wins_b, draws, epoch)
                    logger.info(
                        "Tournament: %s vs %s — %dW %dL %dD",
                        entry_a.display_name, entry_b.display_name,
                        wins_a, wins_b, draws,
                    )

                self._stop_event.wait(self.pause_seconds)
        except Exception:
            logger.exception("Tournament thread crashed")
        finally:
            conn.close()
            logger.info("Tournament thread stopped")

    # ── DB helpers ───────────────────────────────────────────

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _current_epoch(self, conn: sqlite3.Connection) -> int:
        """Read the current training epoch from the DB."""
        try:
            row = conn.execute(
                "SELECT current_epoch FROM training_state WHERE id = 1"
            ).fetchone()
            return row["current_epoch"] if row else 0
        except Exception:
            return 0

    def _load_entries(self, conn: sqlite3.Connection) -> list[OpponentEntry]:
        rows = conn.execute(
            "SELECT * FROM league_entries ORDER BY elo_rating DESC"
        ).fetchall()
        return [OpponentEntry.from_db_row(r) for r in rows]

    def _pick_pair(
        self, conn: sqlite3.Connection, entries: list[OpponentEntry],
    ) -> tuple[OpponentEntry, OpponentEntry] | None:
        """Pick the pair with the fewest head-to-head games."""
        if len(entries) < 2:
            return None

        # Count existing h2h games for each pair
        h2h_counts: dict[tuple[int, int], int] = {}
        rows = conn.execute(
            "SELECT learner_id, opponent_id, SUM(wins + losses + draws) as total "
            "FROM league_results GROUP BY learner_id, opponent_id"
        ).fetchall()
        for row in rows:
            key = (row["learner_id"], row["opponent_id"])
            h2h_counts[key] = row["total"]

        # Find the pair with minimum total games (both directions)
        best_pair = None
        best_count = float("inf")
        for a, b in itertools.combinations(entries, 2):
            count = h2h_counts.get((a.id, b.id), 0) + h2h_counts.get((b.id, a.id), 0)
            if count < best_count:
                best_count = count
                best_pair = (a, b)

        return best_pair

    def _record_result(
        self,
        conn: sqlite3.Connection,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
        wins_a: int,
        wins_b: int,
        draws: int,
        epoch: int = 0,
    ) -> None:
        """Record match result and update Elo for both entries."""
        total = wins_a + wins_b + draws

        conn.execute(
            """INSERT INTO league_results
               (epoch, learner_id, opponent_id, wins, losses, draws)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (epoch, entry_a.id, entry_b.id, wins_a, wins_b, draws),
        )

        # Update games_played for both
        conn.execute(
            "UPDATE league_entries SET games_played = games_played + ? WHERE id = ?",
            (total, entry_a.id),
        )
        conn.execute(
            "UPDATE league_entries SET games_played = games_played + ? WHERE id = ?",
            (total, entry_b.id),
        )

        # Elo update
        result_score = (wins_a + 0.5 * draws) / total
        # Re-read current Elo (may have changed since we loaded entries)
        row_a = conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = ?", (entry_a.id,)
        ).fetchone()
        row_b = conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = ?", (entry_b.id,)
        ).fetchone()
        if row_a is None or row_b is None:
            conn.commit()
            return  # entry was evicted mid-match

        elo_a, elo_b = row_a["elo_rating"], row_b["elo_rating"]
        new_a, new_b = compute_elo_update(elo_a, elo_b, result_score, k=self.k_factor)

        conn.execute(
            "UPDATE league_entries SET elo_rating = ? WHERE id = ?", (new_a, entry_a.id),
        )
        conn.execute(
            "UPDATE league_entries SET elo_rating = ? WHERE id = ?", (new_b, entry_b.id),
        )

        # Record Elo history at current training epoch
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (?, ?, ?)",
            (entry_a.id, epoch, new_a),
        )
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (?, ?, ?)",
            (entry_b.id, epoch, new_b),
        )

        conn.commit()
        logger.info(
            "Elo update: %s %.0f->%.0f, %s %.0f->%.0f",
            entry_a.display_name, elo_a, new_a,
            entry_b.display_name, elo_b, new_b,
        )

    # ── Match execution ──────────────────────────────────────

    def _play_match(
        self,
        vecenv: object,
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
        vecenv: object,
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
            dones = np.asarray(step_result.dones)

            # Count results from finished games
            for i in range(self.num_envs):
                if dones[i]:
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
        ckpt = Path(entry.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint missing: {ckpt}")
        model = build_model(entry.architecture, entry.model_params)
        state_dict = torch.load(ckpt, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model
