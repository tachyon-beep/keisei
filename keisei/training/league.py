"""Opponent league: pool management, sampling, and Elo tracking."""

from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


@dataclass
class OpponentEntry:
    """A snapshot in the opponent pool."""

    id: int
    architecture: str
    model_params: dict[str, Any]
    checkpoint_path: str
    elo_rating: float
    created_epoch: int
    games_played: int
    created_at: str

    @classmethod
    def from_db_row(cls, row: tuple) -> OpponentEntry:
        return cls(
            id=row[0],
            architecture=row[1],
            model_params=json.loads(row[2]),
            checkpoint_path=row[3],
            elo_rating=row[4],
            created_epoch=row[5],
            games_played=row[6],
            created_at=row[7],
        )


def compute_elo_update(
    rating_a: float,
    rating_b: float,
    result: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Compute new Elo ratings after a match.

    Args:
        rating_a: Player A's current rating
        rating_b: Player B's current rating
        result: 1.0 = A wins, 0.5 = draw, 0.0 = A loses
        k: K-factor controlling update magnitude

    Returns:
        (new_rating_a, new_rating_b)
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a

    new_a = rating_a + k * (result - expected_a)
    new_b = rating_b + k * ((1.0 - result) - expected_b)

    return new_a, new_b


class OpponentPool:
    """Manages the collection of checkpoint snapshots available as opponents."""

    def __init__(self, db_path: str, league_dir: str, max_pool_size: int = 20) -> None:
        self.db_path = db_path
        self.league_dir = Path(league_dir)
        self.league_dir.mkdir(parents=True, exist_ok=True)
        self.max_pool_size = max_pool_size
        self._pinned: set[int] = set()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def add_snapshot(
        self,
        model: torch.nn.Module,
        architecture: str,
        model_params: dict[str, Any],
        epoch: int,
    ) -> OpponentEntry:
        """Save a checkpoint snapshot and add it to the pool."""
        raw_model = model.module if hasattr(model, "module") else model
        ckpt_path = self.league_dir / f"{architecture}_ep{epoch:05d}.pt"
        torch.save(raw_model.state_dict(), ckpt_path)

        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO league_entries
                   (architecture, model_params, checkpoint_path, created_epoch)
                   VALUES (?, ?, ?, ?)""",
                (architecture, json.dumps(model_params), str(ckpt_path), epoch),
            )
            entry_id = cursor.lastrowid
            conn.commit()
        finally:
            conn.close()

        logger.info(
            "Pool snapshot: %s epoch %d -> %s (id=%d)",
            architecture, epoch, ckpt_path.name, entry_id,
        )

        self._evict_if_needed()

        entry = self._get_entry(entry_id)
        assert entry is not None
        return entry

    def _get_entry(self, entry_id: int) -> OpponentEntry | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, architecture, model_params, checkpoint_path, "
                "elo_rating, created_epoch, games_played, created_at "
                "FROM league_entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
            return OpponentEntry.from_db_row(row) if row else None
        finally:
            conn.close()

    def list_entries(self) -> list[OpponentEntry]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, architecture, model_params, checkpoint_path, "
                "elo_rating, created_epoch, games_played, created_at "
                "FROM league_entries ORDER BY created_epoch ASC"
            ).fetchall()
            return [OpponentEntry.from_db_row(r) for r in rows]
        finally:
            conn.close()

    def _evict_if_needed(self) -> None:
        entries = self.list_entries()
        while len(entries) > self.max_pool_size:
            evicted = False
            for entry in entries:
                if entry.id not in self._pinned:
                    self._delete_entry(entry)
                    entries = self.list_entries()
                    evicted = True
                    break
            if not evicted:
                logger.warning(
                    "All entries pinned, cannot evict to reach max_pool_size=%d",
                    self.max_pool_size,
                )
                break

    def _delete_entry(self, entry: OpponentEntry) -> None:
        ckpt = Path(entry.checkpoint_path)
        if ckpt.exists():
            ckpt.unlink()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM league_entries WHERE id = ?", (entry.id,))
            conn.commit()
        finally:
            conn.close()
        logger.info("Evicted pool entry id=%d (epoch %d)", entry.id, entry.created_epoch)

    def pin(self, entry_id: int) -> None:
        """Pin an entry to prevent eviction."""
        self._pinned.add(entry_id)

    def unpin(self, entry_id: int) -> None:
        """Release a pin."""
        self._pinned.discard(entry_id)

    def load_opponent(self, entry: OpponentEntry, device: str = "cpu") -> torch.nn.Module:
        """Load an opponent model from a pool entry."""
        ckpt = Path(entry.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint missing for pool entry id={entry.id} "
                f"(epoch {entry.created_epoch}): {ckpt}"
            )
        model = build_model(entry.architecture, entry.model_params)
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def update_elo(self, entry_id: int, new_elo: float) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE league_entries SET elo_rating = ? WHERE id = ?",
                (new_elo, entry_id),
            )
            conn.commit()
        finally:
            conn.close()

    def record_result(
        self, epoch: int, learner_id: int, opponent_id: int,
        wins: int, losses: int, draws: int,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO league_results
                   (epoch, learner_id, opponent_id, wins, losses, draws)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (epoch, learner_id, opponent_id, wins, losses, draws),
            )
            conn.execute(
                "UPDATE league_entries SET games_played = games_played + ? WHERE id = ?",
                (wins + losses + draws, opponent_id),
            )
            conn.commit()
        finally:
            conn.close()


class OpponentSampler:
    """Samples opponents from the pool using weighted historical/current-best mix."""

    def __init__(
        self,
        pool: OpponentPool,
        historical_ratio: float = 0.8,
        current_best_ratio: float = 0.2,
        elo_floor: float = 500.0,
    ) -> None:
        self.pool = pool
        self.historical_ratio = historical_ratio
        self.current_best_ratio = current_best_ratio
        self.elo_floor = elo_floor

    def sample(self) -> OpponentEntry:
        """Sample an opponent from the pool."""
        entries = self.pool.list_entries()
        if not entries:
            raise ValueError("Cannot sample from empty opponent pool")
        if len(entries) == 1:
            return entries[0]

        # Current best = most recent entry
        current_best = entries[-1]

        # Historical = all entries above elo_floor (excluding current best)
        historical = [e for e in entries[:-1] if e.elo_rating >= self.elo_floor]

        # Fallback: if no historical entries above floor, use all except current best
        if not historical:
            historical = entries[:-1]

        if random.random() < self.current_best_ratio:
            return current_best
        else:
            return random.choice(historical)
