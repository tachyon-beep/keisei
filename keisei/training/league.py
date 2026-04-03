"""Opponent league: pool management, sampling, and Elo tracking."""

from __future__ import annotations

import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)

# Themed name pool for league entries — Japanese shogi/martial arts themed.
# Each snapshot gets a unique name from this pool, assigned deterministically
# from the epoch number so names are stable across restarts.
LEAGUE_NAMES: list[str] = [
    # Batch 1 — warriors and strategists
    "Takeshi", "Haruka", "Renjiro", "Sakura", "Noboru",
    "Kaede", "Shingen", "Tomoe", "Genryu", "Hana",
    "Kenshin", "Mizuki", "Raiden", "Ayame", "Daisuke",
    "Shiori", "Hayato", "Natsuki", "Sorin", "Yuki",
    "Jubei", "Kasumi", "Tetsuo", "Akane", "Goemon",
    # Batch 2 — poets and wanderers
    "Rin", "Masato", "Chihiro", "Ryoma", "Satsuki",
    "Kojiro", "Fumiko", "Hideki", "Koharu", "Saburo",
    "Mio", "Tadashi", "Hotaru", "Isamu", "Nanami",
    "Shiro", "Kaori", "Benkei", "Sumire", "Yasuo",
    "Tsubaki", "Goro", "Hikari", "Kenji", "Aoi",
    # Batch 3 — legends and scholars
    "Musashi", "Hanzo", "Kagero", "Tsukasa", "Rinko",
    "Souji", "Yukimura", "Makoto", "Azami", "Taro",
    "Hinata", "Shizuka", "Ieyasu", "Momiji", "Tetsu",
    "Kurama", "Suzu", "Dosetsu", "Fubuki", "Sei",
    "Nagato", "Chiyo", "Ranmaru", "Mayumi", "Jinbei",
    # Batch 4 — elements and seasons
    "Arashi", "Tsukimi", "Enma", "Wakaba", "Ryusei",
    "Asuka", "Kagerou", "Tamamo", "Sora", "Inari",
    "Kaze", "Kurenai", "Yamato", "Suzume", "Homura",
    "Michiru", "Hayate", "Kaguya", "Ibuki", "Akira",
    "Minato", "Shinobu", "Sessho", "Uzume", "Ginga",
]


_FLAVOUR_POOLS: dict[str, list[str]] = {
    "Favourite piece": [
        "Gold General", "Silver General", "Knight", "Lance", "Bishop",
        "Rook", "Promoted Pawn", "King", "Dragon Horse", "Dragon King",
    ],
    "Favourite opening": [
        "Static Rook", "Ranging Rook", "Fourth File Rook", "Central Rook",
        "Opposing Rook", "Double Wing Attack", "Snow Roof", "Bear-in-the-Hole",
        "Right King", "Yagura",
    ],
    "Philosophy": [
        "Musashi", "Sun Tzu", "Confucius", "Lao Tzu", "Miyamoto",
        "Leibniz", "Turing", "Shannon", "Von Neumann", "Bellman",
    ],
    "Favourite snack": [
        "Onigiri", "Mochi", "Dango", "Taiyaki", "Senbei",
        "Gradient soup", "Loss crumble", "Entropy tea", "Batch noodles", "Tensor rolls",
    ],
    "Training motto": [
        '"Loss goes down"', '"Explore everything"', '"Patience is policy"',
        '"Trust the gradient"', '"Variance is the enemy"', '"Clip wisely"',
        '"Entropy is freedom"', '"Value the position"', '"Every ply counts"',
        '"Promote early"',
    ],
    "Lucky number": [
        "0.0001", "0.99", "42", "3.14", "2048",
        "0.95", "1e-8", "256", "0.2", "7.5M",
    ],
}


def _generate_flavour_facts(epoch: int, num_facts: int = 3) -> list[list[str]]:
    """Pick random flavour facts for a league entry, deterministic by epoch."""
    rng = random.Random(f"flavour-{epoch}")
    categories = rng.sample(list(_FLAVOUR_POOLS.keys()), min(num_facts, len(_FLAVOUR_POOLS)))
    return [[cat, rng.choice(_FLAVOUR_POOLS[cat])] for cat in categories]


def _generate_display_name(epoch: int, existing_names: set[str]) -> str:
    """Generate a unique display name for a league entry.

    Shuffles the full name pool deterministically by epoch, then picks
    the first name not already in use. With 100 names and max pool
    size 20, collisions are impossible unless the pool exceeds the
    name count.
    """
    rng = random.Random(f"name-{epoch}")
    shuffled = list(LEAGUE_NAMES)
    rng.shuffle(shuffled)
    for name in shuffled:
        if name not in existing_names:
            return name
    # Fallback: all 100 names taken (pool > 100). Add epoch suffix.
    return f"{shuffled[0]} E{epoch}"


@dataclass(frozen=True)
class OpponentEntry:
    """A snapshot in the opponent pool."""

    id: int
    display_name: str
    architecture: str
    model_params: dict[str, Any]
    checkpoint_path: str
    elo_rating: float
    created_epoch: int
    games_played: int
    created_at: str
    flavour_facts: list[list[str]]

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> OpponentEntry:
        raw_facts = row["flavour_facts"] if "flavour_facts" in row.keys() else "[]"
        return cls(
            id=row["id"],
            display_name=row["display_name"],
            architecture=row["architecture"],
            model_params=json.loads(row["model_params"]),
            checkpoint_path=row["checkpoint_path"],
            elo_rating=row["elo_rating"],
            created_epoch=row["created_epoch"],
            games_played=row["games_played"],
            created_at=row["created_at"],
            flavour_facts=json.loads(raw_facts),
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
        # NOTE: Pins are in-memory only — lost on restart. This is a known
        # limitation; persisting to DB is tracked as keisei-76cc7fdc85.
        self._pinned: set[int] = set()
        self._conn = self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        # Single-thread-only: the held connection is not protected by a lock.
        # Do not share an OpponentPool instance across threads.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def close(self) -> None:
        """Close the held database connection."""
        self._conn.close()

    def __enter__(self) -> OpponentPool:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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
        # Atomic write: save to .tmp first, then rename. A crash mid-write
        # leaves the .tmp file (harmless) instead of a corrupt .pt file
        # with a valid DB row pointing at it.
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save(raw_model.state_dict(), tmp_path)
        tmp_path.rename(ckpt_path)

        # Generate a unique themed name and flavour facts for this snapshot
        existing_names = {e.display_name for e in self.list_entries()}
        display_name = _generate_display_name(epoch, existing_names)
        flavour_facts = _generate_flavour_facts(epoch)

        cursor = self._conn.execute(
            """INSERT INTO league_entries
               (display_name, flavour_facts, architecture, model_params, checkpoint_path, created_epoch)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (display_name, json.dumps(flavour_facts), architecture, json.dumps(model_params), str(ckpt_path), epoch),
        )
        entry_id = cursor.lastrowid
        self._evict_if_needed()
        self._conn.commit()

        logger.info(
            "Pool snapshot: %s (%s) epoch %d -> %s (id=%d)",
            display_name, architecture, epoch, ckpt_path.name, entry_id,
        )

        entry = self._get_entry(entry_id)
        assert entry is not None
        return entry

    def _get_entry(self, entry_id: int) -> OpponentEntry | None:
        row = self._conn.execute(
            "SELECT * FROM league_entries WHERE id = ?", (entry_id,),
        ).fetchone()
        return OpponentEntry.from_db_row(row) if row else None

    def list_entries(self) -> list[OpponentEntry]:
        rows = self._conn.execute(
            "SELECT * FROM league_entries ORDER BY created_epoch ASC"
        ).fetchall()
        return [OpponentEntry.from_db_row(r) for r in rows]

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
        # Clean up all FK references before deleting the entry.
        # DB rows are deleted first so a crash never leaves an orphaned row
        # pointing at a missing checkpoint file. An orphaned .pt file on disk
        # is harmless; an orphaned DB row causes FileNotFoundError at load time.
        # Caller is responsible for committing the transaction.
        self._conn.execute(
            "DELETE FROM league_results WHERE learner_id = ? OR opponent_id = ?",
            (entry.id, entry.id),
        )
        self._conn.execute(
            "DELETE FROM elo_history WHERE entry_id = ?",
            (entry.id,),
        )
        self._conn.execute(
            "UPDATE game_snapshots SET opponent_id = NULL WHERE opponent_id = ?",
            (entry.id,),
        )
        self._conn.execute("DELETE FROM league_entries WHERE id = ?", (entry.id,))
        # File cleanup after DB — safe to fail without leaving broken references.
        ckpt = Path(entry.checkpoint_path)
        if ckpt.exists():
            ckpt.unlink()
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
        model = model.to(device)
        model.eval()
        return model

    def update_elo(self, entry_id: int, new_elo: float, epoch: int = 0) -> None:
        self._conn.execute(
            "UPDATE league_entries SET elo_rating = ? WHERE id = ?",
            (new_elo, entry_id),
        )
        self._conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (?, ?, ?)",
            (entry_id, epoch, new_elo),
        )
        self._conn.commit()

    def record_result(
        self, epoch: int, learner_id: int, opponent_id: int,
        wins: int, losses: int, draws: int,
    ) -> None:
        self._conn.execute(
            """INSERT INTO league_results
               (epoch, learner_id, opponent_id, wins, losses, draws)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (epoch, learner_id, opponent_id, wins, losses, draws),
        )
        total_games = wins + losses + draws
        self._conn.execute(
            "UPDATE league_entries SET games_played = games_played + ? WHERE id = ?",
            (total_games, learner_id),
        )
        self._conn.execute(
            "UPDATE league_entries SET games_played = games_played + ? WHERE id = ?",
            (total_games, opponent_id),
        )
        self._conn.commit()


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

        # Current best = highest Elo entry (not most recent — learner may regress)
        current_best = max(entries, key=lambda e: e.elo_rating)

        # Historical = all entries above elo_floor, excluding current_best
        historical = [
            e for e in entries
            if e.id != current_best.id and e.elo_rating >= self.elo_floor
        ]

        # If no historical entries above floor, sample current_best only.
        # The fallback to all entries would defeat the floor's purpose.
        if not historical:
            return current_best

        if random.random() < self.current_best_ratio:
            return current_best
        else:
            return random.choice(historical)
