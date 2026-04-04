"""OpponentStore: tiered pool storage layer with enums, transactions, and lineage."""

from __future__ import annotations

import json
import logging
import pickle
import random
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Generator

import torch

from keisei.training.model_registry import build_model

logger = logging.getLogger(__name__)


class Role(StrEnum):
    FRONTIER_STATIC = "frontier_static"
    RECENT_FIXED = "recent_fixed"
    DYNAMIC = "dynamic"
    UNASSIGNED = "unassigned"


class EntryStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"
    ARCHIVED = "archived"


class EloColumn(StrEnum):
    FRONTIER = "elo_frontier"
    DYNAMIC = "elo_dynamic"
    RECENT = "elo_recent"
    HISTORICAL = "elo_historical"


# Themed name pool for league entries — Japanese shogi/martial arts themed.
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
    "Favourite philosopher": [
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
    """Generate a unique display name for a league entry."""
    rng = random.Random(f"name-{epoch}")
    shuffled = list(LEAGUE_NAMES)
    rng.shuffle(shuffled)
    for name in shuffled:
        if name not in existing_names:
            return name
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
    role: Role = Role.UNASSIGNED
    status: EntryStatus = EntryStatus.ACTIVE
    parent_entry_id: int | None = None
    lineage_group: str | None = None
    protection_remaining: int = 0
    last_match_at: str | None = None
    elo_frontier: float = 1000.0
    elo_dynamic: float = 1000.0
    elo_recent: float = 1000.0
    elo_historical: float = 1000.0

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> OpponentEntry:
        raw_facts = row["flavour_facts"] if "flavour_facts" in row.keys() else "[]"
        keys = row.keys()
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
            role=Role(row["role"]) if "role" in keys else Role.UNASSIGNED,
            status=EntryStatus(row["status"]) if "status" in keys else EntryStatus.ACTIVE,
            parent_entry_id=row["parent_entry_id"] if "parent_entry_id" in keys else None,
            lineage_group=row["lineage_group"] if "lineage_group" in keys else None,
            protection_remaining=row["protection_remaining"] if "protection_remaining" in keys else 0,
            last_match_at=row["last_match_at"] if "last_match_at" in keys else None,
            elo_frontier=row["elo_frontier"] if "elo_frontier" in keys else 1000.0,
            elo_dynamic=row["elo_dynamic"] if "elo_dynamic" in keys else 1000.0,
            elo_recent=row["elo_recent"] if "elo_recent" in keys else 1000.0,
            elo_historical=row["elo_historical"] if "elo_historical" in keys else 1000.0,
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


class OpponentStore:
    """Manages the collection of checkpoint snapshots available as opponents."""

    def __init__(self, db_path: str, league_dir: str) -> None:
        self.db_path = db_path
        self.league_dir = Path(league_dir)
        self.league_dir.mkdir(parents=True, exist_ok=True)
        # NOTE: Pins are in-memory only — lost on restart. This is a known
        # limitation; persisting to DB is tracked as keisei-76cc7fdc85.
        self._pinned: set[int] = set()
        self._lock = threading.RLock()
        self._in_transaction = False
        self._conn = self._open_connection()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def close(self) -> None:
        """Close the held database connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> OpponentStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Transaction support
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Atomic multi-operation context. Holds lock, defers commit."""
        with self._lock:
            self._in_transaction = True
            try:
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            finally:
                self._in_transaction = False

    # ------------------------------------------------------------------
    # Entry CRUD
    # ------------------------------------------------------------------

    def add_entry(
        self,
        model: torch.nn.Module,
        architecture: str,
        model_params: dict[str, Any],
        epoch: int,
        role: Role = Role.UNASSIGNED,
    ) -> OpponentEntry:
        """Save a checkpoint snapshot and add it to the store."""
        raw_model: torch.nn.Module = model.module if hasattr(model, "module") else model  # type: ignore[assignment]

        with self._lock:
            existing_names = {e.display_name for e in self._list_entries_unlocked()}
            display_name = _generate_display_name(epoch, existing_names)
            flavour_facts = _generate_flavour_facts(epoch)

            cursor = self._conn.execute(
                """INSERT INTO league_entries
                   (display_name, flavour_facts, architecture, model_params,
                    checkpoint_path, created_epoch, role, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (display_name, json.dumps(flavour_facts), architecture,
                 json.dumps(model_params), "", epoch, role, EntryStatus.ACTIVE),
            )
            entry_id = cursor.lastrowid
            assert entry_id is not None

            ckpt_path = self.league_dir / f"{architecture}_ep{epoch:05d}_id{entry_id}.pt"
            tmp_path = ckpt_path.with_suffix(".pt.tmp")
            torch.save(raw_model.state_dict(), tmp_path)
            tmp_path.rename(ckpt_path)

            self._conn.execute(
                "UPDATE league_entries SET checkpoint_path = ? WHERE id = ?",
                (str(ckpt_path), entry_id),
            )

            self.log_transition(entry_id, None, role, None, EntryStatus.ACTIVE, "added")

            if not self._in_transaction:
                self._conn.commit()

            logger.info(
                "Store entry: %s (%s) epoch %d -> %s (id=%d)",
                display_name, architecture, epoch, ckpt_path.name, entry_id,
            )

            entry = self._get_entry(entry_id)
            assert entry is not None
            return entry

    def clone_entry(self, source_entry_id: int, new_role: Role, reason: str) -> OpponentEntry:
        """Clone an entry: copy checkpoint file, create new DB row with lineage."""
        with self._lock:
            source = self._get_entry(source_entry_id)
            if source is None:
                raise ValueError(f"Source entry {source_entry_id} not found")
            src_path = Path(source.checkpoint_path)
            existing_names = {e.display_name for e in self._list_entries_unlocked()}
            display_name = _generate_display_name(source.created_epoch, existing_names)
            flavour_facts = _generate_flavour_facts(source.created_epoch + source_entry_id)
            cursor = self._conn.execute(
                """INSERT INTO league_entries
                   (display_name, flavour_facts, architecture, model_params,
                    checkpoint_path, created_epoch, role, status,
                    parent_entry_id, lineage_group)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (display_name, json.dumps(flavour_facts), source.architecture,
                 json.dumps(source.model_params), "", source.created_epoch,
                 new_role, EntryStatus.ACTIVE, source_entry_id,
                 source.lineage_group or f"lineage-{source_entry_id}"),
            )
            entry_id = cursor.lastrowid
            assert entry_id is not None
            dst_path = self.league_dir / f"{source.architecture}_ep{source.created_epoch:05d}_id{entry_id}.pt"
            shutil.copy2(str(src_path), str(dst_path))
            self._conn.execute(
                "UPDATE league_entries SET checkpoint_path = ? WHERE id = ?",
                (str(dst_path), entry_id),
            )
            self.log_transition(entry_id, None, new_role, None, EntryStatus.ACTIVE, reason)
            if not self._in_transaction:
                self._conn.commit()
            entry = self._get_entry(entry_id)
            assert entry is not None
            return entry

    def retire_entry(self, entry_id: int, reason: str) -> None:
        """Mark an entry as retired. Does NOT delete the checkpoint file."""
        with self._lock:
            entry = self._get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Entry {entry_id} not found")
            old_role = entry.role
            self._conn.execute(
                "UPDATE league_entries SET status = ? WHERE id = ?",
                (EntryStatus.RETIRED, entry_id),
            )
            self.log_transition(
                entry_id, old_role, old_role,
                EntryStatus.ACTIVE, EntryStatus.RETIRED, reason,
            )
            if not self._in_transaction:
                self._conn.commit()

    def update_role(self, entry_id: int, new_role: Role, reason: str) -> None:
        """Change the role of an entry."""
        with self._lock:
            entry = self._get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Entry {entry_id} not found")
            old_role = entry.role
            self._conn.execute(
                "UPDATE league_entries SET role = ? WHERE id = ?",
                (new_role, entry_id),
            )
            self.log_transition(
                entry_id, old_role, new_role,
                entry.status, entry.status, reason,
            )
            if not self._in_transaction:
                self._conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_entry(self, entry_id: int) -> OpponentEntry | None:
        """Look up a single entry by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM league_entries WHERE id = ?", (entry_id,),
            ).fetchone()
            return OpponentEntry.from_db_row(row) if row else None

    # Keep _get_entry as internal alias (no lock, for use inside locked blocks)
    def _get_entry(self, entry_id: int) -> OpponentEntry | None:
        row = self._conn.execute(
            "SELECT * FROM league_entries WHERE id = ?", (entry_id,),
        ).fetchone()
        return OpponentEntry.from_db_row(row) if row else None

    def list_all_entries(self) -> list[OpponentEntry]:
        """List all entries regardless of status, ordered by created_epoch."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM league_entries ORDER BY created_epoch ASC",
            ).fetchall()
            return [OpponentEntry.from_db_row(r) for r in rows]

    def get_current_epoch(self) -> int:
        """Read the current training epoch from training_state."""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT current_epoch FROM training_state WHERE id = 1"
                ).fetchone()
                return row["current_epoch"] if row else 0
            except Exception:
                return 0

    def _list_entries_unlocked(self) -> list[OpponentEntry]:
        """Internal: list active entries without acquiring the lock."""
        rows = self._conn.execute(
            "SELECT * FROM league_entries WHERE status = ? ORDER BY created_epoch ASC",
            (EntryStatus.ACTIVE,),
        ).fetchall()
        return [OpponentEntry.from_db_row(r) for r in rows]

    def list_entries(self) -> list[OpponentEntry]:
        """List all active entries."""
        with self._lock:
            return self._list_entries_unlocked()

    def list_by_role(self, role: Role) -> list[OpponentEntry]:
        """List active entries filtered by role."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM league_entries WHERE role = ? AND status = ? ORDER BY created_epoch ASC",
                (role, EntryStatus.ACTIVE),
            ).fetchall()
            return [OpponentEntry.from_db_row(r) for r in rows]

    # ------------------------------------------------------------------
    # Transitions and protection
    # ------------------------------------------------------------------

    def log_transition(
        self,
        entry_id: int,
        from_role: Role | str | None,
        to_role: Role | str | None,
        from_status: EntryStatus | str | None,
        to_status: EntryStatus | str | None,
        reason: str,
    ) -> None:
        """Record a role/status transition in the league_transitions table."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO league_transitions
                   (entry_id, from_role, to_role, from_status, to_status, reason)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (entry_id, str(from_role) if from_role else None,
                 str(to_role) if to_role else None,
                 str(from_status) if from_status else None,
                 str(to_status) if to_status else None,
                 reason),
            )
            if not self._in_transaction:
                self._conn.commit()

    def decrement_protection(self, entry_id: int) -> None:
        """Decrement protection_remaining by 1, floor at 0."""
        with self._lock:
            self._conn.execute(
                """UPDATE league_entries
                   SET protection_remaining = MAX(protection_remaining - 1, 0)
                   WHERE id = ?""",
                (entry_id,),
            )
            if not self._in_transaction:
                self._conn.commit()

    # ------------------------------------------------------------------
    # Pins
    # ------------------------------------------------------------------

    def pin(self, entry_id: int) -> None:
        """Pin an entry to prevent eviction."""
        with self._lock:
            self._pinned.add(entry_id)

    def unpin(self, entry_id: int) -> None:
        """Release a pin."""
        with self._lock:
            self._pinned.discard(entry_id)

    # ------------------------------------------------------------------
    # Bootstrap state
    # ------------------------------------------------------------------

    def is_bootstrapped(self) -> bool:
        """Check whether the league has been bootstrapped."""
        with self._lock:
            row = self._conn.execute(
                "SELECT bootstrapped FROM league_meta WHERE id = 1"
            ).fetchone()
            return bool(row and row[0])

    def set_bootstrapped(self) -> None:
        """Mark the league as bootstrapped."""
        with self._lock:
            self._conn.execute(
                "UPDATE league_meta SET bootstrapped = 1 WHERE id = 1"
            )
            if not self._in_transaction:
                self._conn.commit()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_opponent(self, entry: OpponentEntry, device: str = "cpu") -> torch.nn.Module:
        """Load an opponent model from a store entry."""
        ckpt = Path(entry.checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint missing for store entry id={entry.id} "
                f"(epoch {entry.created_epoch}): {ckpt}"
            )
        model = build_model(entry.architecture, entry.model_params)
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        return model

    def load_all_opponents(self, device: str = "cpu") -> dict[int, torch.nn.Module]:
        """Load all active entries. Skips entries with missing/corrupt checkpoints."""
        models: dict[int, torch.nn.Module] = {}
        for entry in self.list_entries():
            try:
                models[entry.id] = self.load_opponent(entry, device=device)
            except (FileNotFoundError, RuntimeError, pickle.UnpicklingError) as e:
                logger.warning(
                    "Skipping store entry id=%d (epoch %d): %s",
                    entry.id, entry.created_epoch, e,
                )
        return models

    # ------------------------------------------------------------------
    # Elo and match results
    # ------------------------------------------------------------------

    def update_elo(self, entry_id: int, new_elo: float, epoch: int = 0) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE league_entries SET elo_rating = ? WHERE id = ?",
                (new_elo, entry_id),
            )
            self._conn.execute(
                "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (?, ?, ?)",
                (entry_id, epoch, new_elo),
            )
            if not self._in_transaction:
                self._conn.commit()

    def record_result(
        self, epoch: int, learner_id: int, opponent_id: int,
        wins: int, losses: int, draws: int,
        elo_delta_a: float = 0.0, elo_delta_b: float = 0.0,
        *, match_context: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO league_results
                   (epoch, learner_id, opponent_id, wins, losses, draws, elo_delta_a, elo_delta_b)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (epoch, learner_id, opponent_id, wins, losses, draws, elo_delta_a, elo_delta_b),
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
            # Update last_match_at for both participants
            self._conn.execute(
                "UPDATE league_entries SET last_match_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (learner_id,),
            )
            self._conn.execute(
                "UPDATE league_entries SET last_match_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (opponent_id,),
            )
            # Decrement protection for both participants (deduplicate if same)
            self.decrement_protection(learner_id)
            if opponent_id != learner_id:
                self.decrement_protection(opponent_id)
            if not self._in_transaction:
                self._conn.commit()

    # ------------------------------------------------------------------
    # Role-specific Elo
    # ------------------------------------------------------------------

    _ROLE_ELO_SQL: dict[EloColumn, str] = {
        EloColumn.FRONTIER: "UPDATE league_entries SET elo_frontier = ? WHERE id = ?",
        EloColumn.DYNAMIC: "UPDATE league_entries SET elo_dynamic = ? WHERE id = ?",
        EloColumn.RECENT: "UPDATE league_entries SET elo_recent = ? WHERE id = ?",
        EloColumn.HISTORICAL: "UPDATE league_entries SET elo_historical = ? WHERE id = ?",
    }

    def update_role_elo(self, entry_id: int, column: EloColumn, new_elo: float) -> None:
        """Update a specific role Elo column for an entry."""
        sql = self._ROLE_ELO_SQL.get(column)
        if sql is None:
            raise ValueError(f"Invalid Elo column: {column!r}")
        with self._lock:
            self._conn.execute(sql, (new_elo, entry_id))
            if not self._in_transaction:
                self._conn.commit()

    # ------------------------------------------------------------------
    # Historical Library
    # ------------------------------------------------------------------

    def upsert_historical_slot(
        self,
        slot_index: int,
        target_epoch: int,
        entry_id: int | None,
        actual_epoch: int | None,
        selection_mode: str,
    ) -> None:
        """Insert or update a historical library slot."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO historical_library
                   (slot_index, target_epoch, entry_id, actual_epoch, selected_at, selection_mode)
                   VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?)
                   ON CONFLICT(slot_index) DO UPDATE SET
                     target_epoch = excluded.target_epoch,
                     entry_id = excluded.entry_id,
                     actual_epoch = excluded.actual_epoch,
                     selected_at = excluded.selected_at,
                     selection_mode = excluded.selection_mode""",
                (slot_index, target_epoch, entry_id, actual_epoch, selection_mode),
            )
            if not self._in_transaction:
                self._conn.commit()

    def get_historical_slots(self) -> list[dict[str, Any]]:
        """Read all historical library slots joined with entry data."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT h.slot_index, h.target_epoch, h.entry_id, h.actual_epoch,
                   h.selected_at, h.selection_mode,
                   e.display_name, e.checkpoint_path, e.elo_rating, e.elo_historical
                   FROM historical_library h
                   LEFT JOIN league_entries e ON h.entry_id = e.id
                   ORDER BY h.slot_index"""
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Gauntlet results
    # ------------------------------------------------------------------

    def record_gauntlet_result(
        self,
        epoch: int,
        entry_id: int,
        historical_slot: int,
        historical_entry_id: int,
        wins: int,
        losses: int,
        draws: int,
        elo_before: float | None = None,
        elo_after: float | None = None,
    ) -> None:
        """Record a gauntlet matchup result."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO gauntlet_results
                   (epoch, entry_id, historical_slot, historical_entry_id,
                    wins, losses, draws, elo_before, elo_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (epoch, entry_id, historical_slot, historical_entry_id,
                 wins, losses, draws, elo_before, elo_after),
            )
            if not self._in_transaction:
                self._conn.commit()
