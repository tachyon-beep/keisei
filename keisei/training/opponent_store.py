"""OpponentStore: tiered pool storage layer with enums, transactions, and lineage."""

from __future__ import annotations

import json
import logging
import pickle
import random
import shutil
import sqlite3
import threading
from collections import OrderedDict
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


# Map opponent role -> column name for per-role game counters (§13.1).
_ROLE_TO_GAMES_COLUMN: dict[str, str] = {
    "frontier_static": "games_vs_frontier",
    "recent_fixed": "games_vs_recent",
    "dynamic": "games_vs_dynamic",
}


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


def _generate_display_name(epoch: int, existing_names: set[str], entry_count: int = 0) -> str:
    """Generate a unique display name for a league entry.

    Args:
        epoch: Training epoch, used to seed the RNG.
        existing_names: Names already in use (across ALL statuses, not just active).
        entry_count: Total number of entries ever created, used to break ties
            when multiple entries are created at the same epoch.
    """
    rng = random.Random(f"name-{epoch}-{entry_count}")
    shuffled = list(LEAGUE_NAMES)
    rng.shuffle(shuffled)
    for name in shuffled:
        if name not in existing_names:
            return name
    # Pool exhausted — use a monotonic suffix that cannot collide
    return f"{shuffled[0]} #{entry_count}"


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
    optimizer_path: str | None = None
    update_count: int = 0
    last_train_at: str | None = None
    retired_at: str | None = None
    training_enabled: bool = True
    games_vs_frontier: int = 0
    games_vs_dynamic: int = 0
    games_vs_recent: int = 0

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> OpponentEntry:
        keys = row.keys()
        raw_facts = row["flavour_facts"] if "flavour_facts" in keys else "[]"
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
            optimizer_path=row["optimizer_path"] if "optimizer_path" in keys else None,
            update_count=row["update_count"] if "update_count" in keys else 0,
            last_train_at=row["last_train_at"] if "last_train_at" in keys else None,
            retired_at=row["retired_at"] if "retired_at" in keys else None,
            training_enabled=bool(row["training_enabled"]) if "training_enabled" in keys else True,
            games_vs_frontier=row["games_vs_frontier"] if "games_vs_frontier" in keys else 0,
            games_vs_dynamic=row["games_vs_dynamic"] if "games_vs_dynamic" in keys else 0,
            games_vs_recent=row["games_vs_recent"] if "games_vs_recent" in keys else 0,
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
        self._model_cache: OrderedDict[tuple[int, str, str], torch.nn.Module] = OrderedDict()
        self._cache_lock = threading.Lock()
        # Thread-local storage: each thread gets its own SQLite connection,
        # transaction depth counter, and pending filesystem ops list.
        # SQLite WAL mode handles concurrent access from multiple connections
        # natively — reads never block writes and vice versa, with busy_timeout
        # retrying brief write conflicts.
        self._local = threading.local()
        # Open a connection eagerly for the creating thread.
        self._conn  # triggers lazy creation via property

    @staticmethod
    def _new_connection(db_path: str) -> sqlite3.Connection:
        """Open a new WAL-mode connection to the league database."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """Return the calling thread's dedicated connection (created on first access)."""
        try:
            return self._local.conn
        except AttributeError:
            self._local.conn = self._new_connection(self.db_path)
            self._local.transaction_depth = 0
            self._local.pending_fs_ops: list[tuple[str, ...]] = []
            self._local.lock = threading.RLock()
            return self._local.conn

    @_conn.setter
    def _conn(self, value: sqlite3.Connection) -> None:
        """Replace the calling thread's connection (used by tests)."""
        self._local.conn = value

    @property
    def _lock(self) -> threading.RLock:
        """Per-thread lock for serializing DB operations within a single thread.

        With thread-local connections, cross-thread contention is handled by
        SQLite's WAL mode.  This lock only prevents interleaving of operations
        within the same thread (e.g., nested transaction() calls from the same
        thread).
        """
        # Trigger connection init if needed (sets up _local.lock)
        self._conn
        return self._local.lock

    @property
    def _transaction_depth(self) -> int:
        self._conn  # ensure init
        return self._local.transaction_depth

    @_transaction_depth.setter
    def _transaction_depth(self, value: int) -> None:
        self._conn  # ensure init
        self._local.transaction_depth = value

    @property
    def _pending_fs_ops(self) -> list[tuple[str, ...]]:
        self._conn  # ensure init
        return self._local.pending_fs_ops

    @_pending_fs_ops.setter
    def _pending_fs_ops(self, value: list[tuple[str, ...]]) -> None:
        self._conn  # ensure init
        self._local.pending_fs_ops = value

    def close(self) -> None:
        """Close the calling thread's database connection."""
        try:
            conn = self._local.conn
        except AttributeError:
            return
        conn.close()
        del self._local.conn

    def __enter__(self) -> OpponentStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Transaction support
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Atomic multi-operation context. Holds lock, defers commit.

        Supports nesting: only the outermost transaction commits/rollbacks.

        Important: do NOT mix raw ``with self._lock:`` blocks with nested
        ``self.transaction()`` calls — the transaction tracks depth via
        _transaction_depth, but raw lock acquisitions don't increment it.
        A ``transaction()`` inside a raw ``with self._lock:`` would see
        depth=1 (outermost) and commit mid-operation.  All current callers
        use either ``_lock`` (for simple reads) or ``transaction()`` (for
        writes), never both.
        """
        with self._lock:
            self._transaction_depth += 1
            is_outermost = self._transaction_depth == 1
            try:
                yield
                if is_outermost:
                    self._conn.commit()
                    self._finalize_fs_ops()
            except Exception:
                if is_outermost:
                    try:
                        self._conn.rollback()
                    except Exception:
                        logger.error("DB rollback failed", exc_info=True)
                    self._rollback_fs_ops()
                raise
            finally:
                self._transaction_depth -= 1

    def _register_pending_file(self, path: Path) -> None:
        """Register a NEW file for deletion if the outermost transaction rolls back."""
        self._pending_fs_ops.append(("delete", str(path)))

    def _register_overwrite(self, path: Path) -> None:
        """Backup an existing file before overwriting so rollback can restore it.

        If the file doesn't exist yet, falls back to delete-on-rollback.
        """
        if path.exists():
            backup = Path(str(path) + ".rollback-bak")
            shutil.copy2(str(path), str(backup))
            self._pending_fs_ops.append(("restore", str(path), str(backup)))
        else:
            self._pending_fs_ops.append(("delete", str(path)))

    def _finalize_fs_ops(self) -> None:
        """On commit: remove any backup files, clear the ops list."""
        for op in self._pending_fs_ops:
            if op[0] == "restore":
                backup = Path(op[2])
                backup.unlink(missing_ok=True)
        self._pending_fs_ops.clear()

    def _rollback_fs_ops(self) -> None:
        """On rollback: delete new files, restore overwritten files from backups."""
        for op in self._pending_fs_ops:
            try:
                if op[0] == "delete":
                    Path(op[1]).unlink(missing_ok=True)
                elif op[0] == "restore":
                    target, backup = Path(op[1]), Path(op[2])
                    if backup.exists():
                        shutil.copy2(str(backup), str(target))
                        backup.unlink(missing_ok=True)
            except OSError:
                logger.warning("Failed to roll back filesystem op: %s", op)
        self._pending_fs_ops.clear()

    # ------------------------------------------------------------------
    # Entry CRUD
    # ------------------------------------------------------------------

    def _entry_dir(self, entry_id: int) -> Path:
        """Per-entry directory: league/entries/{id:06d}/."""
        return self.league_dir / "entries" / f"{entry_id:06d}"

    @staticmethod
    def _write_metadata(entry_dir: Path, metadata: dict[str, Any]) -> None:
        """Write metadata.json sidecar so checkpoints are self-describing."""
        meta_path = entry_dir / "metadata.json"
        tmp = Path(str(meta_path) + ".tmp")
        tmp.write_text(json.dumps(metadata, indent=2))
        tmp.rename(meta_path)

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

        with self.transaction():
            existing_names = self._all_display_names_unlocked()
            entry_count = self._entry_count_unlocked()
            display_name = _generate_display_name(epoch, existing_names, entry_count)
            flavour_facts = _generate_flavour_facts(epoch)

            training_enabled = 1 if role == Role.DYNAMIC else 0
            cursor = self._conn.execute(
                """INSERT INTO league_entries
                   (display_name, flavour_facts, architecture, model_params,
                    checkpoint_path, created_epoch, role, status, training_enabled)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (display_name, json.dumps(flavour_facts), architecture,
                 json.dumps(model_params), "", epoch, role, EntryStatus.ACTIVE,
                 training_enabled),
            )
            entry_id = cursor.lastrowid
            assert entry_id is not None

            entry_dir = self._entry_dir(entry_id)
            entry_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = entry_dir / "weights.pt"
            tmp_path = Path(str(ckpt_path) + ".tmp")
            torch.save(raw_model.state_dict(), tmp_path)
            tmp_path.rename(ckpt_path)
            self._register_pending_file(ckpt_path)

            try:
                self._conn.execute(
                    "UPDATE league_entries SET checkpoint_path = ? WHERE id = ?",
                    (str(ckpt_path), entry_id),
                )

                self._log_transition_unlocked(entry_id, None, role, None, EntryStatus.ACTIVE, "added")
            except Exception:
                ckpt_path.unlink(missing_ok=True)
                raise

            meta_path = entry_dir / "metadata.json"
            self._write_metadata(entry_dir, {
                "entry_id": entry_id,
                "display_name": display_name,
                "architecture": architecture,
                "model_params": model_params,
                "created_epoch": epoch,
                "role": str(role),
            })
            self._register_pending_file(meta_path)

            logger.info(
                "Store entry: %s (%s) epoch %d -> entries/%06d/ (id=%d)",
                display_name, architecture, epoch, entry_id, entry_id,
            )

            entry = self._get_entry(entry_id)
            assert entry is not None
            return entry

    def clone_entry(self, source_entry_id: int, new_role: Role, reason: str) -> OpponentEntry:
        """Clone an entry: copy checkpoint into new per-entry dir, create DB row with lineage.

        Lifecycle invariants (§5.2, §7.1) are the CALLER's responsibility:
        - Dynamic clones: DynamicManager.admit() sets protection_remaining.
        - Optimizer state: clone copies only weights, not optimizer — fresh
          optimizer is created on first DynamicTrainer.update().
        - Frontier clones: FrontierManager.review() handles retirement.
        """
        with self.transaction():
            source = self._get_entry(source_entry_id)
            if source is None:
                raise ValueError(f"Source entry {source_entry_id} not found")
            if source.status != EntryStatus.ACTIVE:
                raise ValueError(
                    f"Source entry {source_entry_id} is not active "
                    f"(status={source.status})"
                )
            src_path = Path(source.checkpoint_path)
            existing_names = self._all_display_names_unlocked()
            entry_count = self._entry_count_unlocked()
            display_name = _generate_display_name(source.created_epoch, existing_names, entry_count)
            flavour_facts = _generate_flavour_facts(source.created_epoch + source_entry_id)
            # Only Dynamic entries are trainable (§6.1, §6.2, §10).
            training_enabled = 1 if new_role == Role.DYNAMIC else 0
            cursor = self._conn.execute(
                """INSERT INTO league_entries
                   (display_name, flavour_facts, architecture, model_params,
                    checkpoint_path, created_epoch, role, status,
                    parent_entry_id, lineage_group, training_enabled)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (display_name, json.dumps(flavour_facts), source.architecture,
                 json.dumps(source.model_params), "", source.created_epoch,
                 new_role, EntryStatus.ACTIVE, source_entry_id,
                 source.lineage_group or f"lineage-{source_entry_id}",
                 training_enabled),
            )
            entry_id = cursor.lastrowid
            assert entry_id is not None
            entry_dir = self._entry_dir(entry_id)
            entry_dir.mkdir(parents=True, exist_ok=True)
            dst_path = entry_dir / "weights.pt"
            shutil.copy2(str(src_path), str(dst_path))
            self._register_pending_file(dst_path)
            try:
                self._conn.execute(
                    "UPDATE league_entries SET checkpoint_path = ? WHERE id = ?",
                    (str(dst_path), entry_id),
                )
                self._log_transition_unlocked(entry_id, None, new_role, None, EntryStatus.ACTIVE, reason)
            except Exception:
                dst_path.unlink(missing_ok=True)
                raise

            meta_path = entry_dir / "metadata.json"
            self._write_metadata(entry_dir, {
                "entry_id": entry_id,
                "display_name": display_name,
                "architecture": source.architecture,
                "model_params": source.model_params,
                "created_epoch": source.created_epoch,
                "role": str(new_role),
                "parent_entry_id": source_entry_id,
                "lineage_group": source.lineage_group or f"lineage-{source_entry_id}",
            })
            self._register_pending_file(meta_path)

            entry = self._get_entry(entry_id)
            assert entry is not None
            return entry

    def retire_entry(self, entry_id: int, reason: str) -> None:
        """Mark an entry as retired. Does NOT delete the checkpoint file."""
        with self.transaction():
            entry = self._get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Entry {entry_id} not found")
            old_role = entry.role
            self._conn.execute(
                "UPDATE league_entries SET status = ?, retired_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (EntryStatus.RETIRED, entry_id),
            )
            self._log_transition_unlocked(
                entry_id, old_role, old_role,
                EntryStatus.ACTIVE, EntryStatus.RETIRED, reason,
            )

    def update_role(self, entry_id: int, new_role: Role, reason: str) -> None:
        """Change the role of an entry."""
        with self.transaction():
            entry = self._get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Entry {entry_id} not found")
            old_role = entry.role
            training_enabled = 1 if new_role == Role.DYNAMIC else 0
            self._conn.execute(
                "UPDATE league_entries SET role = ?, training_enabled = ? WHERE id = ?",
                (new_role, training_enabled, entry_id),
            )
            self._log_transition_unlocked(
                entry_id, old_role, new_role,
                entry.status, entry.status, reason,
            )

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
            except sqlite3.OperationalError:
                # Table may not exist yet (pre-init or schema mismatch)
                return 0

    def _list_entries_unlocked(self) -> list[OpponentEntry]:
        """Internal: list active entries without acquiring the lock."""
        rows = self._conn.execute(
            "SELECT * FROM league_entries WHERE status = ? ORDER BY created_epoch ASC",
            (EntryStatus.ACTIVE,),
        ).fetchall()
        return [OpponentEntry.from_db_row(r) for r in rows]

    def _all_display_names_unlocked(self) -> set[str]:
        """Internal: return display names across ALL statuses for uniqueness checks."""
        rows = self._conn.execute(
            "SELECT display_name FROM league_entries WHERE display_name != ''",
        ).fetchall()
        return {row["display_name"] for row in rows}

    def _entry_count_unlocked(self) -> int:
        """Internal: total number of entries ever created."""
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM league_entries").fetchone()
        return row["cnt"] if row else 0

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

    def _log_transition_unlocked(
        self,
        entry_id: int,
        from_role: Role | str | None,
        to_role: Role | str | None,
        from_status: EntryStatus | str | None,
        to_status: EntryStatus | str | None,
        reason: str,
    ) -> None:
        """Internal: insert transition row without acquiring lock or committing.

        Must be called inside an active transaction() — the INSERT is only
        committed when the outermost transaction scope exits successfully.
        The ``_unlocked`` suffix signals this requirement.
        """
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
        with self.transaction():
            self._log_transition_unlocked(
                entry_id, from_role, to_role, from_status, to_status, reason,
            )

    def count_unique_opponents(self, entry_id: int) -> int:
        """Count distinct opponents this entry has faced (in either seat)."""
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(DISTINCT other_id) AS cnt FROM (
                       SELECT entry_b_id AS other_id FROM league_results WHERE entry_a_id = ?
                       UNION ALL
                       SELECT entry_a_id AS other_id FROM league_results WHERE entry_b_id = ?
                   )""",
                (entry_id, entry_id),
            ).fetchone()
            return row["cnt"] if row else 0

    def set_protection(self, entry_id: int, count: int) -> None:
        """Set the protection_remaining value for an entry."""
        with self.transaction():
            self._conn.execute(
                "UPDATE league_entries SET protection_remaining = ? WHERE id = ?",
                (count, entry_id),
            )

    def _decrement_protection_unlocked(self, entry_id: int) -> None:
        """Internal: decrement protection without acquiring lock or committing."""
        self._conn.execute(
            """UPDATE league_entries
               SET protection_remaining = MAX(protection_remaining - 1, 0)
               WHERE id = ?""",
            (entry_id,),
        )

    def decrement_protection(self, entry_id: int) -> None:
        """Decrement protection_remaining by 1, floor at 0."""
        with self.transaction():
            self._decrement_protection_unlocked(entry_id)

    # ------------------------------------------------------------------
    # Pins
    # ------------------------------------------------------------------

    def pin(self, entry_id: int) -> None:
        """Pin an entry to prevent eviction."""
        with self._cache_lock:
            self._pinned.add(entry_id)

    def unpin(self, entry_id: int) -> None:
        """Release a pin."""
        with self._cache_lock:
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
        """Mark the league as bootstrapped.

        The league_meta row with id=1 is guaranteed to exist — created by
        init_db() via INSERT OR IGNORE.
        """
        with self.transaction():
            self._conn.execute(
                "UPDATE league_meta SET bootstrapped = 1 WHERE id = 1"
            )

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
        # strict=True: entry.architecture + model_params must exactly match the
        # checkpoint's layer structure.  If architecture evolves, entries need
        # migration (re-save with new structure) — silent partial loads via
        # strict=False would produce corrupt inference.
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        return model

    def load_opponent_cached(
        self, entry: OpponentEntry, device: str = "cpu", max_cached: int = 0,
    ) -> torch.nn.Module:
        """Load with LRU caching.  max_cached=0 disables caching."""
        if max_cached <= 0:
            return self.load_opponent(entry, device=device)
        key = (entry.id, entry.checkpoint_path, device)
        with self._cache_lock:
            if key in self._model_cache:
                self._model_cache.move_to_end(key)
                return self._model_cache[key]
        # Load outside lock (disk I/O + GPU transfer can be slow)
        model = self.load_opponent(entry, device=device)
        with self._cache_lock:
            # Re-check after releasing lock — another thread may have loaded same key
            if key in self._model_cache:
                self._model_cache.move_to_end(key)
                return self._model_cache[key]
            self._model_cache[key] = model
            while len(self._model_cache) > max_cached:
                self._model_cache.popitem(last=False)
        return model

    def cache_size(self) -> int:
        """Number of models currently in the LRU cache."""
        with self._cache_lock:
            return len(self._model_cache)

    def clear_model_cache(self) -> None:
        """Drop all cached models."""
        with self._cache_lock:
            self._model_cache.clear()

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
        with self.transaction():
            self._conn.execute(
                "UPDATE league_entries SET elo_rating = ? WHERE id = ?",
                (new_elo, entry_id),
            )
            self._conn.execute(
                "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (?, ?, ?)",
                (entry_id, epoch, new_elo),
            )

    def carry_forward_elo(self, epoch: int) -> None:
        """Write elo_history entries for all active entries at the given epoch.

        Uses a single INSERT...SELECT to atomically read current elo_rating
        values from league_entries, avoiding TOCTOU races with the tournament
        thread that updates elo_rating concurrently.

        This does NOT modify league_entries.elo_rating — the tournament thread
        owns that field.  This method only fills elo_history gaps so the
        chart shows continuous lines.
        """
        with self.transaction():
            self._conn.execute(
                """INSERT INTO elo_history (entry_id, epoch, elo_rating)
                   SELECT id, ?, elo_rating
                   FROM league_entries
                   WHERE status = ?""",
                (epoch, EntryStatus.ACTIVE),
            )

    def elo_spread(self, entry_id: int, window: int = 0) -> float:
        """Return Elo spread (max - min) from elo_history for an entry.

        Args:
            window: When > 0, only the most recent *window* history points
                are considered (ordered by epoch DESC).  When 0, the full
                lifetime spread is returned (legacy behaviour).

        Returns 0.0 if fewer than 2 history points exist (insufficient data
        to assess volatility).
        """
        with self._lock:
            if window > 0:
                row = self._conn.execute(
                    "SELECT MAX(elo_rating) - MIN(elo_rating) AS spread, "
                    "COUNT(*) AS cnt FROM ("
                    "  SELECT elo_rating FROM elo_history "
                    "  WHERE entry_id = ? ORDER BY epoch DESC LIMIT ?"
                    ")",
                    (entry_id, window),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT MAX(elo_rating) - MIN(elo_rating) AS spread, COUNT(*) AS cnt "
                    "FROM elo_history WHERE entry_id = ?",
                    (entry_id,),
                ).fetchone()
            if row is None or row["cnt"] < 2:
                return 0.0
            return float(row["spread"])

    def record_result(
        self, epoch: int, entry_a_id: int, entry_b_id: int,
        wins_a: int, wins_b: int, draws: int,
        *, match_type: str,
        role_a: str | None = None, role_b: str | None = None,
        elo_before_a: float | None = None, elo_after_a: float | None = None,
        elo_before_b: float | None = None, elo_after_b: float | None = None,
        training_updates_a: int | None = None,
        training_updates_b: int | None = None,
    ) -> None:
        num_games = wins_a + wins_b + draws
        with self.transaction():
            self._conn.execute(
                """INSERT INTO league_results
                   (epoch, entry_a_id, entry_b_id, match_type,
                    role_a, role_b, num_games, wins_a, wins_b, draws,
                    elo_before_a, elo_after_a, elo_before_b, elo_after_b,
                    training_updates_a, training_updates_b)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (epoch, entry_a_id, entry_b_id, match_type,
                 role_a, role_b, num_games, wins_a, wins_b, draws,
                 elo_before_a, elo_after_a, elo_before_b, elo_after_b,
                 training_updates_a, training_updates_b),
            )
            self._conn.execute(
                "UPDATE league_entries SET games_played = games_played + ? WHERE id = ?",
                (num_games, entry_a_id),
            )
            self._conn.execute(
                "UPDATE league_entries SET games_played = games_played + ? WHERE id = ?",
                (num_games, entry_b_id),
            )
            # Update per-role game counters: A played against B's role, and vice versa.
            col_for_b = _ROLE_TO_GAMES_COLUMN.get(role_b) if role_b else None
            col_for_a = _ROLE_TO_GAMES_COLUMN.get(role_a) if role_a else None
            if col_for_b:
                self._conn.execute(
                    f"UPDATE league_entries SET {col_for_b} = {col_for_b} + ? WHERE id = ?",
                    (num_games, entry_a_id),
                )
            if col_for_a:
                self._conn.execute(
                    f"UPDATE league_entries SET {col_for_a} = {col_for_a} + ? WHERE id = ?",
                    (num_games, entry_b_id),
                )
            # Update last_match_at for both participants
            self._conn.execute(
                "UPDATE league_entries SET last_match_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (entry_a_id,),
            )
            self._conn.execute(
                "UPDATE league_entries SET last_match_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = ?",
                (entry_b_id,),
            )
            # Decrement protection for both participants (deduplicate if same)
            self._decrement_protection_unlocked(entry_a_id)
            if entry_b_id != entry_a_id:
                self._decrement_protection_unlocked(entry_b_id)

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
        with self.transaction():
            self._conn.execute(sql, (new_elo, entry_id))

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
        with self.transaction():
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

    # ------------------------------------------------------------------
    # Optimizer persistence (Phase 3)
    # ------------------------------------------------------------------

    def save_weights(self, entry_id: int, state_dict: dict[str, Any]) -> None:
        """Save updated model weights for a Dynamic entry.

        Per §14: overwrites weights.pt atomically via .tmp + rename.
        Uses transaction rollback safety (backup before overwrite).
        """
        with self.transaction():
            entry = self._get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Entry {entry_id} not found")
            ckpt_path = Path(entry.checkpoint_path)
            self._register_overwrite(ckpt_path)
            tmp_path = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
            torch.save(state_dict, tmp_path)
            tmp_path.rename(ckpt_path)
            logger.info(
                "Saved weights for entry %d -> %s", entry_id, ckpt_path,
            )

    def save_optimizer(self, entry_id: int, optimizer_state_dict: dict[str, Any]) -> None:
        """Save an optimizer state dict as optimizer.pt in the entry directory.

        Per §14: Dynamic entries get weights.pt + optimizer.pt + metadata.json.
        Writes atomically via .tmp + rename. Updates optimizer_path in the DB.
        """
        with self.transaction():
            entry = self._get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Entry {entry_id} not found")
            entry_dir = self._entry_dir(entry_id)
            entry_dir.mkdir(parents=True, exist_ok=True)
            opt_path = entry_dir / "optimizer.pt"
            self._register_overwrite(opt_path)
            tmp_path = Path(str(opt_path) + ".tmp")
            torch.save(optimizer_state_dict, tmp_path)
            tmp_path.rename(opt_path)
            self._conn.execute(
                "UPDATE league_entries SET optimizer_path = ? WHERE id = ?",
                (str(opt_path), entry_id),
            )
            # Refresh metadata.json so the sidecar is self-describing (§14).
            meta_path = entry_dir / "metadata.json"
            self._register_overwrite(meta_path)
            refreshed = self._get_entry(entry_id)
            if refreshed is not None:
                self._write_metadata(entry_dir, {
                    "entry_id": entry_id,
                    "display_name": refreshed.display_name,
                    "architecture": refreshed.architecture,
                    "model_params": refreshed.model_params,
                    "created_epoch": refreshed.created_epoch,
                    "role": str(refreshed.role),
                    "parent_entry_id": refreshed.parent_entry_id,
                    "lineage_group": refreshed.lineage_group,
                    "optimizer_path": str(opt_path),
                    "update_count": refreshed.update_count,
                    "last_train_at": refreshed.last_train_at,
                })
            logger.info("Saved optimizer for entry %d -> entries/%06d/optimizer.pt", entry_id, entry_id)

    def load_optimizer(self, entry_id: int, device: str = "cpu") -> dict[str, Any] | None:
        """Load a saved optimizer state dict, or None if unavailable.

        Note: the lock is intentionally released before torch.load to avoid
        serializing all store operations during slow I/O. save_optimizer uses
        atomic rename (tmp -> final), so a concurrent save produces either the
        old or new version — both valid states.
        """
        with self._lock:
            entry = self._get_entry(entry_id)
        if entry is None or entry.optimizer_path is None:
            return None
        opt_path = Path(entry.optimizer_path)
        if not opt_path.exists():
            logger.warning(
                "Optimizer file does not exist for entry %d: %s", entry_id, opt_path,
            )
            return None
        # Optimizer state is self-generated, not from untrusted sources.
        # weights_only=False is needed because Adam state contains Python
        # ints and dicts alongside tensors.
        return torch.load(opt_path, map_location=device, weights_only=False)

    def increment_update_count(self, entry_id: int) -> None:
        """Increment the update_count and set last_train_at to now."""
        with self.transaction():
            self._conn.execute(
                "UPDATE league_entries SET update_count = update_count + 1, "
                "last_train_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
                "WHERE id = ?",
                (entry_id,),
            )

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
        with self.transaction():
            self._conn.execute(
                """INSERT INTO gauntlet_results
                   (epoch, entry_id, historical_slot, historical_entry_id,
                    wins, losses, draws, elo_before, elo_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (epoch, entry_id, historical_slot, historical_entry_id,
                 wins, losses, draws, elo_before, elo_after),
            )
