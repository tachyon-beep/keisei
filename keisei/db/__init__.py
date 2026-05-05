"""SQLite database layer — schema, migrations, read/write helpers.

This is a per-entity package: each table family lives in its own submodule
(``metrics``, ``snapshots``, ``training_state``, ``league``, ``head_to_head``,
``historical``, ``gauntlet``, ``tournament``, ``game_features``, ``style_profiles``,
``showcase``, ``tournament_queue``).  Each submodule contributes a ``DDL`` constant
and the read/write helpers for its tables.  ``init_db`` concatenates the DDL
blocks in their original order and runs them as a single ``executescript``;
the migration registry stays in :mod:`keisei.db._migrations`.

The public read/write API is re-exported from this package, so every existing
``from keisei.db import …`` site keeps working.
"""

from __future__ import annotations

from keisei.db import (
    _migrations,
    game_features,
    gauntlet,
    head_to_head,
    historical,
    league,
    metrics,
    showcase,
    snapshots,
    style_profiles,
    tournament,
    tournament_queue,
    training_state,
)
from keisei.db._connection import _connect
from keisei.db.game_features import (
    read_all_game_features,
    read_game_features_for_checkpoint,
    write_game_features,
)
from keisei.db.head_to_head import backfill_head_to_head, read_head_to_head
from keisei.db.league import read_elo_history, read_league_data
from keisei.db.metrics import read_metrics_since, write_metrics
from keisei.db.snapshots import (
    read_game_snapshots,
    read_game_snapshots_since,
    write_game_snapshots,
)
from keisei.db.style_profiles import read_style_profiles, write_style_profile
from keisei.db.tournament import read_tournament_stats, write_tournament_stats
from keisei.db.training_state import (
    read_training_state,
    update_heartbeat,
    update_training_progress,
    write_epoch_summary,
    write_training_state,
)

SCHEMA_VERSION = 8

# DDL execution order is preserved from the pre-split ``init_db`` so the
# resulting schema is byte-equivalent.
_SCHEMA_VERSION_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);\n"
)
_DDL_MODULES = (
    metrics,
    snapshots,
    training_state,
    league,
    head_to_head,
    historical,
    gauntlet,
    tournament,
    game_features,
    showcase,
    style_profiles,
    tournament_queue,
)


def init_db(db_path: str) -> None:
    """Create tables if they don't exist. Idempotent."""
    conn = _connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA wal_autocheckpoint = 1000")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.executescript(
            _SCHEMA_VERSION_DDL + "".join(m.DDL for m in _DDL_MODULES)
        )
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            # No version row: either a truly fresh DB or a pre-versioning DB.
            # Treat as version 0 so all migrations run (they're idempotent).
            db_version = 0
        else:
            db_version = row[0]

        if db_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {db_version} is newer than "
                f"expected version {SCHEMA_VERSION}. "
                f"Upgrade the application or delete the database."
            )
        if db_version < SCHEMA_VERSION:
            _migrations.apply_migrations(conn, db_version, SCHEMA_VERSION)
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,)
                )
            else:
                conn.execute(
                    "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
                )
        conn.commit()
    finally:
        conn.close()


def wal_checkpoint(db_path: str) -> tuple[int, int, int]:
    """Force a WAL checkpoint to prevent unbounded WAL growth.

    Uses TRUNCATE mode: merges the WAL back into the main DB and resets the
    WAL file to zero length.  This is safe to call while other connections
    are reading — SQLite will checkpoint as many pages as possible and report
    the remainder.

    Returns (busy, log_pages, checkpointed_pages).
    """
    conn = _connect(db_path)
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return (row[0], row[1], row[2]) if row else (0, 0, 0)
    finally:
        conn.close()


__all__ = [
    "SCHEMA_VERSION",
    "_connect",
    "init_db",
    "wal_checkpoint",
    "write_metrics",
    "read_metrics_since",
    "write_game_snapshots",
    "read_game_snapshots",
    "read_game_snapshots_since",
    "write_training_state",
    "read_training_state",
    "update_heartbeat",
    "update_training_progress",
    "write_epoch_summary",
    "read_league_data",
    "read_elo_history",
    "read_head_to_head",
    "backfill_head_to_head",
    "write_tournament_stats",
    "read_tournament_stats",
    "write_game_features",
    "read_game_features_for_checkpoint",
    "read_all_game_features",
    "write_style_profile",
    "read_style_profiles",
]
