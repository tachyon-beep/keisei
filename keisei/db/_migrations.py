"""Schema migration registry and runner.

Each function migrates a database from version (target - 1) to target.  All
functions must be idempotent (safe to re-run on an already-migrated DB) and
operate on a connection passed in by the caller.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _migrate_add_column(
    conn: sqlite3.Connection, table: str, column: str, col_type: str,
) -> None:
    """Add a column to a table if it doesn't already exist."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2: Add columns to league_entries and training_state.

    New *tables* (game_features, style_profiles, tournament_stats, etc.) are
    handled by CREATE TABLE IF NOT EXISTS in init_db().  This migration covers
    columns added to tables that already exist in v1.

    Note: SQLite ALTER TABLE ADD COLUMN does not support expression defaults
    like strftime(...). Constant defaults are used; the expression defaults in
    the CREATE TABLE DDL apply to freshly-created v2 databases.
    """
    # league_entries columns added in v2
    _migrate_add_column(conn, "league_entries", "games_played", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "created_at", "TEXT NOT NULL DEFAULT ''")
    _migrate_add_column(conn, "league_entries", "role", "TEXT NOT NULL DEFAULT 'unassigned'")
    _migrate_add_column(conn, "league_entries", "status", "TEXT NOT NULL DEFAULT 'active'")
    _migrate_add_column(conn, "league_entries", "parent_entry_id", "INTEGER")
    _migrate_add_column(conn, "league_entries", "lineage_group", "TEXT")
    _migrate_add_column(conn, "league_entries", "protection_remaining", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "last_match_at", "TEXT")
    _migrate_add_column(conn, "league_entries", "elo_frontier", "REAL NOT NULL DEFAULT 1000.0")
    _migrate_add_column(conn, "league_entries", "elo_dynamic", "REAL NOT NULL DEFAULT 1000.0")
    _migrate_add_column(conn, "league_entries", "elo_recent", "REAL NOT NULL DEFAULT 1000.0")
    _migrate_add_column(conn, "league_entries", "elo_historical", "REAL NOT NULL DEFAULT 1000.0")
    _migrate_add_column(conn, "league_entries", "optimizer_path", "TEXT")
    _migrate_add_column(conn, "league_entries", "update_count", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "last_train_at", "TEXT")
    _migrate_add_column(conn, "league_entries", "retired_at", "TEXT")
    _migrate_add_column(conn, "league_entries", "training_enabled", "INTEGER NOT NULL DEFAULT 1")
    _migrate_add_column(conn, "league_entries", "games_vs_frontier", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "games_vs_dynamic", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "games_vs_recent", "INTEGER NOT NULL DEFAULT 0")
    # training_state column added in v2
    _migrate_add_column(conn, "training_state", "learner_entry_id", "INTEGER")


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """v2 -> v3: Add showcase tables (created by init_db IF NOT EXISTS)."""
    pass


def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
    """v3 -> v4: Tournament sidecar tables + dynamic_update_worker claim column.

    New tables (tournament_pairing_queue, tournament_worker_heartbeat) are
    created by init_db() via CREATE TABLE IF NOT EXISTS.  This migration
    only covers the ALTER TABLE for the new column on league_entries.
    """
    _migrate_add_column(conn, "league_entries", "dynamic_update_worker", "TEXT")


def _migrate_v4_to_v5(conn: sqlite3.Connection) -> None:
    """v4 -> v5: Backfill head_to_head from historical league_results.

    head_to_head is created by init_db() (CREATE TABLE IF NOT EXISTS) and
    maintained incrementally by record_result() going forward, but DBs that
    accumulated league_results before the write-path landed have an empty
    aggregate table.  Without this migration, the dashboard matchup matrix
    only reflects post-deploy games on those DBs.

    Mirrors the SQL in backfill_head_to_head().  Idempotent: clears the
    table and rebuilds from league_results, canonicalising (a,b) ordering.
    Self-play rows (entry_a_id == entry_b_id) are filtered — head_to_head's
    CHECK constraint forbids a == b, and a single such row would otherwise
    abort the entire INSERT...SELECT and leave the table empty.

    Runs inside an explicit BEGIN IMMEDIATE transaction so a mid-INSERT
    failure cannot leave the aggregates table wiped, even if driver-level
    autocommit semantics change in the future.
    """
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM head_to_head")
        conn.execute(
            """INSERT INTO head_to_head (entry_a_id, entry_b_id, wins_a, wins_b, draws, games, last_epoch)
               SELECT
                   CASE WHEN entry_a_id < entry_b_id THEN entry_a_id ELSE entry_b_id END AS a_id,
                   CASE WHEN entry_a_id < entry_b_id THEN entry_b_id ELSE entry_a_id END AS b_id,
                   SUM(CASE WHEN entry_a_id < entry_b_id THEN wins_a ELSE wins_b END) AS w_a,
                   SUM(CASE WHEN entry_a_id < entry_b_id THEN wins_b ELSE wins_a END) AS w_b,
                   SUM(draws) AS d,
                   SUM(wins_a + wins_b + draws) AS g,
                   MAX(epoch) AS last_ep
               FROM league_results
               WHERE entry_a_id != entry_b_id
               GROUP BY a_id, b_id"""
        )
        conn.execute("COMMIT")
    except Exception:
        logger.exception("v4->v5 head_to_head backfill failed; rolling back")
        conn.execute("ROLLBACK")
        raise


def _migrate_v5_to_v6(conn: sqlite3.Connection) -> None:
    """v5 -> v6: Add (status, enqueued_epoch) index on tournament_pairing_queue.

    Without this index, the staleness UPDATE in claim_next_pairings_batch
    (WHERE status = 'pending' AND enqueued_epoch < ?) walks every pending
    row via idx_pairing_queue_pending (status, priority, id), since
    enqueued_epoch is not covered.  With this index, SQLite range-scans only
    stale rows — O(stale) instead of O(pending).  Idempotent via IF NOT EXISTS.
    """
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pairing_queue_staleness "
        "ON tournament_pairing_queue (status, enqueued_epoch)"
    )


def _migrate_v6_to_v7(conn: sqlite3.Connection) -> None:
    """v6 -> v7: Add showcase_moves.move_heatmap_json column.

    Stores a JSON {usi: probability} dict containing legal moves sharing the
    chosen move's from-square (or drop prefix), used by the showcase tab's
    toggleable policy-preference heatmap overlay. Nullable — pre-migration
    rows render no heatmap, which is the correct fallback.
    """
    _migrate_add_column(conn, "showcase_moves", "move_heatmap_json", "TEXT")


def _migrate_v7_to_v8(conn: sqlite3.Connection) -> None:
    """v7 -> v8: Add showcase_moves.move_usi column.

    The existing usi_notation column actually stores Hodges notation (e.g. 'P-9f'),
    not USI (e.g. '9g9f'). This is legacy misnaming; the dashboard MoveLog and
    CommentaryPanel rely on the Hodges format. The new policy heatmap and
    last-move highlight features need real USI to match against
    SpectatorEnv.legal_moves_with_usi() output. Storing it in a sibling column
    avoids breaking existing consumers. Nullable: pre-migration rows render no
    highlight, which is the correct fallback.
    """
    _migrate_add_column(conn, "showcase_moves", "move_usi", "TEXT")


# Migration registry: maps target version to the function that migrates
# from (target - 1) → target.  Each function receives an open connection
# and must be idempotent (safe to re-run on an already-migrated DB).
_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    2: _migrate_v1_to_v2,
    3: _migrate_v2_to_v3,
    4: _migrate_v3_to_v4,
    5: _migrate_v4_to_v5,
    6: _migrate_v5_to_v6,
    7: _migrate_v6_to_v7,
    8: _migrate_v7_to_v8,
}


def apply_migrations(
    conn: sqlite3.Connection, current_version: int, target_version: int
) -> None:
    """Run every migration in the chain (current_version, target_version]."""
    for target in range(current_version + 1, target_version + 1):
        migrate_fn = _MIGRATIONS.get(target)
        if migrate_fn is not None:
            migrate_fn(conn)
