"""Head-to-head aggregates table — canonical (a < b) pairwise win/loss/draw totals."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from keisei.db._connection import _connect

logger = logging.getLogger(__name__)

DDL = """
-- Standing head-to-head aggregates: maintained incrementally by record_result()
-- Canonical ordering: entry_a_id < entry_b_id to avoid duplicate pairs
CREATE TABLE IF NOT EXISTS head_to_head (
    entry_a_id    INTEGER NOT NULL REFERENCES league_entries(id),
    entry_b_id    INTEGER NOT NULL REFERENCES league_entries(id),
    wins_a        INTEGER NOT NULL DEFAULT 0,
    wins_b        INTEGER NOT NULL DEFAULT 0,
    draws         INTEGER NOT NULL DEFAULT 0,
    games         INTEGER NOT NULL DEFAULT 0,
    last_epoch    INTEGER NOT NULL DEFAULT 0,
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    PRIMARY KEY (entry_a_id, entry_b_id),
    CHECK (entry_a_id < entry_b_id)
);
CREATE INDEX IF NOT EXISTS idx_h2h_entry_a ON head_to_head(entry_a_id);
CREATE INDEX IF NOT EXISTS idx_h2h_entry_b ON head_to_head(entry_b_id);
"""


def read_head_to_head(db_path: str) -> list[dict[str, Any]]:
    """Read all head-to-head aggregates for frontend consumption.

    Returns a list of dicts with entry_a_id, entry_b_id, wins_a, wins_b, draws,
    games, last_epoch. The canonical ordering (entry_a_id < entry_b_id) is
    preserved; the frontend can derive bidirectional stats by mirroring.
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT entry_a_id, entry_b_id, wins_a, wins_b, draws, games, last_epoch
               FROM head_to_head
               ORDER BY games DESC, last_epoch DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc).lower():
            logger.warning("head_to_head table missing in %s; returning empty result", db_path)
            return []
        raise
    finally:
        conn.close()


_BACKFILL_INSERT_SQL = """
INSERT INTO head_to_head (entry_a_id, entry_b_id, wins_a, wins_b, draws, games, last_epoch)
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
GROUP BY a_id, b_id
"""


def apply_backfill(conn: sqlite3.Connection) -> int:
    """Execute the head_to_head backfill on an open connection.

    Wraps DELETE + INSERT...SELECT in BEGIN IMMEDIATE / COMMIT so a
    mid-INSERT failure cannot leave the aggregates table wiped.  Self-play
    rows (entry_a_id == entry_b_id) are filtered: head_to_head's CHECK
    constraint forbids a == b, and a single such row would otherwise abort
    the entire INSERT and leave the table empty.

    Shared between the public ``backfill_head_to_head`` (db_path API) and
    the v4→v5 migration in :mod:`keisei.db._migrations` (connection API).
    Returns the number of pairs inserted.
    """
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM head_to_head")
        cursor = conn.execute(_BACKFILL_INSERT_SQL)
        count = cursor.rowcount
        conn.execute("COMMIT")
        return count
    except Exception:
        logger.exception("head_to_head backfill failed; rolling back")
        conn.execute("ROLLBACK")
        raise


def backfill_head_to_head(db_path: str) -> int:
    """Rebuild head_to_head table from league_results.

    Used for initial migration or repair. Returns the number of pairs inserted.
    """
    conn = _connect(db_path)
    try:
        return apply_backfill(conn)
    finally:
        conn.close()
