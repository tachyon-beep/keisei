"""Tournament stats singleton table — last-round runtime metrics for the dashboard."""

from __future__ import annotations

from typing import Any

from keisei.db._connection import _connect

DDL = """
CREATE TABLE IF NOT EXISTS tournament_stats (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    round_duration_s    REAL NOT NULL DEFAULT 0,
    pairings_requested  INTEGER NOT NULL DEFAULT 0,
    pairings_completed  INTEGER NOT NULL DEFAULT 0,
    total_games         INTEGER NOT NULL DEFAULT 0,
    total_plies         INTEGER NOT NULL DEFAULT 0,
    active_slots        INTEGER NOT NULL DEFAULT 0,
    model_load_time_s   REAL NOT NULL DEFAULT 0,
    model_load_count    INTEGER NOT NULL DEFAULT 0,
    games_per_min       REAL NOT NULL DEFAULT 0,
    updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


def write_tournament_stats(db_path: str, stats: object) -> None:
    """Upsert latest tournament round stats (single-row table)."""
    conn = _connect(db_path)
    try:
        duration = getattr(stats, "round_duration_s", 0.0)
        total_games = getattr(stats, "total_games", 0)
        gpm = (total_games / duration * 60) if duration > 0 else 0.0
        conn.execute(
            """INSERT INTO tournament_stats
               (id, round_duration_s, pairings_requested, pairings_completed,
                total_games, total_plies, active_slots,
                model_load_time_s, model_load_count, games_per_min, updated_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
               ON CONFLICT(id) DO UPDATE SET
                 round_duration_s   = excluded.round_duration_s,
                 pairings_requested = excluded.pairings_requested,
                 pairings_completed = excluded.pairings_completed,
                 total_games        = excluded.total_games,
                 total_plies        = excluded.total_plies,
                 active_slots       = excluded.active_slots,
                 model_load_time_s  = excluded.model_load_time_s,
                 model_load_count   = excluded.model_load_count,
                 games_per_min      = excluded.games_per_min,
                 updated_at         = excluded.updated_at""",
            (duration, getattr(stats, "pairings_requested", 0),
             getattr(stats, "pairings_completed", 0), total_games,
             getattr(stats, "total_plies", 0), getattr(stats, "active_slots", 0),
             getattr(stats, "model_load_time_s", 0.0),
             getattr(stats, "model_load_count", 0), gpm),
        )
        conn.commit()
    finally:
        conn.close()


def read_tournament_stats(db_path: str) -> dict[str, Any] | None:
    """Read latest tournament round stats. Returns None if no data yet."""
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM tournament_stats WHERE id = 1").fetchone()
        return dict(row) if row else None
    except Exception:
        return None
    finally:
        conn.close()
