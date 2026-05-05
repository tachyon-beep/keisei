"""League tables — entries, results, elo history, transitions, meta singleton."""

from __future__ import annotations

import json
from typing import Any

from keisei.db._connection import _connect

DDL = """
CREATE TABLE IF NOT EXISTS league_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name    TEXT NOT NULL DEFAULT '',
    flavour_facts   TEXT NOT NULL DEFAULT '[]',
    architecture    TEXT NOT NULL,
    model_params    TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    elo_rating      REAL NOT NULL DEFAULT 1000.0,
    created_epoch   INTEGER NOT NULL,
    games_played    INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    role            TEXT NOT NULL DEFAULT 'unassigned',
    status          TEXT NOT NULL DEFAULT 'active',
    parent_entry_id INTEGER REFERENCES league_entries(id),
    lineage_group   TEXT,
    protection_remaining INTEGER NOT NULL DEFAULT 0,
    last_match_at   TEXT,
    elo_frontier    REAL NOT NULL DEFAULT 1000.0,
    elo_dynamic     REAL NOT NULL DEFAULT 1000.0,
    elo_recent      REAL NOT NULL DEFAULT 1000.0,
    elo_historical  REAL NOT NULL DEFAULT 1000.0,
    optimizer_path  TEXT,
    update_count    INTEGER NOT NULL DEFAULT 0,
    last_train_at   TEXT,
    retired_at      TEXT,
    training_enabled INTEGER NOT NULL DEFAULT 1,
    games_vs_frontier INTEGER NOT NULL DEFAULT 0,
    games_vs_dynamic  INTEGER NOT NULL DEFAULT 0,
    games_vs_recent   INTEGER NOT NULL DEFAULT 0,
    dynamic_update_worker TEXT
);
CREATE TABLE IF NOT EXISTS league_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch               INTEGER NOT NULL,
    entry_a_id          INTEGER NOT NULL REFERENCES league_entries(id),
    entry_b_id          INTEGER NOT NULL REFERENCES league_entries(id),
    match_type          TEXT NOT NULL,
    role_a              TEXT,
    role_b              TEXT,
    num_games           INTEGER NOT NULL,
    wins_a              INTEGER NOT NULL,
    wins_b              INTEGER NOT NULL,
    draws               INTEGER NOT NULL,
    elo_before_a        REAL,
    elo_after_a         REAL,
    elo_before_b        REAL,
    elo_after_b         REAL,
    training_updates_a  INTEGER,
    training_updates_b  INTEGER,
    recorded_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_league_results_epoch ON league_results(epoch);
CREATE INDEX IF NOT EXISTS idx_league_entries_elo ON league_entries(elo_rating);
CREATE INDEX IF NOT EXISTS idx_league_results_entry_a ON league_results(entry_a_id);
CREATE INDEX IF NOT EXISTS idx_league_results_entry_b ON league_results(entry_b_id);
CREATE TABLE IF NOT EXISTS elo_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    INTEGER NOT NULL REFERENCES league_entries(id),
    epoch       INTEGER NOT NULL,
    elo_rating  REAL NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_elo_history_entry ON elo_history(entry_id);
CREATE INDEX IF NOT EXISTS idx_elo_history_entry_epoch ON elo_history(entry_id, epoch);
CREATE TABLE IF NOT EXISTS league_transitions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    INTEGER NOT NULL REFERENCES league_entries(id),
    from_role   TEXT,
    to_role     TEXT,
    from_status TEXT,
    to_status   TEXT,
    reason      TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_transitions_entry ON league_transitions(entry_id);
CREATE TABLE IF NOT EXISTS league_meta (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    bootstrapped INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO league_meta (id, bootstrapped) VALUES (1, 0);
"""


def read_league_data(
    db_path: str, max_results: int = 500
) -> dict[str, list[dict[str, Any]]]:
    """Read league entries, recent results, historical library, and gauntlet results."""
    conn = _connect(db_path)
    try:
        entries = conn.execute(
            "SELECT id, display_name, flavour_facts, model_params, architecture, "
            "elo_rating, games_played, created_epoch, created_at, "
            "role, status, parent_entry_id, lineage_group, protection_remaining, last_match_at, "
            "elo_frontier, elo_dynamic, elo_recent, elo_historical, "
            "optimizer_path, update_count, last_train_at, "
            "games_vs_frontier, games_vs_dynamic, games_vs_recent "
            "FROM league_entries ORDER BY elo_rating DESC"
        ).fetchall()
        results = conn.execute(
            "SELECT id, epoch, entry_a_id, entry_b_id, match_type, role_a, role_b, "
            "num_games, wins_a, wins_b, draws, "
            "elo_before_a, elo_after_a, elo_before_b, elo_after_b, "
            "training_updates_a, training_updates_b, recorded_at "
            "FROM league_results ORDER BY id DESC LIMIT ?",
            (max_results,),
        ).fetchall()
        parsed_entries = []
        for r in entries:
            e = dict(r)
            if isinstance(e.get("flavour_facts"), str):
                e["flavour_facts"] = json.loads(e["flavour_facts"])
            if isinstance(e.get("model_params"), str):
                e["model_params"] = json.loads(e["model_params"])
            parsed_entries.append(e)

        historical_slots = [dict(r) for r in conn.execute(
            "SELECT h.slot_index, h.target_epoch, h.entry_id, h.actual_epoch, "
            "h.selected_at, h.selection_mode, e.display_name AS entry_name, "
            "e.elo_rating AS entry_elo "
            "FROM historical_library h "
            "LEFT JOIN league_entries e ON h.entry_id = e.id "
            "ORDER BY h.slot_index"
        ).fetchall()]

        # Recent gauntlet results (last 50 distinct epochs)
        gauntlet_results = [dict(r) for r in conn.execute(
            "SELECT g.id, g.epoch, g.entry_id, g.historical_slot, "
            "g.historical_entry_id, g.wins, g.losses, g.draws, "
            "g.elo_before, g.elo_after, g.created_at "
            "FROM gauntlet_results g "
            "WHERE g.epoch >= ("
            "  SELECT COALESCE(MIN(epoch), 0) FROM ("
            "    SELECT DISTINCT epoch FROM gauntlet_results ORDER BY epoch DESC LIMIT 50"
            "  )"
            ") "
            "ORDER BY g.epoch DESC, g.historical_slot"
        ).fetchall()]

        # Recent transitions (last 200) for dashboard admission/eviction/promotion display
        transitions = [dict(r) for r in conn.execute(
            "SELECT id, entry_id, from_role, to_role, from_status, to_status, "
            "reason, created_at "
            "FROM league_transitions ORDER BY id DESC LIMIT 200"
        ).fetchall()]

        return {
            "entries": parsed_entries,
            "results": [dict(r) for r in results],
            "historical_library": historical_slots,
            "gauntlet_results": gauntlet_results,
            "transitions": transitions,
        }
    finally:
        conn.close()


def read_elo_history(db_path: str, *, max_epochs: int = 0) -> list[dict[str, Any]]:
    """Read Elo history points for charting.

    Args:
        max_epochs: When > 0, return only the most recent *max_epochs* worth of
            data (by epoch number). 0 returns the full history.
    """
    conn = _connect(db_path)
    try:
        if max_epochs > 0:
            rows = conn.execute(
                """SELECT entry_id, epoch, elo_rating FROM elo_history
                   WHERE epoch >= (SELECT MAX(epoch) - ? FROM elo_history)
                   ORDER BY epoch, entry_id""",
                (max_epochs,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT entry_id, epoch, elo_rating FROM elo_history ORDER BY epoch, entry_id"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
