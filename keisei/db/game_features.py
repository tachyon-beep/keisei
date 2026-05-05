"""Game features table — per-game feature rows for style profiling."""

from __future__ import annotations

from typing import Any

from keisei.db._connection import _connect

DDL = """
CREATE TABLE IF NOT EXISTS game_features (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id       INTEGER NOT NULL REFERENCES league_entries(id),
    opponent_id         INTEGER NOT NULL REFERENCES league_entries(id),
    epoch               INTEGER NOT NULL,
    side                TEXT NOT NULL,
    result              TEXT NOT NULL,
    total_plies         INTEGER NOT NULL,
    -- §8.1 Opening features
    first_action        INTEGER,
    opening_seq_3       TEXT,
    opening_seq_6       TEXT,
    rook_moved_ply      INTEGER,
    king_displacement_20 INTEGER NOT NULL DEFAULT 0,
    -- §8.2 Tempo and aggression
    first_capture_ply   INTEGER,
    first_check_ply     INTEGER,  -- placeholder: populated when Rust engine exposes check state
    first_drop_ply      INTEGER,
    num_checks          INTEGER NOT NULL DEFAULT 0,  -- placeholder: see first_check_ply
    num_captures        INTEGER NOT NULL DEFAULT 0,
    -- §8.3 Drop and promotion behaviour
    num_drops           INTEGER NOT NULL DEFAULT 0,
    num_promotions      INTEGER NOT NULL DEFAULT 0,
    num_early_drops     INTEGER NOT NULL DEFAULT 0,
    -- §8.4 Positional style proxies
    rook_moves_in_20    INTEGER NOT NULL DEFAULT 0,
    king_moves_in_30    INTEGER NOT NULL DEFAULT 0,
    num_repetitions     INTEGER NOT NULL DEFAULT 0,
    -- §8.5 Termination
    termination_reason  INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_game_features_checkpoint ON game_features(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_game_features_opponent ON game_features(opponent_id);
CREATE INDEX IF NOT EXISTS idx_game_features_epoch ON game_features(epoch);
"""


def write_game_features(db_path: str, features: list[dict[str, Any]]) -> None:
    """Insert per-game feature rows (append-only)."""
    if not features:
        return
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN")
        for f in features:
            conn.execute(
                """INSERT INTO game_features
                   (checkpoint_id, opponent_id, epoch, side, result, total_plies,
                    first_action, opening_seq_3, opening_seq_6,
                    rook_moved_ply, king_displacement_20,
                    first_capture_ply, first_drop_ply,
                    num_captures,
                    num_drops, num_promotions, num_early_drops,
                    rook_moves_in_20, king_moves_in_30, num_repetitions,
                    termination_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f["checkpoint_id"], f["opponent_id"], f["epoch"],
                    f["side"], f["result"], f["total_plies"],
                    f.get("first_action"), f.get("opening_seq_3"), f.get("opening_seq_6"),
                    f.get("rook_moved_ply"), f.get("king_displacement_20", 0),
                    f.get("first_capture_ply"), f.get("first_drop_ply"),
                    f.get("num_captures", 0),
                    f.get("num_drops", 0), f.get("num_promotions", 0),
                    f.get("num_early_drops", 0),
                    f.get("rook_moves_in_20", 0), f.get("king_moves_in_30", 0),
                    f.get("num_repetitions", 0),
                    f.get("termination_reason", 0),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def read_game_features_for_checkpoint(
    db_path: str, checkpoint_id: int
) -> list[dict[str, Any]]:
    """Read all game feature rows for a given checkpoint."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM game_features WHERE checkpoint_id = ? ORDER BY id",
            (checkpoint_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def read_all_game_features(
    db_path: str, *, min_epoch: int | None = None
) -> list[dict[str, Any]]:
    """Read game feature rows for league-wide aggregation.

    Args:
        min_epoch: If set, only return rows with epoch >= this value.
    """
    conn = _connect(db_path)
    try:
        if min_epoch is not None:
            rows = conn.execute(
                "SELECT * FROM game_features WHERE epoch >= ? ORDER BY id",
                (min_epoch,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM game_features ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
