"""Game snapshots table — current state of live demo/training games."""

from __future__ import annotations

from typing import Any

from keisei.db._connection import _connect

DDL = """
CREATE TABLE IF NOT EXISTS game_snapshots (
    game_id           INTEGER PRIMARY KEY,
    board_json        TEXT NOT NULL,
    hands_json        TEXT NOT NULL,
    current_player    TEXT NOT NULL,
    ply               INTEGER NOT NULL,
    is_over           INTEGER NOT NULL,
    result            TEXT NOT NULL,
    sfen              TEXT NOT NULL,
    in_check          INTEGER NOT NULL,
    move_history_json TEXT NOT NULL,
    value_estimate    REAL NOT NULL DEFAULT 0.0,
    game_type         TEXT NOT NULL DEFAULT 'live',
    demo_slot         INTEGER,
    opponent_id       INTEGER REFERENCES league_entries(id),
    updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def write_game_snapshots(db_path: str, snapshots: list[dict[str, Any]]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN")
        for snap in snapshots:
            conn.execute(
                """INSERT OR REPLACE INTO game_snapshots
                   (game_id, board_json, hands_json, current_player, ply,
                    is_over, result, sfen, in_check, move_history_json,
                    value_estimate, game_type, demo_slot, opponent_id, updated_at)
                   VALUES (:game_id, :board_json, :hands_json, :current_player,
                    :ply, :is_over, :result, :sfen, :in_check, :move_history_json,
                    :value_estimate, :game_type, :demo_slot, :opponent_id,
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))""",
                {
                    "game_id": snap["game_id"],
                    "board_json": snap["board_json"],
                    "hands_json": snap["hands_json"],
                    "current_player": snap["current_player"],
                    "ply": snap["ply"],
                    "is_over": snap["is_over"],
                    "result": snap["result"],
                    "sfen": snap["sfen"],
                    "in_check": snap["in_check"],
                    "move_history_json": snap["move_history_json"],
                    "value_estimate": snap.get("value_estimate", 0.0),
                    "game_type": snap.get("game_type", "live"),
                    "demo_slot": snap.get("demo_slot"),
                    "opponent_id": snap.get("opponent_id"),
                },
            )
        conn.commit()
    finally:
        conn.close()


def read_game_snapshots(db_path: str) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM game_snapshots ORDER BY game_id").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def read_game_snapshots_since(
    db_path: str, since_ts: str, since_game_id: int = 0
) -> tuple[list[dict[str, Any]], str, int]:
    """Read game snapshots updated after the composite cursor (since_ts, since_game_id).

    Uses a composite cursor to avoid permanently missing rows when multiple
    game_ids share the same updated_at timestamp.  The cursor is:
      (updated_at > since_ts) OR (updated_at = since_ts AND game_id > since_game_id)

    Returns (rows, max_updated_at, max_game_id_at_that_timestamp).
    """
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM game_snapshots "
            "WHERE updated_at > ? OR (updated_at = ? AND game_id > ?) "
            "ORDER BY updated_at, game_id",
            (since_ts, since_ts, since_game_id),
        ).fetchall()
        max_ts = since_ts
        max_gid = since_game_id
        if rows:
            max_ts = max(dict(r)["updated_at"] for r in rows)
            # Find the highest game_id at the max timestamp for the next cursor
            max_gid = max(
                dict(r)["game_id"] for r in rows if dict(r)["updated_at"] == max_ts
            )
        return [dict(row) for row in rows], max_ts, max_gid
    finally:
        conn.close()
