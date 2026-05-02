# keisei/showcase/db_ops.py
"""Database operations for the showcase sidecar and server."""
from __future__ import annotations

import random
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from keisei.db import _connect

MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.1


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _retry_write(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
    """Execute a write with retry on SQLITE_BUSY."""
    for attempt in range(MAX_RETRIES):
        try:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.05)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")  # pragma: no cover


def queue_match(db_path: str, entry_id_1: str, entry_id_2: str, speed: str) -> int:
    conn = _connect(db_path)
    try:
        cursor = _retry_write(conn,
            "INSERT INTO showcase_queue (entry_id_1, entry_id_2, speed, status, requested_at) VALUES (?, ?, ?, 'pending', ?)",
            (entry_id_1, entry_id_2, speed, _now_iso()))
        assert cursor.lastrowid is not None
        return cursor.lastrowid
    finally:
        conn.close()


def claim_next_match(db_path: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        now = _now_iso()
        row = conn.execute(
            """UPDATE showcase_queue SET status = 'running', started_at = ?
               WHERE id = (SELECT id FROM showcase_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1)
               RETURNING id, entry_id_1, entry_id_2, speed, status, requested_at, started_at""",
            (now,)).fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        conn.close()


def read_queue(db_path: str) -> list[dict[str, Any]]:
    """Read active queue entries (pending + running). Cancelled/completed are excluded."""
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM showcase_queue WHERE status IN ('pending', 'running') ORDER BY id").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def cancel_match(db_path: str, queue_id: int) -> None:
    conn = _connect(db_path)
    try:
        _retry_write(conn, "UPDATE showcase_queue SET status = 'cancelled', completed_at = ? WHERE id = ? AND status = 'pending'", (_now_iso(), queue_id))
    finally:
        conn.close()


def update_queue_speed(db_path: str, queue_id: int, speed: str) -> None:
    conn = _connect(db_path)
    try:
        _retry_write(conn, "UPDATE showcase_queue SET speed = ? WHERE id = ?", (speed, queue_id))
    finally:
        conn.close()


def create_showcase_game(db_path: str, *, queue_id: int, entry_id_black: str, entry_id_white: str,
                          elo_black: float, elo_white: float, name_black: str, name_white: str) -> int:
    conn = _connect(db_path)
    try:
        cursor = _retry_write(conn,
            """INSERT INTO showcase_games (queue_id, entry_id_black, entry_id_white, elo_black, elo_white,
               name_black, name_white, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'in_progress', ?)""",
            (queue_id, entry_id_black, entry_id_white, elo_black, elo_white, name_black, name_white, _now_iso()))
        assert cursor.lastrowid is not None
        return cursor.lastrowid
    finally:
        conn.close()


def complete_queue_entry(db_path: str, queue_id: int) -> None:
    """Mark a running queue entry as completed. Uses retry logic."""
    conn = _connect(db_path)
    try:
        _retry_write(conn,
            "UPDATE showcase_queue SET status = 'completed', completed_at = ? WHERE id = ? AND status = 'running'",
            (_now_iso(), queue_id))
    finally:
        conn.close()


def read_active_showcase_game(db_path: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM showcase_games WHERE status = 'in_progress' ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def write_showcase_move(db_path: str, *, game_id: int, ply: int, action_index: int,
                         usi_notation: str, board_json: str, hands_json: str,
                         current_player: str, in_check: bool, value_estimate: float,
                         top_candidates: str, move_time_ms: int,
                         move_heatmap_json: str | None = None) -> None:
    """Atomic write: INSERT move + UPDATE total_ply in one transaction."""
    conn = _connect(db_path)
    try:
        now = _now_iso()
        for attempt in range(MAX_RETRIES):
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """INSERT OR IGNORE INTO showcase_moves
                       (game_id, ply, action_index, usi_notation, board_json, hands_json,
                        current_player, in_check, value_estimate, top_candidates,
                        move_heatmap_json, move_time_ms, created_at)
                       VALUES (:game_id, :ply, :action_index, :usi_notation, :board_json, :hands_json,
                               :current_player, :in_check, :value_estimate, :top_candidates,
                               :move_heatmap_json, :move_time_ms, :created_at)""",
                    {
                        "game_id": game_id, "ply": ply, "action_index": action_index,
                        "usi_notation": usi_notation, "board_json": board_json,
                        "hands_json": hands_json, "current_player": current_player,
                        "in_check": int(in_check), "value_estimate": value_estimate,
                        "top_candidates": top_candidates, "move_heatmap_json": move_heatmap_json,
                        "move_time_ms": move_time_ms, "created_at": now,
                    })
                conn.execute("UPDATE showcase_games SET total_ply = ? WHERE id = ?", (ply, game_id))
                conn.commit()
                return
            except sqlite3.OperationalError as e:
                conn.rollback()
                if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.05)
                    time.sleep(delay)
                else:
                    raise
    finally:
        conn.close()


def read_showcase_moves_since(db_path: str, game_id: int, since_ply: int) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM showcase_moves WHERE game_id = ? AND ply > ? ORDER BY ply", (game_id, since_ply)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def read_all_showcase_moves(db_path: str, game_id: int) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM showcase_moves WHERE game_id = ? ORDER BY ply", (game_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_game_completed(db_path: str, game_id: int, result: str, total_ply: int) -> None:
    conn = _connect(db_path)
    try:
        _retry_write(conn, "UPDATE showcase_games SET status = ?, completed_at = ?, total_ply = ? WHERE id = ?",
                      (result, _now_iso(), total_ply, game_id))
    finally:
        conn.close()


def mark_game_abandoned(db_path: str, game_id: int, reason: str) -> None:
    conn = _connect(db_path)
    try:
        _retry_write(conn, "UPDATE showcase_games SET status = 'abandoned', abandon_reason = ?, completed_at = ? WHERE id = ?",
                      (reason, _now_iso(), game_id))
    finally:
        conn.close()


def write_heartbeat(db_path: str, pid: int) -> None:
    conn = _connect(db_path)
    try:
        _retry_write(conn,
            """INSERT INTO showcase_heartbeat (id, last_heartbeat, runner_pid)
               VALUES (1, ?, ?) ON CONFLICT(id) DO UPDATE SET last_heartbeat = excluded.last_heartbeat, runner_pid = excluded.runner_pid""",
            (_now_iso(), pid))
    finally:
        conn.close()


def read_heartbeat(db_path: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM showcase_heartbeat WHERE id = 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def cleanup_orphaned_games(db_path: str, stale_after_s: float = 60.0) -> int:
    """Mark orphaned in-progress games as abandoned. Checks heartbeat age first."""
    conn = _connect(db_path)
    try:
        now = _now_iso()
        hb = conn.execute("SELECT last_heartbeat FROM showcase_heartbeat WHERE id = 1").fetchone()
        if hb:
            try:
                last_hb = datetime.fromisoformat(hb["last_heartbeat"].replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - last_hb).total_seconds()
                if age < stale_after_s:
                    return 0
            except (ValueError, TypeError):
                pass
        cursor = conn.execute(
            "UPDATE showcase_games SET status = 'abandoned', abandon_reason = 'crash_recovery', completed_at = ? WHERE status = 'in_progress'",
            (now,))
        count = cursor.rowcount
        conn.execute("UPDATE showcase_queue SET status = 'cancelled', completed_at = ? WHERE status = 'running'", (now,))
        conn.commit()
        return count
    finally:
        conn.close()
