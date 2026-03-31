"""SQLite database layer — schema, migrations, read/write helpers."""

from __future__ import annotations

import sqlite3
from typing import Any

SCHEMA_VERSION = 1


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA wal_autocheckpoint = 1000")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db(db_path: str) -> None:
    """Create tables if they don't exist. Idempotent."""
    conn = _connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS metrics (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            epoch              INTEGER NOT NULL,
            step               INTEGER NOT NULL,
            policy_loss        REAL,
            value_loss         REAL,
            entropy            REAL,
            win_rate           REAL,
            draw_rate          REAL,
            truncation_rate    REAL,
            avg_episode_length REAL,
            gradient_norm      REAL,
            episodes_completed INTEGER,
            timestamp          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_epoch ON metrics(epoch);
        CREATE INDEX IF NOT EXISTS idx_metrics_id ON metrics(id);
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
            updated_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
        CREATE TABLE IF NOT EXISTS training_state (
            id               INTEGER PRIMARY KEY CHECK (id = 1),
            config_json      TEXT NOT NULL,
            display_name     TEXT NOT NULL,
            model_arch       TEXT NOT NULL,
            algorithm_name   TEXT NOT NULL,
            started_at       TEXT NOT NULL,
            current_epoch    INTEGER NOT NULL DEFAULT 0,
            current_step     INTEGER NOT NULL DEFAULT 0,
            checkpoint_path  TEXT,
            status           TEXT NOT NULL DEFAULT 'running',
            heartbeat_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()
    conn.close()


def write_metrics(db_path: str, metrics: dict[str, Any]) -> None:
    conn = _connect(db_path)
    conn.execute(
        """INSERT INTO metrics (epoch, step, policy_loss, value_loss, entropy,
           win_rate, draw_rate, truncation_rate, avg_episode_length,
           gradient_norm, episodes_completed)
           VALUES (:epoch, :step, :policy_loss, :value_loss, :entropy,
           :win_rate, :draw_rate, :truncation_rate, :avg_episode_length,
           :gradient_norm, :episodes_completed)""",
        {
            "epoch": metrics.get("epoch", 0), "step": metrics.get("step", 0),
            "policy_loss": metrics.get("policy_loss"), "value_loss": metrics.get("value_loss"),
            "entropy": metrics.get("entropy"), "win_rate": metrics.get("win_rate"),
            "draw_rate": metrics.get("draw_rate"),
            "truncation_rate": metrics.get("truncation_rate"),
            "avg_episode_length": metrics.get("avg_episode_length"),
            "gradient_norm": metrics.get("gradient_norm"),
            "episodes_completed": metrics.get("episodes_completed"),
        },
    )
    conn.commit()
    conn.close()


def read_metrics_since(db_path: str, since_id: int, limit: int = 500) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM metrics WHERE id > ? ORDER BY id LIMIT ?", (since_id, limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def write_game_snapshots(db_path: str, snapshots: list[dict[str, Any]]) -> None:
    conn = _connect(db_path)
    conn.execute("BEGIN")
    for snap in snapshots:
        conn.execute(
            """INSERT OR REPLACE INTO game_snapshots
               (game_id, board_json, hands_json, current_player, ply,
                is_over, result, sfen, in_check, move_history_json)
               VALUES (:game_id, :board_json, :hands_json, :current_player,
                :ply, :is_over, :result, :sfen, :in_check, :move_history_json)""",
            snap,
        )
    conn.commit()
    conn.close()


def read_game_snapshots(db_path: str) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM game_snapshots ORDER BY game_id").fetchall()
    conn.close()
    return [dict(row) for row in rows]


def write_training_state(db_path: str, state: dict[str, Any]) -> None:
    conn = _connect(db_path)
    conn.execute(
        """INSERT OR REPLACE INTO training_state
           (id, config_json, display_name, model_arch, algorithm_name,
            started_at, current_epoch, current_step, checkpoint_path, status)
           VALUES (1, :config_json, :display_name, :model_arch, :algorithm_name,
            :started_at, :current_epoch, :current_step, :checkpoint_path, :status)""",
        {
            "config_json": state["config_json"], "display_name": state["display_name"],
            "model_arch": state["model_arch"], "algorithm_name": state["algorithm_name"],
            "started_at": state["started_at"],
            "current_epoch": state.get("current_epoch", 0),
            "current_step": state.get("current_step", 0),
            "checkpoint_path": state.get("checkpoint_path"),
            "status": state.get("status", "running"),
        },
    )
    conn.commit()
    conn.close()


def read_training_state(db_path: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM training_state WHERE id = 1").fetchone()
    conn.close()
    return dict(row) if row else None


def update_heartbeat(db_path: str) -> None:
    conn = _connect(db_path)
    conn.execute(
        "UPDATE training_state SET heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
        "WHERE id = 1"
    )
    conn.commit()
    conn.close()


def update_training_progress(
    db_path: str, epoch: int, step: int, checkpoint_path: str | None = None
) -> None:
    conn = _connect(db_path)
    if checkpoint_path is not None:
        conn.execute(
            "UPDATE training_state SET current_epoch = ?, current_step = ?, "
            "checkpoint_path = ?, heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
            "WHERE id = 1",
            (epoch, step, checkpoint_path),
        )
    else:
        conn.execute(
            "UPDATE training_state SET current_epoch = ?, current_step = ?, "
            "heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') WHERE id = 1",
            (epoch, step),
        )
    conn.commit()
    conn.close()
