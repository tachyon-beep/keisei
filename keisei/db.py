"""SQLite database layer — schema, migrations, read/write helpers."""

from __future__ import annotations

import sqlite3
from typing import Any

SCHEMA_VERSION = 2


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """Create tables if they don't exist. Idempotent."""
    conn = _connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA wal_autocheckpoint = 1000")
        conn.execute("PRAGMA busy_timeout = 5000")
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
                black_win_rate     REAL,
                white_win_rate     REAL,
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
                value_estimate    REAL NOT NULL DEFAULT 0.0,
                game_type         TEXT NOT NULL DEFAULT 'live',
                demo_slot         INTEGER,
                opponent_id       INTEGER REFERENCES league_entries(id),
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
            CREATE TABLE IF NOT EXISTS league_entries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                architecture    TEXT NOT NULL,
                model_params    TEXT NOT NULL,
                checkpoint_path TEXT NOT NULL,
                elo_rating      REAL NOT NULL DEFAULT 1000.0,
                created_epoch   INTEGER NOT NULL,
                games_played    INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE TABLE IF NOT EXISTS league_results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch           INTEGER NOT NULL,
                learner_id      INTEGER NOT NULL REFERENCES league_entries(id),
                opponent_id     INTEGER NOT NULL REFERENCES league_entries(id),
                wins            INTEGER NOT NULL,
                losses          INTEGER NOT NULL,
                draws           INTEGER NOT NULL,
                recorded_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE INDEX IF NOT EXISTS idx_league_results_epoch ON league_results(epoch);
            CREATE INDEX IF NOT EXISTS idx_league_entries_elo ON league_entries(elo_rating);
            CREATE TABLE IF NOT EXISTS elo_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id    INTEGER NOT NULL REFERENCES league_entries(id),
                epoch       INTEGER NOT NULL,
                elo_rating  REAL NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE INDEX IF NOT EXISTS idx_elo_history_entry ON elo_history(entry_id);
        """)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()


def write_metrics(db_path: str, metrics: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT INTO metrics (epoch, step, policy_loss, value_loss, entropy,
               win_rate, black_win_rate, white_win_rate, draw_rate,
               truncation_rate, avg_episode_length,
               gradient_norm, episodes_completed)
               VALUES (:epoch, :step, :policy_loss, :value_loss, :entropy,
               :win_rate, :black_win_rate, :white_win_rate, :draw_rate,
               :truncation_rate, :avg_episode_length,
               :gradient_norm, :episodes_completed)""",
            {
                "epoch": metrics.get("epoch", 0), "step": metrics.get("step", 0),
                "policy_loss": metrics.get("policy_loss"), "value_loss": metrics.get("value_loss"),
                "entropy": metrics.get("entropy"), "win_rate": metrics.get("win_rate"),
                "black_win_rate": metrics.get("black_win_rate"),
                "white_win_rate": metrics.get("white_win_rate"),
                "draw_rate": metrics.get("draw_rate"),
                "truncation_rate": metrics.get("truncation_rate"),
                "avg_episode_length": metrics.get("avg_episode_length"),
                "gradient_norm": metrics.get("gradient_norm"),
                "episodes_completed": metrics.get("episodes_completed"),
            },
        )
        conn.commit()
    finally:
        conn.close()


def read_metrics_since(db_path: str, since_id: int, limit: int = 500) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM metrics WHERE id > ? ORDER BY id LIMIT ?", (since_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def write_game_snapshots(db_path: str, snapshots: list[dict[str, Any]]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN")
        for snap in snapshots:
            conn.execute(
                """INSERT OR REPLACE INTO game_snapshots
                   (game_id, board_json, hands_json, current_player, ply,
                    is_over, result, sfen, in_check, move_history_json,
                    value_estimate, game_type, demo_slot, opponent_id)
                   VALUES (:game_id, :board_json, :hands_json, :current_player,
                    :ply, :is_over, :result, :sfen, :in_check, :move_history_json,
                    :value_estimate, :game_type, :demo_slot, :opponent_id)""",
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
    db_path: str, since: str
) -> tuple[list[dict[str, Any]], str]:
    """Read game snapshots updated after `since` timestamp. Returns (rows, max_updated_at)."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM game_snapshots WHERE updated_at > ? ORDER BY game_id",
            (since,),
        ).fetchall()
        max_ts = since
        if rows:
            max_ts = max(dict(r)["updated_at"] for r in rows)
        return [dict(row) for row in rows], max_ts
    finally:
        conn.close()


def write_training_state(db_path: str, state: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
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
    finally:
        conn.close()


def read_training_state(db_path: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM training_state WHERE id = 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_heartbeat(db_path: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            "UPDATE training_state SET heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now') "
            "WHERE id = 1"
        )
        conn.commit()
    finally:
        conn.close()


def update_training_progress(
    db_path: str, epoch: int, step: int, checkpoint_path: str | None = None
) -> None:
    conn = _connect(db_path)
    try:
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
    finally:
        conn.close()


def read_league_data(db_path: str) -> dict[str, list[dict[str, Any]]]:
    """Read all league entries and results."""
    conn = _connect(db_path)
    try:
        entries = conn.execute(
            "SELECT id, architecture, elo_rating, games_played, created_epoch, created_at "
            "FROM league_entries ORDER BY elo_rating DESC"
        ).fetchall()
        results = conn.execute(
            "SELECT id, epoch, learner_id, opponent_id, wins, losses, draws, recorded_at "
            "FROM league_results ORDER BY id DESC"
        ).fetchall()
        return {
            "entries": [dict(r) for r in entries],
            "results": [dict(r) for r in results],
        }
    finally:
        conn.close()


def read_elo_history(db_path: str) -> list[dict[str, Any]]:
    """Read all Elo history points for charting."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT entry_id, epoch, elo_rating FROM elo_history ORDER BY epoch, entry_id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
