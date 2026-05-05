"""Metrics table — per-step training metrics."""

from __future__ import annotations

from typing import Any

from keisei.db._connection import _connect

DDL = """
CREATE TABLE IF NOT EXISTS metrics (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch              INTEGER NOT NULL,
    step               INTEGER NOT NULL,
    policy_loss        REAL,
    value_loss         REAL,
    entropy            REAL,
    win_rate           REAL,
    loss_rate          REAL,
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
"""


def write_metrics(db_path: str, metrics: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT INTO metrics (epoch, step, policy_loss, value_loss, entropy,
               win_rate, loss_rate, black_win_rate, white_win_rate, draw_rate,
               truncation_rate, avg_episode_length,
               gradient_norm, episodes_completed)
               VALUES (:epoch, :step, :policy_loss, :value_loss, :entropy,
               :win_rate, :loss_rate, :black_win_rate, :white_win_rate, :draw_rate,
               :truncation_rate, :avg_episode_length,
               :gradient_norm, :episodes_completed)""",
            {
                "epoch": metrics.get("epoch", 0), "step": metrics.get("step", 0),
                "policy_loss": metrics.get("policy_loss"), "value_loss": metrics.get("value_loss"),
                "entropy": metrics.get("entropy"), "win_rate": metrics.get("win_rate"),
                "loss_rate": metrics.get("loss_rate"),
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
