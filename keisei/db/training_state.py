"""Training-state singleton table — current run config, progress, heartbeat."""

from __future__ import annotations

from typing import Any

from keisei.db._connection import _connect

DDL = """
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
    total_epochs     INTEGER,
    status           TEXT NOT NULL DEFAULT 'running',
    phase            TEXT NOT NULL DEFAULT 'init',
    heartbeat_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    learner_entry_id INTEGER
);
"""


def write_training_state(db_path: str, state: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO training_state
               (id, config_json, display_name, model_arch, algorithm_name,
                started_at, current_epoch, current_step, checkpoint_path,
                total_epochs, status, phase, learner_entry_id)
               VALUES (1, :config_json, :display_name, :model_arch, :algorithm_name,
                :started_at, :current_epoch, :current_step, :checkpoint_path,
                :total_epochs, :status, :phase, :learner_entry_id)""",
            {
                "config_json": state["config_json"], "display_name": state["display_name"],
                "model_arch": state["model_arch"], "algorithm_name": state["algorithm_name"],
                "started_at": state["started_at"],
                "current_epoch": state.get("current_epoch", 0),
                "current_step": state.get("current_step", 0),
                "checkpoint_path": state.get("checkpoint_path"),
                "total_epochs": state.get("total_epochs"),
                "status": state.get("status", "running"),
                "phase": state.get("phase", "init"),
                "learner_entry_id": state.get("learner_entry_id"),
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
    db_path: str, epoch: int, step: int, checkpoint_path: str | None = None,
    phase: str | None = None, learner_entry_id: int | None = None,
) -> None:
    conn = _connect(db_path)
    try:
        parts = ["current_epoch = ?", "current_step = ?",
                 "heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"]
        params: list[int | str] = [epoch, step]
        if checkpoint_path is not None:
            parts.append("checkpoint_path = ?")
            params.append(checkpoint_path)
        if phase is not None:
            parts.append("phase = ?")
            params.append(phase)
        if learner_entry_id is not None:
            parts.append("learner_entry_id = ?")
            params.append(learner_entry_id)
        conn.execute(
            f"UPDATE training_state SET {', '.join(parts)} WHERE id = 1",
            params,
        )
        conn.commit()
    finally:
        conn.close()


def write_epoch_summary(
    db_path: str,
    metrics: dict[str, Any],
    epoch: int,
    step: int,
    checkpoint_path: str | None = None,
) -> None:
    """Write metrics and training progress in a single connection+transaction.

    Batches what was previously 2-3 separate connect/commit/close cycles into
    one, reducing WAL page generation and eliminating redundant WAL index scans.
    Also performs a WAL checkpoint (TRUNCATE) at the end to prevent WAL growth
    across epochs.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN")
        # 1. Metrics INSERT
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
        # 2. Training progress UPDATE
        parts = ["current_epoch = ?", "current_step = ?",
                 "heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"]
        params: list[int | str] = [epoch, step]
        if checkpoint_path is not None:
            parts.append("checkpoint_path = ?")
            params.append(checkpoint_path)
        conn.execute(
            f"UPDATE training_state SET {', '.join(parts)} WHERE id = 1",
            params,
        )
        conn.commit()

        # 3. WAL checkpoint — cheap after a small batch of writes.
        # TRUNCATE mode merges WAL back into main DB and resets WAL to zero
        # length, preventing read degradation from WAL index scans.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
