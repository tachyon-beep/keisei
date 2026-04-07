"""SQLite database layer — schema, migrations, read/write helpers."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

SCHEMA_VERSION = 2


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate_add_column(
    conn: sqlite3.Connection, table: str, column: str, col_type: str,
) -> None:
    """Add a column to a table if it doesn't already exist."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2: Add columns to league_entries and training_state.

    New *tables* (game_features, style_profiles, tournament_stats, etc.) are
    handled by CREATE TABLE IF NOT EXISTS in init_db().  This migration covers
    columns added to tables that already exist in v1.

    Note: SQLite ALTER TABLE ADD COLUMN does not support expression defaults
    like strftime(...). Constant defaults are used; the expression defaults in
    the CREATE TABLE DDL apply to freshly-created v2 databases.
    """
    # league_entries columns added in v2
    _migrate_add_column(conn, "league_entries", "games_played", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "created_at", "TEXT NOT NULL DEFAULT ''")
    _migrate_add_column(conn, "league_entries", "role", "TEXT NOT NULL DEFAULT 'unassigned'")
    _migrate_add_column(conn, "league_entries", "status", "TEXT NOT NULL DEFAULT 'active'")
    _migrate_add_column(conn, "league_entries", "parent_entry_id", "INTEGER")
    _migrate_add_column(conn, "league_entries", "lineage_group", "TEXT")
    _migrate_add_column(conn, "league_entries", "protection_remaining", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "last_match_at", "TEXT")
    _migrate_add_column(conn, "league_entries", "elo_frontier", "REAL NOT NULL DEFAULT 1000.0")
    _migrate_add_column(conn, "league_entries", "elo_dynamic", "REAL NOT NULL DEFAULT 1000.0")
    _migrate_add_column(conn, "league_entries", "elo_recent", "REAL NOT NULL DEFAULT 1000.0")
    _migrate_add_column(conn, "league_entries", "elo_historical", "REAL NOT NULL DEFAULT 1000.0")
    _migrate_add_column(conn, "league_entries", "optimizer_path", "TEXT")
    _migrate_add_column(conn, "league_entries", "update_count", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "last_train_at", "TEXT")
    _migrate_add_column(conn, "league_entries", "retired_at", "TEXT")
    _migrate_add_column(conn, "league_entries", "training_enabled", "INTEGER NOT NULL DEFAULT 1")
    _migrate_add_column(conn, "league_entries", "games_vs_frontier", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "games_vs_dynamic", "INTEGER NOT NULL DEFAULT 0")
    _migrate_add_column(conn, "league_entries", "games_vs_recent", "INTEGER NOT NULL DEFAULT 0")
    # training_state column added in v2
    _migrate_add_column(conn, "training_state", "learner_entry_id", "INTEGER")


# Migration registry: maps target version to the function that migrates
# from (target - 1) → target.  Each function receives an open connection
# and must be idempotent (safe to re-run on an already-migrated DB).
_MIGRATIONS: dict[int, callable] = {
    2: _migrate_v1_to_v2,
}


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
                games_vs_recent   INTEGER NOT NULL DEFAULT 0
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
            CREATE TABLE IF NOT EXISTS elo_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id    INTEGER NOT NULL REFERENCES league_entries(id),
                epoch       INTEGER NOT NULL,
                elo_rating  REAL NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE INDEX IF NOT EXISTS idx_elo_history_entry ON elo_history(entry_id);
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
            CREATE INDEX IF NOT EXISTS idx_league_results_entry_a ON league_results(entry_a_id);
            CREATE INDEX IF NOT EXISTS idx_league_results_entry_b ON league_results(entry_b_id);
            CREATE TABLE IF NOT EXISTS league_meta (
                id           INTEGER PRIMARY KEY CHECK (id = 1),
                bootstrapped INTEGER NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO league_meta (id, bootstrapped) VALUES (1, 0);
            CREATE TABLE IF NOT EXISTS historical_library (
                slot_index     INTEGER NOT NULL PRIMARY KEY,
                target_epoch   INTEGER NOT NULL,
                entry_id       INTEGER REFERENCES league_entries(id),
                actual_epoch   INTEGER,
                selected_at    TEXT NOT NULL,
                selection_mode TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS gauntlet_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch               INTEGER NOT NULL,
                entry_id            INTEGER NOT NULL REFERENCES league_entries(id),
                historical_slot     INTEGER NOT NULL,
                historical_entry_id INTEGER NOT NULL REFERENCES league_entries(id),
                wins                INTEGER NOT NULL,
                losses              INTEGER NOT NULL,
                draws               INTEGER NOT NULL,
                elo_before          REAL,
                elo_after           REAL,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE INDEX IF NOT EXISTS idx_gauntlet_epoch ON gauntlet_results(epoch);
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
            CREATE TABLE IF NOT EXISTS style_profiles (
                checkpoint_id       INTEGER PRIMARY KEY REFERENCES league_entries(id),
                recomputed_at       TEXT NOT NULL,
                profile_status      TEXT NOT NULL DEFAULT 'insufficient',
                games_sampled       INTEGER NOT NULL DEFAULT 0,
                raw_metrics_json    TEXT NOT NULL DEFAULT '{}',
                percentile_json     TEXT NOT NULL DEFAULT '{}',
                primary_style       TEXT,
                secondary_traits    TEXT NOT NULL DEFAULT '[]',
                commentary_json     TEXT NOT NULL DEFAULT '[]',
                updated_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
        """)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            # No version row: either a truly fresh DB or a pre-versioning DB.
            # Treat as version 0 so all migrations run (they're idempotent).
            db_version = 0
        else:
            db_version = row[0]

        if db_version > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {db_version} is newer than "
                f"expected version {SCHEMA_VERSION}. "
                f"Upgrade the application or delete the database."
            )
        if db_version < SCHEMA_VERSION:
            for target in range(db_version + 1, SCHEMA_VERSION + 1):
                migrate_fn = _MIGRATIONS.get(target)
                if migrate_fn is not None:
                    migrate_fn(conn)
            if row is None:
                conn.execute(
                    "INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,)
                )
            else:
                conn.execute(
                    "UPDATE schema_version SET version = ?", (SCHEMA_VERSION,)
                )
        conn.commit()
    finally:
        conn.close()


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


def write_style_profile(db_path: str, profile: dict[str, Any]) -> None:
    """Upsert a single checkpoint style profile."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT INTO style_profiles
               (checkpoint_id, recomputed_at, profile_status, games_sampled,
                raw_metrics_json, percentile_json, primary_style,
                secondary_traits, commentary_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                       strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
               ON CONFLICT(checkpoint_id) DO UPDATE SET
                 recomputed_at    = excluded.recomputed_at,
                 profile_status   = excluded.profile_status,
                 games_sampled    = excluded.games_sampled,
                 raw_metrics_json = excluded.raw_metrics_json,
                 percentile_json  = excluded.percentile_json,
                 primary_style    = excluded.primary_style,
                 secondary_traits = excluded.secondary_traits,
                 commentary_json  = excluded.commentary_json,
                 updated_at       = excluded.updated_at""",
            (
                profile["checkpoint_id"],
                profile["recomputed_at"],
                profile["profile_status"],
                profile["games_sampled"],
                json.dumps(profile.get("raw_metrics", {})),
                json.dumps(profile.get("percentiles", {})),
                profile.get("primary_style"),
                json.dumps(profile.get("secondary_traits", [])),
                json.dumps(profile.get("commentary", [])),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def read_style_profiles(db_path: str) -> list[dict[str, Any]]:
    """Read all style profiles for the league."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM style_profiles ORDER BY checkpoint_id"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["raw_metrics"] = json.loads(d.pop("raw_metrics_json"))
            d["percentiles"] = json.loads(d.pop("percentile_json"))
            d["secondary_traits"] = json.loads(d["secondary_traits"])
            d["commentary"] = json.loads(d.pop("commentary_json"))
            result.append(d)
        return result
    finally:
        conn.close()
