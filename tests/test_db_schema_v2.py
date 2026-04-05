"""Tests for DB schema: league tables, game_snapshots, and schema version."""

import sqlite3
import time

import pytest

from keisei.db import (
    SCHEMA_VERSION,
    init_db,
    read_elo_history,
    read_metrics_since,
    read_training_state,
    update_heartbeat,
    update_training_progress,
    write_metrics,
    write_training_state,
)

pytestmark = pytest.mark.integration


def _get_schema_version(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def _get_table_columns(db_path: str, table: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(f"PRAGMA table_info({table})")
        return [row[1] for row in cursor.fetchall()]
    finally:
        conn.close()


def _table_exists(db_path: str, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        return row is not None
    finally:
        conn.close()


class TestSchemaV2:
    def test_creates_league_tables(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        assert _table_exists(db_path, "league_entries")
        assert _table_exists(db_path, "league_results")

    def test_game_snapshots_has_new_columns(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "game_snapshots")
        assert "game_type" in cols
        assert "demo_slot" in cols

    def test_schema_version_is_current(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        assert _get_schema_version(db_path) == SCHEMA_VERSION

    def test_league_entries_columns(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "league_entries")
        assert "display_name" in cols
        assert "architecture" in cols
        assert "elo_rating" in cols
        assert "checkpoint_path" in cols
        assert "created_epoch" in cols

    def test_league_results_columns(self, tmp_path):
        db_path = str(tmp_path / "fresh.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "league_results")
        assert "learner_id" in cols
        assert "opponent_id" in cols
        assert "wins" in cols
        assert "draws" in cols

    def test_creates_elo_history_table(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "elo_history" in tables

    def test_elo_history_columns(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(elo_history)").fetchall()]
        conn.close()
        assert cols == ["id", "entry_id", "epoch", "elo_rating", "recorded_at"]

    def test_game_snapshots_has_opponent_id(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(game_snapshots)").fetchall()]
        conn.close()
        assert "opponent_id" in cols


class TestSchemaV5:
    """Phase 2: historical library, gauntlet results, role Elo columns."""

    def test_league_entries_has_role_elo_columns(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        conn.close()
        assert "elo_frontier" in cols
        assert "elo_dynamic" in cols
        assert "elo_recent" in cols
        assert "elo_historical" in cols

    def test_role_elo_defaults_to_1000(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES ('resnet', '{}', '/tmp/x.pt', 1)"
        )
        conn.commit()
        row = conn.execute("SELECT elo_frontier, elo_dynamic, elo_recent, elo_historical FROM league_entries").fetchone()
        conn.close()
        assert row == (1000.0, 1000.0, 1000.0, 1000.0)

    def test_historical_library_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        assert _table_exists(db_path, "historical_library")

    def test_historical_library_columns(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "historical_library")
        assert "slot_index" in cols
        assert "target_epoch" in cols
        assert "entry_id" in cols
        assert "actual_epoch" in cols
        assert "selected_at" in cols
        assert "selection_mode" in cols

    def test_gauntlet_results_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        assert _table_exists(db_path, "gauntlet_results")

    def test_gauntlet_results_columns(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        cols = _get_table_columns(db_path, "gauntlet_results")
        assert "epoch" in cols
        assert "entry_id" in cols
        assert "historical_slot" in cols
        assert "historical_entry_id" in cols
        assert "wins" in cols
        assert "losses" in cols
        assert "draws" in cols
        assert "elo_before" in cols
        assert "elo_after" in cols

    def test_mismatched_version_raises(self, tmp_path):
        """A database with a different schema version should raise RuntimeError."""
        db_path = str(tmp_path / "old.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version VALUES (99)")
        conn.commit()
        conn.close()
        import pytest
        with pytest.raises(RuntimeError, match="schema version 99"):
            init_db(db_path)

    def test_idempotent_init(self, tmp_path):
        """Running init_db twice should be a no-op."""
        db_path = str(tmp_path / "idempotent.db")
        init_db(db_path)
        init_db(db_path)  # Should not raise
        assert _get_schema_version(db_path) == SCHEMA_VERSION


class TestWriteAndReadMetrics:
    """Tests for write_metrics and read_metrics_since."""

    def test_write_metrics_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "metrics.db")
        init_db(db_path)
        metrics = {
            "epoch": 1,
            "step": 42,
            "policy_loss": 0.5,
            "value_loss": 0.3,
            "entropy": 1.2,
            "win_rate": 0.55,
            "loss_rate": 0.40,
            "black_win_rate": 0.60,
            "white_win_rate": 0.50,
            "draw_rate": 0.05,
            "truncation_rate": 0.02,
            "avg_episode_length": 120.5,
            "gradient_norm": 0.8,
            "episodes_completed": 100,
        }
        write_metrics(db_path, metrics)
        rows = read_metrics_since(db_path, since_id=0)
        assert len(rows) == 1
        row = rows[0]
        assert row["epoch"] == 1
        assert row["step"] == 42
        assert row["policy_loss"] == 0.5
        assert row["value_loss"] == 0.3
        assert row["entropy"] == 1.2
        assert row["win_rate"] == 0.55
        assert row["loss_rate"] == 0.40
        assert row["black_win_rate"] == 0.60
        assert row["white_win_rate"] == 0.50
        assert row["draw_rate"] == 0.05
        assert row["truncation_rate"] == 0.02
        assert row["avg_episode_length"] == 120.5
        assert row["gradient_norm"] == 0.8
        assert row["episodes_completed"] == 100
        assert "id" in row
        assert "timestamp" in row

    def test_read_metrics_since_filters_by_id(self, tmp_path):
        db_path = str(tmp_path / "metrics_filter.db")
        init_db(db_path)
        for i in range(1, 4):
            write_metrics(db_path, {"epoch": i, "step": i * 10})
        all_rows = read_metrics_since(db_path, since_id=0)
        assert len(all_rows) == 3
        first_id = all_rows[0]["id"]
        filtered = read_metrics_since(db_path, since_id=first_id)
        assert len(filtered) == 2
        assert all(r["id"] > first_id for r in filtered)


class TestWriteAndReadTrainingState:
    """Tests for write_training_state, read_training_state, update_training_progress."""

    def _make_state(self, **overrides):
        base = {
            "config_json": "{}",
            "display_name": "TestRun",
            "model_arch": "resnet",
            "algorithm_name": "katago_ppo",
            "started_at": "2026-01-01T00:00:00Z",
            "current_epoch": 0,
            "current_step": 0,
            "checkpoint_path": None,
            "total_epochs": 100,
            "status": "running",
        }
        base.update(overrides)
        return base

    def test_write_training_state_roundtrip(self, tmp_path):
        db_path = str(tmp_path / "state.db")
        init_db(db_path)
        state = self._make_state(
            current_epoch=3,
            status="paused",
            checkpoint_path="/tmp/ckpt.pt",
        )
        write_training_state(db_path, state)
        result = read_training_state(db_path)
        assert result is not None
        assert result["current_epoch"] == 3
        assert result["status"] == "paused"
        assert result["checkpoint_path"] == "/tmp/ckpt.pt"

    def test_update_training_progress(self, tmp_path):
        db_path = str(tmp_path / "progress.db")
        init_db(db_path)
        write_training_state(db_path, self._make_state())
        update_training_progress(
            db_path, epoch=5, step=100, checkpoint_path="/x.pt", phase="league"
        )
        result = read_training_state(db_path)
        assert result is not None
        assert result["current_epoch"] == 5
        assert result["current_step"] == 100
        assert result["checkpoint_path"] == "/x.pt"
        assert result["phase"] == "league"


class TestUpdateHeartbeat:
    """Tests for update_heartbeat."""

    def test_update_heartbeat_updates_existing(self, tmp_path):
        db_path = str(tmp_path / "heartbeat.db")
        init_db(db_path)
        state = {
            "config_json": "{}",
            "display_name": "HB",
            "model_arch": "resnet",
            "algorithm_name": "katago_ppo",
            "started_at": "2026-01-01T00:00:00Z",
            "current_epoch": 0,
            "current_step": 0,
            "checkpoint_path": None,
            "total_epochs": 10,
            "status": "running",
        }
        write_training_state(db_path, state)
        before = read_training_state(db_path)
        assert before is not None
        hb_before = before["heartbeat_at"]
        # Small delay so the timestamp changes
        time.sleep(1.1)
        update_heartbeat(db_path)
        after = read_training_state(db_path)
        assert after is not None
        hb_after = after["heartbeat_at"]
        assert hb_after > hb_before


class TestReadEloHistory:
    """Tests for read_elo_history ordering."""

    def test_read_elo_history_ordering(self, tmp_path):
        db_path = str(tmp_path / "elo.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        # Create a league entry to satisfy FK
        conn.execute(
            "INSERT INTO league_entries (id, architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES (1, 'resnet', '{}', '/tmp/x.pt', 1)"
        )
        conn.execute(
            "INSERT INTO league_entries (id, architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES (2, 'resnet', '{}', '/tmp/y.pt', 1)"
        )
        # Insert elo_history rows in non-sorted order
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (2, 10, 1100.0)"
        )
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (1, 5, 1050.0)"
        )
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (1, 10, 1080.0)"
        )
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (2, 5, 1020.0)"
        )
        conn.commit()
        conn.close()

        rows = read_elo_history(db_path)
        assert len(rows) == 4
        # Should be ordered by (epoch, entry_id)
        assert (rows[0]["epoch"], rows[0]["entry_id"]) == (5, 1)
        assert (rows[1]["epoch"], rows[1]["entry_id"]) == (5, 2)
        assert (rows[2]["epoch"], rows[2]["entry_id"]) == (10, 1)
        assert (rows[3]["epoch"], rows[3]["entry_id"]) == (10, 2)
