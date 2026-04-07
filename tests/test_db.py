import json
import sqlite3
from pathlib import Path

import pytest

from keisei.db import (
    SCHEMA_VERSION,
    _connect,
    init_db,
    read_elo_history,
    read_game_snapshots,
    read_league_data,
    read_metrics_since,
    read_tournament_stats,
    read_training_state,
    update_heartbeat,
    update_training_progress,
    write_game_snapshots,
    write_metrics,
    write_tournament_stats,
    write_training_state,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


@pytest.fixture
def db(db_path: Path) -> Path:
    init_db(str(db_path))
    return db_path


def test_init_creates_tables(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "schema_version" in tables
    assert "metrics" in tables
    assert "game_snapshots" in tables
    assert "training_state" in tables
    conn.close()


def test_init_is_idempotent(db_path: Path) -> None:
    init_db(str(db_path))
    init_db(str(db_path))


def test_schema_version(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    conn.close()


def test_wal_mode_enabled(db: Path) -> None:
    conn = sqlite3.connect(str(db))
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    conn.close()


def test_metrics_round_trip(db: Path) -> None:
    write_metrics(str(db), {
        "epoch": 1, "step": 100, "policy_loss": 2.5, "value_loss": 0.8,
        "entropy": 5.1, "win_rate": 0.52, "draw_rate": 0.1,
        "truncation_rate": 0.05, "avg_episode_length": 120.5,
        "gradient_norm": 1.2, "episodes_completed": 50,
    })
    rows = read_metrics_since(str(db), since_id=0)
    assert len(rows) == 1
    row = rows[0]
    assert row["epoch"] == 1
    assert row["step"] == 100
    assert abs(row["policy_loss"] - 2.5) < 1e-6
    assert abs(row["win_rate"] - 0.52) < 1e-6
    assert row["episodes_completed"] == 50
    assert "id" in row
    assert "timestamp" in row


def test_metrics_since_filters(db: Path) -> None:
    for i in range(5):
        write_metrics(str(db), {"epoch": i, "step": i * 10})
    rows = read_metrics_since(str(db), since_id=3)
    assert len(rows) == 2
    assert rows[0]["epoch"] == 3
    assert rows[1]["epoch"] == 4


def test_game_snapshots_round_trip(db: Path) -> None:
    board: list[dict[str, object] | None] = [None] * 81
    board[0] = {"type": "king", "color": "black", "promoted": False, "row": 0, "col": 0}
    hands = {"black": {"pawn": 2}, "white": {"pawn": 0}}
    history = [{"action": 42, "notation": "P-7f"}]
    snapshots = [{
        "game_id": 0,
        "board_json": json.dumps(board),
        "hands_json": json.dumps(hands),
        "current_player": "black",
        "ply": 10, "is_over": 0, "result": "in_progress",
        "sfen": "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "in_check": 0,
        "move_history_json": json.dumps(history),
        "value_estimate": 0.42,
    }]
    write_game_snapshots(str(db), snapshots)
    result = read_game_snapshots(str(db))
    assert len(result) == 1
    assert result[0]["game_id"] == 0
    assert result[0]["ply"] == 10
    assert json.loads(result[0]["board_json"])[0]["type"] == "king"
    assert json.loads(result[0]["move_history_json"]) == history


def test_game_snapshots_overwrite(db: Path) -> None:
    snap = {
        "game_id": 0, "board_json": "[]", "hands_json": "{}",
        "current_player": "black", "ply": 1, "is_over": 0,
        "result": "in_progress", "sfen": "startpos", "in_check": 0,
        "move_history_json": "[]",
        "value_estimate": 0.0,
    }
    write_game_snapshots(str(db), [snap])
    snap["ply"] = 99
    write_game_snapshots(str(db), [snap])
    result = read_game_snapshots(str(db))
    assert len(result) == 1
    assert result[0]["ply"] == 99


def test_training_state_write_and_read(db: Path) -> None:
    write_training_state(str(db), {
        "config_json": '{"test": true}',
        "display_name": "Hikaru",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
    })
    state = read_training_state(str(db))
    assert state is not None
    assert state["display_name"] == "Hikaru"
    assert state["status"] == "running"
    assert state["current_epoch"] == 0


def test_update_heartbeat(db: Path) -> None:
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
    })
    old_state = read_training_state(str(db))
    update_heartbeat(str(db))
    new_state = read_training_state(str(db))
    assert old_state is not None
    assert new_state is not None
    assert new_state["heartbeat_at"] >= old_state["heartbeat_at"]


def test_update_training_progress(db: Path) -> None:
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
    })
    update_training_progress(str(db), epoch=5, step=500, checkpoint_path="/tmp/ckpt.pt")
    state = read_training_state(str(db))
    assert state is not None
    assert state["current_epoch"] == 5
    assert state["current_step"] == 500
    assert state["checkpoint_path"] == "/tmp/ckpt.pt"


# ---------------------------------------------------------------------------
# High gap: update_training_progress without checkpoint_path
# ---------------------------------------------------------------------------


def test_update_training_progress_no_checkpoint(db: Path) -> None:
    """Calling update_training_progress without checkpoint_path should update
    epoch/step but leave checkpoint_path unchanged."""
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
    })
    # First set a checkpoint
    update_training_progress(str(db), epoch=5, step=500, checkpoint_path="/tmp/ckpt.pt")
    # Then update without checkpoint_path
    update_training_progress(str(db), epoch=10, step=1000)
    state = read_training_state(str(db))
    assert state is not None
    assert state["current_epoch"] == 10
    assert state["current_step"] == 1000
    # checkpoint_path should remain from the previous update
    assert state["checkpoint_path"] == "/tmp/ckpt.pt"


def test_training_state_learner_entry_id(db: Path) -> None:
    """learner_entry_id should be stored and retrieved from training_state."""
    init_db(str(db))
    write_training_state(str(db), {
        "config_json": '{"test": true}',
        "display_name": "Hikaru",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
        "learner_entry_id": 42,
    })
    state = read_training_state(str(db))
    assert state is not None
    assert state["learner_entry_id"] == 42


def test_training_state_learner_entry_id_defaults_null(db: Path) -> None:
    """learner_entry_id should default to None when not provided."""
    init_db(str(db))
    write_training_state(str(db), {
        "config_json": '{"test": true}',
        "display_name": "Hikaru",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
    })
    state = read_training_state(str(db))
    assert state is not None
    assert state["learner_entry_id"] is None


def test_update_training_progress_with_learner_entry_id(db: Path) -> None:
    """update_training_progress should update learner_entry_id when provided."""
    init_db(str(db))
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
    })
    update_training_progress(str(db), epoch=5, step=500, learner_entry_id=99)
    state = read_training_state(str(db))
    assert state is not None
    assert state["learner_entry_id"] == 99


def test_update_training_progress_preserves_learner_entry_id(db: Path) -> None:
    """update_training_progress without learner_entry_id should not clear it."""
    init_db(str(db))
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
        "learner_entry_id": 42,
    })
    update_training_progress(str(db), epoch=10, step=1000)
    state = read_training_state(str(db))
    assert state is not None
    assert state["learner_entry_id"] == 42


def test_read_metrics_since_with_limit(db: Path) -> None:
    """read_metrics_since should respect the limit parameter."""
    for i in range(10):
        write_metrics(str(db), {"epoch": i, "step": i * 10})
    rows = read_metrics_since(str(db), since_id=0, limit=3)
    assert len(rows) == 3
    assert rows[0]["epoch"] == 0
    assert rows[2]["epoch"] == 2


class TestLeagueDataReaders:
    def test_read_league_data_empty(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        data = read_league_data(db_path)
        assert data["entries"] == []
        assert data["results"] == []

    def test_read_league_data_with_entries(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES ('transformer', '{}', '/tmp/ckpt.pt', 5)"
        )
        conn.execute(
            "INSERT INTO league_results (epoch, entry_a_id, entry_b_id, match_type, num_games, wins_a, wins_b, draws) "
            "VALUES (5, 1, 1, 'calibration', 5, 3, 1, 1)"
        )
        conn.commit()
        conn.close()
        data = read_league_data(db_path)
        assert len(data["entries"]) == 1
        assert data["entries"][0]["architecture"] == "transformer"
        assert len(data["results"]) == 1
        assert data["results"][0]["wins_a"] == 3

    def test_read_elo_history_empty(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        history = read_elo_history(db_path)
        assert history == []

    def test_read_elo_history_with_data(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES ('transformer', '{}', '/tmp/ckpt.pt', 5)"
        )
        conn.execute("INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (1, 5, 1050.0)")
        conn.execute("INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (1, 6, 1100.0)")
        conn.commit()
        conn.close()
        history = read_elo_history(db_path)
        assert len(history) == 2
        assert history[0]["elo_rating"] == 1050.0
        assert history[1]["epoch"] == 6

    def test_read_league_data_includes_games_vs_counts(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch, "
            "games_vs_frontier, games_vs_dynamic, games_vs_recent) "
            "VALUES ('transformer', '{}', '/tmp/ckpt.pt', 5, 10, 20, 30)"
        )
        conn.commit()
        conn.close()
        data = read_league_data(db_path)
        entry = data["entries"][0]
        assert entry["games_vs_frontier"] == 10
        assert entry["games_vs_dynamic"] == 20
        assert entry["games_vs_recent"] == 30


class TestSchemaV4:
    def test_league_entries_has_role_column(self, tmp_path):
        db_path = str(tmp_path / "v4.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        conn.close()
        assert "role" in cols
        assert "status" in cols
        assert "parent_entry_id" in cols
        assert "lineage_group" in cols
        assert "protection_remaining" in cols
        assert "last_match_at" in cols

    def test_league_transitions_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v4.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "league_transitions" in tables

    def test_league_meta_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v4.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT bootstrapped FROM league_meta WHERE id = 1").fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 0

    def test_league_results_entry_indexes_exist(self, tmp_path):
        db_path = str(tmp_path / "v4.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        indexes = [r[1] for r in conn.execute("PRAGMA index_list(league_results)").fetchall()]
        conn.close()
        assert "idx_league_results_entry_a" in indexes
        assert "idx_league_results_entry_b" in indexes


# ---------------------------------------------------------------------------
# Critical gap: tournament stats write / read round-trip
# ---------------------------------------------------------------------------


class TestTournamentStats:
    """write_tournament_stats / read_tournament_stats were never directly tested.

    These feed the dashboard's tournament monitoring panel.  A schema mismatch
    or getattr bug here silently delivers wrong data — or crashes the server
    poll loop.
    """

    def test_read_returns_none_when_empty(self, db: Path) -> None:
        """Before any writes, read_tournament_stats should return None."""
        result = read_tournament_stats(str(db))
        assert result is None

    def test_write_and_read_round_trip(self, db: Path) -> None:
        """Write a RoundStats-like object, read it back, verify all fields."""
        from keisei.training.concurrent_matches import RoundStats

        stats = RoundStats(
            round_duration_s=12.5,
            pairings_requested=4,
            pairings_completed=3,
            total_games=192,
            total_plies=48000,
            active_slots=4,
            model_load_time_s=2.1,
            model_load_count=8,
        )
        write_tournament_stats(str(db), stats)
        row = read_tournament_stats(str(db))

        assert row is not None
        assert row["round_duration_s"] == pytest.approx(12.5)
        assert row["pairings_requested"] == 4
        assert row["pairings_completed"] == 3
        assert row["total_games"] == 192
        assert row["total_plies"] == 48000
        assert row["active_slots"] == 4
        assert row["model_load_time_s"] == pytest.approx(2.1)
        assert row["model_load_count"] == 8
        # games_per_min is computed: 192 / 12.5 * 60 = 921.6
        assert row["games_per_min"] == pytest.approx(921.6)
        assert "updated_at" in row

    def test_upsert_overwrites_on_second_write(self, db: Path) -> None:
        """Second write should update the single row, not insert a duplicate."""
        from keisei.training.concurrent_matches import RoundStats

        stats1 = RoundStats(total_games=100, round_duration_s=10.0)
        write_tournament_stats(str(db), stats1)

        stats2 = RoundStats(total_games=200, round_duration_s=8.0)
        write_tournament_stats(str(db), stats2)

        row = read_tournament_stats(str(db))
        assert row is not None
        assert row["total_games"] == 200
        assert row["games_per_min"] == pytest.approx(200 / 8.0 * 60)

    def test_games_per_min_zero_duration(self, db: Path) -> None:
        """Zero-duration round should yield games_per_min=0, not division error."""
        from keisei.training.concurrent_matches import RoundStats

        stats = RoundStats(total_games=50, round_duration_s=0.0)
        write_tournament_stats(str(db), stats)

        row = read_tournament_stats(str(db))
        assert row is not None
        assert row["games_per_min"] == pytest.approx(0.0)

    def test_accepts_plain_object_with_attrs(self, db: Path) -> None:
        """write_tournament_stats uses getattr — any object with matching attrs works."""
        from types import SimpleNamespace

        stats = SimpleNamespace(
            round_duration_s=5.0,
            pairings_requested=2,
            pairings_completed=2,
            total_games=64,
            total_plies=12000,
            active_slots=2,
            model_load_time_s=1.0,
            model_load_count=4,
        )
        write_tournament_stats(str(db), stats)
        row = read_tournament_stats(str(db))
        assert row is not None
        assert row["total_games"] == 64


def test_init_db_migrates_learner_entry_id(tmp_path: Path) -> None:
    """init_db should add learner_entry_id column to existing training_state tables."""
    db = tmp_path / "migrate.db"
    # Create a DB with the old schema (no learner_entry_id)
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE training_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        config_json TEXT NOT NULL,
        display_name TEXT NOT NULL,
        model_arch TEXT NOT NULL,
        algorithm_name TEXT NOT NULL,
        started_at TEXT NOT NULL,
        current_epoch INTEGER NOT NULL DEFAULT 0,
        current_step INTEGER NOT NULL DEFAULT 0,
        checkpoint_path TEXT,
        total_epochs INTEGER,
        status TEXT NOT NULL DEFAULT 'running',
        phase TEXT NOT NULL DEFAULT 'init',
        heartbeat_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    )""")
    conn.execute(
        "INSERT INTO training_state (id, config_json, display_name, model_arch, "
        "algorithm_name, started_at) VALUES (1, '{}', 'Test', 'resnet', 'ppo', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    # Run init_db — should migrate
    init_db(str(db))

    # Verify the column exists and existing data is preserved
    state = read_training_state(str(db))
    assert state is not None
    assert state["display_name"] == "Test"
    assert state["learner_entry_id"] is None


class TestForeignKeyEnforcement:
    """db.py._connect() must enable PRAGMA foreign_keys=ON so FK constraints
    are enforced on every connection, not just OpponentPool's."""

    def test_db_connect_enforces_foreign_keys(self, db: Path) -> None:
        """_connect() should enable FK enforcement — inserting a league_results
        row with a nonexistent entry_a_id must raise IntegrityError."""
        conn = _connect(str(db))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO league_results (epoch, entry_a_id, entry_b_id, match_type, num_games, wins_a, wins_b, draws) "
                "VALUES (1, 9999, 9999, 'calibration', 1, 1, 0, 0)"
            )
        conn.close()
