"""Gap-analysis tests for keisei.server.app: WebSocket polling, heartbeat
staleness, keepalive pings, and training-state diff logic."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from keisei.db import (
    init_db,
    update_heartbeat,
    update_training_progress,
    write_game_snapshots,
    write_metrics,
    write_training_state,
)
from keisei.server.app import (
    TEST_ALLOWED_HOSTS,
    _training_alive,
    create_app,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server_db(tmp_path: Path) -> str:
    """Initialised DB with a training_state row and fresh heartbeat."""
    path = str(tmp_path / "server_test.db")
    init_db(path)
    write_training_state(path, {
        "config_json": "{}",
        "display_name": "TestBot",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    # Ensure heartbeat is fresh
    update_heartbeat(path)
    return path


# ===================================================================
# _training_alive — unit tests
# ===================================================================


class TestTrainingAlive:
    """Direct tests for the heartbeat staleness check."""

    def test_alive_with_fresh_heartbeat(self, server_db: str) -> None:
        assert _training_alive(server_db) is True

    def test_stale_with_old_heartbeat(self, server_db: str) -> None:
        """If heartbeat_at is older than HEARTBEAT_STALE_S, training is stale."""
        # Write a very old heartbeat by manipulating the DB directly
        import sqlite3
        conn = sqlite3.connect(server_db)
        conn.execute(
            "UPDATE training_state SET heartbeat_at = '2020-01-01T00:00:00Z' WHERE id = 1"
        )
        conn.commit()
        conn.close()
        assert _training_alive(server_db) is False

    def test_missing_training_state(self, tmp_path: Path) -> None:
        """No training_state row → not alive."""
        path = str(tmp_path / "empty.db")
        init_db(path)
        assert _training_alive(path) is False

    def test_empty_heartbeat_string(self, server_db: str) -> None:
        """heartbeat_at = '' → not alive."""
        import sqlite3
        conn = sqlite3.connect(server_db)
        conn.execute(
            "UPDATE training_state SET heartbeat_at = '' WHERE id = 1"
        )
        conn.commit()
        conn.close()
        assert _training_alive(server_db) is False

    def test_malformed_heartbeat(self, server_db: str) -> None:
        """Unparseable heartbeat_at → not alive (exception caught)."""
        import sqlite3
        conn = sqlite3.connect(server_db)
        conn.execute(
            "UPDATE training_state SET heartbeat_at = 'not-a-date' WHERE id = 1"
        )
        conn.commit()
        conn.close()
        assert _training_alive(server_db) is False

    def test_nonexistent_db(self) -> None:
        """DB file doesn't exist → not alive."""
        assert _training_alive("/tmp/nonexistent_keisei_test_xyz.db") is False


# ===================================================================
# WebSocket polling — metrics_update push
# ===================================================================


class TestWSMetricsUpdate:
    """After init, new metrics written to DB should be pushed as metrics_update."""

    def test_metrics_update_pushed(self, server_db: str, ws_connect) -> None:
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)

        # Patch poll interval to be very short
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999):
            with ws_connect(app) as ws:
                # Consume init message
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"

                # Write new metrics
                write_metrics(server_db, {
                    "epoch": 1, "step": 200,
                    "policy_loss": 0.5, "value_loss": 0.3,
                })

                # Wait for the poll to pick it up
                msg = ws.receive_json(mode="text")
                # Could be game_update or metrics_update — drain until we find metrics
                attempts = 0
                while msg["type"] != "metrics_update" and attempts < 10:
                    msg = ws.receive_json(mode="text")
                    attempts += 1

                assert msg["type"] == "metrics_update"
                assert len(msg["rows"]) >= 1
                assert msg["rows"][0]["epoch"] == 1


# ===================================================================
# WebSocket polling — game_update push
# ===================================================================


class TestWSGameUpdate:
    """Game snapshots written after init should be pushed as game_update."""

    def test_game_update_pushed(self, server_db: str, ws_connect) -> None:
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)

        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999):
            with ws_connect(app) as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"

                # Write a game snapshot
                write_game_snapshots(server_db, [{
                    "game_id": 0, "board_json": "[]", "hands_json": "{}",
                    "current_player": "black", "ply": 42, "is_over": 0,
                    "result": "in_progress", "sfen": "startpos",
                    "in_check": 0, "move_history_json": "[]",
                    "value_estimate": 0.0,
                }])

                # Drain until game_update
                msg = ws.receive_json(mode="text")
                attempts = 0
                while msg["type"] != "game_update" and attempts < 10:
                    msg = ws.receive_json(mode="text")
                    attempts += 1

                assert msg["type"] == "game_update"
                assert len(msg["snapshots"]) >= 1
                assert msg["snapshots"][0]["ply"] == 42


# ===================================================================
# WebSocket polling — training_status push (diff logic)
# ===================================================================


class TestWSTrainingStatusDiff:
    """training_status is only pushed when epoch or status changes."""

    def test_status_pushed_on_epoch_change(self, server_db: str, ws_connect) -> None:
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)

        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999):
            with ws_connect(app) as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"

                # Update epoch in training_state
                update_training_progress(server_db, epoch=5, step=500)

                # Drain until training_status
                found = False
                for _ in range(20):
                    msg = ws.receive_json(mode="text")
                    if msg["type"] == "training_status":
                        assert msg["epoch"] == 5
                        found = True
                        break

                assert found, "Expected training_status message after epoch change"

    def test_no_status_push_when_epoch_unchanged(self, server_db: str, ws_connect) -> None:
        """When epoch and status don't change, no training_status is pushed.
        We force a metrics_update (observable) and check no training_status
        arrives before it."""
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)

        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999):
            with ws_connect(app) as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"

                # Just update heartbeat (no epoch/status change)
                update_heartbeat(server_db)

                # Write a metric so the poll loop produces a metrics_update
                write_metrics(server_db, {"epoch": 0, "step": 1})

                # Drain until we see metrics_update — collect everything on the way
                messages = []
                for _ in range(20):
                    msg = ws.receive_json(mode="text")
                    messages.append(msg)
                    if msg["type"] == "metrics_update":
                        break

                status_msgs = [m for m in messages if m["type"] == "training_status"]
                assert len(status_msgs) == 0, (
                    f"No training_status expected when epoch unchanged, got {status_msgs}"
                )


# ===================================================================
# WebSocket — keepalive ping
# ===================================================================


class TestWSKeepalive:
    """The server sends periodic ping messages."""

    def test_ping_received(self, server_db: str, ws_connect) -> None:
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)

        # Set ping interval very short so we get one quickly
        with patch("keisei.server.app.WS_PING_INTERVAL_S", 0.05), \
             patch("keisei.server.app.POLL_INTERVAL_S", 999):
            with ws_connect(app) as ws:
                # First message is init (from _poll_and_push)
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"

                # Next should be a ping from _keepalive
                msg = ws.receive_json(mode="text")
                assert msg["type"] == "ping"


# ===================================================================
# WebSocket — DB error during poll (graceful handling)
# ===================================================================


    # NOTE: DB-error-during-poll test omitted.  The async TaskGroup +
    # except* error-handling path cannot be exercised reliably with
    # Starlette's synchronous TestClient because the sync adapter
    # blocks waiting for the WS tasks, and the exception propagation
    # doesn't translate cleanly.  The error path is exercised
    # indirectly by the _training_alive tests (which cover DB failures)
    # and by test_healthz_db_missing (which exercises _db_accessible).


# ===================================================================
# M3 — _get_system_stats() psutil-unavailable fallback
# ===================================================================


class TestGetSystemStats:
    """Tests for _get_system_stats() psutil availability paths."""

    def test_psutil_unavailable_returns_none_fields(self) -> None:
        """When psutil import fails, CPU/RAM fields should be None."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("psutil not installed")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            from keisei.server.app import _get_system_stats
            stats = _get_system_stats()

        assert stats["cpu_percent"] is None
        assert stats["ram_used_gb"] is None
        assert stats["ram_total_gb"] is None

    def test_psutil_available_returns_values(self) -> None:
        """When psutil is available, CPU/RAM fields should be numeric."""
        from keisei.server.app import _get_system_stats
        stats = _get_system_stats()

        # psutil is installed in this test env, so we expect real values
        assert isinstance(stats["cpu_percent"], (int, float))
        assert isinstance(stats["ram_used_gb"], (int, float))
        assert isinstance(stats["ram_total_gb"], (int, float))


# ===================================================================
# H2 — system_stats in training_status WebSocket push
# ===================================================================


class TestWSTrainingStatusSystemStats:
    """H2: training_status push includes system_stats with expected structure."""

    def test_system_stats_present_in_training_status(self, server_db: str, ws_connect) -> None:
        """Trigger a training_status push and verify system_stats field."""
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)

        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999):
            with ws_connect(app) as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"

                # Change epoch to trigger a training_status push
                update_training_progress(server_db, epoch=10, step=1000)

                # Drain until training_status
                found = False
                for _ in range(20):
                    msg = ws.receive_json(mode="text")
                    if msg["type"] == "training_status":
                        # Verify system_stats is present
                        assert "system_stats" in msg, \
                            "training_status message must include system_stats"
                        sys_stats = msg["system_stats"]
                        assert isinstance(sys_stats, dict)

                        # Must have CPU/RAM keys (values may be None if psutil missing)
                        assert "cpu_percent" in sys_stats
                        assert "ram_used_gb" in sys_stats
                        assert "ram_total_gb" in sys_stats
                        # Must have gpus key (list, possibly empty)
                        assert "gpus" in sys_stats
                        assert isinstance(sys_stats["gpus"], list)

                        found = True
                        break

                assert found, "Expected training_status message with system_stats"


# ===================================================================
# HIGH-2 — _poll_and_push with pre-existing game snapshots at init
# ===================================================================


class TestWSInitWithPreExistingGames:
    """HIGH-2: When games already exist in DB before WebSocket connect,
    init message should include them and last_game_ts should be set."""

    def test_init_includes_pre_existing_games(self, server_db: str, ws_connect) -> None:
        """Seed DB with game snapshots before connecting; verify init message."""
        # Write games BEFORE WebSocket connect
        write_game_snapshots(server_db, [
            {
                "game_id": 0, "board_json": "[]", "hands_json": "{}",
                "current_player": "black", "ply": 10, "is_over": 0,
                "result": "in_progress", "sfen": "startpos",
                "in_check": 0, "move_history_json": "[]",
                "value_estimate": 0.5,
            },
            {
                "game_id": 1, "board_json": "[]", "hands_json": "{}",
                "current_player": "white", "ply": 20, "is_over": 0,
                "result": "in_progress", "sfen": "startpos",
                "in_check": 0, "move_history_json": "[]",
                "value_estimate": -0.3,
            },
        ])

        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)

        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999):
            with ws_connect(app) as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"
                # Init should include the 2 pre-existing games
                assert len(init_msg["games"]) == 2
                plies = {g["ply"] for g in init_msg["games"]}
                assert plies == {10, 20}

    def test_subsequent_updates_after_pre_existing_games(self, server_db: str, ws_connect) -> None:
        """After init with pre-existing games, updating an existing game should push game_update.

        We update an existing game_id (not create a new one) to ensure the
        updated_at timestamp changes and the poll detects it.
        """
        import time as _time

        write_game_snapshots(server_db, [{
            "game_id": 0, "board_json": "[]", "hands_json": "{}",
            "current_player": "black", "ply": 5, "is_over": 0,
            "result": "in_progress", "sfen": "startpos",
            "in_check": 0, "move_history_json": "[]",
            "value_estimate": 0.0,
        }])

        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)

        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999):
            with ws_connect(app) as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"
                assert len(init_msg["games"]) == 1

                # Wait briefly to ensure updated_at will differ
                _time.sleep(1.1)

                # Update existing game with new ply — this changes updated_at
                write_game_snapshots(server_db, [{
                    "game_id": 0, "board_json": "[updated]", "hands_json": "{}",
                    "current_player": "white", "ply": 99, "is_over": 0,
                    "result": "in_progress", "sfen": "startpos",
                    "in_check": 0, "move_history_json": "[]",
                    "value_estimate": 0.1,
                }])

                # Drain until game_update
                found = False
                for _ in range(20):
                    msg = ws.receive_json(mode="text")
                    if msg["type"] == "game_update":
                        new_plies = {s["ply"] for s in msg["snapshots"]}
                        assert 99 in new_plies
                        found = True
                        break

                assert found, "Expected game_update with updated snapshot after init"
