"""Tests for showcase WebSocket extensions."""
from __future__ import annotations

import contextlib
import json
import time
from concurrent.futures import CancelledError
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from keisei.db import init_db, write_training_state
from keisei.server.app import create_app, TEST_ALLOWED_HOSTS
from keisei.showcase.db_ops import (
    queue_match,
    claim_next_match,
    create_showcase_game,
    write_showcase_move,
    write_heartbeat,
    read_queue,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def server_db(tmp_path: Path) -> str:
    path = str(tmp_path / "server_test.db")
    init_db(path)
    write_training_state(path, {
        "config_json": "{}",
        "display_name": "test-model",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-01-01T00:00:00Z",
        "status": "idle",
        "current_epoch": 0,
        "current_step": 0,
    })
    return path


@contextlib.contextmanager
def _ws_connect(app, path="/ws"):
    """WebSocket connect that tolerates server-side CancelledError on teardown.

    The bidirectional _receive_commands coroutine can race with
    Starlette's sync TestClient teardown, producing a harmless
    CancelledError from the background thread.
    """
    client = TestClient(app, raise_server_exceptions=False)
    try:
        with client.websocket_connect(path) as ws:
            yield ws
    except CancelledError:
        pass


class TestShowcaseInit:
    def test_init_message_contains_showcase(self, server_db: str) -> None:
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999), \
             patch("keisei.server.app.SHOWCASE_POLL_INTERVAL_S", 999):
            with _ws_connect(app) as ws:
                msg = ws.receive_json()
                assert msg["type"] == "init"
                assert "showcase" in msg
                assert "queue" in msg["showcase"]
                assert "sidecar_alive" in msg["showcase"]

    def test_init_showcase_with_active_game(self, server_db: str) -> None:
        qid = queue_match(server_db, "e1", "e2", "normal")
        claim_next_match(server_db)
        game_id = create_showcase_game(server_db, queue_id=qid,
            entry_id_black="e1", entry_id_white="e2",
            elo_black=1500, elo_white=1480, name_black="A", name_white="B")
        write_showcase_move(server_db, game_id=game_id, ply=1, action_index=42,
            usi_notation="7g7f", board_json="[]", hands_json="{}",
            current_player="white", in_check=False, value_estimate=0.5,
            top_candidates="[]", move_time_ms=10)

        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999), \
             patch("keisei.server.app.SHOWCASE_POLL_INTERVAL_S", 999):
            with _ws_connect(app) as ws:
                msg = ws.receive_json()
                assert msg["showcase"]["game"] is not None
                assert len(msg["showcase"]["moves"]) == 1


class TestShowcaseCommands:
    def test_request_match_creates_queue_entry(self, server_db: str) -> None:
        from keisei.db import _connect
        conn = _connect(server_db)
        conn.execute("INSERT INTO league_entries (id, display_name, architecture, model_params, checkpoint_path, elo_rating, status, created_epoch) VALUES (1, 'A', 'resnet', '{}', '/tmp/a.pt', 1500, 'active', 0)")
        conn.execute("INSERT INTO league_entries (id, display_name, architecture, model_params, checkpoint_path, elo_rating, status, created_epoch) VALUES (2, 'B', 'resnet', '{}', '/tmp/b.pt', 1480, 'active', 0)")
        conn.commit()
        conn.close()

        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999), \
             patch("keisei.server.app.SHOWCASE_POLL_INTERVAL_S", 999):
            with _ws_connect(app) as ws:
                ws.receive_json()  # init
                ws.send_json({
                    "type": "request_showcase_match",
                    "entry_id_1": "1",
                    "entry_id_2": "2",
                    "speed": "normal",
                })
                time.sleep(0.1)
                queue = read_queue(server_db)
                assert len(queue) == 1
                assert queue[0]["entry_id_1"] == "1"

    def test_request_match_validates_self_match(self, server_db: str) -> None:
        from keisei.db import _connect
        conn = _connect(server_db)
        conn.execute("INSERT INTO league_entries (id, display_name, architecture, model_params, checkpoint_path, elo_rating, status, created_epoch) VALUES (1, 'A', 'resnet', '{}', '/tmp/a.pt', 1500, 'active', 0)")
        conn.commit()
        conn.close()

        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999), \
             patch("keisei.server.app.SHOWCASE_POLL_INTERVAL_S", 999):
            with _ws_connect(app) as ws:
                ws.receive_json()  # init
                ws.send_json({
                    "type": "request_showcase_match",
                    "entry_id_1": "1",
                    "entry_id_2": "1",
                    "speed": "normal",
                })
                time.sleep(0.1)
                msg = ws.receive_json()
                assert msg["type"] == "showcase_error"

    def test_invalid_speed_rejected(self, server_db: str) -> None:
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999), \
             patch("keisei.server.app.SHOWCASE_POLL_INTERVAL_S", 999):
            with _ws_connect(app) as ws:
                ws.receive_json()  # init
                ws.send_json({
                    "type": "request_showcase_match",
                    "entry_id_1": "1",
                    "entry_id_2": "2",
                    "speed": "turbo",
                })
                time.sleep(0.1)
                msg = ws.receive_json()
                assert msg["type"] == "showcase_error"
