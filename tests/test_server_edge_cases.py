"""Edge-case tests for keisei.server.app functions not covered by
test_server.py or test_server_gaps.py."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from starlette.testclient import TestClient

from keisei.db import init_db, update_heartbeat, write_training_state
from keisei.server.app import _db_accessible, _get_system_stats, create_app


class TestGetSystemStatsNvidiaSmi:
    """nvidia-smi failure modes in _get_system_stats."""

    def test_nvidia_smi_timeout_returns_empty_gpus(self) -> None:
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)):
            stats = _get_system_stats()
        assert stats["gpus"] == []

    def test_nvidia_smi_nonzero_exit_returns_no_gpus_key(self) -> None:
        # When returncode != 0, the code skips the gpus block entirely —
        # no "gpus" key is set (only the except branch sets stats["gpus"] = []).
        mock_result = Mock(returncode=1, stdout="", stderr="NVIDIA-SMI not found")
        with patch("subprocess.run", return_value=mock_result):
            stats = _get_system_stats()
        assert "gpus" not in stats

    def test_nvidia_smi_malformed_csv_returns_empty_gpus(self) -> None:
        mock_result = Mock(returncode=0, stdout="garbage,only_two\n")
        with patch("subprocess.run", return_value=mock_result):
            stats = _get_system_stats()
        assert stats["gpus"] == []

    def test_nvidia_smi_non_numeric_values_returns_empty_gpus(self) -> None:
        mock_result = Mock(returncode=0, stdout="N/A, N/A, N/A\n")
        with patch("subprocess.run", return_value=mock_result):
            stats = _get_system_stats()
        assert stats["gpus"] == []

    def test_nvidia_smi_file_not_found_returns_empty_gpus(self) -> None:
        with patch("subprocess.run",
                   side_effect=FileNotFoundError("nvidia-smi not found")):
            stats = _get_system_stats()
        assert stats["gpus"] == []

    def test_nvidia_smi_multi_gpu_parsed_correctly(self) -> None:
        mock_result = Mock(
            returncode=0,
            stdout="85, 4096, 8192\n42, 2048, 8192\n",
        )
        with patch("subprocess.run", return_value=mock_result):
            stats = _get_system_stats()
        assert len(stats["gpus"]) == 2
        assert stats["gpus"][0] == {"util_percent": 85, "mem_used_mb": 4096, "mem_total_mb": 8192}
        assert stats["gpus"][1] == {"util_percent": 42, "mem_used_mb": 2048, "mem_total_mb": 8192}


class TestDbAccessibleEdgeCases:
    """Edge cases for _db_accessible beyond what test_healthz covers."""

    def test_db_exists_but_no_schema_version_table(self, tmp_path: Path) -> None:
        """A SQLite DB that exists but wasn't init'd by keisei."""
        db_file = str(tmp_path / "foreign.db")
        conn = sqlite3.connect(db_file)
        conn.execute("CREATE TABLE other (id INTEGER)")
        conn.commit()
        conn.close()
        assert _db_accessible(db_file) is False

    def test_db_file_is_not_sqlite(self, tmp_path: Path) -> None:
        """A non-SQLite file at the path."""
        bad_file = tmp_path / "not_a_db.db"
        bad_file.write_text("this is not a database")
        assert _db_accessible(str(bad_file)) is False

    def test_db_path_is_directory(self, tmp_path: Path) -> None:
        """Path points to a directory, not a file."""
        assert _db_accessible(str(tmp_path)) is False

    def test_db_accessible_with_valid_db(self, tmp_path: Path) -> None:
        """A properly initialized keisei DB should be accessible."""
        db_file = str(tmp_path / "good.db")
        init_db(db_file)
        assert _db_accessible(db_file) is True


@pytest.fixture
def edge_db(tmp_path: Path) -> str:
    """Initialized DB with fresh heartbeat for edge-case tests."""
    path = str(tmp_path / "edge.db")
    init_db(path)
    write_training_state(path, {
        "config_json": "{}",
        "display_name": "TestBot",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
    })
    update_heartbeat(path)
    return path


class TestWSDbErrorDuringPoll:
    """The WebSocket should close cleanly when the DB fails mid-poll,
    not crash or hang."""

    @pytest.mark.skip(
        reason=(
            "Starlette sync TestClient cannot reliably exercise except* with TaskGroup: "
            "the background tasks (TaskGroup + asyncio event loop) keep running inside "
            "the sync thread, causing ws.receive_json() to block indefinitely after the "
            "DB error triggers a disconnect. The error-handling path (ws_endpoint lines "
            "118-131 in app.py) is only exercisable via a real async client (e.g. "
            "httpx.AsyncClient with anyio). See test_server_gaps.py lines 279-286."
        )
    )
    def test_ws_closes_on_db_read_failure(self, edge_db: str) -> None:
        """Simulate DB failure after init by making read_metrics_since raise."""
        app = create_app(edge_db)
        call_count = 0

        original_read_metrics = __import__(
            "keisei.db", fromlist=["read_metrics_since"]
        ).read_metrics_since

        def failing_read_metrics(db_path, since_id, limit=500):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise sqlite3.OperationalError("database is locked")
            return original_read_metrics(db_path, since_id, limit)

        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01), \
             patch("keisei.server.app.WS_PING_INTERVAL_S", 999), \
             patch("keisei.server.app.read_metrics_since", failing_read_metrics):
            client = TestClient(app)
            with client.websocket_connect("/ws") as ws:
                init_msg = ws.receive_json()
                assert init_msg["type"] == "init"

                try:
                    for _ in range(5):
                        data = ws.receive_json(mode="text")
                        if data.get("type") == "ping":
                            continue
                except Exception:
                    pass  # Connection closed — expected outcome


class TestMainEntryPoint:
    """Tests for the keisei-serve CLI entry point."""

    def test_main_passes_config_to_create_app(self, tmp_path: Path) -> None:
        """main() loads config and passes db_path to create_app."""
        config_file = tmp_path / "test.toml"
        config_file.write_text(
            '[display]\ndb_path = "/tmp/test_keisei.db"\nmoves_per_minute = 60\n'
        )

        mock_app = MagicMock()

        with patch("sys.argv", ["keisei-serve", "--config", str(config_file),
                                "--host", "0.0.0.0", "--port", "9999"]), \
             patch("keisei.server.app.create_app", return_value=mock_app) as mock_create, \
             patch("uvicorn.run") as mock_uvicorn_run:
            from keisei.server.app import main
            main()

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert call_args[0][0] == "/tmp/test_keisei.db"

        mock_uvicorn_run.assert_called_once_with(
            mock_app, host="0.0.0.0", port=9999
        )

    def test_main_uses_default_host_and_port(self, tmp_path: Path) -> None:
        """Without --host/--port, defaults to 127.0.0.1:8000."""
        config_file = tmp_path / "test.toml"
        config_file.write_text(
            '[display]\ndb_path = "/tmp/test_keisei.db"\nmoves_per_minute = 60\n'
        )

        mock_app = MagicMock()

        with patch("sys.argv", ["keisei-serve", "--config", str(config_file)]), \
             patch("keisei.server.app.create_app", return_value=mock_app), \
             patch("uvicorn.run") as mock_uvicorn_run:
            from keisei.server.app import main
            main()

        mock_uvicorn_run.assert_called_once_with(
            mock_app, host="127.0.0.1", port=8000
        )

    def test_main_missing_config_flag_exits(self) -> None:
        """--config is required; omitting it should cause SystemExit."""
        with patch("sys.argv", ["keisei-serve"]), \
             pytest.raises(SystemExit):
            from keisei.server.app import main
            main()
