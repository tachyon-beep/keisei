"""Edge-case tests for keisei.server.app functions not covered by
test_server.py or test_server_gaps.py."""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from keisei.server.app import _db_accessible, _get_system_stats


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
        from keisei.db import init_db
        db_file = str(tmp_path / "good.db")
        init_db(db_file)
        assert _db_accessible(db_file) is True
