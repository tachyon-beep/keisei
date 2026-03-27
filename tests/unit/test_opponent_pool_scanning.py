"""Tests for OpponentPool directory scanning."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pytest

from keisei.evaluation.opponents.opponent_pool import OpponentPool


@pytest.fixture
def pool() -> OpponentPool:
    return OpponentPool(pool_size=20)


class TestScanFindsCheckpoints:
    def test_scan_finds_checkpoints(self, pool: OpponentPool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                (Path(tmpdir) / f"checkpoint_ts{i}.pth").write_bytes(b"fake")
            added = pool.scan_directory(tmpdir)
            assert added == 3
            assert len(list(pool.get_all())) == 3


class TestScanSkipsDuplicates:
    def test_scan_skips_duplicates(self, pool: OpponentPool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "checkpoint_ts0.pth"
            ckpt.write_bytes(b"fake")
            (Path(tmpdir) / "checkpoint_ts1.pth").write_bytes(b"fake")

            pool.add_checkpoint(ckpt)
            assert len(list(pool.get_all())) == 1

            added = pool.scan_directory(tmpdir)
            assert added == 1  # only the new one
            assert len(list(pool.get_all())) == 2


class TestScanNonexistentDirectory:
    def test_scan_nonexistent_directory(self, pool: OpponentPool) -> None:
        result = pool.scan_directory("/nonexistent/path/that/does/not/exist")
        assert result == 0


class TestScanRespectsPattern:
    def test_scan_respects_pattern(self, pool: OpponentPool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "checkpoint_ts0.pth").write_bytes(b"fake")
            (Path(tmpdir) / "checkpoint_ts1.pth").write_bytes(b"fake")
            (Path(tmpdir) / "other_file.txt").write_bytes(b"fake")
            (Path(tmpdir) / "model.onnx").write_bytes(b"fake")

            added = pool.scan_directory(tmpdir)
            assert added == 2
            paths = {p.name for p in pool.get_all()}
            assert paths == {"checkpoint_ts0.pth", "checkpoint_ts1.pth"}


class TestScanSortsByMtime:
    def test_scan_sorts_by_mtime(self, pool: OpponentPool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with controlled modification times
            oldest = Path(tmpdir) / "checkpoint_ts_oldest.pth"
            middle = Path(tmpdir) / "checkpoint_ts_middle.pth"
            newest = Path(tmpdir) / "checkpoint_ts_newest.pth"

            oldest.write_bytes(b"fake")
            os.utime(oldest, (1000, 1000))

            middle.write_bytes(b"fake")
            os.utime(middle, (2000, 2000))

            newest.write_bytes(b"fake")
            os.utime(newest, (3000, 3000))

            pool.scan_directory(tmpdir)
            all_paths = list(pool.get_all())
            assert len(all_paths) == 3
            # Oldest first, newest last
            assert all_paths[0].name == "checkpoint_ts_oldest.pth"
            assert all_paths[1].name == "checkpoint_ts_middle.pth"
            assert all_paths[2].name == "checkpoint_ts_newest.pth"


class TestKnownPaths:
    def test_known_paths(self, pool: OpponentPool) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "checkpoint_ts0.pth"
            p2 = Path(tmpdir) / "checkpoint_ts1.pth"
            p1.write_bytes(b"fake")
            p2.write_bytes(b"fake")
            pool.add_checkpoint(p1)
            pool.add_checkpoint(p2)

            known = pool.known_paths()
            assert isinstance(known, set)
            assert p1.resolve() in known
            assert p2.resolve() in known
            assert len(known) == 2
