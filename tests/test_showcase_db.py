"""Tests for showcase database tables and operations."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from keisei.db import init_db, _connect


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


class TestShowcaseSchema:
    def test_schema_version_is_3(self, db: str) -> None:
        conn = _connect(db)
        try:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row["version"] == 3
        finally:
            conn.close()

    def test_showcase_queue_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute("SELECT id, entry_id_1, entry_id_2, speed, status, requested_at, started_at, completed_at FROM showcase_queue LIMIT 0")
        finally:
            conn.close()

    def test_showcase_games_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute("SELECT id, queue_id, entry_id_black, entry_id_white, elo_black, elo_white, name_black, name_white, status, abandon_reason, started_at, completed_at, total_ply FROM showcase_games LIMIT 0")
        finally:
            conn.close()

    def test_showcase_moves_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute("SELECT id, game_id, ply, action_index, usi_notation, board_json, hands_json, current_player, in_check, value_estimate, top_candidates, move_time_ms, created_at FROM showcase_moves LIMIT 0")
        finally:
            conn.close()

    def test_showcase_heartbeat_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute("SELECT id, last_heartbeat, runner_pid FROM showcase_heartbeat LIMIT 0")
        finally:
            conn.close()

    def test_showcase_queue_one_running_constraint(self, db: str) -> None:
        """Only one queue entry can have status='running' at a time."""
        conn = _connect(db)
        try:
            conn.execute(
                "INSERT INTO showcase_queue (entry_id_1, entry_id_2, speed, status, requested_at) VALUES ('a', 'b', 'normal', 'running', '2026-01-01T00:00:00Z')"
            )
            conn.commit()
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO showcase_queue (entry_id_1, entry_id_2, speed, status, requested_at) VALUES ('c', 'd', 'normal', 'running', '2026-01-01T00:00:00Z')"
                )
                conn.commit()
        finally:
            conn.close()


from keisei.showcase.db_ops import (
    queue_match,
    claim_next_match,
    read_queue,
    cancel_match,
    update_queue_speed,
    create_showcase_game,
    write_showcase_move,
    read_showcase_moves_since,
    read_active_showcase_game,
    mark_game_completed,
    mark_game_abandoned,
    write_heartbeat,
    read_heartbeat,
    cleanup_orphaned_games,
)


class TestQueueOperations:
    def test_queue_match_creates_pending_row(self, db: str) -> None:
        qid = queue_match(db, "entry-1", "entry-2", "normal")
        assert qid > 0
        rows = read_queue(db)
        assert len(rows) == 1
        assert rows[0]["status"] == "pending"
        assert rows[0]["entry_id_1"] == "entry-1"

    def test_claim_next_match_returns_pending(self, db: str) -> None:
        queue_match(db, "entry-1", "entry-2", "normal")
        claimed = claim_next_match(db)
        assert claimed is not None
        assert claimed["status"] == "running"

    def test_claim_next_match_returns_none_when_empty(self, db: str) -> None:
        claimed = claim_next_match(db)
        assert claimed is None

    def test_claim_next_match_is_atomic(self, db: str) -> None:
        """Second claim returns None — first claim took the only row."""
        queue_match(db, "entry-1", "entry-2", "normal")
        first = claim_next_match(db)
        second = claim_next_match(db)
        assert first is not None
        assert second is None

    def test_cancel_match(self, db: str) -> None:
        qid = queue_match(db, "entry-1", "entry-2", "normal")
        cancel_match(db, qid)
        # read_queue excludes cancelled rows, so verify via direct DB query
        conn = _connect(db)
        try:
            row = conn.execute("SELECT status FROM showcase_queue WHERE id = ?", (qid,)).fetchone()
            assert row["status"] == "cancelled"
        finally:
            conn.close()
        # read_queue should return empty (cancelled excluded)
        assert len(read_queue(db)) == 0

    def test_update_queue_speed(self, db: str) -> None:
        qid = queue_match(db, "entry-1", "entry-2", "normal")
        claim_next_match(db)
        update_queue_speed(db, qid, "fast")
        rows = read_queue(db)
        assert rows[0]["speed"] == "fast"


class TestGameOperations:
    def test_create_and_read_game(self, db: str) -> None:
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="ModelA", name_white="ModelB",
        )
        game = read_active_showcase_game(db)
        assert game is not None
        assert game["id"] == game_id
        assert game["status"] == "in_progress"

    def test_write_and_read_moves(self, db: str) -> None:
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )
        write_showcase_move(db, game_id=game_id, ply=1, action_index=42,
            usi_notation="7g7f", board_json="[]", hands_json="{}",
            current_player="white", in_check=False, value_estimate=0.52,
            top_candidates='[{"usi":"7g7f","probability":0.3}]', move_time_ms=15)
        moves = read_showcase_moves_since(db, game_id, since_ply=0)
        assert len(moves) == 1
        assert moves[0]["ply"] == 1
        assert moves[0]["usi_notation"] == "7g7f"

    def test_read_moves_since_filters_by_ply(self, db: str) -> None:
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )
        for ply in range(1, 6):
            write_showcase_move(db, game_id=game_id, ply=ply, action_index=ply,
                usi_notation=f"move{ply}", board_json="[]", hands_json="{}",
                current_player="black" if ply % 2 == 0 else "white",
                in_check=False, value_estimate=0.5, top_candidates="[]", move_time_ms=10)
        moves = read_showcase_moves_since(db, game_id, since_ply=3)
        assert len(moves) == 2
        assert moves[0]["ply"] == 4

    def test_mark_game_completed(self, db: str) -> None:
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )
        mark_game_completed(db, game_id, "black_win", total_ply=50)
        game = read_active_showcase_game(db)
        assert game is None  # no active game

    def test_mark_game_abandoned(self, db: str) -> None:
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )
        mark_game_abandoned(db, game_id, "crash_recovery")
        game = read_active_showcase_game(db)
        assert game is None


class TestHeartbeat:
    def test_write_and_read_heartbeat(self, db: str) -> None:
        write_heartbeat(db, pid=12345)
        hb = read_heartbeat(db)
        assert hb is not None
        assert hb["runner_pid"] == 12345
        assert hb["last_heartbeat"] is not None

    def test_read_heartbeat_returns_none_when_empty(self, db: str) -> None:
        hb = read_heartbeat(db)
        assert hb is None


class TestCrashRecovery:
    def test_cleanup_orphaned_games(self, db: str) -> None:
        qid = queue_match(db, "e1", "e2", "normal")
        claim_next_match(db)
        create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )
        # Simulate crash: game is in_progress, queue is running
        count = cleanup_orphaned_games(db)
        assert count == 1
        game = read_active_showcase_game(db)
        assert game is None


import concurrent.futures
import threading


class TestSQLiteConcurrency:
    """Verify showcase DB operations work under concurrent access."""

    def test_concurrent_move_writes(self, db: str) -> None:
        """Multiple threads writing moves should not lose data."""
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )

        errors: list[Exception] = []

        def write_move(ply: int) -> None:
            try:
                write_showcase_move(
                    db, game_id=game_id, ply=ply, action_index=ply,
                    usi_notation=f"move{ply}", board_json="[]", hands_json="{}",
                    current_player="black" if ply % 2 == 0 else "white",
                    in_check=False, value_estimate=0.5,
                    top_candidates="[]", move_time_ms=10,
                )
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(write_move, i) for i in range(1, 21)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        moves = read_showcase_moves_since(db, game_id, since_ply=0)
        assert len(moves) == 20

    def test_concurrent_queue_claim(self, db: str) -> None:
        """Only one thread should successfully claim a match."""
        queue_match(db, "e1", "e2", "normal")

        results: list[dict | None] = []
        lock = threading.Lock()

        def try_claim() -> None:
            result = claim_next_match(db)
            with lock:
                results.append(result)

        threads = [threading.Thread(target=try_claim) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        claimed = [r for r in results if r is not None]
        assert len(claimed) == 1, f"Expected 1 claim, got {len(claimed)}"

    def test_read_during_write(self, db: str) -> None:
        """Reads should not block during writes (WAL mode)."""
        qid = queue_match(db, "e1", "e2", "normal")
        game_id = create_showcase_game(
            db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B",
        )

        for ply in range(1, 6):
            write_showcase_move(
                db, game_id=game_id, ply=ply, action_index=ply,
                usi_notation=f"move{ply}", board_json="[]", hands_json="{}",
                current_player="white", in_check=False, value_estimate=0.5,
                top_candidates="[]", move_time_ms=10,
            )

        read_errors: list[Exception] = []

        def concurrent_read() -> None:
            try:
                for _ in range(10):
                    read_showcase_moves_since(db, game_id, since_ply=0)
            except Exception as e:
                read_errors.append(e)

        def concurrent_write() -> None:
            try:
                for ply in range(6, 16):
                    write_showcase_move(
                        db, game_id=game_id, ply=ply, action_index=ply,
                        usi_notation=f"move{ply}", board_json="[]", hands_json="{}",
                        current_player="black", in_check=False, value_estimate=0.5,
                        top_candidates="[]", move_time_ms=10,
                    )
            except Exception as e:
                read_errors.append(e)

        t_read = threading.Thread(target=concurrent_read)
        t_write = threading.Thread(target=concurrent_write)
        t_read.start()
        t_write.start()
        t_read.join()
        t_write.join()

        assert len(read_errors) == 0
