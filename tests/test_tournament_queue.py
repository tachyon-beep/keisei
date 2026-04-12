"""Tests for tournament pairing queue, worker heartbeat, and dynamic claim ops."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from keisei.db import _connect, init_db
from keisei.training.tournament_queue import (
    ClaimedPairing,
    claim_dynamic_update,
    claim_next_pairing,
    enqueue_pairings,
    get_active_queue_depth,
    get_round_status,
    get_worker_health,
    mark_pairing_done,
    release_dynamic_update,
    reset_stale_playing,
    write_worker_heartbeat,
)


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def _seed_entries(db: str, n: int = 3) -> list[int]:
    """Insert minimal league_entries rows and return their IDs."""
    conn = _connect(db)
    try:
        ids = []
        for i in range(n):
            cur = conn.execute(
                "INSERT INTO league_entries "
                "(display_name, architecture, model_params, "
                "checkpoint_path, elo_rating, created_epoch, "
                "games_played, created_at, flavour_facts, "
                "role, status, update_count) "
                "VALUES (?, 'resnet', '{}', ?, 1000.0, ?, 0, "
                "'2026-01-01', '[]', 'dynamic', 'active', 0)",
                (f"entry{i}", f"/p/{i}.pt", i),
            )
            ids.append(cur.lastrowid)
        conn.commit()
        return ids
    finally:
        conn.close()


# ── Schema tests ─────────────────────────────────────────────────────


class TestTournamentQueueSchema:
    def test_schema_version_is_4(self, db: str) -> None:
        conn = _connect(db)
        try:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row["version"] == 4
        finally:
            conn.close()

    def test_pairing_queue_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute(
                "SELECT id, round_id, entry_a_id, entry_b_id, games_target, "
                "status, worker_id, claimed_at, completed_at, enqueued_epoch, "
                "priority FROM tournament_pairing_queue LIMIT 0"
            )
        finally:
            conn.close()

    def test_pairing_queue_indexes_exist(self, db: str) -> None:
        conn = _connect(db)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='tournament_pairing_queue'"
            ).fetchall()
            names = {r["name"] for r in rows}
            assert "idx_pairing_queue_pending" in names
            assert "idx_pairing_queue_round" in names
        finally:
            conn.close()

    def test_worker_heartbeat_table_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            conn.execute(
                "SELECT worker_id, pid, device, last_seen, "
                "pairings_done FROM tournament_worker_heartbeat LIMIT 0"
            )
        finally:
            conn.close()

    def test_league_entries_has_dynamic_update_worker(self, db: str) -> None:
        conn = _connect(db)
        try:
            cols = {
                r["name"] for r in conn.execute(
                    "PRAGMA table_info(league_entries)"
                ).fetchall()
            }
            assert "dynamic_update_worker" in cols
        finally:
            conn.close()


# ── Pairing queue ops ────────────────────────────────────────────────


class TestPairingQueueOps:
    def test_enqueue_inserts_rows(self, db: str) -> None:
        ids = _seed_entries(db)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3)],
        )
        conn = _connect(db)
        try:
            rows = conn.execute(
                "SELECT * FROM tournament_pairing_queue ORDER BY id"
            ).fetchall()
            assert len(rows) == 2
            assert rows[0]["round_id"] == 1
            assert rows[0]["status"] == "pending"
            assert rows[0]["games_target"] == 3
            assert rows[0]["enqueued_epoch"] == 5
        finally:
            conn.close()

    def test_enqueue_empty_is_noop(self, db: str) -> None:
        enqueue_pairings(db, round_id=1, epoch=0, pairings=[])
        assert get_active_queue_depth(db) == 0

    def test_enqueue_with_priorities(self, db: str) -> None:
        ids = _seed_entries(db)
        enqueue_pairings(
            db, round_id=1, epoch=0,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3)],
            priorities=[1.0, 5.0],
        )
        claimed = claim_next_pairing(db, worker_id="w0")
        assert claimed is not None
        assert claimed.entry_a_id == ids[0]
        assert claimed.entry_b_id == ids[2]

    def test_claim_returns_one(self, db: str) -> None:
        ids = _seed_entries(db)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3)],
        )
        claimed = claim_next_pairing(db, worker_id="w0")
        assert claimed is not None
        assert claimed.worker_id == "w0"
        assert claimed.status == "playing"
        claimed2 = claim_next_pairing(db, worker_id="w1")
        assert claimed2 is not None
        assert claimed2.id != claimed.id
        assert claim_next_pairing(db, worker_id="w2") is None

    def test_claim_is_atomic_under_concurrent_workers(self, db: str) -> None:
        ids = _seed_entries(db)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 1) for _ in range(100)],
        )
        claimed_by: dict[str, list[int]] = {f"w{i}": [] for i in range(4)}
        lock = threading.Lock()

        def worker_loop(wid: str) -> None:
            while True:
                claim = claim_next_pairing(db, worker_id=wid)
                if claim is None:
                    return
                with lock:
                    claimed_by[wid].append(claim.id)

        threads = [threading.Thread(target=worker_loop, args=(w,)) for w in claimed_by]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_ids = [i for lst in claimed_by.values() for i in lst]
        assert len(all_ids) == 100
        assert len(set(all_ids)) == 100

    def test_mark_done_updates_status(self, db: str) -> None:
        ids = _seed_entries(db)
        enqueue_pairings(db, round_id=1, epoch=5, pairings=[(ids[0], ids[1], 3)])
        claim = claim_next_pairing(db, worker_id="w0")
        assert claim is not None
        mark_pairing_done(db, claim.id, status="done")
        conn = _connect(db)
        try:
            row = conn.execute(
                "SELECT status, completed_at FROM tournament_pairing_queue WHERE id = ?",
                (claim.id,),
            ).fetchone()
            assert row["status"] == "done"
            assert row["completed_at"] is not None
        finally:
            conn.close()

    def test_mark_done_rejects_invalid_status(self, db: str) -> None:
        with pytest.raises(ValueError, match="invalid status"):
            mark_pairing_done(db, 1, status="nonsense")

    def test_get_round_status(self, db: str) -> None:
        ids = _seed_entries(db)
        enqueue_pairings(
            db, round_id=7, epoch=5,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3), (ids[1], ids[2], 3)],
        )
        claim = claim_next_pairing(db, worker_id="w0")
        assert claim is not None
        mark_pairing_done(db, claim.id, status="done")
        claim2 = claim_next_pairing(db, worker_id="w0")
        assert claim2 is not None
        status = get_round_status(db, round_id=7)
        assert status == {"pending": 1, "playing": 1, "done": 1}

    def test_get_active_queue_depth(self, db: str) -> None:
        ids = _seed_entries(db)
        assert get_active_queue_depth(db) == 0
        enqueue_pairings(
            db, round_id=1, epoch=0,
            pairings=[(ids[0], ids[1], 1) for _ in range(5)],
        )
        assert get_active_queue_depth(db) == 5
        claim = claim_next_pairing(db, worker_id="w0")
        assert claim is not None
        assert get_active_queue_depth(db) == 5
        mark_pairing_done(db, claim.id, status="done")
        assert get_active_queue_depth(db) == 4

    def test_reset_stale_playing_by_worker(self, db: str) -> None:
        ids = _seed_entries(db)
        enqueue_pairings(db, round_id=1, epoch=0, pairings=[(ids[0], ids[1], 1)])
        claim = claim_next_pairing(db, worker_id="w0")
        assert claim is not None
        reset_count = reset_stale_playing(db, worker_id="w0")
        assert reset_count == 1
        reclaimed = claim_next_pairing(db, worker_id="w1")
        assert reclaimed is not None
        assert reclaimed.worker_id == "w1"

    def test_reset_stale_playing_all(self, db: str) -> None:
        ids = _seed_entries(db)
        enqueue_pairings(
            db, round_id=1, epoch=0,
            pairings=[(ids[0], ids[1], 1), (ids[0], ids[2], 1)],
        )
        claim_next_pairing(db, worker_id="w0")
        claim_next_pairing(db, worker_id="w1")
        reset_count = reset_stale_playing(db)
        assert reset_count == 2
        assert get_active_queue_depth(db) == 2


# ── Worker heartbeat ─────────────────────────────────────────────────


class TestWorkerHeartbeat:
    def test_write_and_read(self, db: str) -> None:
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1")
        health = get_worker_health(db, stale_after_seconds=60)
        assert len(health) == 1
        assert health[0].worker_id == "w0"
        assert health[0].pid == 1234
        assert health[0].device == "cuda:1"

    def test_upserts_not_duplicates(self, db: str) -> None:
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1")
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1")
        conn = _connect(db)
        try:
            row = conn.execute(
                "SELECT COUNT(*) as n FROM tournament_worker_heartbeat"
            ).fetchone()
            assert row["n"] == 1
        finally:
            conn.close()

    def test_pairings_done_tracked(self, db: str) -> None:
        write_worker_heartbeat(
            db, worker_id="w0", pid=1234, device="cuda:1", pairings_done=5,
        )
        health = get_worker_health(db, stale_after_seconds=60)
        assert health[0].pairings_done == 5

    def test_stale_workers_excluded(self, db: str) -> None:
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1")
        time.sleep(1.1)
        health = get_worker_health(db, stale_after_seconds=1)
        assert len(health) == 0


# ── DynamicTrainer claim ─────────────────────────────────────────────


class TestDynamicUpdateClaim:
    def test_claim_returns_true_when_available(self, db: str) -> None:
        ids = _seed_entries(db, 1)
        assert claim_dynamic_update(db, entry_id=ids[0], worker_id="w0") is True

    def test_claim_returns_false_when_held(self, db: str) -> None:
        ids = _seed_entries(db, 1)
        assert claim_dynamic_update(db, entry_id=ids[0], worker_id="w0") is True
        assert claim_dynamic_update(db, entry_id=ids[0], worker_id="w1") is False

    def test_release_allows_reclaim(self, db: str) -> None:
        ids = _seed_entries(db, 1)
        assert claim_dynamic_update(db, entry_id=ids[0], worker_id="w0") is True
        release_dynamic_update(db, entry_id=ids[0], worker_id="w0")
        assert claim_dynamic_update(db, entry_id=ids[0], worker_id="w1") is True

    def test_release_noop_when_wrong_owner(self, db: str) -> None:
        ids = _seed_entries(db, 1)
        claim_dynamic_update(db, entry_id=ids[0], worker_id="w0")
        release_dynamic_update(db, entry_id=ids[0], worker_id="w1")
        assert claim_dynamic_update(db, entry_id=ids[0], worker_id="w2") is False

    def test_concurrent_claim_single_winner(self, db: str) -> None:
        ids = _seed_entries(db, 1)
        winners: list[str] = []
        lock = threading.Lock()

        def attempt(wid: str) -> None:
            got = claim_dynamic_update(db, entry_id=ids[0], worker_id=wid)
            if got:
                with lock:
                    winners.append(wid)

        threads = [threading.Thread(target=attempt, args=(f"w{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(winners) == 1
