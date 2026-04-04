"""Tests for keisei.training.tournament — round-robin Elo calibration.

Covers: _generate_round pairing logic, _record_result Elo writes,
lifecycle (start/stop), bye carry-forward, DB helpers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from keisei.db import init_db
from keisei.training.league import OpponentEntry
from keisei.training.tournament import LeagueTournament

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    id: int,
    elo: float = 1000.0,
    name: str | None = None,
    checkpoint_path: str = "/fake/ckpt.pt",
) -> OpponentEntry:
    """Create an OpponentEntry for testing without touching the DB."""
    return OpponentEntry(
        id=id,
        display_name=name or f"Bot-{id}",
        architecture="resnet",
        model_params={"hidden_size": 16},
        checkpoint_path=checkpoint_path,
        elo_rating=elo,
        created_epoch=id,
        games_played=0,
        created_at="2026-04-01T00:00:00Z",
        flavour_facts=[],
    )


def _make_tournament(db_path: str, league_dir: str, **kwargs) -> LeagueTournament:
    """Create a LeagueTournament with CPU device to avoid GPU requirement."""
    defaults = dict(device="cpu", num_envs=1, games_per_match=1)
    defaults.update(kwargs)
    return LeagueTournament(db_path, league_dir, **defaults)


def _insert_entry(conn: sqlite3.Connection, entry: OpponentEntry) -> None:
    """Insert an OpponentEntry into the league_entries table."""
    conn.execute(
        """INSERT INTO league_entries
           (id, display_name, flavour_facts, architecture, model_params,
            checkpoint_path, elo_rating, created_epoch, games_played, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry.id, entry.display_name, json.dumps(entry.flavour_facts),
            entry.architecture, json.dumps(entry.model_params),
            entry.checkpoint_path, entry.elo_rating, entry.created_epoch,
            entry.games_played, entry.created_at,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tournament_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "tournament.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def league_dir(tmp_path: Path) -> str:
    d = tmp_path / "league"
    d.mkdir()
    return str(d)


# ===========================================================================
# _generate_round — pairing logic
# ===========================================================================


class TestGenerateRound:
    """Test the round-robin pairing algorithm."""

    def test_even_pool_produces_n_over_2_pairings(self, tournament_db, league_dir):
        """4 entries → 2 pairings, no bye."""
        t = _make_tournament(tournament_db, league_dir)
        entries = [_make_entry(i) for i in range(1, 5)]
        pairings, bye = t._generate_round(entries)
        assert len(pairings) == 2
        assert bye is None

    def test_odd_pool_produces_bye(self, tournament_db, league_dir):
        """3 entries → 1 pairing + 1 bye."""
        t = _make_tournament(tournament_db, league_dir)
        entries = [_make_entry(i) for i in range(1, 4)]
        pairings, bye = t._generate_round(entries)
        assert len(pairings) == 1
        assert bye is not None
        # Bye entry should not appear in any pairing
        paired_ids = {e.id for pair in pairings for e in pair}
        assert bye.id not in paired_ids

    def test_two_entries_produces_one_pairing(self, tournament_db, league_dir):
        """Minimum viable pool: 2 entries → 1 pairing."""
        t = _make_tournament(tournament_db, league_dir)
        entries = [_make_entry(1), _make_entry(2)]
        pairings, bye = t._generate_round(entries)
        assert len(pairings) == 1
        assert bye is None
        ids = {pairings[0][0].id, pairings[0][1].id}
        assert ids == {1, 2}

    def test_single_entry_produces_no_pairings(self, tournament_db, league_dir):
        """1 entry → no pairings (need at least 2 distinct IDs)."""
        t = _make_tournament(tournament_db, league_dir)
        entries = [_make_entry(1)]
        pairings, bye = t._generate_round(entries)
        assert pairings == []
        assert bye is None

    def test_empty_pool_produces_no_pairings(self, tournament_db, league_dir):
        t = _make_tournament(tournament_db, league_dir)
        pairings, bye = t._generate_round([])
        assert pairings == []
        assert bye is None

    def test_no_self_matches(self, tournament_db, league_dir):
        """No entry should be paired with itself."""
        t = _make_tournament(tournament_db, league_dir)
        entries = [_make_entry(i) for i in range(1, 9)]
        # Run multiple times since shuffle is random
        for _ in range(20):
            pairings, _ = t._generate_round(entries)
            for a, b in pairings:
                assert a.id != b.id, f"Self-match: {a.id} vs {b.id}"

    def test_duplicate_ids_filtered(self, tournament_db, league_dir):
        """If entries list has duplicate IDs (e.g. from DB restart bug),
        self-matches are filtered out."""
        t = _make_tournament(tournament_db, league_dir)
        # Two entries with the same ID
        entries = [_make_entry(1), _make_entry(1)]
        pairings, bye = t._generate_round(entries)
        # Only 1 unique ID → no valid pairings
        assert pairings == []

    def test_large_pool_all_entries_paired_or_bye(self, tournament_db, league_dir):
        """With N entries, exactly N-1 or N entries are used (depending on parity)."""
        t = _make_tournament(tournament_db, league_dir)
        entries = [_make_entry(i) for i in range(1, 11)]  # 10 entries
        pairings, bye = t._generate_round(entries)
        paired_ids = {e.id for pair in pairings for e in pair}
        if bye:
            paired_ids.add(bye.id)
        # All entries should be accounted for (paired or bye)
        assert paired_ids == {e.id for e in entries}


# ===========================================================================
# _record_result — Elo writes
# ===========================================================================


class TestRecordResult:
    """Test Elo recording and DB writes."""

    def test_elo_updates_written_correctly(self, tournament_db, league_dir):
        """After recording a result, both entries' Elo ratings should change."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        entry_a = _make_entry(1, elo=1000.0)
        entry_b = _make_entry(2, elo=1000.0)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        t = _make_tournament(tournament_db, league_dir, k_factor=32.0)
        conn = t._open_connection()

        # A wins all games
        t._record_result(conn, entry_a, entry_b, wins_a=10, wins_b=0, draws=0, epoch=5)

        row_a = conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = ?", (1,)
        ).fetchone()
        row_b = conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = ?", (2,)
        ).fetchone()

        assert row_a["elo_rating"] > 1000.0, "Winner's Elo should increase"
        assert row_b["elo_rating"] < 1000.0, "Loser's Elo should decrease"
        conn.close()

    def test_league_results_row_inserted(self, tournament_db, league_dir):
        """A league_results row should be inserted with win/loss/draw counts."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1))
        _insert_entry(conn, _make_entry(2))
        conn.close()

        t = _make_tournament(tournament_db, league_dir)
        conn = t._open_connection()
        t._record_result(
            conn, _make_entry(1), _make_entry(2),
            wins_a=5, wins_b=3, draws=2, epoch=10,
        )

        row = conn.execute("SELECT * FROM league_results").fetchone()
        assert row is not None
        assert row["wins"] == 5
        assert row["losses"] == 3
        assert row["draws"] == 2
        assert row["epoch"] == 10
        conn.close()

    def test_elo_history_rows_inserted(self, tournament_db, league_dir):
        """Both entries should get elo_history rows at the given epoch."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1))
        _insert_entry(conn, _make_entry(2))
        conn.close()

        t = _make_tournament(tournament_db, league_dir)
        conn = t._open_connection()
        t._record_result(
            conn, _make_entry(1), _make_entry(2),
            wins_a=5, wins_b=5, draws=0, epoch=7,
        )

        rows = conn.execute(
            "SELECT * FROM elo_history WHERE epoch = ?", (7,)
        ).fetchall()
        assert len(rows) == 2
        entry_ids = {r["entry_id"] for r in rows}
        assert entry_ids == {1, 2}
        conn.close()

    def test_games_played_incremented(self, tournament_db, league_dir):
        """games_played should increase by total games for both entries."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1))
        _insert_entry(conn, _make_entry(2))
        conn.close()

        t = _make_tournament(tournament_db, league_dir)
        conn = t._open_connection()
        t._record_result(
            conn, _make_entry(1), _make_entry(2),
            wins_a=3, wins_b=4, draws=1, epoch=1,
        )

        row_a = conn.execute(
            "SELECT games_played FROM league_entries WHERE id = 1"
        ).fetchone()
        row_b = conn.execute(
            "SELECT games_played FROM league_entries WHERE id = 2"
        ).fetchone()
        assert row_a["games_played"] == 8  # 3 + 4 + 1
        assert row_b["games_played"] == 8
        conn.close()

    def test_evicted_entry_graceful_skip(self, tournament_db, league_dir):
        """If an entry was evicted mid-match, _record_result should not crash."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        # Only insert entry 1, not entry 2
        _insert_entry(conn, _make_entry(1))
        conn.close()

        t = _make_tournament(tournament_db, league_dir)
        conn = t._open_connection()
        # Should not raise — entry_b doesn't exist
        t._record_result(
            conn, _make_entry(1), _make_entry(2),
            wins_a=5, wins_b=0, draws=0, epoch=1,
        )
        # No league_results row should be written
        row = conn.execute("SELECT COUNT(*) as cnt FROM league_results").fetchone()
        assert row["cnt"] == 0
        conn.close()

    def test_draw_result_elo_changes_minimal(self, tournament_db, league_dir):
        """Equal-rated opponents drawing should have near-zero Elo change."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1, elo=1000.0))
        _insert_entry(conn, _make_entry(2, elo=1000.0))
        conn.close()

        t = _make_tournament(tournament_db, league_dir, k_factor=16.0)
        conn = t._open_connection()
        t._record_result(
            conn, _make_entry(1), _make_entry(2),
            wins_a=5, wins_b=5, draws=0, epoch=1,
        )

        row_a = conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = 1"
        ).fetchone()
        # 50/50 result between equal-rated players → minimal Elo change
        assert abs(row_a["elo_rating"] - 1000.0) < 1.0
        conn.close()


# ===========================================================================
# DB helpers
# ===========================================================================


class TestDBHelpers:
    """Test _current_epoch, _load_entries, _open_connection."""

    def test_current_epoch_from_training_state(self, tournament_db, league_dir):
        """_current_epoch should read from training_state table."""
        from keisei.db import write_training_state

        write_training_state(tournament_db, {
            "config_json": "{}",
            "display_name": "Test",
            "model_arch": "resnet",
            "algorithm_name": "ppo",
            "started_at": "2026-04-01T00:00:00Z",
            "current_epoch": 42,
        })

        t = _make_tournament(tournament_db, league_dir)
        conn = t._open_connection()
        assert t._current_epoch(conn) == 42
        conn.close()

    def test_current_epoch_returns_zero_when_no_state(self, tournament_db, league_dir):
        """_current_epoch should return 0 when no training_state row exists."""
        t = _make_tournament(tournament_db, league_dir)
        conn = t._open_connection()
        assert t._current_epoch(conn) == 0
        conn.close()

    def test_load_entries_returns_opponent_entries(self, tournament_db, league_dir):
        """_load_entries should return OpponentEntry objects from the DB."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1, elo=1200.0))
        _insert_entry(conn, _make_entry(2, elo=800.0))
        conn.close()

        t = _make_tournament(tournament_db, league_dir)
        conn = t._open_connection()
        entries = t._load_entries(conn)
        assert len(entries) == 2
        # Should be ordered by elo_rating DESC
        assert entries[0].elo_rating == 1200.0
        assert entries[1].elo_rating == 800.0
        conn.close()


# ===========================================================================
# Lifecycle — start/stop
# ===========================================================================


class TestLifecycle:
    """Test thread start/stop behavior."""

    def test_start_stop_lifecycle(self, tournament_db, league_dir):
        """start() should spawn a thread; stop() should join it."""
        t = _make_tournament(tournament_db, league_dir)
        assert not t.is_running

        # Mock _run_loop to avoid needing VecEnv
        with patch.object(t, "_run_loop", side_effect=lambda: t._stop_event.wait()):
            t.start()
            assert t.is_running
            t.stop(timeout=2.0)
            assert not t.is_running

    def test_double_start_is_idempotent(self, tournament_db, league_dir):
        """Calling start() twice should not spawn a second thread."""
        t = _make_tournament(tournament_db, league_dir)

        with patch.object(t, "_run_loop", side_effect=lambda: t._stop_event.wait()):
            t.start()
            thread1 = t._thread
            t.start()  # should be a no-op
            assert t._thread is thread1
            t.stop(timeout=2.0)

    def test_stop_without_start(self, tournament_db, league_dir):
        """stop() when no thread is running should not raise."""
        t = _make_tournament(tournament_db, league_dir)
        t.stop()  # no-op, should not raise

    def test_is_running_reflects_thread_state(self, tournament_db, league_dir):
        """is_running should be False after the thread naturally exits."""
        t = _make_tournament(tournament_db, league_dir)
        exit_event = threading.Event()

        def quick_exit():
            exit_event.set()  # signal that thread started
            # Exit immediately

        with patch.object(t, "_run_loop", side_effect=quick_exit):
            t.start()
            exit_event.wait(timeout=2.0)
            # Give the thread time to actually finish
            t._thread.join(timeout=2.0)
            assert not t.is_running


# ===========================================================================
# Bye carry-forward (integration-level, but no VecEnv)
# ===========================================================================


class TestByeCarryForward:
    """Test that bye entries get their Elo recorded in elo_history."""

    def test_bye_elo_history_written(self, tournament_db, league_dir):
        """When a bye entry exists, its current Elo should be inserted
        into elo_history at the current epoch."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        bye_entry = _make_entry(3, elo=1100.0)
        _insert_entry(conn, bye_entry)
        conn.close()

        t = _make_tournament(tournament_db, league_dir)
        conn = t._open_connection()

        # Simulate the bye carry-forward from _run_loop
        epoch = 10
        row = conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = ?", (bye_entry.id,)
        ).fetchone()
        assert row is not None
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (?, ?, ?)",
            (bye_entry.id, epoch, row["elo_rating"]),
        )
        conn.commit()

        history = conn.execute(
            "SELECT * FROM elo_history WHERE entry_id = ? AND epoch = ?",
            (bye_entry.id, epoch),
        ).fetchone()
        assert history is not None
        assert history["elo_rating"] == 1100.0
        conn.close()
