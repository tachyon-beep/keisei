"""Tests for keisei.training.tournament — round-robin Elo calibration.

Covers: lifecycle (start/stop), _current_epoch, _load_model, _run_loop
integration with OpponentStore and MatchScheduler.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from keisei.config import ConcurrencyConfig, MatchSchedulerConfig
from keisei.db import init_db
from keisei.training.match_scheduler import MatchScheduler
from keisei.training.concurrent_matches import ConcurrentMatchPool
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role, compute_elo_update
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


def _make_scheduler() -> MatchScheduler:
    """Create a MatchScheduler with default config."""
    config = MatchSchedulerConfig()
    return MatchScheduler(config)


def _make_tournament(store: OpponentStore, scheduler: MatchScheduler | None = None, **kwargs) -> LeagueTournament:
    """Create a LeagueTournament with CPU device to avoid GPU requirement."""
    if scheduler is None:
        scheduler = _make_scheduler()
    defaults: dict[str, object] = dict(device="cpu", num_envs=1, games_per_match=1)
    defaults.update(kwargs)
    return LeagueTournament(store=store, scheduler=scheduler, **defaults)  # type: ignore[arg-type]


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


@pytest.fixture
def store(tournament_db: str, league_dir: str) -> OpponentStore:
    s = OpponentStore(tournament_db, league_dir)
    yield s
    s.close()


# ===========================================================================
# MatchScheduler.generate_round — pairing logic (formerly _generate_round)
# ===========================================================================


class TestGenerateRound:
    """Test the round-robin pairing algorithm via MatchScheduler."""

    def test_even_pool_produces_full_round_robin(self):
        """4 entries → 6 pairings (full round-robin, N*(N-1)/2)."""
        scheduler = _make_scheduler()
        entries = [_make_entry(i) for i in range(1, 5)]
        pairings = scheduler.generate_round(entries)
        assert len(pairings) == 6  # 4*3/2

    def test_two_entries_produces_one_pairing(self):
        """Minimum viable pool: 2 entries → 1 pairing."""
        scheduler = _make_scheduler()
        entries = [_make_entry(1), _make_entry(2)]
        pairings = scheduler.generate_round(entries)
        assert len(pairings) == 1
        ids = {pairings[0][0].id, pairings[0][1].id}
        assert ids == {1, 2}

    def test_single_entry_produces_no_pairings(self):
        """1 entry → no pairings."""
        scheduler = _make_scheduler()
        entries = [_make_entry(1)]
        pairings = scheduler.generate_round(entries)
        assert pairings == []

    def test_empty_pool_produces_no_pairings(self):
        scheduler = _make_scheduler()
        pairings = scheduler.generate_round([])
        assert pairings == []

    def test_no_self_matches(self):
        """No entry should be paired with itself."""
        scheduler = _make_scheduler()
        entries = [_make_entry(i) for i in range(1, 9)]
        pairings = scheduler.generate_round(entries)
        for a, b in pairings:
            assert a.id != b.id, f"Self-match: {a.id} vs {b.id}"

    def test_large_pool_all_pairs_covered(self):
        """With N entries, exactly N*(N-1)/2 pairings are generated."""
        scheduler = _make_scheduler()
        entries = [_make_entry(i) for i in range(1, 11)]  # 10 entries
        pairings = scheduler.generate_round(entries)
        assert len(pairings) == 45  # 10*9/2


# ===========================================================================
# Store-based result recording (replaces _record_result tests)
# ===========================================================================


class TestRecordResult:
    """Test Elo recording via OpponentStore."""

    def test_elo_updates_via_store(self, store, tournament_db):
        """Full pipeline: compute_elo_update → record_result → update_elo → verify DB."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        entry_a = _make_entry(1, elo=1000.0)
        entry_b = _make_entry(2, elo=1000.0)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        # Compute Elo update: A wins all games
        result_score = 1.0  # A wins everything
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=result_score, k=32.0)

        # Verify compute_elo_update produced meaningful deltas
        assert new_a > 1000.0
        assert new_b < 1000.0

        store.record_result(
            epoch=5, learner_id=1, opponent_id=2,
            wins=10, losses=0, draws=0,
            elo_delta_a=round(new_a - 1000.0, 1),
            elo_delta_b=round(new_b - 1000.0, 1),
        )
        store.update_elo(1, new_a, epoch=5)
        store.update_elo(2, new_b, epoch=5)

        # Verify DB values match what compute_elo_update returned
        check_conn = sqlite3.connect(tournament_db)
        check_conn.row_factory = sqlite3.Row
        row_a = check_conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = ?", (1,)
        ).fetchone()
        row_b = check_conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = ?", (2,)
        ).fetchone()
        assert row_a["elo_rating"] == pytest.approx(new_a)
        assert row_b["elo_rating"] == pytest.approx(new_b)
        check_conn.close()

    def test_league_results_row_inserted(self, store, tournament_db):
        """A league_results row should be inserted with win/loss/draw counts."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1))
        _insert_entry(conn, _make_entry(2))
        conn.close()

        store.record_result(
            epoch=10, learner_id=1, opponent_id=2,
            wins=5, losses=3, draws=2,
        )

        check_conn = sqlite3.connect(tournament_db)
        check_conn.row_factory = sqlite3.Row
        row = check_conn.execute("SELECT * FROM league_results").fetchone()
        assert row is not None
        assert row["wins"] == 5
        assert row["losses"] == 3
        assert row["draws"] == 2
        assert row["epoch"] == 10
        check_conn.close()

    def test_elo_history_rows_inserted(self, store, tournament_db):
        """update_elo writes both league_entries and elo_history rows."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1))
        _insert_entry(conn, _make_entry(2))
        conn.close()

        store.update_elo(1, 1010.0, epoch=7)
        store.update_elo(2, 990.0, epoch=7)

        check_conn = sqlite3.connect(tournament_db)
        check_conn.row_factory = sqlite3.Row
        rows = check_conn.execute(
            "SELECT * FROM elo_history WHERE epoch = ? ORDER BY entry_id", (7,)
        ).fetchall()
        assert len(rows) == 2
        # Verify both the entry_ids and the Elo values stored in history
        assert rows[0]["entry_id"] == 1
        assert rows[0]["elo_rating"] == pytest.approx(1010.0)
        assert rows[1]["entry_id"] == 2
        assert rows[1]["elo_rating"] == pytest.approx(990.0)
        check_conn.close()

    def test_games_played_incremented(self, store, tournament_db):
        """games_played should increase by total games (wins+losses+draws) for both entries."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1))
        _insert_entry(conn, _make_entry(2))
        conn.close()

        # Verify initial state
        pre_conn = sqlite3.connect(tournament_db)
        pre_conn.row_factory = sqlite3.Row
        pre_a = pre_conn.execute("SELECT games_played FROM league_entries WHERE id = 1").fetchone()
        pre_b = pre_conn.execute("SELECT games_played FROM league_entries WHERE id = 2").fetchone()
        assert pre_a["games_played"] == 0, "Initial games_played should be 0"
        assert pre_b["games_played"] == 0, "Initial games_played should be 0"
        pre_conn.close()

        store.record_result(
            epoch=1, learner_id=1, opponent_id=2,
            wins=3, losses=4, draws=1,
        )

        check_conn = sqlite3.connect(tournament_db)
        check_conn.row_factory = sqlite3.Row
        row_a = check_conn.execute("SELECT games_played FROM league_entries WHERE id = 1").fetchone()
        row_b = check_conn.execute("SELECT games_played FROM league_entries WHERE id = 2").fetchone()
        assert row_a["games_played"] == 8  # 0 + 3 + 4 + 1
        assert row_b["games_played"] == 8
        check_conn.close()

    def test_draw_result_elo_changes_minimal(self, store, tournament_db):
        """Draw between unequal-rated opponents: weaker gains, stronger loses slightly."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        _insert_entry(conn, _make_entry(1, elo=1100.0))
        _insert_entry(conn, _make_entry(2, elo=900.0))
        conn.close()

        result_score = 0.5  # draw
        new_a, new_b = compute_elo_update(1100.0, 900.0, result=result_score, k=16.0)

        # Draw between unequal players: stronger (A) loses a little, weaker (B) gains a little
        assert new_a < 1100.0, "Stronger player should lose Elo on a draw"
        assert new_b > 900.0, "Weaker player should gain Elo on a draw"

        store.update_elo(1, new_a, epoch=1)
        store.update_elo(2, new_b, epoch=1)

        check_conn = sqlite3.connect(tournament_db)
        check_conn.row_factory = sqlite3.Row
        row_a = check_conn.execute("SELECT elo_rating FROM league_entries WHERE id = 1").fetchone()
        row_b = check_conn.execute("SELECT elo_rating FROM league_entries WHERE id = 2").fetchone()
        # Both should have changed from their starting values
        assert row_a["elo_rating"] == pytest.approx(new_a)
        assert row_b["elo_rating"] == pytest.approx(new_b)
        # But changes should be small (draw ≈ expected result)
        assert abs(row_a["elo_rating"] - 1100.0) < 10.0
        assert abs(row_b["elo_rating"] - 900.0) < 10.0
        check_conn.close()


# ===========================================================================
# DB helpers
# ===========================================================================


class TestDBHelpers:
    """Test _current_epoch and store.list_entries."""

    def test_current_epoch_from_training_state(self, store, tournament_db):
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

        t = _make_tournament(store)
        assert t._current_epoch() == 42

    def test_current_epoch_returns_zero_when_no_state(self, store):
        """_current_epoch should return 0 when no training_state row exists."""
        t = _make_tournament(store)
        assert t._current_epoch() == 0

    def test_list_entries_returns_opponent_entries(self, store, tournament_db):
        """store.list_entries should return OpponentEntry objects ordered by created_epoch ASC."""
        conn = sqlite3.connect(tournament_db)
        conn.row_factory = sqlite3.Row
        # Insert in reverse Elo order to verify sort is by epoch, not Elo
        _insert_entry(conn, _make_entry(2, elo=800.0))
        _insert_entry(conn, _make_entry(1, elo=1200.0))
        conn.close()

        entries = store.list_entries()
        assert len(entries) == 2
        assert isinstance(entries[0], OpponentEntry)
        # Ordered by created_epoch ASC (id==created_epoch in _make_entry)
        assert entries[0].created_epoch <= entries[1].created_epoch


# ===========================================================================
# Lifecycle — start/stop
# ===========================================================================


class TestLifecycle:
    """Test thread start/stop behavior."""

    def test_start_stop_lifecycle(self, store):
        """start() should spawn a thread; stop() should join it."""
        t = _make_tournament(store)
        assert not t.is_running

        # Mock _run_loop to avoid needing VecEnv
        with patch.object(t, "_run_loop", side_effect=lambda: t._stop_event.wait()):
            t.start()
            assert t.is_running
            t.stop(timeout=2.0)
            assert not t.is_running

    def test_double_start_is_idempotent(self, store):
        """Calling start() twice should not spawn a second thread."""
        t = _make_tournament(store)

        with patch.object(t, "_run_loop", side_effect=lambda: t._stop_event.wait()):
            t.start()
            thread1 = t._thread
            t.start()  # should be a no-op
            assert t._thread is thread1
            t.stop(timeout=2.0)

    def test_stop_without_start(self, store):
        """stop() when no thread is running should not raise."""
        t = _make_tournament(store)
        t.stop()  # no-op, should not raise

    def test_is_running_reflects_thread_state(self, store):
        """is_running should be False after the thread naturally exits."""
        t = _make_tournament(store)
        exit_event = threading.Event()

        def quick_exit():
            exit_event.set()  # signal that thread started
            # Exit immediately

        with patch.object(t, "_run_loop", side_effect=quick_exit):
            t.start()
            exit_event.wait(timeout=2.0)
            # Give the thread time to actually finish
            assert t._thread is not None
            t._thread.join(timeout=2.0)
            assert not t.is_running


# ===========================================================================
# Constructor validation
# ===========================================================================


class TestConstructor:
    """Test that the new constructor accepts store and scheduler."""

    def test_constructor_stores_references(self, store):
        scheduler = _make_scheduler()
        t = LeagueTournament(store=store, scheduler=scheduler, device="cpu")
        assert t.store is store
        assert t.scheduler is scheduler

    def test_constructor_defaults(self, store):
        scheduler = _make_scheduler()
        t = LeagueTournament(store=store, scheduler=scheduler, device="cpu")
        assert t.num_envs == 64
        assert t.max_ply == 512
        assert t.games_per_match == 64
        assert t.k_factor == 16.0
        assert t.pause_seconds == 5.0
        assert t.min_pool_size == 3
        assert str(t.device) == "cpu"
        assert t.learner_entry_id is None

    def test_constructor_overrides(self, store):
        scheduler = _make_scheduler()
        t = LeagueTournament(
            store=store, scheduler=scheduler, device="cpu",
            num_envs=8, games_per_match=16, k_factor=32.0,
        )
        assert t.num_envs == 8
        assert t.games_per_match == 16
        assert t.k_factor == 32.0


# ===========================================================================
# Concurrent pool integration
# ===========================================================================


class TestConcurrentTournament:
    """Test ConcurrentMatchPool integration with LeagueTournament."""

    def test_tournament_accepts_concurrent_pool(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        scheduler = _make_scheduler()
        config = ConcurrencyConfig(
            parallel_matches=2, envs_per_match=2, total_envs=4, max_resident_models=4,
        )
        pool = ConcurrentMatchPool(config)
        t = _make_tournament(store, scheduler, concurrent_pool=pool)
        assert t.concurrent_pool is pool
        store.close()

    def test_tournament_without_pool_still_works(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        t = _make_tournament(store)
        assert t.concurrent_pool is None
        store.close()
