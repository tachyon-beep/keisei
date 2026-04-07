"""Tests for LeagueTournament._run_concurrent_round() Elo write path.

GAP-C2: The _run_concurrent_round method (tournament.py lines 286-431)
handles the full concurrent Elo update path including role-specific Elo,
record_result() calls, and dynamic_trainer integration.  These tests mock
ConcurrentMatchPool.run_round() to return controlled MatchResult objects
and verify the downstream write calls.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from keisei.config import ConcurrencyConfig, MatchSchedulerConfig, RoleEloConfig
from keisei.db import init_db
from keisei.training.concurrent_matches import (
    ConcurrentMatchPool,
    MatchResult,
    RoundStats,
)
from keisei.training.match_scheduler import MatchScheduler
from keisei.training.opponent_store import (
    EntryStatus,
    EloColumn,
    OpponentEntry,
    OpponentStore,
    Role,
)
from keisei.training.role_elo import RoleEloTracker
from keisei.training.tournament import LeagueTournament

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    id: int,
    elo: float = 1000.0,
    role: Role = Role.FRONTIER_STATIC,
    status: EntryStatus = EntryStatus.ACTIVE,
) -> OpponentEntry:
    return OpponentEntry(
        id=id,
        display_name=f"Bot-{id}",
        architecture="resnet",
        model_params={"hidden_size": 16},
        checkpoint_path="/fake/ckpt.pt",
        elo_rating=elo,
        created_epoch=id,
        games_played=0,
        created_at="2026-04-01T00:00:00Z",
        flavour_facts=[],
        role=role,
        status=status,
    )


def _insert_entry(conn: sqlite3.Connection, entry: OpponentEntry) -> None:
    conn.execute(
        """INSERT INTO league_entries
           (id, display_name, flavour_facts, architecture, model_params,
            checkpoint_path, elo_rating, created_epoch, games_played, created_at, role)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry.id, entry.display_name, json.dumps(entry.flavour_facts),
            entry.architecture, json.dumps(entry.model_params),
            entry.checkpoint_path, entry.elo_rating, entry.created_epoch,
            entry.games_played, entry.created_at, entry.role.value,
        ),
    )
    conn.commit()


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    path = str(tmp_path / "tournament.db")
    init_db(path)
    return path


@pytest.fixture
def league_dir(tmp_path: Path) -> str:
    d = tmp_path / "league"
    d.mkdir()
    return str(d)


@pytest.fixture
def store(db_path: str, league_dir: str) -> OpponentStore:
    s = OpponentStore(db_path, league_dir)
    yield s
    s.close()


def _make_tournament(
    store: OpponentStore,
    concurrent_pool: ConcurrentMatchPool | None = None,
    role_elo_tracker: RoleEloTracker | None = None,
    dynamic_trainer: MagicMock | None = None,
) -> LeagueTournament:
    config = ConcurrencyConfig(
        parallel_matches=2, envs_per_match=2, total_envs=4, max_resident_models=4,
    )
    pool = concurrent_pool or ConcurrentMatchPool(config)
    scheduler = MatchScheduler(MatchSchedulerConfig())
    return LeagueTournament(
        store=store,
        scheduler=scheduler,
        device="cpu",
        num_envs=4,
        games_per_match=3,
        concurrent_pool=pool,
        role_elo_tracker=role_elo_tracker,
        dynamic_trainer=dynamic_trainer,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcurrentRoundEloWrites:
    """Verify _run_concurrent_round writes correct Elo data to the store."""

    def test_record_result_called_with_correct_args(self, store, db_path) -> None:
        """Mock run_round to return a controlled MatchResult, verify record_result args."""
        conn = sqlite3.connect(db_path)
        entry_a = _make_entry(1, elo=1000.0, role=Role.FRONTIER_STATIC)
        entry_b = _make_entry(2, elo=1000.0, role=Role.FRONTIER_STATIC)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        tournament = _make_tournament(store)
        match_result = MatchResult(
            entry_a=entry_a, entry_b=entry_b,
            a_wins=5, b_wins=2, draws=1, rollout=None,
        )
        mock_stats = RoundStats()

        with patch.object(
            tournament.concurrent_pool, "run_round",
            return_value=([match_result], mock_stats),
        ):
            vecenv = MagicMock()
            tournament._run_concurrent_round(
                vecenv, [(entry_a, entry_b)], epoch=10,
            )

        # Verify Elo was updated in the DB
        check_conn = sqlite3.connect(db_path)
        check_conn.row_factory = sqlite3.Row
        row_a = check_conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = 1"
        ).fetchone()
        row_b = check_conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = 2"
        ).fetchone()
        # A won more games → A's Elo should increase
        assert row_a["elo_rating"] > 1000.0
        assert row_b["elo_rating"] < 1000.0

        # Verify a league_results row was inserted
        result_row = check_conn.execute(
            "SELECT * FROM league_results"
        ).fetchone()
        assert result_row is not None
        assert result_row["wins_a"] == 5
        assert result_row["wins_b"] == 2
        assert result_row["draws"] == 1
        assert result_row["epoch"] == 10
        assert result_row["match_type"] == "calibration"
        check_conn.close()

    def test_retired_entry_skipped(self, store, db_path) -> None:
        """Entries retired mid-round should not get Elo updates."""
        conn = sqlite3.connect(db_path)
        entry_a = _make_entry(1, elo=1000.0)
        entry_b = _make_entry(2, elo=1000.0)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        # Retire entry_b mid-round
        conn.execute(
            "UPDATE league_entries SET status = 'retired' WHERE id = 2"
        )
        conn.commit()
        conn.close()

        tournament = _make_tournament(store)
        match_result = MatchResult(
            entry_a=entry_a, entry_b=entry_b,
            a_wins=3, b_wins=0, draws=0, rollout=None,
        )
        mock_stats = RoundStats()

        with patch.object(
            tournament.concurrent_pool, "run_round",
            return_value=([match_result], mock_stats),
        ):
            tournament._run_concurrent_round(
                MagicMock(), [(entry_a, entry_b)], epoch=5,
            )

        # No league_results row should be inserted
        check_conn = sqlite3.connect(db_path)
        count = check_conn.execute(
            "SELECT COUNT(*) FROM league_results"
        ).fetchone()[0]
        assert count == 0, "Retired entry should not produce league_results"
        # Elo should not change
        row_a = check_conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = 1"
        ).fetchone()
        assert row_a[0] == 1000.0, "Elo should not change for retired match"
        check_conn.close()

    def test_zero_total_games_skipped(self, store, db_path) -> None:
        """Results with 0 total games should be silently skipped."""
        conn = sqlite3.connect(db_path)
        entry_a = _make_entry(1)
        entry_b = _make_entry(2)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        tournament = _make_tournament(store)
        empty_result = MatchResult(
            entry_a=entry_a, entry_b=entry_b,
            a_wins=0, b_wins=0, draws=0, rollout=None,
        )
        mock_stats = RoundStats()

        with patch.object(
            tournament.concurrent_pool, "run_round",
            return_value=([empty_result], mock_stats),
        ):
            tournament._run_concurrent_round(
                MagicMock(), [(entry_a, entry_b)], epoch=5,
            )

        check_conn = sqlite3.connect(db_path)
        count = check_conn.execute(
            "SELECT COUNT(*) FROM league_results"
        ).fetchone()[0]
        assert count == 0, "Zero-game result should not produce league_results"
        check_conn.close()

    def test_match_type_set_correctly_for_training_match(self, store, db_path) -> None:
        """D-vs-D match should have match_type='train'."""
        conn = sqlite3.connect(db_path)
        entry_a = _make_entry(1, role=Role.DYNAMIC)
        entry_b = _make_entry(2, role=Role.DYNAMIC)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        tournament = _make_tournament(store)
        match_result = MatchResult(
            entry_a=entry_a, entry_b=entry_b,
            a_wins=2, b_wins=1, draws=0, rollout=None,
        )
        mock_stats = RoundStats()

        with patch.object(
            tournament.concurrent_pool, "run_round",
            return_value=([match_result], mock_stats),
        ):
            tournament._run_concurrent_round(
                MagicMock(), [(entry_a, entry_b)], epoch=5,
            )

        check_conn = sqlite3.connect(db_path)
        check_conn.row_factory = sqlite3.Row
        row = check_conn.execute("SELECT match_type FROM league_results").fetchone()
        assert row["match_type"] == "train"
        check_conn.close()

    def test_role_elo_tracker_updates_called(self, store, db_path) -> None:
        """When role_elo_tracker is set, update_from_result should be called."""
        conn = sqlite3.connect(db_path)
        entry_a = _make_entry(1, role=Role.FRONTIER_STATIC)
        entry_b = _make_entry(2, role=Role.FRONTIER_STATIC)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        role_tracker = MagicMock(spec=RoleEloTracker)
        role_tracker.k_for_context.return_value = 16.0
        role_tracker.columns_for_context.return_value = (
            EloColumn.FRONTIER, EloColumn.FRONTIER,
        )

        tournament = _make_tournament(store, role_elo_tracker=role_tracker)
        match_result = MatchResult(
            entry_a=entry_a, entry_b=entry_b,
            a_wins=3, b_wins=1, draws=0, rollout=None,
        )
        mock_stats = RoundStats()

        with patch.object(
            tournament.concurrent_pool, "run_round",
            return_value=([match_result], mock_stats),
        ):
            tournament._run_concurrent_round(
                MagicMock(), [(entry_a, entry_b)], epoch=5,
            )

        role_tracker.update_from_result.assert_called_once()
        args = role_tracker.update_from_result.call_args
        # First arg is entry_a (re-fetched), second is entry_b
        assert args[0][0].id == 1
        assert args[0][1].id == 2
        # result_score should be 1.0 (A won majority)
        assert args[0][2] == 1.0
        # context should be 'frontier'
        assert args[0][3] == "frontier"

    def test_dynamic_trainer_record_match_called(self, store, db_path) -> None:
        """When dynamic_trainer is set and rollout is present, record_match should be called."""
        conn = sqlite3.connect(db_path)
        entry_a = _make_entry(1, role=Role.DYNAMIC)
        entry_b = _make_entry(2, role=Role.DYNAMIC)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        mock_trainer = MagicMock()
        mock_trainer.should_update.return_value = False  # don't trigger update

        mock_rollout = MagicMock()
        tournament = _make_tournament(store, dynamic_trainer=mock_trainer)
        match_result = MatchResult(
            entry_a=entry_a, entry_b=entry_b,
            a_wins=2, b_wins=1, draws=0, rollout=mock_rollout,
        )
        mock_stats = RoundStats()

        with patch.object(
            tournament.concurrent_pool, "run_round",
            return_value=([match_result], mock_stats),
        ):
            tournament._run_concurrent_round(
                MagicMock(), [(entry_a, entry_b)], epoch=5,
            )

        # Both entries are DYNAMIC, so record_match should be called for each
        assert mock_trainer.record_match.call_count == 2
        # Verify side=0 for entry_a and side=1 for entry_b
        calls = mock_trainer.record_match.call_args_list
        assert calls[0] == call(1, mock_rollout, side=0)
        assert calls[1] == call(2, mock_rollout, side=1)

    def test_priority_scorer_advanced_after_round(self, store, db_path) -> None:
        """Priority scorer should be notified after concurrent round completes."""
        conn = sqlite3.connect(db_path)
        entry_a = _make_entry(1)
        entry_b = _make_entry(2)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        tournament = _make_tournament(store)
        mock_scorer = MagicMock()
        tournament.scheduler._priority_scorer = mock_scorer

        match_result = MatchResult(
            entry_a=entry_a, entry_b=entry_b,
            a_wins=3, b_wins=1, draws=0, rollout=None,
        )
        mock_stats = RoundStats()

        with patch.object(
            tournament.concurrent_pool, "run_round",
            return_value=([match_result], mock_stats),
        ):
            tournament._run_concurrent_round(
                MagicMock(), [(entry_a, entry_b)], epoch=5,
            )

        mock_scorer.record_result.assert_called_once_with(entry_a.id, entry_b.id)
        mock_scorer.record_round_result.assert_called_once_with(entry_a.id, entry_b.id)
        mock_scorer.advance_round.assert_called_once()

    def test_no_advance_round_when_no_games_played(self, store, db_path) -> None:
        """If all results have 0 games, advance_round should NOT be called."""
        conn = sqlite3.connect(db_path)
        entry_a = _make_entry(1)
        entry_b = _make_entry(2)
        _insert_entry(conn, entry_a)
        _insert_entry(conn, entry_b)
        conn.close()

        tournament = _make_tournament(store)
        mock_scorer = MagicMock()
        tournament.scheduler._priority_scorer = mock_scorer

        empty_result = MatchResult(
            entry_a=entry_a, entry_b=entry_b,
            a_wins=0, b_wins=0, draws=0, rollout=None,
        )
        mock_stats = RoundStats()

        with patch.object(
            tournament.concurrent_pool, "run_round",
            return_value=([empty_result], mock_stats),
        ):
            tournament._run_concurrent_round(
                MagicMock(), [(entry_a, entry_b)], epoch=5,
            )

        mock_scorer.advance_round.assert_not_called()
