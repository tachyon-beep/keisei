"""Tests for TournamentWorker — claim/play/record loop and lifecycle."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import pytest

from keisei.db import _connect, init_db
from keisei.training.match_utils import MatchOutcome
from keisei.training.opponent_store import OpponentStore
from keisei.training.tournament_queue import (
    enqueue_pairings,
    get_round_status,
    claim_next_pairing,
)
from keisei.training.tournament_runner import TournamentWorker


def _mock_load_opponent(entry, device="cpu"):
    """Return a tiny dummy model instead of loading from disk."""
    m = torch.nn.Linear(4, 4)
    m.eval()
    return m


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture
def league_dir(tmp_path: Path) -> str:
    d = str(tmp_path / "league")
    Path(d).mkdir(exist_ok=True)
    return d


@pytest.fixture
def store(db: str, league_dir: str) -> OpponentStore:
    return OpponentStore(db_path=db, league_dir=league_dir)


def _seed_entries(db: str, n: int = 3) -> list[int]:
    """Insert minimal league_entries rows and return IDs."""
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


def _set_current_epoch(db: str, epoch: int) -> None:
    """Write current_epoch into training_state for the worker to read."""
    conn = _connect(db)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO training_state "
            "(id, config_json, display_name, model_arch, algorithm_name, "
            "started_at, current_epoch, current_step, status, phase) "
            "VALUES (1, '{}', 'test', 'resnet', 'ppo', "
            "'2026-01-01', ?, 0, 'running', 'rollout')",
            (epoch,),
        )
        conn.commit()
    finally:
        conn.close()


def _mock_play_fn(**kwargs: object) -> MatchOutcome:
    """play_match replacement that returns a fixed outcome without any model inference."""
    return MatchOutcome(a_wins=2, b_wins=1, draws=0)


class TestTournamentWorker:
    def _make_worker(
        self, db: str, league_dir: str, *,
        play_fn: object = None,
        stop_event: threading.Event | None = None,
        **kwargs: object,
    ) -> TournamentWorker:
        worker = TournamentWorker(
            db_path=db, league_dir=league_dir,
            worker_id="test-w0", device="cpu",
            stop_event=stop_event or threading.Event(),
            play_fn=play_fn or _mock_play_fn,
            vecenv_factory=lambda: MagicMock(),
            **kwargs,
        )
        worker.store.load_opponent = _mock_load_opponent  # type: ignore[method-assign]
        return worker

    def test_claims_and_completes_one_pairing(
        self, db: str, league_dir: str, store: OpponentStore,
    ) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5, pairings=[(ids[0], ids[1], 3)],
        )
        stop = threading.Event()
        call_count = 0

        def counting_play_fn(*args: object, **kwargs: object) -> MatchOutcome:
            nonlocal call_count
            call_count += 1
            stop.set()
            return MatchOutcome(a_wins=2, b_wins=1, draws=0)

        worker = self._make_worker(db, league_dir, play_fn=counting_play_fn, stop_event=stop)
        worker.run()
        assert call_count == 1
        status = get_round_status(db, round_id=1)
        assert status.get("done", 0) == 1

    def test_elo_updates_after_match(
        self, db: str, league_dir: str, store: OpponentStore,
    ) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5, pairings=[(ids[0], ids[1], 3)],
        )
        stop = threading.Event()

        def play_then_stop(*args: object, **kwargs: object) -> MatchOutcome:
            stop.set()
            return MatchOutcome(a_wins=3, b_wins=0, draws=0)

        worker = self._make_worker(db, league_dir, play_fn=play_then_stop, stop_event=stop)
        worker.run()
        a = store.get_entry(ids[0])
        b = store.get_entry(ids[1])
        assert a is not None and b is not None
        assert a.elo_rating > 1000.0
        assert b.elo_rating < 1000.0

    def test_graceful_stop_on_event(self, db: str, league_dir: str) -> None:
        stop = threading.Event()
        stop.set()
        worker = self._make_worker(db, league_dir, stop_event=stop)
        worker.run()
        assert worker._pairings_done == 0

    def test_marks_failed_on_exception(self, db: str, league_dir: str) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5, pairings=[(ids[0], ids[1], 3)],
        )
        stop = threading.Event()

        def exploding_play(*args: object, **kwargs: object) -> MatchOutcome:
            stop.set()
            raise RuntimeError("boom")

        worker = self._make_worker(db, league_dir, play_fn=exploding_play, stop_event=stop)
        worker.run()
        status = get_round_status(db, round_id=1)
        assert status.get("failed", 0) == 1

    def test_expires_stale_pairings(self, db: str, league_dir: str) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 100)
        enqueue_pairings(
            db, round_id=1, epoch=5, pairings=[(ids[0], ids[1], 3)],
        )
        stop = threading.Event()
        call_count = 0

        def counting_play(*args: object, **kwargs: object) -> MatchOutcome:
            nonlocal call_count
            call_count += 1
            return MatchOutcome(a_wins=1, b_wins=0, draws=0)

        worker = self._make_worker(
            db, league_dir, play_fn=counting_play, stop_event=stop,
            max_staleness_epochs=50,
        )
        t = threading.Thread(target=worker.run)
        t.start()
        t.join(timeout=5.0)
        if t.is_alive():
            stop.set()
            t.join(timeout=2.0)
        assert call_count == 0
        status = get_round_status(db, round_id=1)
        assert status.get("expired", 0) == 1

    def test_startup_sweep_resets_own_pairings(self, db: str, league_dir: str) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5, pairings=[(ids[0], ids[1], 3)],
        )
        claim = claim_next_pairing(db, worker_id="test-w0")
        assert claim is not None
        assert get_round_status(db, round_id=1) == {"playing": 1}

        stop = threading.Event()
        stop.set()
        worker = self._make_worker(db, league_dir, stop_event=stop)
        worker.run()
        assert get_round_status(db, round_id=1) == {"pending": 1}
