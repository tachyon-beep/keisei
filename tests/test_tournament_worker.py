"""Tests for TournamentWorker — batched claim/play/record loop and lifecycle."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from keisei.config import ConcurrencyConfig
from keisei.db import _connect, init_db
from keisei.training.concurrent_matches import MatchResult, RoundStats
from keisei.training.opponent_store import OpponentEntry, OpponentStore
from keisei.training.tournament_queue import (
    claim_next_pairing,
    enqueue_pairings,
    get_round_status,
)
from keisei.training.tournament_runner import TournamentWorker


def _mock_load_opponent(entry: OpponentEntry, device: str = "cpu") -> torch.nn.Module:
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


@pytest.fixture
def concurrency() -> ConcurrencyConfig:
    return ConcurrencyConfig(
        parallel_matches=4, envs_per_match=2, total_envs=8,
        max_resident_models=4,
    )


def _seed_entries(db: str, n: int = 3) -> list[int]:
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


class _FakePool:
    """ConcurrentMatchPool stand-in that returns deterministic results."""

    def __init__(
        self, concurrency: ConcurrencyConfig, *,
        per_pairing_outcome: tuple[int, int, int] = (2, 1, 0),
        raise_on_run: Exception | None = None,
    ) -> None:
        self.config = concurrency
        self._outcome = per_pairing_outcome
        self._raise = raise_on_run
        self.run_round_calls: list[list[tuple[OpponentEntry, OpponentEntry]]] = []

    def run_round(
        self, vecenv: Any,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
        **kwargs: Any,
    ) -> tuple[list[MatchResult], RoundStats]:
        self.run_round_calls.append(list(pairings))
        if self._raise is not None:
            raise self._raise
        a_wins, b_wins, draws = self._outcome
        results = [
            MatchResult(
                entry_a=a, entry_b=b,
                a_wins=a_wins, b_wins=b_wins, draws=draws,
                rollout=None, feature_tracker=None,
            )
            for a, b in pairings
        ]
        stats = RoundStats(
            pairings_requested=len(pairings),
            pairings_completed=len(pairings),
            total_games=len(pairings) * (a_wins + b_wins + draws),
            active_slots=min(self.config.parallel_matches, len(pairings)),
        )
        return results, stats


def _make_worker(
    db: str, league_dir: str, concurrency: ConcurrencyConfig,
    *, pool: _FakePool | None = None,
    stop_event: threading.Event | None = None,
    **kwargs: Any,
) -> TournamentWorker:
    pool = pool if pool is not None else _FakePool(concurrency)
    worker = TournamentWorker(
        db_path=db, league_dir=league_dir,
        worker_id="test-w0", device="cpu",
        concurrency=concurrency,
        stop_event=stop_event or threading.Event(),
        pool=pool,
        vecenv_factory=lambda: MagicMock(),
        **kwargs,
    )
    worker.store.load_opponent = _mock_load_opponent  # type: ignore[method-assign]
    return worker


class TestBatchedLoop:
    def test_claims_batch_and_runs_round_once(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3), (ids[1], ids[2], 3)],
        )
        stop = threading.Event()
        pool = _FakePool(concurrency)

        def stop_after_run(orig_run_round):  # noqa: ANN001
            def wrapper(*args: Any, **kw: Any) -> Any:
                out = orig_run_round(*args, **kw)
                stop.set()
                return out
            return wrapper
        pool.run_round = stop_after_run(pool.run_round)  # type: ignore[method-assign]

        worker = _make_worker(db, league_dir, concurrency, pool=pool, stop_event=stop)
        worker.run()

        assert len(pool.run_round_calls) == 1
        assert len(pool.run_round_calls[0]) == 3
        status = get_round_status(db, round_id=1)
        assert status.get("done", 0) == 3

    def test_batch_size_capped_at_parallel_matches_times_two(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
        """Worker claims enough for initial slots + backfill (2× parallel_matches)."""
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        # 20 pairings, parallel_matches=4 -> expect batch of 8 (2×4).
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 1) for _ in range(20)],
        )
        stop = threading.Event()
        pool = _FakePool(concurrency)

        def stop_after_first(orig):  # noqa: ANN001
            def wrapper(*args: Any, **kw: Any) -> Any:
                out = orig(*args, **kw)
                stop.set()
                return out
            return wrapper
        pool.run_round = stop_after_first(pool.run_round)  # type: ignore[method-assign]

        worker = _make_worker(db, league_dir, concurrency, pool=pool, stop_event=stop)
        worker.run()

        assert len(pool.run_round_calls) == 1
        # 2 × parallel_matches = 8
        assert len(pool.run_round_calls[0]) == 8
        status = get_round_status(db, round_id=1)
        assert status.get("done", 0) == 8
        assert status.get("pending", 0) == 12

    def test_multiple_iterations_drain_queue(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 1) for _ in range(12)],
        )
        stop = threading.Event()
        pool = _FakePool(concurrency)
        # Stop after 2 iterations (2 × 8 = 16 but only 12 pending → second batch 4).
        iteration = {"n": 0}

        def counting_run(orig):  # noqa: ANN001
            def wrapper(*args: Any, **kw: Any) -> Any:
                iteration["n"] += 1
                out = orig(*args, **kw)
                if iteration["n"] >= 2:
                    stop.set()
                return out
            return wrapper
        pool.run_round = counting_run(pool.run_round)  # type: ignore[method-assign]

        worker = _make_worker(db, league_dir, concurrency, pool=pool, stop_event=stop)
        worker.run()

        assert iteration["n"] == 2
        assert len(pool.run_round_calls[0]) == 8
        assert len(pool.run_round_calls[1]) == 4
        status = get_round_status(db, round_id=1)
        assert status.get("done", 0) == 12

    def test_elo_updated_for_each_result(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
        store: OpponentStore,
    ) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3)],
        )
        stop = threading.Event()
        pool = _FakePool(concurrency, per_pairing_outcome=(3, 0, 0))

        def stop_after(orig):  # noqa: ANN001
            def wrapper(*args: Any, **kw: Any) -> Any:
                out = orig(*args, **kw)
                stop.set()
                return out
            return wrapper
        pool.run_round = stop_after(pool.run_round)  # type: ignore[method-assign]

        worker = _make_worker(db, league_dir, concurrency, pool=pool, stop_event=stop)
        worker.run()

        a = store.get_entry(ids[0])
        b = store.get_entry(ids[1])
        c = store.get_entry(ids[2])
        assert a is not None and b is not None and c is not None
        # entry 0 beat 1 and 2 → rating up
        assert a.elo_rating > 1000.0
        assert b.elo_rating < 1000.0
        assert c.elo_rating < 1000.0

    def test_marks_batch_failed_on_exception(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3)],
        )
        stop = threading.Event()
        pool = _FakePool(concurrency, raise_on_run=RuntimeError("boom"))

        orig = pool.run_round

        def one_shot(*args: Any, **kw: Any) -> Any:
            try:
                return orig(*args, **kw)
            finally:
                stop.set()
        pool.run_round = one_shot  # type: ignore[method-assign]

        worker = _make_worker(db, league_dir, concurrency, pool=pool, stop_event=stop)
        worker.run()
        status = get_round_status(db, round_id=1)
        assert status.get("failed", 0) == 2

    def test_stale_pairings_expired_via_batch_claim(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
        ids = _seed_entries(db)
        _set_current_epoch(db, 100)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3)],
        )
        stop = threading.Event()
        pool = _FakePool(concurrency)
        worker = _make_worker(
            db, league_dir, concurrency, pool=pool, stop_event=stop,
            max_staleness_epochs=50,
        )
        # Background run: there's nothing fresh to play, loop will poll forever.
        t = threading.Thread(target=worker.run)
        t.start()
        t.join(timeout=5.0)
        stop.set()
        t.join(timeout=2.0)

        assert pool.run_round_calls == []
        status = get_round_status(db, round_id=1)
        assert status.get("expired", 0) == 2

    def test_missing_result_marks_pairing_failed(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
        """If run_round returns fewer MatchResults than pairings claimed,
        the unplayed pairings must be marked 'failed' (e.g. load_fn failure
        inside the pool)."""
        ids = _seed_entries(db)
        _set_current_epoch(db, 5)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(ids[0], ids[1], 3), (ids[0], ids[2], 3)],
        )
        stop = threading.Event()
        pool = _FakePool(concurrency)

        def partial_results(vecenv: Any, pairings: list[Any], **kw: Any) -> Any:
            pool.run_round_calls.append(list(pairings))
            stop.set()
            # Return only the first pairing's result
            a, b = pairings[0]
            res = MatchResult(
                entry_a=a, entry_b=b, a_wins=2, b_wins=1, draws=0,
                rollout=None, feature_tracker=None,
            )
            return [res], RoundStats(pairings_requested=2, pairings_completed=1)
        pool.run_round = partial_results  # type: ignore[method-assign]

        worker = _make_worker(db, league_dir, concurrency, pool=pool, stop_event=stop)
        worker.run()
        status = get_round_status(db, round_id=1)
        assert status.get("done", 0) == 1
        assert status.get("failed", 0) == 1


class TestLifecycle:
    def test_graceful_stop_on_event(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
        stop = threading.Event()
        stop.set()
        worker = _make_worker(db, league_dir, concurrency, stop_event=stop)
        worker.run()
        assert worker._pairings_done == 0

    def test_startup_sweep_resets_own_pairings(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
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
        worker = _make_worker(db, league_dir, concurrency, stop_event=stop)
        worker.run()
        assert get_round_status(db, round_id=1) == {"pending": 1}


class TestConstruction:
    def test_pool_constructed_from_concurrency_when_not_injected(
        self, db: str, league_dir: str, concurrency: ConcurrencyConfig,
    ) -> None:
        """When no pool is injected, worker builds a real ConcurrentMatchPool
        from the provided ConcurrencyConfig."""
        from keisei.training.concurrent_matches import ConcurrentMatchPool

        worker = TournamentWorker(
            db_path=db, league_dir=league_dir,
            worker_id="test-w0", device="cpu",
            concurrency=concurrency,
            stop_event=threading.Event(),
            vecenv_factory=lambda: MagicMock(),
        )
        assert isinstance(worker._pool, ConcurrentMatchPool)
        assert worker._pool.config is concurrency
