"""End-to-end test for the tournament sidecar architecture.

Verifies the full round-trip: dispatcher enqueues pairings, worker claims
and completes them, all through the real DB layer (no mocks).

Marked @pytest.mark.integration — excluded from default pytest runs.
Run explicitly with: uv run pytest -m integration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from keisei.config import ConcurrencyConfig, MatchSchedulerConfig
from keisei.db import _connect, init_db
from keisei.training.concurrent_matches import MatchResult, RoundStats
from keisei.training.match_scheduler import MatchScheduler
from keisei.training.opponent_store import OpponentStore
from keisei.training.tournament_dispatcher import TournamentDispatcher
from keisei.training.tournament_queue import get_round_status
from keisei.training.tournament_runner import TournamentWorker


def _seed_db(db: str, league_dir: str, n_entries: int = 4) -> list[int]:
    """Seed DB with training_state and league entries."""
    conn = _connect(db)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO training_state "
            "(id, config_json, display_name, model_arch, algorithm_name, "
            "started_at, current_epoch, current_step, status, phase) "
            "VALUES (1, '{}', 'test', 'resnet', 'ppo', "
            "'2026-01-01', 10, 0, 'running', 'rollout')",
        )
        ids = []
        for i in range(n_entries):
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


@pytest.mark.integration
def test_dispatcher_and_worker_round_trip(tmp_path: Path) -> None:
    """Dispatcher enqueues a round; worker claims all pairings and
    completes them. Verifies the full dispatcher→queue→worker→DB flow
    without subprocess spawning (both run in-process)."""
    db = str(tmp_path / "test.db")
    league_dir = str(tmp_path / "league")
    Path(league_dir).mkdir()
    init_db(db)

    ids = _seed_db(db, league_dir, n_entries=4)
    store = OpponentStore(db_path=db, league_dir=league_dir)
    sched = MatchScheduler(MatchSchedulerConfig())
    dispatcher = TournamentDispatcher(
        store=store, scheduler=sched, games_per_match=3,
    )

    round_id = dispatcher.enqueue_round(epoch=10)
    assert round_id is not None

    status = get_round_status(db, round_id)
    n_pairings = status.get("pending", 0)
    assert n_pairings == 6

    import threading
    stop = threading.Event()
    concurrency = ConcurrencyConfig(
        parallel_matches=3, envs_per_match=2, total_envs=6,
        max_resident_models=4,
    )
    completed = 0
    observed_active_slots: list[int] = []

    class _RealRoundPool:
        """Stand-in pool that records active_slots per call, proving the
        worker hands the whole batch to run_round (not one-pairing-at-a-time)."""

        def __init__(self) -> None:
            self.config = concurrency

        def run_round(self, vecenv, pairings, **kwargs):  # noqa: ANN001, ANN201
            nonlocal completed
            observed_active_slots.append(
                min(concurrency.parallel_matches, len(pairings))
            )
            results = [
                MatchResult(
                    entry_a=a, entry_b=b,
                    a_wins=1, b_wins=1, draws=1,
                    rollout=None, feature_tracker=None,
                )
                for a, b in pairings
            ]
            completed += len(results)
            if completed >= n_pairings:
                stop.set()
            return results, RoundStats(
                pairings_requested=len(pairings),
                pairings_completed=len(pairings),
                total_games=3 * len(pairings),
                active_slots=min(concurrency.parallel_matches, len(pairings)),
            )

    import torch
    worker = TournamentWorker(
        db_path=db, league_dir=league_dir,
        worker_id="itest-w0", device="cpu",
        concurrency=concurrency,
        games_per_match=3,
        stop_event=stop,
        pool=_RealRoundPool(),
        vecenv_factory=lambda: MagicMock(),
    )
    worker.store.load_opponent = lambda entry, device="cpu": torch.nn.Linear(4, 4)  # type: ignore[method-assign]

    t = threading.Thread(target=worker.run)
    t.start()
    t.join(timeout=30.0)
    if t.is_alive():
        stop.set()
        t.join(timeout=5.0)

    final_status = get_round_status(db, round_id)
    assert final_status.get("done", 0) == n_pairings, (
        f"Expected {n_pairings} done, got {final_status}"
    )
    assert dispatcher.check_round_completion(round_id) is True
    # Regression guard for the single-slot-at-a-time bug: at least one
    # run_round call must have been handed > 1 pairing.  Without batching,
    # every call would see exactly 1 slot active.
    assert max(observed_active_slots) > 1, (
        f"Worker never ran multiple slots concurrently: "
        f"observed active_slots = {observed_active_slots}"
    )
