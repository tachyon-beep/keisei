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

from keisei.config import MatchSchedulerConfig
from keisei.db import _connect, init_db
from keisei.training.match_scheduler import MatchScheduler
from keisei.training.match_utils import MatchOutcome
from keisei.training.opponent_store import OpponentStore, Role
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
    completed = 0

    def play_fn(*args, **kwargs):
        nonlocal completed
        completed += 1
        if completed >= n_pairings:
            stop.set()
        return MatchOutcome(a_wins=1, b_wins=1, draws=1)

    import torch
    worker = TournamentWorker(
        db_path=db, league_dir=league_dir,
        worker_id="itest-w0", device="cpu",
        games_per_match=3,
        stop_event=stop,
        play_fn=play_fn,
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
