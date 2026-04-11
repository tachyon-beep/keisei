# Tournament Sidecar Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `LeagueTournament._run_loop` from the training process into a standalone subprocess ("tournament worker") that communicates with training exclusively through SQLite, following the established showcase sidecar pattern. Training and tournament become physically separate processes with their own CUDA contexts, their own GILs, and no shared in-memory state — eliminating the last category of coupling between the two loops.

**Architecture:** `LeagueTournament` today runs as a `threading.Thread` inside the training process, sharing Python's GIL and PyTorch's CUDA context with the training rollout. This plan extracts it into a separate process launched via `python -m keisei.training.tournament_runner`. A tiny in-training **dispatcher** generates pairings and enqueues them into a new `tournament_pairing_queue` table; one or more **worker** subprocesses atomically claim pairings, play matches on cuda:1, and write results back to the same database. `PriorityScorer` and `HistoricalGauntlet` stay in the dispatcher (training-process side); `ConcurrentMatchPool`, `DynamicTrainer`, and `StyleProfiler` move to the worker. Elo read-modify-write is protected by `BEGIN IMMEDIATE` transactions. DynamicTrainer updates are serialized per-entry via an atomic claim column.

**Tech Stack:** Python 3.13, SQLite WAL, PyTorch (CUDA contexts isolated per-process), subprocess + signal handling

**Depends on:** Phase 1 (commit `2e1a928`) — the sample-K rollout change is already merged. The sidecar work is independently valuable but the Phase 1 measurements give us a clean baseline to detect any regression introduced by the process-topology change.

**Does NOT include:** The SQLite → Postgres migration (Phase 3). This plan deliberately stays on SQLite so the process-topology refactor can be validated in isolation. Postgres comes afterwards as a mechanical swap-out of the DB layer.

---

## File Map

### New files

| File | Purpose |
|------|---------|
| `keisei/training/tournament_runner.py` | Sidecar process entry point: `ShellTournamentWorker` class + `python -m` CLI. Owns the per-worker main loop, CUDA context, match execution, result write-back. |
| `keisei/training/tournament_dispatcher.py` | Training-side dispatcher: generates pairings via `MatchScheduler`, enqueues them to the DB, advances round state. Owns `PriorityScorer` and `MatchScheduler` references. Runs in the training process at round boundaries. |
| `keisei/training/tournament_queue.py` | DB ops for the pairing queue and worker heartbeat: `enqueue_pairings`, `claim_next_pairing`, `mark_pairing_done`, `get_round_status`, `write_worker_heartbeat`, `get_worker_health`, `claim_dynamic_update`, `release_dynamic_update`. Atomic-claim idioms mirror `keisei/showcase/db_ops.py`. |
| `tests/test_tournament_queue.py` | Atomic-claim correctness under concurrent threads (real threads in-process simulate multi-worker claims). Pairing lifecycle, worker health, Dynamic-update claim. |
| `tests/test_tournament_dispatcher.py` | Dispatcher unit tests: generate pairings, enqueue, advance round. Priority scorer state isolation. |
| `tests/test_tournament_worker.py` | Worker runner unit tests: claim → play → record loop, graceful shutdown on stop_event, crash-safe pairing status updates. |
| `tests/test_tournament_sidecar_integration.py` | End-to-end with a real `subprocess.Popen`'d worker: training enqueues, worker claims and plays, results land in DB. Slow — marked `@pytest.mark.integration`. |

### Modified files

| File | Changes |
|------|---------|
| `keisei/db.py` | Bump `SCHEMA_VERSION` to 4. Add `_migrate_v3_to_v4` (new tables, no data transform). Add `CREATE TABLE IF NOT EXISTS` for `tournament_pairing_queue`, `tournament_worker_heartbeat`. Add `dynamic_update_worker` TEXT column to `league_entries` via ALTER TABLE. |
| `keisei/training/opponent_store.py` | Replace existing `update_elo` with `BEGIN IMMEDIATE` / read-compute-write pattern to guarantee correctness under concurrent worker writes. Add `claim_dynamic_update` / `release_dynamic_update` helpers to `OpponentStore` or expose them from `tournament_queue.py`. |
| `keisei/training/katago_loop.py` | Remove `self._tournament: LeagueTournament` field and all `tournament.start()`, `tournament.stop()`, `tournament.learner_entry_id = …` calls. Replace with `self._dispatcher: TournamentDispatcher`. Dispatcher invocation at round boundaries (once per N epochs, configurable). `learner_entry_id` is written to a `training_state` singleton table on rotation. |
| `keisei/training/tournament.py` | Keep the existing `LeagueTournament` class intact as the "in-process" implementation path. Gated by a new `league.tournament_mode` config field (values: `"in_process"` [default, today's behavior] and `"sidecar"` [new]). When `"sidecar"`, training skips instantiating `LeagueTournament` entirely. |
| `keisei/config.py` | Add `LeagueConfig.tournament_mode: str = "in_process"` (values: `"in_process"` \| `"sidecar"`) with validation in `__post_init__`. Add `LeagueConfig.dispatcher_max_queue_depth: int = 400` (D1 — adaptive dispatch cap). Add `LeagueConfig.max_staleness_epochs: int = 50` (D2 — pairing expiration on claim). |
| `keisei-500k-league.toml` | After Phase 2 lands and is stable: set `tournament_mode = "sidecar"` to flip the user's active run to the new topology. Kept separate from the implementation tasks so the default behavior is unchanged for existing configs. |
| `run.sh` | Add tournament worker as a supervised child process, gated on `tournament_enabled == true && tournament_mode == "sidecar"` from the config (D3). New `--no-tournament` opt-out CLI flag parallel to `--no-showcase`. Parse `tournament_mode` and `tournament_device` in the existing TOML preflight block. Launch in both foreground and background modes. Add worker to the monitor loop's restart-on-death block and to `cleanup()` / `--stop` teardown. Separate log file. Details in **Task 11a**. |
| `tests/test_league_tournament.py` | Split: existing tests that cover the `LeagueTournament._run_loop` logic stay and gain an `@pytest.fixture` for the in-process path. New tests in `tests/test_tournament_worker.py` cover the sidecar-mode class. |

---

## Phase 2 requires NOT changing

- **`PriorityScorer`**: stays in the dispatcher (training-process side). The scorer holds in-memory round-to-round state (pair frequencies, under-sample bonuses, repeat penalties) that cannot be sharded across workers. Workers never touch it.
- **`MatchScheduler`**: stays in the dispatcher. Generates pairings, classifies matches, owns tier ratios. Workers consume already-generated pairings from the queue and do not need scheduler state.
- **`ConcurrentMatchPool`**: moves to the worker. Used by a single worker to run multiple matches in parallel within its own process. Each worker instantiates its own pool.
- **`DynamicTrainer`**: moves to the worker. Runs backprop on cuda:1 in the worker's CUDA context. Persists updated weights to disk; training picks them up on the next opponent pool refresh via the existing checkpoint-path polling path.
- **`HistoricalLibrary`**: stays in the dispatcher. The library advances on a schedule driven by training epoch count; the dispatcher is the natural owner.
- **`HistoricalGauntlet`**: stays in the dispatcher. The gauntlet reads the latest learner checkpoint from disk and runs a short evaluation; it's periodic and dispatcher-scoped (once per N rounds), not per-pairing. Workers never execute gauntlet matches.
- **`StyleProfiler`**: moves to the worker. Profile recomputation runs every 5 rounds inside today's `LeagueTournament._run_loop`. Moving it to the worker keeps it out of the training process. Alternative: also move to the dispatcher if worker-side CPU time becomes a concern, but initial cut keeps it where the match data was written.

---

## Resolved design decisions

Five open questions were discussed before this plan was locked. Decisions
recorded here so the implementation doesn't re-open them.

### D1. Dispatcher cadence — adaptive, queue-depth-gated

**Decision:** The dispatcher is called from training at every epoch boundary and only enqueues a new round if the current queue depth is below `dispatcher_max_queue_depth` (default **400**, ≈ 2 rounds of work at pool=20).

```python
# pseudocode inside the epoch loop
if self._dispatcher is not None:
    pending_plus_playing = get_active_queue_depth(db_path)
    if pending_plus_playing < self.config.league.dispatcher_max_queue_depth:
        self._dispatcher.enqueue_round(epoch=epoch_i)
```

**Why adaptive instead of fixed cadence:** self-regulating. If the worker is keeping up, the queue drains and the dispatcher enqueues roughly one round per epoch — matches today's continuous-loop behavior. If the worker is slow or down, the queue fills to cap and the dispatcher naturally stops. No need to tune interval counts; one parameter controls worst-case queue size.

**Rejected alternative:** fixed `dispatcher_interval_epochs` cadence. Less code but requires the user to know the right cadence up front and doesn't adapt to worker health.

### D2. Worker-down handling — cap, expire, warn; no dispatcher shutdown

**Decision:** Three defensive layers with no automatic dispatcher-side blocking.

1. **Queue cap (D1):** adaptive dispatch prevents unbounded growth. Worst case ≈ `dispatcher_max_queue_depth` rows (~400).
2. **Expire on claim (Task 7):** when a worker claims a pairing, if `current_epoch - pairing.enqueued_epoch > max_staleness_epochs` (default **50**, ~100 min at 2min/epoch), the pairing is marked `'expired'` and skipped rather than played against a pool snapshot that's evolved far past what the pairing represented.
3. **Heartbeat warnings only:** the dispatcher calls `get_worker_health(db, stale_after_seconds=60)` before enqueueing and logs a warning if zero healthy workers are visible. **It does NOT block enqueueing.** Training shouldn't care whether the tournament worker is running; blocking dispatch on worker health re-couples the two processes.
4. **Startup sweep:** on worker start, before entering the main loop, reset any `'playing'` rows owned by this worker_id (crash recovery). Then reset `'playing'` rows owned by any worker_id not currently in `get_worker_health()` (recover from other workers' crashes).

**Rejected alternative:** actively halt dispatching when no workers are alive. Couples training to tournament health in exactly the way this refactor is meant to prevent.

### D3. Launch convention — config-gated, launched by `run.sh`, matching showcase exactly

**Decision:** The tournament worker is launched as a supervised child of `run.sh`, following the **showcase sidecar pattern verbatim**. (Initial draft of this plan had the worker manually launched; that was wrong — showcase is config-gated and supervised, and tournament should follow the same convention for consistency.)

Launch gate:

```
launch_tournament_worker = (
    config.league.tournament_enabled == true
    AND config.league.tournament_mode == "sidecar"
    AND --no-tournament flag not passed
)
```

Concretely in `run.sh` (full details in **Task 11a**):

1. Parse `tournament_mode` and `tournament_device` from the TOML in the existing `eval "$(uv run python -c ...)"` block alongside `DB_PATH` / `USE_DDP`.
2. Add `--no-tournament` CLI option parallel to `--no-showcase`.
3. Build `TOURNAMENT_CMD` array:
   ```bash
   TOURNAMENT_CMD=(uv run python -m keisei.training.tournament_runner
                   --db-path "$DB_PATH"
                   --worker-id worker-0
                   --device "$TOURNAMENT_DEVICE")
   ```
4. Launch the worker wherever showcase is launched (foreground mode and `--background` mode), gated on `TOURNAMENT_ENABLED == true && TOURNAMENT_MODE == "sidecar" && NO_TOURNAMENT != true`.
5. Add the worker to the monitor loop's restart-on-death block (same pattern as showcase at run.sh:~360).
6. Add `TOURNAMENT_PID` to `cleanup()` and to the `--stop` pkill teardown.
7. Separate log file `${CONFIG_STEM}_tournament_${TIMESTAMP}.log`.

**Why this is better than manual launch:**
- `run.sh` already IS a supervisor — adding another supervised child is trivial.
- One-command ergonomics: `./run.sh config.toml` starts training + dashboard + showcase + tournament.
- Consistency: all sidecars follow one launch pattern.
- Process isolation is not affected — a worker crash is caught by the monitor loop and restarted, same as showcase/dashboard today. A training crash takes the worker with it via `cleanup()`, which is the correct behavior (stopping training should stop all children).
- Backward compatibility: configs without `tournament_mode` default to `"in_process"` and `run.sh` skips launching the worker — no behavior change for existing configs.

### D4. DynamicTrainer optimizer state — already solved, no plan change

**Decision:** No change required. Optimizer state is already persisted to disk after every `DynamicTrainer.update()` (dynamic_trainer.py:400 calls `self.store.save_optimizer(entry.id, optimizer.state_dict())`) and reloaded at the start of the next update (dynamic_trainer.py:227 calls `self.store.load_optimizer(entry_id)`). The `persist_optimizer_for_dynamic` config flag is validated as `must be True`.

Sidecar worker restart loads optimizer state from disk on its first update per entry, exactly as a crashed training process would today.

**What IS lost on worker crash:** the in-memory rollout buffer per Dynamic entry (`DynamicTrainer._rollout_buffers`, a bounded deque). A crash drops not-yet-used match data that was accumulating toward the `update_every_matches` threshold. This is a bounded self-healing loss — subsequent matches refill the buffer — and no Elo updates are lost (Elo writes land in the DB on match completion, before any Dynamic update).

### D5. Integration tests — one subprocess test, CPU-only, marked `@pytest.mark.integration`

**Decision:** Exactly one subprocess-based integration test (`tests/test_tournament_sidecar_integration.py`), running on CPU, marked with `@pytest.mark.integration` so it's excluded from default `pytest` runs and only runs via `uv run pytest -m integration`.

**Scope of the subprocess test:** catch IPC-layer bugs only — CLI arg parsing, signal handling, DB path resolution, subprocess exit codes. It launches the worker via `subprocess.Popen`, waits up to 60s for one pairing to reach `'done'`, then terminates the worker.

**Everything else is a unit test of the `TournamentWorker` class directly.** The worker constructor accepts injected `stop_event` and `vecenv_factory` arguments so its `run()` method is fully testable in-process with a mock VecEnv. No subprocess, no CUDA init, no process startup latency.

**Concurrent-claim correctness uses threaded tests, not process tests.** `threading.Thread` against a shared SQLite DB is sufficient to validate WAL-mode atomicity — SQLite's locking is process-agnostic, so thread-level correctness guarantees process-level correctness.

**Runbook gate:** the runbook (Task 11) includes a pre-merge checklist: run `uv run pytest -m integration` before merging any change to `tournament_runner.py` or `tournament_queue.py`.

---

## Task 1: Database schema additions

**Files:**
- Modify: `keisei/db.py` (schema version, migration, new tables, new column)
- Create: `tests/test_tournament_queue.py` (schema tests)

- [ ] **Step 1: Write failing test for schema version 4 and new tables**

```python
# tests/test_tournament_queue.py — schema portion
"""Tests for the tournament pairing queue and worker heartbeat tables."""
from __future__ import annotations

import sqlite3
from pathlib import Path
import pytest
from keisei.db import _connect, init_db


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


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

    def test_pairing_queue_status_index_exists(self, db: str) -> None:
        conn = _connect(db)
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND tbl_name='tournament_pairing_queue'"
            ).fetchall()
            names = {r["name"] for r in rows}
            assert "idx_pairing_queue_pending" in names
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

    def test_league_entries_has_dynamic_update_worker_column(self, db: str) -> None:
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tournament_queue.py::TestTournamentQueueSchema -v`

Expected: all 5 FAIL with "no such table" / "no such column" / "version == 3".

- [ ] **Step 3: Bump schema version and add migration + CREATE TABLE statements**

In `keisei/db.py`:

1. Change `SCHEMA_VERSION = 3` → `SCHEMA_VERSION = 4`.
2. Add a `_migrate_v3_to_v4` function that runs the ALTER TABLE for `dynamic_update_worker`:
   ```python
   def _migrate_v3_to_v4(conn: sqlite3.Connection) -> None:
       """v3 -> v4: Add tournament pairing queue, worker heartbeat,
       and dynamic_update_worker claim column on league_entries.
       
       New tables are created by init_db() via CREATE TABLE IF NOT EXISTS.
       Only the ALTER TABLE needs explicit migration.
       """
       cols = {r["name"] for r in conn.execute("PRAGMA table_info(league_entries)").fetchall()}
       if "dynamic_update_worker" not in cols:
           conn.execute("ALTER TABLE league_entries ADD COLUMN dynamic_update_worker TEXT")
   ```
3. Register it in `_MIGRATIONS`: `4: _migrate_v3_to_v4`.
4. Inside `init_db()`'s `executescript()` block, add:
   ```sql
   CREATE TABLE IF NOT EXISTS tournament_pairing_queue (
       id             INTEGER PRIMARY KEY AUTOINCREMENT,
       round_id       INTEGER NOT NULL,
       entry_a_id     INTEGER NOT NULL REFERENCES league_entries(id),
       entry_b_id     INTEGER NOT NULL REFERENCES league_entries(id),
       games_target   INTEGER NOT NULL,
       status         TEXT NOT NULL DEFAULT 'pending',
       worker_id      TEXT,
       claimed_at     TEXT,
       completed_at   TEXT,
       enqueued_epoch INTEGER NOT NULL,
       priority       REAL NOT NULL DEFAULT 0.0
   );
   CREATE INDEX IF NOT EXISTS idx_pairing_queue_pending
       ON tournament_pairing_queue (status, priority DESC, id);
   CREATE INDEX IF NOT EXISTS idx_pairing_queue_round
       ON tournament_pairing_queue (round_id);

   CREATE TABLE IF NOT EXISTS tournament_worker_heartbeat (
       worker_id      TEXT PRIMARY KEY,
       pid            INTEGER NOT NULL,
       device         TEXT NOT NULL,
       last_seen      TEXT NOT NULL,
       pairings_done  INTEGER NOT NULL DEFAULT 0
   );
   ```
5. For fresh databases, the `dynamic_update_worker` column is added to the `league_entries` CREATE TABLE statement directly (so new DBs don't need the ALTER path).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tournament_queue.py::TestTournamentQueueSchema -v`

Expected: all 5 PASS.

- [ ] **Step 5: Verify existing schema migration tests still pass**

Run: `uv run pytest tests/test_db.py tests/test_league_config.py -v`

Expected: no regressions. If `tests/test_db.py` asserts `SCHEMA_VERSION == 3` anywhere, update those assertions to 4.

---

## Task 2: Pairing queue DB operations

**Files:**
- Create: `keisei/training/tournament_queue.py` (module with DB ops)
- Modify: `tests/test_tournament_queue.py` (add op tests)

- [ ] **Step 1: Write failing tests for enqueue, claim, mark_done**

Add to `tests/test_tournament_queue.py`:

```python
from keisei.training.tournament_queue import (
    enqueue_pairings,
    claim_next_pairing,
    mark_pairing_done,
    get_round_status,
)


class TestPairingQueueOps:
    def _seed_entries(self, db: str) -> tuple[int, int, int]:
        """Insert 3 minimal league_entries rows and return their IDs."""
        conn = _connect(db)
        try:
            ids = []
            for i in range(3):
                cur = conn.execute(
                    "INSERT INTO league_entries "
                    "(display_name, architecture, model_params_json, "
                    "checkpoint_path, elo_rating, created_epoch, "
                    "games_played, created_at, flavour_facts_json, "
                    "role, status, update_count) "
                    "VALUES (?, 'resnet', '{}', ?, 1000.0, ?, 0, "
                    "'2026-01-01', '[]', 'dynamic', 'active', 0)",
                    (f"entry{i}", f"/p/{i}.pt", i),
                )
                ids.append(cur.lastrowid)
            conn.commit()
            return tuple(ids)
        finally:
            conn.close()

    def test_enqueue_pairings_inserts_rows(self, db: str) -> None:
        a, b, c = self._seed_entries(db)
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(a, b, 3), (a, c, 3)],
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

    def test_claim_next_pairing_returns_one(self, db: str) -> None:
        a, b, c = self._seed_entries(db)
        enqueue_pairings(
            db, round_id=1, epoch=5, pairings=[(a, b, 3), (a, c, 3)],
        )
        claimed = claim_next_pairing(db, worker_id="w0")
        assert claimed is not None
        assert claimed.worker_id == "w0"
        assert claimed.status == "playing"
        # Second worker claims the other pairing
        claimed2 = claim_next_pairing(db, worker_id="w1")
        assert claimed2 is not None
        assert claimed2.id != claimed.id
        # No more pairings
        assert claim_next_pairing(db, worker_id="w2") is None

    def test_claim_is_atomic_under_concurrent_workers(self, db: str) -> None:
        """Two threads racing claim_next_pairing must never claim the same row."""
        import threading
        a, b, _ = self._seed_entries(db)
        # Enqueue 100 pairings
        enqueue_pairings(
            db, round_id=1, epoch=5,
            pairings=[(a, b, 1) for _ in range(100)],
        )
        claimed_by_worker: dict[str, list[int]] = {"w0": [], "w1": [], "w2": [], "w3": []}
        lock = threading.Lock()

        def worker_loop(wid: str) -> None:
            while True:
                claim = claim_next_pairing(db, worker_id=wid)
                if claim is None:
                    return
                with lock:
                    claimed_by_worker[wid].append(claim.id)

        threads = [threading.Thread(target=worker_loop, args=(w,)) for w in claimed_by_worker]
        for t in threads: t.start()
        for t in threads: t.join()

        all_ids = [i for lst in claimed_by_worker.values() for i in lst]
        assert len(all_ids) == 100
        assert len(set(all_ids)) == 100  # every ID claimed exactly once

    def test_mark_pairing_done_updates_status(self, db: str) -> None:
        a, b, _ = self._seed_entries(db)
        enqueue_pairings(db, round_id=1, epoch=5, pairings=[(a, b, 3)])
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

    def test_get_round_status_counts_by_state(self, db: str) -> None:
        a, b, c = self._seed_entries(db)
        enqueue_pairings(
            db, round_id=7, epoch=5,
            pairings=[(a, b, 3), (a, c, 3), (b, c, 3)],
        )
        # Claim one, complete another
        claim = claim_next_pairing(db, worker_id="w0")
        assert claim is not None
        mark_pairing_done(db, claim.id, status="done")
        claim2 = claim_next_pairing(db, worker_id="w0")
        assert claim2 is not None
        # 1 pending, 1 playing, 1 done
        status = get_round_status(db, round_id=7)
        assert status == {"pending": 1, "playing": 1, "done": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tournament_queue.py::TestPairingQueueOps -v`

Expected: `ImportError: cannot import name 'enqueue_pairings' from 'keisei.training.tournament_queue'`.

- [ ] **Step 3: Implement `keisei/training/tournament_queue.py`**

```python
"""Database operations for the tournament pairing queue and worker heartbeat.

All operations are idempotent where possible and use atomic SQL patterns
(conditional UPDATE, BEGIN IMMEDIATE) to guarantee correctness under
concurrent workers sharing the same SQLite database in WAL mode.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from keisei.db import _connect


@dataclass(frozen=True)
class ClaimedPairing:
    id: int
    round_id: int
    entry_a_id: int
    entry_b_id: int
    games_target: int
    worker_id: str
    status: str
    enqueued_epoch: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enqueue_pairings(
    db_path: str,
    *,
    round_id: int,
    epoch: int,
    pairings: list[tuple[int, int, int]],  # (a_id, b_id, games_target)
    priorities: list[float] | None = None,
) -> None:
    """Insert a batch of pairings for a round.

    ``priorities`` is optional; when provided it must have the same length
    as ``pairings`` and governs claim order (highest first).
    """
    if not pairings:
        return
    if priorities is not None and len(priorities) != len(pairings):
        raise ValueError("priorities length must match pairings length")
    now = _now_iso()
    rows = [
        (
            round_id, a, b, games_target,
            (priorities[i] if priorities is not None else 0.0),
            epoch, now,
        )
        for i, (a, b, games_target) in enumerate(pairings)
    ]
    conn = _connect(db_path)
    try:
        with conn:  # transaction
            conn.executemany(
                "INSERT INTO tournament_pairing_queue "
                "(round_id, entry_a_id, entry_b_id, games_target, "
                "priority, enqueued_epoch, claimed_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, 'pending')",
                [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows],
            )
    finally:
        conn.close()


def claim_next_pairing(
    db_path: str, *, worker_id: str,
) -> ClaimedPairing | None:
    """Atomically claim the highest-priority pending pairing.

    Uses a conditional UPDATE that only writes if the row is still pending,
    so two concurrent workers can never claim the same row. Returns None
    if no pending pairings remain.
    """
    conn = _connect(db_path)
    try:
        # BEGIN IMMEDIATE acquires the writer lock up front, serializing
        # the SELECT + UPDATE against other workers even under WAL.
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT id FROM tournament_pairing_queue "
                "WHERE status = 'pending' "
                "ORDER BY priority DESC, id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return None
            now = _now_iso()
            conn.execute(
                "UPDATE tournament_pairing_queue "
                "SET status = 'playing', worker_id = ?, claimed_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (worker_id, now, row["id"]),
            )
            full = conn.execute(
                "SELECT * FROM tournament_pairing_queue WHERE id = ?",
                (row["id"],),
            ).fetchone()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return ClaimedPairing(
            id=full["id"], round_id=full["round_id"],
            entry_a_id=full["entry_a_id"], entry_b_id=full["entry_b_id"],
            games_target=full["games_target"], worker_id=full["worker_id"],
            status=full["status"], enqueued_epoch=full["enqueued_epoch"],
        )
    finally:
        conn.close()


def mark_pairing_done(
    db_path: str, pairing_id: int, *, status: str = "done",
) -> None:
    """Mark a pairing as 'done' or 'failed'. Sets completed_at."""
    if status not in ("done", "failed", "expired"):
        raise ValueError(f"invalid status: {status!r}")
    now = _now_iso()
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE tournament_pairing_queue "
                "SET status = ?, completed_at = ? WHERE id = ?",
                (status, now, pairing_id),
            )
    finally:
        conn.close()


def get_round_status(db_path: str, round_id: int) -> dict[str, int]:
    """Return {status: count} for a given round."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) as n FROM tournament_pairing_queue "
            "WHERE round_id = ? GROUP BY status",
            (round_id,),
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tournament_queue.py::TestPairingQueueOps -v`

Expected: all 5 PASS, including the concurrent-claim test (which would catch any non-atomic claim pattern).

---

## Task 3: Worker heartbeat operations

**Files:**
- Modify: `keisei/training/tournament_queue.py` (add heartbeat ops)
- Modify: `tests/test_tournament_queue.py` (add heartbeat tests)

- [ ] **Step 1: Write failing tests**

```python
from keisei.training.tournament_queue import (
    write_worker_heartbeat,
    get_worker_health,
)


class TestWorkerHeartbeat:
    def test_write_heartbeat_inserts_row(self, db: str) -> None:
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1")
        health = get_worker_health(db, stale_after_seconds=60)
        assert len(health) == 1
        assert health[0].worker_id == "w0"
        assert health[0].pid == 1234
        assert health[0].device == "cuda:1"

    def test_write_heartbeat_upserts(self, db: str) -> None:
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1")
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1")
        conn = _connect(db)
        try:
            rows = conn.execute(
                "SELECT COUNT(*) as n FROM tournament_worker_heartbeat"
            ).fetchone()
            assert rows["n"] == 1  # upsert, not duplicate insert
        finally:
            conn.close()

    def test_pairings_done_increments(self, db: str) -> None:
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1", pairings_done=5)
        health = get_worker_health(db, stale_after_seconds=60)
        assert health[0].pairings_done == 5

    def test_stale_workers_excluded_from_health(self, db: str) -> None:
        """get_worker_health filters by last_seen age."""
        import time
        write_worker_heartbeat(db, worker_id="w0", pid=1234, device="cuda:1")
        # With stale_after_seconds=0, the worker should be considered stale
        health = get_worker_health(db, stale_after_seconds=0)
        # Implementation note: if last_seen is "right now" with second resolution,
        # stale_after=0 may still see it as fresh. Use small negative number to
        # force stale classification, or sleep. Prefer: mock the time source.
        time.sleep(1.1)
        health_after = get_worker_health(db, stale_after_seconds=1)
        assert len(health_after) == 0
```

- [ ] **Step 2: Verify tests fail** (missing imports).

- [ ] **Step 3: Implement heartbeat ops in `tournament_queue.py`**

```python
@dataclass(frozen=True)
class WorkerHealth:
    worker_id: str
    pid: int
    device: str
    last_seen: str
    pairings_done: int


def write_worker_heartbeat(
    db_path: str, *,
    worker_id: str, pid: int, device: str, pairings_done: int | None = None,
) -> None:
    """Upsert a worker heartbeat row. Sets last_seen to now."""
    now = _now_iso()
    conn = _connect(db_path)
    try:
        with conn:
            if pairings_done is None:
                conn.execute(
                    "INSERT INTO tournament_worker_heartbeat "
                    "(worker_id, pid, device, last_seen, pairings_done) "
                    "VALUES (?, ?, ?, ?, 0) "
                    "ON CONFLICT(worker_id) DO UPDATE SET "
                    "pid = excluded.pid, device = excluded.device, "
                    "last_seen = excluded.last_seen",
                    (worker_id, pid, device, now),
                )
            else:
                conn.execute(
                    "INSERT INTO tournament_worker_heartbeat "
                    "(worker_id, pid, device, last_seen, pairings_done) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(worker_id) DO UPDATE SET "
                    "pid = excluded.pid, device = excluded.device, "
                    "last_seen = excluded.last_seen, "
                    "pairings_done = excluded.pairings_done",
                    (worker_id, pid, device, now, pairings_done),
                )
    finally:
        conn.close()


def get_worker_health(
    db_path: str, *, stale_after_seconds: int = 60,
) -> list[WorkerHealth]:
    """Return workers whose last_seen is newer than (now - stale_after_seconds)."""
    conn = _connect(db_path)
    try:
        cutoff = datetime.now(timezone.utc).timestamp() - stale_after_seconds
        rows = conn.execute(
            "SELECT * FROM tournament_worker_heartbeat"
        ).fetchall()
        alive = []
        for r in rows:
            try:
                last = datetime.fromisoformat(r["last_seen"]).timestamp()
            except ValueError:
                continue
            if last >= cutoff:
                alive.append(WorkerHealth(
                    worker_id=r["worker_id"], pid=r["pid"],
                    device=r["device"], last_seen=r["last_seen"],
                    pairings_done=r["pairings_done"],
                ))
        return alive
    finally:
        conn.close()
```

- [ ] **Step 4: Verify tests pass**.

---

## Task 4: DynamicTrainer single-writer claim operations

**Files:**
- Modify: `keisei/training/tournament_queue.py` (add `claim_dynamic_update`, `release_dynamic_update`)
- Modify: `tests/test_tournament_queue.py` (add claim tests)

- [ ] **Step 1: Write failing tests**

```python
from keisei.training.tournament_queue import (
    claim_dynamic_update,
    release_dynamic_update,
)


class TestDynamicUpdateClaim:
    def test_claim_returns_true_when_available(self, db: str) -> None:
        a, _, _ = self._seed_entries(db)
        assert claim_dynamic_update(db, entry_id=a, worker_id="w0") is True

    def test_claim_returns_false_when_held(self, db: str) -> None:
        a, _, _ = self._seed_entries(db)
        assert claim_dynamic_update(db, entry_id=a, worker_id="w0") is True
        assert claim_dynamic_update(db, entry_id=a, worker_id="w1") is False

    def test_release_allows_reclaim(self, db: str) -> None:
        a, _, _ = self._seed_entries(db)
        assert claim_dynamic_update(db, entry_id=a, worker_id="w0") is True
        release_dynamic_update(db, entry_id=a, worker_id="w0")
        assert claim_dynamic_update(db, entry_id=a, worker_id="w1") is True

    def test_release_no_op_when_worker_mismatch(self, db: str) -> None:
        """Releasing a claim held by a different worker is a no-op (guards
        against stale release after crash + respawn)."""
        a, _, _ = self._seed_entries(db)
        claim_dynamic_update(db, entry_id=a, worker_id="w0")
        release_dynamic_update(db, entry_id=a, worker_id="w1")  # wrong owner
        assert claim_dynamic_update(db, entry_id=a, worker_id="w2") is False

    def test_concurrent_claim_single_winner(self, db: str) -> None:
        """Two threads racing the same entry — exactly one wins."""
        import threading
        a, _, _ = self._seed_entries(db)
        winners = []
        lock = threading.Lock()

        def attempt(wid: str) -> None:
            got = claim_dynamic_update(db, entry_id=a, worker_id=wid)
            if got:
                with lock:
                    winners.append(wid)

        threads = [threading.Thread(target=attempt, args=(f"w{i}",)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(winners) == 1
```

- [ ] **Step 2: Verify tests fail**.

- [ ] **Step 3: Implement claim/release**

```python
def claim_dynamic_update(
    db_path: str, *, entry_id: int, worker_id: str,
) -> bool:
    """Atomically claim the Dynamic update slot for an entry.

    Returns True if this worker successfully claimed the slot, False if
    another worker already holds it. The claim is a single UPDATE with
    WHERE dynamic_update_worker IS NULL — if two workers race, exactly
    one will see rows affected = 1.
    """
    conn = _connect(db_path)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE league_entries SET dynamic_update_worker = ? "
                "WHERE id = ? AND dynamic_update_worker IS NULL",
                (worker_id, entry_id),
            )
            return cur.rowcount == 1
    finally:
        conn.close()


def release_dynamic_update(
    db_path: str, *, entry_id: int, worker_id: str,
) -> None:
    """Release the Dynamic update claim. No-op if the claim is held by
    a different worker (guards against stale release after respawn)."""
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE league_entries SET dynamic_update_worker = NULL "
                "WHERE id = ? AND dynamic_update_worker = ?",
                (entry_id, worker_id),
            )
    finally:
        conn.close()
```

- [ ] **Step 4: Verify tests pass**.

---

## Task 5: Concurrency-safe Elo updates

**Files:**
- Modify: `keisei/training/opponent_store.py` (rewrite `update_elo` with `BEGIN IMMEDIATE`)
- Modify: `tests/test_opponent_store.py` (concurrent Elo test)

**Problem:** `update_elo` today reads `elo_rating`, computes the new value, writes it back. Under a single in-process thread this is fine. Under concurrent workers, two workers reading 1500 and writing 1510 / 1520 respectively loses one update entirely.

**Fix:** Wrap the read-compute-write in `BEGIN IMMEDIATE`. SQLite in WAL mode serializes writers, so `BEGIN IMMEDIATE` blocks other writers until commit, guaranteeing the RMW is atomic.

- [ ] **Step 1: Write a failing test for concurrent Elo updates**

```python
# tests/test_opponent_store.py — add concurrency test
class TestConcurrentEloUpdates:
    def test_two_workers_racing_elo_do_not_lose_updates(self, tmp_path):
        """Two threads each apply +10 to the same entry's Elo; final
        value must be 1020, not 1010 or 1000 (lost update detection)."""
        import threading
        db = str(tmp_path / "test.db")
        init_db(db)
        store = OpponentStore(db_path=db)
        entry_id = store.create_entry(
            display_name="test", architecture="resnet",
            model_params={}, checkpoint_path="/p/0.pt",
            created_epoch=0, role=Role.DYNAMIC,
        )

        def worker() -> None:
            # Simulate a worker applying a +10 Elo delta via the
            # concurrency-safe update_elo method.
            current = store.get_entry(entry_id)
            new_elo = current.elo_rating + 10.0
            store.update_elo(entry_id, new_elo, epoch=0)

        # Launch 100 workers each applying +10. With correct locking,
        # the final Elo should be 1000 + 100*10 = 2000.
        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads: t.start()
        for t in threads: t.join()
        final = store.get_entry(entry_id)
        assert final.elo_rating == 2000.0, (
            f"Expected 2000.0 but got {final.elo_rating}: "
            f"{int((2000.0 - final.elo_rating) / 10.0)} updates were lost"
        )
```

This test as-written will still fail under the proposed `BEGIN IMMEDIATE` fix because `update_elo` takes a pre-computed value, not a delta. To actually catch lost updates, the test needs the read and the compute inside the locked transaction.

Two options:
- **(a)** Make `update_elo` take a callback `compute: (old_elo) -> new_elo` and run it inside the transaction. Caller-side computation happens with the lock held.
- **(b)** Add a new method `update_elo_with_fn(entry_id, compute_fn, epoch)` that does the read+compute+write atomically, and migrate the tournament result-recording paths to use it.

Recommendation: **(b)**, because it keeps `update_elo` compatible with existing callers (training's in-process Elo updates use pre-computed values). Only the worker's result-recording path needs the atomic read-modify-write.

Revised test:

```python
def test_two_workers_racing_elo_do_not_lose_updates(self, tmp_path):
    import threading
    db = str(tmp_path / "test.db")
    init_db(db)
    store = OpponentStore(db_path=db)
    entry_id = store.create_entry(...)

    def worker() -> None:
        store.update_elo_atomic(
            entry_id, compute=lambda old: old + 10.0, epoch=0,
        )

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads: t.start()
    for t in threads: t.join()
    final = store.get_entry(entry_id)
    assert final.elo_rating == 2000.0
```

- [ ] **Step 2: Verify test fails** (method doesn't exist).

- [ ] **Step 3: Implement `update_elo_atomic` in `OpponentStore`**

```python
def update_elo_atomic(
    self, entry_id: int,
    compute: Callable[[float], float],
    *, epoch: int = 0,
) -> float:
    """Atomically update an entry's Elo via a caller-supplied compute function.
    
    Reads the current Elo, applies ``compute(old_elo) -> new_elo``, and
    writes the result — all inside a BEGIN IMMEDIATE transaction so
    concurrent writers never lose updates. Returns the new Elo value.
    
    Use this path for tournament worker result recording. The non-atomic
    update_elo remains for training-side updates that compute the new
    value from in-memory state the worker doesn't see.
    """
    conn = self._get_conn()  # thread-local connection
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT elo_rating FROM league_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            raise KeyError(f"entry {entry_id} not found")
        old_elo = row["elo_rating"]
        new_elo = compute(old_elo)
        conn.execute(
            "UPDATE league_entries SET elo_rating = ? WHERE id = ?",
            (new_elo, entry_id),
        )
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) "
            "VALUES (?, ?, ?)",
            (entry_id, epoch, new_elo),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return new_elo
```

- [ ] **Step 4: Verify test passes.**

- [ ] **Step 5: Migrate `tournament.py::_record_match_result` to use `update_elo_atomic`**

The existing path computes `new_a_elo, new_b_elo = compute_elo_update(...)` from in-memory values and then calls `store.update_elo(...)`. Replace with two calls to `store.update_elo_atomic` passing a compute function that takes the old Elo:

```python
def _record_match_result(self, ...) -> bool:
    ...
    def compute_a(old_elo: float) -> float:
        new_a, _ = compute_elo_update(old_elo, current_b.elo_rating, result=result_score, k=k)
        return new_a
    def compute_b(old_elo: float) -> float:
        _, new_b = compute_elo_update(current_a.elo_rating, old_elo, result=result_score, k=k)
        return new_b
    self.store.update_elo_atomic(entry_a_id, compute_a, epoch=epoch)
    self.store.update_elo_atomic(entry_b_id, compute_b, epoch=epoch)
```

Note: this isn't a true rating system — pairing both sides of an Elo exchange through independent atomic transactions can drift if two unrelated matches involving the same entry interleave. A true fix requires computing both sides inside a single transaction that locks both rows. For initial rollout, **defer that refinement** and accept the minor drift (Elo is a statistical estimate anyway; single-vote errors wash out over hundreds of matches). File a follow-up issue to improve rating-pair atomicity after the sidecar is deployed and measured.

---

## Task 6: Tournament dispatcher

**Files:**
- Create: `keisei/training/tournament_dispatcher.py`
- Create: `tests/test_tournament_dispatcher.py`

- [ ] **Step 1: Write failing tests for the dispatcher**

```python
# tests/test_tournament_dispatcher.py
from keisei.training.tournament_dispatcher import TournamentDispatcher
from keisei.training.match_scheduler import MatchScheduler
from keisei.training.opponent_store import OpponentStore
from keisei.training.tournament_queue import get_round_status

class TestTournamentDispatcher:
    def test_enqueues_round_pairings(self, tmp_path):
        db = str(tmp_path / "test.db")
        # ... set up store with 5 entries ...
        sched = MatchScheduler(MatchSchedulerConfig())
        dispatcher = TournamentDispatcher(store=store, scheduler=sched, games_per_match=3)
        round_id = dispatcher.enqueue_round(epoch=5)
        status = get_round_status(db, round_id=round_id)
        assert status["pending"] == 10  # C(5,2) = 10 pairings

    def test_round_id_monotonic(self, tmp_path):
        # Two rounds should get distinct round_ids
        ...

    def test_skip_when_pool_too_small(self, tmp_path):
        # Pool size < min_pool_size → returns None, no enqueue
        ...

    def test_skip_when_queue_full(self, tmp_path):
        # If pending pairings >= max_queue_depth → skip and log warning
        ...

    def test_advance_round_triggers_priority_scorer(self, tmp_path):
        # When all pairings in a round complete, dispatcher calls
        # priority_scorer.advance_round() exactly once
        ...
```

- [ ] **Step 2: Verify tests fail**.

- [ ] **Step 3: Implement `TournamentDispatcher`**

Key points:
- Holds references to `MatchScheduler`, `OpponentStore`, `PriorityScorer`, `HistoricalGauntlet`, `HistoricalLibrary`.
- `enqueue_round(epoch)` → list entries, call `scheduler.generate_round(...)`, convert to queue rows via `enqueue_pairings`, return round_id.
- `advance_round(round_id)` → check `get_round_status`; if all done, call `priority_scorer.advance_round()`, trigger gauntlet-if-due, etc.
- Round IDs are monotonic integers persisted in a `tournament_round_counter` singleton table (or derived from `MAX(round_id) + 1` on the queue).
- Observability: log round generation and advance.

(Full implementation sketch — approximately 200 lines, mirrors today's `_run_loop` body sans the actual match playing.)

- [ ] **Step 4: Verify tests pass**.

---

## Task 7: Tournament worker runner

**Files:**
- Create: `keisei/training/tournament_runner.py`
- Create: `tests/test_tournament_worker.py`

- [ ] **Step 1: Write failing tests for the worker class**

```python
# tests/test_tournament_worker.py
from keisei.training.tournament_runner import TournamentWorker

class TestTournamentWorker:
    def test_worker_loop_claims_and_plays_one_pairing(self, tmp_path, mock_vecenv):
        """Inject a mock VecEnv and model loader; verify the worker's
        inner loop claims a pairing, plays the match, and marks it done."""
        db = str(tmp_path / "test.db")
        # Seed DB with 2 entries and 1 queued pairing
        ...
        worker = TournamentWorker(
            db_path=db, worker_id="w0", device="cpu",
            vecenv_factory=lambda: mock_vecenv,
            stop_after_pairings=1,  # test harness
        )
        worker.run()
        status = get_round_status(db, round_id=1)
        assert status == {"done": 1}

    def test_worker_heartbeats_during_loop(self, tmp_path, mock_vecenv):
        ...

    def test_worker_gracefully_stops_on_stop_event(self, tmp_path, mock_vecenv):
        ...

    def test_worker_marks_failed_on_exception(self, tmp_path, mock_vecenv):
        """If play_match raises, the pairing is marked 'failed' so it
        doesn't stay in 'playing' state forever."""
        ...

    def test_worker_skips_expired_pairings(self, tmp_path, mock_vecenv):
        """Pairings older than max_staleness_epochs are marked expired
        and not played."""
        ...
```

- [ ] **Step 2: Verify tests fail**.

- [ ] **Step 3: Implement `TournamentWorker`**

```python
"""Tournament worker sidecar: claims pairings from the queue and plays them.

Usage:
    python -m keisei.training.tournament_runner \\
        --db-path data/keisei-500k-league.db \\
        --device cuda:1 \\
        --worker-id worker-0 \\
        [--games-per-match 3] [--max-cached 22]

The runner is a thin CLI wrapper around the TournamentWorker class.
TournamentWorker is unit-testable (inject stop events, mock vecenv).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from typing import Callable

import torch

from keisei.training.concurrent_matches import ConcurrentMatchPool
from keisei.training.dynamic_trainer import DynamicTrainer
from keisei.training.opponent_store import OpponentStore
from keisei.training.tournament_queue import (
    claim_dynamic_update,
    claim_next_pairing,
    mark_pairing_done,
    release_dynamic_update,
    write_worker_heartbeat,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 2.0
HEARTBEAT_INTERVAL_SECONDS = 10.0


class TournamentWorker:
    """Main loop for a single tournament worker process."""

    def __init__(
        self, *,
        db_path: str,
        worker_id: str,
        device: str,
        games_per_match: int = 3,
        max_ply: int = 512,
        max_cached: int = 22,
        max_staleness_epochs: int = 50,
        stop_event: threading.Event | None = None,
        vecenv_factory: Callable[[], object] | None = None,
    ) -> None:
        self.db_path = db_path
        self.worker_id = worker_id
        self.device = torch.device(device)
        self.games_per_match = games_per_match
        self.max_ply = max_ply
        self.max_cached = max_cached
        self.max_staleness_epochs = max_staleness_epochs
        self._stop_event = stop_event or threading.Event()
        self._vecenv_factory = vecenv_factory
        self._pairings_done = 0
        self._last_heartbeat = 0.0

        self.store = OpponentStore(db_path=db_path)
        # DynamicTrainer is optional — only needed if any Dynamic entries
        # are in the pool. Constructed lazily on first Dynamic match.
        self._dynamic_trainer: DynamicTrainer | None = None

    def _maybe_heartbeat(self) -> None:
        now = time.monotonic()
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
            write_worker_heartbeat(
                self.db_path, worker_id=self.worker_id,
                pid=os.getpid(), device=str(self.device),
                pairings_done=self._pairings_done,
            )
            self._last_heartbeat = now

    def run(self) -> None:
        """Main loop: claim → play → record → heartbeat → repeat."""
        logger.info(
            "Tournament worker %s starting on %s",
            self.worker_id, self.device,
        )
        self._maybe_heartbeat()

        # Lazy-import so the module can be imported without shogi_gym
        if self._vecenv_factory is None:
            from shogi_gym import VecEnv
            def _default_factory() -> object:
                return VecEnv(
                    num_envs=64, max_ply=self.max_ply,
                    observation_mode="katago",
                    action_mode="spatial",
                )
            self._vecenv_factory = _default_factory
        vecenv = self._vecenv_factory()

        try:
            while not self._stop_event.is_set():
                self._maybe_heartbeat()
                claim = claim_next_pairing(
                    self.db_path, worker_id=self.worker_id,
                )
                if claim is None:
                    self._stop_event.wait(POLL_INTERVAL_SECONDS)
                    continue
                # Check staleness
                current_epoch = self.store.get_current_epoch()
                if current_epoch - claim.enqueued_epoch > self.max_staleness_epochs:
                    logger.info(
                        "Expiring stale pairing %d (enqueued_epoch=%d, "
                        "current_epoch=%d)",
                        claim.id, claim.enqueued_epoch, current_epoch,
                    )
                    mark_pairing_done(self.db_path, claim.id, status="expired")
                    continue
                try:
                    self._play_claimed_pairing(vecenv, claim, current_epoch)
                    mark_pairing_done(self.db_path, claim.id, status="done")
                    self._pairings_done += 1
                except Exception:
                    logger.exception(
                        "Failed to play pairing %d (%d vs %d)",
                        claim.id, claim.entry_a_id, claim.entry_b_id,
                    )
                    mark_pairing_done(self.db_path, claim.id, status="failed")
        finally:
            logger.info("Tournament worker %s stopping", self.worker_id)

    def _play_claimed_pairing(self, vecenv, claim, epoch: int) -> None:
        """Play a single claimed pairing and write results to the DB.
        
        Reuses the existing `play_match` helper from match_utils and the
        existing `_record_match_result` logic from tournament.py, but
        rerouted to go through update_elo_atomic and with DynamicTrainer
        update gated by claim_dynamic_update.
        """
        # ... detailed implementation: load models, play, record, update Elo,
        #     trigger DynamicTrainer update under single-writer claim ...


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tournament worker sidecar")
    p.add_argument("--db-path", required=True)
    p.add_argument("--worker-id", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--games-per-match", type=int, default=3)
    p.add_argument("--max-ply", type=int, default=512)
    p.add_argument("--max-cached", type=int, default=22)
    p.add_argument("--max-staleness-epochs", type=int, default=50)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    stop_event = threading.Event()

    def _handle_signal(signum, frame):
        logger.info("Received signal %d, stopping worker gracefully", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    worker = TournamentWorker(
        db_path=args.db_path,
        worker_id=args.worker_id,
        device=args.device,
        games_per_match=args.games_per_match,
        max_ply=args.max_ply,
        max_cached=args.max_cached,
        max_staleness_epochs=args.max_staleness_epochs,
        stop_event=stop_event,
    )
    worker.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Verify tests pass**.

---

## Task 8: Config fields for tournament mode and adaptive dispatch

**Files:**
- Modify: `keisei/config.py` (add `tournament_mode`, `dispatcher_max_queue_depth`, `max_staleness_epochs` to `LeagueConfig`)
- Modify: `tests/test_league_config.py` (validation tests)

- [ ] **Step 1: Add failing validation tests**

```python
def test_tournament_mode_defaults_to_in_process():
    cfg = LeagueConfig()
    assert cfg.tournament_mode == "in_process"

def test_tournament_mode_accepts_sidecar():
    cfg = LeagueConfig(tournament_mode="sidecar")
    assert cfg.tournament_mode == "sidecar"

def test_tournament_mode_rejects_invalid():
    with pytest.raises(ValueError, match="tournament_mode"):
        LeagueConfig(tournament_mode="nonsense")

def test_dispatcher_max_queue_depth_default():
    cfg = LeagueConfig()
    assert cfg.dispatcher_max_queue_depth == 400

def test_dispatcher_max_queue_depth_must_be_positive():
    with pytest.raises(ValueError, match="dispatcher_max_queue_depth"):
        LeagueConfig(dispatcher_max_queue_depth=0)

def test_max_staleness_epochs_default():
    cfg = LeagueConfig()
    assert cfg.max_staleness_epochs == 50

def test_max_staleness_epochs_must_be_positive():
    with pytest.raises(ValueError, match="max_staleness_epochs"):
        LeagueConfig(max_staleness_epochs=0)
```

- [ ] **Step 2: Verify tests fail**.

- [ ] **Step 3: Add the fields and validation**

```python
# In LeagueConfig
tournament_mode: str = "in_process"           # "in_process" | "sidecar"
dispatcher_max_queue_depth: int = 400          # D1: adaptive dispatch cap
max_staleness_epochs: int = 50                 # D2: pairing expiration threshold

# In __post_init__
if self.tournament_mode not in ("in_process", "sidecar"):
    raise ValueError(
        f"tournament_mode must be 'in_process' or 'sidecar', "
        f"got {self.tournament_mode!r}"
    )
if self.dispatcher_max_queue_depth < 1:
    raise ValueError(
        f"dispatcher_max_queue_depth must be >= 1, "
        f"got {self.dispatcher_max_queue_depth}"
    )
if self.max_staleness_epochs < 1:
    raise ValueError(
        f"max_staleness_epochs must be >= 1, "
        f"got {self.max_staleness_epochs}"
    )
```

- [ ] **Step 4: Verify tests pass**.

---

## Task 9: Training-side integration

**Files:**
- Modify: `keisei/training/katago_loop.py` (gate tournament/dispatcher selection on `tournament_mode`)
- Modify: `tests/test_katago_loop_integration.py` (assertion updates)

- [ ] **Step 1: Write failing integration test**

```python
def test_sidecar_mode_does_not_start_tournament_thread(tmp_path):
    """When tournament_mode='sidecar', the training loop does NOT
    instantiate LeagueTournament — the dispatcher is used instead."""
    cfg = _minimal_league_config(tournament_mode="sidecar")
    loop = KataGoTrainingLoop(config=cfg, ...)
    assert loop._tournament is None
    assert loop._dispatcher is not None
```

- [ ] **Step 2: Gate tournament instantiation in katago_loop.py**

Around line 660 (where `self._tournament` is conditionally constructed), replace with:

```python
self._tournament: LeagueTournament | None = None
self._dispatcher: TournamentDispatcher | None = None

if config.league is not None and config.league.enabled and self.dist_ctx.is_main:
    if config.league.tournament_mode == "in_process":
        if config.league.tournament_enabled:
            # Existing in-process path — unchanged.
            self._tournament = LeagueTournament(...)
    elif config.league.tournament_mode == "sidecar":
        self._dispatcher = TournamentDispatcher(
            store=self.store,
            scheduler=self.scheduler,
            priority_scorer=priority_scorer,
            games_per_match=config.league.tournament_games_per_match,
            historical_library=historical_library,
            gauntlet=gauntlet,
        )
```

- [ ] **Step 3: Call dispatcher at round boundaries (adaptive, per D1)**

Add to the epoch loop (inside `_run_training_body`, near the end of each epoch). The dispatcher is called every epoch but only actually enqueues a round when the current queue depth is below the configured cap — adaptive dispatch per **D1**. Also emits a warning if no healthy workers are visible (per **D2**); does NOT block enqueueing.

```python
if self._dispatcher is not None:
    try:
        from keisei.training.tournament_queue import (
            get_active_queue_depth, get_worker_health,
        )
        cap = self.config.league.dispatcher_max_queue_depth
        depth = get_active_queue_depth(self.db_path)
        if depth < cap:
            # D2 warn-only heartbeat check: log if no workers are alive,
            # but never block dispatch on worker health.
            alive = get_worker_health(self.db_path, stale_after_seconds=60)
            if not alive:
                logger.warning(
                    "Dispatching tournament round with no healthy workers "
                    "(queue depth %d / %d) — queue will fill to cap",
                    depth, cap,
                )
            round_id = self._dispatcher.enqueue_round(epoch=epoch_i)
            if round_id is not None:
                logger.info(
                    "Dispatched tournament round %d (queue depth %d/%d)",
                    round_id, depth, cap,
                )
        # else: queue is at capacity, skip this epoch silently. Worker
        # will drain the queue and dispatch resumes next epoch.
    except Exception:
        logger.exception("Dispatcher failed at epoch %d", epoch_i)
```

`get_active_queue_depth(db_path)` is a new helper added to `keisei/training/tournament_queue.py`:

```python
def get_active_queue_depth(db_path: str) -> int:
    """Count pending + playing pairings across all rounds."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n FROM tournament_pairing_queue "
            "WHERE status IN ('pending', 'playing')"
        ).fetchone()
        return int(row["n"])
    finally:
        conn.close()
```

- [ ] **Step 4: Remove `learner_entry_id` cross-thread setter**

In sidecar mode, the worker reads `learner_entry_id` from a DB column instead of a cross-thread setter. Add a `training_state` singleton table:

```sql
CREATE TABLE IF NOT EXISTS training_state (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    learner_entry_id  INTEGER,
    current_epoch     INTEGER NOT NULL DEFAULT 0,
    updated_at        TEXT NOT NULL
);
INSERT OR IGNORE INTO training_state (id, updated_at) VALUES (1, '2026-04-12T00:00:00Z');
```

Training writes to this table on rotation; workers read from it. Add `OpponentStore.set_learner_entry_id(entry_id)` and `OpponentStore.get_learner_entry_id()` helpers. Existing `get_current_epoch()` should point at this table rather than wherever it reads from today.

Gate the existing `self._tournament.learner_entry_id = new_entry.id` line so it only runs in in-process mode; in sidecar mode, the rotation path calls `self.store.set_learner_entry_id(new_entry.id)`.

- [ ] **Step 5: Run integration tests**

Run: `uv run pytest tests/test_katago_loop_integration.py tests/test_tournament_dispatcher.py -v`

Expected: all pass. Sidecar-mode test confirms no tournament thread, dispatcher is wired up.

---

## Task 10: End-to-end integration test

**Files:**
- Create: `tests/test_tournament_sidecar_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/test_tournament_sidecar_integration.py
"""End-to-end test: real subprocess worker claims and plays a pairing
written by an in-process dispatcher. Verifies the full round-trip
without mocks at the DB layer.

Marked @pytest.mark.integration so it can be excluded from fast CI runs.
"""
import subprocess
import sys
import time
import pytest
from keisei.training.tournament_queue import get_round_status

@pytest.mark.integration
def test_subprocess_worker_claims_and_completes_pairing(tmp_path):
    db = str(tmp_path / "test.db")
    # ... set up DB with 2 entries and 1 enqueued pairing ...
    
    proc = subprocess.Popen([
        sys.executable, "-m", "keisei.training.tournament_runner",
        "--db-path", db,
        "--worker-id", "itest-worker",
        "--device", "cpu",  # integration test on CPU for CI portability
    ])
    try:
        # Poll queue for up to 60s for the pairing to reach 'done' state
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            status = get_round_status(db, round_id=1)
            if status.get("done", 0) >= 1:
                break
            time.sleep(0.5)
        assert status.get("done", 0) == 1, (
            f"Pairing did not complete within 60s: {status}"
        )
    finally:
        proc.terminate()
        proc.wait(timeout=10)
```

- [ ] **Step 2: Run the integration test in isolation**

Run: `uv run pytest tests/test_tournament_sidecar_integration.py -v -m integration`

Expected: PASS. If it fails, investigate DB contention, signal handling, or model loading in subprocess context.

- [ ] **Step 3: Add `integration` marker to pyproject.toml** (if not already present)

```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests that spawn subprocesses or require real services",
]
```

---

## Task 11a: `run.sh` — config-gated tournament worker launch

Per **D3**, the tournament worker is launched by `run.sh` as a supervised child process, matching the existing showcase sidecar pattern. The showcase code in `run.sh` is the reference — mirror it for tournament.

**Files:**
- Modify: `run.sh` (parse config, add CLI flag, build command, launch, supervise, cleanup)

- [ ] **Step 1: Extend the TOML preflight block to parse `tournament_mode`, `tournament_enabled`, `tournament_device`**

Current block (`run.sh:~155`) uses `uv run python -c ...` with a heredoc to extract `DB_PATH`, `CKPT_DIR`, `USE_DDP` from the config. Extend the Python script to also emit:

```python
league = raw.get('league', {})
tournament_mode = league.get('tournament_mode', 'in_process')
tournament_enabled = league.get('tournament_enabled', False)
tournament_device = league.get('tournament_device', 'cuda:1')
print(f'TOURNAMENT_MODE={tournament_mode!r}')
print(f'TOURNAMENT_ENABLED={str(tournament_enabled).lower()}')
print(f'TOURNAMENT_DEVICE={tournament_device!r}')
```

- [ ] **Step 2: Add `--no-tournament` CLI flag**

Parallel to `--no-showcase`:

```bash
NO_TOURNAMENT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        ...
        --no-tournament) NO_TOURNAMENT=true; shift ;;
        ...
    esac
done
```

Update the help text at the top of `run.sh` to document it.

- [ ] **Step 3: Build `TOURNAMENT_CMD` array**

After the `SHOWCASE_CMD` definition (~line 240):

```bash
# ---- Build tournament worker sidecar command ----
TOURNAMENT_CMD=(uv run python -m keisei.training.tournament_runner
                --db-path "$DB_PATH"
                --worker-id worker-0
                --device "$TOURNAMENT_DEVICE")
```

- [ ] **Step 4: Gate the launch**

Introduce a helper variable for the gate condition so it can be reused in both foreground and background paths:

```bash
LAUNCH_TOURNAMENT=false
if [[ "$TOURNAMENT_ENABLED" == "true" \
      && "$TOURNAMENT_MODE" == "sidecar" \
      && "$NO_TOURNAMENT" != true ]]; then
    LAUNCH_TOURNAMENT=true
fi
```

- [ ] **Step 5: Launch in background mode (if `--background`)**

In the existing `if [[ "$BACKGROUND" == true ]]; then ... fi` block, after the showcase launch:

```bash
if [[ "$LAUNCH_TOURNAMENT" == true ]]; then
    echo "Starting tournament worker sidecar ..."
    nohup "${TOURNAMENT_CMD[@]}" > "$TOURNAMENT_LOG" 2>&1 &
    TOURNAMENT_PID=$!
    echo "$TOURNAMENT_PID" >> "$PIDFILE"
    echo "  Tournament PID: $TOURNAMENT_PID (log: $TOURNAMENT_LOG)"
fi
```

- [ ] **Step 6: Launch in foreground mode and wire into the monitor**

After the showcase foreground launch:

```bash
TOURNAMENT_PID=""
if [[ "$LAUNCH_TOURNAMENT" == true ]]; then
    echo "Starting tournament worker sidecar"
    echo "  Log:    $TOURNAMENT_LOG"
    "${TOURNAMENT_CMD[@]}" > "$TOURNAMENT_LOG" 2>&1 &
    TOURNAMENT_PID=$!
    echo "  PID:    $TOURNAMENT_PID"
fi
```

Add to the status banner after the showcase entry:

```bash
if [[ -n "$TOURNAMENT_PID" ]]; then
echo "  Tournament: PID $TOURNAMENT_PID ($TOURNAMENT_DEVICE)"
echo "  Tournament log: $TOURNAMENT_LOG"
fi
```

Add to the monitor loop's restart-on-death block (after showcase restart):

```bash
if [[ -n "$TOURNAMENT_PID" ]] && ! kill -0 "$TOURNAMENT_PID" 2>/dev/null; then
    echo "Tournament worker died, restarting..."
    "${TOURNAMENT_CMD[@]}" >> "$TOURNAMENT_LOG" 2>&1 &
    TOURNAMENT_PID=$!
    echo "  Tournament restarted (PID $TOURNAMENT_PID)"
fi
```

- [ ] **Step 7: Add to `cleanup()` trap**

```bash
cleanup() {
    echo ""
    echo "Shutting down..."
    [[ -n "${TRAIN_PID:-}" ]] && kill "$TRAIN_PID" 2>/dev/null && echo "  Trainer stopped (PID $TRAIN_PID)"
    [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null && echo "  Server stopped (PID $SERVER_PID)"
    [[ -n "${SHOWCASE_PID:-}" ]] && kill "$SHOWCASE_PID" 2>/dev/null && echo "  Showcase stopped (PID $SHOWCASE_PID)"
    [[ -n "${TOURNAMENT_PID:-}" ]] && kill "$TOURNAMENT_PID" 2>/dev/null && echo "  Tournament stopped (PID $TOURNAMENT_PID)"
    exit 0
}
```

And to the training-exited shutdown path inside the monitor loop.

- [ ] **Step 8: Add to `--stop` teardown**

In the stop-mode block at the top of `run.sh`:

```bash
pkill -f 'keisei.training.tournament_runner' 2>/dev/null && echo "  Killed tournament worker" || true
```

- [ ] **Step 9: Add `TOURNAMENT_LOG` alongside the other log variables**

```bash
TOURNAMENT_LOG="logs/${CONFIG_STEM}_tournament_${TIMESTAMP}.log"
```

- [ ] **Step 10: Test the launch manually**

1. Set `tournament_mode = "sidecar"` in a test config.
2. Run `./run.sh <config>` in foreground.
3. Verify tournament worker log file is created and populated.
4. Verify `sqlite3 <db> "SELECT * FROM tournament_worker_heartbeat"` shows the worker within 15s.
5. Ctrl+C and verify all four processes (training + dashboard + showcase + tournament) are cleanly stopped.
6. Re-run with `--no-tournament`; verify tournament worker is NOT launched, but the other three still are.
7. Re-run with default `tournament_mode = "in_process"`; verify tournament worker is NOT launched (backward compat).

---

## Task 11: Documentation and runbook

**Files:**
- Create: `docs/superpowers/runbooks/tournament-sidecar.md`

- [ ] **Step 1: Write the runbook**

Sections:
1. **What is the tournament sidecar?** — quick architectural overview, reference to this plan.
2. **Configuration** — `tournament_mode = "sidecar"` in the TOML + optional `--no-tournament` override. No separate launch command; `run.sh` handles it.
3. **Scaling to N workers** — deferred (initial deployment is N=1). Document the knobs that will unlock it: a worker_id CLI arg and a per-worker device assignment. Future `run.sh` could launch multiple workers from a config list.
4. **Diagnosing problems** —
   - Queue depth growing? Check `get_round_status` and `get_worker_health`. If no alive workers, check tournament log file.
   - Worker crashed? `run.sh`'s monitor loop auto-restarts it. Check tournament log for the restart entry and the original crash traceback above it.
   - Pairings stuck in `'playing'` state? `run.sh`'s restart pulls the worker up and its startup sweep resets them.
   - DynamicTrainer update race? Check `dynamic_update_worker` column in `league_entries`; stale values are released by the worker on next successful claim.
5. **Pre-merge test gate** — run `uv run pytest -m integration` before merging any change to `tournament_runner.py` or `tournament_queue.py` (D5).
6. **Verifying decoupling** — compare training's per-epoch timing logs with `tournament_enabled=false` vs `tournament_mode="sidecar"`. After Phase 1, both should be identical within 5% noise. Any additional rollout time in sidecar mode indicates residual coupling worth investigating.

---

## Task 12: Rollout and validation

- [ ] **Step 1: Merge Phase 2 to main with `tournament_mode` default = `in_process`**

No behavior change for existing configs. All tests pass. The sidecar infrastructure ships behind a feature flag — `run.sh` won't launch the worker for any config that doesn't opt in.

- [ ] **Step 2: Flip `keisei-500k-league.toml` to sidecar mode**

```toml
[league]
tournament_mode = "sidecar"
```

No other config changes. `tournament_enabled` is already true, `tournament_device` is already `cuda:1`.

- [ ] **Step 3: Launch with the new config**

```bash
./run.sh keisei-500k-league.toml
```

`run.sh` will now launch four processes: training, dashboard, showcase, tournament worker. Expected output includes a "Tournament: PID ..." line in the banner.

- [ ] **Step 4: Validate**

- Heartbeat visible: `sqlite3 data/keisei-500k-league.db "SELECT * FROM tournament_worker_heartbeat;"` updates every ~10s
- Pairings flowing: `sqlite3 data/keisei-500k-league.db "SELECT status, COUNT(*) FROM tournament_pairing_queue GROUP BY status;"` shows pending → playing → done progression
- Queue depth bounded: never exceeds `dispatcher_max_queue_depth` (default 400)
- No stale `'playing'` pairings after a worker restart test (`pkill -f tournament_runner`; wait for monitor to restart; verify all prior `'playing'` rows transitioned to `'pending'` or `'done'`)
- Training epoch timing unchanged from Phase 1 baseline (~35s rollout at pool=20 with K=4)
- Epoch-to-epoch timing variance ≤ 5% (down from ~10-15% under in-process tournament thread)
- Process isolation spot-check: `kill -9` the tournament worker PID; training continues unaffected, monitor restarts the worker within 30s

- [ ] **Step 5: After a stable run (24+ hours), file follow-up issues**

- Evaluate whether Elo-pair atomicity (Task 5 caveat) is observable in practice — if so, follow up with single-transaction Elo updates that lock both entry rows
- Evaluate whether a second worker on cuda:2 (if available) provides any benefit — if yes, file Phase 2.1 for N>1 deployment plumbing in `run.sh`
- Decide whether to schedule Phase 3 (Postgres migration) based on any SQLite contention observed in the sidecar run

---

## Out of scope (file as follow-ups)

- **N>1 worker deployment.** The infrastructure supports it, but initial deployment is N=1 until we've validated the single-worker case in production.
- **Postgres migration.** Deferred to Phase 3.
- **Dispatcher as its own process.** The dispatcher lives inside the training process in this plan. Moving it to its own process is a possible future evolution if the training process has any measurable overhead from the dispatcher calls, but there's no reason to expect that given how small the dispatcher is (a few hundred bytes of state, one DB write per round).
- **Real-time pairing priority adjustment.** The priority column exists in the queue table but is initially set to a fixed 0.0. A future improvement could let the dispatcher rewrite priorities as the round progresses (e.g., to demote pairings for entries that have already played enough).
- **Pairing queue visualization in the webui.** The dashboard could show queue depth, worker health, and recent match throughput. Valuable operationally but not required for correctness.

---

## Review checklist (before starting implementation)

- [x] Design decisions D1-D5 are recorded in this document.
- [ ] File map has no unexpected dependencies (no circular imports between dispatcher and worker, no shared in-memory state).
- [ ] Task ordering is correct: schema → queue ops → dispatcher → worker → training integration → run.sh → tests → rollout.
- [ ] Every task has a failing test before implementation (TDD gate).
- [ ] Rollout plan (Task 12) defines concrete success criteria.
- [ ] Out-of-scope items are filed as follow-ups, not silently dropped.
