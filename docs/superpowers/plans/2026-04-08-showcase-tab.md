# Showcase Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third "Showcase" tab to the webui where two league models play each other at human-watchable speed on CPU, with commentary (top-3 moves, win probability graph, eval bar).

**Architecture:** Sidecar Python process runs games on CPU via `SpectatorEnv`, writes moves to SQLite. Existing FastAPI server polls those tables and pushes updates via WebSocket. Svelte frontend renders a new tab reusing `Board.svelte`, `PieceTray.svelte`, etc.

**Tech Stack:** Python 3.13, FastAPI, SQLite WAL, Svelte 4, PyTorch (CPU-only), shogi-gym (`SpectatorEnv`), uPlot (win probability graph)

**Spec:** `docs/superpowers/specs/2026-04-08-showcase-tab-design.md`

---

## File Map

### New files

| File | Purpose |
|------|---------|
| `keisei/showcase/__init__.py` | Package init |
| `keisei/showcase/runner.py` | Sidecar process: game loop, model loading, DB writes, heartbeat |
| `keisei/showcase/inference.py` | Model loading + forward pass (CPU-only, both contracts) |
| `keisei/showcase/db_ops.py` | Showcase-specific DB read/write functions |
| `tests/test_showcase_db.py` | DB schema, read/write, concurrency tests |
| `tests/test_showcase_runner.py` | Runner logic: queue claiming, game loop, crash recovery |
| `tests/test_showcase_inference.py` | Model loading, CPU enforcement, dual contracts |
| `tests/test_server_showcase.py` | WebSocket: showcase polling, client commands, validation |
| `webui/src/stores/showcase.js` | Svelte store for showcase state |
| `webui/src/stores/showcase.test.js` | Store unit tests |
| `webui/src/lib/ShowcaseView.svelte` | Top-level tab layout |
| `webui/src/lib/MatchControls.svelte` | Entry dropdowns, speed selector, start/cancel |
| `webui/src/lib/CommentaryPanel.svelte` | Top-3 candidates, move annotation |
| `webui/src/lib/WinProbGraph.svelte` | Win probability over time (uPlot) |
| `webui/src/lib/MatchQueue.svelte` | Queue display + sidecar health |

### Modified files

| File | Changes |
|------|---------|
| `keisei/db.py` | Add showcase tables to `init_db()`, increment `SCHEMA_VERSION` to 3, add migration |
| `keisei/server/app.py` | Add `_receive_commands()` coroutine, `_poll_showcase()`, extend init message, refactor `_poll_and_push` |
| `webui/src/lib/ws.js` | Handle `showcase_update`, `showcase_status`, `showcase_error`; add `sendShowcaseCommand()` |
| `webui/src/lib/TabBar.svelte` | Add showcase tab to `tabs` array |
| `webui/src/App.svelte` | Add `{:else if $activeTab === 'showcase'}` branch with `ShowcaseView` |

---

## Task 1: Database Schema — Showcase Tables

**Files:**
- Modify: `keisei/db.py:9` (SCHEMA_VERSION), `keisei/db.py:70-72` (_MIGRATIONS), `keisei/db.py:75-299` (init_db)
- Create: `tests/test_showcase_db.py`

- [ ] **Step 1: Write failing test for schema version 3**

```python
# tests/test_showcase_db.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_showcase_db.py -v`
Expected: FAIL — schema_version is 2, tables don't exist

- [ ] **Step 3: Add showcase tables to init_db() and bump schema version**

In `keisei/db.py`, make these changes:

At line 9, change:
```python
SCHEMA_VERSION = 3
```

After line 64 (after `_migrate_v1_to_v2`), add:
```python
def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """v2 -> v3: Add showcase tables (created by init_db IF NOT EXISTS)."""
    # New tables only — no ALTER TABLE needed. The CREATE TABLE IF NOT EXISTS
    # statements in init_db() handle table creation for both fresh and
    # migrated databases.
    pass
```

Update `_MIGRATIONS` at line 70:
```python
_MIGRATIONS: dict[int, callable] = {
    2: _migrate_v1_to_v2,
    3: _migrate_v2_to_v3,
}
```

Inside `init_db()`, after the existing `CREATE TABLE IF NOT EXISTS` statements (before the schema version check at ~line 300), add the following SQL to the `executescript()` call:

```sql
CREATE TABLE IF NOT EXISTS showcase_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id_1  TEXT NOT NULL,
    entry_id_2  TEXT NOT NULL,
    speed       TEXT NOT NULL DEFAULT 'normal',
    status      TEXT NOT NULL DEFAULT 'pending',
    requested_at TEXT NOT NULL,
    started_at  TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_showcase_queue_status ON showcase_queue(status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_showcase_queue_one_running
    ON showcase_queue(status) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS showcase_games (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    queue_id        INTEGER NOT NULL REFERENCES showcase_queue(id),
    entry_id_black  TEXT NOT NULL,
    entry_id_white  TEXT NOT NULL,
    elo_black       REAL,
    elo_white       REAL,
    name_black      TEXT,
    name_white      TEXT,
    status          TEXT NOT NULL DEFAULT 'in_progress',
    abandon_reason  TEXT,
    started_at      TEXT NOT NULL,
    completed_at    TEXT,
    total_ply       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_showcase_games_status ON showcase_games(status);

CREATE TABLE IF NOT EXISTS showcase_moves (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id         INTEGER NOT NULL REFERENCES showcase_games(id),
    ply             INTEGER NOT NULL,
    action_index    INTEGER NOT NULL,
    usi_notation    TEXT NOT NULL,
    board_json      TEXT NOT NULL,
    hands_json      TEXT NOT NULL,
    current_player  TEXT NOT NULL,
    in_check        INTEGER NOT NULL DEFAULT 0,
    value_estimate  REAL,
    top_candidates  TEXT,
    move_time_ms    INTEGER,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_showcase_moves_game_ply ON showcase_moves(game_id, ply);

CREATE TABLE IF NOT EXISTS showcase_heartbeat (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    last_heartbeat  TEXT NOT NULL,
    runner_pid      INTEGER
);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_showcase_db.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `uv run pytest tests/test_db.py -v`
Expected: All existing DB tests still PASS (schema migration handles v2→v3)

- [ ] **Step 6: Commit**

```bash
git add keisei/db.py tests/test_showcase_db.py
git commit -m "feat(db): add showcase tables and bump schema to v3"
```

---

## Task 2: Showcase DB Operations

**Files:**
- Create: `keisei/showcase/__init__.py`
- Create: `keisei/showcase/db_ops.py`
- Modify: `tests/test_showcase_db.py`

- [ ] **Step 1: Create the showcase package**

```python
# keisei/showcase/__init__.py
"""Showcase sidecar: model-vs-model games at watchable speed."""
```

- [ ] **Step 2: Write failing tests for DB operations**

Append to `tests/test_showcase_db.py`:

```python
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
        rows = read_queue(db)
        assert rows[0]["status"] == "cancelled"

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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_showcase_db.py::TestQueueOperations -v`
Expected: FAIL — `ImportError: cannot import name 'queue_match' from 'keisei.showcase.db_ops'`

- [ ] **Step 4: Implement db_ops.py**

```python
# keisei/showcase/db_ops.py
"""Database operations for the showcase sidecar and server."""
from __future__ import annotations

import json
import os
import random
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from keisei.db import _connect

MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.1


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _retry_write(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()) -> sqlite3.Cursor:
    """Execute a write with retry on SQLITE_BUSY."""
    for attempt in range(MAX_RETRIES):
        try:
            cursor = conn.execute(sql, params)
            conn.commit()
            return cursor
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 0.05)
                time.sleep(delay)
            else:
                raise
    raise RuntimeError("unreachable")  # pragma: no cover


# ── Queue operations ──────────────────────────────────────


def queue_match(db_path: str, entry_id_1: str, entry_id_2: str, speed: str) -> int:
    """Insert a pending match request. Returns the queue ID."""
    conn = _connect(db_path)
    try:
        cursor = _retry_write(conn,
            "INSERT INTO showcase_queue (entry_id_1, entry_id_2, speed, status, requested_at) VALUES (?, ?, ?, 'pending', ?)",
            (entry_id_1, entry_id_2, speed, _now_iso()),
        )
        return cursor.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def claim_next_match(db_path: str) -> dict[str, Any] | None:
    """Atomically claim the next pending match. Returns None if none available."""
    conn = _connect(db_path)
    try:
        now = _now_iso()
        row = conn.execute(
            """UPDATE showcase_queue
               SET status = 'running', started_at = ?
               WHERE id = (
                   SELECT id FROM showcase_queue
                   WHERE status = 'pending'
                   ORDER BY id ASC
                   LIMIT 1
               )
               RETURNING id, entry_id_1, entry_id_2, speed, status, requested_at, started_at""",
            (now,),
        ).fetchone()
        conn.commit()
        return dict(row) if row else None
    finally:
        conn.close()


def read_queue(db_path: str) -> list[dict[str, Any]]:
    """Read all non-completed queue entries (pending + running)."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM showcase_queue WHERE status IN ('pending', 'running') ORDER BY id",
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def cancel_match(db_path: str, queue_id: int) -> None:
    """Cancel a pending match request."""
    conn = _connect(db_path)
    try:
        _retry_write(conn,
            "UPDATE showcase_queue SET status = 'cancelled', completed_at = ? WHERE id = ? AND status = 'pending'",
            (_now_iso(), queue_id),
        )
    finally:
        conn.close()


def update_queue_speed(db_path: str, queue_id: int, speed: str) -> None:
    """Update the speed of a running match."""
    conn = _connect(db_path)
    try:
        _retry_write(conn,
            "UPDATE showcase_queue SET speed = ? WHERE id = ?",
            (speed, queue_id),
        )
    finally:
        conn.close()


# ── Game operations ───────────────────────────────────────


def create_showcase_game(
    db_path: str,
    *,
    queue_id: int,
    entry_id_black: str,
    entry_id_white: str,
    elo_black: float,
    elo_white: float,
    name_black: str,
    name_white: str,
) -> int:
    """Create a new showcase game. Returns the game ID."""
    conn = _connect(db_path)
    try:
        cursor = _retry_write(conn,
            """INSERT INTO showcase_games
               (queue_id, entry_id_black, entry_id_white, elo_black, elo_white,
                name_black, name_white, status, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'in_progress', ?)""",
            (queue_id, entry_id_black, entry_id_white, elo_black, elo_white,
             name_black, name_white, _now_iso()),
        )
        return cursor.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def read_active_showcase_game(db_path: str) -> dict[str, Any] | None:
    """Read the current in-progress showcase game, if any."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM showcase_games WHERE status = 'in_progress' ORDER BY id DESC LIMIT 1",
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def write_showcase_move(
    db_path: str,
    *,
    game_id: int,
    ply: int,
    action_index: int,
    usi_notation: str,
    board_json: str,
    hands_json: str,
    current_player: str,
    in_check: bool,
    value_estimate: float,
    top_candidates: str,
    move_time_ms: int,
) -> None:
    """Write a single showcase move to the database."""
    conn = _connect(db_path)
    try:
        _retry_write(conn,
            """INSERT INTO showcase_moves
               (game_id, ply, action_index, usi_notation, board_json, hands_json,
                current_player, in_check, value_estimate, top_candidates, move_time_ms, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (game_id, ply, action_index, usi_notation, board_json, hands_json,
             current_player, int(in_check), value_estimate, top_candidates,
             move_time_ms, _now_iso()),
        )
        # Also update total_ply on the game row
        _retry_write(conn,
            "UPDATE showcase_games SET total_ply = ? WHERE id = ?",
            (ply, game_id),
        )
    finally:
        conn.close()


def read_showcase_moves_since(
    db_path: str, game_id: int, since_ply: int,
) -> list[dict[str, Any]]:
    """Read moves for a game with ply > since_ply."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM showcase_moves WHERE game_id = ? AND ply > ? ORDER BY ply",
            (game_id, since_ply),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def read_all_showcase_moves(db_path: str, game_id: int) -> list[dict[str, Any]]:
    """Read all moves for a game (for cold-start init)."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM showcase_moves WHERE game_id = ? ORDER BY ply",
            (game_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_game_completed(db_path: str, game_id: int, result: str, total_ply: int) -> None:
    """Mark a showcase game as completed with a result."""
    conn = _connect(db_path)
    try:
        _retry_write(conn,
            "UPDATE showcase_games SET status = ?, completed_at = ?, total_ply = ? WHERE id = ?",
            (result, _now_iso(), total_ply, game_id),
        )
    finally:
        conn.close()


def mark_game_abandoned(db_path: str, game_id: int, reason: str) -> None:
    """Mark a showcase game as abandoned."""
    conn = _connect(db_path)
    try:
        _retry_write(conn,
            "UPDATE showcase_games SET status = 'abandoned', abandon_reason = ?, completed_at = ? WHERE id = ?",
            (reason, _now_iso(), game_id),
        )
    finally:
        conn.close()


# ── Heartbeat ─────────────────────────────────────────────


def write_heartbeat(db_path: str, pid: int) -> None:
    """Upsert the singleton heartbeat row."""
    conn = _connect(db_path)
    try:
        _retry_write(conn,
            """INSERT INTO showcase_heartbeat (id, last_heartbeat, runner_pid)
               VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE SET last_heartbeat = excluded.last_heartbeat, runner_pid = excluded.runner_pid""",
            (_now_iso(), pid),
        )
    finally:
        conn.close()


def read_heartbeat(db_path: str) -> dict[str, Any] | None:
    """Read the heartbeat row."""
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM showcase_heartbeat WHERE id = 1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ── Crash recovery ────────────────────────────────────────


def cleanup_orphaned_games(db_path: str) -> int:
    """Mark orphaned in-progress games as abandoned and reset running queue entries.
    Called on sidecar startup. Returns number of games cleaned up."""
    conn = _connect(db_path)
    try:
        now = _now_iso()
        # Abandon in-progress games
        cursor = conn.execute(
            "UPDATE showcase_games SET status = 'abandoned', abandon_reason = 'crash_recovery', completed_at = ? WHERE status = 'in_progress'",
            (now,),
        )
        count = cursor.rowcount
        # Reset running queue entries to cancelled
        conn.execute(
            "UPDATE showcase_queue SET status = 'cancelled', completed_at = ? WHERE status = 'running'",
            (now,),
        )
        conn.commit()
        return count
    finally:
        conn.close()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_showcase_db.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/showcase/__init__.py keisei/showcase/db_ops.py tests/test_showcase_db.py
git commit -m "feat(showcase): add DB operations for queue, games, moves, heartbeat"
```

---

## Task 3: Showcase Inference Module

**Files:**
- Create: `keisei/showcase/inference.py`
- Create: `tests/test_showcase_inference.py`

- [ ] **Step 1: Write failing tests for inference**

```python
# tests/test_showcase_inference.py
"""Tests for showcase CPU-only model inference."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from keisei.showcase.inference import (
    enforce_cpu_only,
    load_model_for_showcase,
    run_inference,
    ModelCache,
)
from keisei.training.model_registry import build_model


class TestCPUEnforcement:
    def test_enforce_cpu_only_sets_env_var(self) -> None:
        # enforce_cpu_only should set CUDA_VISIBLE_DEVICES=""
        enforce_cpu_only(cpu_threads=2)
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


class TestModelLoading:
    @pytest.fixture
    def resnet_checkpoint(self, tmp_path: Path) -> tuple[Path, str, dict[str, Any]]:
        arch = "resnet"
        params = {"hidden_size": 32, "num_layers": 2}
        model = build_model(arch, params)
        ckpt_path = tmp_path / "weights.pt"
        torch.save(model.state_dict(), ckpt_path)
        return ckpt_path, arch, params

    def test_load_model_returns_eval_mode(self, resnet_checkpoint: tuple[Path, str, dict]) -> None:
        path, arch, params = resnet_checkpoint
        model = load_model_for_showcase(path, arch, params)
        assert not model.training

    def test_load_model_all_params_on_cpu(self, resnet_checkpoint: tuple[Path, str, dict]) -> None:
        path, arch, params = resnet_checkpoint
        model = load_model_for_showcase(path, arch, params)
        for name, param in model.named_parameters():
            assert param.device == torch.device("cpu"), f"{name} on {param.device}"

    def test_load_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_for_showcase(tmp_path / "nonexistent.pt", "resnet", {"hidden_size": 32, "num_layers": 2})


class TestInference:
    @pytest.fixture
    def resnet_model(self, tmp_path: Path) -> nn.Module:
        params = {"hidden_size": 32, "num_layers": 2}
        model = build_model("resnet", params)
        model.eval()
        return model

    def test_run_inference_returns_policy_and_value(self, resnet_model: nn.Module) -> None:
        obs = np.random.randn(50, 9, 9).astype(np.float32)
        policy_logits, win_prob = run_inference(resnet_model, obs, "resnet")
        assert isinstance(policy_logits, np.ndarray)
        assert isinstance(win_prob, float)
        assert 0.0 <= win_prob <= 1.0

    def test_run_inference_se_resnet(self, tmp_path: Path) -> None:
        params = {"channels": 32, "num_blocks": 2}
        model = build_model("se_resnet", params)
        model.eval()
        obs_channels = params.get("obs_channels", 50)
        obs = np.random.randn(obs_channels, 9, 9).astype(np.float32)
        policy_logits, win_prob = run_inference(model, obs, "se_resnet")
        assert isinstance(policy_logits, np.ndarray)
        assert isinstance(win_prob, float)
        assert 0.0 <= win_prob <= 1.0


class TestModelCache:
    @pytest.fixture
    def cache(self) -> ModelCache:
        return ModelCache(max_size=2)

    @pytest.fixture
    def resnet_checkpoint(self, tmp_path: Path) -> tuple[Path, str, dict[str, Any]]:
        arch = "resnet"
        params = {"hidden_size": 32, "num_layers": 2}
        model = build_model(arch, params)
        ckpt_path = tmp_path / "weights.pt"
        torch.save(model.state_dict(), ckpt_path)
        return ckpt_path, arch, params

    def test_cache_hit(self, cache: ModelCache, resnet_checkpoint: tuple[Path, str, dict]) -> None:
        path, arch, params = resnet_checkpoint
        m1 = cache.get_or_load("entry-1", str(path), arch, params)
        m2 = cache.get_or_load("entry-1", str(path), arch, params)
        assert m1 is m2

    def test_cache_evicts_oldest(self, cache: ModelCache, tmp_path: Path) -> None:
        params = {"hidden_size": 32, "num_layers": 2}
        paths = []
        for i in range(3):
            model = build_model("resnet", params)
            p = tmp_path / f"weights_{i}.pt"
            torch.save(model.state_dict(), p)
            paths.append(p)

        cache.get_or_load("e1", str(paths[0]), "resnet", params)
        cache.get_or_load("e2", str(paths[1]), "resnet", params)
        cache.get_or_load("e3", str(paths[2]), "resnet", params)  # should evict e1
        assert cache.size == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_showcase_inference.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement inference.py**

```python
# keisei/showcase/inference.py
"""CPU-only model loading and inference for showcase games."""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from keisei.training.model_registry import build_model, _get_spec

logger = logging.getLogger(__name__)


def enforce_cpu_only(cpu_threads: int = 2) -> None:
    """Set environment and torch config for CPU-only inference.

    MUST be called before any other torch operations in the process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(1)


def load_model_for_showcase(
    checkpoint_path: Path | str,
    architecture: str,
    model_params: dict[str, Any],
) -> nn.Module:
    """Load a model checkpoint for CPU-only showcase inference.

    Raises FileNotFoundError if checkpoint doesn't exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    model = build_model(architecture, model_params)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Disable AMP for KataGo-style models
    if hasattr(model, "configure_amp"):
        model.configure_amp(enabled=False)

    # Runtime assertion: no parameters on GPU
    for name, param in model.named_parameters():
        assert param.device == torch.device("cpu"), (
            f"Parameter {name} on {param.device} — GPU leak in showcase"
        )

    return model


def run_inference(
    model: nn.Module,
    obs: np.ndarray,
    architecture: str,
) -> tuple[np.ndarray, float]:
    """Run a single forward pass. Returns (policy_logits, win_probability).

    win_probability is normalized to [0, 1] regardless of model contract:
    - scalar (resnet/mlp/transformer): tanh output mapped from [-1,1] to [0,1]
    - multi_head (se_resnet): softmax(value_logits)[0] = P(win)
    """
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()

    with torch.inference_mode():
        output = model(obs_tensor)

    spec = _get_spec(architecture)

    if spec.contract == "multi_head":
        # KataGoOutput: .policy_logits (batch, 9, 9, 139), .value_logits (batch, 3)
        policy_logits = output.policy_logits.squeeze(0).reshape(-1)
        win_prob = torch.softmax(output.value_logits.squeeze(0), dim=0)[0].item()
    else:
        # Tuple: (policy_logits, value)
        policy_logits_t, value_tensor = output
        policy_logits = policy_logits_t.squeeze(0)
        win_prob = (value_tensor.squeeze(0).item() + 1.0) / 2.0

    return policy_logits.numpy(), float(win_prob)


class ModelCache:
    """LRU cache for loaded models, keyed on (entry_id, checkpoint_path)."""

    def __init__(self, max_size: int = 2) -> None:
        self._cache: OrderedDict[tuple[str, str], nn.Module] = OrderedDict()
        self._max_size = max_size

    @property
    def size(self) -> int:
        return len(self._cache)

    def get_or_load(
        self,
        entry_id: str,
        checkpoint_path: str,
        architecture: str,
        model_params: dict[str, Any],
    ) -> nn.Module:
        key = (entry_id, checkpoint_path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        model = load_model_for_showcase(checkpoint_path, architecture, model_params)
        self._cache[key] = model
        while len(self._cache) > self._max_size:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("Evicted model %s from cache", evicted_key)
        return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_showcase_inference.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/showcase/inference.py tests/test_showcase_inference.py
git commit -m "feat(showcase): CPU-only inference module with model cache"
```

---

## Task 4: Showcase Runner (Sidecar Process)

**Files:**
- Create: `keisei/showcase/runner.py`
- Create: `tests/test_showcase_runner.py`

- [ ] **Step 1: Write failing tests for the runner**

```python
# tests/test_showcase_runner.py
"""Tests for the showcase sidecar runner."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.db import init_db
from keisei.showcase.db_ops import (
    queue_match,
    claim_next_match,
    read_active_showcase_game,
    read_showcase_moves_since,
    read_heartbeat,
    cleanup_orphaned_games,
    create_showcase_game,
)
from keisei.showcase.runner import ShowcaseRunner


@pytest.fixture
def db(tmp_path: Path) -> str:
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


@pytest.fixture
def mock_spectator_env() -> MagicMock:
    """Mock SpectatorEnv that plays a 3-move game."""
    env = MagicMock()
    env.action_space_size.return_value = 11259

    move_count = 0
    def mock_step(action: int) -> dict:
        nonlocal move_count
        move_count += 1
        return {
            "board": [None] * 81,
            "hands": {"black": {}, "white": {}},
            "current_player": "white" if move_count % 2 == 1 else "black",
            "ply": move_count,
            "is_over": move_count >= 3,
            "result": "checkmate" if move_count >= 3 else "in_progress",
            "in_check": False,
            "sfen": "startpos",
            "move_history": [{"action": i, "notation": f"move{i}"} for i in range(1, move_count + 1)],
        }

    def mock_reset() -> dict:
        nonlocal move_count
        move_count = 0
        return {
            "board": [None] * 81,
            "hands": {"black": {}, "white": {}},
            "current_player": "black",
            "ply": 0,
            "is_over": False,
            "result": "in_progress",
            "in_check": False,
            "sfen": "startpos",
            "move_history": [],
        }

    env.step.side_effect = mock_step
    env.reset.side_effect = mock_reset
    env.legal_actions.return_value = [42, 100, 200]
    env.get_observation.return_value = np.zeros((50, 9, 9), dtype=np.float32)
    env.is_over.side_effect = lambda: move_count >= 3
    return env


@pytest.fixture
def mock_model() -> MagicMock:
    """Mock model returning tuple (policy_logits, value)."""
    model = MagicMock()
    model.training = False
    model.eval.return_value = model

    def mock_forward(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]
        policy = torch.randn(batch, 11259)
        value = torch.tensor([[0.3]])
        return policy, value

    model.side_effect = mock_forward
    model.named_parameters.return_value = []
    return model


class TestShowcaseRunner:
    def test_cleanup_on_startup(self, db: str) -> None:
        """Runner cleans up orphaned games from previous crashes."""
        # Create an orphaned in-progress game
        qid = queue_match(db, "e1", "e2", "normal")
        claim_next_match(db)
        create_showcase_game(db, queue_id=qid, entry_id_black="e1", entry_id_white="e2",
            elo_black=1500.0, elo_white=1480.0, name_black="A", name_white="B")

        runner = ShowcaseRunner(db_path=db)
        runner._startup_cleanup()

        assert read_active_showcase_game(db) is None

    def test_run_single_game(self, db: str, mock_spectator_env: MagicMock, mock_model: MagicMock) -> None:
        """Runner plays a complete game and writes moves to DB."""
        qid = queue_match(db, "e1", "e2", "normal")

        runner = ShowcaseRunner(db_path=db)

        with patch.object(runner, "_create_env", return_value=mock_spectator_env), \
             patch.object(runner, "_load_models", return_value=(mock_model, mock_model, "resnet", "resnet")):
            match = claim_next_match(db)
            runner._run_game(match)

        game = read_active_showcase_game(db)
        assert game is None  # game completed

        # Moves were written — mock plays 3 moves before checkmate
        # read all moves for the game (game_id=1)
        moves = read_showcase_moves_since(db, 1, since_ply=0)
        assert len(moves) == 3

    def test_heartbeat_written(self, db: str) -> None:
        runner = ShowcaseRunner(db_path=db)
        runner._write_heartbeat()
        hb = read_heartbeat(db)
        assert hb is not None

    def test_speed_from_queue(self, db: str) -> None:
        """Runner reads speed from the queue row."""
        runner = ShowcaseRunner(db_path=db)
        assert runner._get_delay("slow") == 4.0
        assert runner._get_delay("normal") == 2.0
        assert runner._get_delay("fast") == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_showcase_runner.py -v`
Expected: FAIL — `ImportError: cannot import name 'ShowcaseRunner'`

- [ ] **Step 3: Implement runner.py**

```python
# keisei/showcase/runner.py
"""Showcase sidecar: plays model-vs-model games at watchable speed.

Usage:
    python -m keisei.showcase.runner --db-path path/to/db.sqlite \\
        [--cpu-threads 2] [--auto-showcase-interval 1800] [--no-auto-showcase]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# GPU firewall — MUST be before any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

from keisei.showcase.db_ops import (
    claim_next_match,
    cleanup_orphaned_games,
    create_showcase_game,
    mark_game_abandoned,
    mark_game_completed,
    queue_match,
    read_queue,
    update_queue_speed,
    write_heartbeat,
    write_showcase_move,
)
from keisei.showcase.inference import (
    ModelCache,
    enforce_cpu_only,
    load_model_for_showcase,
    run_inference,
)

MAX_PLY = 512
SPEED_DELAYS = {"slow": 4.0, "normal": 2.0, "fast": 0.5}
HEARTBEAT_INTERVAL = 10.0
POLL_INTERVAL = 5.0


class ShowcaseRunner:
    """Main sidecar runner for showcase games."""

    def __init__(
        self,
        db_path: str,
        cpu_threads: int = 2,
        auto_showcase_interval: int = 1800,
        auto_showcase_enabled: bool = True,
    ) -> None:
        self.db_path = db_path
        self.cpu_threads = cpu_threads
        self.auto_showcase_interval = auto_showcase_interval
        self.auto_showcase_enabled = auto_showcase_enabled
        self.model_cache = ModelCache(max_size=2)
        self._stop_event = threading.Event()
        self._speed_event = threading.Event()
        self._last_auto_showcase = 0.0

    def _startup_cleanup(self) -> None:
        """Clean up orphaned games from previous crashes."""
        count = cleanup_orphaned_games(self.db_path)
        if count > 0:
            logger.info("Cleaned up %d orphaned showcase game(s)", count)

    def _write_heartbeat(self) -> None:
        write_heartbeat(self.db_path, pid=os.getpid())

    def _get_delay(self, speed: str) -> float:
        return SPEED_DELAYS.get(speed, 2.0)

    def _create_env(self) -> Any:
        """Create a SpectatorEnv instance."""
        from shogi_gym import SpectatorEnv
        return SpectatorEnv(max_ply=MAX_PLY)

    def _load_models(
        self, match: dict[str, Any],
    ) -> tuple[Any, Any, str, str]:
        """Load the two models for a match from the league DB.

        Returns (model_black, model_white, arch_black, arch_white).
        """
        from keisei.db import _connect

        conn = _connect(self.db_path)
        try:
            e1 = conn.execute(
                "SELECT * FROM league_entries WHERE id = ?", (match["entry_id_1"],)
            ).fetchone()
            e2 = conn.execute(
                "SELECT * FROM league_entries WHERE id = ?", (match["entry_id_2"],)
            ).fetchone()
        finally:
            conn.close()

        if e1 is None or e2 is None:
            raise ValueError(f"League entry not found: {match['entry_id_1']} or {match['entry_id_2']}")

        def _load_entry(entry: Any) -> tuple[Any, str]:
            arch = entry["architecture"]
            params_raw = entry["model_params"]
            params = json.loads(params_raw) if isinstance(params_raw, str) else params_raw
            model = self.model_cache.get_or_load(
                str(entry["id"]), entry["checkpoint_path"], arch, params,
            )
            return model, arch

        model_black, arch_black = _load_entry(e1)
        model_white, arch_white = _load_entry(e2)
        return model_black, model_white, arch_black, arch_white

    def _run_game(self, match: dict[str, Any]) -> None:
        """Play a single showcase game to completion."""
        try:
            model_black, model_white, arch_black, arch_white = self._load_models(match)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Cannot start showcase game: %s", e)
            return

        # Look up entry info for display
        from keisei.db import _connect
        conn = _connect(self.db_path)
        try:
            e1 = conn.execute("SELECT * FROM league_entries WHERE id = ?", (match["entry_id_1"],)).fetchone()
            e2 = conn.execute("SELECT * FROM league_entries WHERE id = ?", (match["entry_id_2"],)).fetchone()
        finally:
            conn.close()

        game_id = create_showcase_game(
            self.db_path,
            queue_id=match["id"],
            entry_id_black=match["entry_id_1"],
            entry_id_white=match["entry_id_2"],
            elo_black=e1["elo_rating"] if e1 else 0.0,
            elo_white=e2["elo_rating"] if e2 else 0.0,
            name_black=e1["display_name"] if e1 else "Unknown",
            name_white=e2["display_name"] if e2 else "Unknown",
        )

        env = self._create_env()
        state = env.reset()
        logger.info("Showcase game %d started: %s vs %s", game_id,
                     e1["display_name"] if e1 else "?", e2["display_name"] if e2 else "?")

        ply = 0
        while not self._stop_event.is_set() and not env.is_over() and ply < MAX_PLY:
            # Select the right model for current player
            is_black_turn = state["current_player"] == "black"
            model = model_black if is_black_turn else model_white
            arch = arch_black if is_black_turn else arch_white

            # Get observation and run inference
            obs = env.get_observation()
            start_ms = time.monotonic()
            policy_logits, win_prob = run_inference(model, obs, arch)
            inference_ms = int((time.monotonic() - start_ms) * 1000)

            # Mask illegal moves and select action
            legal = env.legal_actions()
            mask = np.full(policy_logits.shape, -1e9)
            mask[legal] = 0.0
            masked_logits = policy_logits + mask

            # Softmax for probabilities
            probs = np.exp(masked_logits - masked_logits.max())
            probs = probs / probs.sum()

            # Select top-3 candidates
            top_indices = np.argsort(probs)[::-1][:3]
            top_candidates = []
            for idx in top_indices:
                if probs[idx] > 0.001:
                    top_candidates.append({"action": int(idx), "probability": round(float(probs[idx]), 4)})

            # Pick the best move (greedy)
            action = int(np.argmax(masked_logits))

            # Step the environment
            state = env.step(action)
            ply = state["ply"]

            # Extract USI notation from move history
            usi_notation = state["move_history"][-1]["notation"] if state["move_history"] else f"action_{action}"

            # Add USI to top candidates
            for tc in top_candidates:
                # We only have USI for the move that was played — for candidates,
                # we'd need to decode actions. For now, just use action index.
                tc["usi"] = usi_notation if tc["action"] == action else f"a{tc['action']}"

            # Write move to DB
            write_showcase_move(
                self.db_path,
                game_id=game_id,
                ply=ply,
                action_index=action,
                usi_notation=usi_notation,
                board_json=json.dumps(state["board"]),
                hands_json=json.dumps(state["hands"]),
                current_player=state["current_player"],
                in_check=state.get("in_check", False),
                value_estimate=win_prob,
                top_candidates=json.dumps(top_candidates),
                move_time_ms=inference_ms,
            )

            # Read current speed from queue (allows mid-game speed changes)
            from keisei.showcase.db_ops import _connect
            conn = _connect(self.db_path)
            try:
                row = conn.execute(
                    "SELECT speed FROM showcase_queue WHERE id = ?", (match["id"],)
                ).fetchone()
                speed = row["speed"] if row else match.get("speed", "normal")
            finally:
                conn.close()

            # Wait for pacing delay (interruptible)
            delay = self._get_delay(speed)
            self._speed_event.wait(timeout=delay)
            self._speed_event.clear()

        # Determine result
        if self._stop_event.is_set():
            mark_game_abandoned(self.db_path, game_id, "shutdown")
            logger.info("Showcase game %d abandoned (shutdown)", game_id)
        elif ply >= MAX_PLY:
            mark_game_completed(self.db_path, game_id, "draw", total_ply=ply)
            logger.info("Showcase game %d ended: draw (max ply)", game_id)
        else:
            result = state.get("result", "in_progress")
            # Map SpectatorEnv result to our status
            if result == "checkmate":
                # The player who just moved won
                winner = "white" if state["current_player"] == "black" else "black"
                status = f"{winner}_win"
            elif result in ("repetition", "perpetual_check", "impasse", "max_moves"):
                status = "draw"
            else:
                status = "draw"
            mark_game_completed(self.db_path, game_id, status, total_ply=ply)
            logger.info("Showcase game %d ended: %s (%d ply)", game_id, status, ply)

        # Mark queue entry as completed
        conn = _connect(self.db_path)
        try:
            from keisei.showcase.db_ops import _now_iso
            conn.execute(
                "UPDATE showcase_queue SET status = 'completed', completed_at = ? WHERE id = ?",
                (_now_iso(), match["id"]),
            )
            conn.commit()
        finally:
            conn.close()

    def _maybe_auto_showcase(self) -> None:
        """Queue an auto-showcase match if conditions are met."""
        if not self.auto_showcase_enabled:
            return
        if time.monotonic() - self._last_auto_showcase < self.auto_showcase_interval:
            return

        # Check no pending or running matches
        queue = read_queue(self.db_path)
        if queue:
            return

        # Get top-2 league entries by Elo
        from keisei.db import _connect
        conn = _connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT id FROM league_entries WHERE status = 'active' ORDER BY elo_rating DESC LIMIT 2"
            ).fetchall()
        finally:
            conn.close()

        if len(rows) < 2:
            return

        queue_match(self.db_path, str(rows[0]["id"]), str(rows[1]["id"]), "normal")
        self._last_auto_showcase = time.monotonic()
        logger.info("Auto-showcase: queued top-2 league entries")

    def run(self) -> None:
        """Main loop. Blocks until stop_event is set."""
        enforce_cpu_only(self.cpu_threads)
        self._startup_cleanup()
        self._write_heartbeat()

        logger.info("Showcase runner started (pid=%d, db=%s)", os.getpid(), self.db_path)

        heartbeat_time = time.monotonic()

        while not self._stop_event.is_set():
            # Heartbeat
            now = time.monotonic()
            if now - heartbeat_time >= HEARTBEAT_INTERVAL:
                self._write_heartbeat()
                heartbeat_time = now

            # Try to claim a match
            match = claim_next_match(self.db_path)
            if match is not None:
                self._run_game(match)
                continue

            # No match — maybe auto-showcase
            self._maybe_auto_showcase()

            # Wait before polling again
            self._stop_event.wait(timeout=POLL_INTERVAL)

        logger.info("Showcase runner stopped")

    def stop(self) -> None:
        """Signal the runner to stop."""
        self._stop_event.set()
        self._speed_event.set()  # wake any sleeping move


def main() -> None:
    parser = argparse.ArgumentParser(description="Showcase sidecar runner")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--cpu-threads", type=int, default=2, help="PyTorch CPU threads")
    parser.add_argument("--auto-showcase-interval", type=int, default=1800,
                        help="Seconds between auto-showcase matches")
    parser.add_argument("--no-auto-showcase", action="store_true",
                        help="Disable auto-showcase")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    runner = ShowcaseRunner(
        db_path=args.db_path,
        cpu_threads=args.cpu_threads,
        auto_showcase_interval=args.auto_showcase_interval,
        auto_showcase_enabled=not args.no_auto_showcase,
    )

    def handle_signal(signum: int, frame: Any) -> None:
        logger.info("Received signal %d, stopping...", signum)
        runner.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    runner.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_showcase_runner.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/showcase/runner.py tests/test_showcase_runner.py
git commit -m "feat(showcase): sidecar runner with game loop, heartbeat, auto-showcase"
```

---

## Task 5: WebSocket Server — Showcase Polling and Client Commands

**Files:**
- Modify: `keisei/server/app.py`
- Create: `tests/test_server_showcase.py`

- [ ] **Step 1: Write failing tests for server showcase support**

```python
# tests/test_server_showcase.py
"""Tests for showcase WebSocket extensions."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from keisei.db import init_db, write_training_state
from keisei.server.app import create_app, TEST_ALLOWED_HOSTS
from keisei.showcase.db_ops import (
    queue_match,
    claim_next_match,
    create_showcase_game,
    write_showcase_move,
    write_heartbeat,
    read_queue,
)


@pytest.fixture
def server_db(tmp_path: Path) -> str:
    path = str(tmp_path / "server_test.db")
    init_db(path)
    # Write minimal training state so init message works
    write_training_state(path, {
        "status": "idle", "current_epoch": 0, "current_step": 0,
    })
    return path


class TestShowcaseInit:
    def test_init_message_contains_showcase(self, server_db: str) -> None:
        """Init message includes showcase data."""
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01):
            client = TestClient(app)
            with client.websocket_connect("/ws") as ws:
                msg = ws.receive_json()
                assert msg["type"] == "init"
                assert "showcase" in msg
                assert "queue" in msg["showcase"]
                assert "sidecar_alive" in msg["showcase"]

    def test_init_showcase_with_active_game(self, server_db: str) -> None:
        """Init message includes active game and all moves."""
        qid = queue_match(server_db, "e1", "e2", "normal")
        claim_next_match(server_db)
        game_id = create_showcase_game(server_db, queue_id=qid,
            entry_id_black="e1", entry_id_white="e2",
            elo_black=1500, elo_white=1480, name_black="A", name_white="B")
        write_showcase_move(server_db, game_id=game_id, ply=1, action_index=42,
            usi_notation="7g7f", board_json="[]", hands_json="{}",
            current_player="white", in_check=False, value_estimate=0.5,
            top_candidates="[]", move_time_ms=10)

        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01):
            client = TestClient(app)
            with client.websocket_connect("/ws") as ws:
                msg = ws.receive_json()
                assert msg["showcase"]["game"] is not None
                assert len(msg["showcase"]["moves"]) == 1


class TestShowcaseCommands:
    def test_request_match_creates_queue_entry(self, server_db: str) -> None:
        """Client can request a showcase match."""
        # Create league entries so validation passes
        from keisei.db import _connect
        conn = _connect(server_db)
        conn.execute("INSERT INTO league_entries (id, display_name, architecture, model_params, checkpoint_path, elo_rating, status) VALUES (1, 'A', 'resnet', '{}', '/tmp/a.pt', 1500, 'active')")
        conn.execute("INSERT INTO league_entries (id, display_name, architecture, model_params, checkpoint_path, elo_rating, status) VALUES (2, 'B', 'resnet', '{}', '/tmp/b.pt', 1480, 'active')")
        conn.commit()
        conn.close()

        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01):
            client = TestClient(app)
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()  # init
                ws.send_json({
                    "type": "request_showcase_match",
                    "entry_id_1": "1",
                    "entry_id_2": "2",
                    "speed": "normal",
                })
                # Give server time to process
                import time; time.sleep(0.1)
                queue = read_queue(server_db)
                assert len(queue) == 1
                assert queue[0]["entry_id_1"] == "1"

    def test_request_match_validates_self_match(self, server_db: str) -> None:
        """Server rejects match where both entries are the same."""
        from keisei.db import _connect
        conn = _connect(server_db)
        conn.execute("INSERT INTO league_entries (id, display_name, architecture, model_params, checkpoint_path, elo_rating, status) VALUES (1, 'A', 'resnet', '{}', '/tmp/a.pt', 1500, 'active')")
        conn.commit()
        conn.close()

        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01):
            client = TestClient(app)
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()  # init
                ws.send_json({
                    "type": "request_showcase_match",
                    "entry_id_1": "1",
                    "entry_id_2": "1",
                    "speed": "normal",
                })
                import time; time.sleep(0.1)
                # Should receive error
                msg = ws.receive_json()
                assert msg["type"] == "showcase_error"

    def test_invalid_speed_rejected(self, server_db: str) -> None:
        app = create_app(server_db, allowed_hosts=TEST_ALLOWED_HOSTS)
        with patch("keisei.server.app.POLL_INTERVAL_S", 0.01):
            client = TestClient(app)
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()  # init
                ws.send_json({
                    "type": "request_showcase_match",
                    "entry_id_1": "1",
                    "entry_id_2": "2",
                    "speed": "turbo",
                })
                import time; time.sleep(0.1)
                msg = ws.receive_json()
                assert msg["type"] == "showcase_error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server_showcase.py -v`
Expected: FAIL — no showcase support in server yet

- [ ] **Step 3: Add `_receive_commands()` and `_poll_showcase()` to app.py**

In `keisei/server/app.py`, add these imports at the top:

```python
from keisei.showcase.db_ops import (
    queue_match as showcase_queue_match,
    read_queue as showcase_read_queue,
    read_active_showcase_game,
    read_all_showcase_moves,
    read_showcase_moves_since,
    read_heartbeat as showcase_read_heartbeat,
    cancel_match as showcase_cancel_match,
    update_queue_speed as showcase_update_speed,
)
```

Add a constant:

```python
SHOWCASE_POLL_INTERVAL_S = 0.5
VALID_SPEEDS = frozenset({"slow", "normal", "fast"})
MAX_SHOWCASE_QUEUE_DEPTH = 5
```

Add the `_receive_commands()` coroutine:

```python
async def _receive_commands(ws: WebSocket, db_path: str) -> None:
    """Handle client-to-server WebSocket messages."""
    import time as _time
    last_match_request_time = 0.0  # rate limit: 1 match request per 10s per connection

    while True:
        try:
            data = await ws.receive_json()
        except Exception:
            return  # connection closed

        msg_type = data.get("type")

        try:
            if msg_type == "request_showcase_match":
                now = _time.monotonic()
                if now - last_match_request_time < 10.0:
                    await asyncio.wait_for(ws.send_json({
                        "type": "showcase_error",
                        "message": "Rate limited — wait 10 seconds between match requests",
                        "request_type": "request_showcase_match",
                    }), timeout=WS_SEND_TIMEOUT_S)
                    continue
                last_match_request_time = now
                await _handle_match_request(ws, db_path, data)
            elif msg_type == "change_showcase_speed":
                await _handle_speed_change(ws, db_path, data)
            elif msg_type == "cancel_showcase_match":
                await _handle_cancel(ws, db_path, data)
            else:
                await asyncio.wait_for(ws.send_json({
                    "type": "showcase_error",
                    "message": f"Unknown message type: {msg_type}",
                    "request_type": msg_type,
                }), timeout=WS_SEND_TIMEOUT_S)
        except Exception as e:
            logger.warning("Error handling client command %s: %s", msg_type, e)


async def _handle_match_request(ws: WebSocket, db_path: str, data: dict) -> None:
    entry_id_1 = str(data.get("entry_id_1", ""))
    entry_id_2 = str(data.get("entry_id_2", ""))
    speed = data.get("speed", "normal")

    # Validate speed
    if speed not in VALID_SPEEDS:
        await asyncio.wait_for(ws.send_json({
            "type": "showcase_error",
            "message": f"Invalid speed: {speed}. Must be one of: {', '.join(sorted(VALID_SPEEDS))}",
            "request_type": "request_showcase_match",
        }), timeout=WS_SEND_TIMEOUT_S)
        return

    # Validate self-match
    if entry_id_1 == entry_id_2:
        await asyncio.wait_for(ws.send_json({
            "type": "showcase_error",
            "message": "Cannot match an entry against itself",
            "request_type": "request_showcase_match",
        }), timeout=WS_SEND_TIMEOUT_S)
        return

    # Validate entries exist in active league pool
    def _validate_entries() -> str | None:
        conn = _connect(db_path)
        try:
            for eid in (entry_id_1, entry_id_2):
                row = conn.execute(
                    "SELECT id FROM league_entries WHERE id = ? AND status = 'active'", (eid,)
                ).fetchone()
                if not row:
                    return f"Entry {eid} not found in active league pool"
        finally:
            conn.close()
        return None

    error = await asyncio.to_thread(_validate_entries)
    if error:
        await asyncio.wait_for(ws.send_json({
            "type": "showcase_error",
            "message": error,
            "request_type": "request_showcase_match",
        }), timeout=WS_SEND_TIMEOUT_S)
        return

    # Check queue depth
    queue = await asyncio.to_thread(showcase_read_queue, db_path)
    pending_count = sum(1 for q in queue if q["status"] == "pending")
    if pending_count >= MAX_SHOWCASE_QUEUE_DEPTH:
        await asyncio.wait_for(ws.send_json({
            "type": "showcase_error",
            "message": f"Queue full ({pending_count} pending matches, max {MAX_SHOWCASE_QUEUE_DEPTH})",
            "request_type": "request_showcase_match",
        }), timeout=WS_SEND_TIMEOUT_S)
        return

    # Queue the match
    await asyncio.to_thread(showcase_queue_match, db_path, entry_id_1, entry_id_2, speed)


async def _handle_speed_change(ws: WebSocket, db_path: str, data: dict) -> None:
    speed = data.get("speed", "")
    if speed not in VALID_SPEEDS:
        await asyncio.wait_for(ws.send_json({
            "type": "showcase_error",
            "message": f"Invalid speed: {speed}",
            "request_type": "change_showcase_speed",
        }), timeout=WS_SEND_TIMEOUT_S)
        return

    # Find the running queue entry and update its speed
    queue = await asyncio.to_thread(showcase_read_queue, db_path)
    running = [q for q in queue if q["status"] == "running"]
    if not running:
        await asyncio.wait_for(ws.send_json({
            "type": "showcase_error",
            "message": "No active match to change speed on",
            "request_type": "change_showcase_speed",
        }), timeout=WS_SEND_TIMEOUT_S)
        return

    await asyncio.to_thread(showcase_update_speed, db_path, running[0]["id"], speed)


async def _handle_cancel(ws: WebSocket, db_path: str, data: dict) -> None:
    queue_id = data.get("queue_id")
    if queue_id is None:
        await asyncio.wait_for(ws.send_json({
            "type": "showcase_error",
            "message": "Missing queue_id",
            "request_type": "cancel_showcase_match",
        }), timeout=WS_SEND_TIMEOUT_S)
        return

    await asyncio.to_thread(showcase_cancel_match, db_path, int(queue_id))
```

Add the `_poll_showcase()` coroutine:

```python
async def _poll_showcase(ws: WebSocket, db_path: str) -> None:
    """Poll showcase tables and push updates."""
    last_move_ply = 0
    last_game_id: int | None = None

    while True:
        await asyncio.sleep(SHOWCASE_POLL_INTERVAL_S)

        game = await asyncio.to_thread(read_active_showcase_game, db_path)
        queue = await asyncio.to_thread(showcase_read_queue, db_path)
        hb = await asyncio.to_thread(showcase_read_heartbeat, db_path)

        sidecar_alive = False
        if hb:
            from datetime import datetime, timezone
            try:
                last_hb = datetime.fromisoformat(hb["last_heartbeat"].replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - last_hb).total_seconds()
                sidecar_alive = age < 30
            except (ValueError, TypeError):
                pass

        # Send queue/status update
        try:
            await asyncio.wait_for(ws.send_json({
                "type": "showcase_status",
                "queue": queue,
                "active_game_id": game["id"] if game else None,
                "sidecar_alive": sidecar_alive,
                "queue_depth": sum(1 for q in queue if q["status"] == "pending"),
            }), timeout=WS_SEND_TIMEOUT_S)
        except Exception:
            return

        # Send new moves if game is active
        if game:
            current_game_id = game["id"]
            if current_game_id != last_game_id:
                last_move_ply = 0
                last_game_id = current_game_id

            moves = await asyncio.to_thread(
                read_showcase_moves_since, db_path, current_game_id, last_move_ply,
            )
            if moves:
                last_move_ply = max(m["ply"] for m in moves)
                try:
                    await asyncio.wait_for(ws.send_json({
                        "type": "showcase_update",
                        "game": dict(game),
                        "new_moves": moves,
                    }), timeout=WS_SEND_TIMEOUT_S)
                except Exception:
                    return
```

Modify the init message in `_poll_and_push()` to include showcase data:

After the existing init fields (around line 252), add:

```python
# Showcase init data (cold-start support)
showcase_game = await asyncio.to_thread(read_active_showcase_game, db_path)
showcase_moves = []
if showcase_game:
    showcase_moves = await asyncio.to_thread(read_all_showcase_moves, db_path, showcase_game["id"])
showcase_queue = await asyncio.to_thread(showcase_read_queue, db_path)
showcase_hb = await asyncio.to_thread(showcase_read_heartbeat, db_path)
showcase_alive = False
if showcase_hb:
    from datetime import datetime, timezone
    try:
        last_hb = datetime.fromisoformat(showcase_hb["last_heartbeat"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - last_hb).total_seconds()
        showcase_alive = age < 30
    except (ValueError, TypeError):
        pass
```

Then add to the init message dict:

```python
"showcase": {
    "game": dict(showcase_game) if showcase_game else None,
    "moves": showcase_moves,
    "queue": showcase_queue,
    "sidecar_alive": showcase_alive,
},
```

Modify the WebSocket handler's `TaskGroup` to add both new coroutines:

```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(_poll_and_push(websocket, db_path))
    tg.create_task(_keepalive(websocket))
    tg.create_task(_receive_commands(websocket, db_path))
    tg.create_task(_poll_showcase(websocket, db_path))
```

Also add the `_connect` import if not already present:

```python
from keisei.db import _connect
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server_showcase.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run existing server tests for regressions**

Run: `uv run pytest tests/test_server_websocket.py -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/server/app.py tests/test_server_showcase.py
git commit -m "feat(server): showcase WebSocket polling, client commands, and init support"
```

---

## Task 6: Frontend — Showcase Store and WebSocket Handling

**Files:**
- Create: `webui/src/stores/showcase.js`
- Create: `webui/src/stores/showcase.test.js`
- Modify: `webui/src/lib/ws.js`

- [ ] **Step 1: Create showcase store**

```javascript
// webui/src/stores/showcase.js
import { writable, derived } from 'svelte/store'

/** Active showcase game metadata */
export const showcaseGame = writable(null)

/** All moves in the current showcase game */
export const showcaseMoves = writable([])

/** Queue of pending/running showcase matches */
export const showcaseQueue = writable([])

/** Current speed setting */
export const showcaseSpeed = writable('normal')

/** Whether the sidecar process is alive */
export const sidecarAlive = writable(false)

/** Latest board state from most recent move */
export const showcaseBoard = derived(showcaseMoves, moves => {
  if (moves.length === 0) return null
  return moves[moves.length - 1]
})

/** Win probability history for the graph */
export const winProbHistory = derived(showcaseMoves, moves => {
  return moves.map(m => ({ ply: m.ply, value: m.value_estimate }))
})

/** Number of pending matches in queue */
export const queueDepth = derived(showcaseQueue, q =>
  q.filter(e => e.status === 'pending').length
)
```

- [ ] **Step 2: Write store tests**

```javascript
// webui/src/stores/showcase.test.js
import { describe, it, expect } from 'vitest'
import { get } from 'svelte/store'
import {
  showcaseGame, showcaseMoves, showcaseQueue,
  showcaseBoard, winProbHistory, queueDepth, sidecarAlive,
} from './showcase.js'

describe('showcase stores', () => {
  it('showcaseBoard returns null when no moves', () => {
    showcaseMoves.set([])
    expect(get(showcaseBoard)).toBeNull()
  })

  it('showcaseBoard returns latest move', () => {
    showcaseMoves.set([
      { ply: 1, board_json: 'b1', value_estimate: 0.5 },
      { ply: 2, board_json: 'b2', value_estimate: 0.6 },
    ])
    expect(get(showcaseBoard).ply).toBe(2)
  })

  it('winProbHistory maps moves to ply/value pairs', () => {
    showcaseMoves.set([
      { ply: 1, value_estimate: 0.5 },
      { ply: 2, value_estimate: 0.7 },
    ])
    const history = get(winProbHistory)
    expect(history).toEqual([
      { ply: 1, value: 0.5 },
      { ply: 2, value: 0.7 },
    ])
  })

  it('queueDepth counts pending entries', () => {
    showcaseQueue.set([
      { id: 1, status: 'running' },
      { id: 2, status: 'pending' },
      { id: 3, status: 'pending' },
    ])
    expect(get(queueDepth)).toBe(2)
  })
})
```

- [ ] **Step 3: Run store tests**

Run: `cd webui && npx vitest run src/stores/showcase.test.js`
Expected: All tests PASS

- [ ] **Step 4: Update ws.js to handle showcase messages and add send capability**

In `webui/src/lib/ws.js`, add the showcase store imports at the top:

```javascript
import {
  showcaseGame, showcaseMoves, showcaseQueue, sidecarAlive,
} from '../stores/showcase.js'
```

Add a `sendShowcaseCommand` export function after the `connect()` function:

```javascript
export function sendShowcaseCommand(message) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message))
  }
}
```

In the `handleMessage()` switch statement, extend the `'init'` case (after the existing init handling, before `break`):

```javascript
      // Showcase init (cold-start support)
      if (msg.showcase) {
        showcaseGame.set(msg.showcase.game || null)
        showcaseMoves.set(msg.showcase.moves || [])
        showcaseQueue.set(msg.showcase.queue || [])
        sidecarAlive.set(msg.showcase.sidecar_alive || false)
      }
```

Add new cases before the `'ping'` case:

```javascript
    case 'showcase_update':
      showcaseGame.set(msg.game || null)
      showcaseMoves.update(existing => {
        const newMoves = msg.new_moves || []
        // Append only new moves (deduplicate by ply)
        const maxPly = existing.length > 0 ? existing[existing.length - 1].ply : 0
        const fresh = newMoves.filter(m => m.ply > maxPly)
        return [...existing, ...fresh]
      })
      break

    case 'showcase_status':
      showcaseQueue.set(msg.queue || [])
      sidecarAlive.set(msg.sidecar_alive || false)
      break

    case 'showcase_error':
      console.warn('[ws] showcase error:', msg.message)
      // TODO: surface to UI via a toast/notification store (Task 7 handles display)
      break
```

- [ ] **Step 5: Run frontend tests**

Run: `cd webui && npx vitest run`
Expected: All tests PASS (including new showcase store tests)

- [ ] **Step 6: Commit**

```bash
git add webui/src/stores/showcase.js webui/src/stores/showcase.test.js webui/src/lib/ws.js
git commit -m "feat(webui): showcase store and WebSocket message handling"
```

---

## Task 7: Frontend — Showcase Tab Components

**Files:**
- Create: `webui/src/lib/ShowcaseView.svelte`
- Create: `webui/src/lib/MatchControls.svelte`
- Create: `webui/src/lib/CommentaryPanel.svelte`
- Create: `webui/src/lib/WinProbGraph.svelte`
- Create: `webui/src/lib/MatchQueue.svelte`
- Modify: `webui/src/lib/TabBar.svelte`
- Modify: `webui/src/App.svelte`

- [ ] **Step 1: Add showcase tab to TabBar.svelte**

In `webui/src/lib/TabBar.svelte`, update the tabs array (line 5-8):

```javascript
const tabs = [
  { id: 'training', label: 'Training' },
  { id: 'league', label: 'League' },
  { id: 'showcase', label: 'Showcase' },
]
```

- [ ] **Step 2: Create MatchControls.svelte**

```svelte
<!-- webui/src/lib/MatchControls.svelte -->
<script>
  import { leagueEntries } from '../stores/league.js'
  import { showcaseQueue, queueDepth, sidecarAlive, showcaseSpeed } from '../stores/showcase.js'
  import { sendShowcaseCommand } from './ws.js'

  let selectedEntry1 = ''
  let selectedEntry2 = ''
  let speed = 'normal'

  $: activeEntries = ($leagueEntries || []).filter(e => e.status === 'active')
  $: canStart = selectedEntry1 && selectedEntry2 && selectedEntry1 !== selectedEntry2
    && $sidecarAlive && $queueDepth < 5

  function requestMatch() {
    if (!canStart) return
    sendShowcaseCommand({
      type: 'request_showcase_match',
      entry_id_1: selectedEntry1,
      entry_id_2: selectedEntry2,
      speed,
    })
  }

  function changeSpeed(newSpeed) {
    speed = newSpeed
    showcaseSpeed.set(newSpeed)
    sendShowcaseCommand({ type: 'change_showcase_speed', speed: newSpeed })
  }
</script>

<div class="match-controls">
  <div class="entry-selectors">
    <select bind:value={selectedEntry1} aria-label="Black player">
      <option value="">Select black...</option>
      {#each activeEntries as entry}
        <option value={String(entry.id)}>
          {entry.display_name} ({entry.elo_rating?.toFixed(0) ?? '?'})
        </option>
      {/each}
    </select>

    <span class="vs">vs</span>

    <select bind:value={selectedEntry2} aria-label="White player">
      <option value="">Select white...</option>
      {#each activeEntries as entry}
        <option value={String(entry.id)}>
          {entry.display_name} ({entry.elo_rating?.toFixed(0) ?? '?'})
        </option>
      {/each}
    </select>
  </div>

  <div class="speed-controls">
    <span class="label">Speed:</span>
    {#each ['slow', 'normal', 'fast'] as s}
      <button
        class:active={speed === s}
        on:click={() => changeSpeed(s)}
      >{s}</button>
    {/each}
  </div>

  <button class="start-btn" on:click={requestMatch} disabled={!canStart}>
    Start Match
  </button>

  {#if !$sidecarAlive}
    <div class="warning">Showcase engine is offline</div>
  {:else if $queueDepth >= 5}
    <div class="warning">Queue full ({$queueDepth} pending)</div>
  {/if}
</div>

<style>
  .match-controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-bottom: 1px solid var(--border);
  }
  .entry-selectors { display: flex; align-items: center; gap: 8px; }
  .vs { font-weight: 600; color: var(--text-muted); font-size: 13px; }
  select {
    padding: 6px 8px;
    min-height: 36px;
    font-size: 13px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  .speed-controls { display: flex; align-items: center; gap: 4px; }
  .speed-controls .label { font-size: 12px; color: var(--text-secondary); }
  .speed-controls button {
    padding: 4px 10px;
    min-height: 32px;
    font-size: 12px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    text-transform: capitalize;
  }
  .speed-controls button.active {
    border-color: var(--tab-active-border);
    color: var(--tab-active-border);
    background: var(--tab-active-bg);
  }
  .start-btn {
    padding: 6px 16px;
    min-height: 36px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid var(--accent-teal);
    border-radius: 4px;
    background: var(--accent-teal);
    color: #fff;
    cursor: pointer;
  }
  .start-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .warning { font-size: 12px; color: var(--accent-gold); }
</style>
```

- [ ] **Step 3: Create CommentaryPanel.svelte**

```svelte
<!-- webui/src/lib/CommentaryPanel.svelte -->
<script>
  import { showcaseBoard } from '../stores/showcase.js'

  $: move = $showcaseBoard
  $: topCandidates = (() => {
    if (!move?.top_candidates) return []
    try {
      return typeof move.top_candidates === 'string'
        ? JSON.parse(move.top_candidates) : move.top_candidates
    } catch { return [] }
  })()
  $: winProb = move?.value_estimate ?? 0.5
</script>

<div class="commentary">
  <h3 class="section-label">Commentary</h3>

  <div class="eval-display">
    <span class="label">Win probability</span>
    <div class="eval-bar-container">
      <div class="eval-bar-fill" style="width: {winProb * 100}%"></div>
    </div>
    <span class="eval-value">{(winProb * 100).toFixed(1)}%</span>
  </div>

  {#if move}
    <div class="last-move">
      <span class="label">Last move</span>
      <span class="value">{move.usi_notation}
        {#if topCandidates.length > 0}
          ({(topCandidates.find(c => c.usi === move.usi_notation)?.probability * 100 || 0).toFixed(1)}%)
        {/if}
      </span>
    </div>

    <div class="candidates">
      <span class="label">Top candidates</span>
      {#each topCandidates as c, i}
        <div class="candidate" class:chosen={c.usi === move.usi_notation}>
          <span class="rank">{i + 1}.</span>
          <span class="move-name">{c.usi}</span>
          <span class="prob">{(c.probability * 100).toFixed(1)}%</span>
        </div>
      {/each}
    </div>

    {#if move.move_time_ms != null}
      <div class="inference-time">
        <span class="label">Inference</span>
        <span class="value">{move.move_time_ms}ms</span>
      </div>
    {/if}
  {:else}
    <div class="no-data">Waiting for moves...</div>
  {/if}
</div>

<style>
  .commentary { display: flex; flex-direction: column; gap: 10px; padding: 8px; }
  .section-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin: 0; }
  .label { font-size: 12px; color: var(--text-secondary); display: block; margin-bottom: 2px; }
  .value { font-size: 13px; color: var(--text-primary); }
  .eval-display { display: flex; flex-direction: column; gap: 4px; }
  .eval-bar-container { height: 8px; background: var(--bg-secondary, #333); border-radius: 4px; overflow: hidden; }
  .eval-bar-fill { height: 100%; background: var(--accent-teal); transition: width 0.3s ease; }
  .eval-value { font-size: 14px; font-weight: 600; color: var(--text-primary); }
  .candidates { display: flex; flex-direction: column; gap: 2px; }
  .candidate { display: flex; gap: 6px; font-size: 13px; padding: 2px 4px; border-radius: 3px; }
  .candidate.chosen { background: var(--tab-active-bg); }
  .rank { color: var(--text-muted); width: 1.5em; }
  .move-name { color: var(--text-primary); flex: 1; }
  .prob { color: var(--text-secondary); }
  .inference-time { font-size: 12px; color: var(--text-muted); }
  .no-data { font-size: 13px; color: var(--text-muted); font-style: italic; }
</style>
```

- [ ] **Step 4: Create WinProbGraph.svelte**

```svelte
<!-- webui/src/lib/WinProbGraph.svelte -->
<script>
  import { onMount, onDestroy } from 'svelte'
  import { winProbHistory } from '../stores/showcase.js'
  import uPlot from 'uplot'

  let chartEl
  let chart = null

  function buildData(history) {
    if (!history || history.length === 0) return [[], []]
    return [
      history.map(h => h.ply),
      history.map(h => h.value),
    ]
  }

  const opts = {
    width: 300,
    height: 120,
    cursor: { show: false },
    legend: { show: false },
    scales: { y: { range: [0, 1] } },
    axes: [
      { show: true, size: 20, font: '10px sans-serif', stroke: 'var(--text-muted)' },
      { show: true, size: 30, font: '10px sans-serif', stroke: 'var(--text-muted)',
        values: (u, vals) => vals.map(v => (v * 100).toFixed(0) + '%') },
    ],
    series: [
      {},
      { stroke: 'var(--accent-teal)', width: 2, fill: 'rgba(0, 180, 180, 0.1)' },
    ],
  }

  onMount(() => {
    chart = new uPlot(opts, buildData($winProbHistory), chartEl)
  })

  onDestroy(() => {
    if (chart) chart.destroy()
  })

  $: if (chart && $winProbHistory) {
    chart.setData(buildData($winProbHistory))
  }
</script>

<div class="win-prob-graph">
  <h3 class="section-label">Win Probability</h3>
  <div bind:this={chartEl}></div>
</div>

<style>
  .win-prob-graph { padding: 8px; }
  .section-label { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin: 0 0 4px; }
</style>
```

- [ ] **Step 5: Create MatchQueue.svelte**

```svelte
<!-- webui/src/lib/MatchQueue.svelte -->
<script>
  import { showcaseQueue, sidecarAlive } from '../stores/showcase.js'
  import { sendShowcaseCommand } from './ws.js'

  function cancelMatch(queueId) {
    sendShowcaseCommand({ type: 'cancel_showcase_match', queue_id: queueId })
  }
</script>

<div class="match-queue">
  <div class="sidecar-status">
    <span class="dot" class:alive={$sidecarAlive} class:dead={!$sidecarAlive}></span>
    <span class="status-text">{$sidecarAlive ? 'Engine online' : 'Engine offline'}</span>
  </div>

  {#if $showcaseQueue.length > 0}
    <div class="queue-list">
      {#each $showcaseQueue as q}
        <div class="queue-item" class:running={q.status === 'running'}>
          <span class="q-status">{q.status}</span>
          <span class="q-entries">{q.entry_id_1} vs {q.entry_id_2}</span>
          <span class="q-speed">{q.speed}</span>
          {#if q.status === 'pending'}
            <button class="cancel-btn" on:click={() => cancelMatch(q.id)}>Cancel</button>
          {/if}
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .match-queue { padding: 8px; font-size: 12px; }
  .sidecar-status { display: flex; align-items: center; gap: 6px; margin-bottom: 8px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot.alive { background: var(--accent-teal); }
  .dot.dead { background: var(--accent-gold); }
  .status-text { color: var(--text-secondary); }
  .queue-list { display: flex; flex-direction: column; gap: 4px; }
  .queue-item { display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 4px; border: 1px solid var(--border); }
  .queue-item.running { border-color: var(--accent-teal); }
  .q-status { font-weight: 600; text-transform: uppercase; font-size: 10px; min-width: 60px; }
  .q-entries { flex: 1; }
  .q-speed { color: var(--text-muted); text-transform: capitalize; }
  .cancel-btn { font-size: 11px; padding: 2px 8px; border: 1px solid var(--border); border-radius: 3px; background: transparent; color: var(--text-secondary); cursor: pointer; }
</style>
```

- [ ] **Step 6: Create ShowcaseView.svelte**

```svelte
<!-- webui/src/lib/ShowcaseView.svelte -->
<script>
  import { showcaseGame, showcaseMoves, showcaseBoard, sidecarAlive } from '../stores/showcase.js'
  import { safeParse } from './safeParse.js'
  import Board from './Board.svelte'
  import PieceTray from './PieceTray.svelte'
  import MoveLog from './MoveLog.svelte'
  import EvalBar from './EvalBar.svelte'
  import MatchControls from './MatchControls.svelte'
  import CommentaryPanel from './CommentaryPanel.svelte'
  import WinProbGraph from './WinProbGraph.svelte'
  import MatchQueue from './MatchQueue.svelte'

  $: move = $showcaseBoard
  $: board = move ? safeParse(move.board_json, []) : []
  $: hands = move ? safeParse(move.hands_json, {}) : {}
  $: game = $showcaseGame
  $: moveHistoryJson = JSON.stringify(
    ($showcaseMoves || []).map(m => ({
      action: m.action_index,
      notation: m.usi_notation,
    }))
  )
</script>

<div class="showcase-view">
  <MatchControls />

  {#if !$sidecarAlive}
    <div class="offline-banner">
      Showcase engine is offline. Start the sidecar to enable live matches.
    </div>
  {/if}

  {#if game}
    <div class="game-area">
      <div class="game-header">
        <span class="player black">{game.name_black} ({game.elo_black?.toFixed(0) ?? '?'})</span>
        <span class="vs">vs</span>
        <span class="player white">{game.name_white} ({game.elo_white?.toFixed(0) ?? '?'})</span>
        <span class="ply">Ply {game.total_ply}</span>
        {#if game.status !== 'in_progress'}
          <span class="result">{game.status.replaceAll('_', ' ')}</span>
        {/if}
      </div>

      <div class="game-content">
        <div class="board-side">
          <PieceTray color="white" hand={hands.white || {}} />
          <Board
            board={board}
            inCheck={!!move?.in_check}
            currentPlayer={move?.current_player || 'black'}
          />
          <PieceTray color="black" hand={hands.black || {}} />
        </div>

        <div class="eval-side">
          <EvalBar
            value={(move?.value_estimate ?? 0.5) * 2 - 1}
            currentPlayer={move?.current_player || 'black'}
          />
        </div>

        <div class="commentary-side">
          <CommentaryPanel />
          <WinProbGraph />
        </div>

        <div class="moves-side">
          <MoveLog
            moveHistoryJson={moveHistoryJson}
            currentPlayer={move?.current_player || 'black'}
          />
        </div>
      </div>
    </div>
  {:else}
    <div class="no-game">
      <p>No match in progress.</p>
      <p class="hint">Select two entries above and start a match!</p>
    </div>
  {/if}

  <MatchQueue />
</div>

<style>
  .showcase-view {
    display: flex;
    flex-direction: column;
    height: 100%;
    overflow: hidden;
  }
  .offline-banner {
    padding: 8px 16px;
    background: var(--accent-gold);
    color: #000;
    font-size: 13px;
    font-weight: 600;
    text-align: center;
  }
  .game-area { flex: 1; overflow: hidden; display: flex; flex-direction: column; }
  .game-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 14px;
  }
  .player { font-weight: 600; }
  .player.black { color: var(--text-primary); }
  .player.white { color: var(--text-secondary); }
  .vs { color: var(--text-muted); font-size: 12px; }
  .ply { color: var(--text-muted); font-size: 12px; margin-left: auto; }
  .result { color: var(--accent-teal); font-weight: 600; text-transform: capitalize; }
  .game-content {
    flex: 1;
    display: flex;
    gap: 16px;
    padding: 8px;
    overflow: hidden;
    min-height: 0;
  }
  .board-side {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    justify-content: center;
  }
  .eval-side { display: flex; flex-shrink: 0; }
  .commentary-side {
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 280px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
  }
  .moves-side {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .no-game {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    gap: 8px;
  }
  .hint { font-size: 13px; }
</style>
```

- [ ] **Step 7: Update App.svelte to render ShowcaseView**

In `webui/src/App.svelte`, add the import at the top of the `<script>` block:

```javascript
import ShowcaseView from './lib/ShowcaseView.svelte'
```

Then change the conditional rendering (around line 155-238). Replace:

```svelte
{:else}
    <LeagueView />
  {/if}
```

With:

```svelte
{:else if $activeTab === 'league'}
    <LeagueView />
  {:else if $activeTab === 'showcase'}
    <ShowcaseView />
  {/if}
```

- [ ] **Step 8: Build and verify**

Run: `cd webui && npm run build`
Expected: Build succeeds with no errors

- [ ] **Step 9: Commit**

```bash
git add webui/src/lib/ShowcaseView.svelte webui/src/lib/MatchControls.svelte \
  webui/src/lib/CommentaryPanel.svelte webui/src/lib/WinProbGraph.svelte \
  webui/src/lib/MatchQueue.svelte webui/src/lib/TabBar.svelte webui/src/App.svelte
git commit -m "feat(webui): showcase tab with board, commentary, controls, and queue"
```

---

## Task 8: SQLite Concurrency Tests

**Files:**
- Modify: `tests/test_showcase_db.py`

- [ ] **Step 1: Add concurrency tests**

Append to `tests/test_showcase_db.py`:

```python
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

        # Write some moves
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
```

- [ ] **Step 2: Add write_showcase_move import at top of file**

Make sure `write_showcase_move` and `read_showcase_moves_since` are in the imports from `keisei.showcase.db_ops` at the top of the test file.

- [ ] **Step 3: Run concurrency tests**

Run: `uv run pytest tests/test_showcase_db.py::TestSQLiteConcurrency -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_showcase_db.py
git commit -m "test(showcase): SQLite concurrency tests for moves and queue claiming"
```

---

## Task 9: Integration Test — Full Showcase Game

**Files:**
- Modify: `tests/test_showcase_runner.py`

- [ ] **Step 1: Add integration test with real (tiny) model**

Append to `tests/test_showcase_runner.py`:

```python
class TestShowcaseIntegration:
    """Integration tests using real (tiny) models."""

    @pytest.fixture
    def tiny_model_checkpoint(self, tmp_path: Path) -> tuple[str, Path]:
        """Create a tiny MLP model checkpoint."""
        from keisei.training.model_registry import build_model
        arch = "mlp"
        params = {}  # use defaults
        model = build_model(arch, params)
        ckpt = tmp_path / "tiny.pt"
        torch.save(model.state_dict(), ckpt)
        return arch, ckpt

    def test_inference_cpu_only(self, tiny_model_checkpoint: tuple[str, Path]) -> None:
        """Verify inference runs on CPU and produces valid output."""
        from keisei.showcase.inference import load_model_for_showcase, run_inference
        arch, ckpt = tiny_model_checkpoint
        model = load_model_for_showcase(ckpt, arch, {})

        # All params should be on CPU
        for name, param in model.named_parameters():
            assert param.device == torch.device("cpu"), f"{name} on {param.device}"

        # Run inference
        obs = np.random.randn(50, 9, 9).astype(np.float32)
        policy, win_prob = run_inference(model, obs, arch)
        assert policy.shape[0] > 0
        assert 0.0 <= win_prob <= 1.0

    def test_cuda_not_available_in_showcase(self) -> None:
        """After enforce_cpu_only, CUDA should not be available."""
        from keisei.showcase.inference import enforce_cpu_only
        enforce_cpu_only(cpu_threads=1)
        # Note: this test may not work if CUDA is not installed,
        # but it should not fail — torch.cuda.is_available() returns False
        # when CUDA_VISIBLE_DEVICES=""
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
```

- [ ] **Step 2: Add missing import**

Add `import os` at the top of the test file if not already present.

- [ ] **Step 3: Run integration tests**

Run: `uv run pytest tests/test_showcase_runner.py::TestShowcaseIntegration -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_showcase_runner.py
git commit -m "test(showcase): integration tests with real model and CPU enforcement"
```

---

## Task 10: Final Wiring and Verification

**Files:**
- Create: `keisei/showcase/__main__.py`

- [ ] **Step 1: Create `__main__.py` entry point**

```python
# keisei/showcase/__main__.py
"""Entry point for `python -m keisei.showcase.runner`."""
from keisei.showcase.runner import main

main()
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/test_showcase_db.py tests/test_showcase_runner.py tests/test_showcase_inference.py tests/test_server_showcase.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run existing test suite for regressions**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: No regressions in existing tests

- [ ] **Step 4: Build frontend**

Run: `cd webui && npm run build`
Expected: Build succeeds

- [ ] **Step 5: Run frontend tests**

Run: `cd webui && npx vitest run`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/showcase/__main__.py
git commit -m "feat(showcase): add __main__.py entry point for sidecar process"
```

- [ ] **Step 7: Final commit with all remaining changes**

Run `git status` to check for any unstaged files. Stage and commit anything missed.

---

## Deferred Work

These items from the spec are deliberately deferred from this plan:

- **Refactoring `_poll_and_push` into coroutine-per-domain**: The spec recommends splitting the 150-line function into separate coroutines (metrics, games, league, showcase). This plan adds showcase polling as a separate `_poll_showcase()` coroutine in the TaskGroup, which is the right first step. The full refactor of the existing polling should be a separate task to avoid scope creep.
- **Checkpoint TOCTOU test**: The spec mentions testing what happens when a checkpoint file is deleted between queue claim and `torch.load()`. The runner handles this with a try/except in `_load_models()`, but a dedicated test for this edge case should be added as follow-up.
- **`_poll_and_push` refactoring**: Existing metrics/games/league polling should eventually be split into separate coroutines for maintainability. Track as a separate filigree task.

## Post-Implementation Checklist

After all tasks are complete:

- [ ] Verify `python -m keisei.showcase.runner --help` shows usage
- [ ] Verify the Showcase tab appears in the webui when built
- [ ] Benchmark CPU inference latency against the actual deployed model architecture
- [ ] If SE-ResNet latency > 400ms, adjust the Fast preset from 0.5s to 1.0s
- [ ] Conduct user acceptance testing with 2+ observers watching showcase games
