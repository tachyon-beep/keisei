# Fix Learner Entry Lookup in League Dashboard

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the league dashboard showing dashes/zeros for Frontier Elo, League Elo, Challenge, Gauntlet by propagating `learner_entry_id` from the training loop to the frontend.

**Architecture:** Add a `learner_entry_id` column to `training_state`, write it from the training loop on bootstrap and seat rotation, send it via WebSocket, and use it in the frontend's `learnerEntry` derived store to look up by ID instead of display_name.

**Tech Stack:** Python (SQLite, FastAPI/WebSocket), JavaScript (Svelte stores)

---

### Task 1: Add `learner_entry_id` column to DB schema and functions

**Files:**
- Modify: `keisei/db.py:67-81` (schema), `keisei/db.py:331-356` (write_training_state), `keisei/db.py:380-401` (update_training_progress)
- Test: `tests/test_db.py`

- [ ] **Step 1: Write failing tests for learner_entry_id in training_state**

Add to `tests/test_db.py` after the existing `test_update_training_progress_no_checkpoint` test (~line 202):

```python
def test_training_state_learner_entry_id(db: Path) -> None:
    """learner_entry_id should be stored and retrieved from training_state."""
    init_db(str(db))
    write_training_state(str(db), {
        "config_json": '{"test": true}',
        "display_name": "Hikaru",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
        "learner_entry_id": 42,
    })
    state = read_training_state(str(db))
    assert state is not None
    assert state["learner_entry_id"] == 42


def test_training_state_learner_entry_id_defaults_null(db: Path) -> None:
    """learner_entry_id should default to None when not provided."""
    init_db(str(db))
    write_training_state(str(db), {
        "config_json": '{"test": true}',
        "display_name": "Hikaru",
        "model_arch": "resnet",
        "algorithm_name": "ppo",
        "started_at": "2026-04-01T00:00:00Z",
    })
    state = read_training_state(str(db))
    assert state is not None
    assert state["learner_entry_id"] is None


def test_update_training_progress_with_learner_entry_id(db: Path) -> None:
    """update_training_progress should update learner_entry_id when provided."""
    init_db(str(db))
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
    })
    update_training_progress(str(db), epoch=5, step=500, learner_entry_id=99)
    state = read_training_state(str(db))
    assert state is not None
    assert state["learner_entry_id"] == 99


def test_update_training_progress_preserves_learner_entry_id(db: Path) -> None:
    """update_training_progress without learner_entry_id should not clear it."""
    init_db(str(db))
    write_training_state(str(db), {
        "config_json": "{}", "display_name": "X", "model_arch": "resnet",
        "algorithm_name": "ppo", "started_at": "2026-04-01T00:00:00Z",
        "learner_entry_id": 42,
    })
    update_training_progress(str(db), epoch=10, step=1000)
    state = read_training_state(str(db))
    assert state is not None
    assert state["learner_entry_id"] == 42
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py::test_training_state_learner_entry_id tests/test_db.py::test_training_state_learner_entry_id_defaults_null tests/test_db.py::test_update_training_progress_with_learner_entry_id tests/test_db.py::test_update_training_progress_preserves_learner_entry_id -v`

Expected: FAIL — `learner_entry_id` not in schema/function signatures.

- [ ] **Step 3: Add column to schema and update DB functions**

In `keisei/db.py`:

**Schema** (line 80, before the closing `);` of training_state):
Add after `heartbeat_at` line:
```python
                learner_entry_id INTEGER REFERENCES league_entries(id)
```

**write_training_state** (lines 331-356): Add `learner_entry_id` to the INSERT columns and VALUES:
```python
def write_training_state(db_path: str, state: dict[str, Any]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """INSERT OR REPLACE INTO training_state
               (id, config_json, display_name, model_arch, algorithm_name,
                started_at, current_epoch, current_step, checkpoint_path,
                total_epochs, status, phase, learner_entry_id)
               VALUES (1, :config_json, :display_name, :model_arch, :algorithm_name,
                :started_at, :current_epoch, :current_step, :checkpoint_path,
                :total_epochs, :status, :phase, :learner_entry_id)""",
            {
                "config_json": state["config_json"], "display_name": state["display_name"],
                "model_arch": state["model_arch"], "algorithm_name": state["algorithm_name"],
                "started_at": state["started_at"],
                "current_epoch": state.get("current_epoch", 0),
                "current_step": state.get("current_step", 0),
                "checkpoint_path": state.get("checkpoint_path"),
                "total_epochs": state.get("total_epochs"),
                "status": state.get("status", "running"),
                "phase": state.get("phase", "init"),
                "learner_entry_id": state.get("learner_entry_id"),
            },
        )
        conn.commit()
    finally:
        conn.close()
```

**update_training_progress** (lines 380-401): Add optional `learner_entry_id` parameter:
```python
def update_training_progress(
    db_path: str, epoch: int, step: int, checkpoint_path: str | None = None,
    phase: str | None = None, learner_entry_id: int | None = None,
) -> None:
    conn = _connect(db_path)
    try:
        parts = ["current_epoch = ?", "current_step = ?",
                 "heartbeat_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')"]
        params: list[int | str] = [epoch, step]
        if checkpoint_path is not None:
            parts.append("checkpoint_path = ?")
            params.append(checkpoint_path)
        if phase is not None:
            parts.append("phase = ?")
            params.append(phase)
        if learner_entry_id is not None:
            parts.append("learner_entry_id = ?")
            params.append(learner_entry_id)
        conn.execute(
            f"UPDATE training_state SET {', '.join(parts)} WHERE id = 1",
            params,
        )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v`
Expected: ALL PASS (including the 4 new tests and all existing ones).

- [ ] **Step 5: Commit**

```bash
git add keisei/db.py tests/test_db.py
git commit -m "feat: add learner_entry_id column to training_state schema"
```

---

### Task 2: Propagate learner_entry_id from training loop to DB

**Files:**
- Modify: `keisei/training/katago_loop.py:730-752` (write_training_state call), `keisei/training/katago_loop.py:1496` (update_training_progress call), `keisei/training/katago_loop.py:1526-1528` (checkpoint progress call), `keisei/training/katago_loop.py:1550-1551` (_rotate_seat)

- [ ] **Step 1: Add learner_entry_id to initial write_training_state call**

In `keisei/training/katago_loop.py`, in the `write_training_state` call (~line 730), add the field to the dict. The `_learner_entry_id` is set at line 610 before this call happens (at line 730):

Change the dict in the `write_training_state` call to include:
```python
                    "learner_entry_id": self._learner_entry_id,
```

Add it after the `"started_at"` entry (line 749), so the full call reads:
```python
            write_training_state(
                self.db_path,
                {
                    "config_json": json.dumps(
                        {
                            "training": {
                                "num_games": self.config.training.num_games,
                                "algorithm": self.config.training.algorithm,
                            },
                            "model": {
                                "architecture": self.config.model.architecture,
                                "params": dict(self.config.model.params),
                            },
                        }
                    ),
                    "display_name": self.config.model.display_name,
                    "model_arch": self.config.model.architecture,
                    "algorithm_name": self.config.training.algorithm,
                    "started_at": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "learner_entry_id": self._learner_entry_id,
                },
            )
```

- [ ] **Step 2: Propagate on seat rotation**

In `_rotate_seat` (~line 1551), after `self._learner_entry_id = new_entry.id`, add a DB write so the frontend picks up the new ID on the next heartbeat. The simplest approach: add `learner_entry_id` to the `update_training_progress` call at line 1591 (the heartbeat). But the rotation also needs an immediate write. Add after line 1555:

```python
        try:
            update_training_progress(
                self.db_path, epoch + 1, self.global_step,
                learner_entry_id=new_entry.id,
            )
        except Exception:
            logger.exception("Failed to write learner_entry_id after seat rotation")
```

- [ ] **Step 3: Run existing tests to ensure no regressions**

Run: `uv run pytest tests/test_katago_loop.py tests/test_db.py -v`
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "feat: propagate learner_entry_id to training_state DB"
```

---

### Task 3: Send learner_entry_id through WebSocket to frontend

**Files:**
- Modify: `keisei/server/app.py:269-285` (training_status WS message)
- Modify: `webui/src/lib/ws.js:111-126` (training_status handler)
- Test: `tests/test_server_websocket.py`

- [ ] **Step 1: Write failing test for learner_entry_id in WS init message**

Find the existing WebSocket init test in `tests/test_server_websocket.py` and add a test that checks `training_state` includes `learner_entry_id`. First read the test file to find the right fixture/pattern, then add:

```python
def test_init_message_includes_learner_entry_id(server_db: str) -> None:
    """The init WS message should include learner_entry_id from training_state."""
    from keisei.db import read_training_state
    state = read_training_state(server_db)
    # training_state is sent as-is from read_training_state in init message
    # The field must be present (even if None) so the frontend can use it
    assert "learner_entry_id" in state
```

This test confirms the DB-level field is present. The WebSocket `init` message sends `training_state` directly from `read_training_state()`, so if the field is in the DB result it flows through automatically.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server_websocket.py::test_init_message_includes_learner_entry_id -v`
Expected: FAIL (or skip if the fixture creates a fresh DB — which would PASS since Task 1 added the column). If it passes, that confirms the init path works.

- [ ] **Step 3: Add learner_entry_id to the training_status poll message**

The `init` message sends `training_state` as a full dict from `read_training_state()`, which already includes `learner_entry_id` (Task 1 added it to the schema). But the *poll* path at `app.py:269-285` constructs a custom dict that cherry-picks fields. Add `learner_entry_id` to it.

In `keisei/server/app.py`, in the `training_status` WS message construction (~line 270-285), add after `"system_stats"`:

```python
                    "learner_entry_id": new_state.get("learner_entry_id"),
```

So the full block becomes:
```python
            await asyncio.wait_for(
                ws.send_json({
                    "type": "training_status",
                    "status": new_state.get("status"),
                    "phase": new_state.get("phase", ""),
                    "heartbeat_at": new_state.get("heartbeat_at"),
                    "epoch": new_state.get("current_epoch"),
                    "step": new_state.get("current_step"),
                    "episodes": total_episodes,
                    "config_json": new_state.get("config_json"),
                    "display_name": new_state.get("display_name"),
                    "model_arch": new_state.get("model_arch"),
                    "total_epochs": new_state.get("total_epochs"),
                    "system_stats": sys_stats,
                    "learner_entry_id": new_state.get("learner_entry_id"),
                }),
                timeout=WS_SEND_TIMEOUT_S,
            )
```

- [ ] **Step 4: Update frontend WS handler to store learner_entry_id**

In `webui/src/lib/ws.js`, in the `training_status` handler (lines 111-126), the handler already spreads the old state and merges new fields. Add `learner_entry_id`:

```javascript
    case 'training_status':
      trainingState.update(state => ({
        ...state,
        status: msg.status,
        phase: msg.phase || state?.phase,
        heartbeat_at: msg.heartbeat_at,
        current_epoch: msg.epoch,
        current_step: msg.step,
        episodes: msg.episodes ?? state?.episodes,
        config_json: msg.config_json || state?.config_json,
        display_name: msg.display_name || state?.display_name,
        model_arch: msg.model_arch || state?.model_arch,
        total_epochs: msg.total_epochs ?? state?.total_epochs,
        system_stats: msg.system_stats || state?.system_stats,
        learner_entry_id: msg.learner_entry_id ?? state?.learner_entry_id,
      }))
      break
```

Note: use `??` (not `||`) so that `null` from the message doesn't clobber an existing numeric ID with a falsy check, but explicit `null` from the server does override.

- [ ] **Step 5: Run server tests to verify no regressions**

Run: `uv run pytest tests/test_server_websocket.py -v`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/server/app.py webui/src/lib/ws.js
git commit -m "feat: send learner_entry_id through WebSocket to frontend"
```

---

### Task 4: Update frontend learnerEntry store to use ID lookup

**Files:**
- Modify: `webui/src/stores/league.js:208-216` (learnerEntry derived store)

- [ ] **Step 1: Update learnerEntry to look up by ID with display_name fallback**

In `webui/src/stores/league.js`, replace the `learnerEntry` derived store (lines 208-216):

```javascript
/** The league entry matching the current learner (by learner_entry_id from trainingState, with display_name fallback) */
export const learnerEntry = derived(
  [leagueEntries, trainingState],
  ([$entries, $state]) => {
    if (!$state) return null
    const id = $state.learner_entry_id
    if (id != null) {
      return $entries.find(e => e.id === id) || null
    }
    // Fallback for older backends that don't send learner_entry_id
    const name = $state.display_name
    if (!name) return null
    return $entries.find(e => e.display_name === name) || null
  }
)
```

- [ ] **Step 2: Verify the webui builds without errors**

Run: `cd webui && npm run build` (or whatever the build command is — check `webui/package.json` for the build script)
Expected: Build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
git add webui/src/stores/league.js
git commit -m "fix: look up learner entry by ID instead of display_name"
```

---

### Task 5: Add schema migration for existing databases

**Files:**
- Modify: `keisei/db.py:19-25` (init_db function)
- Test: `tests/test_db.py`

- [ ] **Step 1: Write failing test for migration of old schema**

Add to `tests/test_db.py`:

```python
def test_init_db_migrates_learner_entry_id(tmp_path: Path) -> None:
    """init_db should add learner_entry_id column to existing training_state tables."""
    db = tmp_path / "migrate.db"
    # Create a DB with the old schema (no learner_entry_id)
    conn = sqlite3.connect(str(db))
    conn.execute("""CREATE TABLE training_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        config_json TEXT NOT NULL,
        display_name TEXT NOT NULL,
        model_arch TEXT NOT NULL,
        algorithm_name TEXT NOT NULL,
        started_at TEXT NOT NULL,
        current_epoch INTEGER NOT NULL DEFAULT 0,
        current_step INTEGER NOT NULL DEFAULT 0,
        checkpoint_path TEXT,
        total_epochs INTEGER,
        status TEXT NOT NULL DEFAULT 'running',
        phase TEXT NOT NULL DEFAULT 'init',
        heartbeat_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
    )""")
    conn.execute(
        "INSERT INTO training_state (id, config_json, display_name, model_arch, "
        "algorithm_name, started_at) VALUES (1, '{}', 'Test', 'resnet', 'ppo', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    # Run init_db — should migrate
    init_db(str(db))

    # Verify the column exists and existing data is preserved
    state = read_training_state(str(db))
    assert state is not None
    assert state["display_name"] == "Test"
    assert state["learner_entry_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py::test_init_db_migrates_learner_entry_id -v`
Expected: FAIL — column doesn't get added to existing tables.

- [ ] **Step 3: Add migration logic to init_db**

In `keisei/db.py`, add migration logic at the end of `init_db()`, after the `executescript` block but before `conn.close()`. Add after the existing schema creation (after line ~175, before the `finally` block):

```python
        # Migrations — add columns that may be missing from older schemas.
        # SQLite ignores ALTER TABLE ADD COLUMN if the column already exists
        # only in newer versions; we catch the error for broad compatibility.
        _migrate_add_column(
            conn,
            "training_state",
            "learner_entry_id",
            "INTEGER REFERENCES league_entries(id)",
        )
```

And add this helper function before `init_db`:

```python
def _migrate_add_column(
    conn: sqlite3.Connection, table: str, column: str, col_type: str,
) -> None:
    """Add a column to a table if it doesn't already exist."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e).lower():
            raise
```

- [ ] **Step 4: Run all DB tests**

Run: `uv run pytest tests/test_db.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add keisei/db.py tests/test_db.py
git commit -m "feat: add schema migration for learner_entry_id column"
```

---

### Task 6: Full integration verification

- [ ] **Step 1: Run entire test suite**

Run: `uv run pytest -x -v`
Expected: ALL PASS. Pay attention to tests in `test_katago_loop.py`, `test_server_websocket.py`, `test_db.py`, and `test_db_league_schema.py`.

- [ ] **Step 2: Verify the live fix**

If training is running, verify the fix works end-to-end:
1. Restart the server so it picks up the code changes (`keisei-serve`)
2. The server calls `init_db` on startup (app.py:133), which will run the migration
3. The next heartbeat from the training loop will write `learner_entry_id`
4. Refresh the dashboard — the league tab should now show actual Elo values

Note: The training process itself also needs to be restarted for it to write `learner_entry_id` to the DB, since the change is in `katago_loop.py`. Until restart, the server will see `learner_entry_id = NULL` and the frontend will fall back to display_name matching (which still won't work). A full restart of both training and server is needed.

- [ ] **Step 3: Final commit (if any fixups needed)**

```bash
git add -u
git commit -m "fix: learner entry lookup — full integration"
```
