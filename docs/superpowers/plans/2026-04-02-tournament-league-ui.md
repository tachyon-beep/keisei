# Tournament/League UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface league standings, Elo trends, match history, and opponent identity in the Keisei training dashboard via a new League tab and revised Training layout.

**Architecture:** Backend adds an `elo_history` table and `opponent_id` column, league data DB readers, and a 5s-cadence WebSocket `league_update` message. Frontend adds a tab bar, league stores, player cards, league view components, and refactors the metrics grid to mini sparklines with click-to-expand.

**Tech Stack:** Python/FastAPI/SQLite (backend), Svelte 4/Vite/uPlot/Vitest (frontend)

**Spec:** `docs/superpowers/specs/2026-04-02-tournament-league-ui-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `webui/src/stores/navigation.js` | `activeTab` writable store |
| `webui/src/stores/league.js` | `leagueEntries`, `leagueResults`, `eloHistory` writables + `leagueRanked` derived |
| `webui/src/lib/TabBar.svelte` | Tab toggle (Training \| League) |
| `webui/src/lib/PlayerCard.svelte` | Learner/opponent identity card |
| `webui/src/lib/LeagueView.svelte` | League tab container |
| `webui/src/lib/LeagueTable.svelte` | Sortable Elo leaderboard |
| `webui/src/lib/MatchHistory.svelte` | Inline match results |
| `webui/src/lib/eloChartData.js` | Transform flat elo_history → grouped chart series |
| `webui/src/lib/eloChartData.test.js` | Tests for chart data helper |
| `webui/src/stores/league.test.js` | Tests for league stores |

### Modified Files

| File | Changes |
|------|---------|
| `keisei/db.py` | Add `elo_history` table, `opponent_id` column on `game_snapshots`, `read_league_data()` + `read_elo_history()` helpers |
| `keisei/training/league.py` | `update_elo()` writes to `elo_history`, `_delete_entry()` cleans up history |
| `keisei/server/app.py` | Import league readers, add `LEAGUE_POLL_INTERVAL_S`, league polling in `_poll_and_push()`, extend `init` message |
| `webui/src/lib/ws.js` | Import league stores, handle `league_update` case, extend `init` case |
| `webui/src/stores/games.js` | Add `selectedOpponent` derived store |
| `webui/src/App.svelte` | Tab routing, 3-column layout, player cards, revised metrics area |
| `webui/src/lib/StatusIndicator.svelte` | Integrate TabBar, remove player name (moved to PlayerCard) |
| `webui/src/lib/MetricsGrid.svelte` | 4×1 mini sparklines with click-to-expand |
| `webui/src/lib/MetricsChart.svelte` | Add `compact` prop for mini mode |
| `webui/src/app.css` | Tab bar and player card CSS variables |
| `tests/test_league.py` | Tests for `update_elo` elo_history writes, `_delete_entry` cascade |
| `tests/test_server.py` | Test `init` message includes league data |
| `webui/src/lib/ws.test.js` | Tests for `league_update` and extended `init` |
| `webui/src/stores/games.test.js` | Tests for `selectedOpponent` |

---

## Task 1: DB Schema — `elo_history` table and `opponent_id` column

**Files:**
- Modify: `keisei/db.py:24-98` (inside `init_db` executescript)
- Modify: `keisei/db.py:148-163` (`write_game_snapshots`)
- Test: `tests/test_db_schema_v2.py`

- [ ] **Step 1: Write failing test for `elo_history` table existence**

Add to `tests/test_db_schema_v2.py`:

```python
def test_creates_elo_history_table(self, tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    conn.close()
    assert "elo_history" in tables
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db_schema_v2.py::TestSchemaV2::test_creates_elo_history_table -v`
Expected: FAIL — `elo_history` not in tables

- [ ] **Step 3: Write failing test for `elo_history` columns**

Add to `tests/test_db_schema_v2.py`:

```python
def test_elo_history_columns(self, tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(elo_history)").fetchall()]
    conn.close()
    assert cols == ["id", "entry_id", "epoch", "elo_rating", "recorded_at"]
```

- [ ] **Step 4: Write failing test for `opponent_id` column on `game_snapshots`**

Add to `tests/test_db_schema_v2.py`:

```python
def test_game_snapshots_has_opponent_id(self, tmp_path):
    db_path = str(tmp_path / "test.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(game_snapshots)").fetchall()]
    conn.close()
    assert "opponent_id" in cols
```

- [ ] **Step 5: Implement schema changes in `init_db()`**

In `keisei/db.py`, add to the `executescript` block after the `league_entries` index:

```sql
CREATE TABLE IF NOT EXISTS elo_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id    INTEGER NOT NULL REFERENCES league_entries(id),
    epoch       INTEGER NOT NULL,
    elo_rating  REAL NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_elo_history_entry ON elo_history(entry_id);
```

And add `opponent_id` column to the `game_snapshots` CREATE TABLE, after `demo_slot`:

```sql
opponent_id       INTEGER REFERENCES league_entries(id),
```

- [ ] **Step 6: Update `write_game_snapshots` to include `opponent_id`**

In `keisei/db.py`, update the `write_game_snapshots` INSERT to include `game_type`, `demo_slot`, and `opponent_id`:

```python
def write_game_snapshots(db_path: str, snapshots: list[dict[str, Any]]) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN")
        for snap in snapshots:
            conn.execute(
                """INSERT OR REPLACE INTO game_snapshots
                   (game_id, board_json, hands_json, current_player, ply,
                    is_over, result, sfen, in_check, move_history_json,
                    value_estimate, game_type, demo_slot, opponent_id)
                   VALUES (:game_id, :board_json, :hands_json, :current_player,
                    :ply, :is_over, :result, :sfen, :in_check, :move_history_json,
                    :value_estimate, :game_type, :demo_slot, :opponent_id)""",
                {
                    "game_id": snap["game_id"],
                    "board_json": snap["board_json"],
                    "hands_json": snap["hands_json"],
                    "current_player": snap["current_player"],
                    "ply": snap["ply"],
                    "is_over": snap["is_over"],
                    "result": snap["result"],
                    "sfen": snap["sfen"],
                    "in_check": snap["in_check"],
                    "move_history_json": snap["move_history_json"],
                    "value_estimate": snap.get("value_estimate", 0.0),
                    "game_type": snap.get("game_type", "live"),
                    "demo_slot": snap.get("demo_slot"),
                    "opponent_id": snap.get("opponent_id"),
                },
            )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 7: Run all three schema tests to verify they pass**

Run: `uv run pytest tests/test_db_schema_v2.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add keisei/db.py tests/test_db_schema_v2.py
git commit -m "feat(db): add elo_history table and opponent_id to game_snapshots"
```

---

## Task 2: DB Readers — `read_league_data()` and `read_elo_history()`

**Files:**
- Modify: `keisei/db.py` (add new functions at end of file)
- Test: `tests/test_db.py`

- [ ] **Step 1: Write failing tests for league data readers**

Add to `tests/test_db.py`:

```python
from keisei.db import init_db, read_league_data, read_elo_history


class TestLeagueDataReaders:
    def test_read_league_data_empty(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        data = read_league_data(db_path)
        assert data["entries"] == []
        assert data["results"] == []

    def test_read_league_data_with_entries(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES ('transformer', '{}', '/tmp/ckpt.pt', 5)"
        )
        conn.execute(
            "INSERT INTO league_results (epoch, learner_id, opponent_id, wins, losses, draws) "
            "VALUES (5, 1, 1, 3, 1, 1)"
        )
        conn.commit()
        conn.close()
        data = read_league_data(db_path)
        assert len(data["entries"]) == 1
        assert data["entries"][0]["architecture"] == "transformer"
        assert len(data["results"]) == 1
        assert data["results"][0]["wins"] == 3

    def test_read_elo_history_empty(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        history = read_elo_history(db_path)
        assert history == []

    def test_read_elo_history_with_data(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
            "VALUES ('transformer', '{}', '/tmp/ckpt.pt', 5)"
        )
        conn.execute("INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (1, 5, 1050.0)")
        conn.execute("INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (1, 6, 1100.0)")
        conn.commit()
        conn.close()
        history = read_elo_history(db_path)
        assert len(history) == 2
        assert history[0]["elo_rating"] == 1050.0
        assert history[1]["epoch"] == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py::TestLeagueDataReaders -v`
Expected: FAIL — `read_league_data` / `read_elo_history` not defined

- [ ] **Step 3: Implement `read_league_data()` and `read_elo_history()`**

Add to end of `keisei/db.py`:

```python
def read_league_data(db_path: str) -> dict[str, list[dict[str, Any]]]:
    """Read all league entries and results."""
    conn = _connect(db_path)
    try:
        entries = conn.execute(
            "SELECT id, architecture, elo_rating, games_played, created_epoch, created_at "
            "FROM league_entries ORDER BY elo_rating DESC"
        ).fetchall()
        results = conn.execute(
            "SELECT id, epoch, learner_id, opponent_id, wins, losses, draws, recorded_at "
            "FROM league_results ORDER BY id DESC"
        ).fetchall()
        return {
            "entries": [dict(r) for r in entries],
            "results": [dict(r) for r in results],
        }
    finally:
        conn.close()


def read_elo_history(db_path: str) -> list[dict[str, Any]]:
    """Read all Elo history points for charting."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT entry_id, epoch, elo_rating FROM elo_history ORDER BY epoch, entry_id"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py::TestLeagueDataReaders -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/db.py tests/test_db.py
git commit -m "feat(db): add read_league_data and read_elo_history readers"
```

---

## Task 3: OpponentPool — Write Elo history on update, cascade on delete

**Files:**
- Modify: `keisei/training/league.py:211-220` (`update_elo`) and `keisei/training/league.py:171-186` (`_delete_entry`)
- Test: `tests/test_league.py`

- [ ] **Step 1: Write failing test for `update_elo` writing to `elo_history`**

Add to `TestOpponentPool` in `tests/test_league.py`:

```python
def test_update_elo_writes_history(self, league_db, league_dir):
    import sqlite3
    pool = OpponentPool(league_db, str(league_dir), max_pool_size=5)
    model = torch.nn.Linear(10, 10)
    pool.add_snapshot(model, "resnet", {}, epoch=0)
    entry = pool.list_entries()[0]

    pool.update_elo(entry.id, 1050.0, epoch=3)
    pool.update_elo(entry.id, 1100.0, epoch=5)

    conn = sqlite3.connect(league_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT entry_id, epoch, elo_rating FROM elo_history ORDER BY epoch"
    ).fetchall()
    conn.close()
    assert len(rows) == 2
    assert dict(rows[0]) == {"entry_id": entry.id, "epoch": 3, "elo_rating": 1050.0}
    assert dict(rows[1]) == {"entry_id": entry.id, "epoch": 5, "elo_rating": 1100.0}
```

- [ ] **Step 2: Write failing test for `_delete_entry` cascading to `elo_history`**

Add to `TestOpponentPool` in `tests/test_league.py`:

```python
def test_delete_entry_cascades_elo_history(self, league_db, league_dir):
    import sqlite3
    pool = OpponentPool(league_db, str(league_dir), max_pool_size=1)
    model = torch.nn.Linear(10, 10)
    pool.add_snapshot(model, "resnet", {}, epoch=0)
    entry = pool.list_entries()[0]
    pool.update_elo(entry.id, 1050.0, epoch=1)

    # Adding a second snapshot triggers eviction of the first
    pool.add_snapshot(model, "resnet", {}, epoch=1)

    conn = sqlite3.connect(league_db)
    count = conn.execute("SELECT COUNT(*) FROM elo_history WHERE entry_id = ?", (entry.id,)).fetchone()[0]
    conn.close()
    assert count == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_league.py::TestOpponentPool::test_update_elo_writes_history tests/test_league.py::TestOpponentPool::test_delete_entry_cascades_elo_history -v`
Expected: FAIL — `update_elo()` doesn't accept `epoch` parameter

- [ ] **Step 4: Implement `update_elo` with `elo_history` write**

In `keisei/training/league.py`, replace the `update_elo` method:

```python
def update_elo(self, entry_id: int, new_elo: float, epoch: int = 0) -> None:
    conn = self._connect()
    try:
        conn.execute(
            "UPDATE league_entries SET elo_rating = ? WHERE id = ?",
            (new_elo, entry_id),
        )
        conn.execute(
            "INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (?, ?, ?)",
            (entry_id, epoch, new_elo),
        )
        conn.commit()
    finally:
        conn.close()
```

- [ ] **Step 5: Implement `_delete_entry` cascade to `elo_history`**

In `keisei/training/league.py`, update `_delete_entry` to delete elo_history rows before the entry. Add this line after the `DELETE FROM league_results` line:

```python
conn.execute(
    "DELETE FROM elo_history WHERE entry_id = ?",
    (entry.id,),
)
```

- [ ] **Step 6: Update existing callers of `update_elo` to pass `epoch`**

Search the codebase for calls to `update_elo`. The `epoch` parameter defaults to `0`, so existing callers won't break, but they should pass the real epoch where available. Check `keisei/training/evaluate.py` and `keisei/training/katago_loop.py` for calls.

Run: `uv run grep -rn "update_elo" keisei/` to find all call sites and update them.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_league.py::TestOpponentPool -v`
Expected: All PASS (including pre-existing tests)

- [ ] **Step 8: Commit**

```bash
git add keisei/training/league.py tests/test_league.py
git commit -m "feat(league): write elo_history on update, cascade on delete"
```

---

## Task 4: Server — League data in WebSocket init and poll

**Files:**
- Modify: `keisei/server/app.py:16-21` (imports), `keisei/server/app.py:140-215` (`_poll_and_push`)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write failing test for league data in init message**

Add to `tests/test_server.py`:

```python
def test_ws_init_includes_league_data(db_path: str) -> None:
    app = create_app(db_path)
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "init"
        assert "league_entries" in msg
        assert "league_results" in msg
        assert "elo_history" in msg
        assert isinstance(msg["league_entries"], list)
```

- [ ] **Step 2: Write failing test for league data in init with entries present**

Add to `tests/test_server.py`:

```python
def test_ws_init_league_data_populated(db_path: str) -> None:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO league_entries (architecture, model_params, checkpoint_path, created_epoch) "
        "VALUES ('transformer', '{}', '/tmp/ckpt.pt', 5)"
    )
    conn.execute("INSERT INTO elo_history (entry_id, epoch, elo_rating) VALUES (1, 5, 1050.0)")
    conn.commit()
    conn.close()

    app = create_app(db_path)
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert len(msg["league_entries"]) == 1
        assert msg["league_entries"][0]["architecture"] == "transformer"
        assert len(msg["elo_history"]) == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py::test_ws_init_includes_league_data tests/test_server.py::test_ws_init_league_data_populated -v`
Expected: FAIL — `league_entries` not in init message

- [ ] **Step 4: Implement league data in `_poll_and_push`**

In `keisei/server/app.py`, add imports at top:

```python
from keisei.db import (
    read_game_snapshots,
    read_game_snapshots_since,
    read_league_data,
    read_elo_history,
    read_metrics_since,
    read_training_state,
)
```

Add constant:

```python
LEAGUE_POLL_INTERVAL_S = 5.0
```

In `_poll_and_push`, after the init `state` fetch but before sending the init message, add:

```python
league_data = await asyncio.to_thread(read_league_data, db_path)
elo_history = await asyncio.to_thread(read_elo_history, db_path)
```

Extend the init message:

```python
await asyncio.wait_for(
    ws.send_json({
        "type": "init",
        "games": games,
        "metrics": metrics,
        "training_state": state,
        "league_entries": league_data["entries"],
        "league_results": league_data["results"],
        "elo_history": elo_history,
    }),
    timeout=WS_SEND_TIMEOUT_S,
)
```

Add league polling inside the `while True` loop. Track state with:

```python
last_league_entry_count = len(league_data["entries"])
last_league_result_id = league_data["results"][0]["id"] if league_data["results"] else 0
league_poll_elapsed = 0.0
```

Then inside the loop, after the existing polls:

```python
league_poll_elapsed += POLL_INTERVAL_S
if league_poll_elapsed >= LEAGUE_POLL_INTERVAL_S:
    league_poll_elapsed = 0.0
    new_league = await asyncio.to_thread(read_league_data, db_path)
    new_elo_hist = await asyncio.to_thread(read_elo_history, db_path)
    new_entry_count = len(new_league["entries"])
    new_result_id = new_league["results"][0]["id"] if new_league["results"] else 0
    if new_entry_count != last_league_entry_count or new_result_id != last_league_result_id:
        last_league_entry_count = new_entry_count
        last_league_result_id = new_result_id
        await asyncio.wait_for(
            ws.send_json({
                "type": "league_update",
                "entries": new_league["entries"],
                "results": new_league["results"],
                "elo_history": new_elo_hist,
            }),
            timeout=WS_SEND_TIMEOUT_S,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/server/app.py tests/test_server.py
git commit -m "feat(server): include league data in WebSocket init and poll"
```

---

## Task 5: Frontend stores — navigation, league, selectedOpponent

**Files:**
- Create: `webui/src/stores/navigation.js`
- Create: `webui/src/stores/league.js`
- Create: `webui/src/stores/league.test.js`
- Modify: `webui/src/stores/games.js`
- Modify: `webui/src/stores/games.test.js`

- [ ] **Step 1: Create `navigation.js`**

```javascript
import { writable } from 'svelte/store'

export const activeTab = writable('training')
```

- [ ] **Step 2: Write failing tests for league stores**

Create `webui/src/stores/league.test.js`:

```javascript
import { describe, it, expect, beforeEach } from 'vitest'
import { get } from 'svelte/store'
import { leagueEntries, leagueResults, eloHistory, leagueRanked } from './league.js'

beforeEach(() => {
  leagueEntries.set([])
  leagueResults.set([])
  eloHistory.set([])
})

describe('leagueRanked', () => {
  it('returns empty array when no entries', () => {
    expect(get(leagueRanked)).toEqual([])
  })

  it('sorts entries by elo_rating descending and injects rank', () => {
    leagueEntries.set([
      { id: 1, architecture: 'a', elo_rating: 900, games_played: 10, created_epoch: 1 },
      { id: 2, architecture: 'b', elo_rating: 1200, games_played: 20, created_epoch: 2 },
      { id: 3, architecture: 'c', elo_rating: 1100, games_played: 15, created_epoch: 3 },
    ])
    const ranked = get(leagueRanked)
    expect(ranked[0]).toEqual(expect.objectContaining({ id: 2, rank: 1, elo_rating: 1200 }))
    expect(ranked[1]).toEqual(expect.objectContaining({ id: 3, rank: 2, elo_rating: 1100 }))
    expect(ranked[2]).toEqual(expect.objectContaining({ id: 1, rank: 3, elo_rating: 900 }))
  })
})
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd webui && npx vitest run src/stores/league.test.js`
Expected: FAIL — `./league.js` not found

- [ ] **Step 4: Create `league.js`**

Create `webui/src/stores/league.js`:

```javascript
import { writable, derived } from 'svelte/store'

export const leagueEntries = writable([])
export const leagueResults = writable([])
export const eloHistory = writable([])

export const leagueRanked = derived(leagueEntries, ($entries) => {
  const sorted = [...$entries].sort((a, b) => b.elo_rating - a.elo_rating)
  return sorted.map((entry, i) => ({ ...entry, rank: i + 1 }))
})
```

- [ ] **Step 5: Run league store tests to verify they pass**

Run: `cd webui && npx vitest run src/stores/league.test.js`
Expected: All PASS

- [ ] **Step 6: Write failing test for `selectedOpponent`**

Add to `webui/src/stores/games.test.js`:

```javascript
import { leagueEntries } from './league.js'
import { selectedOpponent } from './games.js'

describe('selectedOpponent derived store', () => {
  beforeEach(() => {
    leagueEntries.set([])
  })

  it('returns null when game has no opponent_id', () => {
    games.set([{ game_id: 0, opponent_id: null }])
    selectedGameId.set(0)
    expect(get(selectedOpponent)).toBeNull()
  })

  it('returns null when no league entries exist', () => {
    games.set([{ game_id: 0, opponent_id: 5 }])
    selectedGameId.set(0)
    expect(get(selectedOpponent)).toBeNull()
  })

  it('returns opponent entry when opponent_id matches a league entry', () => {
    leagueEntries.set([
      { id: 5, architecture: 'transformer_ep00008', elo_rating: 1180, games_played: 124 },
    ])
    games.set([{ game_id: 0, opponent_id: 5 }])
    selectedGameId.set(0)
    const opp = get(selectedOpponent)
    expect(opp).toEqual({
      architecture: 'transformer_ep00008',
      elo_rating: 1180,
      games_played: 124,
    })
  })
})
```

- [ ] **Step 7: Run test to verify it fails**

Run: `cd webui && npx vitest run src/stores/games.test.js`
Expected: FAIL — `selectedOpponent` not exported from games.js

- [ ] **Step 8: Implement `selectedOpponent` in `games.js`**

Add to `webui/src/stores/games.js`:

```javascript
import { writable, derived } from 'svelte/store'
import { leagueEntries } from './league.js'

export const games = writable([])
export const selectedGameId = writable(0)
export const selectedGame = derived(
  [games, selectedGameId],
  ([$games, $id]) => $games.find(g => g.game_id === $id) || $games[0] || null
)

export const selectedOpponent = derived(
  [selectedGame, leagueEntries],
  ([$game, $entries]) => {
    if (!$game?.opponent_id) return null
    const entry = $entries.find(e => e.id === $game.opponent_id)
    if (!entry) return null
    return {
      architecture: entry.architecture,
      elo_rating: entry.elo_rating,
      games_played: entry.games_played,
    }
  }
)
```

- [ ] **Step 9: Run all store tests to verify they pass**

Run: `cd webui && npx vitest run src/stores/`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add webui/src/stores/navigation.js webui/src/stores/league.js webui/src/stores/league.test.js webui/src/stores/games.js webui/src/stores/games.test.js
git commit -m "feat(stores): add navigation, league stores and selectedOpponent"
```

---

## Task 6: WebSocket handler — `league_update` and extended `init`

**Files:**
- Modify: `webui/src/lib/ws.js`
- Modify: `webui/src/lib/ws.test.js`

- [ ] **Step 1: Write failing tests**

Add to `webui/src/lib/ws.test.js`:

```javascript
import { leagueEntries, leagueResults, eloHistory } from '../stores/league.js'

beforeEach(() => {
  // ... add to existing beforeEach:
  leagueEntries.set([])
  leagueResults.set([])
  eloHistory.set([])
})

describe('handleMessage — init with league data', () => {
  it('populates league stores from init message', () => {
    handleMessage({
      type: 'init',
      games: [],
      metrics: [],
      training_state: null,
      league_entries: [{ id: 1, architecture: 'resnet', elo_rating: 1050 }],
      league_results: [{ id: 1, epoch: 5, wins: 3, losses: 1, draws: 1 }],
      elo_history: [{ entry_id: 1, epoch: 5, elo_rating: 1050 }],
    })
    expect(get(leagueEntries)).toHaveLength(1)
    expect(get(leagueResults)).toHaveLength(1)
    expect(get(eloHistory)).toHaveLength(1)
  })

  it('handles init with no league fields gracefully', () => {
    handleMessage({ type: 'init', games: [], metrics: [], training_state: null })
    expect(get(leagueEntries)).toEqual([])
    expect(get(leagueResults)).toEqual([])
    expect(get(eloHistory)).toEqual([])
  })
})

describe('handleMessage — league_update', () => {
  it('replaces all league stores', () => {
    leagueEntries.set([{ id: 99 }])
    handleMessage({
      type: 'league_update',
      entries: [{ id: 1 }, { id: 2 }],
      results: [{ id: 10 }],
      elo_history: [{ entry_id: 1, epoch: 5, elo_rating: 1050 }],
    })
    expect(get(leagueEntries)).toHaveLength(2)
    expect(get(leagueResults)).toHaveLength(1)
    expect(get(eloHistory)).toHaveLength(1)
  })

  it('handles missing fields gracefully', () => {
    handleMessage({ type: 'league_update' })
    expect(get(leagueEntries)).toEqual([])
    expect(get(leagueResults)).toEqual([])
    expect(get(eloHistory)).toEqual([])
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd webui && npx vitest run src/lib/ws.test.js`
Expected: FAIL — league stores not populated

- [ ] **Step 3: Implement ws.js changes**

Add imports at top of `webui/src/lib/ws.js`:

```javascript
import { leagueEntries, leagueResults, eloHistory } from '../stores/league.js'
```

Extend the `init` case in `handleMessage`:

```javascript
case 'init':
  games.set(msg.games || [])
  metrics.set(msg.metrics || [])
  trainingState.set(msg.training_state || null)
  leagueEntries.set(msg.league_entries || [])
  leagueResults.set(msg.league_results || [])
  eloHistory.set(msg.elo_history || [])
  if (msg.games?.length > 0) {
    selectedGameId.update(id => id ?? 0)
  }
  break
```

Add new case after `training_status`:

```javascript
case 'league_update':
  leagueEntries.set(msg.entries || [])
  leagueResults.set(msg.results || [])
  eloHistory.set(msg.elo_history || [])
  break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd webui && npx vitest run src/lib/ws.test.js`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add webui/src/lib/ws.js webui/src/lib/ws.test.js
git commit -m "feat(ws): handle league_update and extend init with league data"
```

---

## Task 7: Elo chart data helper

**Files:**
- Create: `webui/src/lib/eloChartData.js`
- Create: `webui/src/lib/eloChartData.test.js`

- [ ] **Step 1: Write failing tests**

Create `webui/src/lib/eloChartData.test.js`:

```javascript
import { describe, it, expect } from 'vitest'
import { buildEloChartData } from './eloChartData.js'

describe('buildEloChartData', () => {
  it('returns empty structure for empty input', () => {
    const result = buildEloChartData([], [])
    expect(result.xData).toEqual([])
    expect(result.series).toEqual([])
  })

  it('groups single entry into one series', () => {
    const history = [
      { entry_id: 1, epoch: 5, elo_rating: 1000 },
      { entry_id: 1, epoch: 10, elo_rating: 1050 },
    ]
    const entries = [{ id: 1, architecture: 'resnet', elo_rating: 1050 }]
    const result = buildEloChartData(history, entries)
    expect(result.xData).toEqual([5, 10])
    expect(result.series).toHaveLength(1)
    expect(result.series[0].label).toBe('resnet (1050)')
    expect(result.series[0].data).toEqual([1000, 1050])
  })

  it('groups multiple entries with shared epoch axis', () => {
    const history = [
      { entry_id: 1, epoch: 5, elo_rating: 1000 },
      { entry_id: 2, epoch: 5, elo_rating: 900 },
      { entry_id: 1, epoch: 10, elo_rating: 1050 },
      { entry_id: 2, epoch: 10, elo_rating: 950 },
    ]
    const entries = [
      { id: 1, architecture: 'resnet', elo_rating: 1050 },
      { id: 2, architecture: 'transformer', elo_rating: 950 },
    ]
    const result = buildEloChartData(history, entries)
    expect(result.xData).toEqual([5, 10])
    expect(result.series).toHaveLength(2)
    expect(result.series[0].data).toEqual([1000, 1050])
    expect(result.series[1].data).toEqual([900, 950])
  })

  it('fills null for epochs where an entry has no data', () => {
    const history = [
      { entry_id: 1, epoch: 5, elo_rating: 1000 },
      { entry_id: 1, epoch: 10, elo_rating: 1050 },
      { entry_id: 2, epoch: 10, elo_rating: 900 },
    ]
    const entries = [
      { id: 1, architecture: 'a', elo_rating: 1050 },
      { id: 2, architecture: 'b', elo_rating: 900 },
    ]
    const result = buildEloChartData(history, entries)
    expect(result.xData).toEqual([5, 10])
    expect(result.series[0].data).toEqual([1000, 1050]) // entry 1: present at both
    expect(result.series[1].data).toEqual([null, 900])   // entry 2: missing at epoch 5
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd webui && npx vitest run src/lib/eloChartData.test.js`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `eloChartData.js`**

Create `webui/src/lib/eloChartData.js`:

```javascript
const COLORS = ['#4ade80', '#60a5fa', '#f59e0b', '#a78bfa', '#f472b6']

/**
 * Transform flat elo_history array into grouped chart data for MetricsChart.
 * @param {Array<{entry_id: number, epoch: number, elo_rating: number}>} history
 * @param {Array<{id: number, architecture: string, elo_rating: number}>} entries
 * @returns {{ xData: number[], series: Array<{label: string, data: number[], color: string}> }}
 */
export function buildEloChartData(history, entries) {
  if (history.length === 0) return { xData: [], series: [] }

  // Collect unique epochs (sorted)
  const epochSet = new Set(history.map(h => h.epoch))
  const xData = [...epochSet].sort((a, b) => a - b)
  const epochIndex = new Map(xData.map((e, i) => [e, i]))

  // Collect unique entry_ids in the order they appear in entries
  const entryIds = entries.map(e => e.id)
  const entryMap = new Map(entries.map(e => [e.id, e]))

  const series = entryIds
    .filter(id => history.some(h => h.entry_id === id))
    .map((id, i) => {
      const entry = entryMap.get(id)
      const data = new Array(xData.length).fill(null)
      for (const h of history) {
        if (h.entry_id === id) {
          data[epochIndex.get(h.epoch)] = h.elo_rating
        }
      }
      return {
        label: `${entry.architecture} (${Math.round(entry.elo_rating)})`,
        data,
        color: COLORS[i % COLORS.length],
      }
    })

  return { xData, series }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd webui && npx vitest run src/lib/eloChartData.test.js`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add webui/src/lib/eloChartData.js webui/src/lib/eloChartData.test.js
git commit -m "feat: add eloChartData helper for Elo trend chart"
```

---

## Task 8: CSS variables and TabBar component

**Files:**
- Modify: `webui/src/app.css`
- Create: `webui/src/lib/TabBar.svelte`

- [ ] **Step 1: Add CSS variables for tabs and player cards**

Add to `webui/src/app.css` inside `:root`:

```css
--tab-active-bg: #1a3a2a;
--tab-active-border: var(--accent-green);
--tab-inactive-border: var(--border);
--player-learner: var(--accent-green);
--player-opponent: var(--accent-blue);
```

- [ ] **Step 2: Create `TabBar.svelte`**

Create `webui/src/lib/TabBar.svelte`:

```svelte
<script>
  import { activeTab } from '../stores/navigation.js'

  const tabs = [
    { id: 'training', label: 'Training' },
    { id: 'league', label: 'League' },
  ]
</script>

<div class="tab-bar" role="tablist" aria-label="Dashboard views">
  {#each tabs as tab}
    <button
      role="tab"
      aria-selected={$activeTab === tab.id}
      class:active={$activeTab === tab.id}
      on:click={() => activeTab.set(tab.id)}
    >
      {tab.label}
    </button>
  {/each}
</div>

<style>
  .tab-bar {
    display: flex;
    gap: 4px;
  }

  button {
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    border: 1px solid var(--tab-inactive-border);
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }

  button:hover {
    border-color: var(--text-secondary);
  }

  button:focus-visible {
    outline: 2px solid var(--accent-blue);
    outline-offset: 2px;
  }

  button.active {
    border-color: var(--tab-active-border);
    color: var(--tab-active-border);
    background: var(--tab-active-bg);
  }

  @media (prefers-reduced-motion: reduce) {
    button { transition: none; }
  }
</style>
```

- [ ] **Step 3: Integrate TabBar into StatusIndicator**

In `webui/src/lib/StatusIndicator.svelte`, replace the `.right` div content. Import and render TabBar:

```svelte
<script>
  // ... existing imports ...
  import TabBar from './TabBar.svelte'
</script>
```

Replace the `.right` div:

```svelte
<div class="right">
  <TabBar />
</div>
```

Remove the `☗ {displayName}` player name — it will move to `PlayerCard` in Task 9.

- [ ] **Step 4: Verify manually — run dev server**

Run: `cd webui && npm run dev`
Open browser, verify tab bar renders in status header. Clicking tabs should toggle (no view routing yet — that comes in Task 10).

- [ ] **Step 5: Commit**

```bash
git add webui/src/app.css webui/src/lib/TabBar.svelte webui/src/lib/StatusIndicator.svelte
git commit -m "feat: add TabBar component and integrate into status header"
```

---

## Task 9: PlayerCard component

**Files:**
- Create: `webui/src/lib/PlayerCard.svelte`

- [ ] **Step 1: Create `PlayerCard.svelte`**

```svelte
<script>
  /** @type {'learner' | 'opponent'} */
  export let role = 'learner'
  /** @type {string} */
  export let name = ''
  /** @type {number | null} */
  export let elo = null
  /** @type {string} */
  export let detail = ''

  $: icon = role === 'learner' ? '☗' : '☖'
  $: roleLabel = role === 'learner' ? 'Learner' : 'Opponent'
  $: colorClass = role === 'learner' ? 'learner' : 'opponent'
</script>

<div
  class="player-card {colorClass}"
  aria-label="{roleLabel}: {name}{elo != null ? ', Elo ' + Math.round(elo) : ''}"
>
  <div class="header">
    <span class="role">{icon} {roleLabel}</span>
    {#if elo != null}
      <span class="elo-badge">{Math.round(elo)}</span>
    {/if}
  </div>
  <div class="name">{name || '—'}</div>
  {#if detail}
    <div class="detail">{detail}</div>
  {/if}
</div>

<style>
  .player-card {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
    background: var(--bg-secondary);
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .role {
    font-weight: 700;
    font-size: 12px;
  }

  .player-card.learner .role { color: var(--player-learner); }
  .player-card.opponent .role { color: var(--player-opponent); }

  .elo-badge {
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    font-family: monospace;
  }

  .player-card.learner .elo-badge {
    background: #1a3a2a;
    color: var(--player-learner);
  }

  .player-card.opponent .elo-badge {
    background: #1a1a2e;
    color: var(--player-opponent);
  }

  .name {
    font-size: 12px;
    color: var(--text-primary);
    margin-top: 6px;
  }

  .detail {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
  }
</style>
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/PlayerCard.svelte
git commit -m "feat: add PlayerCard component for learner/opponent identity"
```

---

## Task 10: MetricsChart — add `compact` prop

**Files:**
- Modify: `webui/src/lib/MetricsChart.svelte`
- Modify: `webui/src/lib/chartHelpers.js`

- [ ] **Step 1: Add `compact` prop to `MetricsChart.svelte`**

Add to the `<script>` section:

```javascript
export let compact = false
```

Conditionally suppress legend and annotation:

```svelte
{#if annotation && !compact}
  <div class="annotation">{annotation}</div>
{/if}
```

Update `getOpts()` to suppress legend in compact mode:

```javascript
function getOpts() {
  const w = container ? container.clientWidth : width
  return buildChartOpts({ width: w, height, series, compact })
}
```

- [ ] **Step 2: Update `buildChartOpts` to accept `compact`**

In `webui/src/lib/chartHelpers.js`, add `compact = false` parameter:

```javascript
export function buildChartOpts({ width, height, series, compact = false }) {
  return {
    width,
    height,
    padding: compact ? [4, 4, 0, 0] : [8, 8, 0, 0],
    cursor: { show: !compact },
    legend: { show: !compact },
    scales: { x: { time: false } },
    axes: [
      {
        show: !compact,
        stroke: DARK_THEME.textColor,
        grid: { stroke: DARK_THEME.gridColor, width: 0.5 },
        ticks: { stroke: DARK_THEME.axisColor },
        font: '12px sans-serif',
        values: (u, vals) => vals.map(v => Number.isInteger(v) ? v : ''),
        incrs: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
      },
      {
        show: !compact,
        stroke: DARK_THEME.textColor,
        grid: { stroke: DARK_THEME.gridColor, width: 0.5 },
        ticks: { stroke: DARK_THEME.axisColor },
        font: '12px sans-serif',
      },
    ],
    series: [
      { label: 'X' },
      ...series.map(s => ({
        label: s.label,
        stroke: s.color,
        width: compact ? 1 : 1.5,
        fill: s.color + '20',
      })),
    ],
  }
}
```

- [ ] **Step 3: Run existing chart tests to verify no regressions**

Run: `cd webui && npx vitest run src/lib/chartHelpers.test.js`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add webui/src/lib/MetricsChart.svelte webui/src/lib/chartHelpers.js
git commit -m "feat(chart): add compact prop for mini sparkline mode"
```

---

## Task 11: MetricsGrid — mini sparklines with click-to-expand

**Files:**
- Modify: `webui/src/lib/MetricsGrid.svelte`

- [ ] **Step 1: Rewrite MetricsGrid**

Replace `webui/src/lib/MetricsGrid.svelte`:

```svelte
<script>
  import { metrics } from '../stores/metrics.js'
  import MetricsChart from './MetricsChart.svelte'
  import { extractColumns } from './metricsColumns.js'

  $: columns = extractColumns($metrics)

  let expandedIndex = null

  const charts = [
    { title: 'Policy & Value Loss', xKey: 'steps', series: (c) => [
      { label: 'Policy', data: c.policyLoss, color: '#f59e0b' },
      { label: 'Value', data: c.valueLoss, color: '#60a5fa' },
    ]},
    { title: 'Win Rate', xKey: 'epochs', series: (c) => [
      { label: '☗ Black', data: c.blackWinRate, color: '#e0e0e0' },
      { label: '☖ White', data: c.whiteWinRate, color: '#60a5fa' },
      { label: 'Draw', data: c.drawRate, color: '#f59e0b' },
    ]},
    { title: 'Avg Episode Length', xKey: 'epochs', series: (c) => [
      { label: 'Episode Length', data: c.avgEpLen, color: '#a78bfa' },
    ], annotation: 'Longer games = more strategic play' },
    { title: 'Policy Entropy', xKey: 'steps', series: (c) => [
      { label: 'Entropy', data: c.entropy, color: '#f472b6' },
    ], annotation: 'Falling entropy = agent becoming more decisive' },
  ]

  function handleClick(index) {
    expandedIndex = expandedIndex === index ? null : index
  }
</script>

<div class="metrics-grid">
  <h2 class="grid-header">
    Training Metrics {#if $metrics.length > 0}— Epoch {$metrics[$metrics.length - 1]?.epoch ?? '?'}{/if}
  </h2>

  {#if expandedIndex != null}
    {@const chart = charts[expandedIndex]}
    <div class="expanded-chart">
      <button class="collapse-btn" on:click={() => expandedIndex = null} aria-label="Collapse chart">✕</button>
      <MetricsChart
        title={chart.title}
        xData={columns[chart.xKey]}
        series={chart.series(columns)}
        height={280}
        annotation={chart.annotation || null}
        compact={false}
      />
    </div>
  {/if}

  <div class="grid">
    {#each charts as chart, i}
      <button
        class="mini-chart-btn"
        class:active={expandedIndex === i}
        on:click={() => handleClick(i)}
        aria-expanded={expandedIndex === i}
        aria-label="{chart.title} — click to {expandedIndex === i ? 'collapse' : 'expand'}"
      >
        <MetricsChart
          title={chart.title}
          xData={columns[chart.xKey]}
          series={chart.series(columns)}
          height={100}
          compact={true}
        />
      </button>
    {/each}
  </div>
</div>

<style>
  .metrics-grid {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 12px;
  }

  h2.grid-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 8px;
  }

  .mini-chart-btn {
    all: unset;
    cursor: pointer;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    transition: border-color 0.15s;
  }

  .mini-chart-btn:hover {
    border-color: var(--text-secondary);
  }

  .mini-chart-btn:focus-visible {
    outline: 2px solid var(--accent-blue);
    outline-offset: 2px;
  }

  .mini-chart-btn.active {
    border-color: var(--accent-green);
  }

  .expanded-chart {
    margin-bottom: 10px;
    position: relative;
  }

  .collapse-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    z-index: 1;
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 2px 6px;
    font-size: 12px;
  }

  .collapse-btn:hover {
    color: var(--text-primary);
    border-color: var(--text-secondary);
  }

  @media (max-width: 768px) {
    .grid {
      grid-template-columns: 1fr 1fr;
    }
  }

  @media (max-width: 480px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .mini-chart-btn { transition: none; }
  }
</style>
```

- [ ] **Step 2: Verify manually — run dev server**

Run: `cd webui && npm run dev`
Expected: 4 mini charts in a row, clicking expands one above the row, clicking again collapses.

- [ ] **Step 3: Commit**

```bash
git add webui/src/lib/MetricsGrid.svelte
git commit -m "feat: mini sparkline metrics with click-to-expand"
```

---

## Task 12: League view components — LeagueTable, MatchHistory, LeagueView

**Files:**
- Create: `webui/src/lib/MatchHistory.svelte`
- Create: `webui/src/lib/LeagueTable.svelte`
- Create: `webui/src/lib/LeagueView.svelte`

- [ ] **Step 1: Create `MatchHistory.svelte`**

```svelte
<script>
  import { leagueResults, leagueEntries } from '../stores/league.js'

  /** @type {number} */
  export let entryId

  $: matches = $leagueResults.filter(
    r => r.learner_id === entryId || r.opponent_id === entryId
  ).sort((a, b) => b.epoch - a.epoch)

  function opponentName(result) {
    const oppId = result.learner_id === entryId ? result.opponent_id : result.learner_id
    const entry = $leagueEntries.find(e => e.id === oppId)
    return entry ? entry.architecture : `#${oppId}`
  }
</script>

<div class="match-history">
  {#if matches.length === 0}
    <p class="empty">No matches recorded</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>Epoch</th>
          <th>Opponent</th>
          <th class="num">W</th>
          <th class="num">L</th>
          <th class="num">D</th>
        </tr>
      </thead>
      <tbody>
        {#each matches as m}
          <tr>
            <td>{m.epoch}</td>
            <td>{opponentName(m)}</td>
            <td class="num win">{m.wins}</td>
            <td class="num loss">{m.losses}</td>
            <td class="num draw">{m.draws}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .match-history {
    max-height: 200px;
    overflow-y: auto;
    padding: 8px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    margin: 4px 0;
  }

  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  thead { color: var(--text-muted); }
  th, td { text-align: left; padding: 3px 8px; }
  th.num, td.num { text-align: right; width: 40px; font-family: monospace; }
  .win { color: var(--accent-green); }
  .loss { color: var(--danger); }
  .draw { color: var(--accent-amber); }
  .empty { color: var(--text-muted); font-size: 12px; text-align: center; padding: 12px; }
</style>
```

- [ ] **Step 2: Create `LeagueTable.svelte`**

```svelte
<script>
  import { leagueRanked } from '../stores/league.js'
  import MatchHistory from './MatchHistory.svelte'

  let sortColumn = 'elo_rating'
  let sortAsc = false
  let expandedId = null

  $: sorted = (() => {
    const entries = [...$leagueRanked]
    entries.sort((a, b) => {
      const av = a[sortColumn], bv = b[sortColumn]
      return sortAsc ? (av > bv ? 1 : -1) : (bv > av ? 1 : -1)
    })
    return entries.map((e, i) => ({ ...e, rank: i + 1 }))
  })()

  function toggleSort(col) {
    if (sortColumn === col) {
      sortAsc = !sortAsc
    } else {
      sortColumn = col
      sortAsc = false
    }
  }

  function toggleExpand(id) {
    expandedId = expandedId === id ? null : id
  }

  function sortIndicator(col) {
    if (sortColumn !== col) return ''
    return sortAsc ? ' ▲' : ' ▼'
  }
</script>

<div class="league-table">
  <h2 class="section-header">Elo Leaderboard</h2>
  {#if sorted.length === 0}
    <p class="empty">No league entries yet. League data appears once opponent pool training begins.</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th class="num">#</th>
          <th><button class="sort-btn" on:click={() => toggleSort('architecture')}>Model{sortIndicator('architecture')}</button></th>
          <th class="num"><button class="sort-btn" on:click={() => toggleSort('elo_rating')}>Elo{sortIndicator('elo_rating')}</button></th>
          <th class="num"><button class="sort-btn" on:click={() => toggleSort('games_played')}>Games{sortIndicator('games_played')}</button></th>
          <th class="num"><button class="sort-btn" on:click={() => toggleSort('created_epoch')}>Epoch{sortIndicator('created_epoch')}</button></th>
        </tr>
      </thead>
      <tbody>
        {#each sorted as entry}
          <tr
            class:top={entry.rank === 1}
            class:expanded={expandedId === entry.id}
            on:click={() => toggleExpand(entry.id)}
            aria-expanded={expandedId === entry.id}
            tabindex="0"
            on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggleExpand(entry.id) }}}
          >
            <td class="num rank">{entry.rank}</td>
            <td>{entry.architecture}</td>
            <td class="num elo">{Math.round(entry.elo_rating)}</td>
            <td class="num">{entry.games_played}</td>
            <td class="num">{entry.created_epoch}</td>
          </tr>
          {#if expandedId === entry.id}
            <tr class="history-row">
              <td colspan="5">
                <MatchHistory entryId={entry.id} />
              </td>
            </tr>
          {/if}
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .league-table { padding: 12px; }

  .section-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead { color: var(--text-muted); font-size: 12px; }
  th, td { text-align: left; padding: 6px 10px; }
  th.num, td.num { text-align: right; }

  .sort-btn {
    all: unset;
    cursor: pointer;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
  }
  .sort-btn:hover { color: var(--text-primary); }

  tbody tr {
    border-bottom: 1px solid var(--border-subtle);
    cursor: pointer;
    color: var(--text-primary);
    transition: background 0.1s;
  }
  tbody tr:hover { background: var(--bg-secondary); }
  tbody tr:focus-visible { outline: 2px solid var(--accent-blue); outline-offset: -2px; }

  tr.top .rank { color: var(--accent-amber); font-weight: 700; }
  tr.top .elo { color: var(--accent-green); font-weight: 700; }
  tr.expanded { background: var(--bg-secondary); }

  .history-row { cursor: default; }
  .history-row:hover { background: transparent; }
  .history-row td { padding: 0; }

  .elo { font-family: monospace; }
  .empty { color: var(--text-muted); font-size: 13px; padding: 24px; text-align: center; }

  @media (prefers-reduced-motion: reduce) {
    tbody tr { transition: none; }
  }
</style>
```

- [ ] **Step 3: Create `LeagueView.svelte`**

```svelte
<script>
  import { eloHistory, leagueEntries } from '../stores/league.js'
  import { buildEloChartData } from './eloChartData.js'
  import LeagueTable from './LeagueTable.svelte'
  import MetricsChart from './MetricsChart.svelte'

  $: chartData = buildEloChartData($eloHistory, $leagueEntries)
</script>

<div class="league-view">
  <LeagueTable />

  <div class="elo-chart-section">
    <h2 class="section-header">Elo Over Time</h2>
    {#if chartData.xData.length > 0}
      <MetricsChart
        title=""
        xData={chartData.xData}
        series={chartData.series}
        height={250}
      />
    {:else}
      <p class="empty">Elo history will appear after league matches are played.</p>
    {/if}
  </div>
</div>

<style>
  .league-view {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .elo-chart-section {
    padding: 0 12px 12px;
  }

  .section-header {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
  }

  .empty {
    color: var(--text-muted);
    font-size: 13px;
    text-align: center;
    padding: 24px;
  }
</style>
```

- [ ] **Step 4: Commit**

```bash
git add webui/src/lib/MatchHistory.svelte webui/src/lib/LeagueTable.svelte webui/src/lib/LeagueView.svelte
git commit -m "feat: add LeagueView, LeagueTable, and MatchHistory components"
```

---

## Task 13: App.svelte — Revised layout with tab routing and player cards

**Files:**
- Modify: `webui/src/App.svelte`

- [ ] **Step 1: Rewrite App.svelte**

This is the main integration point. Replace the entire file:

```svelte
<script>
  import { onMount } from 'svelte'
  import { connect, disconnect } from './lib/ws.js'
  import { games, selectedGame, selectedOpponent } from './stores/games.js'
  import { activeTab } from './stores/navigation.js'
  import { trainingState } from './stores/training.js'
  import { leagueEntries } from './stores/league.js'
  import StatusIndicator from './lib/StatusIndicator.svelte'
  import GameThumbnail from './lib/GameThumbnail.svelte'
  import Board from './lib/Board.svelte'
  import PieceTray from './lib/PieceTray.svelte'
  import MoveLog from './lib/MoveLog.svelte'
  import EvalBar from './lib/EvalBar.svelte'
  import MetricsGrid from './lib/MetricsGrid.svelte'
  import PlayerCard from './lib/PlayerCard.svelte'
  import LeagueView from './lib/LeagueView.svelte'
  import { safeParse } from './lib/safeParse.js'

  onMount(() => {
    connect()
    return disconnect
  })

  $: game = $selectedGame
  $: board = game ? safeParse(game.board_json, game.board || []) : []
  $: hands = game ? safeParse(game.hands_json, game.hands || {}) : {}
  $: moveHistory = game?.move_history_json || '[]'

  let boardAreaHeight = 0

  $: lastMoveIdx = (() => {
    try {
      const history = safeParse(moveHistory, [])
      if (history.length === 0) return -1
      return -1
    } catch { return -1 }
  })()

  // Learner info from training state
  $: learnerName = $trainingState?.display_name || $trainingState?.model_arch || 'Learner'
  $: learnerElo = (() => {
    if (!$leagueEntries.length) return null
    const epoch = $trainingState?.current_epoch ?? -1
    const match = [...$leagueEntries]
      .filter(e => e.created_epoch <= epoch)
      .sort((a, b) => b.created_epoch - a.created_epoch)[0]
    return match?.elo_rating ?? null
  })()
  $: learnerDetail = $trainingState
    ? `${$trainingState.model_arch || ''} · Epoch ${$trainingState.current_epoch || 0} · ${($trainingState.current_step || 0).toLocaleString()} steps`
    : ''

  // Opponent info from selected game
  $: opp = $selectedOpponent
  $: opponentName = opp ? opp.architecture : 'Self-play'
  $: opponentElo = opp?.elo_rating ?? null
  $: opponentDetail = opp ? `${opp.games_played} games played` : ''
</script>

<div class="app">
  <a href="#game-panel" class="skip-nav">Skip to game</a>
  <StatusIndicator />

  {#if $activeTab === 'training'}
    <div class="main-content">
      <aside class="thumbnail-panel" aria-label="Game list">
        <h2 class="section-label">Games ({$games.length})</h2>
        <div class="thumb-grid">
          {#each $games.slice(0, 16) as g (g.game_id)}
            <GameThumbnail game={g} />
          {/each}
        </div>
      </aside>

      <div class="player-panel">
        <PlayerCard role="learner" name={learnerName} elo={learnerElo} detail={learnerDetail} />
        <div class="vs-separator">VS</div>
        <PlayerCard role="opponent" name={opponentName} elo={opponentElo} detail={opponentDetail} />
      </div>

      <main id="game-panel" class="game-panel" aria-label="Game viewer">
        {#if game}
          <div class="game-view">
            <div class="board-area" bind:clientHeight={boardAreaHeight}>
              <PieceTray color="white" hand={hands.white || {}} />
              <Board
                board={board}
                inCheck={!!game.in_check}
                currentPlayer={game.current_player || 'black'}
                lastMoveIdx={lastMoveIdx}
              />
              <PieceTray color="black" hand={hands.black || {}} />
            </div>

            <div class="eval-area" style="height: {boardAreaHeight}px">
              <EvalBar
                value={game.value_estimate || 0}
                currentPlayer={game.current_player || 'black'}
              />
            </div>

            <div class="info-area" style="height: {boardAreaHeight}px">
              <div class="game-info">
                <div class="info-row">
                  <span class="label">Game {(game.game_id || 0) + 1}</span>
                  <span class="value">{game.current_player || 'black'} to move</span>
                </div>
                <div class="info-row">
                  <span class="label">Ply</span>
                  <span class="value">{game.ply || 0}</span>
                </div>
                <div class="info-row">
                  <span class="label">Result</span>
                  <span class="value result"
                    class:in-progress={game.result === 'in_progress'}
                    class:terminal={game.result !== 'in_progress'}
                  >
                    {#if game.result === 'in_progress'}In progress{:else}&#10003; {(game.result || '').replaceAll('_', ' ')}{/if}
                  </span>
                </div>
              </div>

              <MoveLog
                moveHistoryJson={moveHistory}
                currentPlayer={game.current_player || 'black'}
              />
            </div>
          </div>
        {:else}
          <div class="no-game">
            <p>Waiting for game data&hellip;</p>
            <p class="no-game-hint">Connect a training session to see live games.</p>
          </div>
        {/if}
      </main>
    </div>

    <section class="metrics-panel" aria-label="Training metrics">
      <MetricsGrid />
    </section>
  {:else}
    <LeagueView />
  {/if}
</div>

<style>
  .skip-nav {
    position: absolute;
    left: -9999px;
    top: 0;
    z-index: 100;
    padding: 8px 16px;
    background: var(--accent-blue);
    color: #fff;
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    border-radius: 0 0 4px 0;
  }

  .skip-nav:focus { left: 0; }

  .app {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    background: var(--bg-primary);
  }

  .main-content {
    display: flex;
    flex: 0 0 auto;
    gap: 0;
    border-bottom: 1px solid var(--border);
    align-items: stretch;
  }

  .thumbnail-panel {
    flex: 0 0 auto;
    border-right: 1px solid var(--border);
    padding: 8px;
    overflow: hidden;
  }

  .section-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }

  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 6px;
  }

  .player-panel {
    flex: 0 0 auto;
    width: 170px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
    border-right: 1px solid var(--border);
  }

  .vs-separator {
    text-align: center;
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
  }

  .game-panel {
    flex: 1 1 auto;
    padding: 16px;
    overflow-y: auto;
  }

  .game-view {
    display: flex;
    align-items: stretch;
    gap: 16px;
  }

  .board-area {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    justify-content: center;
  }

  .eval-area {
    display: flex;
    flex-shrink: 0;
  }

  .info-area {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
    width: 40ch;
    overflow: hidden;
  }

  .game-info {
    background: var(--bg-primary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
    font-size: 12px;
  }

  .info-row .label { color: var(--text-secondary); }
  .info-row .value { color: var(--text-primary); }

  .result.in-progress { color: var(--accent-amber); }
  .result.terminal { color: var(--accent-green); }

  .no-game {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    color: var(--text-muted);
    gap: 8px;
  }

  .no-game-hint { font-size: 12px; color: var(--text-muted); }

  .metrics-panel { padding: 12px 16px; }

  @media (max-width: 768px) {
    .main-content { flex-direction: column; }

    .thumbnail-panel {
      width: 100%;
      border-right: none;
      border-bottom: 1px solid var(--border);
      max-height: 160px;
    }

    .thumb-grid {
      grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    }

    .player-panel {
      width: 100%;
      flex-direction: row;
      border-right: none;
      border-bottom: 1px solid var(--border);
      justify-content: center;
    }

    .vs-separator { writing-mode: horizontal-tb; }

    .game-view { flex-direction: column; }
    .board-area { align-self: center; }
    .info-area { min-width: unset; }
  }

  @media (max-width: 480px) {
    .game-panel { padding: 8px; }
    .metrics-panel { padding: 8px; }
  }
</style>
```

- [ ] **Step 2: Verify manually — run dev server**

Run: `cd webui && npm run dev`
Expected:
- Training tab shows 3-column layout: 4×4 thumbnails | player cards (VS) | board + info
- League tab shows leaderboard + Elo chart
- Tab switching works
- Mini charts expand/collapse
- Player cards show learner info, "Self-play" for opponent when no league match

- [ ] **Step 3: Commit**

```bash
git add webui/src/App.svelte
git commit -m "feat: revised App layout with tab routing, player cards, and league view"
```

---

## Task 14: Add `aria-live` to status bar

**Files:**
- Modify: `webui/src/lib/StatusIndicator.svelte`

- [ ] **Step 1: Add `aria-live="polite"` to stats div**

In `StatusIndicator.svelte`, add `aria-live="polite"` to the `.stats` div:

```svelte
<div class="stats" aria-live="polite">
```

- [ ] **Step 2: Commit**

```bash
git add webui/src/lib/StatusIndicator.svelte
git commit -m "a11y: add aria-live to training status updates"
```

---

## Task 15: Run full test suites

**Files:** None (verification only)

- [ ] **Step 1: Run all backend tests**

Run: `uv run pytest tests/ -v --timeout=30`
Expected: All PASS

- [ ] **Step 2: Run all frontend tests**

Run: `cd webui && npx vitest run`
Expected: All PASS

- [ ] **Step 3: Run frontend build**

Run: `cd webui && npm run build`
Expected: Build succeeds with no errors

- [ ] **Step 4: If any failures, fix and recommit**

Address any failures found in steps 1-3 before proceeding.

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: resolve test/build issues from tournament UI integration"
```
