# Tiered Opponent Pool Phase 2: Historical Library & Role-Specific Elo

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5-slot Historical Library with log-spaced milestone selection and periodic gauntlet benchmarks, plus four role-specific Elo ratings that track performance per opponent tier.

**Architecture:** Three new components plugging into the existing Phase 1 infrastructure: HistoricalLibrary (milestone selection), HistoricalGauntlet (benchmark runner integrated into tournament thread), RoleEloTracker (per-context Elo updates). All additive — no existing Phase 1 interfaces change.

**Tech Stack:** Python 3.13, SQLite (WAL mode), PyTorch, frozen dataclasses, uv for deps.

**Spec:** `docs/superpowers/specs/2026-04-05-tiered-opponent-pool-phase2-design.md`

**Prerequisite:** Phase 1 must be complete. This plan assumes the following exist:
- `keisei/training/opponent_store.py` — `OpponentStore` with `transaction()`, `RLock`, `_in_transaction` commit discipline
- `keisei/training/tiered_pool.py` — `TieredPool` orchestrator with `on_epoch_end(epoch)`
- `keisei/training/match_scheduler.py` — `MatchScheduler` with `generate_round(entries)`
- `keisei/training/tournament.py` — `LeagueTournament` using `OpponentStore` and `MatchScheduler`
- `keisei/config.py` — `LeagueConfig` with nested `FrontierStaticConfig`, `RecentFixedConfig`, `DynamicConfig`, `MatchSchedulerConfig`
- `keisei/db.py` — Schema v4 with `role`, `status`, `parent_entry_id`, `lineage_group`, `protection_remaining`, `last_match_at` on `league_entries`
- `keisei/training/opponent_store.py` — `Role`, `EntryStatus` StrEnums, `OpponentEntry` with defaults on new fields, `compute_elo_update` function

---

## Critical Implementation Notes

### Commit Discipline (inherited from Phase 1)

OpponentStore uses `threading.RLock()`. All mutating methods check `self._in_transaction` — if True (inside a `transaction()` block), they skip commit. If False (standalone call), they auto-commit. Phase 2 methods on OpponentStore must follow this same pattern. See the Phase 1 plan's Critical Implementation Notes for details.

### Phase 2 is Additive

Phase 2 does NOT modify any Phase 1 interfaces. It adds new tables, new columns, new classes, and new methods. The existing `elo_rating` column, `record_result`, `update_elo`, and all tier manager logic remain unchanged. Role-specific Elo runs in parallel with the composite rating.

### Gauntlet Threading

The gauntlet runs synchronously on the tournament thread after each round-robin round completes (when due). No new background thread is created. The tournament's `_stop_event` is checked between gauntlet matchups for graceful shutdown.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `keisei/training/historical_library.py` | Create | `HistoricalLibrary` class, `HistoricalSlot` dataclass |
| `keisei/training/historical_gauntlet.py` | Create | `HistoricalGauntlet` class — periodic benchmark runner |
| `keisei/training/role_elo.py` | Create | `RoleEloTracker` — per-context Elo updates |
| `keisei/config.py` | Modify | Add `HistoricalLibraryConfig`, `GauntletConfig`, `RoleEloConfig` to `LeagueConfig` |
| `keisei/db.py` | Modify | Schema v5: `historical_library` table, `gauntlet_results` table, 4 Elo columns on `league_entries`, update `read_league_data` |
| `keisei/training/opponent_store.py` | Modify | Add `update_role_elo`, `upsert_historical_slot`, `get_historical_slots`, `record_gauntlet_result` |
| `keisei/training/tiered_pool.py` | Modify | Wire HistoricalLibrary + gauntlet into `on_epoch_end`, add `get_historical_slots()` |
| `keisei/training/tournament.py` | Modify | Run gauntlet after round-robin when due |
| `keisei/training/katago_loop.py` | Modify | Pass history/gauntlet to tournament, log role Elo |
| `tests/test_historical_library.py` | Create | Tests for HistoricalLibrary |
| `tests/test_historical_gauntlet.py` | Create | Tests for HistoricalGauntlet |
| `tests/test_role_elo.py` | Create | Tests for RoleEloTracker |

---

### Task 1: Config Dataclasses

**Files:**
- Modify: `keisei/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for new config dataclasses**

In `tests/test_config.py`, add:

```python
from keisei.config import (
    HistoricalLibraryConfig,
    GauntletConfig,
    RoleEloConfig,
    LeagueConfig,
)


class TestPhase2Configs:
    def test_historical_library_defaults(self):
        cfg = HistoricalLibraryConfig()
        assert cfg.slots == 5
        assert cfg.refresh_interval_epochs == 100
        assert cfg.min_epoch_for_selection == 10

    def test_gauntlet_defaults(self):
        cfg = GauntletConfig()
        assert cfg.enabled is True
        assert cfg.interval_epochs == 100
        assert cfg.games_per_matchup == 16
        assert cfg.include_dynamic_topn == 0

    def test_role_elo_defaults(self):
        cfg = RoleEloConfig()
        assert cfg.frontier_k == 16.0
        assert cfg.dynamic_k == 24.0
        assert cfg.recent_k == 32.0
        assert cfg.historical_k == 12.0

    def test_league_config_has_phase2_nested(self):
        cfg = LeagueConfig()
        assert isinstance(cfg.history, HistoricalLibraryConfig)
        assert isinstance(cfg.gauntlet, GauntletConfig)
        assert isinstance(cfg.role_elo, RoleEloConfig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::TestPhase2Configs -v --no-header 2>&1 | tail -5`
Expected: ImportError — the new config classes don't exist yet.

- [ ] **Step 3: Implement the config dataclasses**

In `keisei/config.py`, add before `LeagueConfig`:

```python
@dataclass(frozen=True)
class HistoricalLibraryConfig:
    slots: int = 5
    refresh_interval_epochs: int = 100
    min_epoch_for_selection: int = 10


@dataclass(frozen=True)
class GauntletConfig:
    enabled: bool = True
    interval_epochs: int = 100
    games_per_matchup: int = 16
    include_dynamic_topn: int = 0


@dataclass(frozen=True)
class RoleEloConfig:
    frontier_k: float = 16.0
    dynamic_k: float = 24.0
    recent_k: float = 32.0
    historical_k: float = 12.0
```

Add to `LeagueConfig` fields:

```python
    # Phase 2: Historical Library & Role Elo
    history: HistoricalLibraryConfig = HistoricalLibraryConfig()
    gauntlet: GauntletConfig = GauntletConfig()
    role_elo: RoleEloConfig = RoleEloConfig()
```

Update `load_config` to parse nested `[league]` sub-tables for these:

```python
        history_raw = lg.pop("history", {})
        gauntlet_raw = lg.pop("gauntlet", {})
        role_elo_raw = lg.pop("role_elo", {})
        league_config = LeagueConfig(
            **lg,
            # ... existing Phase 1 nested configs ...
            history=HistoricalLibraryConfig(**history_raw),
            gauntlet=GauntletConfig(**gauntlet_raw),
            role_elo=RoleEloConfig(**role_elo_raw),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v --no-header 2>&1 | tail -10`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/config.py tests/test_config.py
git commit -m "feat(config): add Phase 2 config dataclasses (history, gauntlet, role_elo)"
```

---

### Task 2: Schema Update (db.py)

**Files:**
- Modify: `keisei/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write failing tests for new schema**

In `tests/test_db.py`, add:

```python
class TestSchemaV5:
    def test_historical_library_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # Insert a slot and read it back
        conn.execute(
            """INSERT INTO historical_library
               (slot_index, target_epoch, entry_id, actual_epoch, selected_at, selection_mode)
               VALUES (0, 100, NULL, NULL, '2026-01-01T00:00:00Z', 'log_spaced')"""
        )
        conn.commit()
        row = conn.execute("SELECT * FROM historical_library WHERE slot_index = 0").fetchone()
        conn.close()
        assert row is not None
        assert row["target_epoch"] == 100
        assert row["selection_mode"] == "log_spaced"

    def test_league_entries_has_role_elo_columns(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        conn.close()
        assert "elo_frontier" in cols
        assert "elo_dynamic" in cols
        assert "elo_recent" in cols
        assert "elo_historical" in cols

    def test_gauntlet_results_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "gauntlet_results" in tables

    def test_gauntlet_results_index_exists(self, tmp_path):
        db_path = str(tmp_path / "v5.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        indexes = [r[1] for r in conn.execute("PRAGMA index_list(gauntlet_results)").fetchall()]
        conn.close()
        assert "idx_gauntlet_epoch" in indexes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py::TestSchemaV5 -v --no-header 2>&1 | tail -5`
Expected: FAIL — tables/columns don't exist.

- [ ] **Step 3: Update init_db schema**

In `keisei/db.py`, change `SCHEMA_VERSION = 4` to `SCHEMA_VERSION = 5`.

Add new columns to the `CREATE TABLE IF NOT EXISTS league_entries` block:

```sql
                elo_frontier     REAL NOT NULL DEFAULT 1000.0,
                elo_dynamic      REAL NOT NULL DEFAULT 1000.0,
                elo_recent       REAL NOT NULL DEFAULT 1000.0,
                elo_historical   REAL NOT NULL DEFAULT 1000.0
```

Add new tables after the existing schema:

```sql
            CREATE TABLE IF NOT EXISTS historical_library (
                slot_index      INTEGER PRIMARY KEY,
                target_epoch    INTEGER NOT NULL,
                entry_id        INTEGER REFERENCES league_entries(id),
                actual_epoch    INTEGER,
                selected_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
                selection_mode  TEXT NOT NULL DEFAULT 'log_spaced'
            );
            CREATE TABLE IF NOT EXISTS gauntlet_results (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch               INTEGER NOT NULL,
                entry_id            INTEGER NOT NULL REFERENCES league_entries(id),
                historical_slot     INTEGER NOT NULL,
                historical_entry_id INTEGER NOT NULL REFERENCES league_entries(id),
                wins                INTEGER NOT NULL,
                losses              INTEGER NOT NULL,
                draws               INTEGER NOT NULL,
                elo_before          REAL,
                elo_after           REAL,
                created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE INDEX IF NOT EXISTS idx_gauntlet_epoch ON gauntlet_results(epoch);
```

Update `read_league_data` to include the four new Elo columns in the entries SELECT:

```python
        entries = conn.execute(
            "SELECT id, display_name, flavour_facts, model_params, architecture, "
            "elo_rating, games_played, created_epoch, created_at, "
            "role, status, parent_entry_id, lineage_group, protection_remaining, last_match_at, "
            "elo_frontier, elo_dynamic, elo_recent, elo_historical "
            "FROM league_entries WHERE status = 'active' ORDER BY elo_rating DESC"
        ).fetchall()
```

Add a new function to read historical library data:

```python
def read_historical_library(db_path: str) -> list[dict[str, Any]]:
    """Read historical library slots for dashboard."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT h.slot_index, h.target_epoch, h.actual_epoch, h.selection_mode, "
            "h.selected_at, h.entry_id, e.display_name, e.elo_rating, e.elo_historical, "
            "e.created_epoch "
            "FROM historical_library h "
            "LEFT JOIN league_entries e ON h.entry_id = e.id "
            "ORDER BY h.slot_index"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def read_gauntlet_results(db_path: str, limit: int = 50) -> list[dict[str, Any]]:
    """Read recent gauntlet results for dashboard."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            "SELECT g.*, e.display_name as entry_name, h.display_name as historical_name "
            "FROM gauntlet_results g "
            "LEFT JOIN league_entries e ON g.entry_id = e.id "
            "LEFT JOIN league_entries h ON g.historical_entry_id = h.id "
            "ORDER BY g.id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v --no-header 2>&1 | tail -10`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/db.py tests/test_db.py
git commit -m "feat(db): schema v5 with historical_library, gauntlet_results, role Elo columns"
```

---

### Task 3: OpponentStore Extensions

**Files:**
- Modify: `keisei/training/opponent_store.py`
- Test: `tests/test_opponent_store.py`

- [ ] **Step 1: Write failing tests for new store methods**

In `tests/test_opponent_store.py`, add:

```python
class TestRoleEloMethods:
    def test_update_role_elo(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        store.update_role_elo(entry.id, "elo_frontier", 1050.0)
        # Read back directly
        conn = sqlite3.connect(store_db)
        row = conn.execute(
            "SELECT elo_frontier FROM league_entries WHERE id = ?", (entry.id,)
        ).fetchone()
        conn.close()
        assert row[0] == 1050.0

    def test_update_role_elo_invalid_column_raises(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        with pytest.raises(ValueError, match="Invalid role Elo column"):
            store.update_role_elo(entry.id, "elo_nonexistent", 1050.0)


class TestHistoricalSlotMethods:
    def test_upsert_historical_slot(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RETIRED)
        store.upsert_historical_slot(
            slot_index=0, target_epoch=100, entry_id=entry.id,
            actual_epoch=100, selection_mode="log_spaced",
        )
        slots = store.get_historical_slots()
        assert len(slots) == 1
        assert slots[0]["slot_index"] == 0
        assert slots[0]["entry_id"] == entry.id

    def test_upsert_historical_slot_overwrites(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        e1 = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RETIRED)
        e2 = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RETIRED)
        store.upsert_historical_slot(0, 50, e1.id, 50, "fallback")
        store.upsert_historical_slot(0, 100, e2.id, 100, "log_spaced")
        slots = store.get_historical_slots()
        assert len(slots) == 1
        assert slots[0]["entry_id"] == e2.id
        assert slots[0]["selection_mode"] == "log_spaced"

    def test_get_historical_slots_empty(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        assert store.get_historical_slots() == []

    def test_upsert_historical_slot_null_entry(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        store.upsert_historical_slot(0, 100, None, None, "log_spaced")
        slots = store.get_historical_slots()
        assert len(slots) == 1
        assert slots[0]["entry_id"] is None


class TestGauntletResultMethods:
    def test_record_gauntlet_result(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=1, role=Role.RECENT_FIXED)
        historical = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RETIRED)
        store.record_gauntlet_result(
            epoch=100, entry_id=learner.id, historical_slot=0,
            historical_entry_id=historical.id,
            wins=10, losses=5, draws=1,
            elo_before=1000.0, elo_after=1012.0,
        )
        conn = sqlite3.connect(store_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM gauntlet_results").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["wins"] == 10
        assert rows[0]["elo_after"] == 1012.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_opponent_store.py::TestRoleEloMethods -v --no-header 2>&1 | tail -5`
Expected: AttributeError — methods don't exist.

- [ ] **Step 3: Implement new OpponentStore methods**

In `keisei/training/opponent_store.py`, add these methods to `OpponentStore`:

```python
    _VALID_ROLE_ELO_COLUMNS = frozenset({
        "elo_frontier", "elo_dynamic", "elo_recent", "elo_historical",
    })

    def update_role_elo(self, entry_id: int, role_column: str, new_elo: float) -> None:
        """Update a specific role Elo column."""
        if role_column not in self._VALID_ROLE_ELO_COLUMNS:
            raise ValueError(
                f"Invalid role Elo column: {role_column}. "
                f"Valid: {sorted(self._VALID_ROLE_ELO_COLUMNS)}"
            )
        with self._lock:
            self._conn.execute(
                f"UPDATE league_entries SET {role_column} = ? WHERE id = ?",
                (new_elo, entry_id),
            )
            if not self._in_transaction:
                self._conn.commit()

    def upsert_historical_slot(
        self,
        slot_index: int,
        target_epoch: int,
        entry_id: int | None,
        actual_epoch: int | None,
        selection_mode: str,
    ) -> None:
        """Insert or update a historical library slot."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO historical_library
                   (slot_index, target_epoch, entry_id, actual_epoch, selection_mode, selected_at)
                   VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                   ON CONFLICT(slot_index) DO UPDATE SET
                   target_epoch = excluded.target_epoch,
                   entry_id = excluded.entry_id,
                   actual_epoch = excluded.actual_epoch,
                   selection_mode = excluded.selection_mode,
                   selected_at = excluded.selected_at""",
                (slot_index, target_epoch, entry_id, actual_epoch, selection_mode),
            )
            if not self._in_transaction:
                self._conn.commit()

    def get_historical_slots(self) -> list[dict[str, Any]]:
        """Read all historical library slots, joined with entry data."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT h.slot_index, h.target_epoch, h.actual_epoch,
                          h.selection_mode, h.selected_at, h.entry_id,
                          e.display_name, e.elo_rating, e.elo_historical,
                          e.created_epoch, e.checkpoint_path, e.architecture,
                          e.model_params
                   FROM historical_library h
                   LEFT JOIN league_entries e ON h.entry_id = e.id
                   ORDER BY h.slot_index"""
            ).fetchall()
            return [dict(r) for r in rows]

    def record_gauntlet_result(
        self,
        epoch: int,
        entry_id: int,
        historical_slot: int,
        historical_entry_id: int,
        wins: int,
        losses: int,
        draws: int,
        elo_before: float,
        elo_after: float,
    ) -> None:
        """Record a gauntlet match result."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO gauntlet_results
                   (epoch, entry_id, historical_slot, historical_entry_id,
                    wins, losses, draws, elo_before, elo_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (epoch, entry_id, historical_slot, historical_entry_id,
                 wins, losses, draws, elo_before, elo_after),
            )
            if not self._in_transaction:
                self._conn.commit()
```

Also extend `OpponentEntry` with the new Elo fields (with defaults for backward compatibility):

```python
    # Phase 2 role-specific Elo
    elo_frontier: float = 1000.0
    elo_dynamic: float = 1000.0
    elo_recent: float = 1000.0
    elo_historical: float = 1000.0
```

Update `from_db_row` to read these columns (with fallback to 1000.0 if column doesn't exist in the row, for test compatibility):

```python
        elo_frontier = row["elo_frontier"] if "elo_frontier" in row.keys() else 1000.0
        elo_dynamic = row["elo_dynamic"] if "elo_dynamic" in row.keys() else 1000.0
        elo_recent = row["elo_recent"] if "elo_recent" in row.keys() else 1000.0
        elo_historical = row["elo_historical"] if "elo_historical" in row.keys() else 1000.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_opponent_store.py -v --no-header 2>&1 | tail -15`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/opponent_store.py tests/test_opponent_store.py
git commit -m "feat(store): add role Elo, historical slot, and gauntlet result methods"
```

---

### Task 4: RoleEloTracker

**Files:**
- Create: `keisei/training/role_elo.py`
- Create: `tests/test_role_elo.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_role_elo.py`:

```python
"""Tests for RoleEloTracker — role-specific Elo updates."""

import sqlite3

import pytest
import torch

from keisei.config import RoleEloConfig
from keisei.db import init_db
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role, EntryStatus
from keisei.training.role_elo import RoleEloTracker


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "elo.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    return OpponentStore(db_path, str(league_dir))


def _add(store, epoch, role, elo=1000.0):
    model = torch.nn.Linear(10, 10)
    entry = store.add_entry(model, "resnet", {}, epoch=epoch, role=role)
    if elo != 1000.0:
        store.update_elo(entry.id, elo)
    return store._get_entry(entry.id)


class TestRoleEloTracker:
    def test_frontier_match_updates_elo_frontier(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        a = _add(store, 1, Role.DYNAMIC)
        b = _add(store, 2, Role.FRONTIER_STATIC)
        tracker.update_from_result(a, b, result_score=0.75, match_context="frontier")
        updated_a = store._get_entry(a.id)
        assert updated_a.elo_frontier != 1000.0  # should have changed
        assert updated_a.elo_dynamic == 1000.0  # should NOT have changed
        assert updated_a.elo_recent == 1000.0
        assert updated_a.elo_historical == 1000.0

    def test_dynamic_match_updates_elo_dynamic(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        a = _add(store, 1, Role.DYNAMIC)
        b = _add(store, 2, Role.DYNAMIC)
        tracker.update_from_result(a, b, result_score=0.6, match_context="dynamic")
        updated_a = store._get_entry(a.id)
        assert updated_a.elo_dynamic != 1000.0

    def test_recent_match_updates_elo_recent(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        a = _add(store, 1, Role.RECENT_FIXED)
        b = _add(store, 2, Role.RECENT_FIXED)
        tracker.update_from_result(a, b, result_score=0.5, match_context="recent")
        updated_a = store._get_entry(a.id)
        # Draw between equals — no change expected
        assert abs(updated_a.elo_recent - 1000.0) < 0.1

    def test_historical_match_updates_elo_historical(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        a = _add(store, 1, Role.DYNAMIC)
        b = _add(store, 50, Role.RETIRED)
        tracker.update_from_result(a, b, result_score=1.0, match_context="historical")
        updated_a = store._get_entry(a.id)
        assert updated_a.elo_historical > 1000.0

    def test_k_factors_differ_by_context(self, store):
        config = RoleEloConfig(frontier_k=16.0, dynamic_k=32.0)
        tracker = RoleEloTracker(store, config)
        a1 = _add(store, 1, Role.DYNAMIC, elo=1000.0)
        b1 = _add(store, 2, Role.FRONTIER_STATIC, elo=1000.0)
        tracker.update_from_result(a1, b1, result_score=1.0, match_context="frontier")
        delta_frontier = store._get_entry(a1.id).elo_frontier - 1000.0

        a2 = _add(store, 3, Role.DYNAMIC, elo=1000.0)
        b2 = _add(store, 4, Role.DYNAMIC, elo=1000.0)
        tracker.update_from_result(a2, b2, result_score=1.0, match_context="dynamic")
        delta_dynamic = store._get_entry(a2.id).elo_dynamic - 1000.0

        # Dynamic K=32 is 2x frontier K=16, so delta should be ~2x
        assert abs(delta_dynamic / delta_frontier - 2.0) < 0.1

    def test_composite_elo_not_modified(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        a = _add(store, 1, Role.DYNAMIC, elo=1500.0)
        b = _add(store, 2, Role.FRONTIER_STATIC, elo=1500.0)
        tracker.update_from_result(a, b, result_score=1.0, match_context="frontier")
        # Composite elo_rating should be untouched
        assert store._get_entry(a.id).elo_rating == 1500.0

    def test_get_role_elos(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        a = _add(store, 1, Role.DYNAMIC)
        store.update_role_elo(a.id, "elo_frontier", 1100.0)
        store.update_role_elo(a.id, "elo_dynamic", 1200.0)
        elos = tracker.get_role_elos(a.id)
        assert elos["elo_frontier"] == 1100.0
        assert elos["elo_dynamic"] == 1200.0
        assert elos["elo_recent"] == 1000.0
        assert elos["elo_historical"] == 1000.0

    def test_cross_tier_dynamic_vs_recent(self, store):
        """Dynamic vs Recent Fixed: elo_dynamic on Dynamic, elo_recent on Recent Fixed."""
        tracker = RoleEloTracker(store, RoleEloConfig())
        a = _add(store, 1, Role.DYNAMIC)
        b = _add(store, 2, Role.RECENT_FIXED)
        tracker.update_from_result(a, b, result_score=0.75, match_context="dynamic_recent")
        updated_a = store._get_entry(a.id)
        updated_b = store._get_entry(b.id)
        assert updated_a.elo_dynamic != 1000.0
        assert updated_a.elo_recent == 1000.0  # Dynamic's recent Elo unchanged
        assert updated_b.elo_recent != 1000.0
        assert updated_b.elo_dynamic == 1000.0  # Recent's dynamic Elo unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_role_elo.py -v --no-header 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement role_elo.py**

Create `keisei/training/role_elo.py`:

```python
"""RoleEloTracker — per-context Elo updates for the tiered opponent pool."""

from __future__ import annotations

import logging

from keisei.config import RoleEloConfig
from keisei.training.opponent_store import OpponentEntry, OpponentStore, compute_elo_update

logger = logging.getLogger(__name__)

# Maps match_context string to (elo_column_for_a, elo_column_for_b, k_factor_attr)
_CONTEXT_MAP = {
    "frontier": ("elo_frontier", "elo_frontier", "frontier_k"),
    "dynamic": ("elo_dynamic", "elo_dynamic", "dynamic_k"),
    "recent": ("elo_recent", "elo_recent", "recent_k"),
    "historical": ("elo_historical", "elo_historical", "historical_k"),
    "dynamic_recent": ("elo_dynamic", "elo_recent", "dynamic_k"),
    "dynamic_frontier": ("elo_frontier", "elo_frontier", "frontier_k"),
    "recent_frontier": ("elo_frontier", "elo_frontier", "frontier_k"),
}


class RoleEloTracker:
    """Computes and stores role-specific Elo updates."""

    def __init__(self, store: OpponentStore, config: RoleEloConfig) -> None:
        self.store = store
        self.config = config

    def update_from_result(
        self,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
        result_score: float,
        match_context: str,
    ) -> None:
        """Update role-specific Elo for both entries based on match context.

        Args:
            entry_a: First participant.
            entry_b: Second participant.
            result_score: 1.0 = A wins, 0.5 = draw, 0.0 = A loses.
            match_context: One of "frontier", "dynamic", "recent", "historical",
                          "dynamic_recent", "dynamic_frontier", "recent_frontier".
        """
        if match_context not in _CONTEXT_MAP:
            logger.warning("Unknown match context %r — skipping role Elo update", match_context)
            return

        col_a, col_b, k_attr = _CONTEXT_MAP[match_context]
        k = getattr(self.config, k_attr)

        elo_a = getattr(entry_a, col_a)
        elo_b = getattr(entry_b, col_b)

        new_a, new_b = compute_elo_update(elo_a, elo_b, result_score, k=k)

        self.store.update_role_elo(entry_a.id, col_a, new_a)
        self.store.update_role_elo(entry_b.id, col_b, new_b)

        logger.debug(
            "Role Elo [%s]: %s %.0f->%.0f, %s %.0f->%.0f",
            match_context, entry_a.display_name, elo_a, new_a,
            entry_b.display_name, elo_b, new_b,
        )

    def get_role_elos(self, entry_id: int) -> dict[str, float]:
        """Get all role-specific Elo values for an entry."""
        entry = self.store._get_entry(entry_id)
        if entry is None:
            return {}
        return {
            "elo_frontier": entry.elo_frontier,
            "elo_dynamic": entry.elo_dynamic,
            "elo_recent": entry.elo_recent,
            "elo_historical": entry.elo_historical,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_role_elo.py -v --no-header 2>&1 | tail -15`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/role_elo.py tests/test_role_elo.py
git commit -m "feat(league): add RoleEloTracker with per-context K-factors"
```

---

### Task 5: HistoricalLibrary

**Files:**
- Create: `keisei/training/historical_library.py`
- Create: `tests/test_historical_library.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_historical_library.py`:

```python
"""Tests for HistoricalLibrary — log-spaced milestone selection."""

import math

import pytest
import torch

from keisei.config import HistoricalLibraryConfig
from keisei.db import init_db
from keisei.training.opponent_store import OpponentStore, Role, EntryStatus
from keisei.training.historical_library import HistoricalLibrary, compute_log_spaced_targets


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "hist.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    return OpponentStore(db_path, str(league_dir))


def _add(store, epoch, role=Role.RETIRED):
    model = torch.nn.Linear(10, 10)
    entry = store.add_entry(model, "resnet", {}, epoch=epoch, role=role)
    if role == Role.RETIRED:
        store.retire_entry(entry.id, "archived for history")
    return store._get_entry(entry.id)


class TestLogSpacedTargets:
    def test_targets_at_10000(self):
        targets = compute_log_spaced_targets(10000, slots=5)
        assert len(targets) == 5
        assert targets[0] == 1
        assert targets[-1] == 10000
        # Middle targets should be roughly log-spaced
        assert 5 < targets[1] < 20
        assert 50 < targets[2] < 200
        assert 500 < targets[3] < 2000

    def test_targets_at_250000(self):
        targets = compute_log_spaced_targets(250000, slots=5)
        assert targets[0] == 1
        assert targets[-1] == 250000

    def test_targets_at_small_epoch(self):
        targets = compute_log_spaced_targets(10, slots=5)
        assert len(targets) == 5
        assert targets[0] >= 1

    def test_targets_at_epoch_1(self):
        targets = compute_log_spaced_targets(1, slots=5)
        # All targets should be 1 when E=1
        assert all(t >= 1 for t in targets)


class TestHistoricalLibrary:
    def test_refresh_populates_slots(self, store):
        # Create entries at various epochs
        for epoch in [1, 10, 50, 200, 1000, 5000, 10000]:
            _add(store, epoch)
        lib = HistoricalLibrary(store, HistoricalLibraryConfig())
        lib.refresh(10000)
        slots = lib.get_slots()
        assert len(slots) == 5
        filled = [s for s in slots if s["entry_id"] is not None]
        assert len(filled) >= 4  # Most slots should find a nearby checkpoint

    def test_refresh_idempotent(self, store):
        for epoch in [1, 100, 1000, 10000]:
            _add(store, epoch)
        lib = HistoricalLibrary(store, HistoricalLibraryConfig())
        lib.refresh(10000)
        slots1 = lib.get_slots()
        lib.refresh(10000)
        slots2 = lib.get_slots()
        for s1, s2 in zip(slots1, slots2):
            assert s1["entry_id"] == s2["entry_id"]
            assert s1["actual_epoch"] == s2["actual_epoch"]

    def test_early_training_fallback(self, store):
        """With only 2 checkpoints, slots should use fallback mode."""
        _add(store, 1)
        _add(store, 10)
        lib = HistoricalLibrary(store, HistoricalLibraryConfig())
        lib.refresh(10)
        slots = lib.get_slots()
        assert len(slots) == 5
        fallback_count = sum(1 for s in slots if s.get("selection_mode") == "fallback")
        assert fallback_count >= 1

    def test_is_due_for_refresh(self, store):
        lib = HistoricalLibrary(store, HistoricalLibraryConfig(refresh_interval_epochs=100))
        assert lib.is_due_for_refresh(100)
        assert lib.is_due_for_refresh(200)
        assert not lib.is_due_for_refresh(50)
        assert not lib.is_due_for_refresh(0)

    def test_no_checkpoints_produces_empty_slots(self, store):
        lib = HistoricalLibrary(store, HistoricalLibraryConfig())
        lib.refresh(1000)
        slots = lib.get_slots()
        assert len(slots) == 5
        assert all(s["entry_id"] is None for s in slots)

    def test_prefers_retired_entries(self, store):
        """When active and retired entries are equidistant, prefer retired."""
        model = torch.nn.Linear(10, 10)
        active = store.add_entry(model, "resnet", {}, epoch=100, role=Role.DYNAMIC)
        retired = _add(store, 101)  # retired, epoch 101
        lib = HistoricalLibrary(store, HistoricalLibraryConfig())
        lib.refresh(100)
        slots = lib.get_slots()
        # The slot targeting ~100 should prefer the retired entry
        slot_at_100 = [s for s in slots if s["entry_id"] is not None and abs(s["actual_epoch"] - 100) <= 5]
        if slot_at_100:
            assert slot_at_100[0]["entry_id"] == retired.id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_historical_library.py -v --no-header 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement historical_library.py**

Create `keisei/training/historical_library.py`:

```python
"""HistoricalLibrary — log-spaced milestone selection for long-range regression detection."""

from __future__ import annotations

import logging
import math
from typing import Any

from keisei.config import HistoricalLibraryConfig
from keisei.training.opponent_store import OpponentStore

logger = logging.getLogger(__name__)


def compute_log_spaced_targets(current_epoch: int, slots: int = 5) -> list[int]:
    """Compute log-spaced milestone target epochs.

    For current epoch E and slots i = 0..slots-1:
        T_i = round(exp(log(max(E, 2)) * i / (slots - 1)))

    Produces approximately {1, 10, 100, 1000, 10000} at E=10000.
    """
    e = max(current_epoch, 2)
    targets = []
    for i in range(slots):
        t = round(math.exp(math.log(e) * i / max(slots - 1, 1)))
        targets.append(max(1, t))
    return targets


class HistoricalLibrary:
    """Manages 5 log-spaced milestone slots for historical regression detection."""

    def __init__(self, store: OpponentStore, config: HistoricalLibraryConfig) -> None:
        self.store = store
        self.config = config

    def refresh(self, current_epoch: int) -> None:
        """Recompute log-spaced targets and snap to nearest archived checkpoints."""
        if current_epoch < self.config.min_epoch_for_selection:
            logger.info("Epoch %d < min_epoch_for_selection %d, skipping refresh",
                        current_epoch, self.config.min_epoch_for_selection)
            # Still populate empty slots for dashboard invariant
            for i in range(self.config.slots):
                self.store.upsert_historical_slot(i, 0, None, None, "fallback")
            return

        targets = compute_log_spaced_targets(current_epoch, self.config.slots)
        candidates = self._get_candidate_entries()

        for slot_index, target in enumerate(targets):
            if not candidates:
                self.store.upsert_historical_slot(slot_index, target, None, None, "log_spaced")
                continue

            # Find nearest candidate
            best = min(candidates, key=lambda e: abs(e["created_epoch"] - target))
            distance = abs(best["created_epoch"] - target)

            # Distance threshold: 50% of gap to nearest neighbor target
            if slot_index > 0 and slot_index < len(targets) - 1:
                neighbor_gap = min(
                    abs(targets[slot_index] - targets[slot_index - 1]),
                    abs(targets[slot_index + 1] - targets[slot_index]),
                )
            elif slot_index == 0:
                neighbor_gap = abs(targets[1] - targets[0]) if len(targets) > 1 else target
            else:
                neighbor_gap = abs(targets[-1] - targets[-2]) if len(targets) > 1 else target

            threshold = max(neighbor_gap * 0.5, 1)

            if distance <= threshold:
                mode = "log_spaced"
            elif candidates:
                mode = "fallback"
            else:
                self.store.upsert_historical_slot(slot_index, target, None, None, "log_spaced")
                continue

            self.store.upsert_historical_slot(
                slot_index, target, best["id"], best["created_epoch"], mode,
            )

        logger.info(
            "Historical library refreshed at epoch %d: targets=%s",
            current_epoch, targets,
        )

    def get_slots(self) -> list[dict[str, Any]]:
        """Returns the 5 historical library slots with entry data."""
        return self.store.get_historical_slots()

    def is_due_for_refresh(self, epoch: int) -> bool:
        """True if epoch aligns with refresh interval."""
        return epoch > 0 and epoch % self.config.refresh_interval_epochs == 0

    def _get_candidate_entries(self) -> list[dict[str, Any]]:
        """Get all entries eligible for historical selection.

        Prefers retired/archived entries (stable) over active entries.
        Returns dicts with id, created_epoch, status.
        """
        with self.store._lock:
            rows = self.store._conn.execute(
                """SELECT id, created_epoch, status FROM league_entries
                   ORDER BY
                       CASE WHEN status IN ('retired', 'archived') THEN 0 ELSE 1 END,
                       created_epoch ASC"""
            ).fetchall()
            return [dict(r) for r in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_historical_library.py -v --no-header 2>&1 | tail -15`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/historical_library.py tests/test_historical_library.py
git commit -m "feat(league): add HistoricalLibrary with log-spaced milestone selection"
```

---

### Task 6: HistoricalGauntlet

**Files:**
- Create: `keisei/training/historical_gauntlet.py`
- Create: `tests/test_historical_gauntlet.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_historical_gauntlet.py`:

```python
"""Tests for HistoricalGauntlet — periodic benchmark runner."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest
import torch

from keisei.config import GauntletConfig, RoleEloConfig
from keisei.db import init_db
from keisei.training.opponent_store import OpponentStore, Role
from keisei.training.role_elo import RoleEloTracker
from keisei.training.historical_gauntlet import HistoricalGauntlet


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "gauntlet.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    return OpponentStore(db_path, str(league_dir))


class TestGauntletIsDue:
    def test_is_due_at_interval(self):
        gauntlet = HistoricalGauntlet(
            store=MagicMock(), role_elo_tracker=MagicMock(),
            config=GauntletConfig(interval_epochs=100),
        )
        assert gauntlet.is_due(100)
        assert gauntlet.is_due(200)
        assert not gauntlet.is_due(50)
        assert not gauntlet.is_due(0)

    def test_is_due_disabled(self):
        gauntlet = HistoricalGauntlet(
            store=MagicMock(), role_elo_tracker=MagicMock(),
            config=GauntletConfig(enabled=False),
        )
        assert not gauntlet.is_due(100)


class TestGauntletRunning:
    def test_run_gauntlet_records_results(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        gauntlet = HistoricalGauntlet(
            store=store, role_elo_tracker=tracker,
            config=GauntletConfig(games_per_matchup=4),
        )
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RECENT_FIXED)
        historical = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RETIRED)
        store.retire_entry(historical.id, "archived")
        historical = store._get_entry(historical.id)

        slots = [
            {"slot_index": 0, "entry_id": historical.id, "actual_epoch": 50,
             "checkpoint_path": historical.checkpoint_path,
             "architecture": "resnet", "model_params": "{}"},
        ]

        # Mock the actual game playing — we test the recording, not the inference
        with patch.object(gauntlet, "_play_matchup", return_value=(3, 1, 0)):
            gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots)

        conn = sqlite3.connect(store.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM gauntlet_results").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["wins"] == 3
        assert rows[0]["losses"] == 1

    def test_run_gauntlet_skips_empty_slots(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        gauntlet = HistoricalGauntlet(
            store=store, role_elo_tracker=tracker,
            config=GauntletConfig(),
        )
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RECENT_FIXED)

        slots = [
            {"slot_index": 0, "entry_id": None, "actual_epoch": None},
            {"slot_index": 1, "entry_id": None, "actual_epoch": None},
        ]

        gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots)

        conn = sqlite3.connect(store.db_path)
        count = conn.execute("SELECT COUNT(*) FROM gauntlet_results").fetchone()[0]
        conn.close()
        assert count == 0  # nothing played

    def test_run_gauntlet_updates_elo_historical(self, store):
        tracker = RoleEloTracker(store, RoleEloConfig())
        gauntlet = HistoricalGauntlet(
            store=store, role_elo_tracker=tracker,
            config=GauntletConfig(games_per_matchup=4),
        )
        model = torch.nn.Linear(10, 10)
        learner = store.add_entry(model, "resnet", {}, epoch=100, role=Role.RECENT_FIXED)
        historical = store.add_entry(model, "resnet", {}, epoch=50, role=Role.RETIRED)
        store.retire_entry(historical.id, "archived")
        historical = store._get_entry(historical.id)

        slots = [
            {"slot_index": 0, "entry_id": historical.id, "actual_epoch": 50,
             "checkpoint_path": historical.checkpoint_path,
             "architecture": "resnet", "model_params": "{}"},
        ]

        with patch.object(gauntlet, "_play_matchup", return_value=(4, 0, 0)):
            gauntlet.run_gauntlet(epoch=100, learner_entry=learner, historical_slots=slots)

        updated = store._get_entry(learner.id)
        assert updated.elo_historical > 1000.0  # should have increased after 4-0 win
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_historical_gauntlet.py -v --no-header 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement historical_gauntlet.py**

Create `keisei/training/historical_gauntlet.py`:

```python
"""HistoricalGauntlet — periodic learner-vs-history benchmark runner."""

from __future__ import annotations

import logging
from typing import Any

from keisei.config import GauntletConfig
from keisei.training.opponent_store import OpponentEntry, OpponentStore, compute_elo_update
from keisei.training.role_elo import RoleEloTracker

logger = logging.getLogger(__name__)


class HistoricalGauntlet:
    """Runs periodic benchmark matches between the learner and historical milestones.

    Designed to run synchronously on the tournament thread after round-robin
    matches complete. Uses the same device and VecEnv as the tournament.
    """

    def __init__(
        self,
        store: OpponentStore,
        role_elo_tracker: RoleEloTracker,
        config: GauntletConfig,
    ) -> None:
        self.store = store
        self.role_elo_tracker = role_elo_tracker
        self.config = config

    def is_due(self, epoch: int) -> bool:
        """True if the gauntlet should run at this epoch."""
        if not self.config.enabled:
            return False
        return epoch > 0 and epoch % self.config.interval_epochs == 0

    def run_gauntlet(
        self,
        epoch: int,
        learner_entry: OpponentEntry,
        historical_slots: list[dict[str, Any]],
        stop_event: Any | None = None,
    ) -> None:
        """Play the learner against each non-empty historical slot.

        Args:
            epoch: Current training epoch.
            learner_entry: The learner's current OpponentEntry.
            historical_slots: List of slot dicts from HistoricalLibrary.get_slots().
            stop_event: Optional threading.Event for graceful shutdown.
        """
        played = 0
        for slot in historical_slots:
            if stop_event and stop_event.is_set():
                break

            if slot.get("entry_id") is None:
                continue

            historical_entry_id = slot["entry_id"]
            slot_index = slot["slot_index"]
            historical_entry = self.store._get_entry(historical_entry_id)
            if historical_entry is None:
                logger.warning("Historical entry %d not found, skipping slot %d",
                               historical_entry_id, slot_index)
                continue

            elo_before = learner_entry.elo_historical

            try:
                wins, losses, draws = self._play_matchup(learner_entry, historical_entry)
            except Exception:
                logger.exception("Gauntlet matchup failed: slot %d", slot_index)
                continue

            total = wins + losses + draws
            if total == 0:
                continue

            # Update elo_historical via RoleEloTracker
            result_score = (wins + 0.5 * draws) / total
            self.role_elo_tracker.update_from_result(
                learner_entry, historical_entry,
                result_score=result_score,
                match_context="historical",
            )

            # Re-read to get updated elo
            updated = self.store._get_entry(learner_entry.id)
            elo_after = updated.elo_historical if updated else elo_before

            self.store.record_gauntlet_result(
                epoch=epoch,
                entry_id=learner_entry.id,
                historical_slot=slot_index,
                historical_entry_id=historical_entry_id,
                wins=wins, losses=losses, draws=draws,
                elo_before=elo_before, elo_after=elo_after,
            )

            # Update learner_entry reference for next iteration's elo_before
            if updated:
                learner_entry = updated

            played += 1
            logger.info(
                "Gauntlet slot %d (epoch %d): %dW %dL %dD, elo_historical %.0f->%.0f",
                slot_index, slot.get("actual_epoch", 0),
                wins, losses, draws, elo_before, elo_after,
            )

        logger.info("Gauntlet complete at epoch %d: %d/%d slots played",
                     epoch, played, len(historical_slots))

    def _play_matchup(
        self,
        learner_entry: OpponentEntry,
        historical_entry: OpponentEntry,
    ) -> tuple[int, int, int]:
        """Play games_per_matchup games between learner and a historical entry.

        This method is overridden in integration when a VecEnv and device are
        available. The base implementation raises NotImplementedError — tests
        mock this method.
        """
        raise NotImplementedError(
            "_play_matchup must be overridden or mocked. "
            "In production, the tournament runner calls this with VecEnv access."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_historical_gauntlet.py -v --no-header 2>&1 | tail -15`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/historical_gauntlet.py tests/test_historical_gauntlet.py
git commit -m "feat(league): add HistoricalGauntlet benchmark runner"
```

---

### Task 7: TieredPool Integration

**Files:**
- Modify: `keisei/training/tiered_pool.py`
- Test: `tests/test_tiered_pool.py`

- [ ] **Step 1: Write failing tests**

In `tests/test_tiered_pool.py`, add:

```python
from keisei.training.historical_library import HistoricalLibrary


class TestHistoricalIntegration:
    def test_get_historical_slots(self, pool_setup):
        pool, store, _ = pool_setup
        # Add some retired entries for the library
        model = torch.nn.Linear(10, 10)
        for epoch in [1, 100, 1000, 5000, 10000]:
            e = store.add_entry(model, "resnet", {}, epoch=epoch, role=Role.UNASSIGNED)
            store.retire_entry(e.id, "archived")
        # Refresh historical library
        pool.historical_library.refresh(10000)
        slots = pool.get_historical_slots()
        assert len(slots) == 5

    def test_on_epoch_end_triggers_refresh(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        for epoch in [1, 50, 100]:
            e = store.add_entry(model, "resnet", {}, epoch=epoch, role=Role.UNASSIGNED)
            store.retire_entry(e.id, "archived")
        # on_epoch_end at refresh interval should trigger
        pool.on_epoch_end(100)
        slots = pool.get_historical_slots()
        # Should have slots populated (even if some are NULL)
        assert len(slots) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tiered_pool.py::TestHistoricalIntegration -v --no-header 2>&1 | tail -5`
Expected: AttributeError — `historical_library` doesn't exist on TieredPool.

- [ ] **Step 3: Update tiered_pool.py**

In `keisei/training/tiered_pool.py`, add imports:

```python
from keisei.training.historical_library import HistoricalLibrary
from keisei.training.role_elo import RoleEloTracker
```

In `TieredPool.__init__`, add after the existing manager creation:

```python
        # Phase 2: Historical Library & Role Elo
        self.historical_library = HistoricalLibrary(store, config.history)
        self.role_elo_tracker = RoleEloTracker(store, config.role_elo)
```

Add new method:

```python
    def get_historical_slots(self) -> list[dict]:
        """Get the 5 historical library slots."""
        return self.historical_library.get_slots()
```

Update `on_epoch_end` to add historical refresh:

```python
    def on_epoch_end(self, epoch: int) -> None:
        if self.frontier_manager.is_due_for_review(epoch):
            self.frontier_manager.review(epoch)
        if self.historical_library.is_due_for_refresh(epoch):
            self.historical_library.refresh(epoch)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tiered_pool.py -v --no-header 2>&1 | tail -10`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/tiered_pool.py tests/test_tiered_pool.py
git commit -m "feat(league): wire HistoricalLibrary and RoleEloTracker into TieredPool"
```

---

### Task 8: Tournament Integration

**Files:**
- Modify: `keisei/training/tournament.py`
- Modify: `keisei/training/katago_loop.py`

- [ ] **Step 1: Update tournament constructor**

In `keisei/training/tournament.py`, add to constructor parameters:

```python
        historical_library: HistoricalLibrary | None = None,
        gauntlet: HistoricalGauntlet | None = None,
```

Store as instance variables:

```python
        self.historical_library = historical_library
        self.gauntlet = gauntlet
```

Add imports:

```python
from keisei.training.historical_library import HistoricalLibrary
from keisei.training.historical_gauntlet import HistoricalGauntlet
```

- [ ] **Step 2: Add gauntlet hook to tournament loop**

In the `_run_loop` method, after the round-robin pairings loop completes (after the "Tournament round E%d complete" log line), add:

```python
                # Historical gauntlet (runs after round-robin when due)
                if self.gauntlet and self.gauntlet.is_due(epoch):
                    if self.historical_library:
                        self.historical_library.refresh(epoch)
                    slots = self.historical_library.get_slots() if self.historical_library else []
                    learner_entry = self._get_learner_entry()
                    if learner_entry and slots:
                        # Override _play_matchup with our VecEnv-backed version
                        self.gauntlet._play_matchup = lambda a, b: self._play_gauntlet_match(vecenv, a, b)
                        self.gauntlet.run_gauntlet(
                            epoch=epoch, learner_entry=learner_entry,
                            historical_slots=slots, stop_event=self._stop_event,
                        )
                        logger.info("Historical gauntlet completed at epoch %d", epoch)
```

Add helper method to get the learner entry:

```python
    def _get_learner_entry(self) -> OpponentEntry | None:
        """Get the current learner entry from the store."""
        entries = self.store.list_entries()
        if not entries:
            return None
        # The learner entry is the most recent by created_epoch
        return max(entries, key=lambda e: e.created_epoch)
```

Add the VecEnv-backed gauntlet match method:

```python
    def _play_gauntlet_match(
        self,
        vecenv: object,
        learner_entry: OpponentEntry,
        historical_entry: OpponentEntry,
    ) -> tuple[int, int, int]:
        """Play gauntlet games using the tournament's VecEnv."""
        return self._play_match(vecenv, learner_entry, historical_entry)
```

Note: `_play_match` already handles loading models, playing games, and returning (wins_a, wins_b, draws). The gauntlet reuses this infrastructure. The `games_per_match` parameter in `_play_match` controls how many games are played — for the gauntlet this should use `gauntlet.config.games_per_matchup`. Update `_play_match` to accept an optional `games_override` parameter, or update the gauntlet to temporarily set `self.games_per_match`. The simplest approach is to pass through:

```python
    def _play_gauntlet_match(self, vecenv, learner_entry, historical_entry):
        original = self.games_per_match
        self.games_per_match = self.gauntlet.config.games_per_matchup
        try:
            return self._play_match(vecenv, learner_entry, historical_entry)
        finally:
            self.games_per_match = original
```

- [ ] **Step 3: Update katago_loop.py tournament construction**

In `keisei/training/katago_loop.py`, update the `LeagueTournament` construction to pass the new parameters:

```python
                from keisei.training.historical_gauntlet import HistoricalGauntlet

                gauntlet = None
                if config.league.gauntlet.enabled:
                    gauntlet = HistoricalGauntlet(
                        store=self.store,
                        role_elo_tracker=self.tiered_pool.role_elo_tracker,
                        config=config.league.gauntlet,
                    )

                self._tournament = LeagueTournament(
                    store=self.store,
                    scheduler=self.scheduler,
                    historical_library=self.tiered_pool.historical_library,
                    gauntlet=gauntlet,
                    device=tournament_device,
                    # ... rest of existing params ...
                )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v --no-header -x 2>&1 | tail -20`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/tournament.py keisei/training/katago_loop.py
git commit -m "feat(league): integrate HistoricalGauntlet into tournament runner"
```

---

### Task 9: Role Elo in Tournament Round-Robin

**Files:**
- Modify: `keisei/training/tournament.py`

- [ ] **Step 1: Add role Elo tracking to tournament match results**

In the tournament's match result recording (the loop that iterates over pairings), after the existing Elo update, add role-specific Elo updates. Import `RoleEloTracker` and determine match context from the participants' roles:

```python
from keisei.training.role_elo import RoleEloTracker
```

Add `role_elo_tracker: RoleEloTracker | None = None` to the constructor. Store as `self.role_elo_tracker`.

In the match result section, after existing Elo recording, add:

```python
                    # Role-specific Elo update
                    if self.role_elo_tracker:
                        context = self._determine_match_context(entry_a, entry_b)
                        result_score = (wins_a + 0.5 * draws) / total
                        self.role_elo_tracker.update_from_result(
                            entry_a, entry_b, result_score, match_context=context,
                        )
```

Add helper to determine match context:

```python
    def _determine_match_context(
        self, entry_a: OpponentEntry, entry_b: OpponentEntry,
    ) -> str:
        """Determine which Elo context applies based on participant roles."""
        from keisei.training.opponent_store import Role
        roles = {entry_a.role, entry_b.role}
        if Role.FRONTIER_STATIC in roles:
            if Role.DYNAMIC in roles:
                return "dynamic_frontier"
            if Role.RECENT_FIXED in roles:
                return "recent_frontier"
            return "frontier"
        if roles == {Role.DYNAMIC}:
            return "dynamic"
        if roles == {Role.RECENT_FIXED}:
            return "recent"
        if Role.DYNAMIC in roles and Role.RECENT_FIXED in roles:
            return "dynamic_recent"
        return "dynamic"  # fallback
```

- [ ] **Step 2: Update katago_loop.py to pass role_elo_tracker**

```python
                self._tournament = LeagueTournament(
                    store=self.store,
                    scheduler=self.scheduler,
                    historical_library=self.tiered_pool.historical_library,
                    gauntlet=gauntlet,
                    role_elo_tracker=self.tiered_pool.role_elo_tracker,
                    # ... rest of existing params ...
                )
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -v --no-header -x 2>&1 | tail -20`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add keisei/training/tournament.py keisei/training/katago_loop.py
git commit -m "feat(league): add role-specific Elo tracking to tournament round-robin"
```

---

### Task 10: Integration Test

**Files:**
- Create or modify: `tests/test_tiered_pool.py`

- [ ] **Step 1: Write full gauntlet integration test**

In `tests/test_tiered_pool.py`, add:

```python
class TestGauntletIntegration:
    """Full cycle: create entries, retire, refresh library, run gauntlet, verify results."""

    def test_full_gauntlet_cycle(self, pool_setup):
        pool, store, db_path = pool_setup
        model = torch.nn.Linear(10, 10)

        # Create entries at various epochs and retire them (historical candidates)
        for epoch in [1, 10, 50, 200, 500, 1000, 2000, 5000, 10000]:
            e = store.add_entry(model, "resnet", {}, epoch=epoch, role=Role.UNASSIGNED)
            store.retire_entry(e.id, "archived for history test")

        # Create a "learner" entry
        learner = store.add_entry(model, "resnet", {}, epoch=10001, role=Role.RECENT_FIXED)

        # Refresh historical library
        pool.historical_library.refresh(10000)
        slots = pool.get_historical_slots()
        filled = [s for s in slots if s["entry_id"] is not None]
        assert len(filled) >= 4, f"Expected at least 4 filled slots, got {len(filled)}"

        # Verify slots span the epoch range
        epochs = sorted(s["actual_epoch"] for s in filled)
        assert epochs[0] < 100, "First slot should be early training"
        assert epochs[-1] > 1000, "Last slot should be recent"

    def test_role_elo_independence(self, pool_setup):
        """Role-specific Elo updates should not affect composite elo_rating."""
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.FRONTIER_STATIC)
        original_elo_a = a.elo_rating

        pool.role_elo_tracker.update_from_result(a, b, result_score=1.0, match_context="frontier")

        updated_a = store._get_entry(a.id)
        assert updated_a.elo_rating == original_elo_a  # composite unchanged
        assert updated_a.elo_frontier != 1000.0  # role-specific changed
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_tiered_pool.py -v --no-header 2>&1 | tail -15`
Expected: All pass.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_tiered_pool.py
git commit -m "test(league): add gauntlet integration tests and role Elo independence test"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Verify all new modules are importable**

```bash
uv run python -c "
from keisei.training.historical_library import HistoricalLibrary, compute_log_spaced_targets
from keisei.training.historical_gauntlet import HistoricalGauntlet
from keisei.training.role_elo import RoleEloTracker
from keisei.config import HistoricalLibraryConfig, GauntletConfig, RoleEloConfig
print('All Phase 2 imports OK')
"
```

- [ ] **Step 2: Verify no references to removed or renamed symbols**

```bash
grep -r "elo_rating.*role_specific\|role_specific_elo" keisei/ tests/ --include="*.py" | head -5
```
Expected: No results.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All pass.

- [ ] **Step 4: Commit if any fixups needed**

```bash
git status
# Only commit if there are changes
git diff --stat && git add -A && git commit -m "fix: final Phase 2 fixups"
```
