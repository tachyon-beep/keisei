# Tiered Opponent Pool Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the flat 20-seat OpponentPool into a tiered system with three roles (Frontier Static, Recent Fixed, Dynamic), all frozen, with role-aware matchmaking.

**Architecture:** OpponentStore (DB/filesystem storage) + three tier managers (FrontierManager, RecentFixedManager, DynamicManager) + TieredPool orchestrator + MatchScheduler. Managers own tier-specific policy; store owns data; orchestrator coordinates cross-tier operations.

**Tech Stack:** Python 3.13, SQLite (WAL mode), PyTorch, frozen dataclasses, StrEnum, threading locks, uv for deps.

**Spec:** `docs/superpowers/specs/2026-04-05-tiered-opponent-pool-phase1-design.md`

---

## Critical Implementation Notes

These notes capture design decisions from a 6-reviewer panel. Read before implementing.

### Locking and Commit Discipline

OpponentStore uses `threading.RLock()` (reentrant lock), NOT `threading.Lock()`.

**Commit rule:** No store mutating method (add_entry, clone_entry, retire_entry, update_role,
record_result, update_elo, log_transition, decrement_protection) calls `self._conn.commit()`
when inside a transaction. Each method checks `self._in_transaction`:

```python
def some_mutating_method(self, ...):
    with self._lock:
        # ... do SQL work ...
        if not self._in_transaction:
            self._conn.commit()
```

The `transaction()` context manager sets `self._in_transaction = True` on entry, commits
on success, rolls back on exception, and resets the flag on exit. This ensures:
- Standalone calls auto-commit (backward compatible)
- Calls inside `transaction()` defer commit (atomicity preserved)
- No double-commit bugs

### league.py Shim

league.py is kept as a re-export shim until the final cleanup task (Task 12). This prevents
ImportError in callers that haven't been updated yet. Do NOT run `uv run pytest tests/`
as a full suite until Task 11. Use targeted test commands between Tasks 3 and 10.

### OpponentEntry Defaults

All new fields on OpponentEntry have defaults so that existing test code constructing
OpponentEntry with only the original 10 fields continues to work without modification
until those tests are explicitly updated.

### Bootstrap Assignment Order

Recent Fixed is assigned FIRST (most recent by epoch), THEN Frontier Static
(Elo-spread quintile from remaining entries). This preserves the "fresh blood"
semantic -- the most recent snapshots always go to Recent Fixed, not to Frontier.

### Tournament Format

The tournament uses deterministic full round-robin: every active entry plays every other
active entry once per round, with best-of-3 games per pair. With 20 entries this is
190 pairings × 3 games = 570 games per round. This replaces the old format of 10 random
pairings × 64 games = 640 games per round. Fewer total games, but every pair is calibrated
every round — this eliminates pair starvation and makes promotion criteria (unique_opponents,
min_games_for_review) predictably satisfiable.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `keisei/training/opponent_store.py` | Create | OpponentStore class (storage layer), OpponentEntry dataclass, enums (Role, EntryStatus), compute_elo_update, name/flavour generation |
| `keisei/training/tier_managers.py` | Create | FrontierManager, RecentFixedManager, DynamicManager, ReviewOutcome enum |
| `keisei/training/tiered_pool.py` | Create | TieredPool orchestrator |
| `keisei/training/match_scheduler.py` | Create | MatchScheduler (replaces OpponentSampler) |
| `keisei/config.py` | Modify | Add nested tier config dataclasses, update LeagueConfig |
| `keisei/db.py` | Modify | Update schema to v4 with new columns/tables |
| `keisei/training/katago_loop.py` | Modify | Replace OpponentPool/OpponentSampler with TieredPool/MatchScheduler |
| `keisei/training/tournament.py` | Modify | Use OpponentStore instead of raw SQL |
| `keisei/training/demonstrator.py` | Modify | Update imports (OpponentPool -> OpponentStore), replace self.pool.pin/unpin with self.store.pin/unpin |
| `keisei/training/league.py` | Rewrite as shim | Re-export shim for backward compatibility until final cleanup |
| `tests/test_opponent_store.py` | Create | Tests for OpponentStore |
| `tests/test_tier_managers.py` | Create | Tests for all three tier managers |
| `tests/test_tiered_pool.py` | Create | Tests for TieredPool orchestrator |
| `tests/test_match_scheduler.py` | Create | Tests for MatchScheduler |
| `tests/test_tournament.py` | Modify | Update imports from league.py to opponent_store.py, update OpponentEntry construction |
| `tests/test_demonstrator.py` | Modify | Update imports from league.py to opponent_store.py, update OpponentEntry construction |
| `tests/test_katago_loop.py` | Modify | Update imports, replace OpponentPool references |
| `tests/test_config.py` | Modify | Update tests that reference removed LeagueConfig fields (max_pool_size, historical_ratio, current_best_ratio) |
| `tests/test_league.py` | Delete | Replaced by the four new test files (deleted in final cleanup Task 12) |

---

### Task 1: Config Dataclasses

**Files:**
- Modify: `keisei/config.py`
- Test: `tests/test_config.py` (existing -- add cases)

- [ ] **Step 1: Write failing tests for new config dataclasses**

In `tests/test_config.py`, add:

```python
from keisei.config import (
    FrontierStaticConfig,
    RecentFixedConfig,
    DynamicConfig,
    MatchSchedulerConfig,
)


class TestTierConfigs:
    def test_frontier_static_defaults(self):
        cfg = FrontierStaticConfig()
        assert cfg.slots == 5
        assert cfg.review_interval_epochs == 250
        assert cfg.min_tenure_epochs == 100
        assert cfg.promotion_margin_elo == 50.0

    def test_recent_fixed_defaults(self):
        cfg = RecentFixedConfig()
        assert cfg.slots == 5
        assert cfg.min_games_for_review == 32
        assert cfg.min_unique_opponents == 6
        assert cfg.promotion_margin_elo == 25.0
        assert cfg.soft_overflow == 1

    def test_dynamic_defaults(self):
        cfg = DynamicConfig()
        assert cfg.slots == 10
        assert cfg.protection_matches == 24
        assert cfg.min_games_before_eviction == 40
        assert cfg.training_enabled is False

    def test_match_scheduler_defaults(self):
        cfg = MatchSchedulerConfig()
        assert cfg.learner_dynamic_ratio == 0.50
        assert cfg.learner_frontier_ratio == 0.30
        assert cfg.learner_recent_ratio == 0.20


class TestLeagueConfigValidation:
    def test_learner_ratios_must_sum_to_one(self):
        import pytest
        bad_sched = MatchSchedulerConfig(
            learner_dynamic_ratio=0.5,
            learner_frontier_ratio=0.5,
            learner_recent_ratio=0.5,
        )
        with pytest.raises(ValueError, match="learner.*ratio.*sum"):
            LeagueConfig(scheduler=bad_sched)

    def test_valid_config_passes(self):
        cfg = LeagueConfig()
        assert cfg.frontier.slots + cfg.recent.slots + cfg.dynamic.slots == 20
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py::TestTierConfigs -v --no-header 2>&1 | tail -5`
Expected: ImportError -- the new config classes don't exist yet.

- [ ] **Step 3: Implement the config dataclasses**

In `keisei/config.py`, add before the existing `LeagueConfig`:

```python
@dataclass(frozen=True)
class FrontierStaticConfig:
    slots: int = 5
    review_interval_epochs: int = 250
    min_tenure_epochs: int = 100
    promotion_margin_elo: float = 50.0


@dataclass(frozen=True)
class RecentFixedConfig:
    slots: int = 5
    min_games_for_review: int = 32
    min_unique_opponents: int = 6
    promotion_margin_elo: float = 25.0
    soft_overflow: int = 1


@dataclass(frozen=True)
class DynamicConfig:
    slots: int = 10
    protection_matches: int = 24
    min_games_before_eviction: int = 40
    training_enabled: bool = False


@dataclass(frozen=True)
class MatchSchedulerConfig:
    learner_dynamic_ratio: float = 0.50
    learner_frontier_ratio: float = 0.30
    learner_recent_ratio: float = 0.20
    tournament_games_per_pair: int = 3  # best-of-3 round-robin
```

Then replace the existing `LeagueConfig` with:

```python
@dataclass(frozen=True)
class LeagueConfig:
    snapshot_interval: int = 10
    epochs_per_seat: int = 50
    initial_elo: float = 1000.0
    elo_k_factor: float = 32.0
    elo_floor: float = 500.0
    color_randomization: bool = True
    per_env_opponents: bool = True
    opponent_device: str | None = None
    tournament_enabled: bool = False
    tournament_device: str | None = None
    tournament_num_envs: int = 64
    tournament_games_per_match: int = 64
    tournament_k_factor: float = 16.0
    tournament_pause_seconds: float = 5.0
    frontier: FrontierStaticConfig = FrontierStaticConfig()
    recent: RecentFixedConfig = RecentFixedConfig()
    dynamic: DynamicConfig = DynamicConfig()
    scheduler: MatchSchedulerConfig = MatchSchedulerConfig()

    def __post_init__(self) -> None:
        if self.epochs_per_seat < 1:
            raise ValueError(
                f"league.epochs_per_seat must be >= 1, got {self.epochs_per_seat}"
            )
        if self.snapshot_interval < 1:
            raise ValueError(
                f"league.snapshot_interval must be >= 1, got {self.snapshot_interval}"
            )
        s = self.scheduler
        learner_sum = s.learner_dynamic_ratio + s.learner_frontier_ratio + s.learner_recent_ratio
        if abs(learner_sum - 1.0) > 1e-6:
            raise ValueError(
                f"learner mix ratio sum must be 1.0, got {learner_sum}"
            )
        if self.scheduler.tournament_games_per_pair < 1:
            raise ValueError(
                f"tournament_games_per_pair must be >= 1, got {self.scheduler.tournament_games_per_pair}"
            )
```

Update `load_config` to parse the nested `[league]` section. Replace the `LeagueConfig(**raw["league"])` call with nested parsing using ONE canonical key per sub-config (no dual-key fallback):

```python
    league_config = None
    if "league" in raw:
        lg = dict(raw["league"])
        frontier_raw = lg.pop("frontier", {})
        recent_raw = lg.pop("recent", {})
        dynamic_raw = lg.pop("dynamic", {})
        scheduler_raw = lg.pop("scheduler", {})
        league_config = LeagueConfig(
            **lg,
            frontier=FrontierStaticConfig(**frontier_raw),
            recent=RecentFixedConfig(**recent_raw),
            dynamic=DynamicConfig(**dynamic_raw),
            scheduler=MatchSchedulerConfig(**scheduler_raw),
        )
```

Remove the old `max_pool_size`, `historical_ratio`, and `current_best_ratio` fields and their validation.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v --no-header 2>&1 | tail -10`
Expected: All pass (new tests + existing tests -- some existing tests may need updating if they rely on removed fields like `max_pool_size`).

- [ ] **Step 5: Fix any existing tests that broke**

Check for tests relying on `max_pool_size`, `historical_ratio`, or `current_best_ratio` fields and update them to use the new nested config structure.

Run: `uv run pytest tests/test_config.py -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add keisei/config.py tests/test_config.py
git commit -m "feat(config): add nested tier config dataclasses for tiered pool"
```

---

### Task 2: Schema Update (db.py)

**Files:**
- Modify: `keisei/db.py`
- Test: `tests/test_db.py` (existing -- add cases)

- [ ] **Step 1: Write failing test for new schema**

In `tests/test_db.py`, add:

```python
class TestSchemaV4:
    def test_league_entries_has_role_column(self, tmp_path):
        db_path = str(tmp_path / "v4.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        cols = [c[1] for c in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        conn.close()
        assert "role" in cols
        assert "status" in cols
        assert "parent_entry_id" in cols
        assert "lineage_group" in cols
        assert "protection_remaining" in cols
        assert "last_match_at" in cols

    def test_league_transitions_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v4.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "league_transitions" in tables

    def test_league_meta_table_exists(self, tmp_path):
        db_path = str(tmp_path / "v4.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT bootstrapped FROM league_meta WHERE id = 1").fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 0

    def test_league_results_learner_index_exists(self, tmp_path):
        db_path = str(tmp_path / "v4.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        indexes = [r[1] for r in conn.execute("PRAGMA index_list(league_results)").fetchall()]
        conn.close()
        assert "idx_league_results_learner" in indexes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_db.py::TestSchemaV4 -v --no-header 2>&1 | tail -5`
Expected: FAIL -- columns/tables don't exist.

- [ ] **Step 3: Update init_db schema**

In `keisei/db.py`, change `SCHEMA_VERSION = 3` to `SCHEMA_VERSION = 4`.

In the `CREATE TABLE IF NOT EXISTS league_entries` block, add the new columns:

```sql
                role            TEXT NOT NULL DEFAULT 'unassigned',
                status          TEXT NOT NULL DEFAULT 'active',
                parent_entry_id INTEGER REFERENCES league_entries(id),
                lineage_group   TEXT,
                protection_remaining INTEGER NOT NULL DEFAULT 0,
                last_match_at   TEXT
```

After the `elo_history` table creation, add:

```sql
            CREATE TABLE IF NOT EXISTS league_transitions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id    INTEGER NOT NULL REFERENCES league_entries(id),
                from_role   TEXT,
                to_role     TEXT,
                from_status TEXT,
                to_status   TEXT,
                reason      TEXT,
                created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );
            CREATE INDEX IF NOT EXISTS idx_transitions_entry ON league_transitions(entry_id);
            CREATE INDEX IF NOT EXISTS idx_league_results_learner ON league_results(learner_id);
            CREATE TABLE IF NOT EXISTS league_meta (
                id           INTEGER PRIMARY KEY CHECK (id = 1),
                bootstrapped INTEGER NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO league_meta (id, bootstrapped) VALUES (1, 0);
```

Also update `read_league_data` to include the new columns in its SELECT:

```python
def read_league_data(db_path: str) -> dict[str, list[dict[str, Any]]]:
    conn = _connect(db_path)
    try:
        entries = conn.execute(
            "SELECT id, display_name, flavour_facts, model_params, architecture, "
            "elo_rating, games_played, created_epoch, created_at, "
            "role, status, parent_entry_id, lineage_group, protection_remaining, last_match_at "
            "FROM league_entries WHERE status = 'active' ORDER BY elo_rating DESC"
        ).fetchall()
        # ... rest unchanged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_db.py -v --no-header 2>&1 | tail -10`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/db.py tests/test_db.py
git commit -m "feat(db): update schema to v4 with tier columns and transitions table"
```

---

### Task 3: OpponentStore

**Files:**
- Create: `keisei/training/opponent_store.py`
- Rewrite: `keisei/training/league.py` (becomes re-export shim)
- Create: `tests/test_opponent_store.py`

This is the largest task. It moves all contents of `league.py` into `opponent_store.py`, adds the enums (`Role`, `EntryStatus`), extends `OpponentEntry` with new fields, adds the transaction API, `clone_entry`, `retire_entry`, `update_role`, `list_by_role`, `decrement_protection`, and updates `record_result` to also update `last_match_at` and decrement `protection_remaining`. Removes `_evict_if_needed` and `_delete_entry`.

- [ ] **Step 1: Write failing tests for the new enums and OpponentEntry**

Create `tests/test_opponent_store.py`:

```python
"""Tests for OpponentStore -- the tiered pool storage layer."""

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from keisei.db import init_db
from keisei.training.opponent_store import (
    OpponentEntry,
    OpponentStore,
    Role,
    EntryStatus,
    compute_elo_update,
)


class TestEnums:
    def test_role_values(self):
        assert Role.FRONTIER_STATIC == "frontier_static"
        assert Role.RECENT_FIXED == "recent_fixed"
        assert Role.DYNAMIC == "dynamic"
        assert Role.UNASSIGNED == "unassigned"

    def test_entry_status_values(self):
        assert EntryStatus.ACTIVE == "active"
        assert EntryStatus.RETIRED == "retired"
        assert EntryStatus.ARCHIVED == "archived"

    def test_role_from_string(self):
        assert Role("frontier_static") is Role.FRONTIER_STATIC

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            Role("invalid")


@pytest.fixture
def store_db(tmp_path):
    db_path = str(tmp_path / "store.db")
    init_db(db_path)
    return db_path


@pytest.fixture
def league_dir(tmp_path):
    d = tmp_path / "checkpoints" / "league"
    d.mkdir(parents=True)
    return d


class TestOpponentEntry:
    def test_from_db_row_with_new_fields(self, tmp_path):
        db_path = str(tmp_path / "entry.db")
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """INSERT INTO league_entries
               (display_name, flavour_facts, architecture, model_params,
                checkpoint_path, created_epoch, role, status)
               VALUES ('Takeshi', '[]', 'resnet', '{}', '/p', 10,
                       'frontier_static', 'active')"""
        )
        conn.commit()
        row = conn.execute("SELECT * FROM league_entries WHERE id = 1").fetchone()
        conn.close()
        entry = OpponentEntry.from_db_row(row)
        assert entry.role is Role.FRONTIER_STATIC
        assert entry.status is EntryStatus.ACTIVE
        assert entry.parent_entry_id is None
        assert entry.protection_remaining == 0


class TestOpponentStoreBasics:
    def test_add_entry_with_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.RECENT_FIXED)
        assert entry.role is Role.RECENT_FIXED
        assert entry.status is EntryStatus.ACTIVE

    def test_list_by_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)
        store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)
        assert len(store.list_by_role(Role.FRONTIER_STATIC)) == 1
        assert len(store.list_by_role(Role.DYNAMIC)) == 1

    def test_retire_entry_logs_transition(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        store.retire_entry(entry.id, "evicted")
        # Entry should now be retired
        active = store.list_entries()
        assert len(active) == 0
        # Transition logged
        conn = sqlite3.connect(store_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM league_transitions WHERE entry_id = ?", (entry.id,)
        ).fetchall()
        conn.close()
        assert len(rows) >= 1
        assert rows[-1]["to_status"] == "retired"

    def test_retire_entry_does_not_delete_checkpoint(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        ckpt = Path(entry.checkpoint_path)
        assert ckpt.exists()
        store.retire_entry(entry.id, "evicted")
        assert ckpt.exists()  # file NOT deleted

    def test_clone_entry(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        source = store.add_entry(model, "resnet", {"h": 16}, epoch=5, role=Role.RECENT_FIXED)
        clone = store.clone_entry(source.id, Role.DYNAMIC, "promoted")
        assert clone.role is Role.DYNAMIC
        assert clone.parent_entry_id == source.id
        assert clone.architecture == source.architecture
        assert clone.model_params == source.model_params
        # Separate checkpoint file
        assert clone.checkpoint_path != source.checkpoint_path
        assert Path(clone.checkpoint_path).exists()
        assert Path(source.checkpoint_path).exists()

    def test_update_role(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        entry = store.add_entry(model, "resnet", {}, epoch=1, role=Role.UNASSIGNED)
        store.update_role(entry.id, Role.FRONTIER_STATIC, "bootstrap")
        updated = store.list_by_role(Role.FRONTIER_STATIC)
        assert len(updated) == 1
        assert updated[0].id == entry.id


class TestOpponentStoreTransaction:
    def test_transaction_commits_on_exit(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with store.transaction():
            store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        assert len(store.list_entries()) == 1

    def test_transaction_rolls_back_on_exception(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
                raise RuntimeError("abort")
        assert len(store.list_entries()) == 0


class TestRecordResultUpdates:
    def test_record_result_updates_last_match_at(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        b = store.add_entry(model, "resnet", {}, epoch=2, role=Role.FRONTIER_STATIC)
        store.record_result(epoch=1, learner_id=a.id, opponent_id=b.id,
                            wins=3, losses=1, draws=0)
        updated_a = store.list_entries()[0] if store.list_entries()[0].id == a.id else store.list_entries()[1]
        assert updated_a.last_match_at is not None

    def test_record_result_decrements_protection(self, store_db, league_dir):
        store = OpponentStore(store_db, str(league_dir))
        model = torch.nn.Linear(10, 10)
        with store.transaction():
            a = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
            store._conn.execute(
                "UPDATE league_entries SET protection_remaining = 5 WHERE id = ?",
                (a.id,),
            )
        store.record_result(epoch=1, learner_id=a.id, opponent_id=a.id,
                            wins=1, losses=0, draws=0)
        conn = sqlite3.connect(store_db)
        row = conn.execute(
            "SELECT protection_remaining FROM league_entries WHERE id = ?", (a.id,)
        ).fetchone()
        conn.close()
        assert row[0] == 4


class TestEloCalculation:
    def test_equal_elo_win(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=1.0, k=32)
        assert abs(new_a - 1016.0) < 0.1
        assert abs(new_b - 984.0) < 0.1

    def test_draw_against_equal(self):
        new_a, new_b = compute_elo_update(1000.0, 1000.0, result=0.5, k=32)
        assert abs(new_a - 1000.0) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_opponent_store.py -v --no-header 2>&1 | tail -5`
Expected: ImportError -- module doesn't exist.

- [ ] **Step 3: Create opponent_store.py**

Create `keisei/training/opponent_store.py` by:

1. Copy all contents of `keisei/training/league.py`.
2. Add `import shutil` at the module level (top of file, not inside methods).
3. Add the `Role` and `EntryStatus` StrEnums at the top.
4. Extend `OpponentEntry` with the new fields (`role`, `status`, `parent_entry_id`, `lineage_group`, `protection_remaining`, `last_match_at`). **All new fields must have defaults** so existing test code constructing OpponentEntry with only the original 10 fields doesn't break:
```python
    role: Role = Role.UNASSIGNED
    status: EntryStatus = EntryStatus.ACTIVE
    parent_entry_id: int | None = None
    lineage_group: str | None = None
    protection_remaining: int = 0
    last_match_at: str | None = None
```
Update `from_db_row` to coerce via `Role(row["role"])` and `EntryStatus(row["status"])`.
5. Rename `OpponentPool` to `OpponentStore`.
6. In the constructor, change `self._lock = threading.Lock()` to `self._lock = threading.RLock()` and initialize `self._in_transaction = False`.
7. Remove `_evict_if_needed` and `_delete_entry` methods entirely.
8. Update `add_entry` to accept a `role` parameter (default `Role.UNASSIGNED`) and insert it into the DB.
9. Add `transaction()` context manager (see below).
10. Add `clone_entry(source_entry_id, new_role, reason)` -- copies checkpoint file, creates new DB row with lineage (see below).
11. Add `retire_entry(entry_id, reason)` -- sets status=retired, calls `log_transition`.
12. Add `update_role(entry_id, new_role, reason)` -- changes role, calls `log_transition`.
13. Add `list_by_role(role)` -- `SELECT * FROM league_entries WHERE role = ? AND status = 'active'`.
14. Add `list_entries()` -- filters to `status = 'active'` only.
15. Add `log_transition(entry_id, from_role, to_role, from_status, to_status, reason)`.
16. Add `decrement_protection(entry_id)` -- decrements protection_remaining by 1, floor 0.
17. Update `record_result` to also update `last_match_at` and decrement `protection_remaining` on both participants.
18. Update `load_opponent` to use `map_location="cpu"` then `.to(device)`.
19. Remove `OpponentSampler` class (it's replaced by MatchScheduler).
20. Add `is_bootstrapped()` and `set_bootstrapped()` methods:
```python
    def is_bootstrapped(self) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT bootstrapped FROM league_meta WHERE id = 1").fetchone()
            return bool(row and row[0] == 1)

    def set_bootstrapped(self) -> None:
        with self._lock:
            self._conn.execute("UPDATE league_meta SET bootstrapped = 1 WHERE id = 1")
            if not self._in_transaction:
                self._conn.commit()
```
21. Keep `pin(entry_id)` and `unpin(entry_id)` methods from OpponentPool -- they are used by DemonstratorRunner to protect entries during demo matches. They remain as-is (in-memory `_pinned` set).

**IMPORTANT: All store mutating methods (add_entry, clone_entry, retire_entry, update_role, log_transition, record_result, update_elo, decrement_protection, set_bootstrapped) do NOT call `self._conn.commit()` when inside a transaction. They each check `self._in_transaction` and only commit if False. The `transaction()` context manager is the only thing that controls commit timing when active. Public query methods (list_entries, list_by_role, load_opponent, is_bootstrapped, etc.) and transaction() itself acquire `_lock`.**

Key implementation for `transaction()`. Uses RLock so store methods can be called inside `transaction()` without deadlock:

```python
from contextlib import contextmanager

@contextmanager
def transaction(self):
    """Atomic multi-operation context. Holds lock, defers commit."""
    with self._lock:
        self._in_transaction = True
        try:
            yield
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._in_transaction = False
```

Key implementation for `clone_entry`. Note: clone_entry does NOT acquire `_lock` directly -- callers must use `transaction()` for multi-step atomicity or the method auto-commits when called standalone (via the `_in_transaction` check):

```python
def clone_entry(self, source_entry_id: int, new_role: Role, reason: str) -> OpponentEntry:
    """Clone an entry: copy checkpoint file, create new DB row with lineage."""
    with self._lock:
        source = self._get_entry(source_entry_id)
        if source is None:
            raise ValueError(f"Source entry {source_entry_id} not found")

        # Copy checkpoint file
        src_path = Path(source.checkpoint_path)
        existing_names = {e.display_name for e in self._list_entries_unlocked()}
        display_name = _generate_display_name(source.created_epoch, existing_names)
        flavour_facts = _generate_flavour_facts(source.created_epoch + source_entry_id)

        cursor = self._conn.execute(
            """INSERT INTO league_entries
               (display_name, flavour_facts, architecture, model_params,
                checkpoint_path, created_epoch, role, status,
                parent_entry_id, lineage_group)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (display_name, json.dumps(flavour_facts), source.architecture,
             json.dumps(source.model_params), "", source.created_epoch,
             new_role, EntryStatus.ACTIVE, source_entry_id,
             source.lineage_group or f"lineage-{source_entry_id}"),
        )
        entry_id = cursor.lastrowid

        dst_path = self.league_dir / f"{source.architecture}_ep{source.created_epoch:05d}_id{entry_id}.pt"
        shutil.copy2(str(src_path), str(dst_path))

        self._conn.execute(
            "UPDATE league_entries SET checkpoint_path = ? WHERE id = ?",
            (str(dst_path), entry_id),
        )

        self.log_transition(entry_id, None, new_role, None, EntryStatus.ACTIVE, reason)

        if not self._in_transaction:
            self._conn.commit()

        entry = self._get_entry(entry_id)
        assert entry is not None
        return entry
```

All other mutating methods follow the same pattern:

```python
def retire_entry(self, entry_id: int, reason: str) -> None:
    with self._lock:
        # ... do SQL work ...
        if not self._in_transaction:
            self._conn.commit()

def update_role(self, entry_id: int, new_role: Role, reason: str) -> None:
    with self._lock:
        # ... do SQL work ...
        if not self._in_transaction:
            self._conn.commit()

def record_result(self, ...) -> None:
    with self._lock:
        # ... do SQL work ...
        if not self._in_transaction:
            self._conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_opponent_store.py -v --no-header 2>&1 | tail -15`
Expected: All pass.

- [ ] **Step 5: Replace league.py with re-export shim**

Replace the contents of `keisei/training/league.py` with a backward-compatibility shim:

```python
"""Backward-compatibility shim -- will be removed in the final cleanup task."""
from keisei.training.opponent_store import (  # noqa: F401
    OpponentEntry,
    OpponentStore as OpponentPool,
    Role,
    EntryStatus,
    compute_elo_update,
)

# OpponentSampler is removed -- use MatchScheduler instead.
# This shim exists so that callers not yet updated (tournament.py,
# demonstrator.py, katago_loop.py) continue to import without error
# until their integration tasks are complete.
```

**IMPORTANT: Do NOT run `uv run pytest tests/` as a full suite until Task 11 is complete. Between Tasks 3 and 10, use targeted test commands only (e.g., `uv run pytest tests/test_opponent_store.py`). The re-export shim in league.py prevents ImportError but some callers reference removed attributes like `max_pool_size` that won't exist until their tasks are done.**

Run: `uv run pytest tests/test_opponent_store.py -v`
Expected: Still pass (this file only imports from opponent_store.py).

- [ ] **Step 6: Commit**

```bash
git add keisei/training/opponent_store.py keisei/training/league.py tests/test_opponent_store.py
git commit -m "feat(league): create OpponentStore with enums, transaction API, and clone_entry"
```

---

### Task 4: Tier Managers

**Files:**
- Create: `keisei/training/tier_managers.py`
- Create: `tests/test_tier_managers.py`

- [ ] **Step 1: Write failing tests for FrontierManager**

Create `tests/test_tier_managers.py`:

```python
"""Tests for tier managers: Frontier, RecentFixed, Dynamic."""

import sqlite3
from pathlib import Path

import pytest
import torch

from keisei.config import FrontierStaticConfig, RecentFixedConfig, DynamicConfig
from keisei.db import init_db
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role, EntryStatus
from keisei.training.tier_managers import (
    FrontierManager,
    RecentFixedManager,
    DynamicManager,
    ReviewOutcome,
)


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "tier.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    return OpponentStore(db_path, str(league_dir))


def _add_entry(store, epoch, role=Role.UNASSIGNED, elo=1000.0):
    model = torch.nn.Linear(10, 10)
    entry = store.add_entry(model, "resnet", {}, epoch=epoch, role=role)
    if elo != 1000.0:
        store.update_elo(entry.id, elo)
    return store._get_entry(entry.id)


class TestFrontierManager:
    def test_get_active(self, store):
        _add_entry(store, 1, role=Role.FRONTIER_STATIC)
        _add_entry(store, 2, role=Role.DYNAMIC)
        mgr = FrontierManager(store, FrontierStaticConfig())
        active = mgr.get_active()
        assert len(active) == 1
        assert active[0].role is Role.FRONTIER_STATIC

    def test_select_initial_spans_elo(self, store):
        entries = []
        for i, elo in enumerate([800, 900, 1000, 1100, 1200, 1300, 1400]):
            entries.append(_add_entry(store, i, elo=elo))
        mgr = FrontierManager(store, FrontierStaticConfig(slots=5))
        selected = mgr.select_initial(entries, count=5)
        assert len(selected) == 5
        elos = sorted(e.elo_rating for e in selected)
        # Should span the range, not cluster at top
        assert elos[0] < 1000
        assert elos[-1] > 1200

    def test_select_initial_underfull(self, store):
        entries = [_add_entry(store, i, elo=1000 + i * 100) for i in range(3)]
        mgr = FrontierManager(store, FrontierStaticConfig(slots=5))
        selected = mgr.select_initial(entries, count=5)
        assert len(selected) == 3  # take all

    def test_review_is_noop_phase1(self, store):
        mgr = FrontierManager(store, FrontierStaticConfig())
        mgr.review(epoch=500)  # should not raise

    def test_is_due_for_review(self, store):
        mgr = FrontierManager(store, FrontierStaticConfig(review_interval_epochs=250))
        assert mgr.is_due_for_review(250)
        assert mgr.is_due_for_review(500)
        assert not mgr.is_due_for_review(100)
        assert not mgr.is_due_for_review(0)
```

- [ ] **Step 2: Write failing tests for RecentFixedManager**

Append to `tests/test_tier_managers.py`:

```python
class TestRecentFixedManager:
    def test_admit_creates_recent_fixed_entry(self, store):
        mgr = RecentFixedManager(store, RecentFixedConfig())
        model = torch.nn.Linear(10, 10)
        entry = mgr.admit(model, "resnet", {}, epoch=1)
        assert entry.role is Role.RECENT_FIXED
        assert mgr.count() == 1

    def test_count(self, store):
        mgr = RecentFixedManager(store, RecentFixedConfig(slots=5))
        model = torch.nn.Linear(10, 10)
        for i in range(3):
            mgr.admit(model, "resnet", {}, epoch=i)
        assert mgr.count() == 3

    def test_review_oldest_retire_when_unqualified(self, store):
        """Entry with 0 games should be RETIRE'd."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=0, min_games_for_review=32,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)  # triggers overflow
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.RETIRE
        assert entry.created_epoch == 1  # oldest

    def test_review_oldest_promote_when_qualified(self, store):
        """Entry meeting all criteria should be PROMOTE'd."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=0, min_games_for_review=2,
            min_unique_opponents=2, promotion_margin_elo=25.0,
        ))
        # Add a Dynamic entry so weakest_elo() has a value
        _add_entry(store, 0, role=Role.DYNAMIC, elo=900)

        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        oldest = store.list_by_role(Role.RECENT_FIXED)[0]  # epoch=1

        # Seed qualifying match data: 2 games against 2 distinct opponents
        other1 = _add_entry(store, 10, role=Role.FRONTIER_STATIC, elo=1000)
        other2 = _add_entry(store, 11, role=Role.FRONTIER_STATIC, elo=1000)
        store.record_result(epoch=1, learner_id=oldest.id, opponent_id=other1.id,
                            wins=1, losses=0, draws=0)
        store.record_result(epoch=2, learner_id=oldest.id, opponent_id=other2.id,
                            wins=1, losses=0, draws=0)
        # Set Elo high enough to qualify (>= 900 - 25 = 875)
        store.update_elo(oldest.id, 1000.0)

        mgr.admit(model, "resnet", {}, epoch=3)  # triggers overflow review
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.PROMOTE
        assert entry.id == oldest.id

    def test_review_oldest_promotes_when_dynamic_empty(self, store):
        """When Dynamic tier is empty, promotion should always pass the Elo check."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=1, soft_overflow=0, min_games_for_review=0, min_unique_opponents=0,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)  # triggers overflow
        outcome, entry = mgr.review_oldest()
        assert outcome is ReviewOutcome.PROMOTE

    def test_review_oldest_delay_when_undercalibrated(self, store):
        """Entry with < min_games should DELAY if soft overflow remains."""
        mgr = RecentFixedManager(store, RecentFixedConfig(
            slots=2, soft_overflow=1, min_games_for_review=32,
        ))
        model = torch.nn.Linear(10, 10)
        mgr.admit(model, "resnet", {}, epoch=1)
        mgr.admit(model, "resnet", {}, epoch=2)
        mgr.admit(model, "resnet", {}, epoch=3)  # queue=3, slots+overflow=3 -> no review yet
        assert mgr.count() == 3
        # Only at 4 entries would review trigger (slots=2, overflow=1 -> threshold=3)
        mgr.admit(model, "resnet", {}, epoch=4)  # queue=4 > 3
        outcome, entry = mgr.review_oldest()
        # oldest has 0 games < 32, but soft overflow used = 4-2=2 > soft_overflow=1
        # so no more delay capacity -> RETIRE
        assert outcome is ReviewOutcome.RETIRE
```

- [ ] **Step 3: Write failing tests for DynamicManager**

Append to `tests/test_tier_managers.py`:

```python
class TestDynamicManager:
    def test_admit_clones_entry(self, store):
        source = _add_entry(store, 1, role=Role.RECENT_FIXED)
        mgr = DynamicManager(store, DynamicConfig())
        entry = mgr.admit(source)
        assert entry.role is Role.DYNAMIC
        assert entry.parent_entry_id == source.id
        assert entry.protection_remaining == 24

    def test_evict_weakest_skips_protected(self, store):
        """Protected entries should not be evicted, even if they have the lowest Elo."""
        mgr = DynamicManager(store, DynamicConfig(
            slots=3, protection_matches=0, min_games_before_eviction=0,
        ))
        e1 = _add_entry(store, 1, role=Role.DYNAMIC, elo=800)  # weakest
        e2 = _add_entry(store, 2, role=Role.DYNAMIC, elo=1000)
        e3 = _add_entry(store, 3, role=Role.DYNAMIC, elo=1200)
        # Protect the weakest entry
        with store.transaction():
            store._conn.execute(
                "UPDATE league_entries SET protection_remaining = 10 WHERE id = ?", (e1.id,)
            )
        evicted = mgr.evict_weakest()
        assert evicted is not None
        assert evicted.id == e2.id  # e2 is the weakest ELIGIBLE entry (e1 is protected)
        # Verify e1 is still active
        remaining = store.list_by_role(Role.DYNAMIC)
        assert any(e.id == e1.id for e in remaining), "Protected entry e1 should still be active"

    def test_evict_weakest_picks_lowest_elo(self, store):
        mgr = DynamicManager(store, DynamicConfig(
            slots=2, protection_matches=0, min_games_before_eviction=0,
        ))
        e1 = _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        e2 = _add_entry(store, 2, role=Role.DYNAMIC, elo=1200)
        evicted = mgr.evict_weakest()
        assert evicted is not None
        assert evicted.id == e1.id  # lowest Elo

    def test_evict_weakest_returns_none_when_all_protected(self, store):
        """When all entries are protected, evict_weakest should return None."""
        mgr = DynamicManager(store, DynamicConfig(
            slots=2, protection_matches=0, min_games_before_eviction=100,
        ))
        _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        _add_entry(store, 2, role=Role.DYNAMIC, elo=1200)
        # Both have 0 games < 100 min_games_before_eviction
        evicted = mgr.evict_weakest()
        assert evicted is None

    def test_is_full(self, store):
        mgr = DynamicManager(store, DynamicConfig(slots=2))
        assert not mgr.is_full()
        _add_entry(store, 1, role=Role.DYNAMIC)
        _add_entry(store, 2, role=Role.DYNAMIC)
        assert mgr.is_full()

    def test_weakest_elo_returns_none_when_all_protected(self, store):
        mgr = DynamicManager(store, DynamicConfig(
            slots=2, protection_matches=24, min_games_before_eviction=40,
        ))
        _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        # Entry has protection=0 by default but games_played=0 < 40
        assert mgr.weakest_elo() is None

    def test_admit_returns_none_when_full_and_all_protected(self, store):
        """When Dynamic tier is full and all entries protected, admit returns None."""
        mgr = DynamicManager(store, DynamicConfig(
            slots=1, protection_matches=0, min_games_before_eviction=100,
        ))
        _add_entry(store, 1, role=Role.DYNAMIC, elo=800)
        source = _add_entry(store, 2, role=Role.RECENT_FIXED)
        result = mgr.admit(source)
        assert result is None

    def test_training_enabled_raises(self, store):
        with pytest.raises((AssertionError, NotImplementedError)):
            DynamicManager(store, DynamicConfig(training_enabled=True))
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_tier_managers.py -v --no-header 2>&1 | tail -5`
Expected: ImportError -- module doesn't exist.

- [ ] **Step 5: Implement tier_managers.py**

Create `keisei/training/tier_managers.py` with all three manager classes and the `ReviewOutcome` enum. Each manager takes `store: OpponentStore` and its config dataclass. Key details:

- `FrontierManager.select_initial`: sort by Elo, pick at evenly spaced indices (quintile), tiebreak by `games_played` desc then `created_epoch` asc.
- `FrontierManager.review`: no-op. Log at INFO level: "Phase 1: frontier review is a no-op".
- `RecentFixedManager.__init__`: takes an optional `weakest_elo_fn: Callable[[], float | None]` parameter (or receives a reference to `DynamicManager`). This is set by TieredPool after construction. Used for the promotion Elo check.
- `RecentFixedManager.admit`: calls `store.add_entry(role=Role.RECENT_FIXED)`, returns the entry.
- `RecentFixedManager.review_oldest`: gets oldest active recent_fixed entry, checks promotion criteria, returns `(ReviewOutcome, entry)`. For the Elo check, guard against `weakest_elo()` returning None:
```python
        floor_elo = self._weakest_elo_fn()  # may be None if Dynamic tier is empty
        if floor_elo is None:
            elo_qualified = True  # No Dynamic entries yet -- always promote
        else:
            elo_qualified = entry.elo_rating >= floor_elo - self.config.promotion_margin_elo
```
- `RecentFixedManager.get_unique_opponent_count(entry_id)`: queries `league_results` for distinct opponent count in both seats.
- `DynamicManager.__init__`: asserts `not config.training_enabled`.
- `DynamicManager.admit`: inside `store.transaction()`, calls `store.clone_entry`, then sets `protection_remaining`. If `is_full()` and `evict_weakest()` returns None (all protected), log a warning and return None:
```python
    def admit(self, source_entry: OpponentEntry) -> OpponentEntry | None:
        with self.store.transaction():
            if self.is_full():
                evicted = self.evict_weakest()
                if evicted is None:
                    logger.warning("Dynamic tier full and all entries protected -- cannot admit")
                    return None
            clone = self.store.clone_entry(source_entry.id, Role.DYNAMIC, "promoted from recent_fixed")
            self.store._conn.execute(
                "UPDATE league_entries SET protection_remaining = ? WHERE id = ?",
                (self.config.protection_matches, clone.id),
            )
            return clone
```
- `DynamicManager.evict_weakest`: finds lowest-Elo eligible entry, calls `store.retire_entry`. Returns None if no eligible entries.
- `DynamicManager.weakest_elo`: returns Elo of weakest eligible or None.

**Logging requirements:** All tier managers must include INFO-level logging:
  - `RecentFixedManager.review_oldest`: log outcome, entry_id, games_played, unique_opponents at review time
  - `DynamicManager.admit`: log entry_id, source_entry_id, protection_remaining
  - `DynamicManager.evict_weakest`: log evicted entry_id, Elo, reason
  - `FrontierManager.review`: log "Phase 1: frontier review is a no-op"

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_tier_managers.py -v --no-header 2>&1 | tail -15`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/tier_managers.py tests/test_tier_managers.py
git commit -m "feat(league): add FrontierManager, RecentFixedManager, DynamicManager"
```

---

### Task 5: MatchScheduler

**Files:**
- Create: `keisei/training/match_scheduler.py`
- Create: `tests/test_match_scheduler.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_match_scheduler.py`:

```python
"""Tests for MatchScheduler -- role-weighted opponent selection."""

from collections import Counter

import pytest

from keisei.config import MatchSchedulerConfig
from keisei.training.opponent_store import OpponentEntry, Role, EntryStatus
from keisei.training.match_scheduler import MatchScheduler


def _make_entry(id: int, role: Role, elo: float = 1000.0) -> OpponentEntry:
    return OpponentEntry(
        id=id, display_name=f"e{id}", architecture="resnet",
        model_params={}, checkpoint_path=f"/p/{id}.pt", elo_rating=elo,
        created_epoch=id, games_played=0, created_at="2026-01-01",
        flavour_facts=[], role=role, status=EntryStatus.ACTIVE,
        parent_entry_id=None, lineage_group=None,
        protection_remaining=0, last_match_at=None,
    )


@pytest.fixture
def full_entries():
    entries = {
        Role.FRONTIER_STATIC: [_make_entry(i, Role.FRONTIER_STATIC) for i in range(1, 6)],
        Role.RECENT_FIXED: [_make_entry(i, Role.RECENT_FIXED) for i in range(6, 11)],
        Role.DYNAMIC: [_make_entry(i, Role.DYNAMIC) for i in range(11, 21)],
    }
    return entries


class TestSampleForLearner:
    def test_respects_tier_ratios(self, full_entries):
        sched = MatchScheduler(MatchSchedulerConfig())
        counts = Counter()
        for _ in range(1000):
            entry = sched.sample_for_learner(full_entries)
            counts[entry.role] += 1
        # 50% dynamic, 30% frontier, 20% recent (with tolerance)
        assert 400 < counts[Role.DYNAMIC] < 600
        assert 200 < counts[Role.FRONTIER_STATIC] < 400
        assert 100 < counts[Role.RECENT_FIXED] < 300

    def test_empty_tier_redistributes(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [_make_entry(1, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [_make_entry(2, Role.RECENT_FIXED)],
            Role.DYNAMIC: [],  # empty
        }
        for _ in range(100):
            entry = sched.sample_for_learner(entries)
            assert entry.role != Role.DYNAMIC

    def test_single_tier_available(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [],
            Role.RECENT_FIXED: [_make_entry(1, Role.RECENT_FIXED)],
            Role.DYNAMIC: [],
        }
        entry = sched.sample_for_learner(entries)
        assert entry.role is Role.RECENT_FIXED

    def test_all_empty_raises(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {Role.FRONTIER_STATIC: [], Role.RECENT_FIXED: [], Role.DYNAMIC: []}
        with pytest.raises(ValueError, match="[Nn]o.*entries"):
            sched.sample_for_learner(entries)


class TestGenerateRound:
    def test_produces_all_pairs(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = [_make_entry(i, Role.DYNAMIC) for i in range(5)]
        pairings = sched.generate_round(entries)
        # 5 entries = 10 unique pairs
        assert len(pairings) == 10
        # All pairs are distinct
        pair_ids = {(min(a.id, b.id), max(a.id, b.id)) for a, b in pairings}
        assert len(pair_ids) == 10

    def test_no_self_matches(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = [_make_entry(i, Role.DYNAMIC) for i in range(5)]
        pairings = sched.generate_round(entries)
        for a, b in pairings:
            assert a.id != b.id

    def test_single_entry_produces_no_pairings(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = [_make_entry(1, Role.DYNAMIC)]
        pairings = sched.generate_round(entries)
        assert len(pairings) == 0

    def test_mixed_roles_all_paired(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = [
            _make_entry(1, Role.FRONTIER_STATIC),
            _make_entry(2, Role.RECENT_FIXED),
            _make_entry(3, Role.DYNAMIC),
        ]
        pairings = sched.generate_round(entries)
        assert len(pairings) == 3  # 3 choose 2


class TestEffectiveRatios:
    def test_full_pool_returns_config_ratios(self, full_entries):
        sched = MatchScheduler(MatchSchedulerConfig())
        ratios = sched.effective_ratios(full_entries)
        assert abs(ratios[Role.DYNAMIC] - 0.50) < 0.01
        assert abs(ratios[Role.FRONTIER_STATIC] - 0.30) < 0.01
        assert abs(ratios[Role.RECENT_FIXED] - 0.20) < 0.01

    def test_empty_tier_redistributes(self):
        sched = MatchScheduler(MatchSchedulerConfig())
        entries = {
            Role.FRONTIER_STATIC: [_make_entry(1, Role.FRONTIER_STATIC)],
            Role.RECENT_FIXED: [_make_entry(2, Role.RECENT_FIXED)],
            Role.DYNAMIC: [],
        }
        ratios = sched.effective_ratios(entries)
        assert ratios[Role.DYNAMIC] == 0.0
        assert abs(ratios[Role.FRONTIER_STATIC] + ratios[Role.RECENT_FIXED] - 1.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_match_scheduler.py -v --no-header 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement match_scheduler.py**

Create `keisei/training/match_scheduler.py`:

```python
"""MatchScheduler -- role-weighted opponent selection for the tiered pool."""

from __future__ import annotations

import random

from keisei.config import MatchSchedulerConfig
from keisei.training.opponent_store import OpponentEntry, Role


class MatchScheduler:
    def __init__(self, config: MatchSchedulerConfig) -> None:
        self.config = config
        # Phase 4: add H2H repeat penalty here

    def sample_for_learner(
        self, entries_by_role: dict[Role, list[OpponentEntry]],
    ) -> OpponentEntry:
        ratios = self.effective_ratios(entries_by_role)
        non_empty = {r: w for r, w in ratios.items() if w > 0}
        if not non_empty:
            raise ValueError("No entries available in any tier")
        roles = list(non_empty.keys())
        weights = [non_empty[r] for r in roles]
        chosen_role = random.choices(roles, weights=weights, k=1)[0]
        return random.choice(entries_by_role[chosen_role])

    def generate_round(
        self, entries: list[OpponentEntry],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Generate all N*(N-1)/2 pairings for a full round-robin round.
        
        Each pair plays tournament_games_per_pair games (default 3 — best-of-3).
        With 20 entries: 190 pairings × 3 games = 570 games per round.
        """
        pairings = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                pairings.append((entries[i], entries[j]))
        random.shuffle(pairings)  # randomize order to avoid position bias
        return pairings

    def effective_ratios(
        self, entries_by_role: dict[Role, list[OpponentEntry]],
    ) -> dict[Role, float]:
        raw = {
            Role.DYNAMIC: self.config.learner_dynamic_ratio,
            Role.FRONTIER_STATIC: self.config.learner_frontier_ratio,
            Role.RECENT_FIXED: self.config.learner_recent_ratio,
        }
        non_empty = {r: w for r, w in raw.items() if entries_by_role.get(r)}
        if not non_empty:
            return {r: 0.0 for r in raw}
        total = sum(non_empty.values())
        result = {}
        for role in raw:
            if role in non_empty:
                result[role] = non_empty[role] / total
            else:
                result[role] = 0.0
        return result

```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_match_scheduler.py -v --no-header 2>&1 | tail -10`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/match_scheduler.py tests/test_match_scheduler.py
git commit -m "feat(league): add MatchScheduler with role-weighted sampling"
```

---

### Task 6: TieredPool Orchestrator

**Files:**
- Create: `keisei/training/tiered_pool.py`
- Create: `tests/test_tiered_pool.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_tiered_pool.py`:

```python
"""Tests for TieredPool -- the orchestrator."""

import sqlite3

import pytest
import torch

from keisei.config import LeagueConfig
from keisei.db import init_db
from keisei.training.opponent_store import OpponentStore, Role, EntryStatus
from keisei.training.tiered_pool import TieredPool


@pytest.fixture
def pool_setup(tmp_path):
    db_path = str(tmp_path / "pool.db")
    init_db(db_path)
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    store = OpponentStore(db_path, str(league_dir))
    config = LeagueConfig()
    pool = TieredPool(store, config)
    return pool, store, db_path


class TestSnapshotLearner:
    def test_creates_recent_fixed_entry(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        entry = pool.snapshot_learner(model, "resnet", {}, epoch=1)
        assert entry.role is Role.RECENT_FIXED
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 1


class TestEntriesByRole:
    def test_groups_correctly(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)
        store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)
        by_role = pool.entries_by_role()
        assert len(by_role[Role.FRONTIER_STATIC]) == 1
        assert len(by_role[Role.RECENT_FIXED]) == 1
        assert len(by_role[Role.DYNAMIC]) == 1


class TestBootstrapFromFlatPool:
    def test_full_pool_20_entries(self, pool_setup):
        pool, store, db_path = pool_setup
        model = torch.nn.Linear(10, 10)
        entries = []
        for i in range(20):
            e = store.add_entry(model, "resnet", {}, epoch=i, role=Role.UNASSIGNED)
            store.update_elo(e.id, 800 + i * 20)
            entries.append(e)
        pool.bootstrap_from_flat_pool()
        fs = store.list_by_role(Role.FRONTIER_STATIC)
        rf = store.list_by_role(Role.RECENT_FIXED)
        dy = store.list_by_role(Role.DYNAMIC)
        assert len(fs) == 5
        assert len(rf) == 5
        assert len(dy) == 10
        # Check bootstrapped flag
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT bootstrapped FROM league_meta WHERE id = 1").fetchone()
        conn.close()
        assert row[0] == 1

    def test_small_pool_8_entries(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        for i in range(8):
            e = store.add_entry(model, "resnet", {}, epoch=i, role=Role.UNASSIGNED)
            store.update_elo(e.id, 800 + i * 50)
        pool.bootstrap_from_flat_pool()
        fs = store.list_by_role(Role.FRONTIER_STATIC)
        rf = store.list_by_role(Role.RECENT_FIXED)
        dy = store.list_by_role(Role.DYNAMIC)
        total_active = len(fs) + len(rf) + len(dy)
        assert total_active == 8
        assert len(fs) >= 1
        assert len(rf) >= 1
        assert len(dy) >= 1

    def test_bootstrap_is_idempotent(self, pool_setup):
        pool, store, db_path = pool_setup
        model = torch.nn.Linear(10, 10)
        for i in range(5):
            store.add_entry(model, "resnet", {}, epoch=i, role=Role.UNASSIGNED)
        pool.bootstrap_from_flat_pool()
        # Second call should be no-op
        pool.bootstrap_from_flat_pool()
        # Still only 5 active entries
        assert len(store.list_entries()) == 5


class TestFullLifecycle:
    """Integration test: snapshot -> Recent Fixed -> overflow -> review -> promote -> eviction."""

    def test_lifecycle_snapshot_to_eviction(self, pool_setup):
        pool, store, db_path = pool_setup
        config = pool.config
        model = torch.nn.Linear(10, 10)

        # 1. Fill Recent Fixed (5 slots)
        for i in range(1, 6):
            entry = pool.snapshot_learner(model, "resnet", {}, epoch=i)
            assert entry.role is Role.RECENT_FIXED
        assert len(store.list_by_role(Role.RECENT_FIXED)) == 5

        # 2. Seed some Dynamic entries directly for the promotion threshold
        for i in range(10):
            e = store.add_entry(model, "resnet", {}, epoch=100 + i, role=Role.DYNAMIC)
            store.update_elo(e.id, 900 + i * 10)

        # 3. Next snapshot triggers overflow. Oldest has 0 games -> RETIRE
        pool.snapshot_learner(model, "resnet", {}, epoch=7)
        # Oldest (epoch=1) should be retired since it has 0 games
        rf = store.list_by_role(Role.RECENT_FIXED)
        assert all(e.created_epoch != 1 for e in rf)

        # 4. Verify Dynamic eviction works when tier is full
        dynamic_before = store.list_by_role(Role.DYNAMIC)
        assert len(dynamic_before) == 10


class TestListAllActive:
    def test_returns_all_tiers(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        store.add_entry(model, "resnet", {}, epoch=1, role=Role.FRONTIER_STATIC)
        store.add_entry(model, "resnet", {}, epoch=2, role=Role.RECENT_FIXED)
        store.add_entry(model, "resnet", {}, epoch=3, role=Role.DYNAMIC)
        assert len(pool.list_all_active()) == 3

    def test_excludes_retired(self, pool_setup):
        pool, store, _ = pool_setup
        model = torch.nn.Linear(10, 10)
        e = store.add_entry(model, "resnet", {}, epoch=1, role=Role.DYNAMIC)
        store.retire_entry(e.id, "test")
        assert len(pool.list_all_active()) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tiered_pool.py -v --no-header 2>&1 | tail -5`
Expected: ImportError.

- [ ] **Step 3: Implement tiered_pool.py**

Create `keisei/training/tiered_pool.py`:

```python
"""TieredPool -- orchestrator for the tiered opponent league."""

from __future__ import annotations

import logging

from keisei.config import LeagueConfig
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role
from keisei.training.tier_managers import (
    DynamicManager,
    FrontierManager,
    RecentFixedManager,
    ReviewOutcome,
)

logger = logging.getLogger(__name__)


class TieredPool:
    def __init__(self, store: OpponentStore, config: LeagueConfig) -> None:
        self.store = store
        self.config = config
        self.frontier_manager = FrontierManager(store, config.frontier)
        self.recent_manager = RecentFixedManager(store, config.recent)
        self.dynamic_manager = DynamicManager(store, config.dynamic)
        # Wire up the weakest_elo function for promotion checks
        self.recent_manager.set_weakest_elo_fn(self.dynamic_manager.weakest_elo)

    def snapshot_learner(
        self, model, arch: str, params: dict, epoch: int,
    ) -> OpponentEntry:
        entry = self.recent_manager.admit(model, arch, params, epoch)
        if self.recent_manager.count() > self.config.recent.slots + self.config.recent.soft_overflow:
            outcome, oldest = self.recent_manager.review_oldest()
            if outcome is ReviewOutcome.PROMOTE:
                with self.store.transaction():
                    self.dynamic_manager.admit(oldest)
                    self.store.retire_entry(oldest.id, "promoted to dynamic")
            elif outcome is ReviewOutcome.RETIRE:
                self.store.retire_entry(oldest.id, "did not qualify for dynamic")
            # DELAY: leave it, will be reviewed next cycle
        return entry

    def entries_by_role(self) -> dict[Role, list[OpponentEntry]]:
        return {
            Role.FRONTIER_STATIC: self.store.list_by_role(Role.FRONTIER_STATIC),
            Role.RECENT_FIXED: self.store.list_by_role(Role.RECENT_FIXED),
            Role.DYNAMIC: self.store.list_by_role(Role.DYNAMIC),
        }

    def list_all_active(self) -> list[OpponentEntry]:
        return self.store.list_entries()

    def on_epoch_end(self, epoch: int) -> None:
        if self.frontier_manager.is_due_for_review(epoch):
            self.frontier_manager.review(epoch)

    def bootstrap_from_flat_pool(self) -> None:
        """One-time migration from flat pool. Flag check is inside the transaction for safety."""
        with self.store.transaction():
            if self.store.is_bootstrapped():
                logger.info("Bootstrap already complete, skipping")
                return

            entries = [e for e in self.store.list_entries() if e.role is Role.UNASSIGNED]
            if not entries:
                self.store.set_bootstrapped()
                return

            n = len(entries)

            # --- Assignment order: Recent Fixed FIRST, then Frontier, then Dynamic ---

            # Recent Fixed: most recent N by epoch
            n_recent = max(1, round(n * 0.25)) if n >= 3 else (1 if n >= 2 else 0)
            entries_by_epoch = sorted(entries, key=lambda e: e.created_epoch, reverse=True)
            recent_selected = entries_by_epoch[:n_recent]
            recent_ids = {e.id for e in recent_selected}
            for e in recent_selected:
                self.store.update_role(e.id, Role.RECENT_FIXED, "bootstrap: recent fixed")

            # Frontier Static: quintile Elo spread from REMAINING entries
            remaining_after_recent = [e for e in entries if e.id not in recent_ids]
            n_frontier = max(1, round(n * 0.25)) if n >= 3 else (1 if n >= 1 else 0)
            frontier_selected = self.frontier_manager.select_initial(remaining_after_recent, count=n_frontier)
            frontier_ids = {e.id for e in frontier_selected}
            for e in frontier_selected:
                self.store.update_role(e.id, Role.FRONTIER_STATIC, "bootstrap: frontier static")

            # Dynamic: next by Elo from remainder
            assigned_ids = recent_ids | frontier_ids
            n_dynamic = n - len(recent_selected) - len(frontier_selected)
            dynamic_candidates = sorted(
                [e for e in entries if e.id not in assigned_ids],
                key=lambda e: e.elo_rating, reverse=True,
            )
            for e in dynamic_candidates[:n_dynamic]:
                self.store.update_role(e.id, Role.DYNAMIC, "bootstrap: dynamic")

            # Retire remainder
            dynamic_ids = {e.id for e in dynamic_candidates[:n_dynamic]}
            for e in entries:
                if e.id not in assigned_ids and e.id not in dynamic_ids:
                    self.store.retire_entry(e.id, "bootstrap: excess entry retired")

            self.store.set_bootstrapped()

        logger.info(
            "Bootstrap complete: %d recent, %d frontier, %d dynamic, %d retired",
            len(recent_selected), len(frontier_selected),
            min(n_dynamic, len(dynamic_candidates)),
            max(0, len(entries) - len(recent_selected) - len(frontier_selected) - n_dynamic),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tiered_pool.py -v --no-header 2>&1 | tail -10`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/tiered_pool.py tests/test_tiered_pool.py
git commit -m "feat(league): add TieredPool orchestrator with bootstrap and snapshot_learner"
```

---

### Task 7: Update KataGoTrainingLoop Integration

**Files:**
- Modify: `keisei/training/katago_loop.py`

- [ ] **Step 1: Update imports**

Replace:
```python
from keisei.training.league import OpponentEntry, OpponentPool, OpponentSampler, compute_elo_update
```
With:
```python
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role, compute_elo_update
from keisei.training.tiered_pool import TieredPool
from keisei.training.match_scheduler import MatchScheduler
```

- [ ] **Step 2: Update __init__ league setup**

Replace the `OpponentPool` / `OpponentSampler` initialization block (lines ~562-582) with:

```python
        if config.league is not None:
            league_dir = str(Path(config.training.checkpoint_dir) / "league")
            self.store = OpponentStore(self.db_path, league_dir)
            self.tiered_pool = TieredPool(self.store, config.league)
            self.scheduler = MatchScheduler(config.league.scheduler)
            # Bootstrap snapshot so pool is never empty
            bootstrap_entry = self.tiered_pool.snapshot_learner(
                self._base_model, config.model.architecture,
                dict(config.model.params), epoch=0,
            )
            self._learner_entry_id = bootstrap_entry.id
            logger.info(
                "Tiered league initialized: %d frontier, %d recent, %d dynamic slots",
                config.league.frontier.slots,
                config.league.recent.slots,
                config.league.dynamic.slots,
            )
```

Update instance variable declarations to match:
```python
        self.store: OpponentStore | None = None
        self.tiered_pool: TieredPool | None = None
        self.scheduler: MatchScheduler | None = None
```

Remove references to `self.pool` and `self.sampler`.

- [ ] **Step 3: Update per-epoch opponent sampling**

Replace `self.sampler.sample()` with `self.scheduler.sample_for_learner(self.tiered_pool.entries_by_role())`.

Replace `self.pool.list_entries()` for cached entries with `self.tiered_pool.list_all_active()`.

Replace `self.sampler.sample_from(self._cached_entries)` with `self.scheduler.sample_for_learner(self.tiered_pool.entries_by_role())`.

Replace `self.pool.load_all_opponents(device=opp_device)` with `self.store.load_all_opponents(device=opp_device)`.

- [ ] **Step 4: Update Elo updates and snapshot calls**

Replace all `self.pool.update_elo(...)` with `self.store.update_elo(...)`.
Replace all `self.pool.record_result(...)` with `self.store.record_result(...)`.
Replace `self.pool._get_entry(...)` with `self.store._get_entry(...)`.

- [ ] **Step 5: Update _rotate_seat and periodic snapshots**

Replace:
```python
self.pool.add_snapshot(self._base_model, ...)
```
With:
```python
self.tiered_pool.snapshot_learner(self._base_model, ...)
```

In `_rotate_seat`, update `self._learner_entry_id` from the returned entry.

- [ ] **Step 6: Add on_epoch_end call**

After the Elo update block but before metrics, add:
```python
                if self.tiered_pool is not None:
                    self.tiered_pool.on_epoch_end(epoch_i)
```

- [ ] **Step 7: Run tests**

Run: `uv run pytest tests/test_katago_loop.py -v --no-header -x 2>&1 | tail -20`
Expected: Tests pass (some existing tests may need import updates -- fix as needed).

- [ ] **Step 8: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "feat(league): integrate TieredPool and MatchScheduler into training loop"
```

---

### Task 8: Update Tournament Runner

**Files:**
- Modify: `keisei/training/tournament.py`
- Modify: `keisei/training/katago_loop.py` (tournament construction)

- [ ] **Step 1: Update imports**

Replace:
```python
from keisei.training.league import OpponentEntry, compute_elo_update
```
With:
```python
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role, compute_elo_update
from keisei.training.match_scheduler import MatchScheduler
```

- [ ] **Step 2: Refactor constructor**

Add `store: OpponentStore` and `scheduler: MatchScheduler` as constructor parameters. Remove `db_path` and `league_dir` parameters (the store owns those). Remove `self._open_connection` method.

```python
    def __init__(
        self,
        store: OpponentStore,
        scheduler: MatchScheduler,
        *,
        device: str = "cuda:1",
        num_envs: int = 64,
        max_ply: int = 512,
        games_per_match: int = 64,
        k_factor: float = 16.0,
        pause_seconds: float = 5.0,
        min_pool_size: int = 3,
    ) -> None:
        self.store = store
        self.scheduler = scheduler
        # ... rest unchanged except remove self.db_path and self.league_dir
```

- [ ] **Step 3: Replace raw SQL with OpponentStore calls**

In `_run_loop`:
- Replace `conn = self._open_connection()` -> remove (store manages connection)
- Replace `self._load_entries(conn)` -> `self.store.list_entries()`
- Replace the old `_generate_round` method and the inner pairing loop with `scheduler.generate_round(entries)`. The tournament main loop becomes:

```python
entries = self.store.list_entries()
if len(entries) < self.min_pool_size:
    self._stop_event.wait(self.pause_seconds * 2)
    continue

pairings = self.scheduler.generate_round(entries)
for entry_a, entry_b in pairings:
    if self._stop_event.is_set():
        break
    try:
        wins_a, wins_b, draws = self._play_match(entry_a, entry_b)
    except Exception:
        logger.exception("Match failed: %s vs %s", entry_a.display_name, entry_b.display_name)
        continue
    if wins_a + wins_b + draws > 0:
        self.store.record_result(...)
        self.store.update_elo(...)
    self._stop_event.wait(self.pause_seconds)
```

The `_play_match` method is updated to play `self.scheduler.config.tournament_games_per_pair` games instead of `self.games_per_match`. The `games_per_match` parameter is removed from the constructor -- it's now controlled by `MatchSchedulerConfig.tournament_games_per_pair`.
- Replace `self._record_result(conn, ...)` -> use `self.store.record_result(...)` and `self.store.update_elo(...)`
- Remove `self._open_connection`, `_load_entries`, `_record_result` methods
- Keep `_play_match`, `_play_batch` unchanged
- Update `_load_model` to use `self.store.load_opponent(entry, device=str(self.device))`

- [ ] **Step 4: Update the katago_loop.py tournament construction**

In `katago_loop.py`, update the tournament construction to pass `store` and `scheduler`:

```python
            if config.league.tournament_enabled and self.dist_ctx.is_main:
                tournament_device = (
                    config.league.tournament_device
                    or config.league.opponent_device
                    or str(self.device)
                )
                self._tournament = LeagueTournament(
                    store=self.store,
                    scheduler=self.scheduler,
                    device=tournament_device,
                    num_envs=config.league.tournament_num_envs,
                    games_per_match=config.league.tournament_games_per_match,
                    k_factor=config.league.tournament_k_factor,
                    pause_seconds=config.league.tournament_pause_seconds,
                    max_ply=config.training.max_ply,
                )
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_tournament.py tests/test_katago_loop.py -v --no-header -x 2>&1 | tail -20`
Expected: Pass.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/tournament.py keisei/training/katago_loop.py
git commit -m "refactor(tournament): use OpponentStore and MatchScheduler instead of raw SQL"
```

---

### Task 9: Update Demonstrator and Remaining Imports

**Files:**
- Modify: `keisei/training/demonstrator.py`
- Modify: any other files with `from keisei.training.league import`

- [ ] **Step 1: Update demonstrator imports**

Replace:
```python
from keisei.training.league import OpponentEntry, OpponentPool
```
With:
```python
from keisei.training.opponent_store import OpponentEntry, OpponentStore
```

Replace all references to `OpponentPool` in the file with `OpponentStore`. The `DemonstratorRunner.__init__` takes `pool: OpponentPool` -- rename to `store: OpponentStore`. Update `self.pool` -> `self.store` throughout.

Replace `self.pool.pin(...)` with `self.store.pin(...)` and `self.pool.unpin(...)` with `self.store.unpin(...)`.

- [ ] **Step 2: Update katago_loop.py demonstrator construction**

In the demonstrator initialization, replace `self.pool` with `self.store`.

- [ ] **Step 3: Search for any remaining league.py imports**

Run: `grep -r "from keisei.training.league" keisei/ tests/ --include="*.py"`
Expected: Only `keisei/training/league.py` itself (the shim).

- [ ] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_demonstrator.py tests/test_katago_loop.py -v --no-header 2>&1 | tail -20`
Expected: Pass.

- [ ] **Step 5: Commit**

```bash
git add keisei/training/demonstrator.py keisei/training/katago_loop.py
git commit -m "refactor(demonstrator): update imports from league.py to opponent_store.py"
```

---

### Task 10: Update Remaining Test Files

**Files:**
- Modify: Any test files that import from `keisei.training.league`

- [ ] **Step 1: Find and fix all broken test imports**

Run: `grep -r "from keisei.training.league" tests/ --include="*.py" -l`

For each file found, update imports:
- `OpponentPool` -> `OpponentStore` (from `keisei.training.opponent_store`)
- `OpponentSampler` -> removed (use `MatchScheduler` from `keisei.training.match_scheduler`)
- `OpponentEntry` -> from `keisei.training.opponent_store`
- `compute_elo_update` -> from `keisei.training.opponent_store`

Also update any test code that calls `OpponentPool(...)` to use `OpponentStore(...)` (constructor no longer takes `max_pool_size`).

- [ ] **Step 2: Fix test code that relied on removed features**

Tests that call `pool.add_snapshot(...)` -> use `store.add_entry(model, arch, params, epoch, role=Role.UNASSIGNED)`.
Tests that relied on `_evict_if_needed` -> these are no longer relevant (eviction is handled by tier managers).
Tests that used `OpponentSampler` -> either remove (replaced by MatchScheduler tests) or rewrite using MatchScheduler.

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --no-header 2>&1 | tail -30`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "fix(tests): update all test imports from league.py to new tiered pool modules"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run full test suite with coverage**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All pass, no import errors.

- [ ] **Step 2: Verify no references to deleted league.py (other than the shim itself)**

Run: `grep -r "from keisei.training.league" keisei/ tests/ --include="*.py"`
Expected: Only `keisei/training/league.py` (the shim). No callers should still use it.

- [ ] **Step 3: Verify the new modules are importable**

```bash
uv run python -c "
from keisei.training.opponent_store import OpponentStore, OpponentEntry, Role, EntryStatus
from keisei.training.tier_managers import FrontierManager, RecentFixedManager, DynamicManager, ReviewOutcome
from keisei.training.tiered_pool import TieredPool
from keisei.training.match_scheduler import MatchScheduler
print('All imports OK')
"
```

- [ ] **Step 4: Final commit if any fixups needed**

```bash
git add -A
git status
# Only commit if there are changes
git diff --cached --stat && git commit -m "fix: final fixups for tiered pool Phase 1"
```

---

### Task 12: Delete Old Files (Final Cleanup)

**Files:**
- Delete: `keisei/training/league.py` (the re-export shim)
- Delete: `tests/test_league.py`

This task runs AFTER all imports have been updated and the full test suite passes.

- [ ] **Step 1: Verify no remaining imports of the shim**

Run: `grep -r "from keisei.training.league" keisei/ tests/ --include="*.py"`
Expected: Only the shim file itself. No callers.

- [ ] **Step 2: Verify new tests cover old test scenarios**

Run: `uv run pytest tests/test_opponent_store.py tests/test_tier_managers.py tests/test_tiered_pool.py tests/test_match_scheduler.py -v --no-header 2>&1 | tail -5`
Expected: All pass.

- [ ] **Step 3: Delete league.py and test_league.py**

```bash
git rm keisei/training/league.py
git rm tests/test_league.py
```

- [ ] **Step 4: Run full test suite to confirm nothing breaks**

Run: `uv run pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove league.py shim and test_league.py (replaced by tiered pool modules)"
```
