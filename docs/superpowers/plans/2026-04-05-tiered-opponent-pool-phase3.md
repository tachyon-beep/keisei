# Tiered Opponent Pool Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable PPO training for Dynamic entries from league match data, persist optimizer state, and activate the Frontier Static promotion pipeline so proven Dynamic entries can graduate to stable benchmarks.

**Architecture:** DynamicTrainer (small PPO updates for Dynamic entries) + FrontierPromoter (conservative promotion evaluation) + extensions to OpponentStore (optimizer persistence), DynamicManager (training hooks), FrontierManager (review activation), TieredPool (wiring), and LeagueTournament (match data collection + training trigger).

**Tech Stack:** Python 3.13, SQLite (WAL mode), PyTorch, frozen dataclasses, StrEnum, threading.RLock, uv for deps, `uv run pytest` for all tests.

**Spec:** `docs/superpowers/specs/2026-04-05-tiered-opponent-pool-phase3-design.md`

---

## Critical Implementation Notes

These notes capture design decisions and lessons learned from Phase 1/2 reviews. Read before implementing.

### Locking and Commit Discipline (Inherited from Phase 1)

OpponentStore uses `threading.RLock()` (reentrant lock), NOT `threading.Lock()`.

**Commit rule:** Every new store method that mutates the DB must check `if not self._in_transaction: self._conn.commit()`. This applies to all Phase 3 additions: `save_optimizer`, `load_optimizer`, `increment_update_count`.

```python
def some_mutating_method(self, ...):
    with self._lock:
        # ... do SQL work ...
        if not self._in_transaction:
            self._conn.commit()
```

### No Monkey-Patching

Phase 2 review rejected a monkey-patch pattern. If a class needs injectable behavior, pass callables as constructor parameters. DynamicTrainer and FrontierPromoter receive their dependencies through constructor injection, not by patching attributes on existing objects.

### No Direct Store Internals Access

DynamicTrainer and FrontierPromoter must NOT access `store._conn` or `store._lock` directly. If new DB operations are needed, add public methods to OpponentStore. Phase 3 adds: `save_optimizer()`, `load_optimizer()`, `increment_update_count()`.

### Role.RETIRED Does Not Exist

RETIRED is an `EntryStatus`, not a `Role`. To retire an entry, call `store.retire_entry(entry_id, reason)` which sets `status=EntryStatus.RETIRED`. The entry's role stays unchanged. Use `Role.UNASSIGNED`/`Role.DYNAMIC` + `store.retire_entry()`.

### OpponentEntry Defaults (Inherited from Phase 1)

All new fields on OpponentEntry must have defaults for backward compatibility. Phase 3 adds `optimizer_path`, `update_count`, and `last_train_at` — all default to `None`/`0`/`None`.

### Schema Migration Pattern

Follow the existing v3->v4 pattern: check `db_version < 6`, use `PRAGMA table_info` to check for existing columns before `ALTER TABLE ADD COLUMN`, then `UPDATE schema_version SET version = 6`. Never use `CREATE TABLE ... IF NOT EXISTS` for columns that go on existing tables.

### Test Assertions Must Be Behavioral

Every test must assert specific values, not just "doesn't crash". For example, after a DynamicTrainer update, assert that model weights have changed (compare state_dict checksums before/after). After FrontierPromoter.evaluate(), assert the returned candidate ID matches the expected entry.

### Every Task Has Test Steps

No implementation-only tasks. Even schema migration (Task 1) gets a test verifying the new columns exist and round-trip correctly.

### _get_learner_entry Must Use Explicit ID

Never use `max(epoch)` heuristic to find the learner. Pass the learner's entry ID explicitly.

### DynamicTrainer Is Independent of Learner PPO

DynamicTrainer uses the same PPO clipped objective function as `KataGoPPOAlgorithm.update()` but is a completely separate code path. It does NOT inherit from or compose `KataGoPPOAlgorithm`. The learner's optimizer, scaler, and gradient state are never shared. DynamicTrainer creates its own Adam optimizer per Dynamic entry.

### Match Data Collection Is CPU-Only

`MatchRollout` tensors are stored on CPU to avoid GPU memory pressure. They are only moved to GPU during the training step inside `DynamicTrainer.update()`.

### Streak Tracking Is In-Memory Only

`FrontierPromoter` tracks top-K streak epochs in an in-memory dict. This is intentionally lost on restart (conservative: delays promotion slightly after restart). No new DB column for this.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `keisei/training/dynamic_trainer.py` | Create | `DynamicTrainer` — small PPO updates for Dynamic entries from league match data |
| `keisei/training/frontier_promoter.py` | Create | `FrontierPromoter` — evaluates Dynamic entries for Frontier Static promotion |
| `keisei/training/opponent_store.py` | Modify | Add `optimizer_path`/`update_count`/`last_train_at` to OpponentEntry, add `save_optimizer`, `load_optimizer`, `increment_update_count` methods |
| `keisei/training/tier_managers.py` | Modify | Remove `training_enabled=False` guard from DynamicManager, add `get_trainable()` method; activate FrontierManager.review() |
| `keisei/training/tiered_pool.py` | Modify | Wire DynamicTrainer and FrontierPromoter into TieredPool |
| `keisei/training/tournament.py` | Modify | Collect match rollout data, call DynamicTrainer after trainable matches |
| `keisei/config.py` | Modify | Extend `DynamicConfig` with training params, `FrontierStaticConfig` with promotion params |
| `keisei/db.py` | Modify | Schema v5->v6 migration: `optimizer_path`, `update_count`, `last_train_at` columns |
| `tests/test_dynamic_trainer.py` | Create | Unit tests for DynamicTrainer |
| `tests/test_frontier_promoter.py` | Create | Unit tests for FrontierPromoter |
| `tests/test_phase3_store.py` | Create | Tests for Phase 3 OpponentStore extensions |
| `tests/test_phase3_integration.py` | Create | Integration tests for training + promotion cycles |

---

## Tasks

### Task 1: Schema v6 Migration — optimizer_path, update_count, last_train_at

**Goal:** Add three new columns to `league_entries` for Phase 3 Dynamic training state.

**Files:** `keisei/db.py`, `tests/test_phase3_store.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_store.py::test_schema_v6_migration`):
  - Create a v5 database (using current `init_db`).
  - Manually set `schema_version` to 5.
  - Call `init_db` again (should trigger v5->v6 migration).
  - Query `PRAGMA table_info(league_entries)` and assert `optimizer_path`, `update_count`, `last_train_at` columns exist.
  - Insert a row with the new columns set, read it back, assert values match.
  - Assert `schema_version` is now 6.
- [ ] **Verify test fails** — `uv run pytest tests/test_phase3_store.py::test_schema_v6_migration -x`
- [ ] **Implement:**
  - In `keisei/db.py`, bump `SCHEMA_VERSION = 6` (from 5, which Phase 2 sets).
  - Add migration block:
    ```python
    if db_version < 6:
        cols = [c[1] for c in conn.execute("PRAGMA table_info(league_entries)").fetchall()]
        if "optimizer_path" not in cols:
            conn.execute("ALTER TABLE league_entries ADD COLUMN optimizer_path TEXT")
        if "update_count" not in cols:
            conn.execute("ALTER TABLE league_entries ADD COLUMN update_count INTEGER NOT NULL DEFAULT 0")
        if "last_train_at" not in cols:
            conn.execute("ALTER TABLE league_entries ADD COLUMN last_train_at TEXT")
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
    ```
  - Add the new columns to the CREATE TABLE block (for fresh databases).
  - Update `read_league_data` to include `optimizer_path`, `update_count`, `last_train_at` in the SELECT.
- [ ] **Verify test passes** — `uv run pytest tests/test_phase3_store.py::test_schema_v6_migration -x`
- [ ] **Commit:** `feat(db): add schema v6 migration for Phase 3 Dynamic training columns`

---

### Task 2: OpponentEntry Extensions — New Fields

**Goal:** Add `optimizer_path`, `update_count`, `last_train_at` fields to `OpponentEntry` dataclass with backward-compatible defaults.

**Files:** `keisei/training/opponent_store.py`, `tests/test_phase3_store.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_store.py::test_opponent_entry_new_fields`):
  - Construct `OpponentEntry` with only the original fields (no new fields) — assert it works (backward compat).
  - Construct `OpponentEntry` with `optimizer_path="foo.pt"`, `update_count=5`, `last_train_at="2026-04-05T00:00:00Z"` — assert all three are accessible and correct.
  - Assert defaults: `optimizer_path=None`, `update_count=0`, `last_train_at=None`.
- [ ] **Write failing test** (`tests/test_phase3_store.py::test_opponent_entry_from_db_row_new_fields`):
  - Create a mock `sqlite3.Row`-like object with the new columns populated.
  - Call `OpponentEntry.from_db_row(row)` and assert the new fields are populated.
  - Create a mock row WITHOUT the new columns — assert from_db_row still works with defaults.
- [ ] **Verify tests fail** — `uv run pytest tests/test_phase3_store.py -k "new_fields" -x`
- [ ] **Implement:**
  - Add to `OpponentEntry` dataclass:
    ```python
    optimizer_path: str | None = None
    update_count: int = 0
    last_train_at: str | None = None
    ```
  - Update `from_db_row` to read the new columns with safe fallbacks:
    ```python
    optimizer_path=row["optimizer_path"] if "optimizer_path" in keys else None,
    update_count=row["update_count"] if "update_count" in keys else 0,
    last_train_at=row["last_train_at"] if "last_train_at" in keys else None,
    ```
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_store.py -k "new_fields" -x`
- [ ] **Commit:** `feat(store): add optimizer_path, update_count, last_train_at to OpponentEntry`

---

### Task 3: OpponentStore — save_optimizer, load_optimizer, increment_update_count

**Goal:** Add three new public methods to OpponentStore for Dynamic training state persistence.

**Files:** `keisei/training/opponent_store.py`, `tests/test_phase3_store.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_store.py::test_save_load_optimizer_round_trip`):
  - Create a store with a DB and a temp league directory.
  - Add an entry via `store.add_entry(...)`.
  - Create a dummy optimizer state dict: `{"param_groups": [...], "state": {0: {"step": 10, "exp_avg": torch.zeros(4)}}}`.
  - Call `store.save_optimizer(entry.id, optimizer_state_dict)`.
  - Assert the optimizer file exists at `{checkpoint_path}_optimizer.pt` (with `_optimizer` inserted before `.pt`).
  - Assert `store._get_entry(entry.id).optimizer_path` is not None.
  - Call `loaded = store.load_optimizer(entry.id, device="cpu")`.
  - Assert `loaded` is not None and `loaded["state"][0]["step"] == 10`.
- [ ] **Write failing test** (`tests/test_phase3_store.py::test_load_optimizer_returns_none_when_missing`):
  - Add an entry, do NOT save optimizer.
  - Call `store.load_optimizer(entry.id, device="cpu")`.
  - Assert result is None.
- [ ] **Write failing test** (`tests/test_phase3_store.py::test_save_optimizer_atomic_write`):
  - Save optimizer, assert no `.tmp` files remain in league directory.
- [ ] **Write failing test** (`tests/test_phase3_store.py::test_increment_update_count`):
  - Add an entry. Assert `update_count == 0` and `last_train_at is None`.
  - Call `store.increment_update_count(entry.id)`.
  - Re-read entry. Assert `update_count == 1` and `last_train_at is not None`.
  - Call `store.increment_update_count(entry.id)` again.
  - Assert `update_count == 2` and `last_train_at` is more recent than the first.
- [ ] **Verify tests fail** — `uv run pytest tests/test_phase3_store.py -k "optimizer or increment" -x`
- [ ] **Implement:**
  - `save_optimizer(entry_id, optimizer_state_dict)`:
    ```python
    def save_optimizer(self, entry_id: int, optimizer_state_dict: dict) -> None:
        with self._lock:
            entry = self._get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Entry {entry_id} not found")
            ckpt = Path(entry.checkpoint_path)
            opt_path = ckpt.with_name(ckpt.stem + "_optimizer" + ckpt.suffix)
            tmp_path = opt_path.with_suffix(opt_path.suffix + ".tmp")
            torch.save(optimizer_state_dict, tmp_path)
            tmp_path.rename(opt_path)
            self._conn.execute(
                "UPDATE league_entries SET optimizer_path = ? WHERE id = ?",
                (str(opt_path), entry_id),
            )
            if not self._in_transaction:
                self._conn.commit()
    ```
  - `load_optimizer(entry_id, device)`:
    ```python
    def load_optimizer(self, entry_id: int, device: str = "cpu") -> dict | None:
        with self._lock:
            entry = self._get_entry(entry_id)
            if entry is None:
                raise ValueError(f"Entry {entry_id} not found")
            if entry.optimizer_path is None:
                return None
            opt_path = Path(entry.optimizer_path)
            if not opt_path.exists():
                return None
            return torch.load(opt_path, map_location=device, weights_only=False)
    ```
  - `increment_update_count(entry_id)`:
    ```python
    def increment_update_count(self, entry_id: int) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE league_entries
                   SET update_count = update_count + 1,
                       last_train_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                   WHERE id = ?""",
                (entry_id,),
            )
            if not self._in_transaction:
                self._conn.commit()
    ```
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_store.py -k "optimizer or increment" -x`
- [ ] **Commit:** `feat(store): add save_optimizer, load_optimizer, increment_update_count`

---

### Task 4: Config Extensions — DynamicConfig Training Params + FrontierStaticConfig Promotion Params

**Goal:** Extend `DynamicConfig` and `FrontierStaticConfig` with Phase 3 parameters, with defaults that preserve backward compatibility.

**Files:** `keisei/config.py`, `tests/test_config.py` (or `tests/test_phase3_config.py`)

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_config.py::test_dynamic_config_training_defaults`):
  - Construct `DynamicConfig()` with no args. Assert:
    - `training_enabled == True` (changed from False)
    - `update_epochs_per_batch == 2`
    - `batch_reuse == 1`
    - `lr_scale == 0.25`
    - `grad_clip == 1.0`
    - `update_every_matches == 4`
    - `max_updates_per_minute == 20`
    - `checkpoint_flush_every == 8`
    - `disable_on_error == True`
- [ ] **Write failing test** (`tests/test_phase3_config.py::test_dynamic_config_validation`):
  - Assert `DynamicConfig(lr_scale=0.0)` raises `ValueError`.
  - Assert `DynamicConfig(lr_scale=1.5)` raises `ValueError`.
  - Assert `DynamicConfig(lr_scale=1.0)` does NOT raise.
  - Assert `DynamicConfig(update_every_matches=0)` raises `ValueError`.
  - Assert `DynamicConfig(max_updates_per_minute=0)` raises `ValueError`.
- [ ] **Write failing test** (`tests/test_phase3_config.py::test_frontier_config_promotion_defaults`):
  - Construct `FrontierStaticConfig()` with no args. Assert:
    - `min_games_for_promotion == 64`
    - `topk == 3`
    - `streak_epochs == 50`
    - `max_lineage_overlap == 2`
- [ ] **Write failing test** (`tests/test_phase3_config.py::test_frontier_config_promotion_validation`):
  - Assert `FrontierStaticConfig(min_games_for_promotion=50, min_tenure_epochs=100)` raises `ValueError` (min_games < min_tenure).
- [ ] **Write failing test** (`tests/test_phase3_config.py::test_load_config_with_phase3_fields`):
  - Create a temp TOML file with `[league.dynamic]` containing `training_enabled = true`, `lr_scale = 0.5`.
  - Load config. Assert `config.league.dynamic.lr_scale == 0.5`.
  - Create a TOML without the new fields. Assert config loads with defaults.
- [ ] **Verify tests fail** — `uv run pytest tests/test_phase3_config.py -x`
- [ ] **Implement:**
  - Update `DynamicConfig`:
    ```python
    @dataclass(frozen=True)
    class DynamicConfig:
        slots: int = 10
        protection_matches: int = 24
        min_games_before_eviction: int = 40
        training_enabled: bool = True  # Phase 3: changed from False
        # Phase 3 training fields
        update_epochs_per_batch: int = 2
        batch_reuse: int = 1
        lr_scale: float = 0.25
        grad_clip: float = 1.0
        update_every_matches: int = 4
        max_updates_per_minute: int = 20
        checkpoint_flush_every: int = 8
        disable_on_error: bool = True

        def __post_init__(self) -> None:
            if not (0.0 < self.lr_scale <= 1.0):
                raise ValueError(
                    f"dynamic.lr_scale must be in (0, 1], got {self.lr_scale}"
                )
            if self.update_every_matches < 1:
                raise ValueError(
                    f"dynamic.update_every_matches must be >= 1, got {self.update_every_matches}"
                )
            if self.max_updates_per_minute < 1:
                raise ValueError(
                    f"dynamic.max_updates_per_minute must be >= 1, got {self.max_updates_per_minute}"
                )
    ```
  - Update `FrontierStaticConfig`:
    ```python
    @dataclass(frozen=True)
    class FrontierStaticConfig:
        slots: int = 5
        review_interval_epochs: int = 250
        min_tenure_epochs: int = 100
        promotion_margin_elo: float = 50.0
        # Phase 3 promotion fields
        min_games_for_promotion: int = 64
        topk: int = 3
        streak_epochs: int = 50
        max_lineage_overlap: int = 2

        def __post_init__(self) -> None:
            if self.min_games_for_promotion < self.min_tenure_epochs:
                raise ValueError(
                    f"frontier.min_games_for_promotion ({self.min_games_for_promotion}) must be >= "
                    f"frontier.min_tenure_epochs ({self.min_tenure_epochs})"
                )
    ```
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_config.py -x`
- [ ] **Run existing config tests** — `uv run pytest tests/test_config.py -x` (ensure no regressions from default change `training_enabled=True`)
- [ ] **Commit:** `feat(config): add Phase 3 training and promotion parameters`

---

### Task 5: MatchRollout Dataclass

**Goal:** Create the `MatchRollout` dataclass that captures match replay data for Dynamic training.

**Files:** `keisei/training/dynamic_trainer.py` (start of file), `tests/test_dynamic_trainer.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_match_rollout_construction`):
  - Construct a `MatchRollout` with synthetic tensors:
    - `observations`: shape `(10, 3, 50, 9, 9)` (10 steps, 3 envs, 50 channels)
    - `actions`: shape `(10, 3)`
    - `rewards`: shape `(10, 3)`
    - `dones`: shape `(10, 3)`
    - `legal_masks`: shape `(10, 3, 11259)`
    - `perspective`: shape `(10, 3)`
  - Assert all fields are accessible and have correct shapes.
  - Assert all tensors are on CPU (not GPU).
- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_match_rollout_filter_by_perspective`):
  - Create a MatchRollout with mixed perspectives (some 0, some 1).
  - Filter observations/actions/rewards for perspective == 0 only.
  - Assert the filtered count matches the expected number of perspective-0 steps.
- [ ] **Verify tests fail** — `uv run pytest tests/test_dynamic_trainer.py -k "match_rollout" -x`
- [ ] **Implement:**
  ```python
  @dataclass
  class MatchRollout:
      """Replay data from a league match for Dynamic entry training."""
      observations: torch.Tensor    # (steps, num_envs, obs_channels, 9, 9)
      actions: torch.Tensor         # (steps, num_envs)
      rewards: torch.Tensor         # (steps, num_envs)
      dones: torch.Tensor           # (steps, num_envs)
      legal_masks: torch.Tensor     # (steps, num_envs, action_space)
      perspective: torch.Tensor     # (steps, num_envs) — 0=player_A, 1=player_B
  ```
- [ ] **Verify tests pass** — `uv run pytest tests/test_dynamic_trainer.py -k "match_rollout" -x`
- [ ] **Commit:** `feat(trainer): add MatchRollout dataclass for Dynamic entry training`

---

### Task 6: DynamicTrainer — Core Implementation

**Goal:** Implement `DynamicTrainer` with `should_update`, `is_rate_limited`, `update`, and `get_update_stats` methods.

**Files:** `keisei/training/dynamic_trainer.py`, `tests/test_dynamic_trainer.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_should_update_threshold`):
  - Create a `DynamicTrainer` with `update_every_matches=4`.
  - Call `trainer.record_match(entry_id=1)` three times. Assert `trainer.should_update(1)` is False.
  - Call `trainer.record_match(entry_id=1)` a fourth time. Assert `trainer.should_update(1)` is True.
  - After an update, the counter resets. Assert `trainer.should_update(1)` is False again.
- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_is_rate_limited`):
  - Create a `DynamicTrainer` with `max_updates_per_minute=2`.
  - Mock time so two updates happen in the same minute.
  - Assert `trainer.is_rate_limited()` is False before any updates.
  - Record two updates. Assert `trainer.is_rate_limited()` is True.
  - Advance mocked time by 61 seconds. Assert `trainer.is_rate_limited()` is False.
- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_update_modifies_weights`):
  - Create a small model (e.g., a 2-block SE-ResNet with minimal channels).
  - Create an OpponentStore entry pointing at the model's checkpoint.
  - Create synthetic MatchRollout data with 4 matches' worth of steps.
  - Record the model's parameter checksum before update.
  - Call `trainer.update(entry, match_rollouts, device="cpu")`.
  - Assert the parameter checksum has changed (weights were modified).
  - Assert `store.increment_update_count` was called (update_count > 0).
- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_update_uses_lr_scale`):
  - Create trainer with `lr_scale=0.25` and a known `learner_lr=2e-4`.
  - After calling update, inspect the optimizer's param_groups[0]["lr"].
  - Assert `lr == 2e-4 * 0.25 == 5e-5`.
- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_update_saves_weights_after_update`):
  - Run an update. Check that the checkpoint file has been modified (mtime changed or content differs).
- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_update_saves_optimizer_at_flush_interval`):
  - Create trainer with `checkpoint_flush_every=2`.
  - Run update once (match 4 of 4 accumulated). Assert optimizer file does NOT exist yet (not at flush interval).
  - Accumulate 4 more matches, run update again (now at 8 matches = checkpoint_flush_every). Assert optimizer file EXISTS.
- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_get_update_stats`):
  - Before any updates: `get_update_stats(entry_id)` returns `(0, None)`.
  - After one update: returns `(1, <timestamp>)` where timestamp is not None.
- [ ] **Verify tests fail** — `uv run pytest tests/test_dynamic_trainer.py -k "not match_rollout" -x`
- [ ] **Implement `DynamicTrainer`:**
  - Constructor takes `store: OpponentStore`, `config: DynamicConfig`, `learner_lr: float`.
  - Internal state:
    - `_match_counts: dict[int, int]` — matches accumulated per entry since last update.
    - `_total_matches: dict[int, int]` — total matches per entry (for flush interval).
    - `_update_timestamps: list[float]` — sliding window for rate limiting.
    - `_optimizers: dict[int, torch.optim.Adam]` — cached optimizers per entry.
    - `_disabled_entries: set[int]` — entries with training disabled due to error.
    - `_rollout_buffers: dict[int, list[MatchRollout]]` — accumulated rollouts per entry.
  - `record_match(entry_id: int, rollout: MatchRollout)`: Append rollout to buffer, increment match count.
  - `should_update(entry_id: int) -> bool`: `_match_counts.get(entry_id, 0) >= config.update_every_matches` and entry not in `_disabled_entries`.
  - `is_rate_limited() -> bool`: Count entries in `_update_timestamps` within last 60 seconds; return `count >= config.max_updates_per_minute`.
  - `update(entry: OpponentEntry, device: str) -> bool`:
    1. Load model via `store.load_opponent(entry, device)` — but load in TRAIN mode (remove `model.eval()` call, or call `model.train()` after).
    2. Get or create optimizer. If entry has saved optimizer, load it. Otherwise create fresh Adam with `lr = learner_lr * config.lr_scale`.
    3. Concatenate accumulated rollouts into one batch.
    4. Filter batch by perspective matching this entry (player A or B depending on which side the entry played).
    5. Run `config.update_epochs_per_batch` PPO epochs:
       - For each epoch, shuffle and mini-batch the data.
       - Forward pass: get policy logits and value logits.
       - Compute PPO clipped policy loss (same formula as `KataGoPPOAlgorithm.update()`).
       - Compute value loss (WDL cross-entropy, same as learner, but NO score head loss).
       - Backward pass with gradient clipping at `config.grad_clip`.
       - Optimizer step.
    6. Save updated weights to checkpoint file (atomic: write to .tmp, rename).
    7. If `_total_matches[entry.id] % config.checkpoint_flush_every == 0`, save optimizer state.
    8. Call `store.increment_update_count(entry.id)`.
    9. Reset `_match_counts[entry.id]` and clear accumulated rollouts.
    10. Record timestamp in `_update_timestamps`.
    11. Cache optimizer in `_optimizers[entry.id]`.
    12. Return True on success.
  - On exception during update: if `config.disable_on_error`, add entry to `_disabled_entries`, log transition via store, return False. Otherwise re-raise.
- [ ] **Verify tests pass** — `uv run pytest tests/test_dynamic_trainer.py -x`
- [ ] **Commit:** `feat(trainer): implement DynamicTrainer with PPO updates for Dynamic entries`

---

### Task 7: DynamicTrainer — Error Fallback

**Goal:** Verify that training errors disable the affected entry gracefully.

**Files:** `tests/test_dynamic_trainer.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_error_fallback_disables_entry`):
  - Create trainer with `disable_on_error=True`.
  - Inject a NaN into the rollout rewards to cause a training error (NaN loss).
  - Call `trainer.update(entry, device="cpu")`.
  - Assert the method returns False (not True).
  - Assert entry is in `trainer._disabled_entries`.
  - Assert `trainer.should_update(entry.id)` returns False even after accumulating enough matches.
  - Assert a transition was logged with reason containing "training disabled due to error".
- [ ] **Write failing test** (`tests/test_dynamic_trainer.py::test_error_fallback_disabled_setting`):
  - Create trainer with `disable_on_error=False`.
  - Inject a NaN. Assert the update raises an exception (not caught).
- [ ] **Verify tests fail** — `uv run pytest tests/test_dynamic_trainer.py -k "error_fallback" -x`
- [ ] **Implement:** Error handling in `DynamicTrainer.update()` (should already be scaffolded from Task 6, but verify the behavior matches).
- [ ] **Verify tests pass** — `uv run pytest tests/test_dynamic_trainer.py -k "error_fallback" -x`
- [ ] **Commit:** `test(trainer): verify DynamicTrainer error fallback behavior`

---

### Task 8: FrontierPromoter — Core Implementation

**Goal:** Implement `FrontierPromoter` with `evaluate` and `should_promote` methods, including streak tracking.

**Files:** `keisei/training/frontier_promoter.py`, `tests/test_frontier_promoter.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_frontier_promoter.py::test_evaluate_returns_none_when_no_candidates`):
  - Create promoter with default `FrontierStaticConfig`.
  - Create 3 Dynamic entries with `games_played < min_games_for_promotion`.
  - Call `promoter.evaluate(dynamic_entries, frontier_entries, epoch=100)`.
  - Assert result is None.
- [ ] **Write failing test** (`tests/test_frontier_promoter.py::test_evaluate_returns_none_below_elo_margin`):
  - Create Dynamic entries with enough games, in top-K, held top-K for streak_epochs.
  - But the best Dynamic's Elo is only 20 points above the weakest Frontier (below promotion_margin of 50).
  - Assert `evaluate()` returns None.
- [ ] **Write failing test** (`tests/test_frontier_promoter.py::test_evaluate_returns_candidate_when_all_criteria_met`):
  - Create 5 Frontier entries with Elo [1200, 1150, 1100, 1050, 1000].
  - Create 10 Dynamic entries. Top 3 have Elo [1200, 1150, 1100], all with `games_played=100`.
  - Call `evaluate` repeatedly for `streak_epochs` consecutive epochs to build streak.
  - Assert the top Dynamic entry (Elo 1200, which exceeds weakest Frontier 1000 by 200 > 50) is returned as candidate.
- [ ] **Write failing test** (`tests/test_frontier_promoter.py::test_should_promote_checks_all_five_criteria`):
  - Test each criterion independently:
    1. `games_played` too low -> False.
    2. Not in top-K -> False.
    3. Streak too short -> False.
    4. Elo margin too small -> False.
    5. Lineage overlap exceeded -> False.
  - Test with all criteria met -> True.
- [ ] **Write failing test** (`tests/test_frontier_promoter.py::test_streak_tracking_across_reviews`):
  - Call `evaluate` at epoch 100 with entry A in top-K. Assert A's streak starts.
  - Call `evaluate` at epoch 150 (50 epochs later). Assert A qualifies (streak >= 50).
  - Call `evaluate` at epoch 151 with A dropped out of top-K. Assert A's streak resets.
  - Call `evaluate` at epoch 152 with A back in top-K. Assert A does NOT qualify yet (streak = 1 epoch).
- [ ] **Write failing test** (`tests/test_frontier_promoter.py::test_lineage_overlap_limit`):
  - Create Frontier entries: 2 share `lineage_group="lineage-1"` (at max_lineage_overlap=2).
  - Create a Dynamic entry with `lineage_group="lineage-1"`.
  - Assert `should_promote` returns False due to lineage overlap.
  - Create a Dynamic entry with `lineage_group="lineage-2"`.
  - Assert `should_promote` returns True (different lineage).
- [ ] **Verify tests fail** — `uv run pytest tests/test_frontier_promoter.py -x`
- [ ] **Implement `FrontierPromoter`:**
  - Constructor takes `config: FrontierStaticConfig`.
  - Internal state:
    - `_topk_streaks: dict[int, int]` — `{entry_id: first_seen_in_topk_epoch}`.
  - `evaluate(dynamic_entries, frontier_entries, epoch) -> OpponentEntry | None`:
    1. Sort dynamic entries by `elo_rating` descending.
    2. Identify top-K entries.
    3. Update streak tracking: add new top-K entries, remove entries that dropped out.
    4. For each top-K entry (highest Elo first), call `should_promote`. Return first that qualifies.
    5. Return None if no candidate qualifies.
  - `should_promote(candidate, frontier_entries, epoch) -> bool`:
    1. `candidate.games_played >= config.min_games_for_promotion`
    2. candidate is in top-K (checked by caller, but verify)
    3. `epoch - _topk_streaks[candidate.id] >= config.streak_epochs`
    4. `candidate.elo_rating >= min(f.elo_rating for f in frontier_entries) + config.promotion_margin_elo`
    5. Count frontier entries with same `lineage_group` as candidate; must be `< config.max_lineage_overlap`
- [ ] **Verify tests pass** — `uv run pytest tests/test_frontier_promoter.py -x`
- [ ] **Commit:** `feat(promoter): implement FrontierPromoter with multi-criteria promotion evaluation`

---

### Task 9: DynamicManager Changes — Remove Guard, Add get_trainable

**Goal:** Remove the `training_enabled=False` guard from DynamicManager and add a `get_trainable()` method.

**Files:** `keisei/training/tier_managers.py`, `tests/test_tier_managers.py` (or `tests/test_phase3_managers.py`)

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_managers.py::test_dynamic_manager_allows_training_enabled`):
  - Construct `DynamicManager` with `DynamicConfig(training_enabled=True)`.
  - Assert it does NOT raise (previously Phase 1 had an assert/guard blocking this).
- [ ] **Write failing test** (`tests/test_phase3_managers.py::test_get_trainable_returns_active_dynamic_entries`):
  - Create a DynamicManager. Mock store to return 3 Dynamic entries.
  - Mark one as having training disabled (entry is in disabled_entries set).
  - Call `manager.get_trainable(disabled_entries={entry2.id})`. Assert returns only the other 2 entries.
- [ ] **Write failing test** (`tests/test_phase3_managers.py::test_get_trainable_empty_when_training_disabled`):
  - Create DynamicManager with `training_enabled=False`.
  - Call `manager.get_trainable(disabled_entries=set())`. Assert returns empty list.
- [ ] **Verify tests fail** — `uv run pytest tests/test_phase3_managers.py -x`
- [ ] **Implement:**
  - Remove the `training_enabled=False` guard/assert from `DynamicManager.__init__`.
  - Add method:
    ```python
    def get_trainable(self, disabled_entries: set[int] | None = None) -> list[OpponentEntry]:
        """Return Dynamic entries eligible for training updates."""
        if not self.config.training_enabled:
            return []
        disabled = disabled_entries or set()
        return [e for e in self.store.list_by_role(Role.DYNAMIC)
                if e.id not in disabled]
    ```
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_managers.py -x`
- [ ] **Run Phase 1 tier manager tests** — `uv run pytest tests/ -k "tier_manager" -x` (ensure no regressions)
- [ ] **Commit:** `feat(managers): enable Dynamic training in DynamicManager, add get_trainable`

---

### Task 10: FrontierManager Changes — Activate review()

**Goal:** Activate `FrontierManager.review()` to call FrontierPromoter and orchestrate promotion.

**Files:** `keisei/training/tier_managers.py`, `tests/test_phase3_managers.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_managers.py::test_frontier_review_promotes_candidate`):
  - Create FrontierManager with a FrontierPromoter (passed via constructor parameter, NOT monkey-patched).
  - Set up store with 5 Frontier entries and 10 Dynamic entries.
  - Configure the promoter state so one Dynamic entry meets all promotion criteria.
  - Call `manager.review(epoch=300)`.
  - Assert a new Frontier Static entry was created (via `store.clone_entry`).
  - Assert the weakest/stalest existing Frontier entry was retired (via `store.retire_entry`).
  - Assert the Dynamic entry that was promoted still exists and is still Dynamic (cloning, not moving).
- [ ] **Write failing test** (`tests/test_phase3_managers.py::test_frontier_review_no_promotion_when_no_candidate`):
  - Set up a state where no Dynamic entries meet criteria.
  - Call `manager.review(epoch=300)`.
  - Assert no new entries created, no entries retired.
- [ ] **Write failing test** (`tests/test_phase3_managers.py::test_frontier_review_retires_weakest_or_stalest`):
  - Create 5 Frontier entries. One has the lowest Elo AND has exceeded `min_tenure_epochs`.
  - After promotion, assert that specific entry was retired.
  - Test the stalest path: all Frontier entries have similar Elo but one was created much earlier. Assert the oldest is retired.
- [ ] **Write failing test** (`tests/test_phase3_managers.py::test_frontier_review_never_replaces_more_than_one`):
  - Even if two Dynamic entries qualify, only one is promoted per review window.
- [ ] **Write failing test** (`tests/test_phase3_managers.py::test_frontier_review_skips_promotion_when_frontier_not_full`):
  - If Frontier has fewer than `slots` entries, clone without retiring. Assert no retirement.
- [ ] **Verify tests fail** — `uv run pytest tests/test_phase3_managers.py -k "frontier_review" -x`
- [ ] **Implement:**
  - Modify `FrontierManager.__init__` to accept `promoter: FrontierPromoter | None = None`.
  - Implement `review(epoch)`:
    ```python
    def review(self, epoch: int) -> None:
        if self.promoter is None:
            return  # No-op when promoter not provided (backward compat with Phase 1/2)
        dynamic_entries = self.store.list_by_role(Role.DYNAMIC)
        frontier_entries = self.store.list_by_role(Role.FRONTIER_STATIC)
        candidate = self.promoter.evaluate(dynamic_entries, frontier_entries, epoch)
        if candidate is None:
            return
        # Clone Dynamic -> Frontier Static (frozen, no optimizer)
        new_entry = self.store.clone_entry(candidate.id, Role.FRONTIER_STATIC,
            reason=f"promoted from Dynamic entry {candidate.id} ({candidate.display_name}) at epoch {epoch}")
        # Retire weakest/stalest if at capacity
        if len(frontier_entries) >= self.config.slots:
            self._retire_weakest_or_stalest(frontier_entries, epoch)
    ```
  - Implement `_retire_weakest_or_stalest(frontier_entries, epoch)`:
    - Filter entries that have exceeded `min_tenure_epochs` (created_epoch + min_tenure <= epoch).
    - Among those, retire the one with the lowest Elo. If tied, retire the one with the oldest `created_epoch`.
    - Call `store.retire_entry(entry_id, reason)`.
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_managers.py -k "frontier_review" -x`
- [ ] **Commit:** `feat(managers): activate FrontierManager.review() with promotion pipeline`

---

### Task 11: TieredPool Changes — Wire DynamicTrainer and FrontierPromoter

**Goal:** Create DynamicTrainer and FrontierPromoter in TieredPool constructor and pass them to the appropriate managers.

**Files:** `keisei/training/tiered_pool.py`, `tests/test_phase3_integration.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_tiered_pool_creates_trainer_and_promoter`):
  - Construct a `TieredPool` with Phase 3 config (training_enabled=True).
  - Assert `pool.dynamic_trainer` is a `DynamicTrainer` instance.
  - Assert `pool.frontier_manager.promoter` is a `FrontierPromoter` instance.
- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_tiered_pool_on_epoch_end_calls_review`):
  - Set up TieredPool with mocked store and entries meeting promotion criteria.
  - Call `pool.on_epoch_end(epoch=300)`.
  - Assert `FrontierManager.review()` was called (not a no-op).
- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_tiered_pool_no_trainer_when_training_disabled`):
  - Construct TieredPool with `training_enabled=False`.
  - Assert `pool.dynamic_trainer` is None.
- [ ] **Verify tests fail** — `uv run pytest tests/test_phase3_integration.py -k "tiered_pool" -x`
- [ ] **Implement:**
  - In `TieredPool.__init__`:
    ```python
    # Phase 3: DynamicTrainer
    if config.dynamic.training_enabled:
        self.dynamic_trainer = DynamicTrainer(
            store=self.store,
            config=config.dynamic,
            learner_lr=learner_lr,  # passed from training loop
        )
    else:
        self.dynamic_trainer = None

    # Phase 3: FrontierPromoter
    promoter = FrontierPromoter(config.frontier)
    # Pass promoter to FrontierManager
    self.frontier_manager = FrontierManager(store, config.frontier, promoter=promoter)
    ```
  - In `on_epoch_end(epoch)`, ensure `self.frontier_manager.review(epoch)` is called (replacing the no-op).
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_integration.py -k "tiered_pool" -x`
- [ ] **Commit:** `feat(pool): wire DynamicTrainer and FrontierPromoter into TieredPool`

---

### Task 12: Tournament Changes — Match Data Collection

**Goal:** Extend `_play_batch` to optionally collect match rollout data and return it alongside results.

**Files:** `keisei/training/tournament.py`, `tests/test_phase3_integration.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_play_batch_collects_rollout_when_requested`):
  - Create a mock VecEnv that returns known observations, legal_masks, rewards, etc.
  - Call `_play_batch(vecenv, model_a, model_b, collect_rollout=True)`.
  - Assert the returned MatchRollout is not None.
  - Assert `rollout.observations.shape[0] > 0` (at least some steps collected).
  - Assert all tensors are on CPU.
- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_play_batch_no_rollout_by_default`):
  - Call `_play_batch(vecenv, model_a, model_b)` (no collect_rollout arg).
  - Assert the returned rollout is None (backward compatible).
- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_play_batch_rollout_has_correct_perspective`):
  - With known mock data where player A moves at steps 0, 2, 4 and player B at steps 1, 3, 5.
  - Assert `rollout.perspective` matches the expected pattern.
- [ ] **Verify tests fail** — `uv run pytest tests/test_phase3_integration.py -k "play_batch" -x`
- [ ] **Implement:**
  - Modify `_play_batch` signature to accept `collect_rollout: bool = False`.
  - Return type changes to `tuple[int, int, int, MatchRollout | None]`.
  - When `collect_rollout=True`, accumulate per-step tensors (on CPU) during the play loop:
    - `step_obs.append(obs.cpu())`
    - `step_actions.append(actions.cpu())`
    - `step_rewards.append(torch.from_numpy(rewards))`
    - `step_dones.append(torch.from_numpy(terminated | truncated))`
    - `step_legal_masks.append(legal_masks.cpu())`
    - `step_perspective.append(torch.from_numpy(current_players))`
  - After the loop, stack into a MatchRollout and return it.
  - When `collect_rollout=False`, return None as the fourth element.
  - Update `_play_match` to pass through `collect_rollout` and aggregate rollouts from batches.
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_integration.py -k "play_batch" -x`
- [ ] **Commit:** `feat(tournament): collect match rollout data for Dynamic training`

---

### Task 13: Tournament Changes — Training Trigger After Trainable Matches

**Goal:** After each trainable match (D-vs-D or D-vs-RF), check if a Dynamic entry training update is due and trigger it.

**Files:** `keisei/training/tournament.py`, `tests/test_phase3_integration.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_tournament_triggers_training_after_dynamic_match`):
  - Create a LeagueTournament with a mock DynamicTrainer.
  - Set up two Dynamic entries.
  - Simulate a match (mock `_play_match` to return results + rollout).
  - After the match, assert `trainer.record_match` was called for both Dynamic entries.
  - Configure `should_update` to return True. Assert `trainer.update` was called.
- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_tournament_skips_training_for_non_trainable_match`):
  - Set up a Dynamic vs Frontier Static match.
  - Assert training is NOT triggered (Frontier Static calibration only per spec).
  - Actually, re-read spec: "Dynamic vs Recent Fixed: Dynamic entry may update". So D-vs-FS is NOT trainable.
  - Assert `trainer.record_match` is NOT called for the Dynamic entry in a D-vs-FS match.
- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_tournament_respects_rate_limit`):
  - Configure trainer where `is_rate_limited()` returns True.
  - Assert `trainer.update` is NOT called even when `should_update` returns True.
- [ ] **Write failing test** (`tests/test_phase3_integration.py::test_tournament_no_training_when_trainer_is_none`):
  - Create LeagueTournament with `dynamic_trainer=None`.
  - Simulate a D-vs-D match. Assert no crash, no training attempted.
- [ ] **Verify tests fail** — `uv run pytest tests/test_phase3_integration.py -k "tournament_trigger or tournament_skip or tournament_rate or tournament_no_train" -x`
- [ ] **Implement:**
  - Add `dynamic_trainer: DynamicTrainer | None = None` parameter to `LeagueTournament.__init__`.
  - Add `_is_trainable_match(entry_a, entry_b) -> bool`:
    ```python
    def _is_trainable_match(self, entry_a: OpponentEntry, entry_b: OpponentEntry) -> bool:
        """D-vs-D or D-vs-RF produces training data. D-vs-FS and Historical do not."""
        trainable_roles = {Role.DYNAMIC, Role.RECENT_FIXED}
        return (entry_a.role in trainable_roles and entry_b.role in trainable_roles
                and (entry_a.role == Role.DYNAMIC or entry_b.role == Role.DYNAMIC))
    ```
  - In `_run_loop`, after recording match result:
    ```python
    if self.dynamic_trainer and self._is_trainable_match(entry_a, entry_b):
        # Collect rollout if we have a trainer
        # (rollout was collected during _play_match if trainer exists)
        for entry in [entry_a, entry_b]:
            if entry.role == Role.DYNAMIC:
                self.dynamic_trainer.record_match(entry.id, rollout)
                if (self.dynamic_trainer.should_update(entry.id)
                        and not self.dynamic_trainer.is_rate_limited()):
                    self.dynamic_trainer.update(entry, device=str(self.device))
    ```
  - Modify `_play_match` to pass `collect_rollout=True` when `self.dynamic_trainer is not None` and the match is trainable.
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_integration.py -k "tournament" -x`
- [ ] **Commit:** `feat(tournament): trigger Dynamic training after trainable matches`

---

### Task 14: read_league_data Extension

**Goal:** Include `optimizer_path`, `update_count`, `last_train_at` in the dashboard data query.

**Files:** `keisei/db.py`, `tests/test_phase3_store.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase3_store.py::test_read_league_data_includes_phase3_fields`):
  - Create a DB with schema v6, insert a league entry with `optimizer_path="/tmp/opt.pt"`, `update_count=3`, `last_train_at="2026-04-05T12:00:00Z"`.
  - Call `read_league_data(db_path)`.
  - Assert the returned entry dict contains `optimizer_path`, `update_count`, `last_train_at` with correct values.
- [ ] **Verify test fails** — `uv run pytest tests/test_phase3_store.py::test_read_league_data_includes_phase3_fields -x`
- [ ] **Implement:**
  - Update the SELECT in `read_league_data` to add the three new columns:
    ```python
    "SELECT id, display_name, flavour_facts, model_params, architecture, "
    "elo_rating, games_played, created_epoch, created_at, "
    "role, status, parent_entry_id, lineage_group, protection_remaining, last_match_at, "
    "optimizer_path, update_count, last_train_at "
    "FROM league_entries WHERE status = 'active' ORDER BY elo_rating DESC"
    ```
- [ ] **Verify test passes** — `uv run pytest tests/test_phase3_store.py::test_read_league_data_includes_phase3_fields -x`
- [ ] **Commit:** `feat(db): include Phase 3 columns in read_league_data`

---

### Task 15: End-to-End Integration Test — Dynamic Training Cycle

**Goal:** Test the full cycle: tournament plays D-vs-D match, DynamicTrainer collects rollout, runs update, weights change, store updated.

**Files:** `tests/test_phase3_integration.py`

**TDD Steps:**

- [ ] **Write test** (`tests/test_phase3_integration.py::test_full_dynamic_training_cycle`):
  - Set up:
    - In-memory SQLite DB with schema v6.
    - OpponentStore with temp league directory.
    - Two Dynamic entries created via `store.add_entry(...)` with small test models.
    - DynamicTrainer with `update_every_matches=1` (immediate update for testing).
    - Mock VecEnv that produces deterministic match data.
  - Execute:
    - Play a match between the two Dynamic entries (simulated via `_play_match` or direct `_play_batch` with mock).
    - Feed match data to DynamicTrainer.
    - Call `trainer.update(entry, device="cpu")`.
  - Assert:
    - Model weights differ before vs after update (compare state_dict checksums).
    - `store._get_entry(entry.id).update_count == 1`.
    - `store._get_entry(entry.id).last_train_at is not None`.
    - Checkpoint file on disk has been updated.
- [ ] **Write test** (`tests/test_phase3_integration.py::test_full_frontier_promotion_cycle`):
  - Set up:
    - OpponentStore with 5 Frontier entries (Elo [1200, 1150, 1100, 1050, 1000]) and 10 Dynamic entries.
    - Best Dynamic entry has Elo 1200, games_played=100, lineage_group="lineage-new".
    - FrontierPromoter configured with `streak_epochs=1` (for testing speed).
  - Execute:
    - Call `promoter.evaluate(dynamic_entries, frontier_entries, epoch=100)` twice (epoch 100, 101) to build streak.
    - Call `frontier_manager.review(epoch=101)`.
  - Assert:
    - A new Frontier Static entry exists with the promoted entry's architecture and weights.
    - The weakest Frontier entry (Elo 1000) was retired.
    - The promoted Dynamic entry still exists and is still Dynamic.
    - Total active Frontier entries == 5 (one added, one retired).
- [ ] **Verify tests pass** — `uv run pytest tests/test_phase3_integration.py -k "full_" -x`
- [ ] **Commit:** `test(integration): add end-to-end Dynamic training and Frontier promotion tests`

---

### Task 16: Full Test Suite Verification

**Goal:** Run the complete test suite to verify no regressions from Phase 3 changes.

**Steps:**

- [ ] Run `uv run pytest tests/ -x --timeout=120` — all tests must pass.
- [ ] Run `uv run pytest tests/test_dynamic_trainer.py tests/test_frontier_promoter.py tests/test_phase3_store.py tests/test_phase3_integration.py tests/test_phase3_config.py tests/test_phase3_managers.py -v` — all Phase 3 tests pass with verbose output.
- [ ] Verify no test uses `store._conn` or `store._lock` directly (search test files for `_conn` and `_lock` access from non-store classes).
- [ ] Verify all new OpponentStore methods follow the `if not self._in_transaction: self._conn.commit()` pattern (search for new methods in opponent_store.py).
- [ ] **Commit:** `test(phase3): verify full suite passes with no regressions`

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Task(s) | Status |
|---|---|---|
| Schema v5->v6 (optimizer_path, update_count, last_train_at) | Task 1 | Covered |
| OpponentEntry new fields | Task 2 | Covered |
| OpponentStore extensions (save_optimizer, load_optimizer, increment_update_count) | Task 3 | Covered |
| DynamicConfig training params | Task 4 | Covered |
| FrontierStaticConfig promotion params | Task 4 | Covered |
| Config validation (__post_init__) | Task 4 | Covered |
| MatchRollout dataclass | Task 5 | Covered |
| DynamicTrainer (should_update, update, is_rate_limited, get_update_stats) | Task 6 | Covered |
| DynamicTrainer safety rails (rate limiting, error fallback, checkpoint frequency) | Tasks 6, 7 | Covered |
| FrontierPromoter (evaluate, should_promote, streak tracking) | Task 8 | Covered |
| FrontierPromoter lineage overlap limit | Task 8 | Covered |
| DynamicManager remove guard, add get_trainable | Task 9 | Covered |
| FrontierManager.review() activation | Task 10 | Covered |
| FrontierManager replacement policy (retire weakest/stalest) | Task 10 | Covered |
| TieredPool wiring | Task 11 | Covered |
| Tournament _play_batch rollout collection | Task 12 | Covered |
| Tournament training trigger after trainable matches | Task 13 | Covered |
| Match class rules (D-vs-D, D-vs-RF trainable; D-vs-FS not) | Task 13 | Covered |
| read_league_data extension | Task 14 | Covered |
| Filesystem layout (optimizer alongside weights) | Task 3 | Covered |
| Monitoring (update rate, Elo churn, promotion events, error rate) | Implicitly covered via get_update_stats + DB queries; no separate monitoring task needed |
| Inference-only fallback | Task 7 | Covered |
| Full training disable (training_enabled=False) | Tasks 9, 11 | Covered |

### Placeholder Check

No TBD, TODO, or "fill in later" placeholders remain. All tasks have concrete implementation details.

### Type Consistency

- `OpponentEntry` fields: `optimizer_path: str | None`, `update_count: int`, `last_train_at: str | None` — used consistently across Tasks 1-3, 14.
- `DynamicConfig` fields: all `int`/`float`/`bool` — consistent with `KataGoPPOParams` patterns.
- `FrontierStaticConfig` fields: all `int`/`float` — consistent with existing config.
- `MatchRollout` tensors: all `torch.Tensor`, stored on CPU — consistent across Tasks 5, 12, 13.
- `DynamicTrainer.update` returns `bool` (success/failure) — checked in Task 6, 7, 13.
- `FrontierPromoter.evaluate` returns `OpponentEntry | None` — checked in Tasks 8, 10.
- `_is_trainable_match` uses `Role.DYNAMIC` and `Role.RECENT_FIXED` — consistent with spec match class rules.

### Lessons Applied

1. **Locking:** All new store methods use `with self._lock` + `if not self._in_transaction: self._conn.commit()`. Verified in Tasks 3, 16.
2. **No monkey-patching:** FrontierPromoter passed as constructor parameter to FrontierManager (Task 10), not patched.
3. **Schema migration:** Uses `ALTER TABLE ADD COLUMN` with column existence checks, follows v3->v4 pattern (Task 1).
4. **Role.RETIRED does not exist:** Plan uses `store.retire_entry()` for retirement (Task 10), never references `Role.RETIRED`.
5. **OpponentEntry defaults:** All new fields have defaults (Task 2).
6. **Behavioral assertions:** Every test asserts specific values, not just "doesn't crash" (all tasks).
7. **Every task has test steps:** All 16 tasks include TDD steps.
8. **No direct store internals access:** DynamicTrainer and FrontierPromoter use only public store methods (Tasks 6, 8).
9. **Explicit ID for _get_learner_entry:** Not applicable to Phase 3 (no learner entry lookup), but pattern is followed.
10. **No `uv run grep`:** Not used anywhere in this plan.
