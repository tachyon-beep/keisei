# Tiered Opponent Pool — Phase 1: Role Split (Frozen Only)

## Purpose

Split the existing flat 20-seat `OpponentPool` into a tiered system with three distinct roles: Frontier Static (5), Recent Fixed (5), and Dynamic (10). All entries remain frozen in Phase 1 — Dynamic training is deferred to Phase 3. This phase proves the structural foundation: role-aware storage, tier-specific admission/eviction, cross-tier promotion, weighted matchmaking, and migration from the existing flat pool.

## Context

### Problem

The current `OpponentPool` is a flat FIFO queue of frozen snapshots. Every entry is functionally identical — no role differentiation, no promotion pipeline, no difficulty-spanning benchmarks. The class also conflates storage (DB, filesystem, model loading) with policy (eviction, sampling), making it hard to test and extend.

### Source Design

This spec implements Phase 1 of the Kenshi Mixed League design (`docs/concepts/tiered-opponent-pool.md`). That document defines a 20-seat active league plus a 5-slot historical library. Phase 1 covers the active league role split only.

### Goals

1. Separate storage from policy by splitting `OpponentPool` into `OpponentStore` + tier managers.
2. Assign every league entry a role: `frontier_static`, `recent_fixed`, or `dynamic`.
3. Implement tier-specific admission, eviction, and cross-tier promotion rules.
4. Replace flat opponent sampling with role-weighted matchmaking.
5. Migrate existing flat pools to the tiered schema without data loss.
6. Keep all entries frozen — Dynamic training is out of scope.

### Non-Goals

1. Dynamic training (optimizer state, PPO updates for Dynamic entries).
2. Historical Library (log-spaced milestones, benchmark gauntlets).
3. Role-specific Elo (frontier_benchmark_elo, dynamic_league_elo, etc.).
4. Parallel match scheduling or advanced priority scoring.
5. Dashboard UI changes (role badges, tier panels).

---

## Architecture

### Component Overview

```
KataGoTrainingLoop / LeagueTournament
        │
        ▼
   TieredPool  (orchestrator — coordinates cross-tier operations)
        │
        ├── FrontierManager    (5 slots — stable benchmarks)
        ├── RecentFixedManager (5 slots — FIFO fresh blood)
        ├── DynamicManager     (10 slots — future trainable population)
        └── MatchScheduler     (role-weighted opponent selection)
                │
                ▼
         OpponentStore  (DB + filesystem + model loading)
```

### File Layout

| New file | Contents |
|---|---|
| `keisei/training/opponent_store.py` | `OpponentStore` — renamed/refactored from `league.py` storage layer |
| `keisei/training/tier_managers.py` | `FrontierManager`, `RecentFixedManager`, `DynamicManager`, enums (`Role`, `EntryStatus`, `ReviewOutcome`) |
| `keisei/training/tiered_pool.py` | `TieredPool` orchestrator |
| `keisei/training/match_scheduler.py` | `MatchScheduler` — replaces `OpponentSampler` |

The existing `league.py` is removed. All callers (`katago_loop.py`, `tournament.py`, tests) are updated to import from the new modules in the same PR. No shim — the interface changes are mandatory for Phase 1.

---

## Data Model

### Enums

```python
from enum import StrEnum

class Role(StrEnum):
    FRONTIER_STATIC = "frontier_static"
    RECENT_FIXED = "recent_fixed"
    DYNAMIC = "dynamic"
    UNASSIGNED = "unassigned"

class EntryStatus(StrEnum):
    ACTIVE = "active"
    RETIRED = "retired"
    ARCHIVED = "archived"

class ReviewOutcome(StrEnum):
    PROMOTE = "promote"
    RETIRE = "retire"
    DELAY = "delay"
```

These enums are used in the dataclass type hints, SQL INSERT calls, and `from_db_row` (which coerces via `Role(row["role"])`), providing validation at the DB boundary.

### Schema

New columns on `league_entries`:

| Column | Type | Default | Description |
|---|---|---|---|
| `role` | TEXT NOT NULL | `'unassigned'` | One of `Role` values |
| `status` | TEXT NOT NULL | `'active'` | One of `EntryStatus` values |
| `parent_entry_id` | INTEGER | NULL | FK to self — tracks clone lineage |
| `lineage_group` | TEXT | NULL | Groups entries from the same snapshot chain |
| `protection_remaining` | INTEGER | 0 | Matches remaining in protection window |
| `last_match_at` | TEXT | NULL | Timestamp of most recent match |

New table `league_transitions`:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `entry_id` | INTEGER NOT NULL | FK to `league_entries` |
| `from_role` | TEXT | Role before transition (NULL for creation) |
| `to_role` | TEXT | Role after transition |
| `from_status` | TEXT | Status before transition (NULL for creation) |
| `to_status` | TEXT | Status after transition |
| `reason` | TEXT | Human-readable reason |
| `created_at` | TEXT | Timestamp |

New index on `league_results`:

```sql
CREATE INDEX IF NOT EXISTS idx_league_results_learner ON league_results(learner_id);
```

This supports the `unique_opponents` promotion query without a full table scan.

New single-row table `league_meta`:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK CHECK (id = 1) | Single-row constraint |
| `bootstrapped` | INTEGER NOT NULL DEFAULT 0 | 1 after migration bootstrap completes |

Existing tables `elo_history`, `league_results`, `game_snapshots` are unchanged.

Since there are no inflight runs or live users, the schema is created fresh in `init_db` with all new columns and tables included from the start. No incremental `ALTER TABLE` migration is needed. The full tiered system will be implemented and validated during test/calibration runs before the main training run begins.

### OpponentEntry Extension

The `OpponentEntry` dataclass gains new fields matching the schema columns:

```python
@dataclass(frozen=True)
class OpponentEntry:
    id: int
    display_name: str
    architecture: str
    model_params: dict[str, Any]
    checkpoint_path: str
    elo_rating: float
    created_epoch: int
    games_played: int
    created_at: str
    flavour_facts: list[list[str]]
    # New
    role: Role
    status: EntryStatus
    parent_entry_id: int | None
    lineage_group: str | None
    protection_remaining: int
    last_match_at: str | None
```

`OpponentEntry` is a frozen dataclass. The `model_params` and `flavour_facts` fields are mutable containers but are only constructed fresh from JSON in `from_db_row` — never mutated after construction. Entries are read-only snapshots; all mutations go through `OpponentStore` methods that write to the DB and return fresh entries.

---

## Components

### OpponentStore

Renamed from `OpponentPool`. Retains all existing DB, filesystem, and model-loading responsibilities. Loses all admission/eviction policy.

**Transaction API:**

The store exposes a `transaction()` context manager that holds the lock and defers `commit()` until `__exit__`. This ensures multi-step mutations (e.g., admit + set protection) are atomic:

```python
with store.transaction():
    entry = store.add_entry(role=Role.DYNAMIC, parent_entry_id=source.id, ...)
    store.set_protection(entry.id, protection_matches)
```

Without this, another thread could observe a Dynamic entry with `protection_remaining=0` between the two calls and evict it immediately.

**Public interface:**

| Method | Description |
|---|---|
| `transaction()` | Context manager: holds lock, defers commit until exit |
| `add_entry(model, arch, params, epoch, role)` | Create entry with specified role |
| `clone_entry(source_entry_id, new_role, reason)` | Copy checkpoint file, create new DB row with lineage fields. Avoids loading model to GPU just to save it back |
| `retire_entry(entry_id, reason)` | Set status=retired, log transition. Does NOT delete checkpoint file (may be referenced by clones) |
| `update_role(entry_id, new_role, reason)` | Change role, log transition |
| `list_entries()` | All active entries |
| `list_by_role(role)` | Filtered by role |
| `load_opponent(entry, device)` | Load model from checkpoint. Uses `map_location="cpu"` then `.to(device)` to avoid transient VRAM duplication |
| `load_all_opponents(device)` | Load all active entries |
| `update_elo(entry_id, new_elo, epoch)` | Elo update (unchanged) |
| `record_result(...)` | Match result recording. Also updates `last_match_at` on both participants and decrements `protection_remaining` (floor 0) on both participants |
| `decrement_protection(entry_id)` | Decrement `protection_remaining` by 1 (floor 0) |
| `pin(entry_id)` / `unpin(entry_id)` | Eviction protection (unchanged) |
| `log_transition(entry_id, from_role, to_role, from_status, to_status, reason)` | Audit trail |

The store does **no** eviction on its own. `_evict_if_needed()` is removed. Tier managers call `retire_entry()` explicitly.

**Checkpoint lifecycle:** `retire_entry` sets `status=retired` in the DB and logs a transition but does NOT delete the `.pt` file. This is because cloned entries (Dynamic entries created from Recent Fixed sources) may share lineage with the same checkpoint. Checkpoint file cleanup is a separate concern — orphaned `.pt` files (where no active or retired entry references them) can be cleaned up by a periodic sweep, but this is not part of Phase 1.

### FrontierManager

Owns the 5 Frontier Static slots — stable current-era benchmarks.

**Config:**

```python
@dataclass(frozen=True)
class FrontierStaticConfig:
    slots: int = 5
    review_interval_epochs: int = 250
    min_tenure_epochs: int = 100
    promotion_margin_elo: float = 50.0
```

**Interface:**

| Method | Description |
|---|---|
| `get_active()` | List active Frontier Static entries |
| `select_initial(entries, count=5)` | Bootstrap: pick entries spanning the Elo range |
| `review(epoch)` | Check if any Dynamic qualifies for promotion (no-op in Phase 1) |
| `is_due_for_review(epoch)` | True if epoch aligns with review interval |

**`select_initial` algorithm:** Sort candidates by Elo. Pick entries at indices that approximate even spacing across the Elo range (quintile selection), preferring entries with more `games_played` as tiebreaker, then `created_epoch` ascending as final tiebreaker for determinism. This produces the "comfortably beatable to genuinely challenging" spread from the concept doc. If fewer than 5 candidates exist, take all of them.

**Replacement policy (Phase 3 activation):** When a new Frontier Static is admitted, retire the weakest or stalest eligible entry that has exceeded `min_tenure_epochs`. Never replace more than one per review window.

### RecentFixedManager

Owns the 5 Recent Fixed slots — FIFO queue of latest learner snapshots.

**Config:**

```python
@dataclass(frozen=True)
class RecentFixedConfig:
    slots: int = 5
    min_games_for_review: int = 32
    min_unique_opponents: int = 6
    promotion_margin_elo: float = 25.0
    soft_overflow: int = 1
```

**Interface:**

| Method | Description |
|---|---|
| `get_active()` | List active Recent Fixed entries |
| `admit(model, arch, params, epoch)` | Create frozen Recent Fixed entry, trigger overflow review if needed |
| `review_oldest()` | Returns `ReviewOutcome`: PROMOTE, RETIRE, or DELAY |
| `count()` | Current queue depth |

**Admission flow:**

1. Create a new frozen Recent Fixed entry via `store.add_entry(role=Role.RECENT_FIXED)`.
2. If `count() > slots + soft_overflow`, call `review_oldest()`.
3. On PROMOTE: coordinate with `DynamicManager` via `TieredPool`.
4. On RETIRE: call `store.retire_entry()`.
5. On DELAY: allow temporary overflow (entry stays, reviewed next cycle).

**Review criteria for oldest Recent Fixed:**

- PROMOTE if: `games_played >= min_games_for_review` AND `unique_opponents >= min_unique_opponents` AND `elo >= (weakest Dynamic Elo - promotion_margin_elo)`. Unique opponents = `SELECT COUNT(DISTINCT opponent_id) FROM league_results WHERE learner_id = ? UNION SELECT COUNT(DISTINCT learner_id) FROM league_results WHERE opponent_id = ?` — counts all distinct entries this entry has faced in either seat.
- DELAY if: `games_played < min_games_for_review` AND soft overflow capacity remains (current overflow count < `soft_overflow`)
- RETIRE otherwise

### DynamicManager

Owns the 10 Dynamic slots — future trainable population (frozen in Phase 1).

**Config:**

```python
@dataclass(frozen=True)
class DynamicConfig:
    slots: int = 10
    protection_matches: int = 24
    min_games_before_eviction: int = 40
    training_enabled: bool = False  # Phase 1; raises NotImplementedError if True
```

**Interface:**

| Method | Description |
|---|---|
| `get_active()` | List active Dynamic entries |
| `admit(source_entry)` | Clone a Recent Fixed entry into Dynamic, set protection window |
| `evict_weakest()` | Remove lowest-Elo eligible Dynamic entry |
| `is_full()` | True if at slot capacity |
| `weakest_elo()` | Elo of the weakest eligible (past protection, past min games) Dynamic. Returns `None` if no eligible entries exist |

**Admission flow:**

1. If full, call `evict_weakest()` first.
2. Clone via `store.clone_entry(source_entry.id, Role.DYNAMIC, "promoted from recent_fixed")` — this copies the checkpoint file and creates a new DB row with `parent_entry_id` and `lineage_group` set.
3. Set `protection_remaining = protection_matches` (inside the same `store.transaction()`).
4. In Phase 1: `training_enabled=False`, no optimizer state saved. The constructor asserts `not config.training_enabled` with a message pointing to Phase 3.

**Eviction eligibility:** An entry is eligible for eviction only if `protection_remaining == 0` AND `games_played >= min_games_before_eviction`. Evict the entry with the lowest Elo among eligible entries.

### TieredPool

Orchestrator. Single entry point for the training loop and tournament runner.

**Interface:**

| Method | Description |
|---|---|
| `snapshot_learner(model, arch, params, epoch)` | Main entry: admit to Recent Fixed, handle overflow |
| `entries_by_role()` | Returns `dict[Role, list[OpponentEntry]]` — entries grouped by role |
| `get_opponents_by_mix()` | Return entries sampled by tier mix ratios |
| `on_epoch_end(epoch)` | Periodic housekeeping (Frontier review check) |
| `bootstrap_from_flat_pool(entries)` | One-time migration from flat pool (runs inside a single transaction) |
| `list_all_active()` | All active entries across tiers |

**`snapshot_learner` flow:**

1. `recent_manager.admit(model, arch, params, epoch)`
2. If overflow triggered review and outcome is PROMOTE:
   - `dynamic_manager.admit(source_entry)`
3. If outcome is RETIRE:
   - `store.retire_entry(source_entry.id, "did not qualify for dynamic")`

**`bootstrap_from_flat_pool` algorithm:**

Runs inside a single `store.transaction()` so a crash leaves the DB unchanged:

1. Sort existing active entries by Elo.
2. Select up to 5 entries spanning the Elo range → assign `frontier_static` (quintile spread selection, same as `FrontierManager.select_initial`). If fewer than 5 entries exist, assign proportionally: ~25% frontier, ~25% recent, ~50% dynamic (minimum 1 per tier if entries exist).
3. Most recent up to 5 by epoch (excluding those already assigned) → assign `recent_fixed`.
4. Next up to 10 by Elo (excluding assigned) → assign `dynamic`.
5. Any remainder → `status=retired`.
6. Log all assignments as transitions.
7. Set `league_meta.bootstrapped = 1`.

### MatchScheduler

Replaces `OpponentSampler`. Role-aware weighted selection.

**Config:**

```python
@dataclass(frozen=True)
class MatchSchedulerConfig:
    # Learner opponent mix
    learner_dynamic_ratio: float = 0.50
    learner_frontier_ratio: float = 0.30
    learner_recent_ratio: float = 0.20

    # Tournament match class weights
    match_dynamic_dynamic: float = 0.40
    match_dynamic_recent: float = 0.25
    match_dynamic_frontier: float = 0.20
    match_recent_frontier: float = 0.10
    match_recent_recent: float = 0.05
```

**Interface:**

| Method | Description |
|---|---|
| `sample_for_learner(entries_by_role)` | Pick one opponent using learner tier mix |
| `sample_tournament_pair(entries_by_role)` | Pick a (entry_a, entry_b) pair using match class weights |
| `effective_ratios(entries_by_role)` | Returns the actual ratios after empty-tier redistribution (for logging) |

**`sample_for_learner` algorithm:**

1. Roll against tier ratios to pick a role.
2. If the selected role's tier is empty, redistribute weight proportionally to non-empty tiers.
3. Pick uniformly at random within the selected tier.

**`sample_tournament_pair` algorithm:**

1. Pre-filter match classes to those where both required tiers are non-empty.
2. Normalize remaining weights to sum to 1.0.
3. Roll against filtered weights to pick a match class.
4. Pick two distinct entries from the relevant tiers (or two from the same tier for intra-tier classes).
5. Phase 1 simplified priority: penalize repeat pairings (track recent H2H counts), prefer under-sampled entries. Full priority scoring (uncertainty bonus, lineage penalty) deferred to Phase 4.

---

## Config Changes

Nested config dataclasses in `keisei/config.py`:

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
    match_dynamic_dynamic: float = 0.40
    match_dynamic_recent: float = 0.25
    match_dynamic_frontier: float = 0.20
    match_recent_frontier: float = 0.10
    match_recent_recent: float = 0.05

@dataclass(frozen=True)
class LeagueConfig:
    # Existing fields (unchanged)
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

    # Nested tier configs
    frontier: FrontierStaticConfig = FrontierStaticConfig()
    recent: RecentFixedConfig = RecentFixedConfig()
    dynamic: DynamicConfig = DynamicConfig()
    scheduler: MatchSchedulerConfig = MatchSchedulerConfig()
```

**Validation in `__post_init__`:**

- `scheduler.learner_dynamic_ratio + scheduler.learner_frontier_ratio + scheduler.learner_recent_ratio` must equal 1.0 (within float tolerance)
- Sum of all `scheduler.match_*` weights must equal 1.0 (within float tolerance)
- Total pool capacity is derived: `frontier.slots + recent.slots + dynamic.slots`

The old `max_pool_size`, `historical_ratio`, and `current_best_ratio` fields are removed (no live users to break).

---

## Integration Points

### KataGoTrainingLoop

| Current code | Phase 1 change |
|---|---|
| `self.pool = OpponentPool(...)` | `self.store = OpponentStore(...)`; `self.tiered_pool = TieredPool(store, config)` |
| `self.sampler = OpponentSampler(pool, ...)` | `self.scheduler = MatchScheduler(config.scheduler)` |
| `pool.add_snapshot(model, arch, params, epoch)` | `tiered_pool.snapshot_learner(model, arch, params, epoch)` |
| `sampler.sample_from(entries)` | `scheduler.sample_for_learner(tiered_pool.entries_by_role())` |
| `pool.list_entries()` for per-env assignment | `tiered_pool.list_all_active()` (models loaded same way) |
| End-of-epoch Elo updates | Unchanged — `store.update_elo()` / `store.record_result()` |

### LeagueTournament

The tournament runner must be refactored to use `OpponentStore` for all DB operations — entry reads, result recording, and Elo updates — not just `MatchScheduler` for pairing. It currently opens its own DB connection and bypasses the pool entirely. This is required to prevent two independent code paths writing to the same tables with different semantics.

| Current code | Phase 1 change |
|---|---|
| Own DB connection + raw SQL | Uses `OpponentStore` for all reads/writes |
| Random round-robin pairing | `scheduler.sample_tournament_pair(entries_by_role)` |
| Direct Elo updates | `store.update_elo()` / `store.record_result()` |

### DemonstratorRunner

No changes in Phase 1. It reads from `league_entries` and picks by Elo — the new columns don't affect its queries.

---

## Monitoring (Phase 1 Watch Points)

The systems review identified several feedback dynamics to monitor during Phase 1:

1. **RETIRE vs PROMOTE ratio** — if RETIRE dominates within the first 500 epochs, the calibration rate is too slow for the snapshot cadence. Tune `snapshot_interval` or `min_games_for_review`.
2. **Dynamic tier Elo standard deviation** — if it narrows below ~30 points, the tier is concentrating into a homogeneous band (R2 ratchet). Diversity-aware eviction (Phase 4) may need to be pulled forward.
3. **Frontier Static Elo vs learner Elo** — if the learner exceeds the Frontier ceiling by >100 points, the benchmark anchor has drifted. Phase 3's Frontier review should be prioritized.
4. **`unique_opponents` at review time** — if consistently below 6, tournament throughput is the bottleneck, not snapshot quality.

These are logged to the DB via existing `elo_history` and `league_results` tables. No new observability infrastructure is needed — the dashboard can query these directly.

---

## Testing Strategy

### Unit Tests

- **OpponentStore:** add_entry with role, clone_entry copies checkpoint file and sets lineage, retire_entry logs transition (does not delete file), update_role logs transition, list_by_role filtering, transaction context manager holds lock and defers commit, record_result updates `last_match_at` and decrements `protection_remaining`.
- **FrontierManager:** select_initial picks Elo-spread entries (deterministic tiebreaker), get_active returns only frontier_static, review is no-op in Phase 1, handles underfull pools gracefully.
- **RecentFixedManager:** admit creates recent_fixed entry, overflow triggers review, review_oldest returns correct outcome based on games/Elo/opponents, DELAY respects soft_overflow limit, unique_opponents counts both seats in league_results.
- **DynamicManager:** admit clones with parent lineage via clone_entry, evict_weakest skips protected entries, evict_weakest picks lowest Elo among eligible, constructor asserts training_enabled=False.
- **TieredPool:** snapshot_learner end-to-end flow, bootstrap_from_flat_pool assigns correct roles (full and underfull pools), bootstrap is atomic (simulated crash leaves DB unchanged), on_epoch_end calls frontier review at correct interval, entries_by_role returns correct grouping.
- **MatchScheduler:** sample_for_learner respects tier ratios over many samples, sample_tournament_pair produces correct match class distribution, empty tier fallback pre-filters and redistributes (no infinite re-roll), effective_ratios reports actual ratios, config validation rejects ratios that don't sum to 1.0.

### Integration Tests

- Full lifecycle: snapshot → Recent Fixed → overflow → review → promote to Dynamic → more snapshots → Dynamic eviction.
- Bootstrap: create DB with flat pool entries (both full 20 and underfull 8), run bootstrap, verify role assignments and proportional allocation.
- Training loop smoke test: TieredPool plugs into KataGoTrainingLoop without errors for a few epochs.
- Tournament integration: LeagueTournament uses OpponentStore and MatchScheduler correctly.

---

## Phase 1 Boundaries

**In scope:**
- Storage/policy split (OpponentStore + tier managers)
- Role assignment and lifecycle state machine
- Cross-tier promotion (Recent Fixed → Dynamic) with clone_entry
- Weighted matchmaking (learner mix + tournament match classes)
- Schema creation with all new columns/tables
- Flat-pool bootstrap (atomic, handles underfull pools)
- Nested config dataclasses with ratio validation
- Training loop and tournament integration (tournament refactored to use OpponentStore)
- Monitoring watch points for system dynamics

**Explicitly deferred:**
- Dynamic training (optimizer state, PPO updates) → Phase 3
- Historical Library (log-spaced milestones, gauntlet) → Phase 2
- Role-specific Elo tracking → Phase 2
- Parallel match scheduling → Phase 4
- Advanced priority scoring (uncertainty, lineage penalties) → Phase 4
- Dashboard role badges → separate UI task
- Checkpoint file cleanup for retired entries → future housekeeping task
