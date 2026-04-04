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
| `keisei/training/tier_managers.py` | `FrontierManager`, `RecentFixedManager`, `DynamicManager` |
| `keisei/training/tiered_pool.py` | `TieredPool` orchestrator |
| `keisei/training/match_scheduler.py` | `MatchScheduler` — replaces `OpponentSampler` |

The existing `league.py` becomes a re-export shim (`OpponentPool = OpponentStore`, `OpponentSampler` re-exported) for one release cycle, then is removed.

---

## Data Model

### Schema Migration v3 → v4

New columns on `league_entries`:

| Column | Type | Default | Description |
|---|---|---|---|
| `role` | TEXT NOT NULL | `'unassigned'` | One of: `frontier_static`, `recent_fixed`, `dynamic`, `unassigned` |
| `status` | TEXT NOT NULL | `'active'` | One of: `active`, `retired`, `archived` |
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

New single-row table `league_meta`:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK CHECK (id = 1) | Single-row constraint |
| `bootstrapped` | INTEGER NOT NULL DEFAULT 0 | 1 after migration bootstrap completes |

Existing tables `elo_history`, `league_results`, `game_snapshots` are unchanged.

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
    # New in v4
    role: str              # frontier_static | recent_fixed | dynamic | unassigned
    status: str            # active | retired | archived
    parent_entry_id: int | None
    lineage_group: str | None
    protection_remaining: int
    last_match_at: str | None
```

---

## Components

### OpponentStore

Renamed from `OpponentPool`. Retains all existing DB, filesystem, and model-loading responsibilities. Loses all admission/eviction policy.

**Public interface:**

| Method | Description |
|---|---|
| `add_entry(model, arch, params, epoch, role)` | Create entry with specified role |
| `retire_entry(entry_id, reason)` | Set status=retired, log transition |
| `update_role(entry_id, new_role, reason)` | Change role, log transition |
| `list_entries()` | All active entries |
| `list_by_role(role)` | Filtered by role |
| `load_opponent(entry, device)` | Load model from checkpoint (unchanged) |
| `load_all_opponents(device)` | Load all active entries (unchanged) |
| `update_elo(entry_id, new_elo, epoch)` | Elo update (unchanged) |
| `record_result(...)` | Match result recording (unchanged) |
| `pin(entry_id)` / `unpin(entry_id)` | Eviction protection (unchanged) |
| `log_transition(entry_id, from_role, to_role, from_status, to_status, reason)` | Audit trail |

The store does **no** eviction on its own. `_evict_if_needed()` is removed. Tier managers call `retire_entry()` explicitly.

### FrontierManager

Owns the 5 Frontier Static slots — stable current-era benchmarks.

**Config:**

```python
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

**`select_initial` algorithm:** Sort candidates by Elo. Pick entries at indices that approximate even spacing across the Elo range (quintile selection), preferring entries with more games played as tiebreaker. This produces the "comfortably beatable to genuinely challenging" spread from the concept doc.

**Replacement policy (Phase 3 activation):** When a new Frontier Static is admitted, retire the weakest or stalest eligible entry that has exceeded `min_tenure_epochs`. Never replace more than one per review window.

### RecentFixedManager

Owns the 5 Recent Fixed slots — FIFO queue of latest learner snapshots.

**Config:**

```python
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

1. Create a new frozen Recent Fixed entry via `store.add_entry(role='recent_fixed')`.
2. If `count() > slots + soft_overflow`, call `review_oldest()`.
3. On PROMOTE: coordinate with `DynamicManager` via `TieredPool`.
4. On RETIRE: call `store.retire_entry()`.
5. On DELAY: allow temporary overflow (entry stays, reviewed next cycle).

**Review criteria for oldest Recent Fixed:**

- PROMOTE if: games_played >= `min_games_for_review` AND unique_opponents >= `min_unique_opponents` AND elo >= (weakest Dynamic Elo - `promotion_margin_elo`). Unique opponents = `SELECT COUNT(DISTINCT opponent_id) FROM league_results WHERE learner_id = ?` (counts distinct entries this entry has faced)
- DELAY if: games_played < `min_games_for_review` AND soft overflow capacity remains
- RETIRE otherwise

### DynamicManager

Owns the 10 Dynamic slots — future trainable population (frozen in Phase 1).

**Config:**

```python
slots: int = 10
protection_matches: int = 24
min_games_before_eviction: int = 40
training_enabled: bool = False  # Phase 1
```

**Interface:**

| Method | Description |
|---|---|
| `get_active()` | List active Dynamic entries |
| `admit(source_entry)` | Clone a Recent Fixed entry into Dynamic, set protection window |
| `evict_weakest()` | Remove lowest-Elo eligible Dynamic entry |
| `is_full()` | True if at slot capacity |
| `weakest_elo()` | Elo of the weakest eligible (past protection, past min games) Dynamic |

**Admission flow:**

1. If full, call `evict_weakest()` first.
2. Create new entry via `store.add_entry(role='dynamic')` with `parent_entry_id` pointing to the source Recent Fixed entry and same `lineage_group`.
3. Set `protection_remaining = protection_matches`.
4. In Phase 1: `training_enabled=False`, no optimizer state saved.

**Eviction eligibility:** An entry is eligible for eviction only if `protection_remaining == 0` AND `games_played >= min_games_before_eviction`. Evict the entry with the lowest Elo among eligible entries.

### TieredPool

Orchestrator. Single entry point for the training loop and tournament runner.

**Interface:**

| Method | Description |
|---|---|
| `snapshot_learner(model, arch, params, epoch)` | Main entry: admit to Recent Fixed, handle overflow |
| `get_opponents_by_mix()` | Return entries sampled by tier mix ratios |
| `on_epoch_end(epoch)` | Periodic housekeeping (Frontier review check) |
| `bootstrap_from_flat_pool(entries)` | One-time migration from flat pool |
| `list_all_active()` | All active entries across tiers |

**`snapshot_learner` flow:**

1. `recent_manager.admit(model, arch, params, epoch)`
2. If overflow triggered review and outcome is PROMOTE:
   - `dynamic_manager.admit(source_entry)`
3. If outcome is RETIRE:
   - `store.retire_entry(source_entry.id, "did not qualify for dynamic")`

**`bootstrap_from_flat_pool` algorithm:**

1. Sort existing active entries by Elo.
2. Select 5 entries spanning the Elo range → assign `frontier_static` (quintile spread selection, same as `FrontierManager.select_initial`).
3. Most recent 5 by epoch (excluding those already assigned) → assign `recent_fixed`.
4. Next 10 by Elo (excluding assigned) → assign `dynamic`.
5. Any remainder → `status=retired`.
6. Log all assignments as transitions.
7. Set `league_meta.bootstrapped = 1`.

### MatchScheduler

Replaces `OpponentSampler`. Role-aware weighted selection.

**Config (learner mix):**

```python
learner_dynamic_ratio: float = 0.50
learner_frontier_ratio: float = 0.30
learner_recent_ratio: float = 0.20
```

**Config (tournament match class weights):**

```python
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

**`sample_for_learner` algorithm:**

1. Roll against tier ratios to pick a role.
2. If the selected role's tier is empty, redistribute weight proportionally to non-empty tiers.
3. Pick uniformly at random within the selected tier.

**`sample_tournament_pair` algorithm:**

1. Roll against match class weights to pick a match class (e.g., Dynamic-vs-Dynamic).
2. If either tier in the selected class is empty, re-roll.
3. Pick two distinct entries from the relevant tiers (or two from the same tier for intra-tier classes).
4. Phase 1 simplified priority: penalize repeat pairings (track recent H2H counts), prefer under-sampled entries. Full priority scoring (uncertainty bonus, lineage penalty) deferred to Phase 4.

---

## Config Changes

New fields added to `LeagueConfig` in `keisei/config.py`:

```python
@dataclass(frozen=True)
class LeagueConfig:
    # Existing fields (unchanged)
    max_pool_size: int = 20
    snapshot_interval: int = 10
    epochs_per_seat: int = 50
    historical_ratio: float = 0.8
    current_best_ratio: float = 0.2
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

    # New: tier sizes
    frontier_static_slots: int = 5
    recent_fixed_slots: int = 5
    dynamic_slots: int = 10

    # New: Frontier Static policy
    frontier_review_interval_epochs: int = 250
    frontier_min_tenure_epochs: int = 100
    frontier_promotion_margin_elo: float = 50.0

    # New: Recent Fixed policy
    recent_min_games_for_review: int = 32
    recent_min_unique_opponents: int = 6
    recent_promotion_margin_elo: float = 25.0
    recent_soft_overflow: int = 1

    # New: Dynamic policy
    dynamic_protection_matches: int = 24
    dynamic_min_games_before_eviction: int = 40
    dynamic_training_enabled: bool = False

    # New: learner sampling mix
    learner_dynamic_ratio: float = 0.50
    learner_frontier_ratio: float = 0.30
    learner_recent_ratio: float = 0.20

    # New: tournament match class weights
    match_dynamic_dynamic: float = 0.40
    match_dynamic_recent: float = 0.25
    match_dynamic_frontier: float = 0.20
    match_recent_frontier: float = 0.10
    match_recent_recent: float = 0.05
```

Existing fields remain with their current defaults. The `max_pool_size` field is superseded by the sum of tier slots (5+5+10=20) but kept for backward compatibility with any code that reads it.

---

## Integration Points

### KataGoTrainingLoop

| Current code | Phase 1 change |
|---|---|
| `self.pool = OpponentPool(...)` | `self.store = OpponentStore(...)`; `self.tiered_pool = TieredPool(store, config)` |
| `self.sampler = OpponentSampler(pool, ...)` | `self.scheduler = MatchScheduler(config)` |
| `pool.add_snapshot(model, arch, params, epoch)` | `tiered_pool.snapshot_learner(model, arch, params, epoch)` |
| `sampler.sample_from(entries)` | `scheduler.sample_for_learner(tiered_pool.entries_by_role())` |
| `pool.list_entries()` for per-env assignment | `tiered_pool.list_all_active()` (models loaded same way) |
| End-of-epoch Elo updates | Unchanged — `store.update_elo()` / `store.record_result()` |

### LeagueTournament

| Current code | Phase 1 change |
|---|---|
| Random round-robin pairing | `scheduler.sample_tournament_pair(entries_by_role)` |
| Flat entry list | Entries tagged with role for result logging |

### DemonstratorRunner

No changes in Phase 1. It reads from `league_entries` and picks by Elo — the new columns don't affect its queries.

---

## Migration

### Schema Migration (v3 → v4)

```sql
ALTER TABLE league_entries ADD COLUMN role TEXT NOT NULL DEFAULT 'unassigned';
ALTER TABLE league_entries ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE league_entries ADD COLUMN parent_entry_id INTEGER REFERENCES league_entries(id);
ALTER TABLE league_entries ADD COLUMN lineage_group TEXT;
ALTER TABLE league_entries ADD COLUMN protection_remaining INTEGER NOT NULL DEFAULT 0;
ALTER TABLE league_entries ADD COLUMN last_match_at TEXT;

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

CREATE TABLE IF NOT EXISTS league_meta (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    bootstrapped INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO league_meta (id, bootstrapped) VALUES (1, 0);
```

### Bootstrap (existing pools)

On first startup after migration, if `league_meta.bootstrapped == 0`:

1. Load all active entries.
2. Run `TieredPool.bootstrap_from_flat_pool()` (quintile Elo spread → Frontier Static, most recent → Recent Fixed, next by Elo → Dynamic, remainder retired).
3. Set `bootstrapped = 1`.

### Fresh Start

`init_db` creates `league_entries` with the new columns from the start. Entries accumulate organically through `snapshot_learner()`.

### Rollback Safety

New columns have defaults. Old code reading the table ignores them. Dropping the columns restores v3 schema without data loss.

---

## Testing Strategy

### Unit Tests

- **OpponentStore:** add_entry with role, retire_entry logs transition, update_role logs transition, list_by_role filtering.
- **FrontierManager:** select_initial picks Elo-spread entries, get_active returns only frontier_static, review is no-op in Phase 1.
- **RecentFixedManager:** admit creates recent_fixed entry, overflow triggers review, review_oldest returns correct outcome based on games/Elo/opponents, DELAY respects soft_overflow.
- **DynamicManager:** admit clones with parent lineage, evict_weakest skips protected entries, evict_weakest picks lowest Elo among eligible.
- **TieredPool:** snapshot_learner end-to-end flow, bootstrap_from_flat_pool assigns correct roles, on_epoch_end calls frontier review at correct interval.
- **MatchScheduler:** sample_for_learner respects tier ratios over many samples, sample_tournament_pair produces correct match class distribution, empty tier fallback redistributes weight.

### Integration Tests

- Full lifecycle: snapshot → Recent Fixed → overflow → review → promote to Dynamic → more snapshots → Dynamic eviction.
- Migration: create v3 DB with 15 flat entries, run migration + bootstrap, verify 5 FS + 5 RF + 5 D assigned correctly.
- Training loop smoke test: TieredPool plugs into KataGoTrainingLoop without errors for a few epochs.

---

## Phase 1 Boundaries

**In scope:**
- Storage/policy split (OpponentStore + tier managers)
- Role assignment and lifecycle state machine
- Cross-tier promotion (Recent Fixed → Dynamic)
- Weighted matchmaking (learner mix + tournament match classes)
- Schema migration and flat-pool bootstrap
- Config extension
- Training loop and tournament integration

**Explicitly deferred:**
- Dynamic training (optimizer state, PPO updates) → Phase 3
- Historical Library (log-spaced milestones, gauntlet) → Phase 2
- Role-specific Elo tracking → Phase 2
- Parallel match scheduling → Phase 4
- Advanced priority scoring (uncertainty, lineage penalties) → Phase 4
- Dashboard role badges → separate UI task
