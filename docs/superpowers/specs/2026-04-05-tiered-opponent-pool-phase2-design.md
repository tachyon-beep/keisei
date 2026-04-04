# Tiered Opponent Pool — Phase 2: Historical Library & Role-Specific Elo

## Purpose

Add a 5-slot Historical Library for long-range regression detection and split the single Elo rating into four role-specific views. The Historical Library is separate from the active league — it does not consume competitive seats. Role-specific Elo gives each entry context-appropriate ratings (how it performs against benchmarks, in the live ecosystem, against fresh snapshots, and against historical milestones).

## Context

### Prerequisites

Phase 1 must be complete: OpponentStore, tier managers (FrontierManager, RecentFixedManager, DynamicManager), TieredPool orchestrator, MatchScheduler with round-robin tournament, and the tiered schema (v4) with `role`, `status`, `parent_entry_id`, `lineage_group`, `protection_remaining`, `last_match_at` columns on `league_entries`.

### Source Design

This spec implements Phase 2 of the Kenshi Mixed League design (`docs/concepts/tiered-opponent-pool.md`), sections 4.2, 6.4, 7.4, 9.1–9.3, and 13.2.

### Problem

Phase 1 has a single `elo_rating` column that conflates performance across all tiers. An entry that dominates Recent Fixed opponents but loses to Frontier Static benchmarks gets a blended rating that misrepresents both relationships. The learner has no long-range regression signal — if a training change causes forgetting of skills learned 50k epochs ago, current-era benchmarks won't detect it.

### Goals

1. Add a 5-slot Historical Library with log-spaced milestone selection from archived checkpoints.
2. Run periodic historical gauntlets: learner vs Historical Library entries.
3. Split the single Elo into four role-specific ratings per entry.
4. Provide the dashboard with structured data for a historical panel and multi-Elo display.

### Non-Goals

1. Dynamic training (optimizer state, PPO updates) — Phase 3.
2. Historical Library entries participating in normal league matchmaking — they are benchmark-only.
3. Advanced priority scoring or parallel match scheduling — Phase 4.
4. Dashboard UI implementation — this spec provides the data model and API; UI is a separate task.

---

## Architecture

### Component Overview

```
Phase 1 components (unchanged):
   TieredPool → FrontierManager / RecentFixedManager / DynamicManager
   MatchScheduler → round-robin tournament
   OpponentStore → DB + filesystem

New Phase 2 components:
   HistoricalLibrary        (5-slot milestone manager)
   HistoricalGauntlet       (periodic benchmark runner)
   RoleEloTracker           (per-entry, per-role Elo bookkeeping)
```

### File Layout

| File | Action | Responsibility |
|---|---|---|
| `keisei/training/historical_library.py` | Create | `HistoricalLibrary` class — milestone selection, refresh, slot management |
| `keisei/training/historical_gauntlet.py` | Create | `HistoricalGauntlet` class — periodic learner-vs-history benchmark runner |
| `keisei/training/role_elo.py` | Create | `RoleEloTracker` — computes and stores role-specific Elo updates |
| `keisei/training/opponent_store.py` | Modify | Add `historical_library` table queries, new Elo columns |
| `keisei/training/tiered_pool.py` | Modify | Wire HistoricalLibrary and RoleEloTracker into lifecycle |
| `keisei/training/match_scheduler.py` | Modify | Add gauntlet scheduling to round-robin |
| `keisei/training/katago_loop.py` | Modify | Start gauntlet thread, use role-specific Elo for logging |
| `keisei/config.py` | Modify | Add `HistoricalLibraryConfig` and `RoleEloConfig` |
| `keisei/db.py` | Modify | Schema v5: `historical_library` table, Elo columns on `league_entries` |

---

## Data Model

### Schema Changes (v4 → v5)

**New table `historical_library`:**

| Column | Type | Description |
|---|---|---|
| `slot_index` | INTEGER NOT NULL | 0..4, the milestone slot |
| `target_epoch` | INTEGER NOT NULL | The ideal epoch for this slot (computed by log-spacing formula) |
| `entry_id` | INTEGER | FK to `league_entries` — the archived checkpoint snapped to this slot. NULL if no checkpoint is close enough |
| `actual_epoch` | INTEGER | The `created_epoch` of the snapped entry |
| `selected_at` | TEXT NOT NULL | Timestamp of last selection |
| `selection_mode` | TEXT NOT NULL | `'log_spaced'` or `'fallback'` |

Primary key: `slot_index` (single row per slot, upserted on refresh).

**New columns on `league_entries`:**

| Column | Type | Default | Description |
|---|---|---|---|
| `elo_frontier` | REAL | 1000.0 | Elo derived from matches involving Frontier Static entries |
| `elo_dynamic` | REAL | 1000.0 | Elo derived from matches involving Dynamic entries |
| `elo_recent` | REAL | 1000.0 | Elo derived from matches involving Recent Fixed entries |
| `elo_historical` | REAL | 1000.0 | Elo derived from historical gauntlet matches |

The existing `elo_rating` column is retained as the composite/legacy rating. Phase 2 populates the new columns in parallel; the existing training loop and tournament continue to use `elo_rating` for backward compatibility. Tier managers' eviction/promotion logic continues to use `elo_rating` in Phase 2 — switching to role-specific Elo for decisions is a Phase 4 refinement.

**New table `gauntlet_results`:**

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment |
| `epoch` | INTEGER NOT NULL | Training epoch when gauntlet was run |
| `entry_id` | INTEGER NOT NULL | FK to `league_entries` — the entry being benchmarked (learner or dynamic top-N) |
| `historical_slot` | INTEGER NOT NULL | 0..4, which historical milestone was the opponent |
| `historical_entry_id` | INTEGER NOT NULL | FK to `league_entries` — the historical opponent |
| `wins` | INTEGER NOT NULL | Entry's wins |
| `losses` | INTEGER NOT NULL | Entry's losses |
| `draws` | INTEGER NOT NULL | Draws |
| `elo_before` | REAL | Entry's `elo_historical` before this match |
| `elo_after` | REAL | Entry's `elo_historical` after this match |
| `created_at` | TEXT NOT NULL | Timestamp |

**New index:**

```sql
CREATE INDEX IF NOT EXISTS idx_gauntlet_epoch ON gauntlet_results(epoch);
```

---

## Components

### HistoricalLibrary

Manages the 5-slot milestone library. Does not own models or run matches — it only selects which archived checkpoints fill the slots.

**Config:**

```python
@dataclass(frozen=True)
class HistoricalLibraryConfig:
    slots: int = 5
    refresh_interval_epochs: int = 100
    min_epoch_for_selection: int = 10  # don't select milestones before this epoch
```

**Interface:**

| Method | Description |
|---|---|
| `refresh(current_epoch)` | Recompute log-spaced targets and snap to nearest archived checkpoints |
| `get_slots()` | Returns list of 5 `HistoricalSlot` dataclasses (slot_index, target_epoch, entry, actual_epoch, selection_mode) |
| `is_due_for_refresh(epoch)` | True if epoch aligns with refresh interval |

**Log-spaced target selection:**

For current learner epoch `E` and 5 slots (i = 0..4):

```python
targets = [round(math.exp(math.log(max(E, 2)) * i / 4)) for i in range(5)]
```

This produces approximately:
- At E=10,000: {1, 10, 100, 1000, 10000}
- At E=250,000: {1, 22, 500, 11000, 250000}

**Snapping to nearest checkpoint:**

For each target epoch, find the archived checkpoint (any entry in `league_entries` regardless of `status`) whose `created_epoch` is closest to the target. Prefer entries with `status = 'retired'` or `status = 'archived'` (these are stable — active entries might get their role changed). If no checkpoint is within 50% of the target epoch distance to its neighbors, leave the slot empty (entry_id = NULL).

**Early-training fallback:**

When fewer than 5 distinct milestones exist (e.g., training is young):
- Fill available slots with the closest available frozen checkpoints.
- Use `selection_mode = 'fallback'` to mark these as approximate.
- The dashboard always shows 5 slots, even if some are fallback or NULL.

**Refresh is idempotent:** Running refresh twice at the same epoch produces the same result. The `selected_at` timestamp updates but the slot assignments don't change unless the checkpoint archive has grown.

### HistoricalGauntlet

A periodic benchmark runner that pits the learner (and optionally dynamic top-N) against all Historical Library entries. Runs as a background thread, similar to `LeagueTournament`.

**Config:**

```python
@dataclass(frozen=True)
class GauntletConfig:
    enabled: bool = True
    interval_epochs: int = 100  # run gauntlet every N epochs
    games_per_matchup: int = 16  # games per (entry, historical_slot) pair
    include_dynamic_topn: int = 0  # also benchmark top-N dynamic entries (0 = learner only)
```

**Interface:**

| Method | Description |
|---|---|
| `run_gauntlet(epoch, learner_entry, historical_slots)` | Play learner vs each historical slot, record results |
| `is_due(epoch)` | True if epoch aligns with gauntlet interval |

**Gauntlet flow:**

1. `TieredPool.on_epoch_end(epoch)` checks `gauntlet.is_due(epoch)`.
2. If due, calls `historical_library.refresh(epoch)` to update slot assignments.
3. Calls `gauntlet.run_gauntlet(epoch, learner_entry, slots)`.
4. For each non-empty slot: load the historical model, play `games_per_matchup` games against the learner model.
5. Record results to `gauntlet_results` table.
6. Update `elo_historical` on the learner entry via `RoleEloTracker`.

**Threading:** The gauntlet can run on the tournament thread (since the tournament pauses between rounds anyway) or on its own thread. The simplest approach is to run it synchronously at the end of the tournament round when `is_due(epoch)` triggers — this avoids a third background thread and uses the same GPU. The tournament's `_stop_event` is checked between gauntlet matchups for graceful shutdown.

**Model loading:** Historical models are loaded via `OpponentStore.load_opponent(entry, device)`, same as tournament models. They are loaded on demand and released after the gauntlet completes — they do not persist in memory between gauntlets.

### RoleEloTracker

Computes and stores role-specific Elo updates. Wraps the existing `compute_elo_update` function with role-awareness.

**Interface:**

| Method | Description |
|---|---|
| `update_from_result(entry_a, entry_b, result_score, match_context)` | Compute role-specific Elo deltas based on opponent roles, write to DB |
| `get_role_elos(entry_id)` | Returns dict of `{role: elo_value}` for an entry |

**Match context determines which Elo column is updated:**

| Match involves | Elo column updated on both participants |
|---|---|
| At least one Frontier Static entry | `elo_frontier` |
| Dynamic vs Dynamic | `elo_dynamic` |
| Dynamic vs Recent Fixed | `elo_dynamic` on Dynamic, `elo_recent` on Recent Fixed |
| Recent Fixed vs Recent Fixed | `elo_recent` |
| Historical gauntlet | `elo_historical` |

**K-factors per context (from concept doc section 9.3):**

```python
@dataclass(frozen=True)
class RoleEloConfig:
    frontier_k: float = 16.0    # benchmark should move slowly
    dynamic_k: float = 24.0     # live ecosystem can move faster
    recent_k: float = 32.0      # recent snapshots need rapid calibration
    historical_k: float = 12.0  # historical score should be very stable
```

**Integration with existing Elo:** The existing `elo_rating` column continues to be updated by `OpponentStore.update_elo()` in the training loop and tournament. `RoleEloTracker` updates the *additional* columns (`elo_frontier`, `elo_dynamic`, `elo_recent`, `elo_historical`) based on match context. Both systems run in parallel — the composite `elo_rating` remains the operational number for eviction/promotion in Phase 2.

The `record_result` method on OpponentStore is extended to accept an optional `match_context: str` parameter. When provided, `RoleEloTracker.update_from_result` is called to update the appropriate role-specific Elo column.

---

## Config Changes

New nested config dataclasses in `keisei/config.py`:

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

Added to `LeagueConfig`:

```python
@dataclass(frozen=True)
class LeagueConfig:
    # ... existing Phase 1 fields ...
    
    # Phase 2: Historical Library
    history: HistoricalLibraryConfig = HistoricalLibraryConfig()
    gauntlet: GauntletConfig = GauntletConfig()
    role_elo: RoleEloConfig = RoleEloConfig()
```

---

## Integration Points

### TieredPool

| Change | Description |
|---|---|
| Constructor | Creates `HistoricalLibrary(store, config.history)` and `RoleEloTracker(store, config.role_elo)` |
| `on_epoch_end(epoch)` | After Frontier review check, also checks `historical_library.is_due_for_refresh(epoch)` and triggers refresh + gauntlet |
| `get_historical_slots()` | New method — delegates to `historical_library.get_slots()` |

### MatchScheduler

The round-robin tournament already produces all active-league pairings. The gauntlet is a separate operation that does not go through the scheduler — it's a fixed set of (learner, historical_entry) matchups run by `HistoricalGauntlet` directly.

No changes to `MatchScheduler.generate_round()` or `sample_for_learner()`.

### Tournament Runner (LeagueTournament)

After each round-robin round completes, check if a gauntlet is due:

```python
# After round-robin matches complete
if self.gauntlet and self.gauntlet.is_due(epoch):
    self.historical_library.refresh(epoch)
    slots = self.historical_library.get_slots()
    learner_entry = self.store._get_entry(self._learner_entry_id)
    if learner_entry and slots:
        self.gauntlet.run_gauntlet(epoch, learner_entry, slots)
```

The gauntlet uses the same VecEnv and device as the tournament — no additional GPU resources needed.

### KataGoTrainingLoop

| Change | Description |
|---|---|
| Tournament construction | Pass `historical_library` and `gauntlet` to `LeagueTournament` |
| Logging | Log `elo_historical` alongside `elo_rating` at epoch end |

### OpponentStore

| New method | Description |
|---|---|
| `update_role_elo(entry_id, role_column, new_elo)` | Update a specific role Elo column (`elo_frontier`, `elo_dynamic`, `elo_recent`, `elo_historical`) |
| `upsert_historical_slot(slot_index, target_epoch, entry_id, actual_epoch, selection_mode)` | Insert or update a historical library slot |
| `get_historical_slots()` | Read all 5 slots from `historical_library` table, joined with `league_entries` for entry data |
| `record_gauntlet_result(...)` | Insert into `gauntlet_results` |

### Dashboard / read_league_data

`read_league_data` in `db.py` is extended to include:
- The four role-specific Elo columns in the entries query
- A new `historical_library` key with the 5 slots and their assigned checkpoints
- A new `gauntlet_results` key with recent gauntlet outcomes

---

## Monitoring

1. **Historical Gauntlet Score trend** — the primary regression signal. If the learner's `elo_historical` drops over consecutive gauntlets, something is being forgotten.
2. **Milestone epoch coverage** — verify the 5 slots span the full training range. If slots bunch up (e.g., all in the last 10% of training), the log-spacing formula may need a larger base.
3. **Gauntlet game volume** — 5 slots × 16 games = 80 games per gauntlet. At one gauntlet per 100 epochs, this is minimal overhead.
4. **Role Elo divergence** — if `elo_frontier` and `elo_dynamic` diverge significantly for the same entry, it reveals specialization (good) or inconsistency (investigate).

---

## Testing Strategy

### Unit Tests

- **HistoricalLibrary:** log-spaced target computation at various epochs (10, 1000, 100000), snapping to nearest checkpoint, early-training fallback when < 5 checkpoints exist, refresh idempotency, slot with no nearby checkpoint returns NULL entry.
- **HistoricalGauntlet:** run_gauntlet records results to DB, gauntlet skips empty slots, is_due returns correct epoch alignment.
- **RoleEloTracker:** Frontier match updates `elo_frontier` on both entries, Dynamic-vs-Dynamic updates `elo_dynamic`, cross-tier match updates correct columns, K-factors are role-specific, composite `elo_rating` is NOT modified by role tracker.

### Integration Tests

- Full gauntlet cycle: create entries at various epochs, retire them, run `historical_library.refresh()`, verify 5 slots are filled with sensible checkpoints, run gauntlet, verify `gauntlet_results` and `elo_historical` are populated.
- Role Elo accumulation: run a tournament round, verify role-specific Elo columns are updated based on opponent roles while `elo_rating` is updated independently.

---

## Phase 2 Boundaries

**In scope:**
- `historical_library` table and `HistoricalLibrary` class with log-spaced selection
- `gauntlet_results` table and `HistoricalGauntlet` runner
- Four role-specific Elo columns + `RoleEloTracker`
- Per-context K-factors
- Integration with tournament runner (gauntlet runs after round-robin)
- Dashboard data model extensions

**Explicitly deferred:**
- Using role-specific Elo for eviction/promotion decisions → Phase 4 (currently uses composite `elo_rating`)
- Dynamic training → Phase 3
- Parallel match scheduling → Phase 4
- Dashboard UI implementation → separate task

---

## Remaining Phases

- **Phase 3 — Dynamic Training:** Enable optimizer state persistence for Dynamic entries, small PPO updates from league matches, update caps and checkpoint flush, protection windows with fault fallback, Frontier Static review activation (Dynamic → Frontier promotion).
- **Phase 4 — League Concurrency & Refinement:** Multiple simultaneous pairings on the league GPU, advanced scheduler priority scoring (uncertainty bonus, lineage penalty, diversity bonus), switch eviction/promotion to use role-specific Elo instead of composite.
