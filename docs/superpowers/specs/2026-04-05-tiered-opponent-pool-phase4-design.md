# Tiered Opponent Pool — Phase 4: League Concurrency & Refinement

## Purpose

Run multiple tournament pairings concurrently on the league GPU to increase calibration throughput, add priority scoring so the scheduler favors informative matchups, and switch eviction/promotion decisions to use role-specific Elo instead of the composite rating. This is the final phase — it turns the league from functional into optimized.

## Context

### Prerequisites

Phases 1–3 must be complete:
- Phase 1: OpponentStore, TieredPool, tier managers, MatchScheduler with round-robin, schema v4.
- Phase 2: HistoricalLibrary, HistoricalGauntlet, RoleEloTracker, schema v5 with role-specific Elo columns.
- Phase 3: DynamicTrainer, FrontierPromoter, optimizer persistence, match data collection, schema v6.

### Source Design

This spec implements Phase 4 of the Kenshi Mixed League design (`docs/concepts/tiered-opponent-pool.md`), sections 8.4 (pair priority), 8.5 (concurrency), and 12 (diversity controls).

### Problem

The Phase 1–3 tournament processes 190 round-robin pairings sequentially. Each pairing loads two models, plays 3 games, records results, and optionally runs a Dynamic training step — all serial. With 9 GB free on cuda:1, the GPU is idle most of the time (model loading, DB writes, sleep). Concurrent pairings would overlap inference with I/O, increasing throughput.

Additionally, the round-robin format treats all pairings equally. In practice, some matchups are more informative than others — under-calibrated pairs, newly admitted entries, and pairs with high Elo uncertainty should be prioritized. The current flat round-robin wastes calibration budget on well-known matchups.

Finally, Phase 2 added four role-specific Elo columns, but eviction and promotion still use the composite `elo_rating`. Switching to `elo_dynamic` for Dynamic eviction and `elo_frontier` for Frontier promotion makes these decisions more accurate — an entry's performance against its own tier is more relevant than its blended performance.

### Goals

1. Run multiple tournament pairings concurrently on cuda:1 to increase throughput.
2. Add priority scoring to the scheduler so informative matchups are played first.
3. Switch DynamicManager eviction to use `elo_dynamic` instead of `elo_rating`.
4. Switch FrontierPromoter promotion criteria to use `elo_frontier` instead of `elo_rating`.
5. Add lineage-aware scheduling to penalize overuse of parent/child and close-sibling matchups.

### Non-Goals

1. Multi-GPU tournament (pairings spread across multiple GPUs) — single GPU is sufficient.
2. Dynamic training parallelism (concurrent PPO updates for different entries) — training remains sequential on the tournament thread.
3. Changing the learner's sampling mix ratios — these remain 50/30/20 Dynamic/Frontier/Recent.
4. Real-time dashboard updates — the dashboard polls the DB; no push mechanism needed.

---

## Architecture

### Component Overview

```
Phase 1-3 components (modified):
   MatchScheduler    → priority scoring replaces flat round-robin ordering
   DynamicManager    → eviction uses elo_dynamic
   FrontierPromoter  → promotion uses elo_frontier
   LeagueTournament  → concurrent match execution via worker pool

New Phase 4 components:
   PriorityScorer       (pair scoring function)
   ConcurrentMatchPool  (manages parallel match execution)
```

### File Layout

| File | Action | Responsibility |
|---|---|---|
| `keisei/training/priority_scorer.py` | Create | `PriorityScorer` — computes priority scores for matchups |
| `keisei/training/concurrent_matches.py` | Create | `ConcurrentMatchPool` — manages parallel match workers on one GPU |
| `keisei/training/match_scheduler.py` | Modify | `generate_round` returns priority-ordered pairings instead of shuffled |
| `keisei/training/tier_managers.py` | Modify | DynamicManager.evict_weakest uses `elo_dynamic` |
| `keisei/training/frontier_promoter.py` | Modify | FrontierPromoter.should_promote uses `elo_frontier` |
| `keisei/training/tournament.py` | Modify | Use ConcurrentMatchPool instead of sequential loop |
| `keisei/config.py` | Modify | Add `ConcurrencyConfig` and `PriorityScorerConfig` |

---

## Components

### PriorityScorer

Computes a priority score for each candidate pairing. Higher-priority pairings are played first in each round. This replaces the random shuffle in `generate_round`.

**Config:**

```python
@dataclass(frozen=True)
class PriorityScorerConfig:
    under_sample_weight: float = 1.0      # bonus for pairs with few past games
    uncertainty_weight: float = 0.5        # bonus for pairs with high Elo uncertainty
    recent_fixed_bonus: float = 0.3        # bonus for pairings involving Recent Fixed entries
    diversity_weight: float = 0.3          # bonus for cross-lineage pairings
    repeat_penalty: float = -0.5           # penalty for pairs that have played recently
    lineage_penalty: float = -0.3          # penalty for parent/child or close sibling pairings
```

**Interface:**

| Method | Description |
|---|---|
| `score(entry_a, entry_b, match_history)` | Returns a float priority score for this pairing |
| `score_round(pairings, match_history)` | Score all pairings and return sorted by priority descending |

**Priority formula:**

```
priority =
    under_sample_weight * under_sample_bonus(a, b)
  + uncertainty_weight * uncertainty_bonus(a, b)
  + recent_fixed_bonus * has_recent_fixed(a, b)
  + diversity_weight * lineage_diversity(a, b)
  + repeat_penalty * repeat_count(a, b)
  + lineage_penalty * lineage_closeness(a, b)
```

**Component definitions:**

- `under_sample_bonus(a, b)`: `1.0 / max(1, pair_game_count(a.id, b.id))` — pairs with fewer past games get higher priority. Queried from `league_results` with a `SELECT COUNT(*)` for the pair.
- `uncertainty_bonus(a, b)`: `abs(a.elo_rating - b.elo_rating) < 100 ? 1.0 : 0.0` — pairs within 100 Elo points have high uncertainty about who is stronger. This is a simplified proxy; a proper Bayesian uncertainty model is overkill for Phase 4.
- `has_recent_fixed(a, b)`: `1.0` if either entry has `role == Role.RECENT_FIXED`, else `0.0`. Recent Fixed entries need fast calibration.
- `lineage_diversity(a, b)`: `1.0` if `a.lineage_group != b.lineage_group` (or either is None), else `0.0`. Cross-lineage matchups reveal more about the population than same-lineage ones.
- `repeat_count(a, b)`: number of times this pair has played in the last `N` rounds (tracked in-memory). Higher repeat count → larger penalty.
- `lineage_closeness(a, b)`: `1.0` if `a.parent_entry_id == b.id` or `b.parent_entry_id == a.id` (direct parent/child), `0.5` if same `lineage_group`, else `0.0`.

**Match history:** The scorer maintains an in-memory `Counter[tuple[int, int]]` tracking pair play counts within a sliding window of `N` rounds (default 5). This is the mechanism Phase 1's `_h2h_counts` was scaffolding for but never implemented.

### ConcurrentMatchPool

Manages parallel match execution on a single GPU. Instead of loading two models, playing, unloading for each pairing sequentially, the pool keeps a configurable number of model pairs loaded and interleaves their game steps.

**Config:**

```python
@dataclass(frozen=True)
class ConcurrencyConfig:
    parallel_matches: int = 4          # number of concurrent pairings
    envs_per_match: int = 8            # VecEnv size per pairing
    total_envs: int = 32               # parallel_matches * envs_per_match
    max_resident_models: int = 10      # max models loaded on GPU simultaneously
```

**Constraint:** `parallel_matches * envs_per_match <= total_envs`. Validated at construction.

**Interface:**

| Method | Description |
|---|---|
| `run_round(pairings, play_fn)` | Execute all pairings with up to `parallel_matches` concurrent. Returns list of results. |

**Execution model:**

This is NOT Python threading parallelism (GIL prevents true parallel PyTorch). Instead, it's an **interleaved game loop with resident models and batched environment stepping**:

1. Load up to `parallel_matches` pairs of models (up to `max_resident_models` total).
2. Create a single large VecEnv with `total_envs` environments.
3. Partition the envs: envs 0-7 belong to pair 0, envs 8-15 to pair 1, etc.
4. Each step:
   a. For each partition, run the appropriate model's forward pass on that partition's observations. This is `parallel_matches` separate small forward passes (one per model pair), NOT one big batched forward pass — different partitions use different models, so they cannot be batched into a single kernel.
   b. Step the VecEnv with all actions at once (this IS truly batched — one call for `total_envs` environments).
   c. Check for completed games per partition.
5. When a partition's games are all done, record the results and load the next pairing into that partition's model slot.

The throughput gain comes from three sources: (1) **models stay resident** — no load/unload between matches, avoiding repeated disk I/O and `torch.load` overhead; (2) **VecEnv stepping is batched** — one Rust call advances all `total_envs` environments; (3) **game loops are interleaved** — all active partitions advance each iteration, so slow I/O in one partition doesn't block others from progressing. The inference itself is `parallel_matches` small forward passes per step, each at batch size `envs_per_match`.

**Why this works on one GPU:** Each model is ~120 MB. With `max_resident_models=10`, that's 1.2 GB. The VecEnv observations for `total_envs=32` are tiny (~2 MB). Activations for a forward pass at batch `envs_per_match=8` are ~12 MB per partition. Total additional VRAM for concurrency: ~1.3 GB. With 9 GB free this is comfortable.

**Why not threads:** PyTorch CUDA operations from multiple Python threads serialize on the GIL anyway. The interleaved approach is simpler, more predictable, and avoids stream synchronization complexity. Each forward pass is smaller (batch 8 vs batch 64) but there are more of them; the net effect is similar GPU utilization with much simpler code.

**Dynamic training interaction:** When a trainable match completes in a partition, the training step still runs sequentially (Phase 3's DynamicTrainer). The other partitions pause during the training step because they share the CUDA stream. This is acceptable — training steps are short (~100ms) relative to match duration (~seconds).

### MatchScheduler Changes

`generate_round(entries)` now returns pairings sorted by priority score (highest first) instead of randomly shuffled.

```python
def generate_round(self, entries: list[OpponentEntry]) -> list[tuple[OpponentEntry, OpponentEntry]]:
    pairings = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            pairings.append((entries[i], entries[j]))
    # Score and sort by priority (highest first)
    scored = self.priority_scorer.score_round(pairings, self._match_history)
    return scored
```

The `ConcurrentMatchPool` processes pairings in order, so high-priority pairings run first. If the round is interrupted (stop_event), the most informative pairings have already been played.

### DynamicManager Changes

`evict_weakest()` changes from using `elo_rating` to `elo_dynamic`:

```python
# Before (Phase 1-3):
eligible.sort(key=lambda e: e.elo_rating)

# After (Phase 4):
eligible.sort(key=lambda e: e.elo_dynamic)
```

This is a one-line change. The rationale: an entry's performance against other Dynamic entries (`elo_dynamic`) is more relevant for eviction than its blended performance across all tiers. An entry that is weak against Dynamic opponents but strong against Frontier Static benchmarks is not providing useful competitive pressure in the ecosystem — evict it.

### FrontierPromoter Changes

`should_promote` criterion 4 changes from composite Elo to frontier-specific:

```python
# Before (Phase 1-3):
weakest_frontier_elo = min(f.elo_rating for f in frontier_entries)
elo_qualified = candidate.elo_rating >= weakest_frontier_elo + self.config.promotion_margin_elo

# After (Phase 4):
weakest_frontier_elo = min(f.elo_frontier for f in frontier_entries)
elo_qualified = candidate.elo_frontier >= weakest_frontier_elo + self.config.promotion_margin_elo
```

Also a small change. The rationale: `elo_frontier` measures how well the candidate performs specifically against Frontier Static benchmarks — exactly the context it's being promoted into.

### RecentFixedManager Changes (optional refinement)

The promotion threshold for Recent Fixed → Dynamic could switch to `elo_dynamic` for the Dynamic floor comparison:

```python
# Before:
floor_elo = self.dynamic_manager.weakest_elo()  # uses elo_rating

# After:
floor_elo = self.dynamic_manager.weakest_dynamic_elo()  # uses elo_dynamic
```

This is a minor refinement. If it complicates the manager interface, it can be deferred.

---

## Config Changes

New nested config dataclasses:

```python
@dataclass(frozen=True)
class PriorityScorerConfig:
    under_sample_weight: float = 1.0
    uncertainty_weight: float = 0.5
    recent_fixed_bonus: float = 0.3
    diversity_weight: float = 0.3
    repeat_penalty: float = -0.5
    lineage_penalty: float = -0.3
    repeat_window_rounds: int = 5

@dataclass(frozen=True)
class ConcurrencyConfig:
    parallel_matches: int = 4
    envs_per_match: int = 8
    total_envs: int = 32
    max_resident_models: int = 10
```

Added to `LeagueConfig`:

```python
    priority: PriorityScorerConfig = PriorityScorerConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
```

**Validation:**
- `concurrency.parallel_matches * concurrency.envs_per_match <= concurrency.total_envs`
- `concurrency.max_resident_models >= concurrency.parallel_matches * 2` (each pairing needs 2 models)
- All weights/penalties in PriorityScorerConfig are finite

---

## Integration Points

### Tournament Runner

The tournament's `_run_loop` is refactored from:
```python
for entry_a, entry_b in pairings:
    wins_a, wins_b, draws = self._play_match(vecenv, entry_a, entry_b)
```
To:
```python
results = self.concurrent_pool.run_round(pairings, play_fn=self._play_partition)
for entry_a, entry_b, wins_a, wins_b, draws, rollout in results:
    self._record_and_train(entry_a, entry_b, wins_a, wins_b, draws, rollout)
```

The VecEnv is now created with `total_envs` instead of `num_envs`. The sequential `_play_match` + `_play_batch` pattern is replaced by `ConcurrentMatchPool.run_round` which manages all partitions.

### Learner Opponent Sampling

No changes. The learner still samples opponents via MatchScheduler.sample_for_learner with the same tier mix ratios.

### Dashboard

No schema changes. The existing `league_results` table captures all match data. The priority scorer is transparent to the dashboard — it just affects which matches are played, not how results are recorded.

---

## Monitoring

1. **Tournament throughput** — pairings completed per minute. Should increase with concurrency. If it doesn't, the bottleneck is elsewhere (DB writes, training steps, model loading).
2. **Priority score distribution** — histogram of scores per round. If all scores cluster near zero, the priority weights need tuning. If the top scores are always the same pair, the repeat penalty needs strengthening.
3. **GPU utilization** — should increase from ~30% (sequential) to ~60-80% (concurrent) depending on `parallel_matches`. Monitor with `nvidia-smi`.
4. **Role-specific Elo divergence after operational switch** — `elo_dynamic` and `elo_rating` will diverge as role-specific Elo drives different eviction decisions. Track whether the Dynamic tier becomes more or less diverse after the switch.
5. **Lineage penalty effectiveness** — track the fraction of same-lineage matchups per round. Should decrease after the penalty is activated.

---

## Testing Strategy

### Unit Tests

- **PriorityScorer:** `score` returns higher value for under-sampled pairs, `score` penalizes repeat pairings, lineage penalty works for parent/child and same-group, uncertainty bonus fires for close Elo, Recent Fixed bonus works, all weights at zero produce score of zero.
- **ConcurrentMatchPool:** `run_round` processes correct number of pairings, partitioning is correct (env ranges don't overlap), results are returned in order, interrupted round returns partial results, `max_resident_models` is respected.
- **MatchScheduler:** `generate_round` returns pairings sorted by priority (highest first), not random.
- **DynamicManager:** `evict_weakest` uses `elo_dynamic` not `elo_rating` — entry with low `elo_dynamic` but high `elo_rating` is evicted.
- **FrontierPromoter:** `should_promote` uses `elo_frontier` — entry with high `elo_frontier` but low `elo_rating` passes criterion 4.

### Integration Tests

- Full round with priority scoring: create entries with varying games_played and Elo, run `generate_round`, verify the most under-sampled pair is first.
- Concurrent round execution: run `ConcurrentMatchPool.run_round` with mocked models, verify all pairings complete and results are recorded.
- Role-specific Elo eviction: two Dynamic entries with same `elo_rating` but different `elo_dynamic`, verify the lower `elo_dynamic` is evicted.

---

## Phase 4 Boundaries

**In scope:**
- ConcurrentMatchPool with batched inference across partitioned VecEnv
- PriorityScorer with 6-component scoring formula
- Priority-ordered round-robin (high-priority pairs first)
- DynamicManager eviction uses `elo_dynamic`
- FrontierPromoter promotion uses `elo_frontier`
- Lineage-aware scheduling (parent/child and sibling penalties)
- Repeat-pair penalty with sliding window

**Explicitly deferred (no further phases planned):**
- Multi-GPU tournament distribution
- Dynamic training parallelism (concurrent PPO for different entries)
- Bayesian Elo uncertainty (replaced with simplified 100-point-range proxy)
- Adaptive tier mix ratios for the learner

---

## Design Notes

### Why Batched Inference, Not Threads

Python's GIL prevents true parallel PyTorch execution from threads. Using `torch.cuda.Stream` per pairing is possible but adds complexity (stream synchronization, memory management across streams). The batched approach — one big forward pass across all partitions — is simpler and actually more GPU-efficient because it maximizes kernel occupancy. The tradeoff is that all partitions advance at the speed of the slowest partition per step, but for uniform-architecture models this is negligible.

### Why Not asyncio

PyTorch operations are blocking. `asyncio` would only help with I/O (DB writes, file saves), not with inference or training. The I/O is already fast enough (SQLite WAL, atomic file rename) that the complexity of an async event loop isn't justified.

### Operational Switch to Role-Specific Elo

The switch from `elo_rating` to `elo_dynamic`/`elo_frontier` for operational decisions is intentionally a Phase 4 item because it requires Phase 2's role-specific Elo to be well-calibrated before it drives decisions. After Phases 2-3 have run for enough epochs, the role-specific ratings have accumulated sufficient data to be reliable. Switching earlier would use under-calibrated role Elo for eviction/promotion, producing worse decisions than the composite.
