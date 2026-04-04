# Tiered Opponent Pool — Phase 3: Dynamic Training & Frontier Promotion

## Purpose

Enable the 10 Dynamic entries to train via small PPO updates from league matches, and activate the Frontier Static review/promotion pipeline so proven Dynamic entries can graduate to become stable benchmarks. This transforms the league from a pure evaluation system into a live adaptive ecosystem that generates competitive pressure the learner cannot trivially exploit.

## Context

### Prerequisites

Phases 1 and 2 must be complete:
- Phase 1: OpponentStore (RLock, `_in_transaction` commit discipline), TieredPool, FrontierManager (with `review()` as no-op), DynamicManager (with `training_enabled=False` guard), round-robin tournament, schema v4.
- Phase 2: HistoricalLibrary, HistoricalGauntlet, RoleEloTracker with per-context K-factors, schema v5 with role-specific Elo columns.

### Source Design

This spec implements Phase 3 of the Kenshi Mixed League design (`docs/concepts/tiered-opponent-pool.md`), sections 6.3 (Dynamic behavior), 7.3 (Dynamic → Frontier Static), 10.1–10.4 (Dynamic training mechanics), and 12 (diversity controls).

### Problem

In Phases 1–2, all 20 league entries are frozen — they provide measurement and calibration but no adaptive pressure. The learner can find strategies that exploit specific frozen opponents without those opponents ever adapting. Dynamic training solves this: 10 entries receive small PPO updates from their league matches, developing counters and anti-counters that keep the learner honest.

Additionally, `FrontierManager.review()` is a no-op in Phases 1–2, meaning Frontier Static benchmarks are frozen at their bootstrap values. As the learner outpaces them, the benchmark signal degrades. Phase 3 activates the promotion pipeline: proven Dynamic entries can spawn frozen Frontier Static clones, marching the benchmark tier up the skill ladder.

### Goals

1. Enable PPO training for Dynamic entries from league match data.
2. Persist optimizer state alongside weights for Dynamic entries.
3. Implement safety rails: update caps, checkpoint flush limits, and automatic fallback to inference-only mode.
4. Activate `FrontierManager.review()`: promote proven Dynamic entries to Frontier Static.
5. Implement the Frontier Static replacement policy (retire weakest/stalest after cooldown).

### Non-Goals

1. Training from learner PPO batches — Dynamic entries train only from league matches.
2. Training for Frontier Static, Recent Fixed, or Historical Library entries.
3. Parallel match scheduling or advanced priority scoring — Phase 4.
4. Co-training with the learner (shared gradient paths, shared optimizer) — Dynamic training is fully independent.

---

## Architecture

### Component Overview

```
Phase 1-2 components (modified):
   DynamicManager      → training_enabled=True, optimizer handling
   FrontierManager     → review() activated, promotion pipeline
   OpponentStore       → optimizer_path column, save/load optimizer
   LeagueTournament    → training step after trainable matches

New Phase 3 components:
   DynamicTrainer      (small PPO updates for Dynamic entries)
   FrontierPromoter    (conservative promotion evaluation)
```

### File Layout

| File | Action | Responsibility |
|---|---|---|
| `keisei/training/dynamic_trainer.py` | Create | `DynamicTrainer` — small PPO updates for Dynamic entries from league match data |
| `keisei/training/frontier_promoter.py` | Create | `FrontierPromoter` — evaluates Dynamic entries for Frontier Static promotion |
| `keisei/training/opponent_store.py` | Modify | Add `optimizer_path` column support, `save_optimizer`, `load_optimizer` |
| `keisei/training/tier_managers.py` | Modify | Remove `training_enabled=False` guard from DynamicManager, add training hooks |
| `keisei/training/tiered_pool.py` | Modify | Wire DynamicTrainer and FrontierPromoter |
| `keisei/training/tournament.py` | Modify | Call DynamicTrainer after trainable matches |
| `keisei/config.py` | Modify | Extend `DynamicConfig` with training parameters |
| `keisei/db.py` | Modify | Schema v6: `optimizer_path`, `last_train_at`, `update_count` columns |

---

## Data Model

### Schema Changes (v5 → v6)

**New columns on `league_entries`:**

| Column | Type | Default | Description |
|---|---|---|---|
| `optimizer_path` | TEXT | NULL | Path to saved optimizer state (Dynamic entries only) |
| `update_count` | INTEGER | 0 | Number of training updates applied to this entry |
| `last_train_at` | TEXT | NULL | Timestamp of most recent training update |

These columns are NULL/0 for all non-Dynamic entries. Dynamic entries with `training_enabled=True` populate them after each training step.

### Filesystem Layout

Dynamic entries gain an optimizer file alongside their weights:

```
league/
  resnet_ep00100_id42.pt           # weights (all entries have this)
  resnet_ep00100_id42_optimizer.pt  # optimizer state (Dynamic only)
```

The optimizer file is saved atomically (write to `.tmp`, rename) following the existing checkpoint pattern. On Recent Fixed → Dynamic promotion, a fresh optimizer is created — the source entry's optimizer (if any) is never inherited.

---

## Components

### DynamicTrainer

Runs small PPO updates on Dynamic entries using data collected from league matches. This is NOT the learner's PPO — it's a lightweight, independent training step that gives Dynamic entries just enough adaptation to develop counter-strategies.

**Config (extends existing `DynamicConfig`):**

```python
@dataclass(frozen=True)
class DynamicConfig:
    # Existing Phase 1 fields
    slots: int = 10
    protection_matches: int = 24
    min_games_before_eviction: int = 40
    training_enabled: bool = True  # Phase 3: now True by default

    # New Phase 3 training fields
    update_epochs_per_batch: int = 2      # PPO epochs per training batch
    batch_reuse: int = 1                  # how many times to reuse a batch
    lr_scale: float = 0.25               # learning rate relative to learner LR
    grad_clip: float = 1.0               # gradient clipping norm
    update_every_matches: int = 4         # accumulate N matches before one update
    max_updates_per_minute: int = 20      # hard cap on update rate
    checkpoint_flush_every: int = 8       # save weights+optimizer every N matches
    disable_on_error: bool = True         # fall back to inference-only on training error
```

**Interface:**

| Method | Description |
|---|---|
| `should_update(entry_id)` | True if entry has accumulated enough match data since last update |
| `update(entry, match_data, device)` | Run PPO update on the entry's model using collected match data. Returns updated model or None on error |
| `is_rate_limited()` | True if update rate cap has been reached |
| `get_update_stats(entry_id)` | Returns update_count, last_train_at for an entry |

**Training rule (from concept doc):**

Only these match classes produce training data for Dynamic entries:
- **Dynamic vs Dynamic**: both entries may update from their own perspective
- **Dynamic vs Recent Fixed**: Dynamic entry may update, Recent Fixed never updates

These match classes do NOT produce training data:
- Dynamic vs Frontier Static (calibration only)
- Any match involving Historical Library entries

**Update mechanics:**

1. Tournament plays a trainable match (D-vs-D or D-vs-RF).
2. The match rollout data (observations, actions, rewards, dones) is collected.
3. If the Dynamic entry has accumulated `update_every_matches` worth of data, `DynamicTrainer.update()` is called.
4. The update runs `update_epochs_per_batch` PPO epochs on the accumulated batch with:
   - Learning rate = `learner_lr * lr_scale`
   - Gradient clipping at `grad_clip`
   - Standard PPO clipped objective (same loss function as learner, minus the score head — Dynamic entries don't need score prediction)
5. Updated weights are written back to the entry's checkpoint file.
6. Optimizer state is saved every `checkpoint_flush_every` matches.
7. `update_count` and `last_train_at` are updated in the DB.

**Safety rails:**

- **Rate limiting:** No more than `max_updates_per_minute` updates across all Dynamic entries combined. `is_rate_limited()` tracks a sliding window.
- **Error fallback:** If a training update raises an exception (NaN loss, CUDA error, etc.) and `disable_on_error=True`, the DynamicTrainer logs the error and disables training for that entry. The entry remains in the pool as a frozen inference-only opponent. A transition is logged: `"training disabled due to error: {error}"`.
- **Checkpoint frequency:** Optimizer state is expensive to save. Only flush every `checkpoint_flush_every` matches. Weights are always saved after an update (they're small relative to optimizer state).

**What DynamicTrainer does NOT do:**
- It does not manage admission/eviction (that's DynamicManager).
- It does not decide which matches are trainable (that's the tournament via match class).
- It does not modify the learner (the learner has its own PPO loop).

### FrontierPromoter

Evaluates whether any Dynamic entry qualifies for promotion to Frontier Static. Called by `FrontierManager.review()` which is activated in Phase 3.

**Config (extends existing `FrontierStaticConfig`):**

```python
@dataclass(frozen=True)
class FrontierStaticConfig:
    # Existing Phase 1 fields
    slots: int = 5
    review_interval_epochs: int = 250
    min_tenure_epochs: int = 100
    promotion_margin_elo: float = 50.0

    # New Phase 3 promotion fields
    min_games_for_promotion: int = 64     # minimum calibration matches
    topk: int = 3                         # must be in top-K Dynamic by Elo
    streak_epochs: int = 50               # must have held top-K for this many epochs
    max_lineage_overlap: int = 2          # max entries sharing a lineage_group in Frontier
```

**Interface:**

| Method | Description |
|---|---|
| `evaluate(dynamic_entries, frontier_entries, epoch)` | Returns the best promotion candidate or None |
| `should_promote(candidate, frontier_entries)` | True if a specific candidate meets all criteria |

**Promotion criteria (all must be met):**

1. `games_played >= min_games_for_promotion` (64)
2. Entry is in the top-K Dynamic entries by `elo_rating` (top 3)
3. Entry has been in top-K for at least `streak_epochs` consecutive epochs (50)
4. Entry's `elo_rating` exceeds the weakest Frontier Static's `elo_rating` by `promotion_margin_elo` (50 points)
5. The entry's `lineage_group` does not already have `max_lineage_overlap` entries in the Frontier Static tier (prevents clones of the same snapshot dominating the benchmark tier)

**Promotion flow (orchestrated by TieredPool):**

1. `FrontierManager.review(epoch)` calls `FrontierPromoter.evaluate()`.
2. If a candidate is found:
   a. Clone the candidate into a new Frontier Static entry via `store.clone_entry()`. The clone is frozen — no optimizer state is copied.
   b. Retire the weakest or stalest Frontier Static entry that has exceeded `min_tenure_epochs`.
   c. Never replace more than one Frontier Static per review window.
   d. Log the transition.
3. The Dynamic entry that was promoted continues to exist and train in the Dynamic tier — promotion is by cloning, not moving.

**Streak tracking:**

The `streak_epochs` criterion requires knowing how long an entry has been in the top-K. This is tracked via a simple in-memory dict on `FrontierPromoter`: `{entry_id: first_seen_in_topk_epoch}`. On each review:
- For each top-K Dynamic entry, if it wasn't in the dict, add it with the current epoch.
- For entries that drop out of top-K, remove them from the dict.
- An entry qualifies when `current_epoch - first_seen_in_topk_epoch >= streak_epochs`.

This is in-memory only — lost on restart. On restart, all entries start with a fresh streak counter. This is conservative (delays promotion slightly after restart) and avoids adding another DB column for a Phase 3-specific optimization.

### OpponentStore Extensions

| New method | Description |
|---|---|
| `save_optimizer(entry_id, optimizer_state_dict)` | Save optimizer state to `{checkpoint_path}_optimizer.pt`. Atomic write (tmp + rename). Update `optimizer_path` in DB. |
| `load_optimizer(entry_id, device)` | Load optimizer state dict from `optimizer_path`. Returns None if no optimizer saved. |
| `increment_update_count(entry_id)` | Increment `update_count`, set `last_train_at` to now. |

**Optimizer path convention:** For a weights file at `league/resnet_ep00100_id42.pt`, the optimizer is at `league/resnet_ep00100_id42_optimizer.pt`. Derived from `checkpoint_path` by inserting `_optimizer` before the extension.

### DynamicManager Changes

- Remove the `assert not config.training_enabled` guard from `__init__`.
- `admit(source_entry)` now also creates a fresh optimizer for the cloned entry:
  ```python
  # Inside admit(), after clone_entry:
  if self.config.training_enabled:
      # Create fresh Adam optimizer for the new Dynamic entry
      # Optimizer is created lazily on first training step, not at admission
      pass  # optimizer_path stays NULL until first DynamicTrainer.update()
  ```
- Add `get_trainable()` method: returns active Dynamic entries where `training_enabled=True` and training has not been disabled due to error.

### FrontierManager Changes

- `review(epoch)` is no longer a no-op. It calls `FrontierPromoter.evaluate()` and orchestrates promotion.
- Add `retire_weakest_or_stalest(epoch)` method implementing the replacement policy.

### TieredPool Changes

| Change | Description |
|---|---|
| Constructor | Creates `DynamicTrainer(store, config.dynamic)` and `FrontierPromoter(config.frontier)` |
| `on_epoch_end(epoch)` | Frontier review now calls the promoter instead of being a no-op |

### Tournament Changes

After each trainable match (D-vs-D or D-vs-RF) in the round-robin, the tournament checks if a training update is due:

```python
# After recording match result
if self.dynamic_trainer and self._is_trainable_match(entry_a, entry_b):
    for entry in [entry_a, entry_b]:
        if entry.role == Role.DYNAMIC and self.dynamic_trainer.should_update(entry.id):
            if not self.dynamic_trainer.is_rate_limited():
                self.dynamic_trainer.update(entry, match_data, device=self.device)
```

The training step runs on the tournament thread using the same GPU. Match data (observations, actions, rewards) must be collected during `_play_batch` and passed through.

---

## Config Changes

Extend `DynamicConfig` with training parameters (shown above). Extend `FrontierStaticConfig` with promotion parameters (shown above).

No new top-level config dataclasses needed — the existing nested configs gain new fields with defaults, so existing TOML configs continue to work without changes.

**Validation in `__post_init__`:**

- `dynamic.lr_scale` must be in (0, 1] — Dynamic entries should never train faster than the learner.
- `dynamic.update_every_matches` must be >= 1.
- `dynamic.max_updates_per_minute` must be >= 1.
- `frontier.min_games_for_promotion` must be >= `frontier.min_tenure_epochs` (can't promote before surviving long enough).

---

## Match Data Collection

The tournament's `_play_batch` method currently discards observations and actions after computing rewards. For Dynamic training, the match data must be retained:

**New dataclass `MatchRollout`:**

```python
@dataclass
class MatchRollout:
    observations: torch.Tensor    # (steps, num_envs, obs_channels, 9, 9)
    actions: torch.Tensor         # (steps, num_envs)
    rewards: torch.Tensor         # (steps, num_envs)
    dones: torch.Tensor           # (steps, num_envs)
    legal_masks: torch.Tensor     # (steps, num_envs, action_space)
    perspective: torch.Tensor     # (steps, num_envs) — which player's perspective (0=A, 1=B)
```

`_play_batch` is extended to optionally collect and return this data when `collect_rollout=True`. The data is stored on CPU to avoid GPU memory pressure — it's only moved to GPU during the training step.

**Memory budget:** One match rollout for a 3-game best-of-3 with 512-ply max and num_envs=3 is approximately:
- observations: 3 × 512 × 50 × 9 × 9 × 4 bytes ≈ 25 MB
- actions: 3 × 512 × 8 bytes ≈ 12 KB
- Total per match: ~25 MB on CPU

With `update_every_matches=4`, the buffer holds ~100 MB before an update flushes it. This is acceptable.

---

## Integration Points

### Training Loop (KataGoTrainingLoop)

| Change | Description |
|---|---|
| Tournament construction | Pass `DynamicTrainer` to `LeagueTournament` |
| Config | `dynamic.training_enabled` now defaults to `True` |

The learner's own PPO loop is completely unchanged. Dynamic training happens exclusively on the tournament thread.

### Learner Opponent Sampling

No changes. The learner still samples opponents by tier mix ratios (50% Dynamic, 30% Frontier, 20% Recent). The fact that Dynamic entries are now trainable is transparent to the learner — it just sees a model that might play slightly differently over time.

### Dashboard / read_league_data

`read_league_data` extended to include `optimizer_path`, `update_count`, `last_train_at` in the entries query. The dashboard can show which Dynamic entries are actively training and their update history.

---

## Monitoring

1. **Dynamic update rate** — updates per minute across all Dynamic entries. Should stay below `max_updates_per_minute`. If consistently at the cap, the tournament is generating trainable matches faster than updates can process them.
2. **Dynamic Elo churn** — standard deviation of Dynamic tier Elo over time. With training enabled, Dynamic entries should show more Elo movement than in Phases 1-2. If Elo churn is too high, `lr_scale` may be too aggressive.
3. **Frontier promotion events** — how often Dynamic entries graduate to Frontier Static. If never, the promotion criteria may be too strict. If too frequent (>1 per 500 epochs), the criteria may be too loose and Frontier is becoming unstable.
4. **Training error rate** — entries disabled due to training errors. If >0, investigate the error cause (NaN loss, architecture mismatch, etc.).
5. **Optimizer checkpoint size** — the optimizer state for Adam is 2x the model size (momentum + variance). Monitor disk usage in the league directory.

---

## Safety & Rollback

### Inference-only fallback

If `disable_on_error=True` and a training update fails, the affected Dynamic entry is marked as inference-only (training disabled for that entry only, not the whole tier). This is logged as a transition. The entry continues to participate in matches and Elo updates — it just stops receiving weight updates.

### Full training disable

Setting `dynamic.training_enabled=False` in the TOML config and restarting reverts to Phase 1-2 behavior: all Dynamic entries are frozen. Existing optimizer files are ignored but not deleted. This is the escape hatch if Dynamic training proves destabilizing during calibration runs.

### Checkpoint integrity

Weights and optimizer are saved separately. If a crash occurs between saving weights and saving optimizer, the entry has new weights but stale optimizer state. On next training step, the stale optimizer's momentum will be slightly wrong but Adam recovers quickly (within 1-2 updates). This is acceptable — atomic paired saves would require writing both to a single file, which complicates the load path.

---

## Testing Strategy

### Unit Tests

- **DynamicTrainer:** `should_update` returns True after `update_every_matches` matches, `is_rate_limited` respects `max_updates_per_minute`, `update` applies PPO and modifies weights, error fallback disables training for the entry.
- **FrontierPromoter:** `evaluate` returns None when no candidates meet criteria, returns the best candidate when criteria are met, `should_promote` checks all 5 criteria independently, streak tracking works across multiple review cycles, lineage overlap limit prevents redundant promotions.
- **OpponentStore:** `save_optimizer`/`load_optimizer` round-trip, `increment_update_count` updates DB, `optimizer_path` derived correctly from `checkpoint_path`.

### Integration Tests

- Full training cycle: tournament plays D-vs-D match, DynamicTrainer collects rollout, runs update, weights are modified, Elo updates, repeat.
- Frontier promotion cycle: Dynamic entry trains for many matches, reaches top-K, holds streak, `FrontierManager.review()` promotes it, Frontier tier gets a new entry, old weakest is retired.
- Error fallback: inject a NaN in training data, verify Dynamic entry is marked inference-only and continues to play matches without training.

---

## Phase 3 Boundaries

**In scope:**
- DynamicTrainer with PPO updates from league match data
- Optimizer state persistence (save/load alongside weights)
- Training safety rails (rate limiting, error fallback, checkpoint frequency)
- FrontierPromoter with conservative multi-criteria promotion evaluation
- FrontierManager.review() activated with replacement policy
- Match data collection in tournament `_play_batch`
- Schema v6 with `optimizer_path`, `update_count`, `last_train_at`

**Explicitly deferred:**
- Parallel match scheduling — Phase 4
- Advanced priority scoring (uncertainty bonus, lineage penalty) — Phase 4
- Using role-specific Elo for eviction/promotion decisions — Phase 4
- Score head training for Dynamic entries — unnecessary; Dynamic entries only need policy and value heads
- Co-training or shared gradients between learner and Dynamic entries — out of scope permanently

---

## Remaining Phases

- **Phase 4 — League Concurrency & Refinement:** Multiple simultaneous pairings on the league GPU, advanced scheduler priority scoring (uncertainty bonus, lineage penalty, diversity bonus), switch eviction/promotion to use role-specific Elo instead of composite, adaptive prioritization for Recent Fixed calibration bursts.
