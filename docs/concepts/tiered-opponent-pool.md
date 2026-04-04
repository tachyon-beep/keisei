# Kenshi Mixed League Full Design

## Status

Design proposal for a mixed self-play league on the dedicated league GPU.

This version resolves the tension between:

- wanting a trustworthy yardstick,
- wanting always-fresh recent challengers,
- wanting a genuinely live trainable league, and
- wanting historical regression detection without carrying irrelevant early opponents in the active pool forever.

The result is a **20-seat active league** plus a **5-slot historical library**.

---

## 1. Executive decision

### Active league: 20 seats

- **5 Frontier Static** — frozen, current-era measurement anchors
- **5 Recent Fixed** — the latest learner snapshots, frozen, FIFO probation/challenger queue
- **10 Dynamic** — trainable league population

### Historical library: 5 tagged snapshots

- **5 Historical Milestones** — frozen, log-spaced checkpoints across training history
- not part of normal active-league matchmaking
- used for periodic benchmark gauntlets and long-range regression checks

This gives four distinct jobs without forcing all four into the same always-on queue.

---

## 2. Why this is the right shape

The earlier concept already separated the pool into frozen measurement opponents, trainable/live opponents, and fresh recent snapshots. Later discussion sharpened two additional insights:

1. **Old founding statics should not occupy active seats forever.**
   Once the learner is deep into training, very old opponents stop being useful active benchmarks.

2. **Historical memory still matters.**
   You still want long-range regression detection and a way to answer questions like:
   - Are we better than we were 5k / 50k / 200k epochs ago?
   - Is the dynamic tier actually helping?

3. **The latest fixed snapshots and the best fixed snapshots are not the same thing.**
   Recent snapshots provide fresh talent and probationary candidates. Frontier statics provide a stable current yardstick.

4. **Dynamic league training is valuable only if the measurement tiers remain frozen.**
   Otherwise the entire system becomes a moving target and progress becomes hard to interpret.

The cleanest way to satisfy all four requirements is:

- keep the active league focused on **current competition**, and
- move long-range historical tracking into a **separate historical library**.

---

## 3. Design goals

### Goals

1. Preserve a stable current-era benchmark for the learner.
2. Keep the most recent learner snapshots visible and relevant.
3. Let a subset of the league improve over time using spare league-GPU capacity.
4. Preserve historical checkpoints for regression detection.
5. Avoid strategy collapse and narrow overfitting.
6. Keep the system implementable within the current single-process, two-GPU architecture.

### Non-goals

1. Full population-based training for every stored snapshot.
2. Making one composite Elo the only truth.
3. Treating every archived checkpoint as an active league seat.
4. Preserving early weak opponents in the active pool indefinitely.

---

## 4. Topology

## 4.1 Active league tiers

### A. Frontier Static (5)

**Purpose:** Current measurement yardstick

**Behaviour:**

- frozen
- no optimiser state
- no training updates
- replaced slowly and deliberately

**What belongs here:**

- proven frozen benchmarks near the current frontier
- chosen to span from "comfortably beatable" to "genuinely challenging"
- not necessarily the literal top-5 Elo models

**Reasoning:**
If these are only the absolute strongest five, they can collapse into a very narrow band and stop giving a good difficulty spread. The tier should represent the **current skill frontier**, not only the single strongest cluster.

---

### B. Recent Fixed (5)

**Purpose:** Fresh blood / probation / challenger queue

**Behaviour:**

- frozen
- FIFO queue of latest learner snapshots
- always reflects what the learner has been producing recently
- no training updates

**What belongs here:**

- latest 5 learner snapshots created by the existing snapshot mechanism

**Reasoning:**
This is the clean interpretation of "most recent 5 fixed". It keeps fresh recent talent in the active system without immediately making it dynamic or promoting it into the measurement tier.

---

### C. Dynamic (10)

**Purpose:** Live trainable population

**Behaviour:**

- trainable
- has persistent optimiser state
- receives controlled PPO updates from league matches
- can be admitted from Recent Fixed
- can spawn a frozen Frontier Static clone when proven

**Reasoning:**
This is where the live-league behaviour happens. The dynamic tier exists to develop counters, anti-counters, and competitive pressure that a purely frozen pool cannot provide.

---

## 4.2 Historical library (5)

### D. Historical Milestones (5)

**Purpose:** Long-range regression detection and historical comparison

**Behaviour:**

- frozen
- tagged snapshots, not normal active-league seats
- used in benchmark gauntlets and analytics
- selected automatically from the archive using a spacing rule

**Reasoning:**
This solves the earlier tension directly:

- history remains available,
- but does not consume active competitive capacity.

---

## 5. Lifecycle model

## 5.1 State machine

```text
Learner snapshot
    -> Recent Fixed (frozen, probationary)
        -> Retire / archive only
        -> Dynamic clone (if qualified)
            -> Retire
            -> Frontier Static clone (if promoted at review)

Archived checkpoints
    -> Historical Library tag (if selected as milestone)
```

## 5.2 Important rule

Promotion is by **cloning**, not by moving in place.

That means:

- Recent Fixed -> Dynamic creates a new Dynamic branch with fresh optimiser state.
- Dynamic -> Frontier Static creates a frozen clone for benchmark use.
- Historical Library selection adds a **tag/reference** to an existing frozen checkpoint; it does not mutate its tier.

This keeps lineage and semantics clean.

---

## 6. Tier semantics

## 6.1 Frontier Static

### Purpose

Frontier Static answers:

- Is the learner getting stronger against stable current-era opposition?
- Is the dynamic population producing models worthy of being frozen as benchmarks?
- Did something regress in the last few hundred epochs?

### Selection policy

At any review point, choose 5 frozen models that span the current benchmark band:

- Static 1: easy/current-low benchmark
- Static 2: lower-mid benchmark
- Static 3: mid benchmark
- Static 4: upper-mid benchmark
- Static 5: hard/current frontier benchmark

Selection should prefer:

- proven stability,
- enough calibration games,
- lineage diversity,
- recent relevance.

### Replacement policy

This tier is not permanent.
It should march up the skill ladder slowly.

When a new Frontier Static is admitted:

- retire the weakest or stalest eligible Frontier Static,
- but only after a minimum tenure/cooldown.

Recommended defaults:

- `frontier_static_slots = 5`
- `frontier_review_interval_epochs = 250`
- `frontier_min_tenure_epochs = 100`
- `frontier_promotion_margin_elo = 50`

---

## 6.2 Recent Fixed

### Purpose

Recent Fixed answers:

- What is the learner producing right now?
- Which recent snapshots deserve a live Dynamic seat?
- Are recent snapshots outperforming existing Dynamic models?

### Behaviour

- snapshot enters on the normal learner snapshot cadence
- queue keeps the latest 5
- oldest Recent Fixed is reviewed when the queue overflows

### Admission path

When a new snapshot arrives:

1. create a frozen Recent Fixed entry,
2. schedule calibration matches,
3. if Recent Fixed > 5, review the oldest entry.

### Review outcomes for oldest Recent Fixed

- **Promote to Dynamic clone** if qualified
- **Retire** if too weak / too uncertain
- **Delay review once** if under-calibrated and soft overflow is enabled

Recommended defaults:

- `recent_fixed_slots = 5`
- `recent_min_games_for_review = 32`
- `recent_min_unique_opponents = 6`
- `recent_soft_overflow = 1`

---

## 6.3 Dynamic

### Purpose

Dynamic answers:

- Is the league developing meaningful adaptive pressure?
- Can recent learner snapshots survive inside a live ecosystem?
- Are trainable opponents adding value beyond frozen calibration alone?

### Behaviour

- trainable
- own weights and optimiser state
- updated only from league matches
- never updated from learner PPO batches

### Training rule

Dynamic updates come only from explicitly designated **league training matches**.

Recommended trainable match classes:

- Dynamic vs Dynamic
- Dynamic vs Recent Fixed

Recommended non-trainable calibration classes:

- Dynamic vs Frontier Static
- Recent Fixed vs Frontier Static
- Frontier Static vs Frontier Static
- Historical gauntlet matches

### Protection rule

Every newly admitted Dynamic gets a protection window before it can be evicted.

Recommended defaults:

- `dynamic_slots = 10`
- `dynamic_protection_matches = 24`
- `dynamic_min_games_before_eviction = 40`

---

## 6.4 Historical Library

### Purpose

Historical Library answers:

- Are we still stronger than we were at significant earlier points?
- Did the dynamic league accelerate progress or just reshuffle ratings?
- Did a new change cause long-range forgetting that current-era statics do not reveal?

### Behaviour

- does not participate in normal league churn
- runs in a separate benchmark gauntlet
- snapshots remain frozen forever unless manually pruned from storage

### Selection model

Use **5 log-spaced milestone targets** between epoch 1 and current epoch.

For current learner epoch `E`, define targets:

```text
T_i = round(exp(log(E) * i / 4))   for i = 0..4
```

Then snap each target to the nearest available archived checkpoint.

This produces approximately:

- at 10,000 epochs: {1, 10, 100, 1000, 10000}
- at 250,000 epochs: {1, 22, 500, 11000, 250000}

This is better than fixed spacing because:

- early training is represented,
- middle training is represented,
- recent history is represented,
- milestones do not bunch up at the tail.

### Early-training fallback

Before enough distinct milestones exist:

- keep the library at 5 logical slots,
- fill missing slots with the closest available frozen checkpoints,
- allow some entries to double-duty as current benchmarks in the UI.

That preserves the invariant that the dashboard always shows five historical slots, even when training is young.

---

## 7. Promotion, eviction, and retirement

## 7.1 Recent Fixed -> Dynamic

A Recent Fixed entry is eligible for Dynamic admission only if it satisfies all of:

1. minimum games played
2. minimum unique opponents
3. Elo at or above the weakest eligible Dynamic minus a margin
4. acceptable uncertainty / volatility

Recommended defaults:

- `recent_to_dynamic_min_games = 32`
- `recent_to_dynamic_min_unique_opponents = 6`
- `recent_to_dynamic_margin_elo = 25`

### If qualified

- clone into Dynamic
- initialise a fresh optimiser
- mark protection window
- retire the Recent Fixed source entry

### If not qualified

- retire/archive the Recent Fixed source entry
- do not force it into Dynamic just because the queue turned over

---

## 7.2 Dynamic eviction

A Dynamic entry can be evicted only if:

- it is past protection,
- it has minimum calibration volume,
- it is not currently the best candidate for Frontier Static promotion.

When a new Dynamic is admitted:

- evict the **lowest-Elo eligible Dynamic**.

Use **recent or smoothed Dynamic Elo**, not lifetime raw Elo.

Recommended default:

- `dynamic_eviction_policy = "lowest_recent_elo_eligible"`

---

## 7.3 Dynamic -> Frontier Static

This should be conservative.

A Dynamic entry may spawn a Frontier Static clone only if:

- it has sufficient total calibration matches,
- it has survived long enough in Dynamic to prove stability,
- it has held a top position for a sustained window,
- it exceeds the weakest Frontier Static by a promotion margin,
- it is not too redundant with existing Frontier Static lineage.

Recommended defaults:

- `dynamic_to_frontier_min_games = 64`
- `dynamic_to_frontier_topk = 3`
- `dynamic_to_frontier_streak_epochs = 50`
- `frontier_promotion_margin_elo = 50`
- `frontier_review_interval_epochs = 250`

### Replacement rule

When a new Frontier Static clone is admitted:

- retire the weakest or stalest eligible Frontier Static,
- never replace more than one Frontier Static per review window.

---

## 7.4 Historical refresh

Historical Library refresh is not a promotion in the same sense.
It is a **selection/tagging operation**.

Every `historical_refresh_interval_epochs`:

1. recompute target milestone epochs,
2. snap to nearest archived checkpoints,
3. update the 5 historical-library references.

Recommended default:

- `historical_refresh_interval_epochs = 100`

---

## 8. Matchmaking and scheduling

## 8.1 Scheduler responsibilities

The scheduler must balance:

1. rating accuracy,
2. fast calibration for Recent Fixed,
3. meaningful training pressure for Dynamic,
4. periodic historical benchmarking.

A single flat round-robin is no longer enough.

---

## 8.2 Match classes

### Training matches

- Dynamic vs Dynamic
- Dynamic vs Recent Fixed

### Active-league calibration matches

- Dynamic vs Frontier Static
- Recent Fixed vs Frontier Static
- Recent Fixed vs Recent Fixed (low frequency)
- Frontier Static vs Frontier Static (low frequency)

### Historical benchmark matches

- Learner vs Historical Library
- Dynamic top-N vs Historical Library (optional)
- Frontier Static vs Historical Library (optional, low cadence)

Historical benchmark matches should be scheduled separately and should not dominate the normal active-league queue.

---

## 8.3 Recommended active-league mix

Across league-GPU scheduling volume:

- **40%** Dynamic vs Dynamic
- **25%** Dynamic vs Recent Fixed
- **20%** Dynamic vs Frontier Static
- **10%** Recent Fixed vs Frontier Static
- **5%** Recent Fixed vs Recent Fixed

Then run a separate historical gauntlet every `N` epochs.

---

## 8.4 Pair priority

Priority score:

```text
priority =
    under_sample_bonus
  + uncertainty_bonus
  + recent_fixed_calibration_bonus
  + diversity_bonus
  - repeat_pair_penalty
  - close_lineage_penalty
```

Interpretation:

- under-sampled pairs are prioritised,
- Recent Fixed entries get extra early attention,
- repeated head-to-heads are penalised,
- close relatives are penalised to preserve diversity.

---

## 8.5 Concurrency on GPU 1

Because the dedicated league GPU has headroom, the league runner should support multiple simultaneous pairings.

Recommended starting point:

- `parallel_matches = 4`
- `envs_per_match = 8`
- `league_total_envs = 32`

Then scale cautiously to:

- `parallel_matches = 8`
- `envs_per_match = 8`
- `league_total_envs = 64`

Only increase beyond that after measuring:

- GPU memory pressure
- DB write rates
- checkpoint flush frequency
- tournament latency
- dynamic-update overhead

---

## 9. Rating model

## 9.1 Do not rely on one composite Elo

Track at least four rating views:

### A. Frontier Benchmark Elo

Derived only from matches involving Frontier Static.

This is the primary current-era progress number.

### B. Dynamic League Elo

Used for Dynamic admission, eviction, and promotion.

This is the live ecosystem rating.

### C. Recent Challenge Score

Performance against Recent Fixed.

This tells you whether the learner and dynamic league are handling truly recent snapshots.

### D. Historical Gauntlet Score

Performance against the Historical Library.

This is the long-range regression and historical-progress metric.

---

## 9.2 Operational rule

When the dashboard needs one main number:

- show **Frontier Benchmark Elo** first.

Then also show:

- Dynamic League Elo,
- Recent Challenge Score,
- Historical Gauntlet Score.

---

## 9.3 K-factors

Recommended defaults:

- `frontier_benchmark_k = 16`
- `dynamic_league_k = 24`
- `recent_initial_k = 32`
- `historical_k = 12`

Rationale:

- benchmark should move slowly,
- dynamic league can move faster,
- recent snapshots need rapid early calibration,
- historical score should be stable.

---

## 10. Dynamic training mechanics

## 10.1 Training source

Only Dynamic entries train.

If a Dynamic plays:

- **Dynamic vs Dynamic**: both may update from their own perspective
- **Dynamic vs Recent Fixed**: Dynamic may update, Recent Fixed never updates
- **Dynamic vs Frontier Static**: no training update

---

## 10.2 Update budget

Keep Dynamic updates intentionally small.

Recommended defaults:

- `dynamic_update_epochs = 2`
- `dynamic_batch_reuse = 1`
- `dynamic_lr_scale = 0.25` relative to learner LR
- `dynamic_grad_clip = 1.0`
- `dynamic_update_every_matches = 4`

The goal is not to create ten full co-equal learners. The goal is to create enough adaptation to keep the league sharp.

---

## 10.3 Optimiser handling

Dynamic entries persist:

- weights
- optimiser state
- update count
- last updated time

On Recent Fixed -> Dynamic promotion:

- initialise a **fresh optimiser**
- do not inherit learner optimiser state

That keeps the branch semantics clear.

---

## 10.4 Safety rails

- hard cap updates per minute
- hard cap checkpoint writes per minute
- back-pressure if GPU utilisation or queue depth exceeds threshold
- automatic fall-back to inference-only mode if dynamic training becomes unstable

Recommended defaults:

- `dynamic_max_updates_per_minute = 20`
- `dynamic_checkpoint_flush_every = 8 matches`
- `dynamic_disable_on_error = true`

---

## 11. Learner opponent sampling

The learner should not train only against Dynamic entries.

Recommended learner mixture:

- **50% Dynamic**
- **30% Frontier Static**
- **20% Recent Fixed**

Historical Library should not normally be in the learner PPO stream. It should be used in periodic benchmark gauntlets.

This keeps main training focused on current relevant pressure, without sacrificing measurement.

---

## 12. Diversity controls

To avoid strategy collapse:

1. **Lineage-aware scheduling**
   - penalise parent/child and close-sibling overuse
2. **Frontier exposure requirement**
   - every Dynamic must periodically face Frontier Static
3. **Recent-Fixed admission pressure**
   - keep fresh challengers entering continuously
4. **Historical gauntlet checks**
   - ensure long-range regressions are visible
5. **Protection windows**
   - give new Dynamics time before eviction
6. **Role separation**
   - keep history, measurement, recent challengers, and trainable population conceptually distinct

Optional hard rule:

- no more than 50% of Dynamic seats may come from the last `N` learner snapshots.

---

## 13. Data model

## 13.1 league_entries

Recommended columns:

- `id`
- `role` (`frontier_static`, `recent_fixed`, `dynamic`)
- `status` (`active`, `retired`, `archived`)
- `parent_entry_id`
- `source_epoch`
- `created_at`
- `retired_at`
- `checkpoint_path`
- `optimizer_path` (Dynamic only)
- `training_enabled`
- `elo_frontier`
- `elo_dynamic`
- `elo_recent`
- `games_total`
- `games_vs_frontier`
- `games_vs_dynamic`
- `games_vs_recent`
- `protection_remaining`
- `last_match_at`
- `last_train_at`
- `lineage_group`

## 13.2 historical_library

Recommended columns:

- `slot_index` (0..4)
- `target_epoch`
- `entry_id`
- `actual_epoch`
- `selected_at`
- `selection_mode` (`log_spaced`, `fallback`)

## 13.3 league_matches

Recommended fields:

- `match_type` (`train`, `calibration`, `historical_benchmark`)
- `entry_a_id`
- `entry_b_id`
- `role_a`
- `role_b`
- `num_games`
- `wins_a`
- `wins_b`
- `draws`
- `elo_before_a`
- `elo_after_a`
- `elo_before_b`
- `elo_after_b`
- `training_updates_a`
- `training_updates_b`
- `created_at`

## 13.4 league_transitions

Track lifecycle events:

- Recent Fixed created
- Recent Fixed promoted to Dynamic
- Dynamic evicted
- Dynamic cloned to Frontier Static
- Frontier Static retired
- Historical slot re-pointed

---

## 14. Filesystem layout

```text
checkpoints/500k-league/
  learner/
    latest.pt
  league/
    entries/
      000123/
        weights.pt
        metadata.json
      000124/
        weights.pt
        optimizer.pt
        metadata.json
  history/
    library.json
```

Rules:

- Frontier Static / Recent Fixed: weights + metadata only
- Dynamic: weights + optimiser + metadata
- Historical Library: references existing frozen checkpoints; no duplicate weights required

---

## 15. Dashboard and observability

## 15.1 Separate panels

### Active League

- 5 Frontier Static
- 5 Recent Fixed
- 10 Dynamic
- current admissions and protection windows

### Historical Library

- 5 milestone epochs
- current assigned checkpoints
- learner historical gauntlet results

---

## 15.2 Must-have metrics

- Frontier Benchmark Elo
- Dynamic League Elo
- Recent Challenge Score
- Historical Gauntlet Score
- Dynamic admission queue
- Dynamic evictions
- Frontier promotions
- Historical milestone epochs
- role-specific win rates

---

## 15.3 UI cues

Use distinct role badges/icons everywhere:

- Frontier Static: shield
- Recent Fixed: seedling or spark
- Dynamic: crossed swords
- Historical Library: scroll or archive icon

Never show a flat leaderboard without role markers.

---

## 16. Rollout plan

## Phase 0 — current state

- keep existing background inference-only tournament running
- validate round-robin Elo calibration

## Phase 1 — role split, frozen only

Implement:

- Active League role labels: Frontier Static / Recent Fixed / Dynamic
- but keep Dynamic training disabled
- Recent Fixed FIFO
- Dynamic admission and eviction logic
- Frontier Static review and replacement
- dashboard role badges

## Phase 2 — Historical Library

Implement:

- 5-slot historical library
- log-spaced target selection
- periodic learner historical gauntlet
- historical dashboard panel

## Phase 3 — Dynamic training

Implement:

- Dynamic optimiser state
- training matches and small PPO updates
- update caps and checkpoint persistence
- protection windows and fault fallback

## Phase 4 — league concurrency

Implement:

- multiple simultaneous pairings on GPU 1
- more advanced scheduler scoring
- adaptive prioritisation for Recent Fixed calibration bursts

---

## 17. Concrete TOML shape

```toml
[league]
enabled = true
mode = "mixed"
opponent_device = "cuda:1"
snapshot_interval = 10

[league.active]
frontier_static_slots = 5
recent_fixed_slots = 5
dynamic_slots = 10

[league.frontier_static]
review_interval_epochs = 250
min_tenure_epochs = 100
promotion_margin_elo = 50.0
min_games_for_promotion = 64
replace_policy = "weakest_or_stalest_after_cooldown"
span_selection = true

[league.recent_fixed]
min_games_for_review = 32
min_unique_opponents = 6
promotion_margin_elo = 25.0
soft_overflow = 1
retire_if_below_dynamic_floor = true

[league.dynamic]
training_enabled = false          # enable in phase 3
protection_matches = 24
min_games_before_eviction = 40
update_epochs_per_batch = 2
learning_rate_scale = 0.25
grad_clip = 1.0
update_every_matches = 4
max_updates_per_minute = 20
checkpoint_flush_every = 8

[league.history]
slots = 5
enabled = true
selection = "log_spaced"
refresh_interval_epochs = 100
active_league_participation = false
benchmark_gauntlet_interval_epochs = 100

[league.matchmaking]
parallel_matches = 4
envs_per_match = 8
total_envs = 32
pairing_policy = "role_weighted_sparse_h2h"
dynamic_dynamic_weight = 0.40
dynamic_recent_weight = 0.25
dynamic_frontier_weight = 0.20
recent_frontier_weight = 0.10
recent_recent_weight = 0.05

[league.sampling]
learner_dynamic_ratio = 0.50
learner_frontier_ratio = 0.30
learner_recent_ratio = 0.20

[league.elo]
frontier_benchmark_k = 16.0
dynamic_league_k = 24.0
recent_initial_k = 32.0
historical_k = 12.0
track_role_specific = true

[league.storage]
clone_on_promotion = true
persist_optimizer_for_dynamic = true
```

---

## 18. Final recommendation

Lock in these choices:

1. **Use a 20-seat active league:** 5 Frontier Static, 5 Recent Fixed, 10 Dynamic.
2. **Keep a separate 5-slot Historical Library outside the active league.**
3. **Recent Fixed stays frozen and acts as fresh blood.**
4. **Only Dynamic trains.**
5. **Dynamic may clone into Frontier Static on a slow cadence.**
6. **Historical Library is selected from archived checkpoints using log-spaced targets.**
7. **Frontier Benchmark Elo is the primary headline metric.**
8. **Historical Gauntlet Score is the long-range regression metric.**
9. **Dynamic League Elo is operational, not the canonical progress number.**
10. **Implement role split first; enable Dynamic training only after frozen-role calibration looks sane.**

This design preserves measurement, keeps fresh recent talent visible, gives you a genuinely live league, and avoids wasting active seats on dead historical weight.

---

## 19. Confidence

**WEP:**

- **Highly likely** this topology is cleaner than either a flat 20-seat pool or a fully merged 4-tier active league.
- **Likely** that separating Historical Library from the active league is the key simplification that makes the whole system age well.
- **Likely** that the exact numeric thresholds will need tuning once real match volumes and calibration variance are visible.
