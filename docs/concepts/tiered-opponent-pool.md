# Tiered Opponent Pool: Milestone / Static / Dynamic / Newest

**Status:** Conceptual — not yet implemented.
**Prerequisite:** Let the current round-robin tournament run for several days to validate Elo calibration works correctly before adding complexity.

## Motivation

The current opponent pool treats all entries identically: frozen snapshots that only serve as evaluation targets. This works for progress measurement but doesn't push the learner as hard as it could be pushed. A tiered pool gives us historical context, a competitive yardstick, a co-evolving ecosystem, and fresh blood — four distinct purposes that shouldn't be muddled into one.

## The Four Tiers

### Milestone (5 entries) — Historical Record

- **Icon:** Scroll
- **Purpose:** Log-spaced historical checkpoints spanning the full training arc. "How far have we come?" is answered by these.
- **Behavior:** Frozen. Never train.
- **Selection:** Automatically recomputed as training progresses. Targets are log-spaced across the training timeline so you always have coverage from early, mid, and late training regardless of total epochs elapsed.
- **Early training behavior:** Until 5 real milestones exist, the slots act as extra statics. The UI shows the milestone icon regardless — the user doesn't need to know whether a slot has been claimed by a "real" milestone yet. As milestones arrive, they gradually take over slots that were pulling double duty.

#### Log-Spaced Milestone Selection

Fixed intervals (every 500 or 1000 epochs) waste slots: they bunch up late in training and provide too many early checkpoints. Instead, milestone targets are distributed logarithmically:

```
For 5 milestones at current epoch E:
  targets = [E^(i/4) for i in 0..4]
          ≈ [1, E^0.25, E^0.5, E^0.75, E]
```

At epoch 10,000: targets ≈ {1, 10, 100, 1000, 10000}
At epoch 250,000: targets ≈ {1, 22, 500, 11000, 250000}

The *ratio* between consecutive milestones stays roughly constant, giving even coverage across the full training arc on a log scale. Implementation snaps each target to the nearest existing snapshot.

Milestone targets are recomputed periodically (e.g. every 100 epochs). When a new target set differs from the current milestones, the most out-of-date milestone is replaced by the snapshot closest to the new target. This is gradual — at most one swap per recomputation cycle.

### Static (5 entries) — Competitive Benchmarks

- **Icon:** Shield
- **Purpose:** Frozen benchmarks at the current skill frontier. "Is the learner getting stronger right now?" is answered by win rate against these.
- **Behavior:** Frozen. Never train.
- **Selection:** Promoted from the dynamic tier when a dynamic has proven stable (high Elo, low variance, sufficient games).
- **Retirement:** When a new static is promoted in, the weakest static is retired. The tier should always span "comfortably beatable" to "genuinely challenging" relative to the *current* learner.
- **Regression detection:** If the learner's win rate against a recently-competitive static drops sharply, that signals catastrophic forgetting or a bad hyperparameter change.

#### Static Promotion Criteria

A dynamic model is eligible for promotion to static when:
- It has held a top-3 Elo position among dynamics for 50+ consecutive epochs
- It has >= 20 completed calibration matches
- Promotion cadence is conservative — at most every 200-500 epochs

### Dynamic (15 entries) — Competitive Pressure

- **Icon:** Crossed swords
- **Purpose:** Co-evolving opponents that keep the learner challenged. "Is the training environment hard enough?" is answered by win rate against these.
- **Behavior:** Receive PPO updates from tournament games. Their weights change over time.
- **Training mechanics:**
  - After each tournament match, the losing model (or both) gets a few PPO update steps using rollout data from the match.
  - Each dynamic model needs persistent optimizer state (Adam momentum/variance) saved alongside its checkpoint.
  - Estimated memory per model (b6c96, ~3M params): ~48MB (weights + optimizer + gradients). 15 models = ~720MB — fits easily on a dedicated GPU.
- **Eviction:** When a newest entry graduates to dynamic, the lowest-Elo dynamic with >= 5 completed matchups is evicted.

### Newest (5 entries) — Fresh Blood

- **Icon:** Seedling
- **Purpose:** Rolling window of recent learner snapshots. Shows how the learner has evolved recently.
- **Behavior:** Frozen (no training). Created automatically by the existing snapshot mechanism.
- **Lifecycle:** FIFO — newest in, oldest out. The oldest newest entry graduates to dynamic.

## Lifecycle Flows

```
Learner snapshot ──→ [Newest] ──FIFO──→ [Dynamic] ──promotion──→ [Static]
                        (5)                (15)          ↓           (5)
                                     evict lowest    retire weakest
                                         Elo             static

[Milestone] ← recomputed log-spaced from existing snapshots
    (5)        (acts as extra statics until real milestones arrive)
```

## Graduation: Newest → Dynamic

When a newest entry ages out (pushed by a newer snapshot):

**Hybrid approach (recommended):** Always graduate to dynamic, but eviction targets the lowest-Elo dynamic with >= N completed matchups. A freshly graduated entry with uncalibrated Elo is protected until it's had a fair chance. This keeps fresh blood flowing without discarding potentially strong models that just haven't been calibrated yet.

## Pool Size

Total: **30 entries** (matching the current leaderboard placeholder count)

| Tier | Count | Trains? | Eviction |
|------|-------|---------|----------|
| Milestone | 5 | No | Log-spaced recomputation (gradual swap) |
| Static | 5 | No | Weakest retires on promotion |
| Dynamic | 15 | Yes | Lowest Elo (min games) on graduation |
| Newest | 5 | No | FIFO into dynamic |

## Elo Interpretation

**Critical design constraint:** The composite Elo number becomes harder to interpret with mixed tiers. Three mitigations:

1. **Per-tier Elo breakdown in UI.** Show "vs Static: 1450, vs Dynamic: 1280" alongside the composite. The static number is the true short-term progress metric. The milestone number is the long-term progress metric.

2. **Visual tier indicators everywhere ratings appear.** Every place an entry name shows up (leaderboard, matchup matrix, Elo chart, match history, player cards) must show the tier icon. A flat composite Elo line that's actually hiding "static Elo up, dynamic Elo down" must never be misread as stagnation.

3. **Separate Elo chart series.** Consider splitting the learner's Elo chart into "vs Static" and "vs Dynamic" lines, or at minimum allowing a tier filter on the chart.

**What each tier's Elo means:**
- **Milestone Elo:** "How much better am I than historical me?" — should only ever go up.
- **Static Elo:** "Am I improving against current-frontier benchmarks?" — the primary progress signal.
- **Dynamic Elo:** "Am I keeping up with the co-evolving ecosystem?" — flat is fine here (opponents are improving too).
- **Newest Elo:** "How does the latest snapshot compare to recent history?" — volatile, primarily diagnostic.

## Implementation Phases

### Phase 1: Tiered Pool Management (no training)

Add tier labels to league entries. Update eviction logic to respect tiers. Add tier icons to the UI. All models remain frozen — dynamic models just have the *label* for now.

- DB migration: add `tier TEXT NOT NULL DEFAULT 'newest'` to `league_entries`
- Pool eviction: tier-aware (log-recompute for milestone, FIFO for newest, lowest-Elo for dynamic, retire-weakest for static)
- Graduation logic: oldest newest → dynamic (hybrid approach)
- Milestone recomputation: periodic log-spaced target recalc, snap to nearest snapshot
- Early-training mode: milestone slots act as statics until claimed
- UI: tier icons in leaderboard, match history, matchup matrix, Elo chart legend

### Phase 2: Dynamic Model Training

Add PPO updates for dynamic-tier models after tournament matches.

- Extend tournament to collect rollout buffers (not just win/loss counts)
- Per-model optimizer state: save/load Adam state alongside checkpoint
- Training step after match: 2-4 PPO epochs on collected rollout data
- Checkpoint update: write new weights after training
- Memory budget: validate 15 × ~48MB fits alongside main training

### Phase 3: Per-Tier Elo Analytics

- Per-tier Elo breakdown in player cards and stats banner
- Filterable Elo chart (show/hide by tier)
- "vs Static" and "vs Milestone" progress metrics as first-class dashboard numbers

## Open Questions

- **How many tournament games before graduation Elo is "calibrated enough"?** 5 matchups? 10? Depends on K-factor and opponent diversity.
- **Should dynamic models train against static/milestone models too, or only against each other?** Training against static models would shift dynamic Elo relative to static, which is useful for calibration. But it means static models' Elo changes (from the dynamic model's side), which could confuse the picture. Probably: dynamic trains from all matches, but static/milestone Elo is only updated from inference-only calibration matches.
- **Catastrophic forgetting in dynamics:** A dynamic model that only plays against one opponent repeatedly could overfit to that matchup. The round-robin scheduling (fewest h2h games first) mitigates this, but worth monitoring.
- **Should the learner's main training matches count toward dynamic model training?** Probably not — those are the learner's training signal, not the opponent's. Keep training data sources clean.
- **Milestone checkpoint storage:** Log-spaced milestones need their checkpoint files preserved even after they'd normally be eligible for cleanup. Tag milestone checkpoints as protected.

## Resource Estimates

Current setup (b6c96 SE-ResNet, ~3M params):

| Resource | Current | With Tiered Pool |
|----------|---------|-----------------|
| League GPU VRAM | ~50MB (2 models for tournament) | ~800MB (15 dynamic w/ optimizer + 2 loaded for match) |
| Disk per checkpoint | ~12MB | ~36MB (weights + optimizer state) |
| Total disk (30 entries) | ~360MB | ~660MB (15 dynamic w/ optimizer + 15 inference-only) |
| Tournament match time | ~30s (inference only) | ~90s (inference + 2-4 PPO steps for dynamics) |
| DB writes per match | 5 rows | 5 rows + checkpoint write |

All well within "loads of headroom" territory.
