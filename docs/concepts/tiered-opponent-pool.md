# Tiered Opponent Pool: Milestone / Frontier / Dynamic / Newest

**Status:** Conceptual — not yet implemented.
**Prerequisite:** Ship the training fairness fixes (2026-04-05 spec) and let clean Elo calibration run for several days before adding complexity.
**Last reviewed:** 2026-04-05

## Motivation

The current opponent pool treats all entries identically: frozen snapshots that only serve as evaluation targets. This works for progress measurement but doesn't push the learner as hard as it could be pushed. A tiered pool gives us historical context, a competitive yardstick, a co-evolving ecosystem, and fresh blood — four distinct purposes that shouldn't be muddled into one.

### Core Design Principle

**The visible 20-entry league is an ecosystem, not a benchmark.** Use it for pressure, drama, and opponent selection. Use frozen anchors and milestone checkpoints for progress measurement. The Elo inflation we observed (snapshots entering at ~1650 via carry-forward) demonstrated why these concerns must be separated.

## The Four Tiers

### Milestone (4 entries) — Historical Record

- **Icon:** Scroll
- **Purpose:** Log-spaced historical checkpoints spanning the full training arc. "How far have we come?" is answered by these.
- **Behavior:** Frozen. Never train.
- **Selection:** Automatically recomputed as training progresses. Targets are log-spaced across the training timeline so you always have coverage from early, mid, and late training regardless of total epochs elapsed.
- **Early training behavior:** Until 4 real milestones exist, the slots act as extra frontiers. The UI shows the milestone icon regardless — the user doesn't need to know whether a slot has been claimed by a "real" milestone yet. As milestones arrive, they gradually take over slots that were pulling double duty.

#### Log-Spaced Milestone Selection

Fixed intervals (every 500 or 1000 epochs) waste slots: they bunch up late in training and provide too many early checkpoints. Instead, milestone targets are distributed logarithmically:

```
For 4 milestones at current epoch E:
  targets = [E^(i/3) for i in 0..3]
          ≈ [1, E^0.33, E^0.67, E]
```

At epoch 10,000: targets ~ {1, 22, 464, 10000}
At epoch 250,000: targets ~ {1, 63, 3969, 250000}

The *ratio* between consecutive milestones stays roughly constant, giving even coverage across the full training arc on a log scale. Implementation snaps each target to the nearest existing snapshot.

Milestone targets are recomputed periodically (e.g. every 100 epochs). When a new target set differs from the current milestones, the most out-of-date milestone is replaced by the snapshot closest to the new target. This is gradual — at most one swap per recomputation cycle.

### Frontier (4 entries) — Competitive Benchmarks

> Renamed from "Static" to "Frontier" — the name "static" overclaims stability since the tier still drifts through promotion/retirement. "Frontier" accurately describes what it is: the current skill boundary.

- **Icon:** Shield
- **Purpose:** Frozen benchmarks at the current skill frontier. "Is the learner getting stronger right now?" is answered by win rate against these.
- **Behavior:** Frozen. Never train.
- **Selection:** Promoted from the dynamic tier when a dynamic has proven stable (high Elo, low variance, sufficient games) — see promotion criteria below.
- **Retirement:** When a new frontier is promoted in, the weakest frontier is retired — but only if the learner has cleared it by a significant margin for a sustained period, not simply because a shinier thing appeared.
- **Regression detection:** If the learner's win rate against a recently-competitive frontier drops sharply, that signals catastrophic forgetting or a bad hyperparameter change.

#### Frontier Promotion Criteria

A dynamic model is eligible for promotion to frontier when:
- It has held a top-3 Elo position among dynamics for 50+ consecutive epochs
- It has >= 20 completed calibration matches
- It is strong against anchors (not just farming one corner of the ecology)
- Promotion cadence is conservative — at most every 200-500 epochs

### Dynamic (8 entries) — Competitive Pressure

- **Icon:** Crossed swords
- **Purpose:** Co-evolving opponents that keep the learner challenged. "Is the training environment hard enough?" is answered by win rate against these.
- **Behavior:** Receive PPO updates from tournament games. Their weights change over time.
- **Training mechanics:**
  - After each tournament match, the losing model (or both) gets a few PPO update steps using rollout data from the match.
  - Each dynamic model needs persistent optimizer state (Adam momentum/variance) saved alongside its checkpoint.
  - Estimated memory per model (b6c96, ~3M params): ~48MB (weights + optimizer + gradients). 8 models = ~384MB — fits easily on a dedicated GPU.
- **Eviction:** When a newest entry graduates to dynamic, the lowest-Elo dynamic with >= 5 completed matchups is evicted.

### Newest (4 entries) — Fresh Blood

- **Icon:** Seedling
- **Purpose:** Rolling window of recent learner snapshots. Shows how the learner has evolved recently.
- **Behavior:** Frozen (no training). Created automatically by the existing snapshot mechanism.
- **Lifecycle:** FIFO — newest in, oldest out. The oldest newest entry graduates to dynamic.

## Anchor Bank — The Hidden Truth Source

> Added based on review: the visible league needs a fixed reference frame that never moves.

**3-5 frozen checkpoints** stored outside the visible 20-slot league. Never retired, never trained, never promoted. Their only purpose is calibration.

**Why anchors are necessary:** Milestone Elo answers "how much better am I than old me?" and Frontier Elo answers "am I improving against current-frontier benchmarks?" But Frontier still drifts because it is promoted from Dynamic and retired over time — it is a moving reference frame. The anchor bank gives you one number that is actually pinned down.

**Selection:** Anchors are set manually or at major training milestones (e.g., "first model to beat random >90%", "first model to reach 1200 Frontier Elo"). They should span a wide skill range.

**Elo updates:** Anchor models participate in rating matches (inference-only) but their Elo is tracked separately. The learner's "Anchor Elo" is computed from results against these fixed references only.

**Storage:** Anchor checkpoints are tagged as protected and excluded from all eviction/cleanup logic.

## Lifecycle Flows

```
Learner snapshot ──> [Newest] ──FIFO──> [Dynamic] ──promotion──> [Frontier]
                        (4)                (8)           |           (4)
                                     evict lowest    retire weakest
                                         Elo           frontier

[Milestone] <── recomputed log-spaced from existing snapshots
    (4)         (acts as extra frontiers until real milestones arrive)

[Anchor Bank] ── hidden, 3-5 fixed references, never changes
                  (calibration only, not in the visible league)
```

## Graduation: Newest -> Dynamic

When a newest entry ages out (pushed by a newer snapshot):

**Hybrid approach (recommended):** Always graduate to dynamic, but eviction targets the lowest-Elo dynamic with >= N completed matchups. A freshly graduated entry with uncalibrated Elo is protected until it's had a fair chance. This keeps fresh blood flowing without discarding potentially strong models that just haven't been calibrated yet.

## Pool Size

Visible league: **20 entries** (matching the current leaderboard and max pool size)
Anchor bank: **3-5 entries** (hidden, outside the league)

| Tier | Count | Trains? | Eviction |
|------|-------|---------|----------|
| Milestone | 4 | No | Log-spaced recomputation (gradual swap) |
| Frontier | 4 | No | Weakest retires on promotion (sustained margin required) |
| Dynamic | 8 | Yes | Lowest Elo (min games) on graduation |
| Newest | 4 | No | FIFO into dynamic |
| Anchor | 3-5 | No | Never (manually curated) |

## Separating Training Matches from Rating Matches

> Added based on review: this resolves several open questions at once and must be a hard rule, not a soft guideline.

**Training matches** produce rollout data for PPO updates. They do NOT update Elo. Used for:
- Learner's main training loop (vs pool opponents)
- Dynamic model co-training (vs other dynamics)

**Rating matches** are inference-only with fixed weights. They DO update Elo. Used for:
- Background tournament calibration (all tiers)
- Anchor bank calibration

This separation resolves the question of whether dynamic models should train against frontier/milestone models: dynamics train from training matches only (against other dynamics and the learner), but their Elo is calibrated from rating matches against all tiers including anchors. Frontier/milestone Elo is only updated from rating matches — it is never contaminated by training-match results.

The learner's main training matches do NOT count toward dynamic model training. Keep training data sources clean.

## Elo Interpretation

### Named Elo Outputs

> Renamed based on review: "composite" overclaims authority. Each Elo output should describe what it actually measures.

| Name | Source | What it answers |
|------|--------|-----------------|
| **League Elo** | All rating matches in the visible league | Drama, matchmaking, promotion decisions. Not a benchmark. |
| **Anchor Elo** | Rating matches vs anchor bank only | Absolute-ish progress against fixed references. The closest thing to ground truth. |
| **Frontier Elo** | Rating matches vs frontier tier only | "Am I improving against current-frontier benchmarks?" Primary short-term progress signal. |
| **Milestone Elo** | Rating matches vs milestone tier only | "How much better am I than historical me?" Should only ever go up. Long-arc progress. |
| **Dynamic Elo** | Rating matches vs dynamic tier only | "Am I keeping up with the co-evolving ecosystem?" Flat is fine (opponents are improving too). |

### UI Requirements

1. **Per-tier Elo breakdown in UI.** Show Anchor, Frontier, and League Elo prominently. Milestone and Dynamic Elo available on drill-down.
2. **Visual tier indicators everywhere ratings appear.** Every place an entry name shows up (leaderboard, matchup matrix, Elo chart, match history, player cards) must show the tier icon.
3. **Separate Elo chart series.** Split the learner's Elo chart into per-tier lines, or at minimum allow a tier filter. A flat League Elo line hiding "Frontier up, Dynamic down" must never be misread as stagnation.

## Implementation Phases

> Reordered based on review: measurement must come before more non-stationarity. You need to see whether the system is lying to you before making it more complicated.

### Phase 1: Tiered Pool Management (no training)

Add tier labels to league entries. Update eviction logic to respect tiers. Add tier icons to the UI. All models remain frozen — dynamic models just have the *label* for now.

- DB migration: add `tier TEXT NOT NULL DEFAULT 'newest'` to `league_entries`
- Pool eviction: tier-aware (log-recompute for milestone, FIFO for newest, lowest-Elo for dynamic, retire-weakest for frontier)
- Graduation logic: oldest newest -> dynamic (hybrid approach)
- Milestone recomputation: periodic log-spaced target recalc, snap to nearest snapshot
- Early-training mode: milestone slots act as frontiers until claimed
- UI: tier icons in leaderboard, match history, matchup matrix, Elo chart legend

### Phase 2: Anchor Bank + Rating Pipeline + Per-Tier Analytics

> Moved ahead of dynamic training so we can validate measurement before adding complexity.

- Anchor bank: storage, protection from eviction, manual curation UI
- Separate rating match pipeline: inference-only, results tagged as `match_type='rating'`
- Per-tier Elo computation: separate Elo tracked per tier pairing
- Anchor Elo and Frontier Elo as first-class dashboard numbers
- Filterable Elo chart (show/hide by tier)
- Player cards show per-tier breakdown

### Phase 3: Dynamic Model Training

Only after Phase 2 confirms measurement is working correctly.

- Extend tournament to collect rollout buffers (not just win/loss counts)
- Per-model optimizer state: save/load Adam state alongside checkpoint
- Training step after match: 2-4 PPO epochs on collected rollout data
- Training matches tagged as `match_type='training'` — no Elo updates
- Checkpoint update: write new weights after training
- Memory budget: validate 8 x ~48MB fits alongside main training

## Open Questions

- **How many tournament games before graduation Elo is "calibrated enough"?** 5 matchups? 10? Depends on K-factor and opponent diversity.
- **Catastrophic forgetting in dynamics:** A dynamic model that only plays against one opponent repeatedly could overfit to that matchup. The round-robin scheduling (fewest h2h games first) mitigates this, but worth monitoring.
- **Milestone checkpoint storage:** Log-spaced milestones need their checkpoint files preserved even after they'd normally be eligible for cleanup. Tag milestone checkpoints as protected.
- **Anchor selection criteria:** What constitutes a good anchor? Needs to span skill range. Manual curation is fine initially but may want automation later.
- **Dynamic training data sources:** Currently scoped to tournament matches only. If dynamics stagnate, consider whether learner training matches could provide supplementary signal (with appropriate safeguards).

## Resource Estimates

Current setup (b6c96 SE-ResNet, ~3M params):

| Resource | Current | With Tiered Pool |
|----------|---------|-----------------|
| League GPU VRAM | ~50MB (2 models for tournament) | ~450MB (8 dynamic w/ optimizer + 2 loaded for match) |
| Disk per checkpoint | ~12MB | ~36MB (weights + optimizer state) |
| Total disk (20 entries + anchors) | ~240MB | ~500MB (8 dynamic w/ optimizer + 12 inference-only + anchors) |
| Tournament match time | ~30s (inference only) | ~90s (inference + 2-4 PPO steps for dynamics) |
| DB writes per match | 5 rows | 5 rows + checkpoint write + match_type tag |

All well within "loads of headroom" territory.

## Review History

- **2026-04-05 (initial):** Original four-tier design with 5/5/5 counts and composite Elo.
- **2026-04-05 (review):** Incorporated feedback:
  - Added anchor bank as hidden fixed reference frame for calibration
  - Hard separation of training matches (PPO, no Elo) from rating matches (inference-only, Elo)
  - Reordered phases: measurement (Phase 2) before dynamic training (Phase 3)
  - Tightened frontier promotion: must be strong against anchors, not just farming ecology
  - Tightened frontier retirement: sustained margin required, not just "shinier thing appeared"
  - Fixed count inconsistency: 4/4/8/4 throughout (was 5/5/5 in headers vs 4/4/8/4 in table)
  - Renamed "Static" -> "Frontier", "Composite Elo" -> "League Elo", added "Anchor Elo"
  - Resolved open question on dynamic training sources: training matches only, rating matches for Elo
