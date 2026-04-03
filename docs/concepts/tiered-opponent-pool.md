# Tiered Opponent Pool: Static / Dynamic / Newest

**Status:** Conceptual — not yet implemented.
**Prerequisite:** Let the current round-robin tournament run for several days to validate Elo calibration works correctly before adding complexity.

## Motivation

The current opponent pool treats all entries identically: frozen snapshots that only serve as evaluation targets. This works for progress measurement but doesn't push the learner as hard as it could be pushed. A tiered pool gives us both a fixed yardstick *and* a co-evolving competitive ecosystem.

## The Three Tiers

### Static (5 entries) — Measurement

- **Icon:** Shield
- **Purpose:** Fixed benchmarks that anchor the Elo scale. "Is the learner getting stronger?" is answered by win rate against these.
- **Behavior:** Never train, never get evicted. Manually pinned (existing pin infrastructure).
- **Selection:** Chosen to span the Elo range — e.g. the weakest, a low-mid, mid, high-mid, and strongest snapshot at the time of pinning. Replaced rarely and deliberately (e.g. every few hundred epochs, promote the top dynamic model to static to slowly track the frontier).

### Dynamic (10 entries) — Competitive Pressure

- **Icon:** Crossed swords
- **Purpose:** Co-evolving opponents that keep the learner challenged. "Is the training environment hard enough?" is answered by win rate against these.
- **Behavior:** Receive PPO updates from tournament games. Their weights change over time.
- **Training mechanics:**
  - After each tournament match, the losing model (or both) gets a few PPO update steps using rollout data from the match.
  - Each dynamic model needs persistent optimizer state (Adam momentum/variance) saved alongside its checkpoint.
  - Estimated memory per model (b6c96, ~3M params): ~48MB (weights + optimizer + gradients). 10 models = ~480MB — fits easily on a dedicated GPU.
- **Eviction:** When a newest entry graduates to dynamic, the lowest-Elo dynamic with >= 5 completed matchups is evicted.

### Newest (5 entries) — Fresh Blood

- **Icon:** Seedling
- **Purpose:** Rolling window of recent learner snapshots. Shows how the learner has evolved recently.
- **Behavior:** Frozen (no training). Created automatically by the existing snapshot mechanism.
- **Lifecycle:** FIFO — newest in, oldest out. The oldest newest entry either graduates to dynamic (if its calibrated Elo qualifies) or is simply dropped.

## Graduation: Newest → Dynamic

When a newest entry ages out of the newest queue (pushed out by a newer snapshot), it must go somewhere:

**Option A — Always graduate:** The entry becomes dynamic and the weakest dynamic is evicted. Simple, predictable. Risk: weak snapshots dilute the dynamic pool.

**Option B — Elo-gated:** The entry becomes dynamic only if its calibrated Elo exceeds the lowest dynamic's Elo. Otherwise it's discarded. Requires the entry to have played enough tournament matches for a meaningful Elo (minimum games threshold).

**Option C — Hybrid:** Graduate always, but dynamic eviction targets the lowest-Elo entry with >= N games. A freshly graduated entry with uncalibrated Elo is protected until it's had a fair chance.

**Recommendation:** Option C. It keeps fresh blood flowing without discarding potentially strong models that just haven't been calibrated yet.

## Static Promotion

Periodically (every K epochs, or manually), the highest-Elo dynamic model can be "frozen" and promoted to static, replacing the weakest static benchmark. This means the static tier slowly tracks the frontier:

- Short-term progress: measured against recent static benchmarks.
- Long-term progress: measured against the original static benchmarks (which are now much weaker).

This should be rare and deliberate — perhaps every 200-500 epochs.

## Elo Interpretation

**Critical design constraint:** The composite Elo number becomes harder to interpret with mixed tiers. Two mitigations:

1. **Per-tier Elo breakdown in UI.** Show "vs Static: 1450, vs Dynamic: 1280, vs Newest: 1390" alongside the composite. The static number is the true progress metric.

2. **Visual tier indicators everywhere ratings appear.** Every place an entry name shows up (leaderboard, matchup matrix, Elo chart, match history, player cards) must show the tier icon. A flat composite Elo line that's actually hiding "static Elo up, dynamic Elo down" must never be misread as stagnation.

3. **Separate Elo chart series.** Consider splitting the learner's Elo chart into "vs Static" and "vs Dynamic" lines, or at minimum allowing a tier filter on the chart.

## Implementation Phases

### Phase 1: Tiered Pool Management (no training)

Add tier labels to league entries (DB column: `tier` enum: `static`, `dynamic`, `newest`). Update eviction logic to respect tiers. Add tier icons to the UI. All models remain frozen — dynamic models just have the *label* for now.

- DB migration: add `tier TEXT NOT NULL DEFAULT 'newest'` to `league_entries`
- Pool eviction: tier-aware (never evict static, FIFO for newest, lowest-Elo for dynamic)
- Graduation logic: oldest newest → dynamic (option C)
- UI: tier icons in leaderboard, match history, matchup matrix, Elo chart legend

### Phase 2: Dynamic Model Training

Add PPO updates for dynamic-tier models after tournament matches.

- Extend tournament to collect rollout buffers (not just win/loss counts)
- Per-model optimizer state: save/load Adam state alongside checkpoint
- Training step after match: 2-4 PPO epochs on collected rollout data
- Checkpoint update: write new weights after training
- Memory budget: validate 10 × ~48MB fits alongside main training

### Phase 3: Per-Tier Elo Analytics

- Per-tier Elo breakdown in player cards and stats banner
- Filterable Elo chart (show/hide by tier)
- "vs Static" progress metric as a first-class dashboard number

## Open Questions

- **How many tournament games before graduation Elo is "calibrated enough"?** 5 matchups? 10? Depends on K-factor and opponent diversity.
- **Should dynamic models train against static models too, or only against each other?** Training against static models would shift dynamic Elo relative to static, which is useful for calibration. But it means static models' Elo changes (from the dynamic model's side), which could confuse the picture. Probably: dynamic trains from all matches, but static Elo is only updated from inference-only calibration matches.
- **Catastrophic forgetting:** A dynamic model that only plays against one opponent repeatedly could overfit to that matchup. The round-robin scheduling (fewest h2h games first) mitigates this, but worth monitoring.
- **Should the learner's main training matches count toward dynamic model training?** Probably not — those are the learner's training signal, not the opponent's. Keep training data sources clean.

## Resource Estimates

Current setup (b6c96 SE-ResNet, ~3M params):

| Resource | Current | With Tiered Pool |
|----------|---------|-----------------|
| League GPU VRAM | ~50MB (2 models for tournament) | ~600MB (10 dynamic w/ optimizer + 2 loaded for match) |
| Disk per checkpoint | ~12MB | ~36MB (weights + optimizer state) |
| Tournament match time | ~30s (inference only) | ~90s (inference + 2-4 PPO steps) |
| DB writes per match | 5 rows | 5 rows + checkpoint write |

All well within "loads of headroom" territory.
