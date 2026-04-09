# Cross-Slot Model Batching for Tournament Throughput

**Date:** 2026-04-09
**Status:** Approved
**Goal:** Achieve full round-robin coverage (every pair plays 3 games) in ~1 round per epoch instead of ~8 epochs per round.

## Problem

League tournament rounds are bottlenecked at ~30 games/min despite GPU sitting at ~20% utilization. Two root causes:

1. **`effective_parallel` capped at 11 slots** by `max_resident_models // 2` formula, despite `parallel_matches=128` in config. The formula assumes worst-case 2 unique models per slot, but the LRU cache shares model objects across slots. With 20 pool entries and `max_resident_models=22`, all models fit in cache — the cap is unnecessary.

2. **Per-slot serial inference** in `ConcurrentMatchPool.run_round()`. Each ply iteration runs 2 forward passes per active slot (one per player), each with batch size ~2 (half of `envs_per_match=4`). With 11 slots, that's 22 serial GPU kernel launches per ply, dominated by launch overhead rather than compute.

**Evidence from logs (round 49):** 18 entries, 153 pairings, 469 games, 974.1s (16.2 min), 29 games/min. Round spanned epochs 88-95 (7 epochs). At 20 entries (190 pairings), projected ~20 min per round.

## Changes

### Change 1: Remove `effective_parallel` cap

**File:** `keisei/config.py`

**Current:**
```python
@property
def effective_parallel(self) -> int:
    return min(self.parallel_matches, self.max_resident_models // 2)
```

**After:** `effective_parallel` returns `parallel_matches` unconditionally. The `__post_init__` warning about `max_resident_models < parallel_matches * 2` is removed.

**New validation** in `AppConfig.__post_init__`: if `league` and `concurrency` are both configured, warn when `max_resident_models < max_active_entries` (cache can't hold the full pool, risk of thrashing). This replaces the silent parallelism degradation with an explicit, actionable warning at the right abstraction level.

### Change 2: Cross-slot model batching in `run_round()`

**File:** `keisei/training/concurrent_matches.py`

Replace the per-slot serial inference in the game loop (lines 293-361) with batched inference grouped by model identity.

#### Per-ply structure (before):

```
for slot in active_slots:
    model_a(obs[slot.player_a_envs])    # batch ~2
    model_b(obs[slot.player_b_envs])    # batch ~2
```

#### Per-ply structure (after):

**Phase 1 — Collect batches.** Iterate active slots. For each slot:
- Check zero-legal-actions guard (per-slot, same as current)
- Record `pre_step_players` (per-slot, same as current)
- Collect pre-step rollout data: obs, masks, perspective (per-slot, same as current)
- Split env indices by current player into player_a and player_b sets
- Add `(global_env_indices, slot_ref)` to a dict keyed by `id(model)`

**Phase 2 — Batched forward passes.** For each unique model:
- Concatenate all env indices across all slots that need this model
- Run ONE `torch.no_grad()` forward pass on the concatenated batch
- Apply legal masking, softmax, Categorical.sample()
- Scatter sampled actions back into the global `actions` tensor by index

**Phase 3 — Per-slot rollout actions.** For slots collecting rollout data:
- Extract `actions[s:e]` from the global tensor (same as current `slot._actions.append`)

#### Invariants preserved:
- Slot rollout buffers get exactly one append per ply per slot (obs, masks, perspective, actions, rewards, dones)
- `to_result()` contract unchanged
- Zero-legal guard still per-slot (checked before adding to batches)
- Inactive env handling unchanged (argmax over legal_masks)
- Post-step processing (rewards, dones, game counting) unchanged
- Slot lifecycle (creation, assignment, completion, swap-in) unchanged

#### Expected impact (20 entries, 64 active slots):
- Forward passes per ply: ~128 -> ~20 (one per unique model)
- Batch size: ~2 -> ~12-15 average
- ~6x fewer kernel launches, better GPU SM occupancy
- Combined with effective_parallel fix: waves per round drop from ~14 to ~3
- Projected round time for 190 pairings: ~4-5 min (down from ~20 min)

## Config recommendations

After the code changes, the `keisei-500k-league.toml` `[league.concurrency]` section should be adjusted:

```toml
[league.concurrency]
parallel_matches = 64        # 190 pairings / 64 = 3 waves (was 128, limited to 11)
envs_per_match = 4           # unchanged
total_envs = 256             # 64 * 4 (was 512, 468 envs wasted)
max_resident_models = 22     # unchanged (holds full 20-entry pool)
```

`parallel_matches=64` is sufficient for 3 waves per round. Higher values give diminishing returns (more pairings per wave, but more model groups to forward-pass).

## Testing strategy

1. **Unit tests for batched inference:** Mock models, verify actions are correctly scattered back to the right env indices across slots.
2. **Regression test:** Existing `test_concurrent_matches.py` tests must pass unchanged (slot lifecycle, game counting, rollout collection).
3. **Integration test:** Run a round with known models, verify game counts and Elo updates match the serial path.
4. **Config validation test:** Verify `AppConfig.__post_init__` warns when `max_resident_models < max_active_entries`.

## Out of scope

- `pin_memory` for CPU tensors (tracked separately as keisei-32065dc74b)
- Reducing `total_envs` waste for inactive slots (config change only, no code needed)
- Async model loading or prefetching
