# Training Fairness Fixes

**Date**: 2026-04-05
**Status**: Draft
**Scope**: katago_loop.py, league.py — three changes to fix Elo inflation, color bias, and opponent diversity

## Problem Statement

The training loop has three fairness issues:

1. **Elo inflation via carry-forward**: When the learner rotates to a new snapshot, the old Elo is copied to the new entry. New models enter at ~1650 instead of earning their rating, creating a ratchet effect that inflates the entire pool.

2. **Black-only learner**: `learner_side = 0` is hardcoded — the learner always plays sente. This biases training toward first-mover-advantage positions and makes Elo non-comparable to balanced play.

3. **Single opponent per epoch**: One opponent is sampled per epoch. With 20 pool entries, most opponents go unseen for many epochs, giving narrow training signal and slow Elo calibration.

## Change 1: Remove Elo Carry-Forward

### Design

Delete the carry-forward logic in `_rotate_seat` (lines 1228-1238). New snapshots enter at the DB default of 1000.0 and earn their rating through tournament matches.

### Files Changed

- `katago_loop.py`: Remove lines 1228-1230 and 1237-1238 in `_rotate_seat`

### Important: Two Distinct Carry-Forward Mechanisms

The codebase has **two** carry-forward blocks that must not be confused:

1. **Rotation carry-forward** (lines 1228-1238 in `_rotate_seat`) — copies the old snapshot's Elo to the new snapshot on seat rotation. **This is what we are deleting.**
2. **Chart-continuity carry-forward** (lines 1112-1122) — writes identical Elo values to `elo_history` every epoch for entries that didn't play, so the Elo chart has continuous lines with no gaps. **This must be preserved.**

### Impact

- Learner's displayed Elo "resets" each rotation — this is correct behavior
- Background tournament calibrates the new entry quickly
- Pool average will naturally settle rather than inflating monotonically

## Change 2: Per-Game Color Randomization

### Design

Make `learner_side` a per-env numpy array (`shape=(num_envs,)`) instead of a scalar `int`. Initialize randomly (50/50 black/white). When a game ends (`dones`), re-randomize that env's color for the next game.

### Type Widening

Four functions consume `learner_side` — all use comparisons that work identically with element-wise array comparison. **All four signatures must be widened simultaneously**:

- `to_learner_perspective(rewards, pre_players, learner_side)` (line 102) — `pre_players != learner_side` produces per-env bool array
- `sign_correct_bootstrap(next_values, current_players, learner_side)` (line 116) — same
- `split_merge_step(..., learner_side=learner_side)` (line 248) — `current_players == learner_side` produces per-env mask
- `_negate_where(values, condition)` (line 83) — the shared implementation that all three above delegate to. It calls `torch.tensor(condition, ...)` where `condition` is already the result of the numpy comparison (a bool array). No signature change needed on `_negate_where` itself, but it is the function that actually executes the comparison result.

Type hints widen from `int` to `int | np.ndarray` on `to_learner_perspective`, `sign_correct_bootstrap`, and `split_merge_step`. No logic changes needed in these functions.

### Dtype Consistency

`current_players` is `dtype=np.uint8` (line 689). To avoid silent dtype promotion in numpy comparisons, initialize `learner_side` with an explicit matching dtype:

```python
learner_side = np.random.randint(0, 2, size=num_envs, dtype=np.uint8)
```

### Bootstrap Sign Correction Invariant

`sign_correct_bootstrap` is called at epoch end (line 1010-1012) using `learner_side`. Since `learner_side` is mutated by re-randomization throughout the epoch, the following invariant must hold:

> **Invariant**: At all times, `learner_side[env]` reflects the color assignment for the game **currently running** in that env. Re-randomization happens exclusively at game boundaries (on `dones`), never mid-game.

Any off-by-one in re-randomization timing silently corrupts GAE targets via wrong sign correction. This invariant must be tested explicitly.

### Metric Semantic Change

Previously, `win_rate` meant "learner win rate as black (sente)." After this change, it means "learner win rate across both colors." The `black_win_rate` / `white_win_rate` metrics (lines 1166-1171) track winner color regardless of who the learner is, which remains valid. No dashboard label update is strictly required, but users comparing pre- and post-change win rates will see an apparent discontinuity. Add a log message at the first epoch with color randomization enabled noting the semantic change.

### Config Gate

Add `color_randomization: bool = True` to `LeagueConfig`. When `False`, `learner_side` remains a scalar `0` (current behavior). This allows disabling color randomization without code changes if training quality degrades.

### Step Loop Changes

```python
# Initialization (before step loop):
learner_side = np.random.randint(0, 2, size=num_envs, dtype=np.uint8)
learner_side_t = torch.from_numpy(learner_side).to(device)

# During step (use tensor for GPU comparisons):
learner_moved = pre_players_t == learner_side_t
learner_next = current_players_t == learner_side_t

# After step processing (re-randomize for completed games):
# Use in-place scatter to update only changed positions on the GPU tensor,
# avoiding a full cudaMemcpy on every done step.
done_np = dones.bool().cpu().numpy()
if done_np.any():
    new_sides = np.random.randint(0, 2, size=int(done_np.sum()), dtype=np.uint8)
    learner_side[done_np] = new_sides
    done_indices = torch.from_numpy(np.flatnonzero(done_np)).to(device)
    learner_side_t[done_indices] = torch.from_numpy(new_sides).to(device)
```

**Why in-place scatter**: The original approach (`learner_side_t = torch.from_numpy(learner_side).to(device)`) allocates a new GPU tensor and does a full CPU→GPU copy on every step that has any completed game (~50% of steps). The in-place scatter only copies the changed elements. With 512 envs and ~5-15% games completing per step, this replaces a 512-element `cudaMemcpy` with a ~50-element targeted scatter.

### Files Changed

- `katago_loop.py`: Type hints on 3 functions, initialization at line 763, tensor copy, GPU comparisons at lines 784/801, re-randomization after dones processing

## Change 3: Per-Env Sticky Opponents (Cohort Model)

### Design

Instead of one opponent per epoch, assign each env its own opponent on game reset. Opponent identity is **sticky per env** — it never changes mid-game. On `dones`, the env gets a new opponent sampled from the pool.

### Why Not Mid-Game Swaps

Swapping opponents at fixed step boundaries would:
- Contaminate training data (game started vs A, continued vs B)
- Corrupt Elo attribution (result credited to wrong opponent)
- Create inconsistent game dynamics mid-episode

### Config Gate

Add `per_env_opponents: bool = True` to `LeagueConfig`. When `False`, the current behavior (single opponent per epoch) is preserved. This allows reverting without code changes if batch fragmentation proves too costly.

### Performance Budget

**VRAM**: Models are 14MB each. 20 opponents = 280MB — fits alongside the learner. Note: at epoch boundaries, both the old and new opponent dicts may be alive briefly before Python GC collects the old one (see Memory Management below).

**Compute**: The total number of opponent observations per step is unchanged (~num_envs/2). The difference is batch fragmentation: one batch of ~256 becomes ~10-15 smaller batches spread across opponents. GPU utilization per batch is lower for small batches (batch size ~13 with 20 opponents), and 20 sequential kernel launches have non-trivial overhead on a 4060. Since all opponents share the same architecture (they are learner snapshots), a future optimization could concatenate all opponent observations into one batch and dispatch through a single forward pass. For now, the sequential approach is simpler to implement correctly.

**Benchmark requirement**: Before merging, measure wall-clock step time with 1 opponent vs. 20 opponents at 512 envs. Set a regression threshold: step time must not exceed 2x baseline. If exceeded, implement the concatenated-batch optimization.

### Architecture

**New state per epoch**:
- `opponent_models: dict[int, torch.nn.Module]` — all pool entries loaded, keyed by entry ID
- `env_opponent_ids: np.ndarray` shape `(num_envs,)`, dtype `int64` — which opponent each env is playing
- `opponent_results: dict[int, list[int, int, int]]` — per-opponent `[wins, losses, draws]` accumulated during the epoch, for Elo updates at epoch end

**Epoch start**:
1. **Memory cleanup**: Explicitly `del self._opponent_models` + `torch.cuda.empty_cache()` before loading new opponents, to avoid peak VRAM holding two full sets during the transition (~280MB on a 4060 with 8GB)
2. **Load all pool entries**: via `OpponentPool.load_all_opponents(device)` (see below)
3. **Cache sampler entries**: Snapshot the pool entries list once at epoch start. Use this cached list for all `OpponentSampler.sample()` calls during the epoch, avoiding a `SELECT *` query on every game completion (~2500 queries/epoch with 512 envs)
4. **Sample opponents**: For each env, call `OpponentSampler.sample_from(cached_entries)` to assign an initial opponent
5. **Initialize tracking**: `opponent_results = {entry_id: [0, 0, 0] for entry_id in opponent_models}`

**Pool Freeze Semantics**:

`add_snapshot` and `_evict_if_needed` can modify the pool mid-epoch (periodic snapshots at lines 1134-1140). To avoid stale model references after eviction:

- **Freeze the pool view** for the duration of the epoch. `opponent_models` and the cached entries list are the epoch's "working set."
- Pool mutations (snapshot additions, evictions) still happen at the DB level, but the training loop does not reload opponents mid-epoch.
- On the next epoch start, the fresh `load_all_opponents` picks up any changes.

This is simpler and safer than staleness checks on every game reset.

**Step loop — split_merge_step changes**:

The function currently takes a single `opponent_model`. For backward compatibility, support both the old single-model and new multi-model signatures:

```python
def split_merge_step(
    obs, legal_masks, current_players,
    learner_model,
    opponent_model: torch.nn.Module | None = None,       # legacy single-model
    opponent_models: dict[int, torch.nn.Module] | None = None,  # new multi-model
    env_opponent_ids: np.ndarray | None = None,            # new
    learner_side: int | np.ndarray = 0,
    value_adapter=None,
) -> SplitMergeResult:
```

When `opponent_models` is provided, use the multi-model path. When only `opponent_model` is provided, wrap it as `{0: opponent_model}` and proceed through the same code path. This eliminates duplicate logic.

Inside, the opponent forward pass becomes:
```python
opponent_mask_np = (~learner_mask).cpu().numpy()
for opp_id, model in active_opponents.items():
    opp_env_mask = opponent_mask_np
    if env_opponent_ids is not None:
        opp_env_mask = (env_opponent_ids == opp_id) & opponent_mask_np
    if not opp_env_mask.any():
        continue
    indices = np.flatnonzero(opp_env_mask)
    # forward pass on model with obs[indices], merge actions back
```

**Why `np.flatnonzero`**: `np.ndarray.nonzero()[0]` returns the same result but is fragile — it differs from `torch.Tensor.nonzero()` which returns a 2-D tensor. `np.flatnonzero()` is unambiguous for 1-D arrays and consistent with the existing PyTorch `nonzero(as_tuple=True)[0]` pattern used elsewhere in `split_merge_step`.

**On game reset (dones)**:
- Re-sample opponent for that env using `OpponentSampler.sample_from(cached_entries)`
- Update `env_opponent_ids[env]`

**Elo Attribution (Epoch End)**:

The current epoch-level Elo update uses a single `_current_opponent_entry` (line 1090-1110). With per-env opponents, this must be replaced with per-opponent aggregation:

```python
# During the epoch, on each game completion:
opp_id = env_opponent_ids[env]
if learner_won:
    opponent_results[opp_id][0] += 1  # wins
elif learner_lost:
    opponent_results[opp_id][1] += 1  # losses
else:
    opponent_results[opp_id][2] += 1  # draws

# At epoch end, one Elo update per opponent that played:
for opp_id, (w, l, d) in opponent_results.items():
    total = w + l + d
    if total == 0:
        continue
    result_score = (w + 0.5 * d) / total
    opp_entry = cached_entries_by_id[opp_id]
    new_learner_elo, new_opp_elo = compute_elo_update(
        learner_elo, opp_entry.elo_rating, result=result_score, k=k,
    )
    # Update both Elo ratings in DB
```

This avoids per-game Elo updates (which would cause K=32 volatility) and correctly attributes results to each opponent.

### Sampling Strategy

Add a `sample_from(entries: list[OpponentEntry])` method to `OpponentSampler` that accepts a pre-fetched entries list instead of querying the pool on every call. The existing `sample()` method can delegate to `sample_from(self.pool.list_entries())` for backward compatibility.

At epoch start, cache `entries = pool.list_entries()` and use `sampler.sample_from(entries)` for all per-env sampling during the epoch. This maintains the 80/20 historical/current-best ratio while avoiding ~2500 SQLite queries per epoch. Over an epoch with many game completions across 512 envs, all opponents get natural exposure proportional to their sampling weight.

### Backward Compatibility

When only 1 opponent is in the pool (early training, no league), the cohort degenerates to the current behavior — all envs play the same (only) opponent.

### `OpponentPool.load_all_opponents` Specification

New method on `OpponentPool`. Must reuse the existing `load_opponent` method (which handles `weights_only=True`, atomic checkpoint loading, and `model.eval()`) rather than reimplementing model construction:

```python
def load_all_opponents(self, device: str = "cpu") -> dict[int, torch.nn.Module]:
    """Load all pool entries. Skips entries with missing/corrupt checkpoints."""
    models: dict[int, torch.nn.Module] = {}
    for entry in self.list_entries():
        try:
            models[entry.id] = self.load_opponent(entry, device=device)
        except (FileNotFoundError, RuntimeError) as e:
            logger.warning(
                "Skipping pool entry id=%d (epoch %d): %s",
                entry.id, entry.created_epoch, e,
            )
    return models
```

Per-entry load failures log a warning and exclude that entry from the epoch's working set. The training loop does not crash because one historical checkpoint is corrupted.

### Files Changed

- `katago_loop.py`: 
  - `split_merge_step` signature change (backward-compatible with single-model path)
  - Opponent forward pass loop with `np.flatnonzero`
  - Epoch setup: memory cleanup, load all opponents, cache entries, assign per-env
  - Step loop: pass models dict + env IDs to split_merge_step
  - Dones handling: re-sample opponent per env
  - Elo update: per-opponent W/L/D accumulation, batch update at epoch end
- `league.py`:
  - `OpponentPool.load_all_opponents(device) -> dict[int, Module]` — new method with error handling
  - `OpponentSampler.sample_from(entries) -> OpponentEntry` — new method accepting pre-fetched entries

## Cross-Cutting Concerns

### Dones Processing Order

When Changes 2 and 3 are both active, `learner_side` and `env_opponent_ids` are both per-env arrays that change on `dones`. The processing order on game completion must be:

1. **Finalize pending transitions** — flush the completed game's transition data to the rollout buffer (existing protocol, steps 1-2 in the pending transition block)
2. **Elo attribution** — record the result in `opponent_results[env_opponent_ids[env]]`
3. **Re-randomize color** — update `learner_side[env]` and `learner_side_t[env]`
4. **Re-sample opponent** — update `env_opponent_ids[env]`

Steps 3 and 4 set up state for the **next** game in that env. Steps 1 and 2 consume state from the **completed** game. This ordering ensures no cross-contamination.

### Mixed Cohort Transition Period

After deploying Change 1, the pool will contain a mix of inflated-Elo entries (from pre-fix rotations, rated ~1400-1700) and correctly-rated new entries (starting at 1000). This mixed cohort persists for approximately `max_pool_size × epochs_per_seat` = 20 × 50 = **1000 epochs** until all inflated entries are evicted via FIFO.

During this period:
- `current_best` (max Elo entry) points at an inflated-rated legacy entry, not a newly calibrated one
- The 20% current-best sampling pressure is still directed at the genuinely strongest-playing model (inflation is systematic — relative ordering is still accurate)
- Elo updates from beating inflated opponents may produce artificially large gains for the learner

This is acceptable and self-correcting. Add a log warning at the first rotation after deployment: "Pool contains mixed inflated/reset entries. Elo signal will stabilize after ~{remaining} rotations."

### Deployment Strategy

Consider deploying Change 1 alone first (one-line deletion). Verify pool mean stabilizes over one seat rotation (~50 epochs). Then deploy Changes 2 and 3 together. This isolates the simplest, highest-impact change and provides a clean Elo baseline for measuring the effect of the other changes.

## Testing Strategy

### Change 1 (Elo carry-forward removal)
- Existing `_rotate_seat` tests updated to assert `new_entry.elo_rating == 1000.0` (currently no assertion on Elo value)
- Verify Elo history has no rows for the new entry's ID immediately after rotation
- Verify old entry's final Elo does not appear on the new entry
- Edge case: `_rotate_seat` when old entry was evicted (`old_entry is None`) — should still produce 1000.0

### Change 2 (Color randomization)
- Unit test: `learner_side` array shape, dtype (`uint8`), and randomization
- Unit test: re-randomization only affects done envs (non-done envs unchanged)
- Unit test: **bootstrap sign correction with per-env `learner_side` array** — call `sign_correct_bootstrap` with mixed learner_side values, verify per-env correctness
- Integration test: `to_learner_perspective` and `sign_correct_bootstrap` work with array input
- Integration test: `split_merge_step` produces correct learner/opponent masks with per-env sides
- **Invariant test**: after re-randomization on done envs, assert `learner_side[env]` matches the color for the currently running game (not the completed one)
- **Pending transition interaction test**: verify that when done envs trigger color re-randomization, stale pending transitions for those envs are cleared before new `learner_side` values take effect

### Change 3 (Per-env opponents)
- Unit test: opponent assignment on reset
- Unit test: opponent stickiness (no change mid-game)
- Unit test: Elo attribution uses correct opponent per env — specifically that `opponent_results[opp_id]` accumulates only games played against that opponent
- **Action source verification test**: verify env 0's action comes from model A and env 1's from model B, using deterministic model pairs with distinguishable outputs (not just "valid actions")
- **Concurrent games test**: two games running simultaneously with different opponents, verify each opponent's Elo is updated only by games assigned to it
- Integration test: multiple opponent forward passes produce valid actions
- Test with pool_size=1 (degenerate case)
- Test with all entries below Elo floor except one (de facto pool_size=1)
- Test `load_all_opponents` with one corrupt checkpoint — verify it returns the other models and logs a warning
- Test `sample_from(cached_entries)` produces same distribution as `sample()`

### Cross-Change Interaction Tests

- **Color + opponents combined**: 4 envs, 2 opponents, per-env color assignment. Verify `to_learner_perspective` applies each env's own `learner_side`, and Elo update attributes each result to the correct opponent
- **Elo reset + opponents combined**: new snapshot at 1000 and old snapshot at 1200 both active. Verify Elo attribution separates them correctly when both are assigned to envs simultaneously
- **End-to-end smoke test** (all 3 changes active): run a short training epoch with all changes enabled, verify:
  1. Win/loss rates are approximately equal for black and white over many games
  2. Elo history contains separate entries for each opponent
  3. New snapshots after rotation start at 1000.0

### Performance Regression Test

Measure wall-clock time per step with 1 opponent vs. 20 opponents at 512 envs. Threshold: step time must not exceed 2x baseline with 1 opponent.

## Config Changes

Add to `LeagueConfig` in `config.py`:

```python
color_randomization: bool = True     # Change 2: per-game color randomization
per_env_opponents: bool = True       # Change 3: per-env sticky opponents
```

Both default to `True` (new behavior). Setting to `False` reverts to pre-fix behavior for that change. Change 1 has no config gate — it is a pure bug fix with no reason to revert.

## Out of Scope

- Balanced color pairing for calibration matches (tournament.py) — future work
- Changing the OpponentSampler weighting strategy
- Opponent device placement changes (all opponents go to same device as current single opponent)
