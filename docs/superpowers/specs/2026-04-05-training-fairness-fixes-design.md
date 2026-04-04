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

### Impact

- Learner's displayed Elo "resets" each rotation — this is correct behavior
- Background tournament calibrates the new entry quickly
- Pool average will naturally settle rather than inflating monotonically

## Change 2: Per-Game Color Randomization

### Design

Make `learner_side` a per-env numpy array (`shape=(num_envs,)`) instead of a scalar `int`. Initialize randomly (50/50 black/white). When a game ends (`dones`), re-randomize that env's color for the next game.

### Type Widening

The three functions that consume `learner_side` all use comparisons like `array == learner_side` or `array != learner_side`. These work identically with element-wise array comparison:

- `to_learner_perspective(rewards, pre_players, learner_side)` — `pre_players != learner_side` produces per-env bool array
- `sign_correct_bootstrap(next_values, current_players, learner_side)` — same
- `split_merge_step(..., learner_side=learner_side)` — `current_players == learner_side` produces per-env mask

Type hints widen from `int` to `int | np.ndarray`. No logic changes needed in these functions.

### Step Loop Changes

```
Initialization (before step loop):
  learner_side = np.random.randint(0, 2, size=num_envs)
  learner_side_t = torch.from_numpy(learner_side).to(device)

During step (use tensor for GPU comparisons):
  learner_moved = pre_players_t == learner_side_t
  learner_next = current_players_t == learner_side_t

After step processing (re-randomize for completed games):
  done_np = dones.bool().cpu().numpy()
  if done_np.any():
      learner_side[done_np] = np.random.randint(0, 2, size=done_np.sum())
      learner_side_t = torch.from_numpy(learner_side).to(device)
```

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

### Performance Budget

**VRAM**: Models are 14MB each. 20 opponents = 280MB — trivially fits alongside the learner.

**Compute**: The total number of opponent observations per step is unchanged (~num_envs/2). The difference is batch fragmentation: one batch of ~256 becomes ~10-15 smaller batches spread across opponents. GPU utilization per batch is lower for small batches, but the models are small (14MB) and the total work is the same. On a 4060 this should be a modest overhead, not a bottleneck.

### Architecture

**New state per epoch**:
- `opponent_models: dict[int, torch.nn.Module]` — all pool entries loaded, keyed by entry ID
- `env_opponent_ids: np.ndarray` shape `(num_envs,)` — which opponent each env is playing
- `env_opponent_entries: list[OpponentEntry]` — parallel lookup for Elo updates

**Epoch start**:
1. Load all pool entries into `opponent_models` dict
2. Sample an opponent for each env (using existing `OpponentSampler` logic per env)
3. Store assignments in `env_opponent_ids`

**Step loop — split_merge_step changes**:
The function currently takes a single `opponent_model`. Change to accept the opponent models dict and per-env opponent IDs. Group opponent envs by model, run sequential forward passes per group, merge actions back. The learner forward pass is unchanged.

New signature:
```python
def split_merge_step(
    obs, legal_masks, current_players,
    learner_model,
    opponent_models: dict[int, torch.nn.Module],  # was: opponent_model
    env_opponent_ids: np.ndarray,                   # new
    learner_side: int | np.ndarray = 0,
    value_adapter=None,
) -> SplitMergeResult:
```

Inside, the opponent forward pass becomes:
```
for opp_id, model in opponent_models.items():
    opp_env_mask = (env_opponent_ids == opp_id) & opponent_mask_np
    if not opp_env_mask.any():
        continue
    indices = opp_env_mask.nonzero()[0]
    # forward pass on model with obs[indices], merge actions back
```

**On game reset (dones)**:
- Re-sample opponent for that env (same sampling logic)
- Update `env_opponent_ids[env]`

**Elo updates on game completion**:
- Use `env_opponent_ids[env]` to identify the correct opponent entry
- Elo update is attributed to the opponent that actually played the full game

### Sampling Strategy

Use the existing `OpponentSampler.sample()` per env. This maintains the 80/20 historical/current-best ratio. Over an epoch with many game completions across 512 envs, all opponents get natural exposure proportional to their sampling weight.

### Backward Compatibility

When only 1 opponent is in the pool (early training, no league), the cohort degenerates to the current behavior — all envs play the same (only) opponent.

### Files Changed

- `katago_loop.py`: 
  - `split_merge_step` signature and opponent loop
  - Epoch setup: load all opponents, assign per-env
  - Step loop: pass models dict + env IDs to split_merge_step
  - Dones handling: re-sample opponent per env
  - Elo update: use per-env opponent ID for attribution
- `league.py`:
  - `OpponentPool.load_all_opponents(device) -> dict[int, Module]` — new method

## Testing Strategy

### Change 1 (Elo carry-forward removal)
- Existing `_rotate_seat` tests updated to verify new entry starts at 1000.0
- Verify Elo history shows reset on rotation

### Change 2 (Color randomization)
- Unit test: `learner_side` array shape and randomization
- Unit test: re-randomization only affects done envs
- Integration test: `to_learner_perspective` and `sign_correct_bootstrap` work with array input
- Integration test: `split_merge_step` produces correct learner/opponent masks with per-env sides

### Change 3 (Per-env opponents)
- Unit test: opponent assignment on reset
- Unit test: opponent stickiness (no change mid-game)
- Unit test: Elo attribution uses correct opponent per env
- Integration test: multiple opponent forward passes produce valid actions
- Test with pool_size=1 (degenerate case)

## Out of Scope

- Balanced color pairing for calibration matches (tournament.py) — future work
- Changing the OpponentSampler weighting strategy
- Opponent device placement changes (all opponents go to same device as current single opponent)
