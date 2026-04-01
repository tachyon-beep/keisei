# Score Head Material Balance Fix

## Purpose

Fix the score head training signal: replace game-outcome rewards (±1/76 ≈ ±0.013) with actual material difference from the board state. Compute material balance at **every step** (not just terminals), giving the score head dense gradient signal across the full rollout. Also simplifies the code by removing NaN sentinel masking.

## Context

### Problem

`katago_loop.py` line 275 computes score targets as:
```python
score_targets[terminal_mask] = rewards[terminal_mask] / self.score_norm
```

`rewards` are game outcomes (±1.0, 0.0). `SCORE_NORMALIZATION = 76.0` was designed for material difference range (±200), not for game outcomes. Dividing ±1 by 76 produces ±0.013. With `lambda_score=0.02`, the score loss contribution is ~3.4×10⁻⁷ per sample — effectively zero gradient.

Additionally, non-terminal steps use NaN sentinels with `isnan()` masking in the PPO update. This means the score head only receives gradient from ~0.5% of samples (terminal states). Even with correct targets, this sparse signal limits learning speed.

### Root Cause

The Rust engine's `compute_reward()` returns game outcome only. No material count is exposed via `StepResult`. The training loop has no access to the board's piece inventory.

### Fix

Add `material_balance(pos, perspective) -> i32` to shogi-core. Expose it via `StepMetadata` **at every step**. Use it for score targets for all positions. Remove the NaN sentinel / masking logic — every position gets a real score target.

This is both correct (the score head should learn to estimate material advantage from any board position) and simpler (removes NaN masking, the `isnan()` filter, the `score_valid.any()` branch, and the zero-loss fallback in the PPO update).

## Architecture

### Piece Values

Standard computer Shogi piece values:

| Piece | Base | Promoted |
|-------|------|----------|
| Pawn (FU) | 1 | 7 (Tokin) |
| Lance (KY) | 3 | 6 |
| Knight (KE) | 4 | 6 |
| Silver (GI) | 5 | 6 |
| Gold (KI) | 6 | — |
| Bishop (KA) | 8 | 10 (Horse) |
| Rook (HI) | 10 | 12 (Dragon) |
| King (OU) | 0 | — |

Max one-sided material: 9×1 + 2×3 + 2×4 + 2×5 + 2×6 + 1×8 + 1×10 = 63. With all promotions, theoretical max advantage is ~130. `SCORE_NORMALIZATION = 76.0` maps the typical range to roughly [-1, 1].

### `material_balance` and `piece_value`

**Module:** `shogi-core/src/rules.rs` — alongside the existing `piece_impasse_value()` and `compute_impasse_score()` functions. This follows the established pattern: evaluation-related free functions live in `rules.rs`, not as methods on data types.

Note: `piece_impasse_value` uses simplified values (Bishop/Rook = 5, others = 1) for impasse adjudication. `piece_value` uses standard material values for training. These are distinct evaluation semantics — both belong in `rules.rs` but serve different purposes.

```rust
/// Compute material balance from `perspective`'s point of view.
/// Positive = perspective has more material. Counts board pieces + hand pieces.
/// King is excluded — it is never captured in standard Shogi, so including it
/// would add an identical constant to both sides with no effect on the differential.
///
/// Takes &Position (not &GameState) matching compute_impasse_score pattern.
pub fn material_balance(pos: &Position, perspective: Color) -> i32 {
    let mut balance: i32 = 0;
    // Board pieces
    for sq_idx in 0..81u8 {
        if let Some(piece) = pos.piece_at(Square::new_unchecked(sq_idx)) {
            let value = piece_value(piece.piece_type(), piece.is_promoted());
            if piece.color() == perspective {
                balance += value;
            } else {
                balance -= value;
            }
        }
    }
    // Hand pieces (never promoted)
    for &hpt in &HandPieceType::ALL {
        let pt = hpt.to_piece_type();
        let value = piece_value(pt, false);
        let own = pos.hand_count(perspective, hpt) as i32;
        let opp = pos.hand_count(perspective.opponent(), hpt) as i32;
        balance += value * own;
        balance -= value * opp;
    }
    balance
}

/// Material value of a piece for training score head targets.
/// Distinct from piece_impasse_value() which uses simplified counts for adjudication.
pub fn piece_value(pt: PieceType, promoted: bool) -> i32 {
    match (pt, promoted) {
        (PieceType::Pawn, false) => 1,
        (PieceType::Pawn, true) => 7,     // Tokin
        (PieceType::Lance, false) => 3,
        (PieceType::Lance, true) => 6,
        (PieceType::Knight, false) => 4,
        (PieceType::Knight, true) => 6,
        (PieceType::Silver, false) => 5,
        (PieceType::Silver, true) => 6,
        (PieceType::Gold, _) => 6,         // Gold cannot promote; debug_assert in tests
        (PieceType::Bishop, false) => 8,
        (PieceType::Bishop, true) => 10,   // Horse
        (PieceType::Rook, false) => 10,
        (PieceType::Rook, true) => 12,     // Dragon
        (PieceType::King, _) => 0,         // King excluded from material count
    }
}
```

Cost: ~81 squares + 7 hand piece types = ~88 iterations per call. At 128 envs per step, that's ~11K iterations total — nanoseconds in Rust. Negligible even at every step.

### VecEnv Integration

Add `material_balance_buffer: Vec<i32>` to `VecEnv`. Compute **at every step** (not just terminal), inside the rayon parallel closure, after `make_move` and before writing observations:

```rust
// Compute material balance from the perspective of the player who just moved
// (consistent with how rewards and observations are attributed).
let last_mover = game.position.current_player.opponent();
*material_balance_ptr.offset(i) = material_balance(&game.position, last_mover);
```

This goes in the per-env processing block, outside the `if terminated || truncated` conditional. Every step produces a valid material balance.

**ORDERING CONSTRAINT for terminal states:** At terminal states, material balance must be computed BEFORE the auto-reset line `*game = GameState::with_max_ply(max_ply)` — after reset, the game state is gone. Since material balance is computed for every step (not just terminal), and the terminal block runs after the per-step processing, this is naturally satisfied as long as the material balance write precedes the terminal block.

Expose via `StepMetadata`:

```python
class StepMetadata:
    captured_piece: NDArray[np.uint8]
    termination_reason: NDArray[np.uint8]
    ply_count: NDArray[np.uint16]
    material_balance: NDArray[np.int32]  # per-env, valid every step
```

### Training Loop Change

In `katago_loop.py`, replace the NaN-sentinel score target logic:

```python
# OLD: NaN sentinel with terminal-only targets
score_targets = torch.full((self.num_envs,), float("nan"), device=self.device)
score_targets[terminal_mask] = rewards[terminal_mask] / self.score_norm
```

with:

```python
# NEW: material balance at every step — dense signal, no NaN masking needed
material = torch.tensor(
    np.asarray(step_result.step_metadata.material_balance),
    dtype=torch.float32, device=self.device,
)
score_targets = material / self.score_norm
```

Every position gets a real score target. No NaN, no masking.

### PPO Update Simplification

In `katago_ppo.py`, replace the NaN-masked score loss:

```python
# OLD: NaN masking with fallback
score_valid = ~batch_score_targets.isnan()
if score_valid.any():
    score_loss = F.mse_loss(
        output.score_lead.squeeze(-1)[score_valid],
        batch_score_targets[score_valid],
    )
else:
    score_loss = output.score_lead.sum() * 0.0
```

with:

```python
# NEW: straight MSE over all samples — every position has a real target
score_loss = F.mse_loss(output.score_lead.squeeze(-1), batch_score_targets)
```

This removes 6 lines of branching logic and the NaN-related edge cases entirely.

### Buffer Guard Update

The `KataGoRolloutBuffer.add()` guard threshold must be widened from `> 2.0` to `> 3.0`. With material/76, extreme positions can reach ~1.7. The NaN filter is no longer needed since all targets are real values:

```python
if score_targets.abs().max() > 3.0:
    raise ValueError(
        f"score_targets appear unnormalized: max abs value = "
        f"{score_targets.abs().max().item():.1f}, expected in [-2.0, +2.0] approximately. "
        f"Divide by score_normalization before storing."
    )
```

### SL Pipeline

Add a FIXME comment to `sl/prepare.py` at the score target computation:

```python
# FIXME(keisei-8ad9dd8509): score_targets use game outcome (±1/76 ≈ ±0.013),
# not material difference. The score head will learn near-zero targets from
# this data. Real material scoring requires Rust replay of positions.
score_targets.append(raw_score / SCORE_NORMALIZATION)
```

### What Does NOT Change

- `SCORE_NORMALIZATION = 76.0` — correct for material range
- Score head architecture in `se_resnet.py` — already designed for this
- SL pipeline placeholder — stays as-is until Rust replay is implemented
- `lambda_score = 0.02` — with dense per-step signal and targets in [-1.7, +1.7], the score term is ~1-4% of total loss, which is appropriate

### What Gets Simpler

- `katago_loop.py`: no NaN sentinel, no `terminal_mask` filtering for score targets
- `katago_ppo.py`: no `isnan()` check, no `score_valid.any()` branch, no zero-loss fallback
- `KataGoRolloutBuffer.add()`: no NaN-aware guard logic

## Scope

| Component | Change | Risk |
|-----------|--------|------|
| `shogi-core/src/rules.rs` | Add `material_balance()` + `piece_value()` | Low — pure functions, no side effects |
| `shogi-gym/src/vec_env.rs` | Add buffer, populate every step, expose in StepMetadata | Low — follows existing buffer pattern |
| `shogi_gym/_native.pyi` | Add type stub for material_balance on StepMetadata | Trivial |
| `katago_loop.py` | Replace NaN score targets with per-step material | Low — simpler code |
| `katago_ppo.py` | Remove NaN masking in score loss; straight MSE | Low — simpler code |
| `katago_ppo.py` buffer guard | Widen threshold to 3.0, remove NaN filter | Low |
| `sl/prepare.py` | Add FIXME comment | Trivial |
| Tests (Rust) | material_balance at startpos, after captures, edge cases | Standard |
| Tests (Python) | Mock VecEnv returns material_balance, verify score targets | Standard |

## Non-Goals

- SL pipeline real encoding (deferred to Rust replay implementation)
- Dynamic or learned piece values (YAGNI)

## Dependencies

- None beyond the existing shogi-core API (`Position.piece_at`, `Position.hand_count`, `HandPieceType::ALL`)
- Does not depend on any Plan E work

## Testing

### Rust Unit Tests

1. `test_material_balance_startpos` — should be 0 (symmetric)
2. `test_material_balance_after_capture` — make a capture move, verify balance shifts by piece value
3. `test_material_balance_with_hand_pieces` — position with pieces in hand counted correctly
4. `test_material_balance_promoted_pieces` — promoted pieces use promoted values
5. `test_material_balance_perspective` — Black's balance = -White's balance
6. `test_piece_value_exhaustive` — every piece type returns expected value
7. `test_piece_value_gold_not_promoted` — Gold with promoted=true still returns 6 (defensive)

### Python Integration Tests

1. `test_step_metadata_has_material_balance` — field exists, shape (num_envs,), dtype int32
2. `test_material_balance_zero_at_startpos` — first step after reset returns 0 (symmetric board)
3. `test_material_balance_changes_after_capture` — drive game to a capture, verify nonzero
4. `test_score_targets_are_material_divided_by_norm` — mock VecEnv with known material_balance, verify `score_targets = material / 76.0`
5. `test_score_loss_no_nan_masking` — verify PPO update computes score loss over full batch (no NaN filtering)
