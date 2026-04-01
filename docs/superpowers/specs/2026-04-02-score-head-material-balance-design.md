# Score Head Material Balance Fix

## Purpose

Fix the score head training signal: replace game-outcome rewards (±1/76 ≈ ±0.013) with actual material difference from the board state at terminal positions. The score head currently receives zero useful gradient and learns nothing.

## Context

### Problem

`katago_loop.py` line 275 computes score targets as:
```python
score_targets[terminal_mask] = rewards[terminal_mask] / self.score_norm
```

`rewards` are game outcomes (±1.0, 0.0). `SCORE_NORMALIZATION = 76.0` was designed for material difference range (±200), not for game outcomes. Dividing ±1 by 76 produces ±0.013. With `lambda_score=0.02`, the score loss contribution is ~3.4×10⁻⁷ per sample — effectively zero gradient.

### Root Cause

The Rust engine's `compute_reward()` returns game outcome only. No material count is exposed via `StepResult`. The training loop has no access to the board's piece inventory at terminal states.

### Fix

Add `material_balance(perspective: Color) -> i32` to `GameState` in shogi-core. Expose it via `StepResult`. Use it for score targets instead of rewards.

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

### `GameState.material_balance`

```rust
impl GameState {
    /// Compute material balance from `perspective`'s point of view.
    /// Positive = perspective has more material. Counts board pieces + hand pieces.
    /// King is excluded (always present, never captured in standard Shogi).
    pub fn material_balance(&self, perspective: Color) -> i32 {
        let mut balance: i32 = 0;
        // Board pieces
        for sq_idx in 0..81 {
            if let Some(piece) = self.position.piece_at(Square::new_unchecked(sq_idx)) {
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
            let own = self.position.hand_count(perspective, hpt) as i32;
            let opp = self.position.hand_count(perspective.opponent(), hpt) as i32;
            balance += value * own;
            balance -= value * opp;
        }
        balance
    }
}

fn piece_value(pt: PieceType, promoted: bool) -> i32 {
    match (pt, promoted) {
        (PieceType::Pawn, false) => 1,
        (PieceType::Pawn, true) => 7,     // Tokin
        (PieceType::Lance, false) => 3,
        (PieceType::Lance, true) => 6,
        (PieceType::Knight, false) => 4,
        (PieceType::Knight, true) => 6,
        (PieceType::Silver, false) => 5,
        (PieceType::Silver, true) => 6,
        (PieceType::Gold, _) => 6,         // Gold cannot promote
        (PieceType::Bishop, false) => 8,
        (PieceType::Bishop, true) => 10,   // Horse
        (PieceType::Rook, false) => 10,
        (PieceType::Rook, true) => 12,     // Dragon
        (PieceType::King, _) => 0,         // King excluded from material count
    }
}
```

This runs once per terminal state (not every step). ~81 iterations + 7 hand piece types = trivial cost.

### VecEnv Integration

Add `material_balance_buffer: Vec<i32>` to `VecEnv`. At terminal states (inside the `if terminated || truncated` block in the rayon closure), compute and store:

```rust
let last_mover = game.position.current_player.opponent();
*material_balance_ptr.offset(i) = game.material_balance(last_mover);
```

For non-terminal steps, the value is 0 (unused — Python filters by `terminal_mask`).

Expose via `StepResult`:

```python
class StepResult:
    # ... existing fields ...
    material_balance: NDArray[np.int32]  # per-env material balance at terminal
```

### Training Loop Change

In `katago_loop.py`, replace:

```python
score_targets[terminal_mask] = rewards[terminal_mask] / self.score_norm
```

with:

```python
material = torch.from_numpy(
    np.asarray(step_result.material_balance)
).to(self.device)
score_targets[terminal_mask] = material[terminal_mask].float() / self.score_norm
```

### What Does NOT Change

- `SCORE_NORMALIZATION = 76.0` — correct for material range
- Score head architecture in `se_resnet.py` — already designed for this
- Buffer guard threshold (2.0) — material/76 rarely exceeds 2.0
- Score loss computation in `katago_ppo.py` — MSE on normalized material is correct
- SL pipeline placeholder — stays as-is until Rust replay is implemented

## Scope

| Component | Change | Risk |
|-----------|--------|------|
| `shogi-core` (Rust) | Add `material_balance()` + `piece_value()` | Low — pure function, no side effects |
| `shogi-gym/vec_env.rs` | Add buffer, populate at terminal, expose in StepResult | Low — follows existing buffer pattern |
| `shogi_gym/_native.pyi` | Add type stub | Trivial |
| `shogi_gym/__init__.py` | No change needed (StepResult is auto-generated by PyO3) | None |
| `katago_loop.py` | Replace reward-based score targets with material-based | Low — one-line change + numpy conversion |
| Tests (Rust) | material_balance at startpos, after captures, edge cases | Standard |
| Tests (Python) | Mock VecEnv returns material_balance, verify score scaling | Standard |

## Non-Goals

- Material balance for non-terminal positions (would require per-step computation; deferred)
- SL pipeline real encoding (deferred to Rust replay implementation)
- Dynamic or learned piece values (YAGNI)
- Score targets for draw positions (material_balance is meaningful at draws too — keep it)

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

### Python Integration Tests

1. `test_step_result_has_material_balance` — field exists, shape (num_envs,), dtype int32
2. `test_material_balance_zero_at_startpos` — non-terminal step returns 0
3. `test_material_balance_nonzero_at_terminal` — drive to terminal, verify non-zero
4. `test_score_targets_use_material` — mock VecEnv with known material_balance, verify `score_targets = material / 76.0`
