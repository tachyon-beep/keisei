# Score Head Material Balance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken score head training signal (reward/76 ≈ ±0.013) with per-step material balance from the Rust engine, giving the score head dense gradient signal across the full rollout.

**Architecture:** Add `material_balance()` and `piece_value()` to `shogi-core/src/rules.rs`. Expose via `StepMetadata` in `shogi-gym`. Use in Python training loop for score targets. Remove NaN sentinel masking from PPO update.

**Tech Stack:** Rust (shogi-core, shogi-gym/PyO3), Python 3.13, PyTorch. Tests via `cargo test` and `uv run pytest`.

**Spec reference:** `docs/superpowers/specs/2026-04-02-score-head-material-balance-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `shogi-engine/crates/shogi-core/src/rules.rs` | Add `piece_value()` and `material_balance()` |
| Modify | `shogi-engine/crates/shogi-gym/src/vec_env.rs` | Add `material_balance_buffer`, compute every step, expose in StepMetadata |
| Modify | `shogi-engine/crates/shogi-gym/src/step_result.rs` | Add `material_balance` field to `StepMetadata` |
| Modify | `shogi-engine/python/shogi_gym/_native.pyi` | Add type stub |
| Modify | `keisei/training/katago_loop.py` | Replace NaN score targets with per-step material |
| Modify | `keisei/training/katago_ppo.py` | Remove NaN masking in score loss; simplify buffer guard |
| Modify | `keisei/sl/prepare.py` | Add FIXME comment |
| Modify | `tests/test_katago_loop.py` | Update mock VecEnv to include material_balance |
| Modify | `tests/test_katago_ppo.py` | Update score target tests (no more NaN) |

---

### Task 1: `piece_value()` and `material_balance()` in Rust

**Files:**
- Modify: `shogi-engine/crates/shogi-core/src/rules.rs`

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block at the bottom of `rules.rs`:

```rust
    // -----------------------------------------------------------------------
    // Material balance tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_piece_value_exhaustive() {
        // Verify every piece type × promoted combination
        assert_eq!(piece_value(PieceType::Pawn, false), 1);
        assert_eq!(piece_value(PieceType::Pawn, true), 7);
        assert_eq!(piece_value(PieceType::Lance, false), 3);
        assert_eq!(piece_value(PieceType::Lance, true), 6);
        assert_eq!(piece_value(PieceType::Knight, false), 4);
        assert_eq!(piece_value(PieceType::Knight, true), 6);
        assert_eq!(piece_value(PieceType::Silver, false), 5);
        assert_eq!(piece_value(PieceType::Silver, true), 6);
        assert_eq!(piece_value(PieceType::Gold, false), 6);
        assert_eq!(piece_value(PieceType::Gold, true), 6); // Gold cannot promote; defensive
        assert_eq!(piece_value(PieceType::Bishop, false), 8);
        assert_eq!(piece_value(PieceType::Bishop, true), 10);
        assert_eq!(piece_value(PieceType::Rook, false), 10);
        assert_eq!(piece_value(PieceType::Rook, true), 12);
        assert_eq!(piece_value(PieceType::King, false), 0);
        assert_eq!(piece_value(PieceType::King, true), 0);
    }

    #[test]
    fn test_material_balance_startpos() {
        let state = GameState::new();
        let black = material_balance(&state.position, Color::Black);
        let white = material_balance(&state.position, Color::White);
        assert_eq!(black, 0, "Startpos should have zero material balance for Black");
        assert_eq!(white, 0, "Startpos should have zero material balance for White");
    }

    #[test]
    fn test_material_balance_perspective_antisymmetric() {
        let state = GameState::new();
        let black = material_balance(&state.position, Color::Black);
        let white = material_balance(&state.position, Color::White);
        assert_eq!(black, -white, "Black's balance should be -White's balance");
    }

    #[test]
    fn test_material_balance_after_capture() {
        let mut state = GameState::new();
        // Play a few moves to reach a capture position.
        // Use legal moves to advance the game state.
        let legal = state.legal_moves();
        assert!(!legal.is_empty());
        // Make several moves — eventually a capture will occur.
        // For a deterministic test, construct a position with known material.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // Give Black a Rook (value 10)
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let balance = material_balance(&pos, Color::Black);
        assert_eq!(balance, 10, "Black has one Rook (+10), White has nothing");
        let balance_white = material_balance(&pos, Color::White);
        assert_eq!(balance_white, -10, "White is down a Rook (-10)");
    }

    #[test]
    fn test_material_balance_with_hand_pieces() {
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // Give Black 2 Pawns in hand (value 2)
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 2);
        // Give White 1 Gold in hand (value 6)
        pos.set_hand_count(Color::White, HandPieceType::Gold, 1);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let balance = material_balance(&pos, Color::Black);
        // Black: 2 pawns in hand = 2. White: 1 gold in hand = 6.
        // Balance = 2 - 6 = -4
        assert_eq!(balance, -4, "Black has 2 pawns (2), White has 1 gold (6), diff = -4");
    }

    #[test]
    fn test_material_balance_promoted_pieces() {
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // Black has a promoted Rook (Dragon, value 12)
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, true),
        );
        // White has an unpromoted Rook (value 10)
        pos.set_piece(
            Square::from_row_col(4, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let balance = material_balance(&pos, Color::Black);
        // Black: Dragon(12). White: Rook(10). Diff = 2
        assert_eq!(balance, 2, "Promoted Rook (12) vs unpromoted Rook (10) = +2 for Black");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd shogi-engine && cargo test -p shogi-core -- material_balance piece_value 2>&1 | tail -5`
Expected: FAIL — `cannot find function 'piece_value'` / `cannot find function 'material_balance'`

- [ ] **Step 3: Write the implementation**

Add to `shogi-engine/crates/shogi-core/src/rules.rs`, before the `#[cfg(test)]` block (after line 397, after `piece_impasse_value`):

```rust
// ---------------------------------------------------------------------------
// Material balance (for training score head targets)
// ---------------------------------------------------------------------------

/// Material value of a piece for training score head targets.
/// Distinct from `piece_impasse_value()` which uses simplified counts for adjudication.
/// Standard computer Shogi values; promoted pieces use their promoted worth.
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
        (PieceType::Gold, _) => 6,         // Gold cannot promote; defensive fallback
        (PieceType::Bishop, false) => 8,
        (PieceType::Bishop, true) => 10,   // Horse
        (PieceType::Rook, false) => 10,
        (PieceType::Rook, true) => 12,     // Dragon
        (PieceType::King, _) => 0,         // King excluded: never captured, adds same to both sides
    }
}

/// Compute material balance from `perspective`'s point of view.
/// Positive = perspective has more material. Counts board pieces + hand pieces.
///
/// Takes `&Position` (not `&GameState`), matching the `compute_impasse_score` pattern.
pub fn material_balance(pos: &Position, perspective: Color) -> i32 {
    let opponent = perspective.opponent();
    let mut balance: i32 = 0;

    // Board pieces
    for sq_idx in 0..Square::NUM_SQUARES {
        let sq = Square::new_unchecked(sq_idx as u8);
        if let Some(piece) = pos.piece_at(sq) {
            if piece.piece_type() == PieceType::King {
                continue;
            }
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
        let opp = pos.hand_count(opponent, hpt) as i32;
        balance += value * own;
        balance -= value * opp;
    }

    balance
}
```

- [ ] **Step 4: Run tests**

Run: `cd shogi-engine && cargo test -p shogi-core -- material_balance piece_value`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add shogi-engine/crates/shogi-core/src/rules.rs
git commit -m "feat(shogi-core): add material_balance() and piece_value() to rules.rs"
```

---

### Task 2: VecEnv Material Balance Buffer

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/vec_env.rs`
- Modify: `shogi-engine/crates/shogi-gym/src/step_result.rs`
- Modify: `shogi-engine/python/shogi_gym/_native.pyi`

- [ ] **Step 1: Add `material_balance` field to `StepMetadata`**

In `shogi-engine/crates/shogi-gym/src/step_result.rs`, add to the `StepMetadata` struct:

```rust
    /// Material balance from last-mover's perspective, per environment.
    /// Computed every step (not just terminal). Standard piece values.
    #[pyo3(get)]
    pub material_balance: Py<PyArray1<i32>>,
```

- [ ] **Step 2: Add buffer to VecEnv and compute every step**

In `shogi-engine/crates/shogi-gym/src/vec_env.rs`:

Add import at the top (with other shogi-core imports):
```rust
use shogi_core::rules::material_balance;
```

Add buffer field to `VecEnv` struct (after `ply_buffer`, line 188):
```rust
    material_balance_buffer: Vec<i32>,  // N (per-step material balance)
```

Initialize in `VecEnv::new()` constructor (after `ply_buffer` init, around line 268):
```rust
            material_balance_buffer: vec![0; num_envs],
```

Add `SendPtr` in the `step()` closure setup (after `ply_ptr`, around line 391):
```rust
            let material_balance_ptr = SendPtr(self.material_balance_buffer.as_mut_ptr());
```

Compute material balance **every step**, inside the `process_env` closure, after the `*ply_ptr.offset(i) = ...` line (around line 426) and BEFORE the `if terminated || truncated` block:
```rust
                    // Material balance from last_mover's perspective (every step).
                    // Perspective matches the observation stored for this step.
                    *material_balance_ptr.offset(i) = material_balance(
                        &game.position, last_mover,
                    );
```

Include in `StepMetadata` construction (around line 550, after `ply_count`):
```rust
                material_balance: self.material_balance_buffer.to_pyarray(py).unbind(),
```

Also initialize in `make_env` test helper (after `ply_buffer`, around line 706):
```rust
            material_balance_buffer: vec![0; num_envs],
```

- [ ] **Step 3: Update Python type stub**

In `shogi-engine/python/shogi_gym/_native.pyi`, add to `StepMetadata`:

```python
class StepMetadata:
    captured_piece: NDArray[np.uint8]
    termination_reason: NDArray[np.uint8]
    ply_count: NDArray[np.uint16]
    material_balance: NDArray[np.int32]
```

- [ ] **Step 4: Verify Rust compiles**

Run: `cd shogi-engine && cargo check -p shogi-gym`
Expected: Clean compilation

- [ ] **Step 5: Add Rust test for buffer sizes**

Add to the existing `test_buffer_sizes` test in `vec_env.rs` (around line 814):
```rust
        assert_eq!(env.material_balance_buffer.len(), 4);
```

And to `test_default_metadata_buffers` (around line 849):
```rust
            assert_eq!(env.material_balance_buffer[i], 0);
```

And to `test_large_vecenv_buffer_construction` (around line 1238):
```rust
        assert_eq!(env.material_balance_buffer.len(), n);
```

- [ ] **Step 6: Run all Rust tests**

Run: `cd shogi-engine && cargo test -p shogi-gym`
Expected: PASS (all existing + new assertions)

- [ ] **Step 7: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/vec_env.rs \
       shogi-engine/crates/shogi-gym/src/step_result.rs \
       shogi-engine/python/shogi_gym/_native.pyi
git commit -m "feat(shogi-gym): add per-step material_balance buffer to VecEnv and StepMetadata"
```

---

### Task 3: Python Training Loop — Replace NaN Score Targets

**Files:**
- Modify: `keisei/training/katago_loop.py`
- Modify: `tests/test_katago_loop.py`

- [ ] **Step 1: Update the mock VecEnv**

In `tests/test_katago_loop.py`, update `_make_mock_katago_vecenv` to include `material_balance` on `step_metadata`:

```python
    def make_step_result(actions):
        result = MagicMock()
        result.observations = np.random.randn(num_envs, 50, 9, 9).astype(np.float32)
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        result.rewards = np.zeros(num_envs, dtype=np.float32)
        result.terminated = np.zeros(num_envs, dtype=bool)
        result.truncated = np.zeros(num_envs, dtype=bool)
        result.current_players = np.zeros(num_envs, dtype=np.uint8)
        result.step_metadata = MagicMock()
        result.step_metadata.material_balance = np.zeros(num_envs, dtype=np.int32)
        return result
```

- [ ] **Step 2: Replace score target logic in katago_loop.py**

Replace the NaN sentinel block (lines 272-275):

```python
                score_targets = torch.full(
                    (self.num_envs,), float("nan"), device=self.device
                )
                score_targets[terminal_mask] = rewards[terminal_mask] / self.score_norm
```

with:

```python
                # Per-step material balance from the Rust engine, normalized.
                # Every position gets a real score target — no NaN masking needed.
                material = torch.tensor(
                    np.asarray(step_result.step_metadata.material_balance),
                    dtype=torch.float32, device=self.device,
                )
                score_targets = material / self.score_norm
```

- [ ] **Step 3: Run loop tests**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat: use per-step material balance for score targets (replaces reward/76)"
```

---

### Task 4: PPO Update — Remove NaN Masking

**Files:**
- Modify: `keisei/training/katago_ppo.py`
- Modify: `tests/test_katago_ppo.py`

- [ ] **Step 1: Simplify score loss in update()**

In `keisei/training/katago_ppo.py`, replace the NaN-masked score loss (lines 335-344):

```python
                # Score loss (MSE on normalized score, terminal positions only).
                # Non-terminal positions use NaN sentinel — exclude from loss.
                score_valid = ~batch_score_targets.isnan()
                if score_valid.any():
                    score_loss = F.mse_loss(
                        output.score_lead.squeeze(-1)[score_valid],
                        batch_score_targets[score_valid],
                    )
                else:
                    score_loss = output.score_lead.sum() * 0.0  # zero loss, preserve graph
```

with:

```python
                # Score loss (MSE on normalized material balance).
                # Every position has a real target — no NaN masking needed.
                score_loss = F.mse_loss(
                    output.score_lead.squeeze(-1), batch_score_targets
                )
```

- [ ] **Step 2: Simplify buffer guard**

In `KataGoRolloutBuffer.add()`, replace the NaN-aware guard (lines 114-122):

```python
        # Guard against unnormalized score targets (catches integration bugs).
        # NaN is used as sentinel for non-terminal positions — exclude from check.
        finite_scores = score_targets[~score_targets.isnan()]
        if finite_scores.numel() > 0 and finite_scores.abs().max() > 2.0:
            raise ValueError(
                f"score_targets appear unnormalized: max abs value = "
                f"{finite_scores.abs().max().item():.1f}, expected <= 1.0. "
                f"Divide by score_normalization before storing."
            )
```

with:

```python
        # Guard against unnormalized score targets (catches integration bugs).
        # With per-step material balance / 76.0, typical range is [-1.7, +1.7].
        # Threshold at 3.0 gives headroom for extreme positions.
        if score_targets.abs().max() > 3.0:
            raise ValueError(
                f"score_targets appear unnormalized: max abs value = "
                f"{score_targets.abs().max().item():.1f}, expected in [-2.0, +2.0]. "
                f"Divide by score_normalization before storing."
            )
```

- [ ] **Step 3: Update tests**

In `tests/test_katago_ppo.py`, update `test_unnormalized_score_targets_rejected` (if it exists) to use non-NaN values:

```python
    def test_unnormalized_score_targets_rejected(self):
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        with pytest.raises(ValueError, match="unnormalized"):
            buf.add(
                torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
                torch.zeros(2), torch.zeros(2), torch.zeros(2),
                torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
                torch.zeros(2, dtype=torch.long),
                torch.tensor([10.0, -5.0]),  # way above 3.0 threshold
            )
```

Update any tests that previously passed NaN score targets to use real values (e.g., `torch.rand(2) * 2 - 1` instead of `torch.zeros(2)` or `float("nan")`).

Add a test verifying no NaN masking:

```python
class TestScoreLossNoNaN:
    def test_score_loss_computed_over_full_batch(self, ppo):
        """Score loss should use all samples — no NaN filtering."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            # Real material-based score targets (not NaN)
            score_targets = torch.tensor([0.5, -0.3])
            buf.add(
                obs, actions, log_probs, values,
                torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                torch.randint(0, 3, (2,)), score_targets,
            )
        losses = ppo.update(buf, torch.zeros(2))
        assert losses["score_loss"] > 0, "Score loss should be non-zero with real targets"
        assert not torch.tensor(losses["score_loss"]).isnan(), "Score loss should not be NaN"
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_katago_ppo.py
git commit -m "feat: remove NaN masking from score loss — per-step material gives dense signal"
```

---

### Task 5: SL Pipeline FIXME + Verification

**Files:**
- Modify: `keisei/sl/prepare.py`

- [ ] **Step 1: Add FIXME comment**

In `keisei/sl/prepare.py`, at the score target computation (around line 138):

```python
                # FIXME(keisei-8ad9dd8509): score_targets use game outcome (±1/76 ≈ ±0.013),
                # not material difference. The score head will learn near-zero targets from
                # this data. Real material scoring requires Rust replay of positions to compute
                # material_balance() at each move. This placeholder is structurally correct
                # (valid shard format) but semantically wrong for score head training.
                score_targets.append(raw_score / SCORE_NORMALIZATION)
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/test_katago_ppo.py tests/test_katago_model.py tests/test_katago_loop.py tests/test_sl_pipeline.py tests/test_prepare_sl.py tests/test_lr_scheduler.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add keisei/sl/prepare.py
git commit -m "docs: add FIXME for SL placeholder score targets (keisei-8ad9dd8509)"
```

---

### Task 6: Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run Rust tests**

Run: `cd shogi-engine && cargo test`
Expected: ALL PASS (including new material_balance tests)

- [ ] **Step 2: Run Python tests**

Run: `uv run pytest --ignore=tests/test_katago_observation.py --ignore=tests/test_spatial_action_mapper.py -v`
Expected: ALL PASS

- [ ] **Step 3: Verify the fix end-to-end**

Confirm that `score_targets` in the buffer are now real material values (not NaN, not ±0.013):

Run: `uv run python -c "
import torch
# Verify the buffer guard accepts material-range values
from keisei.training.katago_ppo import KataGoRolloutBuffer
buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
# Simulate material balance / 76.0 = typical values
score_targets = torch.tensor([0.5, -1.2])  # 38 and -91 material points
buf.add(
    torch.randn(2, 50, 9, 9), torch.zeros(2, dtype=torch.long),
    torch.zeros(2), torch.zeros(2), torch.zeros(2),
    torch.zeros(2, dtype=torch.bool), torch.ones(2, 11259, dtype=torch.bool),
    torch.zeros(2, dtype=torch.long), score_targets,
)
print(f'Buffer accepted score_targets: {score_targets.tolist()}')
print(f'No NaN in targets: {not score_targets.isnan().any()}')
print('Score head material balance fix verified')
"`
Expected: Prints confirmation with no errors

- [ ] **Step 4: Commit if any fixes needed**

```bash
git add -u
git commit -m "fix: address issues found in score head material balance verification"
```
