# KataGo Plan A: Rust Engine Extensions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Rust shogi-gym crate with a 50-channel observation generator and an 11,259-action spatial action mapper.

**Architecture:** Two new implementations of existing traits (`ObservationGenerator`, `ActionMapper`) in shogi-gym, selected by mode strings on VecEnv. No changes to shogi-core — it already has `repetition_map`, `is_in_check()`, and `hash_history`.

**Tech Stack:** Rust, PyO3, numpy, rayon. Tests via `cargo test` (Rust) and `uv run pytest` (Python integration).

**Spec reference:** `docs/superpowers/specs/2026-04-01-katago-se-resnet-design.md` — Slices 1 & 2.

---

## Task Dependency Graph

- Tasks 1–4: KataGo observation generator + tests (can be done first, no dependencies)
- Tasks 5–6: Spatial action mapper + tests (independent of 1–4)
- Task 7: VecEnv wiring (depends on both)
- Tasks 8–9: Python integration tests (depend on Task 7 + maturin build)
- Task 10: Full suite verification

## Implementation Note

**Task 3's repetition test helper should be validated interactively first** before writing the full test suite around it. The Gold oscillation pattern assumes standard shogi starting position piece placement — if the engine uses a non-standard piece arrangement for testing or the legal move generator has any quirks, the `expect("Gold 6i-5h should be legal")` panics are harder to debug mid-test-run than they are to pre-validate with a quick `cargo run` snippet. Fifteen minutes of upfront validation of the move sequence saves a potentially confusing Task 3 failure.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `shogi-gym/src/katago_observation.rs` | 50-channel `KataGoObservationGenerator` |
| Create | `shogi-gym/src/spatial_action_mapper.rs` | 11,259-action `SpatialActionMapper` |
| Modify | `shogi-gym/src/lib.rs` | Register new modules + export new PyO3 classes |
| Modify | `shogi-gym/src/vec_env.rs` | Add `observation_mode`/`action_mode` params, generic over generators |
| Create | `tests/test_katago_observation.py` | Python integration tests for 50-channel observations |
| Create | `tests/test_spatial_action_mapper.py` | Python integration tests for spatial mapper |

All paths relative to `shogi-engine/crates/` for Rust files, project root for Python tests.

---

### Task 1: KataGoObservationGenerator — Channels 0–43 (Same as Default)

**Files:**
- Create: `shogi-engine/crates/shogi-gym/src/katago_observation.rs`
- Modify: `shogi-engine/crates/shogi-gym/src/lib.rs`

The first 44 channels are identical to `DefaultObservationGenerator`. Start by cloning the logic and establishing the new 50-channel structure.

- [ ] **Step 1: Create `katago_observation.rs` with 50-channel constants and generator struct**

```rust
// shogi-engine/crates/shogi-gym/src/katago_observation.rs

//! KataGoObservationGenerator — 50-channel observation tensor.
//!
//! Channels 0-43:  Same as DefaultObservationGenerator (piece planes, hand, player, ply)
//! Channels 44-47: Repetition count (4 binary planes: 1×, 2×, 3×, 4×)
//! Channel  48:    Check indicator (1.0 if current player is in check)
//! Channel  49:    Reserved (zeros)

use pyo3::prelude::*;
use shogi_core::{Color, GameState, HandPieceType, PieceType, Square};

use crate::observation::ObservationGenerator;

pub const KATAGO_NUM_CHANNELS: usize = 50;
pub const KATAGO_NUM_SQUARES: usize = 81;
pub const KATAGO_BUFFER_LEN: usize = KATAGO_NUM_CHANNELS * KATAGO_NUM_SQUARES;

const HAND_MAX_COUNTS: [f32; HandPieceType::COUNT] = [18.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0];

#[inline]
fn unpromoted_channel(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn   => 0,
        PieceType::Lance  => 1,
        PieceType::Knight => 2,
        PieceType::Silver => 3,
        PieceType::Gold   => 4,
        PieceType::Bishop => 5,
        PieceType::Rook   => 6,
        PieceType::King   => 7,
    }
}

#[inline]
fn promoted_channel(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn   => 0,
        PieceType::Lance  => 1,
        PieceType::Knight => 2,
        PieceType::Silver => 3,
        PieceType::Bishop => 4,
        PieceType::Rook   => 5,
        _ => panic!("piece type {:?} cannot be promoted", pt),
    }
}

#[pyclass]
pub struct KataGoObservationGenerator;

impl KataGoObservationGenerator {
    pub fn new() -> Self {
        KataGoObservationGenerator
    }
}

impl Default for KataGoObservationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ObservationGenerator for KataGoObservationGenerator {
    fn channels(&self) -> usize {
        KATAGO_NUM_CHANNELS
    }

    fn generate(&self, state: &GameState, perspective: Color, buffer: &mut [f32]) {
        assert_eq!(
            buffer.len(),
            KATAGO_BUFFER_LEN,
            "buffer must have length {} (got {})",
            KATAGO_BUFFER_LEN,
            buffer.len()
        );

        buffer.fill(0.0);

        let pos = &state.position;
        let opponent = perspective.opponent();
        let flip = perspective == Color::White;

        // --- Channels 0-27: Board piece planes (identical to Default) ---
        for idx in 0..KATAGO_NUM_SQUARES {
            let sq = Square::new_unchecked(idx as u8);
            if let Some(piece) = pos.piece_at(sq) {
                let piece_color = piece.color();
                let pt = piece.piece_type();
                let promoted = piece.is_promoted();
                let out_sq = if flip { 80 - idx } else { idx };

                if piece_color == perspective {
                    let ch = if promoted {
                        8 + promoted_channel(pt)
                    } else {
                        unpromoted_channel(pt)
                    };
                    buffer[ch * KATAGO_NUM_SQUARES + out_sq] = 1.0;
                } else {
                    let ch = if promoted {
                        22 + promoted_channel(pt)
                    } else {
                        14 + unpromoted_channel(pt)
                    };
                    buffer[ch * KATAGO_NUM_SQUARES + out_sq] = 1.0;
                }
            }
        }

        // --- Channels 28-34: Current player's hand counts ---
        for &hpt in &HandPieceType::ALL {
            let count = pos.hand_count(perspective, hpt) as f32;
            let max_count = HAND_MAX_COUNTS[hpt.index()];
            let normalized = count / max_count;
            let ch = 28 + hpt.index();
            let start = ch * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(normalized);
        }

        // --- Channels 35-41: Opponent's hand counts ---
        for &hpt in &HandPieceType::ALL {
            let count = pos.hand_count(opponent, hpt) as f32;
            let max_count = HAND_MAX_COUNTS[hpt.index()];
            let normalized = count / max_count;
            let ch = 35 + hpt.index();
            let start = ch * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(normalized);
        }

        // --- Channel 42: Player indicator ---
        let player_indicator = if perspective == Color::Black { 1.0_f32 } else { 0.0_f32 };
        let start = 42 * KATAGO_NUM_SQUARES;
        buffer[start..start + KATAGO_NUM_SQUARES].fill(player_indicator);

        // --- Channel 43: Move count ---
        let move_count = if state.max_ply == 0 {
            0.0_f32
        } else {
            state.ply as f32 / state.max_ply as f32
        };
        let start = 43 * KATAGO_NUM_SQUARES;
        buffer[start..start + KATAGO_NUM_SQUARES].fill(move_count);

        // --- Channels 44-47: Repetition count (binary planes) ---
        let current_hash = pos.hash;
        let rep_count = state.repetition_map.get(&current_hash).copied().unwrap_or(0);
        // rep_count is the number of times this position has been seen BEFORE this occurrence.
        // Channel 44 = seen 1× before, channel 45 = 2×, channel 46 = 3×, channel 47 = 4+×
        if rep_count >= 1 && rep_count <= 3 {
            let ch = 44 + (rep_count as usize - 1);
            let start = ch * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(1.0);
        } else if rep_count >= 4 {
            let start = 47 * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(1.0);
        }
        // rep_count == 0: all repetition channels stay 0.0 (first occurrence)

        // --- Channel 48: Check indicator ---
        if state.is_in_check() {
            let start = 48 * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(1.0);
        }

        // Channel 49: Reserved (already zeroed)
    }
}

#[pymethods]
impl KataGoObservationGenerator {
    #[new]
    pub fn py_new() -> Self {
        KataGoObservationGenerator::new()
    }

    #[getter]
    pub fn channels(&self) -> usize {
        KATAGO_NUM_CHANNELS
    }
}
```

- [ ] **Step 2: Register the module in `lib.rs`**

Add to `shogi-engine/crates/shogi-gym/src/lib.rs`:

```rust
mod katago_observation;
```

And in the `#[pymodule]` function, add:

```rust
m.add_class::<katago_observation::KataGoObservationGenerator>()?;
```

- [ ] **Step 3: Build and verify compilation**

Run: `cd shogi-engine && cargo build`
Expected: Compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/katago_observation.rs shogi-engine/crates/shogi-gym/src/lib.rs
git commit -m "feat(shogi-gym): add KataGoObservationGenerator with 50 channels"
```

---

### Task 2: KataGoObservationGenerator — Rust Unit Tests

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/katago_observation.rs` (add `#[cfg(test)]` module)

- [ ] **Step 1: Add basic structural tests**

Append to `katago_observation.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::GameState;

    fn make_gen() -> KataGoObservationGenerator {
        KataGoObservationGenerator::new()
    }

    fn make_buffer() -> Vec<f32> {
        vec![0.0_f32; KATAGO_BUFFER_LEN]
    }

    #[test]
    fn test_katago_channels() {
        let gen = make_gen();
        assert_eq!(
            <KataGoObservationGenerator as ObservationGenerator>::channels(&gen),
            50
        );
    }

    #[test]
    fn test_katago_buffer_length() {
        let gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);
        assert_eq!(buf.len(), 50 * 81);
    }

    #[test]
    fn test_katago_startpos_pieces_match_default() {
        // Channels 0-27 should be identical to DefaultObservationGenerator
        use crate::observation::{DefaultObservationGenerator, BUFFER_LEN};

        let katago_gen = make_gen();
        let default_gen = DefaultObservationGenerator::new();
        let state = GameState::new();

        let mut katago_buf = make_buffer();
        let mut default_buf = vec![0.0_f32; BUFFER_LEN];

        katago_gen.generate(&state, Color::Black, &mut katago_buf);
        default_gen.generate(&state, Color::Black, &mut default_buf);

        // Compare channels 0-27 (piece planes)
        for ch in 0..28 {
            for sq in 0..81 {
                assert_eq!(
                    katago_buf[ch * 81 + sq],
                    default_buf[ch * 81 + sq],
                    "Mismatch at ch={}, sq={} (piece planes)",
                    ch, sq
                );
            }
        }

        // Compare channels 28-43 (hand + meta)
        for ch in 28..44 {
            for sq in 0..81 {
                assert_eq!(
                    katago_buf[ch * 81 + sq],
                    default_buf[ch * 81 + sq],
                    "Mismatch at ch={}, sq={} (hand/meta planes)",
                    ch, sq
                );
            }
        }
    }

    #[test]
    fn test_katago_no_nan() {
        let gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);
        for (i, val) in buf.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at buffer position {}", i);
        }
    }

    #[test]
    fn test_katago_reserved_channel_49_zero() {
        let gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);
        let start = 49 * 81;
        for i in 0..81 {
            assert_eq!(buf[start + i], 0.0, "ch49[{}] should be 0.0", i);
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cd shogi-engine && cargo test -p shogi-gym katago`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/katago_observation.rs
git commit -m "test(shogi-gym): basic KataGoObservationGenerator unit tests"
```

---

### Task 3: Repetition Channel Tests

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/katago_observation.rs` (add repetition tests to `mod tests`)

We need to construct positions with known repetition counts. The approach: play a sequence of moves that returns to the starting position, incrementing the repetition count.

- [ ] **Step 1: Add repetition tests**

Add to the `mod tests` block in `katago_observation.rs`:

```rust
    /// Helper: play moves to create a known repetition count.
    /// Uses a simple king-oscillation sequence that repeats the position.
    fn make_state_with_repetitions(reps: u8) -> GameState {
        // Start from standard position, then play a sequence that returns
        // the position to the same state. We'll use a simple setup:
        // Move a piece back and forth to increment the repetition counter.
        //
        // From startpos: move Gold 6i-5h, opponent Gold 4a-5b,
        //                move Gold 5h-6i, opponent Gold 5b-4a.
        // This returns to startpos (rep_count increments by 1).
        use shogi_core::Move;

        let mut state = GameState::with_max_ply(500);

        for _ in 0..reps {
            // Square indexing: row * 9 + col, where col = 9 - file.
            // Gold 6i: file 6 -> col 3, rank i -> row 8, sq = 8*9+3 = 75
            // Gold 5h: file 5 -> col 4, rank h -> row 7, sq = 7*9+4 = 67
            // Gold 4a: file 4 -> col 5, rank a -> row 0, sq = 0*9+5 = 5
            // Gold 5b: file 5 -> col 4, rank b -> row 1, sq = 1*9+4 = 13

            // Black: Gold 6i (sq 75) -> 5h (sq 67)
            let moves = state.legal_moves();
            let mv_fwd = moves.iter().find(|m| matches!(m,
                Move::Board { from, to, promote: false }
                if from.index() == 75 && to.index() == 67
            )).expect("Gold 6i-5h should be legal");
            state.make_move(*mv_fwd);

            // White: Gold 4a (sq 5) -> 5b (sq 13)
            let moves = state.legal_moves();
            let mv_fwd = moves.iter().find(|m| matches!(m,
                Move::Board { from, to, promote: false }
                if from.index() == 5 && to.index() == 13
            )).expect("Gold 4a-5b should be legal");
            state.make_move(*mv_fwd);

            // Black: Gold 5h (sq 67) -> 6i (sq 75)
            let moves = state.legal_moves();
            let mv_back = moves.iter().find(|m| matches!(m,
                Move::Board { from, to, promote: false }
                if from.index() == 67 && to.index() == 75
            )).expect("Gold 5h-6i should be legal");
            state.make_move(*mv_back);

            // White: Gold 5b (sq 13) -> 4a (sq 5)
            let moves = state.legal_moves();
            let mv_back = moves.iter().find(|m| matches!(m,
                Move::Board { from, to, promote: false }
                if from.index() == 13 && to.index() == 5
            )).expect("Gold 5b-4a should be legal");
            state.make_move(*mv_back);
        }

        state
    }

    #[test]
    fn test_repetition_channels_zero_at_startpos() {
        let gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);

        // All repetition channels should be 0 at startpos (first occurrence)
        for ch in 44..=47 {
            let start = ch * 81;
            for sq in 0..81 {
                assert_eq!(
                    buf[start + sq], 0.0,
                    "ch{}[{}] should be 0.0 at startpos (no prior repetitions)",
                    ch, sq
                );
            }
        }
    }

    #[test]
    fn test_repetition_channel_44_after_one_repeat() {
        let gen = make_gen();
        let state = make_state_with_repetitions(1);
        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);

        // Position has been seen once before -> channel 44 should be 1.0
        let start44 = 44 * 81;
        for sq in 0..81 {
            assert_eq!(buf[start44 + sq], 1.0, "ch44[{}] should be 1.0 after 1 repetition", sq);
        }

        // Other repetition channels should be 0
        for ch in 45..=47 {
            let start = ch * 81;
            assert_eq!(buf[start], 0.0, "ch{}[0] should be 0.0 after 1 repetition", ch);
        }
    }

    #[test]
    fn test_repetition_channel_45_after_two_repeats() {
        let gen = make_gen();
        let state = make_state_with_repetitions(2);
        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);

        let start45 = 45 * 81;
        assert_eq!(buf[start45], 1.0, "ch45 should be 1.0 after 2 repetitions");

        let start44 = 44 * 81;
        assert_eq!(buf[start44], 0.0, "ch44 should be 0.0 after 2 repetitions");
    }

    #[test]
    fn test_repetition_channel_46_after_three_repeats() {
        let gen = make_gen();
        let state = make_state_with_repetitions(3);
        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);

        let start46 = 46 * 81;
        assert_eq!(buf[start46], 1.0, "ch46 should be 1.0 after 3 repetitions");
    }

    #[test]
    fn test_repetition_channels_mutually_exclusive() {
        let gen = make_gen();

        for reps in 0..=3u8 {
            let state = make_state_with_repetitions(reps);
            let mut buf = make_buffer();
            gen.generate(&state, Color::Black, &mut buf);

            let mut active_channels = Vec::new();
            for ch in 44..=47 {
                if buf[ch * 81] == 1.0 {
                    active_channels.push(ch);
                }
            }

            if reps == 0 {
                assert!(
                    active_channels.is_empty(),
                    "No repetition channel should be active at reps=0, got {:?}",
                    active_channels
                );
            } else {
                assert_eq!(
                    active_channels.len(), 1,
                    "Exactly one repetition channel should be active at reps={}, got {:?}",
                    reps, active_channels
                );
            }
        }
    }
```

- [ ] **Step 2: Run tests**

Run: `cd shogi-engine && cargo test -p shogi-gym katago_observation::tests::test_repetition`
Expected: All repetition tests PASS. If the Gold oscillation moves aren't legal at the expected squares, adjust the move lookup to use the actual legal move indices. The test helper may need debugging if square indices for Gold differ.

- [ ] **Step 3: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/katago_observation.rs
git commit -m "test(shogi-gym): repetition channel tests for KataGoObservationGenerator"
```

---

### Task 4: Check Indicator Channel Tests

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/katago_observation.rs` (add check tests to `mod tests`)

- [ ] **Step 1: Add check indicator tests**

Add to the `mod tests` block:

```rust
    #[test]
    fn test_check_channel_not_in_check() {
        let gen = make_gen();
        let state = GameState::new(); // startpos, nobody in check
        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);

        let start48 = 48 * 81;
        for sq in 0..81 {
            assert_eq!(buf[start48 + sq], 0.0, "ch48[{}] should be 0.0 when not in check", sq);
        }
    }

    #[test]
    fn test_check_channel_in_check() {
        use shogi_core::{Piece, PieceType, Position, Square};

        let gen = make_gen();

        // Construct a position where Black king is in check from a White rook.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // White rook on same file as Black king, giving check
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let state = GameState::from_position(pos, 500);

        // Verify Black IS in check
        assert!(state.is_in_check(), "Black king should be in check from White rook");

        let mut buf = make_buffer();
        gen.generate(&state, Color::Black, &mut buf);

        let start48 = 48 * 81;
        for sq in 0..81 {
            assert_eq!(buf[start48 + sq], 1.0, "ch48[{}] should be 1.0 when in check", sq);
        }
    }

    #[test]
    fn test_check_channel_white_perspective() {
        use shogi_core::{Piece, PieceType, Position, Square};

        let gen = make_gen();

        // White king in check from Black rook
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        pos.current_player = Color::White;
        pos.hash = pos.compute_hash();

        let state = GameState::from_position(pos, 500);
        assert!(state.is_in_check(), "White king should be in check");

        let mut buf = make_buffer();
        gen.generate(&state, Color::White, &mut buf);

        // Channel 48 should be 1.0 from White's perspective
        let start48 = 48 * 81;
        for sq in 0..81 {
            assert_eq!(buf[start48 + sq], 1.0, "ch48[{}] should be 1.0 for White in check", sq);
        }
    }
```

- [ ] **Step 2: Run tests**

Run: `cd shogi-engine && cargo test -p shogi-gym katago_observation::tests::test_check`
Expected: All check tests PASS.

- [ ] **Step 3: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/katago_observation.rs
git commit -m "test(shogi-gym): check indicator channel tests for KataGoObservationGenerator"
```

---

### Task 5: SpatialActionMapper — Core Encoding/Decoding

**Files:**
- Create: `shogi-engine/crates/shogi-gym/src/spatial_action_mapper.rs`
- Modify: `shogi-engine/crates/shogi-gym/src/lib.rs`

- [ ] **Step 1: Create the `SpatialActionMapper` with encoding constants and core logic**

```rust
// shogi-engine/crates/shogi-gym/src/spatial_action_mapper.rs

//! SpatialActionMapper — (9×9×139) = 11,259 action encoding.
//!
//! Flat index: square * 139 + move_type
//!
//! Move types per square:
//!   0-63:    Sliding moves (8 directions × 8 distances, no promotion)
//!   64-127:  Sliding moves with promotion (8 directions × 8 distances)
//!   128:     Knight jump left (no promotion)
//!   129:     Knight jump left (promotion)
//!   130:     Knight jump right (no promotion)
//!   131:     Knight jump right (promotion)
//!   132-138: Drop (7 piece types)
//!
//! Directions (relative to current player, N = toward opponent):
//!   0=N(-1,0) 1=NE(-1,+1) 2=E(0,+1) 3=SE(+1,+1)
//!   4=S(+1,0) 5=SW(+1,-1) 6=W(0,-1) 7=NW(-1,-1)

use pyo3::prelude::*;
use pyo3::types::PyDict;
use shogi_core::{Color, HandPieceType, Move, Square};

use crate::action_mapper::ActionMapper;

pub const SPATIAL_MOVE_TYPES: usize = 139;
pub const SPATIAL_NUM_SQUARES: usize = 81;
pub const SPATIAL_ACTION_SPACE_SIZE: usize = SPATIAL_NUM_SQUARES * SPATIAL_MOVE_TYPES; // 11,259

// Direction deltas: (delta_row, delta_col)
const DIRECTIONS: [(i8, i8); 8] = [
    (-1,  0), // 0: N
    (-1,  1), // 1: NE
    ( 0,  1), // 2: E
    ( 1,  1), // 3: SE
    ( 1,  0), // 4: S
    ( 1, -1), // 5: SW
    ( 0, -1), // 6: W
    (-1, -1), // 7: NW
];

// Knight jump deltas relative to current player:
// Left = (-2, -1), Right = (-2, +1)  (two forward, one to the side)
const KNIGHT_LEFT: (i8, i8) = (-2, -1);
const KNIGHT_RIGHT: (i8, i8) = (-2, 1);

#[pyclass]
pub struct SpatialActionMapper;

impl SpatialActionMapper {
    fn apply_perspective(sq: Square, perspective: Color) -> Square {
        match perspective {
            Color::Black => sq,
            Color::White => sq.flip(),
        }
    }

    /// Compute direction index and distance for a board move.
    /// Returns None if the move doesn't fit the direction+distance scheme (knight move).
    fn direction_and_distance(from_row: i8, from_col: i8, to_row: i8, to_col: i8)
        -> Option<(usize, usize)>
    {
        let dr = to_row - from_row;
        let dc = to_col - from_col;

        if dr == 0 && dc == 0 {
            return None;
        }

        // Normalize to unit direction
        let abs_dr = dr.unsigned_abs() as i8;
        let abs_dc = dc.unsigned_abs() as i8;
        let max_delta = abs_dr.max(abs_dc) as usize;

        if max_delta == 0 {
            return None;
        }

        // Must be along a straight line or diagonal
        let unit_dr = if dr == 0 { 0 } else { dr / abs_dr };
        let unit_dc = if dc == 0 { 0 } else { dc / abs_dc };

        // Check this is actually a straight/diagonal line
        if dr != 0 && dc != 0 && abs_dr != abs_dc {
            return None; // Knight move or invalid
        }

        // Find direction index
        let dir = DIRECTIONS.iter().position(|&(r, c)| r == unit_dr && c == unit_dc)?;
        let distance = max_delta; // 1-indexed (distance 1 = adjacent)

        if distance == 0 || distance > 8 {
            return None;
        }

        Some((dir, distance))
    }

    /// Check if a board move is a knight move, and if so, which one (left/right).
    fn knight_slot(from_row: i8, from_col: i8, to_row: i8, to_col: i8) -> Option<usize> {
        let dr = to_row - from_row;
        let dc = to_col - from_col;

        if (dr, dc) == KNIGHT_LEFT {
            Some(0) // left
        } else if (dr, dc) == KNIGHT_RIGHT {
            Some(1) // right
        } else {
            None
        }
    }
}

impl ActionMapper for SpatialActionMapper {
    fn encode(&self, mv: Move, perspective: Color) -> usize {
        match mv {
            Move::Board { from, to, promote } => {
                let from_p = Self::apply_perspective(from, perspective);
                let to_p = Self::apply_perspective(to, perspective);

                let from_row = from_p.row() as i8;
                let from_col = from_p.col() as i8;
                let to_row = to_p.row() as i8;
                let to_col = to_p.col() as i8;

                let source_sq = from_p.index();

                // Try direction+distance encoding first
                if let Some((dir, distance)) = Self::direction_and_distance(
                    from_row, from_col, to_row, to_col
                ) {
                    let slot = if promote {
                        64 + dir * 8 + (distance - 1) // promoted: slots 64-127
                    } else {
                        dir * 8 + (distance - 1)       // non-promoted: slots 0-63
                    };
                    return source_sq * SPATIAL_MOVE_TYPES + slot;
                }

                // Try knight encoding
                if let Some(knight_side) = Self::knight_slot(
                    from_row, from_col, to_row, to_col
                ) {
                        // Knight slots: 128=left_no_promo, 129=left_promo, 130=right_no_promo, 131=right_promo
                    let slot = 128 + knight_side * 2 + if promote { 1 } else { 0 };
                    return source_sq * SPATIAL_MOVE_TYPES + slot;
                }

                panic!(
                    "Cannot encode board move from ({},{}) to ({},{}) — not a valid direction, distance, or knight move",
                    from_row, from_col, to_row, to_col
                );
            }
            Move::Drop { to, piece_type } => {
                let to_p = Self::apply_perspective(to, perspective);
                let dest_sq = to_p.index();
                let slot = 132 + piece_type.index();
                dest_sq * SPATIAL_MOVE_TYPES + slot
            }
        }
    }

    fn decode(&self, idx: usize, perspective: Color) -> Result<Move, String> {
        if idx >= SPATIAL_ACTION_SPACE_SIZE {
            return Err(format!(
                "action index {} out of range (max {})",
                idx, SPATIAL_ACTION_SPACE_SIZE - 1
            ));
        }

        let square_idx = idx / SPATIAL_MOVE_TYPES;
        let slot = idx % SPATIAL_MOVE_TYPES;

        if slot < 128 {
            // Direction + distance move (slots 0-127)
            let promote = slot >= 64;
            let base_slot = if promote { slot - 64 } else { slot };
            let dir = base_slot / 8;
            let distance = (base_slot % 8) + 1; // 1-indexed

            let from_sq = Square::new_unchecked(square_idx as u8);
            let from_row = from_sq.row() as i8;
            let from_col = from_sq.col() as i8;

            let (dr, dc) = DIRECTIONS[dir];
            let to_row = from_row + dr * distance as i8;
            let to_col = from_col + dc * distance as i8;

            if to_row < 0 || to_row > 8 || to_col < 0 || to_col > 8 {
                return Err(format!(
                    "decoded move goes off board: from ({},{}) dir={} dist={}",
                    from_row, from_col, dir, distance
                ));
            }

            let to_sq_raw = Square::from_row_col(to_row as u8, to_col as u8)
                .map_err(|e| format!("invalid square: {}", e))?;

            let from_real = Self::apply_perspective(from_sq, perspective);
            let to_real = Self::apply_perspective(to_sq_raw, perspective);

            Ok(Move::Board { from: from_real, to: to_real, promote })
        } else if slot < 132 {
            // Knight move (slots 128-131)
            let knight_idx = slot - 128;
            let knight_side = knight_idx / 2; // 0=left, 1=right
            let promote = (knight_idx % 2) == 1;

            let from_sq = Square::new_unchecked(square_idx as u8);
            let from_row = from_sq.row() as i8;
            let from_col = from_sq.col() as i8;

            let (dr, dc) = if knight_side == 0 { KNIGHT_LEFT } else { KNIGHT_RIGHT };
            let to_row = from_row + dr;
            let to_col = from_col + dc;

            if to_row < 0 || to_row > 8 || to_col < 0 || to_col > 8 {
                return Err(format!(
                    "decoded knight move goes off board: from ({},{}) side={}",
                    from_row, from_col, knight_side
                ));
            }

            let to_sq_raw = Square::from_row_col(to_row as u8, to_col as u8)
                .map_err(|e| format!("invalid square: {}", e))?;

            let from_real = Self::apply_perspective(from_sq, perspective);
            let to_real = Self::apply_perspective(to_sq_raw, perspective);

            Ok(Move::Board { from: from_real, to: to_real, promote })
        } else {
            // Drop move (slots 132-138)
            let piece_idx = slot - 132;
            if piece_idx >= HandPieceType::COUNT {
                return Err(format!("invalid drop piece index {}", piece_idx));
            }

            let to_sq = Square::new_unchecked(square_idx as u8);
            let to_real = Self::apply_perspective(to_sq, perspective);
            let piece_type = HandPieceType::ALL[piece_idx];

            Ok(Move::Drop { to: to_real, piece_type })
        }
    }

    fn action_space_size(&self) -> usize {
        SPATIAL_ACTION_SPACE_SIZE
    }
}

#[pymethods]
impl SpatialActionMapper {
    #[new]
    pub fn new() -> Self {
        SpatialActionMapper
    }

    pub fn encode_board_move(
        &self,
        from_sq: u8,
        to_sq: u8,
        promote: bool,
        is_white: bool,
    ) -> PyResult<usize> {
        let from = Square::new(from_sq)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let to = Square::new(to_sq)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if from == to {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "from_sq and to_sq must be different",
            ));
        }
        let perspective = if is_white { Color::White } else { Color::Black };
        let mv = Move::Board { from, to, promote };
        Ok(<Self as ActionMapper>::encode(self, mv, perspective))
    }

    pub fn encode_drop_move(
        &self,
        to_sq: u8,
        piece_type_idx: usize,
        is_white: bool,
    ) -> PyResult<usize> {
        let to = Square::new(to_sq)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if piece_type_idx >= HandPieceType::COUNT {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "piece_type_idx {} out of range (max {})",
                piece_type_idx, HandPieceType::COUNT - 1
            )));
        }
        let piece_type = HandPieceType::ALL[piece_type_idx];
        let perspective = if is_white { Color::White } else { Color::Black };
        let mv = Move::Drop { to, piece_type };
        Ok(<Self as ActionMapper>::encode(self, mv, perspective))
    }

    pub fn decode(&self, py: Python<'_>, idx: usize, is_white: bool) -> PyResult<Py<PyDict>> {
        let perspective = if is_white { Color::White } else { Color::Black };
        let mv = <Self as ActionMapper>::decode(self, idx, perspective)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let dict = PyDict::new(py);
        match mv {
            Move::Board { from, to, promote } => {
                dict.set_item("type", "board")?;
                dict.set_item("from_sq", from.index() as u8)?;
                dict.set_item("to_sq", to.index() as u8)?;
                dict.set_item("promote", promote)?;
            }
            Move::Drop { to, piece_type } => {
                dict.set_item("type", "drop")?;
                dict.set_item("to_sq", to.index() as u8)?;
                dict.set_item("piece_type_idx", piece_type.index())?;
            }
        }
        Ok(dict.into())
    }

    #[getter]
    pub fn action_space_size(&self) -> usize {
        SPATIAL_ACTION_SPACE_SIZE
    }
}
```

- [ ] **Step 2: Register in `lib.rs`**

Add to `shogi-engine/crates/shogi-gym/src/lib.rs`:

```rust
mod spatial_action_mapper;
```

And in the `#[pymodule]` function:

```rust
m.add_class::<spatial_action_mapper::SpatialActionMapper>()?;
```

- [ ] **Step 3: Build**

Run: `cd shogi-engine && cargo build`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/spatial_action_mapper.rs shogi-engine/crates/shogi-gym/src/lib.rs
git commit -m "feat(shogi-gym): add SpatialActionMapper with 11,259-action spatial encoding"
```

---

### Task 6: SpatialActionMapper — Rust Unit Tests

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/spatial_action_mapper.rs` (add `#[cfg(test)]` module)

- [ ] **Step 1: Add core encoding/decoding tests**

Append to `spatial_action_mapper.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn mapper() -> SpatialActionMapper {
        SpatialActionMapper
    }

    fn trait_encode(m: &SpatialActionMapper, mv: Move, p: Color) -> usize {
        <SpatialActionMapper as ActionMapper>::encode(m, mv, p)
    }

    fn trait_decode(m: &SpatialActionMapper, idx: usize, p: Color) -> Result<Move, String> {
        <SpatialActionMapper as ActionMapper>::decode(m, idx, p)
    }

    #[test]
    fn test_action_space_size() {
        assert_eq!(SPATIAL_ACTION_SPACE_SIZE, 11_259);
        let m = mapper();
        assert_eq!(<SpatialActionMapper as ActionMapper>::action_space_size(&m), 11_259);
    }

    #[test]
    fn test_flat_index_contract() {
        // Verify: flat_index = square * 139 + move_type
        let m = mapper();
        // A rook-like move: from sq 40 (e5), direction N (0), distance 4
        let from = Square::new_unchecked(40);
        let to_row = 40i8 / 9 - 4; // row 4 - 4 = row 0
        let to_col = 40i8 % 9;     // col 4
        let to = Square::from_row_col(to_row as u8, to_col as u8).unwrap();
        let mv = Move::Board { from, to, promote: false };
        let idx = trait_encode(&m, mv, Color::Black);

        let expected_slot = 0 * 8 + 3; // dir=N(0), distance=4 -> slot index 3
        let expected = 40 * 139 + expected_slot;
        assert_eq!(idx, expected, "Flat index should be square * 139 + move_type");
    }

    #[test]
    fn test_drop_roundtrip_black() {
        let m = mapper();
        for to_idx in 0u8..81 {
            for &piece_type in &HandPieceType::ALL {
                let to = Square::new_unchecked(to_idx);
                let mv = Move::Drop { to, piece_type };
                let idx = trait_encode(&m, mv, Color::Black);

                // Verify index is in drop range
                let slot = idx % SPATIAL_MOVE_TYPES;
                assert!(slot >= 132 && slot <= 138, "Drop slot {} out of range", slot);

                let decoded = trait_decode(&m, idx, Color::Black).expect("decode failed");
                assert_eq!(decoded, mv, "Drop roundtrip failed for to={}, piece={:?}", to_idx, piece_type);
            }
        }
    }

    #[test]
    fn test_drop_roundtrip_white() {
        let m = mapper();
        for to_idx in 0u8..81 {
            for &piece_type in &HandPieceType::ALL {
                let to = Square::new_unchecked(to_idx);
                let mv = Move::Drop { to, piece_type };
                let idx = trait_encode(&m, mv, Color::White);
                let decoded = trait_decode(&m, idx, Color::White).expect("decode failed");
                assert_eq!(decoded, mv, "White drop roundtrip failed");
            }
        }
    }

    #[test]
    fn test_sliding_move_roundtrip() {
        let m = mapper();
        // Rook at e5 (sq 40, row 4, col 4): move N 1-4, S 1-4, E 1-4, W 1-4
        let from = Square::new_unchecked(40);
        let from_row = 4i8;
        let from_col = 4i8;

        for (dir_idx, (dr, dc)) in DIRECTIONS.iter().enumerate() {
            for dist in 1..=8usize {
                let to_row = from_row + dr * dist as i8;
                let to_col = from_col + dc * dist as i8;
                if to_row < 0 || to_row > 8 || to_col < 0 || to_col > 8 {
                    continue;
                }
                let to = Square::from_row_col(to_row as u8, to_col as u8).unwrap();
                for promote in [false, true] {
                    let mv = Move::Board { from, to, promote };
                    let idx = trait_encode(&m, mv, Color::Black);
                    let decoded = trait_decode(&m, idx, Color::Black)
                        .expect("decode failed");
                    assert_eq!(
                        decoded, mv,
                        "Sliding roundtrip failed: dir={}, dist={}, promote={}",
                        dir_idx, dist, promote
                    );
                }
            }
        }
    }

    #[test]
    fn test_knight_move_roundtrip() {
        let m = mapper();
        // Knight at row 4, col 4 (sq 40): can jump to (2,3) and (2,5)
        let from = Square::from_row_col(4, 4).unwrap();
        let to_left = Square::from_row_col(2, 3).unwrap();
        let to_right = Square::from_row_col(2, 5).unwrap();

        for (to, side_name) in [(to_left, "left"), (to_right, "right")] {
            for promote in [false, true] {
                let mv = Move::Board { from, to, promote };
                let idx = trait_encode(&m, mv, Color::Black);
                let slot = idx % SPATIAL_MOVE_TYPES;
                assert!(
                    slot >= 128 && slot <= 131,
                    "Knight slot {} out of range for {} side",
                    slot, side_name
                );
                let decoded = trait_decode(&m, idx, Color::Black).expect("decode failed");
                assert_eq!(decoded, mv, "Knight roundtrip failed: {}, promote={}", side_name, promote);
            }
        }
    }

    #[test]
    fn test_perspective_flip_board_move() {
        let m = mapper();
        let from = Square::new_unchecked(20);
        let to = Square::new_unchecked(11); // one step N from row 2 col 2 to row 1 col 2
        let mv = Move::Board { from, to, promote: false };

        let idx_black = trait_encode(&m, mv, Color::Black);
        let idx_white = trait_encode(&m, mv, Color::White);

        assert_ne!(idx_black, idx_white, "Perspectives should produce different indices");

        let decoded_black = trait_decode(&m, idx_black, Color::Black).unwrap();
        let decoded_white = trait_decode(&m, idx_white, Color::White).unwrap();

        assert_eq!(decoded_black, mv, "Black roundtrip failed");
        assert_eq!(decoded_white, mv, "White roundtrip failed");
    }

    #[test]
    fn test_perspective_flip_drop_move() {
        let m = mapper();
        let to = Square::new_unchecked(10);
        let mv = Move::Drop { to, piece_type: HandPieceType::Rook };

        let idx_black = trait_encode(&m, mv, Color::Black);
        let idx_white = trait_encode(&m, mv, Color::White);

        assert_ne!(idx_black, idx_white, "Drop perspectives should differ");

        let decoded_black = trait_decode(&m, idx_black, Color::Black).unwrap();
        let decoded_white = trait_decode(&m, idx_white, Color::White).unwrap();

        assert_eq!(decoded_black, mv);
        assert_eq!(decoded_white, mv);
    }

    #[test]
    fn test_decode_out_of_range() {
        let m = mapper();
        assert!(trait_decode(&m, SPATIAL_ACTION_SPACE_SIZE, Color::Black).is_err());
        assert!(trait_decode(&m, usize::MAX, Color::Black).is_err());
    }

    #[test]
    fn test_no_index_collisions_drops() {
        let m = mapper();
        let mut seen = HashSet::new();
        for to_idx in 0u8..81 {
            for &pt in &HandPieceType::ALL {
                let mv = Move::Drop { to: Square::new_unchecked(to_idx), piece_type: pt };
                let idx = trait_encode(&m, mv, Color::Black);
                assert!(seen.insert(idx), "Collision at index {} for drop to={} piece={:?}", idx, to_idx, pt);
            }
        }
        assert_eq!(seen.len(), 81 * 7); // 567 drop actions
    }

    /// Round-trip all legal moves from the starting position.
    #[test]
    fn test_startpos_legal_moves_roundtrip() {
        let m = mapper();
        let state = shogi_core::GameState::new();
        let legal_moves = state.legal_moves();

        assert!(!legal_moves.is_empty(), "Startpos should have legal moves");

        for mv in legal_moves.iter() {
            let idx = trait_encode(&m, *mv, Color::Black);
            assert!(idx < SPATIAL_ACTION_SPACE_SIZE, "Index {} out of range", idx);
            let decoded = trait_decode(&m, idx, Color::Black)
                .unwrap_or_else(|e| panic!("Failed to decode move {:?}: {}", mv, e));
            assert_eq!(
                decoded, *mv,
                "Startpos roundtrip failed for move {:?} (encoded as {})",
                mv, idx
            );
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cd shogi-engine && cargo test -p shogi-gym spatial_action_mapper::tests`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/spatial_action_mapper.rs
git commit -m "test(shogi-gym): comprehensive SpatialActionMapper unit tests"
```

---

### Task 7: VecEnv Mode Parameters

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/vec_env.rs`

Add `observation_mode` and `action_mode` parameters to the VecEnv constructor so it can use either the default or KataGo generators.

- [ ] **Step 1: Make VecEnv generic over observation/action modes**

The VecEnv currently stores `mapper: DefaultActionMapper` and `obs_gen: DefaultObservationGenerator` as concrete types. Change it to use trait objects or an enum dispatch.

Since the mapper and obs_gen are stateless and lightweight, use an enum-based dispatch to avoid trait object overhead:

In `shogi-engine/crates/shogi-gym/src/vec_env.rs`, replace the imports and struct fields:

Add new imports at the top:

```rust
use crate::katago_observation::{
    KataGoObservationGenerator, KATAGO_BUFFER_LEN, KATAGO_NUM_CHANNELS,
};
use crate::spatial_action_mapper::{SpatialActionMapper, SPATIAL_ACTION_SPACE_SIZE, SPATIAL_MOVE_TYPES};
```

Add enum types before the `VecEnv` struct:

```rust
enum ObsMode {
    Default(DefaultObservationGenerator),
    KataGo(KataGoObservationGenerator),
}

impl ObsMode {
    fn channels(&self) -> usize {
        match self {
            ObsMode::Default(_) => NUM_CHANNELS,
            ObsMode::KataGo(_) => KATAGO_NUM_CHANNELS,
        }
    }

    fn buffer_len(&self) -> usize {
        match self {
            ObsMode::Default(_) => BUFFER_LEN,
            ObsMode::KataGo(_) => KATAGO_BUFFER_LEN,
        }
    }

    fn generate(&self, state: &GameState, perspective: Color, buffer: &mut [f32]) {
        match self {
            ObsMode::Default(g) => g.generate(state, perspective, buffer),
            ObsMode::KataGo(g) => g.generate(state, perspective, buffer),
        }
    }
}

enum ActionMode {
    Default(DefaultActionMapper),
    Spatial(SpatialActionMapper),
}

impl ActionMode {
    fn action_space_size(&self) -> usize {
        match self {
            ActionMode::Default(_) => ACTION_SPACE_SIZE,
            ActionMode::Spatial(_) => SPATIAL_ACTION_SPACE_SIZE,
        }
    }

    fn encode(&self, mv: Move, perspective: Color) -> usize {
        match self {
            ActionMode::Default(m) => <DefaultActionMapper as ActionMapper>::encode(m, mv, perspective),
            ActionMode::Spatial(m) => <SpatialActionMapper as ActionMapper>::encode(m, mv, perspective),
        }
    }

    fn decode(&self, idx: usize, perspective: Color) -> Result<Move, String> {
        match self {
            ActionMode::Default(m) => <DefaultActionMapper as ActionMapper>::decode(m, idx, perspective),
            ActionMode::Spatial(m) => <SpatialActionMapper as ActionMapper>::decode(m, idx, perspective),
        }
    }
}
```

Replace the `mapper` and `obs_gen` fields in the `VecEnv` struct:

```rust
mapper: ActionMode,
obs_gen: ObsMode,
obs_buffer_len: usize,    // cached: obs_mode.buffer_len()
action_space: usize,       // cached: action_mode.action_space_size()
num_channels: usize,       // cached: obs_mode.channels()
```

- [ ] **Step 2: Update the constructor to accept mode parameters**

Replace the `#[new]` method:

```rust
#[new]
#[pyo3(signature = (num_envs = 512, max_ply = 500, observation_mode = "default", action_mode = "default"))]
pub fn new(num_envs: usize, max_ply: u32, observation_mode: &str, action_mode: &str) -> PyResult<Self> {
    let obs_mode = match observation_mode {
        "default" => ObsMode::Default(DefaultObservationGenerator::new()),
        "katago" => ObsMode::KataGo(KataGoObservationGenerator::new()),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown observation_mode '{}'. Valid: 'default', 'katago'", observation_mode)
        )),
    };

    let action_mode_enum = match action_mode {
        "default" => ActionMode::Default(DefaultActionMapper),
        "spatial" => ActionMode::Spatial(SpatialActionMapper),
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unknown action_mode '{}'. Valid: 'default', 'spatial'", action_mode)
        )),
    };

    let obs_buf_len = obs_mode.buffer_len();
    let act_space = action_mode_enum.action_space_size();
    let channels = obs_mode.channels();

    let games: Vec<GameState> = (0..num_envs)
        .map(|_| GameState::with_max_ply(max_ply))
        .collect();

    Ok(VecEnv {
        games,
        num_envs,
        max_ply,
        obs_buffer: vec![0.0; num_envs * obs_buf_len],
        legal_mask_buffer: vec![false; num_envs * act_space],
        reward_buffer: vec![0.0; num_envs],
        terminated_buffer: vec![false; num_envs],
        truncated_buffer: vec![false; num_envs],
        captured_buffer: vec![255; num_envs],
        term_reason_buffer: vec![0; num_envs],
        ply_buffer: vec![0; num_envs],
        terminal_obs_buffer: vec![0.0; num_envs * obs_buf_len],
        current_players_buffer: vec![0; num_envs],
        mapper: action_mode_enum,
        obs_gen: obs_mode,
        obs_buffer_len: obs_buf_len,
        action_space: act_space,
        num_channels: channels,
        episodes_completed: AtomicU64::new(0),
        episodes_drawn: AtomicU64::new(0),
        episodes_truncated: AtomicU64::new(0),
        total_episode_ply: AtomicU64::new(0),
    })
}
```

- [ ] **Step 3: Update all methods that reference buffer sizes**

Replace all occurrences of `BUFFER_LEN` with `self.obs_buffer_len`, `ACTION_SPACE_SIZE` with `self.action_space`, and `NUM_CHANNELS` with `self.num_channels` throughout `vec_env.rs`.

In `write_obs_and_mask`:
```rust
fn write_obs_and_mask(&mut self, i: usize) {
    let perspective = self.games[i].position.current_player;

    let obs_start = i * self.obs_buffer_len;
    let obs_slice = &mut self.obs_buffer[obs_start..obs_start + self.obs_buffer_len];
    self.obs_gen.generate(&self.games[i], perspective, obs_slice);

    let mask_start = i * self.action_space;
    let mask_slice = &mut self.legal_mask_buffer[mask_start..mask_start + self.action_space];
    mask_slice.fill(false);
    let mut move_list = MoveList::new();
    self.games[i].generate_legal_moves_into(&mut move_list);
    for mv in move_list.iter() {
        let idx = self.mapper.encode(*mv, perspective);
        mask_slice[idx] = true;
    }
}
```

In `reset`, update the reshape calls:
```rust
let obs_4d = obs_array
    .reshape([self.num_envs, self.num_channels, 9, 9])
    // ...

let mask_2d = mask_array
    .reshape([self.num_envs, self.action_space])
    // ...
```

In `step`, update all references similarly. **Critical borrow checker note:** The existing `step` method uses a rayon parallel closure (`process_env`) that can't borrow `self` mutably while also reading `self.obs_buffer_len`. The existing code uses module-level constants (`BUFFER_LEN`, `ACTION_SPACE_SIZE`) which are implicitly `Copy`. After migrating to struct fields, capture them as local variables before the closure:

```rust
let obs_buf_len = self.obs_buffer_len;
let act_space = self.action_space;
let num_ch = self.num_channels;
// Use these locals inside process_env instead of self.obs_buffer_len, etc.
```

For the mapper and obs_gen, **do NOT clone or move the enum wrappers** — they may not implement `Send` and rayon's `.par_iter()` will reject them. Instead, capture a plain tag enum and reconstruct the zero-size generators inside the closure:

```rust
#[derive(Copy, Clone)]
enum ObsModeTag { Default, KataGo }
#[derive(Copy, Clone)]
enum ActionModeTag { Default, Spatial }

// Before the closure:
let obs_tag = match &self.obs_gen {
    ObsMode::Default(_) => ObsModeTag::Default,
    ObsMode::KataGo(_) => ObsModeTag::KataGo,
};
let act_tag = match &self.mapper {
    ActionMode::Default(_) => ActionModeTag::Default,
    ActionMode::Spatial(_) => ActionModeTag::Spatial,
};

// Inside the parallel closure — no Box, no allocation, pure static dispatch:
match obs_tag {
    ObsModeTag::Default => DefaultObservationGenerator::new()
        .generate(state, perspective, obs_slice),
    ObsModeTag::KataGo  => KataGoObservationGenerator::new()
        .generate(state, perspective, obs_slice),
}

match act_tag {
    ActionModeTag::Default  => DefaultActionMapper.encode(mv, perspective),
    ActionModeTag::Spatial  => SpatialActionMapper.encode(mv, perspective),
}
```

This sidesteps `Send` bound issues entirely. Since all four generators/mappers are zero-size structs, `DefaultObservationGenerator::new()` compiles to nothing — the match arm is just a static dispatch to the concrete impl. No allocation, no vtable, the compiler inlines it. Do NOT use `Box<dyn Trait>` here — it would add heap allocation and vtable indirection on every call for no reason.

- [ ] **Step 4: Update property getters**

```rust
#[getter]
pub fn observation_channels(&self) -> usize {
    self.num_channels
}

#[getter]
pub fn action_space_size(&self) -> usize {
    self.action_space
}
```

- [ ] **Step 5: Build and run existing tests**

Run: `cd shogi-engine && cargo build && cargo test -p shogi-gym`
Expected: All existing tests pass (they use default modes).

- [ ] **Step 6: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/vec_env.rs
git commit -m "feat(shogi-gym): add observation_mode and action_mode params to VecEnv"
```

---

### Task 8: Python Integration Tests — KataGo Observation

**Files:**
- Create: `tests/test_katago_observation.py`

- [ ] **Step 1: Write Python integration tests**

```python
# tests/test_katago_observation.py
"""Integration tests for the KataGo 50-channel observation generator via VecEnv."""

import numpy as np
import pytest

from shogi_gym import VecEnv


@pytest.fixture
def katago_env():
    """Single-env VecEnv with KataGo observation mode."""
    return VecEnv(num_envs=1, max_ply=100, observation_mode="katago", action_mode="default")


@pytest.fixture
def default_env():
    """Single-env VecEnv with default observation mode."""
    return VecEnv(num_envs=1, max_ply=100)


class TestKataGoObservationShape:
    def test_observation_channels(self, katago_env):
        assert katago_env.observation_channels == 50

    def test_reset_observation_shape(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)
        assert obs.shape == (1, 50, 9, 9)

    def test_step_observation_shape(self, katago_env):
        result = katago_env.reset()
        masks = np.array(result.legal_masks)
        # Pick first legal action
        action = int(np.argmax(masks[0]))
        step_result = katago_env.step([action])
        obs = np.array(step_result.observations)
        assert obs.shape == (1, 50, 9, 9)


class TestKataGoObservationContent:
    def test_channels_0_43_match_default(self, katago_env, default_env):
        """First 44 channels should match the default generator."""
        katago_result = katago_env.reset()
        default_result = default_env.reset()

        katago_obs = np.array(katago_result.observations)[0]  # (50, 9, 9)
        default_obs = np.array(default_result.observations)[0]  # (46, 9, 9)

        np.testing.assert_array_equal(
            katago_obs[:44], default_obs[:44],
            err_msg="Channels 0-43 should be identical between KataGo and default"
        )

    def test_no_repetition_at_startpos(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)[0]
        # Channels 44-47 should all be zero at startpos
        for ch in range(44, 48):
            assert np.all(obs[ch] == 0.0), f"Channel {ch} should be all zeros at startpos"

    def test_not_in_check_at_startpos(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)[0]
        assert np.all(obs[48] == 0.0), "Channel 48 should be 0 when not in check"

    def test_reserved_channel_49_zero(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)[0]
        assert np.all(obs[49] == 0.0), "Channel 49 (reserved) should be all zeros"

    def test_no_nan_in_observation(self, katago_env):
        result = katago_env.reset()
        obs = np.array(result.observations)[0]
        assert not np.any(np.isnan(obs)), "No NaN should be present in observation"


class TestKataGoObservationInvalidMode:
    def test_invalid_observation_mode(self):
        with pytest.raises(ValueError, match="Unknown observation_mode"):
            VecEnv(num_envs=1, max_ply=100, observation_mode="invalid")
```

- [ ] **Step 2: Run tests**

Run: `cd shogi-engine && maturin develop --release && cd .. && uv run pytest tests/test_katago_observation.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_katago_observation.py
git commit -m "test: Python integration tests for KataGo observation generator"
```

---

### Task 9: Python Integration Tests — Spatial Action Mapper

**Files:**
- Create: `tests/test_spatial_action_mapper.py`

- [ ] **Step 1: Write Python integration tests**

```python
# tests/test_spatial_action_mapper.py
"""Integration tests for the SpatialActionMapper via VecEnv and direct Python bindings."""

import numpy as np
import pytest

from shogi_gym import SpatialActionMapper, VecEnv


@pytest.fixture
def spatial_env():
    """Single-env VecEnv with spatial action mode."""
    return VecEnv(num_envs=1, max_ply=100, observation_mode="default", action_mode="spatial")


@pytest.fixture
def mapper():
    return SpatialActionMapper()


class TestSpatialActionMapperBasic:
    def test_action_space_size(self, mapper):
        assert mapper.action_space_size == 11259

    def test_action_space_on_env(self, spatial_env):
        assert spatial_env.action_space_size == 11259

    def test_legal_mask_shape(self, spatial_env):
        result = spatial_env.reset()
        masks = np.array(result.legal_masks)
        assert masks.shape == (1, 11259)

    def test_legal_mask_has_legal_moves(self, spatial_env):
        result = spatial_env.reset()
        masks = np.array(result.legal_masks)
        assert masks[0].sum() > 0, "Startpos should have legal moves"


class TestSpatialFlatIndexContract:
    def test_flat_index_is_square_times_139_plus_slot(self, mapper):
        """Verify: flat_index = square * 139 + move_type"""
        # Encode a drop (P*e5, sq 40, piece 0=Pawn)
        idx = mapper.encode_drop_move(40, 0, False)
        expected = 40 * 139 + 132  # slot 132 = first drop piece type
        assert idx == expected, f"Expected {expected}, got {idx}"


class TestSpatialRoundTrip:
    def test_board_move_roundtrip(self, mapper):
        """A simple board move: from sq 40 to sq 31 (one step N from e5 to e4)."""
        idx = mapper.encode_board_move(40, 31, False, False)
        decoded = mapper.decode(idx, False)
        assert decoded["type"] == "board"
        assert decoded["from_sq"] == 40
        assert decoded["to_sq"] == 31
        assert decoded["promote"] is False

    def test_drop_move_roundtrip(self, mapper):
        """Drop a pawn at e5."""
        idx = mapper.encode_drop_move(40, 0, False)
        decoded = mapper.decode(idx, False)
        assert decoded["type"] == "drop"
        assert decoded["to_sq"] == 40
        assert decoded["piece_type_idx"] == 0

    def test_white_perspective_roundtrip(self, mapper):
        """Verify white perspective flipping works for board moves."""
        idx = mapper.encode_board_move(40, 31, False, True)
        decoded = mapper.decode(idx, True)
        assert decoded["from_sq"] == 40
        assert decoded["to_sq"] == 31


class TestSpatialStepExecution:
    def test_step_with_spatial_action(self, spatial_env):
        """Execute a full reset-step cycle with spatial action encoding."""
        result = spatial_env.reset()
        masks = np.array(result.legal_masks)
        action = int(np.argmax(masks[0]))  # pick first legal action
        step_result = spatial_env.step([action])

        obs = np.array(step_result.observations)
        assert obs.shape[0] == 1  # one env
        assert len(step_result.rewards) == 1

    def test_multi_step_stability(self, spatial_env):
        """Run 20 steps without crashing."""
        result = spatial_env.reset()
        for _ in range(20):
            masks = np.array(result.legal_masks) if hasattr(result, 'legal_masks') else np.array(spatial_env.reset().legal_masks)
            step_result = spatial_env.step([int(np.argmax(masks[0]))])
            masks = np.array(step_result.legal_masks)
            assert masks[0].sum() > 0 or any(step_result.terminated) or any(step_result.truncated)
            result = step_result


class TestSpatialInvalidMode:
    def test_invalid_action_mode(self):
        with pytest.raises(ValueError, match="Unknown action_mode"):
            VecEnv(num_envs=1, max_ply=100, action_mode="invalid")


class TestKataGoFullConfig:
    """Test VecEnv with both KataGo observation and spatial action modes."""

    def test_full_katago_config(self):
        env = VecEnv(
            num_envs=2,
            max_ply=50,
            observation_mode="katago",
            action_mode="spatial",
        )
        assert env.observation_channels == 50
        assert env.action_space_size == 11259

        result = env.reset()
        obs = np.array(result.observations)
        masks = np.array(result.legal_masks)
        assert obs.shape == (2, 50, 9, 9)
        assert masks.shape == (2, 11259)

        # Step
        actions = [int(np.argmax(masks[i])) for i in range(2)]
        step_result = env.step(actions)
        assert len(step_result.rewards) == 2
```

- [ ] **Step 2: Run tests**

Run: `cd shogi-engine && maturin develop --release && cd .. && uv run pytest tests/test_spatial_action_mapper.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_spatial_action_mapper.py
git commit -m "test: Python integration tests for SpatialActionMapper and VecEnv modes"
```

---

### Task 10: Run Full Test Suite

**Files:** None (verification only)

- [ ] **Step 1: Run all Rust tests**

Run: `cd shogi-engine && cargo test`
Expected: All tests PASS, including existing DefaultActionMapper and DefaultObservationGenerator tests (no regressions).

- [ ] **Step 2: Build Python bindings**

Run: `cd shogi-engine && maturin develop --release`
Expected: Build succeeds.

- [ ] **Step 3: Run all Python tests**

Run: `uv run pytest -v`
Expected: All tests PASS, including existing tests that use the default VecEnv constructor.

- [ ] **Step 4: Verify backward compatibility**

The default VecEnv constructor `VecEnv(num_envs=128, max_ply=500)` should work identically to before (defaults to `observation_mode="default"`, `action_mode="default"`).

Run: `uv run python -c "from shogi_gym import VecEnv; e = VecEnv(num_envs=2, max_ply=100); r = e.reset(); print('obs shape:', r.observations[0].shape, 'mask shape:', r.legal_masks[0].shape)"`
Expected: `obs shape: (46, 9, 9) mask shape: (13527,)`

- [ ] **Step 5: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: address any issues found in full test suite run"
```
