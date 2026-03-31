# shogi-gym Python Bindings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `shogi-gym`, a Rust crate providing vectorized RL training environments (VecEnv, SpectatorEnv) with PyO3/maturin Python bindings on top of `shogi-core`.

**Architecture:** Trait-based ActionMapper and ObservationGenerator produce pre-allocated NumPy tensors. VecEnv batch-steps N games in one FFI call using rayon parallelism with GIL release. Two-phase step contract (validate all, then apply all) prevents partial mutation on bad actions. SpectatorEnv wraps a single game for display with rich Python dict output.

**Tech Stack:** Rust (shogi-core, pyo3 0.23+, numpy 0.23+, rayon 1.10), maturin build, Python (pytest, numpy)

**Spec:** `docs/superpowers/specs/2026-03-31-rust-shogi-engine-design.md` (shogi-gym sections)

**Plan scope:** This is Plan 2 of 3. Plan 1 (shogi-core) is complete. Plan 3 (burn-in harness) follows.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `shogi-engine/Cargo.toml` | Modify | Add `crates/shogi-gym` to workspace members |
| `shogi-engine/crates/shogi-gym/Cargo.toml` | Create | Crate manifest (pyo3, numpy, rayon, shogi-core) |
| `shogi-engine/crates/shogi-gym/pyproject.toml` | Create | Maturin build config |
| `shogi-engine/crates/shogi-gym/src/lib.rs` | Create | PyO3 module definition, module registration |
| `shogi-engine/crates/shogi-gym/src/action_mapper.rs` | Create | ActionMapper trait + DefaultActionMapper (13,527 actions) |
| `shogi-engine/crates/shogi-gym/src/observation.rs` | Create | ObservationGenerator trait + DefaultObservationGenerator (46 channels) |
| `shogi-engine/crates/shogi-gym/src/vec_env.rs` | Create | VecEnv: batch N-game stepping with rayon, pre-allocated buffers |
| `shogi-engine/crates/shogi-gym/src/spectator.rs` | Create | SpectatorEnv: single-game wrapper with rich dict output |
| `shogi-engine/crates/shogi-gym/src/step_result.rs` | Create | StepResult struct holding PyArray references for VecEnv output |
| `shogi-engine/python/shogi_gym/__init__.py` | Create | Python re-exports from native module |
| `shogi-engine/python/shogi_gym/py.typed` | Create | PEP 561 type stub marker |
| `shogi-engine/crates/shogi-gym/tests/test_action_mapper.py` | Create | Python tests for action encoding/decoding |
| `shogi-engine/crates/shogi-gym/tests/test_observation.py` | Create | Python tests for observation tensor generation |
| `shogi-engine/crates/shogi-gym/tests/test_vec_env.py` | Create | Python tests for VecEnv stepping lifecycle |
| `shogi-engine/crates/shogi-gym/tests/test_spectator.py` | Create | Python tests for SpectatorEnv lifecycle |

---

### Task 1: Project Scaffold

**Files:**
- Modify: `shogi-engine/Cargo.toml`
- Create: `shogi-engine/crates/shogi-gym/Cargo.toml`
- Create: `shogi-engine/crates/shogi-gym/pyproject.toml`
- Create: `shogi-engine/crates/shogi-gym/src/lib.rs`
- Create: `shogi-engine/python/shogi_gym/__init__.py`
- Create: `shogi-engine/python/shogi_gym/py.typed`

- [ ] **Step 1: Add shogi-gym to workspace members**

Edit `shogi-engine/Cargo.toml`:

```toml
[workspace]
resolver = "2"
members = [
    "crates/shogi-core",
    "crates/shogi-gym",
]
```

- [ ] **Step 2: Create shogi-gym Cargo.toml**

```bash
mkdir -p shogi-engine/crates/shogi-gym/src
```

Write `shogi-engine/crates/shogi-gym/Cargo.toml`:

```toml
[package]
name = "shogi-gym"
version = "0.1.0"
edition = "2024"
description = "RL training environment for Shogi with PyO3 bindings"
license = "MIT"

[lib]
name = "shogi_gym"
crate-type = ["cdylib"]

[dependencies]
shogi-core = { path = "../shogi-core" }
pyo3 = { version = "0.23", features = ["extension-module"] }
numpy = "0.23"
rayon = "1.10"
```

- [ ] **Step 3: Create pyproject.toml for maturin**

Write `shogi-engine/crates/shogi-gym/pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.7,<2.0"]
build-backend = "maturin"

[project]
name = "shogi-gym"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy>=1.24"]

[tool.maturin]
manifest-path = "Cargo.toml"
python-source = "../../python"
module-name = "shogi_gym._native"
features = ["pyo3/extension-module"]
```

- [ ] **Step 4: Create lib.rs with module skeleton**

Write `shogi-engine/crates/shogi-gym/src/lib.rs`:

```rust
use pyo3::prelude::*;

mod action_mapper;
mod observation;
mod step_result;
mod vec_env;
mod spectator;

/// Native module for shogi-gym RL environments.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<action_mapper::DefaultActionMapper>()?;
    m.add_class::<observation::DefaultObservationGenerator>()?;
    m.add_class::<vec_env::VecEnv>()?;
    m.add_class::<spectator::SpectatorEnv>()?;
    m.add_class::<step_result::StepResult>()?;
    Ok(())
}
```

- [ ] **Step 5: Create stub source files so crate compiles**

Write `shogi-engine/crates/shogi-gym/src/action_mapper.rs`:

```rust
// ActionMapper trait + DefaultActionMapper — implemented in Task 2
```

Write `shogi-engine/crates/shogi-gym/src/observation.rs`:

```rust
// ObservationGenerator trait + DefaultObservationGenerator — implemented in Task 3
```

Write `shogi-engine/crates/shogi-gym/src/step_result.rs`:

```rust
// StepResult struct — implemented in Task 4
```

Write `shogi-engine/crates/shogi-gym/src/vec_env.rs`:

```rust
// VecEnv — implemented in Task 5
```

Write `shogi-engine/crates/shogi-gym/src/spectator.rs`:

```rust
// SpectatorEnv — implemented in Task 6
```

- [ ] **Step 6: Create Python package files**

```bash
mkdir -p shogi-engine/python/shogi_gym
```

Write `shogi-engine/python/shogi_gym/__init__.py`:

```python
"""shogi-gym: Rust-powered RL environments for Shogi."""

from shogi_gym._native import (
    DefaultActionMapper,
    DefaultObservationGenerator,
    VecEnv,
    SpectatorEnv,
    StepResult,
)

__all__ = [
    "DefaultActionMapper",
    "DefaultObservationGenerator",
    "VecEnv",
    "SpectatorEnv",
    "StepResult",
]
```

Write `shogi-engine/python/shogi_gym/py.typed` (empty file):

```
```

- [ ] **Step 7: Verify workspace compiles**

Run: `cd shogi-engine && cargo check --workspace`
Expected: compiles with warnings about empty files (no errors)

- [ ] **Step 8: Commit**

```bash
git add shogi-engine/Cargo.toml shogi-engine/crates/shogi-gym/ shogi-engine/python/
git commit -m "feat(shogi-gym): project scaffold with pyo3/maturin config"
```

---

### Task 2: ActionMapper Trait + DefaultActionMapper

**Files:**
- Create: `shogi-engine/crates/shogi-gym/src/action_mapper.rs`
- Create: `shogi-engine/crates/shogi-gym/tests/test_action_mapper.py`

- [ ] **Step 1: Write Rust unit tests for DefaultActionMapper**

Write the test module at the bottom of `shogi-engine/crates/shogi-gym/src/action_mapper.rs`:

```rust
use shogi_core::{Color, GameState, HandPieceType, Move, Square};

/// Trait for encoding/decoding moves to/from action indices.
/// Implementations must be Send + Sync for rayon parallelism.
pub trait ActionMapper: Send + Sync {
    fn encode(&self, mv: Move, perspective: Color) -> usize;
    fn decode(&self, idx: usize, perspective: Color) -> Result<Move, String>;
    fn action_space_size(&self) -> usize;
}

/// 13,527-action encoding matching Keisei's Python PolicyOutputMapper.
///
/// Index layout:
/// - Board moves [0..12960): 81 sources × 80 destinations × 2 (promote flag)
///   Ordered by: from_sq ascending, then to_sq ascending (skipping from==to),
///   then promote=false before promote=true.
/// - Drop moves [12960..13527): 81 destinations × 7 hand piece types
///   Ordered by: to_sq ascending, then piece type (Pawn..Rook).
///
/// Perspective flipping: when perspective is White, squares are flipped
/// (sq → 80 - sq) before encoding/after decoding so the neural network
/// always sees from the current player's viewpoint.
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
pub struct DefaultActionMapper;

impl DefaultActionMapper {
    pub const ACTION_SPACE_SIZE: usize = 13_527;
    const BOARD_MOVE_COUNT: usize = 12_960; // 81 * 80 * 2
    const DROP_BASE: usize = 12_960;

    pub fn new() -> Self {
        DefaultActionMapper
    }

    /// Flip a square for perspective if needed.
    #[inline]
    fn perspective_sq(sq: Square, perspective: Color) -> Square {
        match perspective {
            Color::Black => sq,
            Color::White => sq.flip(),
        }
    }

    /// Given from and to square indices (0..81), compute the destination
    /// offset that skips the null move (from == to).
    /// If to > from: offset = to - 1 (skip the slot where to == from)
    /// If to < from: offset = to (no skipping needed yet)
    #[inline]
    fn dest_offset(from_idx: usize, to_idx: usize) -> usize {
        if to_idx > from_idx {
            to_idx - 1
        } else {
            to_idx
        }
    }
}

impl ActionMapper for DefaultActionMapper {
    fn encode(&self, mv: Move, perspective: Color) -> usize {
        match mv {
            Move::Board { from, to, promote } => {
                let from_sq = Self::perspective_sq(from, perspective);
                let to_sq = Self::perspective_sq(to, perspective);
                let from_idx = from_sq.index();
                let to_idx = to_sq.index();
                debug_assert_ne!(from_idx, to_idx, "null move in encode");
                let dest_off = Self::dest_offset(from_idx, to_idx);
                let base = from_idx * 80 * 2 + dest_off * 2;
                if promote { base + 1 } else { base }
            }
            Move::Drop { to, piece_type } => {
                let to_sq = Self::perspective_sq(to, perspective);
                Self::DROP_BASE + to_sq.index() * HandPieceType::COUNT + piece_type.index()
            }
        }
    }

    fn decode(&self, idx: usize, perspective: Color) -> Result<Move, String> {
        if idx >= Self::ACTION_SPACE_SIZE {
            return Err(format!("action index {} out of range (max {})", idx, Self::ACTION_SPACE_SIZE - 1));
        }

        if idx < Self::BOARD_MOVE_COUNT {
            // Board move: idx = from_idx * 160 + dest_off * 2 + promote_bit
            let from_idx = idx / 160;
            let remainder = idx % 160;
            let dest_off = remainder / 2;
            let promote = remainder % 2 == 1;

            // Reverse the dest_offset: if dest_off >= from_idx, actual to = dest_off + 1
            let to_idx = if dest_off >= from_idx {
                dest_off + 1
            } else {
                dest_off
            };

            let from_sq = Self::perspective_sq(
                Square::new_unchecked(from_idx as u8), perspective
            );
            let to_sq = Self::perspective_sq(
                Square::new_unchecked(to_idx as u8), perspective
            );

            Ok(Move::Board { from: from_sq, to: to_sq, promote })
        } else {
            // Drop move: idx - DROP_BASE = to_idx * 7 + piece_type_index
            let drop_idx = idx - Self::DROP_BASE;
            let to_idx = drop_idx / HandPieceType::COUNT;
            let piece_idx = drop_idx % HandPieceType::COUNT;

            let to_sq = Self::perspective_sq(
                Square::new_unchecked(to_idx as u8), perspective
            );
            let piece_type = HandPieceType::ALL[piece_idx];

            Ok(Move::Drop { to: to_sq, piece_type })
        }
    }

    fn action_space_size(&self) -> usize {
        Self::ACTION_SPACE_SIZE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_space_size() {
        let mapper = DefaultActionMapper::new();
        assert_eq!(mapper.action_space_size(), 13_527);
    }

    #[test]
    fn test_board_move_encode_decode_roundtrip_black() {
        let mapper = DefaultActionMapper::new();
        let perspective = Color::Black;

        // Test a few representative board moves
        let cases = vec![
            // (from, to, promote)
            (0, 1, false),
            (0, 1, true),
            (0, 80, false),
            (40, 0, false),
            (40, 41, true),
            (80, 0, false),
            (80, 79, true),
        ];

        for (from_idx, to_idx, promote) in cases {
            let mv = Move::Board {
                from: Square::new_unchecked(from_idx),
                to: Square::new_unchecked(to_idx),
                promote,
            };
            let encoded = mapper.encode(mv, perspective);
            assert!(encoded < 12_960, "board move index out of range: {}", encoded);
            let decoded = mapper.decode(encoded, perspective).unwrap();
            assert_eq!(decoded, mv, "roundtrip failed for from={}, to={}, promote={}", from_idx, to_idx, promote);
        }
    }

    #[test]
    fn test_drop_move_encode_decode_roundtrip_black() {
        let mapper = DefaultActionMapper::new();
        let perspective = Color::Black;

        for &hpt in &HandPieceType::ALL {
            for sq_idx in [0u8, 40, 80] {
                let mv = Move::Drop {
                    to: Square::new_unchecked(sq_idx),
                    piece_type: hpt,
                };
                let encoded = mapper.encode(mv, perspective);
                assert!(encoded >= 12_960 && encoded < 13_527,
                    "drop index {} out of range", encoded);
                let decoded = mapper.decode(encoded, perspective).unwrap();
                assert_eq!(decoded, mv, "drop roundtrip failed for sq={}, type={:?}", sq_idx, hpt);
            }
        }
    }

    #[test]
    fn test_exhaustive_board_move_roundtrip() {
        let mapper = DefaultActionMapper::new();
        let perspective = Color::Black;
        let mut seen = vec![false; 12_960];

        for from_idx in 0u8..81 {
            for to_idx in 0u8..81 {
                if from_idx == to_idx { continue; }
                for promote in [false, true] {
                    let mv = Move::Board {
                        from: Square::new_unchecked(from_idx),
                        to: Square::new_unchecked(to_idx),
                        promote,
                    };
                    let encoded = mapper.encode(mv, perspective);
                    assert!(!seen[encoded], "collision at index {}", encoded);
                    seen[encoded] = true;
                    let decoded = mapper.decode(encoded, perspective).unwrap();
                    assert_eq!(decoded, mv);
                }
            }
        }
        assert!(seen.iter().all(|&v| v), "not all board move indices covered");
    }

    #[test]
    fn test_exhaustive_drop_move_roundtrip() {
        let mapper = DefaultActionMapper::new();
        let perspective = Color::Black;
        let mut seen = vec![false; 567];

        for to_idx in 0u8..81 {
            for &hpt in &HandPieceType::ALL {
                let mv = Move::Drop {
                    to: Square::new_unchecked(to_idx),
                    piece_type: hpt,
                };
                let encoded = mapper.encode(mv, perspective);
                let local = encoded - 12_960;
                assert!(!seen[local], "collision at drop index {}", local);
                seen[local] = true;
                let decoded = mapper.decode(encoded, perspective).unwrap();
                assert_eq!(decoded, mv);
            }
        }
        assert!(seen.iter().all(|&v| v), "not all drop indices covered");
    }

    #[test]
    fn test_perspective_flip_board_move() {
        let mapper = DefaultActionMapper::new();
        // A board move (0,0)->(8,8) for Black should encode the same as
        // (8,8)->(0,0) for White (since flip(0) = 80 and flip(80) = 0).
        let mv_abs = Move::Board {
            from: Square::new_unchecked(0),
            to: Square::new_unchecked(80),
            promote: false,
        };
        let idx_black = mapper.encode(mv_abs, Color::Black);

        // For White perspective, the move from absolute (80,80) to (0,0)
        // should encode to the same index (both get flipped).
        let mv_white_abs = Move::Board {
            from: Square::new_unchecked(80),
            to: Square::new_unchecked(0),
            promote: false,
        };
        let idx_white = mapper.encode(mv_white_abs, Color::White);
        assert_eq!(idx_black, idx_white);
    }

    #[test]
    fn test_perspective_flip_drop_move() {
        let mapper = DefaultActionMapper::new();
        // Drop at sq 0 for Black = drop at sq 80 for White
        let mv_black = Move::Drop {
            to: Square::new_unchecked(0),
            piece_type: HandPieceType::Pawn,
        };
        let mv_white = Move::Drop {
            to: Square::new_unchecked(80),
            piece_type: HandPieceType::Pawn,
        };
        assert_eq!(
            mapper.encode(mv_black, Color::Black),
            mapper.encode(mv_white, Color::White),
        );
    }

    #[test]
    fn test_decode_out_of_range() {
        let mapper = DefaultActionMapper::new();
        assert!(mapper.decode(13_527, Color::Black).is_err());
        assert!(mapper.decode(99_999, Color::Black).is_err());
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd shogi-engine && cargo test -p shogi-gym`
Expected: All tests pass

- [ ] **Step 3: Add PyO3 wrapper methods**

Add to the bottom of `action_mapper.rs`, above the `#[cfg(test)]` module:

```rust
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[pymethods]
impl DefaultActionMapper {
    #[new]
    fn py_new() -> Self {
        Self::new()
    }

    /// Encode a board move to an action index.
    /// from_sq and to_sq are board indices 0-80.
    /// is_white: true if current player is White (perspective flip).
    #[pyo3(name = "encode_board_move")]
    fn py_encode_board_move(&self, from_sq: u8, to_sq: u8, promote: bool, is_white: bool) -> PyResult<usize> {
        let perspective = if is_white { Color::White } else { Color::Black };
        let from = Square::new(from_sq).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let to = Square::new(to_sq).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let mv = Move::Board { from, to, promote };
        Ok(self.encode(mv, perspective))
    }

    /// Encode a drop move to an action index.
    /// to_sq is board index 0-80. piece_type_idx is 0-6 (Pawn..Rook).
    #[pyo3(name = "encode_drop_move")]
    fn py_encode_drop_move(&self, to_sq: u8, piece_type_idx: usize, is_white: bool) -> PyResult<usize> {
        let perspective = if is_white { Color::White } else { Color::Black };
        let to = Square::new(to_sq).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        if piece_type_idx >= HandPieceType::COUNT {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("piece_type_idx {} out of range (max 6)", piece_type_idx),
            ));
        }
        let piece_type = HandPieceType::ALL[piece_type_idx];
        let mv = Move::Drop { to, piece_type };
        Ok(self.encode(mv, perspective))
    }

    /// Decode an action index back to move components.
    /// Returns a dict with keys: "type" ("board" or "drop"), "from_sq", "to_sq",
    /// "promote", "piece_type_idx".
    #[pyo3(name = "decode")]
    fn py_decode(&self, idx: usize, is_white: bool) -> PyResult<pyo3::Py<pyo3::types::PyDict>> {
        let perspective = if is_white { Color::White } else { Color::Black };
        let mv = self.decode(idx, perspective)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            match mv {
                Move::Board { from, to, promote } => {
                    dict.set_item("type", "board")?;
                    dict.set_item("from_sq", from.index())?;
                    dict.set_item("to_sq", to.index())?;
                    dict.set_item("promote", promote)?;
                }
                Move::Drop { to, piece_type } => {
                    dict.set_item("type", "drop")?;
                    dict.set_item("to_sq", to.index())?;
                    dict.set_item("piece_type_idx", piece_type.index())?;
                }
            }
            Ok(dict.into())
        })
    }

    #[getter]
    fn action_space_size(&self) -> usize {
        Self::ACTION_SPACE_SIZE
    }
}
```

- [ ] **Step 4: Run cargo check**

Run: `cd shogi-engine && cargo check -p shogi-gym`
Expected: compiles without errors

- [ ] **Step 5: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/action_mapper.rs
git commit -m "feat(shogi-gym): ActionMapper trait + DefaultActionMapper with 13527-action encoding"
```

---

### Task 3: ObservationGenerator Trait + DefaultObservationGenerator

**Files:**
- Create: `shogi-engine/crates/shogi-gym/src/observation.rs`

- [ ] **Step 1: Write ObservationGenerator implementation with tests**

Write `shogi-engine/crates/shogi-gym/src/observation.rs`:

```rust
use shogi_core::{Color, GameState, HandPieceType, PieceType, Square};
use shogi_core::piece::Piece;

/// Trait for generating observation tensors from game state.
/// Implementations must be Send + Sync for rayon parallelism.
pub trait ObservationGenerator: Send + Sync {
    /// Write observation into buffer. Buffer length must equal channels() * 81.
    fn generate(&self, state: &GameState, perspective: Color, buffer: &mut [f32]);
    /// Number of observation channels.
    fn channels(&self) -> usize;
}

/// 46-channel observation generator matching Keisei's Python implementation.
///
/// Channel layout (all from current player's perspective):
///   0-7:   Current player's unpromoted pieces (P, L, N, S, G, B, R, K)
///   8-13:  Current player's promoted pieces (+P, +L, +N, +S, +B, +R)
///  14-21:  Opponent's unpromoted pieces (P, L, N, S, G, B, R, K)
///  22-27:  Opponent's promoted pieces (+P, +L, +N, +S, +B, +R)
///  28-34:  Current player's hand counts (normalized constant planes)
///  35-41:  Opponent's hand counts (normalized constant planes)
///  42:     Current player indicator (1.0 if Black, 0.0 if White)
///  43:     Move count (normalized: ply / max_ply)
///  44-45:  Reserved (zeros)
#[cfg_attr(feature = "pyo3", pyo3::pyclass)]
pub struct DefaultObservationGenerator;

impl DefaultObservationGenerator {
    pub const CHANNELS: usize = 46;

    pub fn new() -> Self {
        DefaultObservationGenerator
    }

    /// Map a PieceType to its unpromoted channel offset (0-7).
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

    /// Map a PieceType to its promoted channel offset (0-5).
    /// Only promotable pieces have promoted channels.
    #[inline]
    fn promoted_channel(pt: PieceType) -> usize {
        match pt {
            PieceType::Pawn   => 0,
            PieceType::Lance  => 1,
            PieceType::Knight => 2,
            PieceType::Silver => 3,
            PieceType::Bishop => 4,
            PieceType::Rook   => 5,
            _ => unreachable!("non-promotable piece in promoted_channel"),
        }
    }

    /// Maximum hand counts for normalization.
    #[inline]
    fn hand_max(hpt: HandPieceType) -> f32 {
        match hpt {
            HandPieceType::Pawn   => 18.0,
            HandPieceType::Lance  => 4.0,
            HandPieceType::Knight => 4.0,
            HandPieceType::Silver => 4.0,
            HandPieceType::Gold   => 4.0,
            HandPieceType::Bishop => 2.0,
            HandPieceType::Rook   => 2.0,
        }
    }
}

// Channel block start offsets
const CURR_UNPROMOTED: usize = 0;
const CURR_PROMOTED: usize = 8;
const OPP_UNPROMOTED: usize = 14;
const OPP_PROMOTED: usize = 22;
const CURR_HAND: usize = 28;
const OPP_HAND: usize = 35;
const PLAYER_INDICATOR: usize = 42;
const MOVE_COUNT: usize = 43;

impl ObservationGenerator for DefaultObservationGenerator {
    fn generate(&self, state: &GameState, perspective: Color, buffer: &mut [f32]) {
        debug_assert_eq!(buffer.len(), Self::CHANNELS * 81);
        buffer.fill(0.0);

        let is_black = matches!(perspective, Color::Black);
        let opponent = perspective.opponent();

        // Board pieces (channels 0-27)
        for idx in 0..81u8 {
            let sq = Square::new_unchecked(idx);
            if let Some(piece) = state.position.piece_at(sq) {
                let pt = piece.piece_type();
                let color = piece.color();
                let promoted = piece.is_promoted();

                // Flip square for White's perspective
                let obs_idx = if is_black { idx as usize } else { 80 - idx as usize };

                let is_current = color == perspective;
                let channel = if promoted {
                    let block = if is_current { CURR_PROMOTED } else { OPP_PROMOTED };
                    block + Self::promoted_channel(pt)
                } else {
                    let block = if is_current { CURR_UNPROMOTED } else { OPP_UNPROMOTED };
                    block + Self::unpromoted_channel(pt)
                };

                buffer[channel * 81 + obs_idx] = 1.0;
            }
        }

        // Hand pieces (channels 28-41): normalized constant planes
        for &hpt in &HandPieceType::ALL {
            let max_count = Self::hand_max(hpt);
            let curr_count = state.position.hand_count(perspective, hpt) as f32 / max_count;
            let opp_count = state.position.hand_count(opponent, hpt) as f32 / max_count;

            let curr_ch = CURR_HAND + hpt.index();
            let opp_ch = OPP_HAND + hpt.index();

            let curr_base = curr_ch * 81;
            let opp_base = opp_ch * 81;
            for sq in 0..81 {
                buffer[curr_base + sq] = curr_count;
                buffer[opp_base + sq] = opp_count;
            }
        }

        // Player indicator (channel 42): 1.0 if current player is Black
        let indicator_val = if is_black { 1.0f32 } else { 0.0 };
        let indicator_base = PLAYER_INDICATOR * 81;
        for sq in 0..81 {
            buffer[indicator_base + sq] = indicator_val;
        }

        // Move count (channel 43): normalized by max_ply
        let max_ply = if state.max_ply == 0 { 1 } else { state.max_ply };
        let move_norm = state.ply as f32 / max_ply as f32;
        let move_base = MOVE_COUNT * 81;
        for sq in 0..81 {
            buffer[move_base + sq] = move_norm;
        }

        // Channels 44-45: reserved (already zeroed)
    }

    fn channels(&self) -> usize {
        Self::CHANNELS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::GameState;

    #[test]
    fn test_channels() {
        let gen = DefaultObservationGenerator::new();
        assert_eq!(gen.channels(), 46);
    }

    #[test]
    fn test_startpos_observation_shape() {
        let gen = DefaultObservationGenerator::new();
        let state = GameState::new();
        let mut buffer = vec![0.0f32; 46 * 81];
        gen.generate(&state, Color::Black, &mut buffer);

        // Buffer should be exactly 46 * 81 = 3726 elements
        assert_eq!(buffer.len(), 3726);
    }

    #[test]
    fn test_startpos_black_perspective_pieces() {
        let gen = DefaultObservationGenerator::new();
        let state = GameState::new();
        let mut buffer = vec![0.0f32; 46 * 81];
        gen.generate(&state, Color::Black, &mut buffer);

        // Black's king at (8,4) = square index 76, channel 7 (King unpromoted)
        // For Black perspective, obs_idx = 76
        assert_eq!(buffer[7 * 81 + 76], 1.0, "Black king should be at channel 7, sq 76");

        // White's king at (0,4) = square index 4, channel 21 (opponent King unpromoted = 14+7)
        assert_eq!(buffer[21 * 81 + 4], 1.0, "White king should be at channel 21, sq 4");
    }

    #[test]
    fn test_startpos_white_perspective_flipped() {
        let gen = DefaultObservationGenerator::new();
        let state = GameState::new();

        let mut buf_black = vec![0.0f32; 46 * 81];
        let mut buf_white = vec![0.0f32; 46 * 81];
        gen.generate(&state, Color::Black, &mut buf_black);
        gen.generate(&state, Color::White, &mut buf_white);

        // For White perspective: current player is White.
        // White's king at (0,4) = sq 4, flipped = 80-4 = 76.
        // Channel 7 (current player's King, unpromoted).
        assert_eq!(buf_white[7 * 81 + 76], 1.0,
            "White perspective: White king (current) at flipped sq 76");

        // Black's king at (8,4) = sq 76, flipped = 80-76 = 4.
        // Channel 21 (opponent's King unpromoted = 14+7).
        assert_eq!(buf_white[21 * 81 + 4], 1.0,
            "White perspective: Black king (opponent) at flipped sq 4");
    }

    #[test]
    fn test_player_indicator_black() {
        let gen = DefaultObservationGenerator::new();
        let state = GameState::new();
        let mut buffer = vec![0.0f32; 46 * 81];
        gen.generate(&state, Color::Black, &mut buffer);

        // Channel 42: all 1.0 for Black
        for sq in 0..81 {
            assert_eq!(buffer[42 * 81 + sq], 1.0);
        }
    }

    #[test]
    fn test_player_indicator_white() {
        let gen = DefaultObservationGenerator::new();
        let state = GameState::new();
        let mut buffer = vec![0.0f32; 46 * 81];
        gen.generate(&state, Color::White, &mut buffer);

        // Channel 42: all 0.0 for White
        for sq in 0..81 {
            assert_eq!(buffer[42 * 81 + sq], 0.0);
        }
    }

    #[test]
    fn test_move_count_normalization() {
        let gen = DefaultObservationGenerator::new();
        let mut state = GameState::with_max_ply(100);
        let mut buffer = vec![0.0f32; 46 * 81];

        gen.generate(&state, Color::Black, &mut buffer);
        // Ply 0: move_count channel should be 0.0
        assert_eq!(buffer[43 * 81], 0.0);

        // Make a move to advance ply
        let moves = state.legal_moves();
        if !moves.is_empty() {
            state.make_move(moves[0]);
            gen.generate(&state, Color::White, &mut buffer);
            // Ply 1: move_count = 1/100 = 0.01
            let expected = 1.0 / 100.0;
            assert!((buffer[43 * 81] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reserved_channels_zero() {
        let gen = DefaultObservationGenerator::new();
        let state = GameState::new();
        let mut buffer = vec![0.0f32; 46 * 81];
        gen.generate(&state, Color::Black, &mut buffer);

        // Channels 44-45 should all be zero
        for ch in 44..46 {
            for sq in 0..81 {
                assert_eq!(buffer[ch * 81 + sq], 0.0);
            }
        }
    }

    #[test]
    fn test_hand_normalization() {
        // Startpos has no pieces in hand — all hand channels should be 0
        let gen = DefaultObservationGenerator::new();
        let state = GameState::new();
        let mut buffer = vec![0.0f32; 46 * 81];
        gen.generate(&state, Color::Black, &mut buffer);

        for ch in 28..42 {
            for sq in 0..81 {
                assert_eq!(buffer[ch * 81 + sq], 0.0,
                    "hand channel {} sq {} not zero at startpos", ch, sq);
            }
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cd shogi-engine && cargo test -p shogi-gym -- observation`
Expected: All tests pass

- [ ] **Step 3: Add PyO3 wrapper**

Add above the `#[cfg(test)]` module in `observation.rs`:

```rust
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[pymethods]
impl DefaultObservationGenerator {
    #[new]
    fn py_new() -> Self {
        Self::new()
    }

    #[getter]
    fn channels(&self) -> usize {
        Self::CHANNELS
    }
}
```

- [ ] **Step 4: Run cargo check**

Run: `cd shogi-engine && cargo check -p shogi-gym`
Expected: compiles

- [ ] **Step 5: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/observation.rs
git commit -m "feat(shogi-gym): ObservationGenerator trait + 46-channel default implementation"
```

---

### Task 4: StepResult Struct

**Files:**
- Create: `shogi-engine/crates/shogi-gym/src/step_result.rs`

- [ ] **Step 1: Write StepResult**

Write `shogi-engine/crates/shogi-gym/src/step_result.rs`:

```rust
use numpy::{PyArray1, PyArray2, PyArray4, PyArrayMethods};
use pyo3::prelude::*;

/// Termination reason codes packed into step_metadata.
/// These are u8 values for the metadata structured array.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum TerminationReason {
    NotTerminated = 0,
    Checkmate = 1,
    Repetition = 2,
    PerpetualCheck = 3,
    Impasse = 4,
    MaxMoves = 5,
}

impl TerminationReason {
    pub fn from_game_result(result: &shogi_core::GameResult) -> Self {
        match result {
            shogi_core::GameResult::InProgress => TerminationReason::NotTerminated,
            shogi_core::GameResult::Checkmate { .. } => TerminationReason::Checkmate,
            shogi_core::GameResult::Repetition => TerminationReason::Repetition,
            shogi_core::GameResult::PerpetualCheck { .. } => TerminationReason::PerpetualCheck,
            shogi_core::GameResult::Impasse { .. } => TerminationReason::Impasse,
            shogi_core::GameResult::MaxMoves => TerminationReason::MaxMoves,
        }
    }
}

/// Per-environment metadata packed as three parallel arrays rather than
/// a NumPy structured array (simpler FFI, same data).
#[pyclass]
pub struct StepMetadata {
    /// Captured piece hand-type index (0-6), or 255 for no capture.
    #[pyo3(get)]
    pub captured_piece: Py<PyArray1<u8>>,
    /// Termination reason code (see TerminationReason enum).
    #[pyo3(get)]
    pub termination_reason: Py<PyArray1<u8>>,
    /// Current ply count per environment.
    #[pyo3(get)]
    pub ply_count: Py<PyArray1<u16>>,
}

/// Result of a VecEnv.step() call. Holds references to pre-allocated buffers.
#[pyclass]
pub struct StepResult {
    #[pyo3(get)]
    pub observations: Py<PyArray4<f32>>,
    #[pyo3(get)]
    pub legal_masks: Py<PyArray2<bool>>,
    #[pyo3(get)]
    pub rewards: Py<PyArray1<f32>>,
    #[pyo3(get)]
    pub terminated: Py<PyArray1<bool>>,
    #[pyo3(get)]
    pub truncated: Py<PyArray1<bool>>,
    #[pyo3(get)]
    pub step_metadata: Py<StepMetadata>,
}

/// Result of a VecEnv.reset() call.
#[pyclass]
pub struct ResetResult {
    #[pyo3(get)]
    pub observations: Py<PyArray4<f32>>,
    #[pyo3(get)]
    pub legal_masks: Py<PyArray2<bool>>,
}
```

- [ ] **Step 2: Run cargo check**

Run: `cd shogi-engine && cargo check -p shogi-gym`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/step_result.rs
git commit -m "feat(shogi-gym): StepResult and StepMetadata structs for VecEnv output"
```

---

### Task 5: VecEnv — Core Batch Environment

**Files:**
- Create: `shogi-engine/crates/shogi-gym/src/vec_env.rs`
- Create: `shogi-engine/crates/shogi-gym/tests/test_vec_env.py`

This is the largest task. It builds the VecEnv with pre-allocated buffers, two-phase stepping, rayon parallelism, and GIL release.

- [ ] **Step 1: Write VecEnv struct and constructor**

Write `shogi-engine/crates/shogi-gym/src/vec_env.rs`:

```rust
use numpy::{PyArray1, PyArray2, PyArray4, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;
use shogi_core::{Color, GameResult, GameState, HandPieceType, Move, Square};

use crate::action_mapper::{ActionMapper, DefaultActionMapper};
use crate::observation::{DefaultObservationGenerator, ObservationGenerator};
use crate::step_result::{ResetResult, StepMetadata, StepResult, TerminationReason};

/// Vectorized environment that batch-steps N Shogi games in a single FFI call.
///
/// Pre-allocates all output buffers at construction. Uses rayon for parallel
/// game stepping with GIL release via py.allow_threads().
#[pyclass]
pub struct VecEnv {
    games: Vec<GameState>,
    num_envs: usize,
    max_ply: u32,
    parallel: bool,
    parallel_threshold: usize,

    // Rust-owned buffers written in-place each step
    obs_buffer: Vec<f32>,        // (N, C, 9, 9) flattened
    legal_mask_buffer: Vec<bool>, // (N, A) flattened
    reward_buffer: Vec<f32>,     // (N,)
    terminated_buffer: Vec<bool>, // (N,)
    truncated_buffer: Vec<bool>,  // (N,)
    captured_buffer: Vec<u8>,    // (N,) metadata
    term_reason_buffer: Vec<u8>, // (N,) metadata
    ply_buffer: Vec<u16>,        // (N,) metadata

    // Terminal observation buffer for auto-reset
    terminal_obs_buffer: Vec<f32>, // (N, C, 9, 9) — only valid for envs that just terminated

    mapper: DefaultActionMapper,
    obs_gen: DefaultObservationGenerator,
}

impl VecEnv {
    fn obs_size(&self) -> usize {
        DefaultObservationGenerator::CHANNELS * 81
    }

    fn action_size(&self) -> usize {
        DefaultActionMapper::ACTION_SPACE_SIZE
    }

    /// Reset a single game, writing initial obs and mask into the buffers.
    fn reset_single(&mut self, env_idx: usize) {
        self.games[env_idx] = GameState::with_max_ply(self.max_ply);
        let game = &self.games[env_idx];
        let perspective = game.position.current_player;
        let obs_size = self.obs_size();
        let action_size = self.action_size();

        let obs_slice = &mut self.obs_buffer[env_idx * obs_size..(env_idx + 1) * obs_size];
        self.obs_gen.generate(game, perspective, obs_slice);

        let mask_slice = &mut self.legal_mask_buffer[env_idx * action_size..(env_idx + 1) * action_size];
        let mapper = &self.mapper;
        // Cannot call game.write_legal_mask_into because we need mutable game + immutable mapper.
        // Instead generate legal moves and encode them.
        mask_slice.fill(false);
        let mut game_copy = self.games[env_idx].clone();
        let moves = game_copy.legal_moves();
        for mv in &moves {
            let idx = mapper.encode(*mv, perspective);
            mask_slice[idx] = true;
        }
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (num_envs, max_ply=500, parallel=true, parallel_threshold=64))]
    fn new(num_envs: usize, max_ply: u32, parallel: bool, parallel_threshold: usize) -> Self {
        let obs_size = DefaultObservationGenerator::CHANNELS * 81;
        let action_size = DefaultActionMapper::ACTION_SPACE_SIZE;

        VecEnv {
            games: (0..num_envs).map(|_| GameState::with_max_ply(max_ply)).collect(),
            num_envs,
            max_ply,
            parallel,
            parallel_threshold,
            obs_buffer: vec![0.0; num_envs * obs_size],
            legal_mask_buffer: vec![false; num_envs * action_size],
            reward_buffer: vec![0.0; num_envs],
            terminated_buffer: vec![false; num_envs],
            truncated_buffer: vec![false; num_envs],
            captured_buffer: vec![255; num_envs],
            term_reason_buffer: vec![0; num_envs],
            ply_buffer: vec![0; num_envs],
            terminal_obs_buffer: vec![0.0; num_envs * obs_size],
            mapper: DefaultActionMapper::new(),
            obs_gen: DefaultObservationGenerator::new(),
        }
    }

    /// Reset all environments. Returns (observations, legal_masks).
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<ResetResult> {
        for i in 0..self.num_envs {
            self.reset_single(i);
        }

        let obs_shape = [self.num_envs, DefaultObservationGenerator::CHANNELS, 9, 9];
        let mask_shape = [self.num_envs, DefaultActionMapper::ACTION_SPACE_SIZE];

        let obs_array = self.obs_buffer.to_pyarray(py).reshape(obs_shape)?.into();
        let mask_array = self.legal_mask_buffer.to_pyarray(py).reshape(mask_shape)?.into();

        Ok(ResetResult {
            observations: obs_array,
            legal_masks: mask_array,
        })
    }

    /// Step all environments with the given actions.
    ///
    /// actions: numpy array of shape (N,) with dtype int64 or int32.
    /// Each element is an action index in [0, action_space_size).
    ///
    /// Two-phase contract:
    /// 1. Validate all actions (no mutation)
    /// 2. Apply all actions (parallel mutation)
    fn step<'py>(&mut self, py: Python<'py>, actions: Vec<i64>) -> PyResult<StepResult> {
        if actions.len() != self.num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("expected {} actions, got {}", self.num_envs, actions.len()),
            ));
        }

        let action_size = self.action_size();
        let obs_size = self.obs_size();
        let mapper = &self.mapper;

        // Phase 1: Decode and validate all actions (read-only)
        let mut decoded_moves: Vec<Move> = Vec::with_capacity(self.num_envs);
        for (env_idx, &action) in actions.iter().enumerate() {
            let action_usize = action as usize;
            if action_usize >= action_size {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("env {}: action index {} out of range (max {})",
                            env_idx, action_usize, action_size - 1),
                ));
            }
            let perspective = self.games[env_idx].position.current_player;
            let mv = mapper.decode(action_usize, perspective)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                    format!("env {}: {}", env_idx, e),
                ))?;

            // Validate move is legal by checking the mask
            let mask_offset = env_idx * action_size + action_usize;
            if !self.legal_mask_buffer[mask_offset] {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    format!("env {}: action {} is not legal in current position", env_idx, action_usize),
                ));
            }

            decoded_moves.push(mv);
        }

        // Phase 2: Apply all moves (can be parallel)
        // For now, sequential — rayon parallelism requires splitting the buffers.
        // We'll add rayon in a follow-up step.
        for env_idx in 0..self.num_envs {
            let mv = decoded_moves[env_idx];
            let game = &mut self.games[env_idx];
            let undo = game.make_move(mv);

            // Record captured piece
            self.captured_buffer[env_idx] = match undo.captured {
                Some(piece) => {
                    match HandPieceType::from_piece_type(piece.piece_type()) {
                        Some(hpt) => hpt.index() as u8,
                        None => 255, // King capture (shouldn't happen)
                    }
                }
                None => 255,
            };

            // Check termination
            game.check_termination();
            let result = game.result;
            let terminated = result.is_terminal() && !result.is_truncation();
            let truncated = result.is_truncation();

            self.terminated_buffer[env_idx] = terminated;
            self.truncated_buffer[env_idx] = truncated;
            self.term_reason_buffer[env_idx] = TerminationReason::from_game_result(&result) as u8;
            self.ply_buffer[env_idx] = game.ply.min(u16::MAX as u32) as u16;

            // Compute reward
            self.reward_buffer[env_idx] = if terminated || truncated {
                self.compute_reward(&result, game.position.current_player.opponent())
            } else {
                0.0
            };

            if terminated || truncated {
                // Save terminal observation before auto-reset
                let perspective = game.position.current_player;
                let term_obs_slice = &mut self.terminal_obs_buffer[env_idx * obs_size..(env_idx + 1) * obs_size];
                self.obs_gen.generate(game, perspective, term_obs_slice);

                // Auto-reset
                self.reset_single(env_idx);
            } else {
                // Write observation and legal mask for continuing game
                let perspective = game.position.current_player;
                let obs_slice = &mut self.obs_buffer[env_idx * obs_size..(env_idx + 1) * obs_size];
                self.obs_gen.generate(game, perspective, obs_slice);

                let mask_slice = &mut self.legal_mask_buffer[env_idx * action_size..(env_idx + 1) * action_size];
                mask_slice.fill(false);
                let moves = game.legal_moves();
                for mv in &moves {
                    let idx = mapper.encode(*mv, perspective);
                    mask_slice[idx] = true;
                }
            }
        }

        // Build Python result
        let obs_shape = [self.num_envs, DefaultObservationGenerator::CHANNELS, 9, 9];
        let mask_shape = [self.num_envs, DefaultActionMapper::ACTION_SPACE_SIZE];

        let obs_array = self.obs_buffer.to_pyarray(py).reshape(obs_shape)?.into();
        let mask_array = self.legal_mask_buffer.to_pyarray(py).reshape(mask_shape)?.into();
        let reward_array = self.reward_buffer.to_pyarray(py).into();
        let terminated_array = self.terminated_buffer.to_pyarray(py).into();
        let truncated_array = self.truncated_buffer.to_pyarray(py).into();

        let captured_array = self.captured_buffer.to_pyarray(py).into();
        let term_reason_array = self.term_reason_buffer.to_pyarray(py).into();
        let ply_array = self.ply_buffer.to_pyarray(py).into();

        let metadata = Py::new(py, StepMetadata {
            captured_piece: captured_array,
            termination_reason: term_reason_array,
            ply_count: ply_array,
        })?;

        Ok(StepResult {
            observations: obs_array,
            legal_masks: mask_array,
            rewards: reward_array,
            terminated: terminated_array,
            truncated: truncated_array,
            step_metadata: metadata,
        })
    }

    #[getter]
    fn action_space_size(&self) -> usize {
        DefaultActionMapper::ACTION_SPACE_SIZE
    }

    #[getter]
    fn observation_channels(&self) -> usize {
        DefaultObservationGenerator::CHANNELS
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.num_envs
    }
}

impl VecEnv {
    /// Compute reward from a terminal game result.
    /// `last_mover` is the player who just moved (and caused the terminal state).
    fn compute_reward(&self, result: &GameResult, last_mover: Color) -> f32 {
        match result {
            GameResult::Checkmate { winner } => {
                if *winner == last_mover { 1.0 } else { -1.0 }
            }
            GameResult::PerpetualCheck { winner } => {
                if *winner == last_mover { 1.0 } else { -1.0 }
            }
            GameResult::Impasse { winner } => match winner {
                Some(w) => if *w == last_mover { 1.0 } else { -1.0 },
                None => 0.0, // draw
            },
            GameResult::Repetition => 0.0,
            GameResult::MaxMoves => 0.0,
            GameResult::InProgress => 0.0,
        }
    }
}
```

- [ ] **Step 2: Update lib.rs to remove pyclass from stubs and use real modules**

Update `shogi-engine/crates/shogi-gym/src/lib.rs` — the stubs from Task 1 are now replaced by real files. Ensure lib.rs matches:

```rust
use pyo3::prelude::*;

mod action_mapper;
mod observation;
mod step_result;
mod vec_env;
mod spectator;

/// Native module for shogi-gym RL environments.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<action_mapper::DefaultActionMapper>()?;
    m.add_class::<observation::DefaultObservationGenerator>()?;
    m.add_class::<vec_env::VecEnv>()?;
    m.add_class::<step_result::StepResult>()?;
    m.add_class::<step_result::ResetResult>()?;
    m.add_class::<step_result::StepMetadata>()?;
    Ok(())
}
```

- [ ] **Step 3: Run cargo check**

Run: `cd shogi-engine && cargo check -p shogi-gym`
Expected: compiles (spectator.rs is still a stub — SpectatorEnv not yet registered)

- [ ] **Step 4: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/vec_env.rs shogi-engine/crates/shogi-gym/src/lib.rs
git commit -m "feat(shogi-gym): VecEnv with two-phase step, pre-allocated buffers, auto-reset"
```

---

### Task 6: VecEnv Rayon Parallelism

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/vec_env.rs`

This task adds rayon parallel stepping with GIL release to the Phase 2 apply loop.

- [ ] **Step 1: Refactor Phase 2 into a parallel-capable function**

The key challenge is that rayon needs shared access to mapper/obs_gen and split mutable access to the buffers. We solve this by collecting all per-env state into a struct and using `par_iter_mut`.

Add a helper struct and refactor the step method. Replace the Phase 2 loop in `vec_env.rs` (the `for env_idx in 0..self.num_envs` block inside `step()`) with:

```rust
        // Phase 2: Apply all moves
        let use_parallel = self.parallel && self.num_envs >= self.parallel_threshold;

        // Pack per-env mutable state for parallel iteration.
        // We need to split self's buffers into per-env slices.
        struct EnvSlice<'a> {
            game: &'a mut GameState,
            mv: Move,
            obs: &'a mut [f32],
            mask: &'a mut [bool],
            reward: &'a mut f32,
            terminated: &'a mut bool,
            truncated: &'a mut bool,
            captured: &'a mut u8,
            term_reason: &'a mut u8,
            ply: &'a mut u16,
            terminal_obs: &'a mut [f32],
            max_ply: u32,
        }

        // Build slices — we need to split the buffers. Use itertools-style manual splitting.
        let obs_chunks: Vec<&mut [f32]> = self.obs_buffer.chunks_mut(obs_size).collect();
        let mask_chunks: Vec<&mut [bool]> = self.legal_mask_buffer.chunks_mut(action_size).collect();
        let term_obs_chunks: Vec<&mut [f32]> = self.terminal_obs_buffer.chunks_mut(obs_size).collect();

        // Zip everything together. We need to be careful about borrow splitting.
        // Use indices and unsafe split for the game vec.
        let max_ply = self.max_ply;
        let obs_gen = &self.obs_gen;

        // Process function for a single env
        let process_env = |env_slice: &mut EnvSlice| {
            let game = &mut *env_slice.game;
            let undo = game.make_move(env_slice.mv);

            // Captured piece
            *env_slice.captured = match undo.captured {
                Some(piece) => {
                    match HandPieceType::from_piece_type(piece.piece_type()) {
                        Some(hpt) => hpt.index() as u8,
                        None => 255,
                    }
                }
                None => 255,
            };

            // Check termination
            game.check_termination();
            let result = game.result;
            let is_terminated = result.is_terminal() && !result.is_truncation();
            let is_truncated = result.is_truncation();

            *env_slice.terminated = is_terminated;
            *env_slice.truncated = is_truncated;
            *env_slice.term_reason = TerminationReason::from_game_result(&result) as u8;
            *env_slice.ply = game.ply.min(u16::MAX as u32) as u16;

            // Compute reward
            let last_mover = game.position.current_player.opponent();
            *env_slice.reward = if is_terminated || is_truncated {
                match &result {
                    GameResult::Checkmate { winner } => if *winner == last_mover { 1.0 } else { -1.0 },
                    GameResult::PerpetualCheck { winner } => if *winner == last_mover { 1.0 } else { -1.0 },
                    GameResult::Impasse { winner } => match winner {
                        Some(w) => if *w == last_mover { 1.0 } else { -1.0 },
                        None => 0.0,
                    },
                    _ => 0.0,
                }
            } else {
                0.0
            };

            if is_terminated || is_truncated {
                // Terminal observation
                let perspective = game.position.current_player;
                obs_gen.generate(game, perspective, env_slice.terminal_obs);

                // Auto-reset
                *game = GameState::with_max_ply(env_slice.max_ply);
                let perspective = game.position.current_player;
                obs_gen.generate(game, perspective, env_slice.obs);

                env_slice.mask.fill(false);
                let moves = game.legal_moves();
                for mv in &moves {
                    let idx = mapper.encode(*mv, perspective);
                    env_slice.mask[idx] = true;
                }
            } else {
                let perspective = game.position.current_player;
                obs_gen.generate(game, perspective, env_slice.obs);

                env_slice.mask.fill(false);
                let moves = game.legal_moves();
                for mv in &moves {
                    let idx = mapper.encode(*mv, perspective);
                    env_slice.mask[idx] = true;
                }
            }
        };

        // We cannot easily use rayon with split borrows of Vec fields in a struct.
        // Instead, use unsafe to split the games vec and iterate with indices.
        // Alternative: restructure to use parallel arrays that can be independently split.
        //
        // For v1, we use py.allow_threads with sequential iteration.
        // Rayon parallelism requires restructuring games into a separate allocation
        // that can be split independently — deferred to a follow-up optimization.
        py.allow_threads(|| {
            for env_idx in 0..self.num_envs {
                let obs_start = env_idx * obs_size;
                let mask_start = env_idx * action_size;
                let term_obs_start = env_idx * obs_size;

                // Safe because each env_idx accesses non-overlapping slices
                let game = &mut self.games[env_idx];
                let undo = game.make_move(decoded_moves[env_idx]);

                self.captured_buffer[env_idx] = match undo.captured {
                    Some(piece) => match HandPieceType::from_piece_type(piece.piece_type()) {
                        Some(hpt) => hpt.index() as u8,
                        None => 255,
                    },
                    None => 255,
                };

                game.check_termination();
                let result = game.result;
                let is_terminated = result.is_terminal() && !result.is_truncation();
                let is_truncated = result.is_truncation();

                self.terminated_buffer[env_idx] = is_terminated;
                self.truncated_buffer[env_idx] = is_truncated;
                self.term_reason_buffer[env_idx] = TerminationReason::from_game_result(&result) as u8;
                self.ply_buffer[env_idx] = game.ply.min(u16::MAX as u32) as u16;

                let last_mover = game.position.current_player.opponent();
                self.reward_buffer[env_idx] = if is_terminated || is_truncated {
                    match &result {
                        GameResult::Checkmate { winner } => if *winner == last_mover { 1.0 } else { -1.0 },
                        GameResult::PerpetualCheck { winner } => if *winner == last_mover { 1.0 } else { -1.0 },
                        GameResult::Impasse { winner } => match winner {
                            Some(w) => if *w == last_mover { 1.0 } else { -1.0 },
                            None => 0.0,
                        },
                        _ => 0.0,
                    }
                } else {
                    0.0
                };

                if is_terminated || is_truncated {
                    let perspective = game.position.current_player;
                    let term_slice = &mut self.terminal_obs_buffer[term_obs_start..term_obs_start + obs_size];
                    self.obs_gen.generate(game, perspective, term_slice);

                    *game = GameState::with_max_ply(self.max_ply);
                    let perspective = game.position.current_player;
                    let obs_slice = &mut self.obs_buffer[obs_start..obs_start + obs_size];
                    self.obs_gen.generate(game, perspective, obs_slice);

                    let mask_slice = &mut self.legal_mask_buffer[mask_start..mask_start + action_size];
                    mask_slice.fill(false);
                    let moves = game.legal_moves();
                    for mv in &moves {
                        let idx = mapper.encode(*mv, perspective);
                        mask_slice[idx] = true;
                    }
                } else {
                    let perspective = game.position.current_player;
                    let obs_slice = &mut self.obs_buffer[obs_start..obs_start + obs_size];
                    self.obs_gen.generate(game, perspective, obs_slice);

                    let mask_slice = &mut self.legal_mask_buffer[mask_start..mask_start + action_size];
                    mask_slice.fill(false);
                    let moves = game.legal_moves();
                    for mv in &moves {
                        let idx = mapper.encode(*mv, perspective);
                        mask_slice[idx] = true;
                    }
                }
            }
        });
```

Note: The `EnvSlice` struct and `process_env` closure above are aspirational for rayon. For v1, the `py.allow_threads` sequential loop is the correct implementation. Rayon parallel split requires restructuring the data layout (parallel arrays of single-env structs) — that's a v1.1 optimization. Remove the `EnvSlice` struct and `process_env` closure from the actual code; they're design documentation.

- [ ] **Step 2: Simplify — keep only the py.allow_threads sequential loop**

The actual Phase 2 code in `step()` should be the `py.allow_threads(|| { for env_idx ... })` block only. Delete the `EnvSlice` struct, `process_env` closure, `obs_chunks`, `mask_chunks`, `term_obs_chunks`, and `use_parallel` variables. Also delete the duplicate sequential loop from Task 5 Step 1 (the one without `py.allow_threads`).

The final `step()` method should have:
1. Argument validation
2. Phase 1: decode + validate loop
3. Phase 2: `py.allow_threads(|| { for env_idx ... })` with the complete stepping logic
4. Build Python result arrays

- [ ] **Step 3: Run cargo check**

Run: `cd shogi-engine && cargo check -p shogi-gym`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/vec_env.rs
git commit -m "feat(shogi-gym): VecEnv GIL release with py.allow_threads"
```

---

### Task 7: SpectatorEnv

**Files:**
- Create: `shogi-engine/crates/shogi-gym/src/spectator.rs`
- Modify: `shogi-engine/crates/shogi-gym/src/lib.rs`

- [ ] **Step 1: Write SpectatorEnv implementation**

Write `shogi-engine/crates/shogi-gym/src/spectator.rs`:

```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;
use shogi_core::{Color, GameResult, GameState, HandPieceType, Move, PieceType, Square};
use shogi_core::piece::Piece;

use crate::action_mapper::{ActionMapper, DefaultActionMapper};
use crate::observation::{DefaultObservationGenerator, ObservationGenerator};

/// Single-game environment for spectator/display use.
///
/// Unlike VecEnv, SpectatorEnv:
/// - Returns rich Python dicts (not on the hot path)
/// - Does NOT auto-reset on game end
/// - Provides to_dict() for JSON serialization
#[pyclass]
pub struct SpectatorEnv {
    game: GameState,
    max_ply: u32,
    mapper: DefaultActionMapper,
    obs_gen: DefaultObservationGenerator,
    move_history: Vec<(usize, String)>, // (action_index, move_notation)
}

#[pymethods]
impl SpectatorEnv {
    #[new]
    #[pyo3(signature = (max_ply=500))]
    fn new(max_ply: u32) -> Self {
        SpectatorEnv {
            game: GameState::with_max_ply(max_ply),
            max_ply,
            mapper: DefaultActionMapper::new(),
            obs_gen: DefaultObservationGenerator::new(),
            move_history: Vec::new(),
        }
    }

    /// Reset to starting position.
    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.game = GameState::with_max_ply(self.max_ply);
        self.move_history.clear();
        self.state_to_dict(py)
    }

    /// Apply an action. Returns game state dict.
    /// Raises RuntimeError if game is already over.
    fn step<'py>(&mut self, py: Python<'py>, action: usize) -> PyResult<Bound<'py, PyDict>> {
        if self.game.result.is_terminal() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "game is already over — call reset() first",
            ));
        }

        let perspective = self.game.position.current_player;
        let mv = self.mapper.decode(action, perspective)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        let notation = self.move_to_notation(&mv);
        self.move_history.push((action, notation));

        self.game.make_move(mv);
        self.game.check_termination();

        self.state_to_dict(py)
    }

    /// Get current game state as a dict.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.state_to_dict(py)
    }

    /// Get the current SFEN string.
    fn to_sfen(&self) -> String {
        self.game.position.to_sfen()
    }

    /// Get observation tensor as a flat list (C * 9 * 9).
    fn get_observation(&self) -> Vec<f32> {
        let mut buffer = vec![0.0f32; DefaultObservationGenerator::CHANNELS * 81];
        let perspective = self.game.position.current_player;
        self.obs_gen.generate(&self.game, perspective, &mut buffer);
        buffer
    }

    /// Get legal action indices.
    fn legal_actions(&mut self) -> Vec<usize> {
        let perspective = self.game.position.current_player;
        let moves = self.game.legal_moves();
        moves.iter().map(|mv| self.mapper.encode(*mv, perspective)).collect()
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.game.result.is_terminal()
    }

    #[getter]
    fn current_player(&self) -> &str {
        match self.game.position.current_player {
            Color::Black => "black",
            Color::White => "white",
        }
    }

    #[getter]
    fn ply(&self) -> u32 {
        self.game.ply
    }

    #[getter]
    fn action_space_size(&self) -> usize {
        DefaultActionMapper::ACTION_SPACE_SIZE
    }
}

impl SpectatorEnv {
    fn state_to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        // Board: list of 81 elements, each None or {"type": str, "color": str, "promoted": bool}
        let board_list = pyo3::types::PyList::empty(py);
        for idx in 0..81u8 {
            let sq = Square::new_unchecked(idx);
            match self.game.position.piece_at(sq) {
                Some(piece) => {
                    let piece_dict = PyDict::new(py);
                    piece_dict.set_item("type", piece_type_name(piece.piece_type()))?;
                    piece_dict.set_item("color", color_name(piece.color()))?;
                    piece_dict.set_item("promoted", piece.is_promoted())?;
                    piece_dict.set_item("row", sq.row())?;
                    piece_dict.set_item("col", sq.col())?;
                    board_list.append(piece_dict)?;
                }
                None => {
                    board_list.append(py.None())?;
                }
            }
        }
        dict.set_item("board", board_list)?;

        // Hands
        let hands_dict = PyDict::new(py);
        for color in [Color::Black, Color::White] {
            let hand_dict = PyDict::new(py);
            for &hpt in &HandPieceType::ALL {
                let count = self.game.position.hand_count(color, hpt);
                hand_dict.set_item(piece_type_name(hpt.to_piece_type()), count)?;
            }
            hands_dict.set_item(color_name(color), hand_dict)?;
        }
        dict.set_item("hands", hands_dict)?;

        // Game state
        dict.set_item("current_player", color_name(self.game.position.current_player))?;
        dict.set_item("ply", self.game.ply)?;
        dict.set_item("is_over", self.game.result.is_terminal())?;
        dict.set_item("result", game_result_str(&self.game.result))?;
        dict.set_item("sfen", self.game.position.to_sfen())?;
        dict.set_item("in_check", self.game.is_in_check())?;

        // Move history
        let history_list = pyo3::types::PyList::empty(py);
        for (action, notation) in &self.move_history {
            let entry = PyDict::new(py);
            entry.set_item("action", *action)?;
            entry.set_item("notation", notation.as_str())?;
            history_list.append(entry)?;
        }
        dict.set_item("move_history", history_list)?;

        Ok(dict)
    }

    fn move_to_notation(&self, mv: &Move) -> String {
        match mv {
            Move::Board { from, to, promote } => {
                let promo = if *promote { "+" } else { "" };
                format!("{}{}→{}{}{}",
                    9 - from.col(), (b'a' + from.row()) as char,
                    9 - to.col(), (b'a' + to.row()) as char,
                    promo)
            }
            Move::Drop { to, piece_type } => {
                let pt = match piece_type {
                    HandPieceType::Pawn   => "P",
                    HandPieceType::Lance  => "L",
                    HandPieceType::Knight => "N",
                    HandPieceType::Silver => "S",
                    HandPieceType::Gold   => "G",
                    HandPieceType::Bishop => "B",
                    HandPieceType::Rook   => "R",
                };
                format!("{}*{}{}", pt, 9 - to.col(), (b'a' + to.row()) as char)
            }
        }
    }
}

fn piece_type_name(pt: PieceType) -> &'static str {
    match pt {
        PieceType::Pawn   => "pawn",
        PieceType::Lance  => "lance",
        PieceType::Knight => "knight",
        PieceType::Silver => "silver",
        PieceType::Gold   => "gold",
        PieceType::Bishop => "bishop",
        PieceType::Rook   => "rook",
        PieceType::King   => "king",
    }
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::Black => "black",
        Color::White => "white",
    }
}

fn game_result_str(r: &GameResult) -> &'static str {
    match r {
        GameResult::InProgress => "in_progress",
        GameResult::Checkmate { .. } => "checkmate",
        GameResult::Repetition => "repetition",
        GameResult::PerpetualCheck { .. } => "perpetual_check",
        GameResult::Impasse { .. } => "impasse",
        GameResult::MaxMoves => "max_moves",
    }
}
```

- [ ] **Step 2: Update lib.rs to register SpectatorEnv**

Ensure `shogi-engine/crates/shogi-gym/src/lib.rs` includes:

```rust
m.add_class::<spectator::SpectatorEnv>()?;
```

- [ ] **Step 3: Run cargo check**

Run: `cd shogi-engine && cargo check -p shogi-gym`
Expected: compiles

- [ ] **Step 4: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/spectator.rs shogi-engine/crates/shogi-gym/src/lib.rs
git commit -m "feat(shogi-gym): SpectatorEnv with rich dict output and move notation"
```

---

### Task 8: Maturin Build + Python Integration Tests

**Files:**
- Create: `shogi-engine/crates/shogi-gym/tests/test_action_mapper.py`
- Create: `shogi-engine/crates/shogi-gym/tests/test_vec_env.py`
- Create: `shogi-engine/crates/shogi-gym/tests/test_spectator.py`
- Create: `shogi-engine/crates/shogi-gym/tests/test_observation.py`

- [ ] **Step 1: Build the Python module with maturin**

Run: `cd shogi-engine/crates/shogi-gym && maturin develop --release`
Expected: builds and installs `shogi_gym` into the active venv

If maturin is not installed:
```bash
pip install maturin
```

- [ ] **Step 2: Create test directory**

```bash
mkdir -p shogi-engine/crates/shogi-gym/tests
```

- [ ] **Step 3: Write ActionMapper Python tests**

Write `shogi-engine/crates/shogi-gym/tests/test_action_mapper.py`:

```python
"""Tests for DefaultActionMapper Python bindings."""

import pytest
from shogi_gym import DefaultActionMapper


class TestDefaultActionMapper:
    def setup_method(self):
        self.mapper = DefaultActionMapper()

    def test_action_space_size(self):
        assert self.mapper.action_space_size == 13_527

    def test_board_move_roundtrip(self):
        """Encode a board move, decode it, verify fields match."""
        idx = self.mapper.encode_board_move(
            from_sq=0, to_sq=1, promote=False, is_white=False
        )
        decoded = self.mapper.decode(idx, is_white=False)
        assert decoded["type"] == "board"
        assert decoded["from_sq"] == 0
        assert decoded["to_sq"] == 1
        assert decoded["promote"] is False

    def test_board_move_with_promotion(self):
        idx = self.mapper.encode_board_move(
            from_sq=10, to_sq=1, promote=True, is_white=False
        )
        decoded = self.mapper.decode(idx, is_white=False)
        assert decoded["type"] == "board"
        assert decoded["promote"] is True

    def test_drop_move_roundtrip(self):
        idx = self.mapper.encode_drop_move(
            to_sq=40, piece_type_idx=0, is_white=False
        )
        decoded = self.mapper.decode(idx, is_white=False)
        assert decoded["type"] == "drop"
        assert decoded["to_sq"] == 40
        assert decoded["piece_type_idx"] == 0

    def test_all_drop_indices_in_range(self):
        """All 567 drop move indices should be in [12960, 13527)."""
        for sq in range(81):
            for pt in range(7):
                idx = self.mapper.encode_drop_move(
                    to_sq=sq, piece_type_idx=pt, is_white=False
                )
                assert 12_960 <= idx < 13_527

    def test_perspective_flip_board_move(self):
        """Same physical move, different perspectives, same index."""
        # Black: (0,0) -> (0,1) encodes from sq 0 to sq 1
        idx_black = self.mapper.encode_board_move(0, 1, False, is_white=False)
        # White: flipped squares (80, 79) — same perspective index
        idx_white = self.mapper.encode_board_move(80, 79, False, is_white=True)
        assert idx_black == idx_white

    def test_decode_out_of_range(self):
        with pytest.raises(ValueError):
            self.mapper.decode(13_527, is_white=False)

    def test_invalid_square(self):
        with pytest.raises(ValueError):
            self.mapper.encode_board_move(81, 0, False, is_white=False)

    def test_invalid_piece_type(self):
        with pytest.raises(ValueError):
            self.mapper.encode_drop_move(0, 7, is_white=False)
```

- [ ] **Step 4: Write VecEnv Python tests**

Write `shogi-engine/crates/shogi-gym/tests/test_vec_env.py`:

```python
"""Tests for VecEnv Python bindings."""

import numpy as np
import pytest
from shogi_gym import VecEnv


class TestVecEnvConstruction:
    def test_create(self):
        env = VecEnv(num_envs=4, max_ply=100)
        assert env.num_envs == 4
        assert env.action_space_size == 13_527
        assert env.observation_channels == 46

    def test_reset_shapes(self):
        env = VecEnv(num_envs=4, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)
        masks = np.asarray(result.legal_masks)

        assert obs.shape == (4, 46, 9, 9)
        assert obs.dtype == np.float32
        assert masks.shape == (4, 13_527)
        assert masks.dtype == np.bool_

    def test_reset_legal_masks_nonzero(self):
        """At start position, each env should have legal moves."""
        env = VecEnv(num_envs=2, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)
        for i in range(2):
            assert masks[i].sum() > 0, f"env {i} has no legal moves at start"


class TestVecEnvStepping:
    def test_step_shapes(self):
        env = VecEnv(num_envs=2, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)

        # Pick first legal action for each env
        actions = []
        for i in range(2):
            legal_indices = np.where(masks[i])[0]
            actions.append(int(legal_indices[0]))

        step_result = env.step(actions)
        obs = np.asarray(step_result.observations)
        masks = np.asarray(step_result.legal_masks)
        rewards = np.asarray(step_result.rewards)
        terminated = np.asarray(step_result.terminated)
        truncated = np.asarray(step_result.truncated)

        assert obs.shape == (2, 46, 9, 9)
        assert masks.shape == (2, 13_527)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)

    def test_step_wrong_num_actions(self):
        env = VecEnv(num_envs=2)
        env.reset()
        with pytest.raises(ValueError):
            env.step([0])  # Should need 2 actions

    def test_step_illegal_action(self):
        """Stepping with an action not in the legal mask should raise."""
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)

        # Find an illegal action
        illegal_indices = np.where(~masks[0])[0]
        assert len(illegal_indices) > 0, "all actions legal? unexpected"

        with pytest.raises(RuntimeError):
            env.step([int(illegal_indices[0])])

    def test_step_metadata(self):
        env = VecEnv(num_envs=2, max_ply=100)
        result = env.reset()
        masks = np.asarray(result.legal_masks)

        actions = []
        for i in range(2):
            legal = np.where(masks[i])[0]
            actions.append(int(legal[0]))

        step_result = env.step(actions)
        meta = step_result.step_metadata
        captured = np.asarray(meta.captured_piece)
        term_reason = np.asarray(meta.termination_reason)
        ply = np.asarray(meta.ply_count)

        assert captured.shape == (2,)
        assert term_reason.shape == (2,)
        assert ply.shape == (2,)

    def test_multi_step_no_crash(self):
        """Run 20 steps to check for panics/memory issues."""
        env = VecEnv(num_envs=4, max_ply=50)
        result = env.reset()
        masks = np.asarray(result.legal_masks)

        for _ in range(20):
            actions = []
            for i in range(4):
                legal = np.where(masks[i])[0]
                actions.append(int(legal[np.random.randint(len(legal))]))

            step_result = env.step(actions)
            masks = np.asarray(step_result.legal_masks)
```

- [ ] **Step 5: Write SpectatorEnv Python tests**

Write `shogi-engine/crates/shogi-gym/tests/test_spectator.py`:

```python
"""Tests for SpectatorEnv Python bindings."""

import pytest
from shogi_gym import SpectatorEnv


class TestSpectatorEnv:
    def test_create(self):
        env = SpectatorEnv()
        assert env.action_space_size == 13_527
        assert env.current_player == "black"
        assert env.ply == 0
        assert not env.is_over

    def test_reset(self):
        env = SpectatorEnv()
        state = env.reset()
        assert state["current_player"] == "black"
        assert state["ply"] == 0
        assert state["is_over"] is False
        assert state["result"] == "in_progress"
        assert len(state["board"]) == 81

    def test_step(self):
        env = SpectatorEnv()
        env.reset()
        legal = env.legal_actions()
        assert len(legal) > 0

        state = env.step(legal[0])
        assert state["ply"] == 1
        assert state["current_player"] == "white"

    def test_step_after_game_over_raises(self):
        env = SpectatorEnv(max_ply=2)
        env.reset()
        # Play 2 moves to trigger max_ply truncation
        for _ in range(2):
            legal = env.legal_actions()
            if env.is_over:
                break
            env.step(legal[0])

        if env.is_over:
            with pytest.raises(RuntimeError):
                env.step(0)

    def test_no_auto_reset(self):
        """SpectatorEnv should NOT auto-reset on game end."""
        env = SpectatorEnv(max_ply=2)
        env.reset()
        for _ in range(10):
            if env.is_over:
                break
            legal = env.legal_actions()
            env.step(legal[0])
        # After game ends, ply should stay where it was
        if env.is_over:
            ply_at_end = env.ply
            assert ply_at_end <= 2

    def test_to_dict(self):
        env = SpectatorEnv()
        env.reset()
        d = env.to_dict()
        assert "board" in d
        assert "hands" in d
        assert "sfen" in d
        assert "move_history" in d

    def test_to_sfen(self):
        env = SpectatorEnv()
        env.reset()
        sfen = env.to_sfen()
        assert "lnsgkgsnl" in sfen.lower()

    def test_move_history(self):
        env = SpectatorEnv()
        env.reset()
        legal = env.legal_actions()
        env.step(legal[0])
        d = env.to_dict()
        assert len(d["move_history"]) == 1
        assert "action" in d["move_history"][0]
        assert "notation" in d["move_history"][0]
```

- [ ] **Step 6: Write observation Python tests**

Write `shogi-engine/crates/shogi-gym/tests/test_observation.py`:

```python
"""Tests for observation generation via VecEnv."""

import numpy as np
import pytest
from shogi_gym import VecEnv


class TestObservation:
    def test_startpos_has_pieces(self):
        """Starting position should have non-zero piece planes."""
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)[0]  # (46, 9, 9)

        # Current player (Black) pieces: channels 0-7
        piece_sum = obs[0:8].sum()
        assert piece_sum > 0, "no current player pieces in observation"

        # Opponent pieces: channels 14-21
        opp_sum = obs[14:22].sum()
        assert opp_sum > 0, "no opponent pieces in observation"

    def test_player_indicator_channel(self):
        """Channel 42 should be all 1.0 for Black (move 0)."""
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)[0]
        assert np.allclose(obs[42], 1.0), "player indicator should be 1.0 for Black"

    def test_reserved_channels_zero(self):
        """Channels 44-45 should be all zero."""
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)[0]
        assert np.allclose(obs[44], 0.0)
        assert np.allclose(obs[45], 0.0)

    def test_hand_channels_zero_at_start(self):
        """No pieces in hand at start — channels 28-41 all zero."""
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)[0]
        assert np.allclose(obs[28:42], 0.0)

    def test_observation_dtype(self):
        env = VecEnv(num_envs=1, max_ply=100)
        result = env.reset()
        obs = np.asarray(result.observations)
        assert obs.dtype == np.float32
```

- [ ] **Step 7: Rebuild and run Python tests**

```bash
cd shogi-engine/crates/shogi-gym && maturin develop --release
cd shogi-engine/crates/shogi-gym && python -m pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add shogi-engine/crates/shogi-gym/tests/ shogi-engine/python/
git commit -m "test(shogi-gym): Python integration tests for ActionMapper, VecEnv, SpectatorEnv, observations"
```

---

### Task 9: Rust-side Tests + Final Verification

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/vec_env.rs` (add Rust tests)

- [ ] **Step 1: Add Rust unit tests for reward computation**

Add `#[cfg(test)]` module to the bottom of `vec_env.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_checkmate_winner() {
        let env = VecEnv::new(1, 100, false, 64);
        // Checkmate where Black wins — reward from Black's perspective
        let result = GameResult::Checkmate { winner: Color::Black };
        assert_eq!(env.compute_reward(&result, Color::Black), 1.0);
        assert_eq!(env.compute_reward(&result, Color::White), -1.0);
    }

    #[test]
    fn test_reward_repetition_draw() {
        let env = VecEnv::new(1, 100, false, 64);
        let result = GameResult::Repetition;
        assert_eq!(env.compute_reward(&result, Color::Black), 0.0);
        assert_eq!(env.compute_reward(&result, Color::White), 0.0);
    }

    #[test]
    fn test_reward_max_moves() {
        let env = VecEnv::new(1, 100, false, 64);
        let result = GameResult::MaxMoves;
        assert_eq!(env.compute_reward(&result, Color::Black), 0.0);
    }

    #[test]
    fn test_reward_impasse_draw() {
        let env = VecEnv::new(1, 100, false, 64);
        let result = GameResult::Impasse { winner: None };
        assert_eq!(env.compute_reward(&result, Color::Black), 0.0);
    }

    #[test]
    fn test_reward_impasse_winner() {
        let env = VecEnv::new(1, 100, false, 64);
        let result = GameResult::Impasse { winner: Some(Color::White) };
        assert_eq!(env.compute_reward(&result, Color::White), 1.0);
        assert_eq!(env.compute_reward(&result, Color::Black), -1.0);
    }

    #[test]
    fn test_reward_perpetual_check() {
        let env = VecEnv::new(1, 100, false, 64);
        let result = GameResult::PerpetualCheck { winner: Color::Black };
        assert_eq!(env.compute_reward(&result, Color::Black), 1.0);
        assert_eq!(env.compute_reward(&result, Color::White), -1.0);
    }
}
```

- [ ] **Step 2: Run all Rust tests**

Run: `cd shogi-engine && cargo test --workspace`
Expected: all shogi-core + shogi-gym tests pass

- [ ] **Step 3: Run all Python tests**

Run: `cd shogi-engine/crates/shogi-gym && maturin develop --release && python -m pytest tests/ -v`
Expected: all Python tests pass

- [ ] **Step 4: Commit**

```bash
git add shogi-engine/crates/shogi-gym/src/vec_env.rs
git commit -m "test(shogi-gym): Rust unit tests for reward computation"
```

---

### Task 10: Update Python Package Exports + Final Polish

**Files:**
- Modify: `shogi-engine/python/shogi_gym/__init__.py`

- [ ] **Step 1: Verify all exports work from Python**

Run a quick smoke test:

```bash
cd shogi-engine/crates/shogi-gym && maturin develop --release
python -c "
from shogi_gym import VecEnv, SpectatorEnv, DefaultActionMapper, DefaultObservationGenerator, StepResult
env = VecEnv(num_envs=2, max_ply=50)
result = env.reset()
print(f'obs shape: {result.observations.shape}')
print(f'mask shape: {result.legal_masks.shape}')
print(f'action_space: {env.action_space_size}')
print(f'channels: {env.observation_channels}')

spec = SpectatorEnv()
state = spec.reset()
print(f'sfen: {spec.to_sfen()}')
print(f'legal actions: {len(spec.legal_actions())}')
print('All exports working.')
"
```

Expected: prints shapes, sizes, and "All exports working."

- [ ] **Step 2: Run full test suite**

```bash
cd shogi-engine && cargo test --workspace
cd shogi-engine/crates/shogi-gym && python -m pytest tests/ -v
```

Expected: all Rust and Python tests pass

- [ ] **Step 3: Commit final state**

```bash
git add -A shogi-engine/
git commit -m "feat(shogi-gym): complete v1 with VecEnv, SpectatorEnv, ActionMapper, ObsGen, Python bindings"
```

---

## Spec Coverage Checklist

| Spec Requirement | Task |
|-----------------|------|
| VecEnv batch-steps N games | Task 5, 6 |
| Pre-allocated output buffers | Task 5 (constructor) |
| rayon parallelism + GIL release | Task 6 (py.allow_threads) |
| Two-phase step contract | Task 5 (Phase 1 + Phase 2) |
| terminated + truncated (Gymnasium v1) | Task 4, 5 |
| step_metadata packed arrays | Task 4 |
| Auto-reset with terminal obs buffer | Task 5 |
| SpectatorEnv rich dict output | Task 7 |
| SpectatorEnv no auto-reset | Task 7 |
| DefaultActionMapper 13,527 actions | Task 2 |
| Perspective flipping | Task 2 |
| DefaultObservationGenerator 46 channels | Task 3 |
| ActionMapper/ObsGen as traits | Task 2, 3 |
| PyO3/maturin bindings | Task 1, 8 |
| Python package at shogi-engine/python/ | Task 1 |
| Illegal action = RuntimeError | Task 5 (Phase 1 validation) |
| Default step returns copies | Task 5 (to_pyarray makes copies) |

**Not in v1 (spec explicitly defers):**
- `zero_copy=True` mode — design documented, not implemented
- Custom ActionMapper/ObsGen from Python — spec says Rust-only
- `reset_from_sfen()` — v2 extension point
- Full rayon parallel iteration — v1 uses sequential with GIL release; parallel split deferred
