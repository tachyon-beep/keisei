use pyo3::prelude::*;
use pyo3::types::PyDict;
use shogi_core::{Color, HandPieceType, Move, Square};

// ---------------------------------------------------------------------------
// ActionMapper trait
// ---------------------------------------------------------------------------

/// Encode/decode moves to/from action indices with perspective support.
#[allow(dead_code)]
pub trait ActionMapper: Send + Sync {
    fn encode(&self, mv: Move, perspective: Color) -> Result<usize, String>;
    fn decode(&self, idx: usize, perspective: Color) -> Result<Move, String>;
    fn action_space_size(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Encoding constants
// ---------------------------------------------------------------------------

/// Board moves: 81 sources × 80 destinations × 2 (promote flag) = 12,960
const BOARD_MOVE_COUNT: usize = 81 * 80 * 2;

/// Drop moves: 81 destinations × 7 hand piece types = 567
const DROP_MOVE_COUNT: usize = 81 * 7;

/// Total action space: 12,960 + 567 = 13,527
pub const ACTION_SPACE_SIZE: usize = BOARD_MOVE_COUNT + DROP_MOVE_COUNT;

// ---------------------------------------------------------------------------
// DefaultActionMapper
// ---------------------------------------------------------------------------

/// 13,527-action encoding for Shogi moves.
///
/// Board moves [0..12960):
///   index = from_idx * 160 + dest_offset * 2 + promote_bit
///   dest_offset = if to_idx > from_idx { to_idx - 1 } else { to_idx }
///
/// Drop moves [12960..13527):
///   index = 12960 + to_idx * 7 + piece_type.index()
///
/// Perspective flipping: when perspective is White, squares are flipped
/// via sq.flip() (which does 80 - sq.index()) before encoding / after decoding.
#[pyclass]
pub struct DefaultActionMapper;

impl DefaultActionMapper {
    fn apply_perspective(sq: Square, perspective: Color) -> Square {
        match perspective {
            Color::Black => sq,
            Color::White => sq.flip(),
        }
    }

    fn encode_board_internal(from_idx: usize, to_idx: usize, promote: bool) -> usize {
        let dest_offset = if to_idx > from_idx {
            to_idx - 1
        } else {
            to_idx
        };
        let promote_bit = if promote { 1 } else { 0 };
        from_idx * 160 + dest_offset * 2 + promote_bit
    }

    fn decode_board_internal(idx: usize) -> (usize, usize, bool) {
        let from_idx = idx / 160;
        let remainder = idx % 160;
        let dest_offset = remainder / 2;
        let promote = (remainder % 2) == 1;
        // Reconstruct to_idx from dest_offset
        let to_idx = if dest_offset >= from_idx {
            dest_offset + 1
        } else {
            dest_offset
        };
        (from_idx, to_idx, promote)
    }
}

impl ActionMapper for DefaultActionMapper {
    fn encode(&self, mv: Move, perspective: Color) -> Result<usize, String> {
        Ok(match mv {
            Move::Board { from, to, promote } => {
                let from_p = Self::apply_perspective(from, perspective);
                let to_p = Self::apply_perspective(to, perspective);
                Self::encode_board_internal(from_p.index(), to_p.index(), promote)
            }
            Move::Drop { to, piece_type } => {
                let to_p = Self::apply_perspective(to, perspective);
                BOARD_MOVE_COUNT + to_p.index() * HandPieceType::COUNT + piece_type.index()
            }
        })
    }

    fn decode(&self, idx: usize, perspective: Color) -> Result<Move, String> {
        if idx >= ACTION_SPACE_SIZE {
            return Err(format!(
                "action index {} is out of range (max {})",
                idx,
                ACTION_SPACE_SIZE - 1
            ));
        }

        if idx < BOARD_MOVE_COUNT {
            let (from_idx, to_idx, promote) = Self::decode_board_internal(idx);
            let from_p = Square::new_unchecked(from_idx as u8);
            let to_p = Square::new_unchecked(to_idx as u8);
            // Un-apply perspective
            let from = Self::apply_perspective(from_p, perspective);
            let to = Self::apply_perspective(to_p, perspective);
            Ok(Move::Board { from, to, promote })
        } else {
            let drop_idx = idx - BOARD_MOVE_COUNT;
            let to_idx = drop_idx / HandPieceType::COUNT;
            let piece_idx = drop_idx % HandPieceType::COUNT;
            let to_p = Square::new_unchecked(to_idx as u8);
            let to = Self::apply_perspective(to_p, perspective);
            let piece_type = HandPieceType::ALL[piece_idx];
            Ok(Move::Drop { to, piece_type })
        }
    }

    fn action_space_size(&self) -> usize {
        ACTION_SPACE_SIZE
    }
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------

#[pymethods]
impl DefaultActionMapper {
    #[new]
    pub fn new() -> Self {
        DefaultActionMapper
    }

    /// Encode a board move to an action index.
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
        self.encode(mv, perspective)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Encode a drop move to an action index.
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
                "piece_type_idx {} is out of range (max {})",
                piece_type_idx,
                HandPieceType::COUNT - 1
            )));
        }
        let piece_type = HandPieceType::ALL[piece_type_idx];
        let perspective = if is_white { Color::White } else { Color::Black };
        let mv = Move::Drop { to, piece_type };
        self.encode(mv, perspective)
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Decode an action index into a move dict.
    ///
    /// For board moves, returns: {"type": "board", "from_sq": u8, "to_sq": u8, "promote": bool}
    /// For drop moves, returns:  {"type": "drop", "to_sq": u8, "piece_type_idx": usize}
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

    /// Total number of actions in the action space.
    #[getter]
    pub fn action_space_size(&self) -> usize {
        ACTION_SPACE_SIZE
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn mapper() -> DefaultActionMapper {
        DefaultActionMapper
    }

    /// Convenience wrapper so tests can call the trait method without PyO3 ambiguity.
    fn trait_encode(m: &DefaultActionMapper, mv: Move, perspective: Color) -> usize {
        <DefaultActionMapper as ActionMapper>::encode(m, mv, perspective).unwrap()
    }

    fn trait_decode(m: &DefaultActionMapper, idx: usize, perspective: Color) -> Result<Move, String> {
        <DefaultActionMapper as ActionMapper>::decode(m, idx, perspective)
    }

    #[test]
    fn test_action_space_size() {
        assert_eq!(ACTION_SPACE_SIZE, 13_527);
        let m = mapper();
        assert_eq!(<DefaultActionMapper as ActionMapper>::action_space_size(&m), 13_527);
    }

    #[test]
    fn test_board_move_encode_decode_roundtrip_black() {
        let m = mapper();
        let perspective = Color::Black;

        // Representative moves
        let cases = vec![
            // (from, to, promote)
            (0usize, 1usize, false),
            (0, 1, true),
            (40, 41, false),
            (40, 39, true),
            (80, 79, false),
            (10, 70, false),
            (70, 10, true),
        ];

        for (from_idx, to_idx, promote) in cases {
            let from = Square::new_unchecked(from_idx as u8);
            let to = Square::new_unchecked(to_idx as u8);
            let mv = Move::Board { from, to, promote };
            let idx = trait_encode(&m, mv, perspective);
            assert!(idx < BOARD_MOVE_COUNT, "idx={idx} should be in board range");
            let decoded = trait_decode(&m, idx, perspective).expect("decode failed");
            assert_eq!(
                decoded, mv,
                "roundtrip failed for from={from_idx} to={to_idx} promote={promote}"
            );
        }
    }

    #[test]
    fn test_drop_move_encode_decode_roundtrip_black() {
        let m = mapper();
        let perspective = Color::Black;

        for &piece_type in &HandPieceType::ALL {
            for to_idx in 0u8..81 {
                let to = Square::new_unchecked(to_idx);
                let mv = Move::Drop { to, piece_type };
                let idx = trait_encode(&m, mv, perspective);
                assert!(
                    idx >= BOARD_MOVE_COUNT && idx < ACTION_SPACE_SIZE,
                    "idx={idx} should be in drop range"
                );
                let decoded = trait_decode(&m, idx, perspective).expect("decode failed");
                assert_eq!(
                    decoded, mv,
                    "roundtrip failed for to={to_idx} piece={piece_type:?}"
                );
            }
        }
    }

    #[test]
    fn test_exhaustive_board_move_roundtrip() {
        let m = mapper();
        let perspective = Color::Black;
        let mut seen = HashSet::new();

        for from_idx in 0u8..81 {
            for to_idx in 0u8..81 {
                if from_idx == to_idx {
                    continue;
                }
                for &promote in &[false, true] {
                    let from = Square::new_unchecked(from_idx);
                    let to = Square::new_unchecked(to_idx);
                    let mv = Move::Board { from, to, promote };
                    let idx = trait_encode(&m, mv, perspective);

                    assert!(idx < BOARD_MOVE_COUNT, "idx={idx} out of board range");
                    assert!(seen.insert(idx), "duplicate index {idx} for from={from_idx} to={to_idx} promote={promote}");

                    let decoded = trait_decode(&m, idx, perspective).expect("decode failed");
                    assert_eq!(decoded, mv, "roundtrip failed");
                }
            }
        }

        assert_eq!(seen.len(), BOARD_MOVE_COUNT, "not all board indices covered");
    }

    #[test]
    fn test_exhaustive_drop_move_roundtrip() {
        let m = mapper();
        let perspective = Color::Black;
        let mut seen = HashSet::new();

        for to_idx in 0u8..81 {
            for &piece_type in &HandPieceType::ALL {
                let to = Square::new_unchecked(to_idx);
                let mv = Move::Drop { to, piece_type };
                let idx = trait_encode(&m, mv, perspective);

                assert!(
                    idx >= BOARD_MOVE_COUNT && idx < ACTION_SPACE_SIZE,
                    "idx={idx} out of drop range"
                );
                assert!(
                    seen.insert(idx),
                    "duplicate index {idx} for to={to_idx} piece={piece_type:?}"
                );

                let decoded = trait_decode(&m, idx, perspective).expect("decode failed");
                assert_eq!(decoded, mv, "roundtrip failed");
            }
        }

        assert_eq!(seen.len(), DROP_MOVE_COUNT, "not all drop indices covered");
    }

    #[test]
    fn test_perspective_flip_board_move() {
        let m = mapper();

        // A physical board move: from=20, to=30, no promote
        let from = Square::new_unchecked(20);
        let to = Square::new_unchecked(30);
        let mv = Move::Board { from, to, promote: false };

        // The same physical move encoded from Black's and White's perspectives
        // should produce DIFFERENT indices (each sees the board from their side),
        // but encoding from White and then decoding from White should give back
        // the original move (same physical squares).
        let idx_black = trait_encode(&m, mv, Color::Black);
        let idx_white = trait_encode(&m, mv, Color::White);

        // The indices differ because the board is flipped
        assert_ne!(
            idx_black, idx_white,
            "Black and White perspectives should yield different indices"
        );

        // Roundtrip from White perspective
        let decoded_white = trait_decode(&m, idx_white, Color::White)
            .expect("decode from White failed");
        assert_eq!(decoded_white, mv, "White perspective roundtrip failed");

        // Cross-check: Black's index decoded from Black gives back the original move
        let decoded_black = trait_decode(&m, idx_black, Color::Black)
            .expect("decode from Black failed");
        assert_eq!(decoded_black, mv, "Black perspective roundtrip failed");
    }

    #[test]
    fn test_perspective_flip_drop_move() {
        let m = mapper();

        let to = Square::new_unchecked(10);
        let mv = Move::Drop { to, piece_type: HandPieceType::Rook };

        let idx_black = trait_encode(&m, mv, Color::Black);
        let idx_white = trait_encode(&m, mv, Color::White);

        // Different square indices due to flip(10) = 70
        assert_ne!(idx_black, idx_white);

        // Roundtrip from White
        let decoded = trait_decode(&m, idx_white, Color::White).expect("decode failed");
        assert_eq!(decoded, mv);

        // Roundtrip from Black
        let decoded = trait_decode(&m, idx_black, Color::Black).expect("decode failed");
        assert_eq!(decoded, mv);
    }

    #[test]
    fn test_decode_out_of_range() {
        let m = mapper();
        assert!(trait_decode(&m, ACTION_SPACE_SIZE, Color::Black).is_err());
        assert!(trait_decode(&m, ACTION_SPACE_SIZE + 1, Color::Black).is_err());
        assert!(trait_decode(&m, usize::MAX, Color::Black).is_err());
    }

    #[test]
    fn test_exhaustive_board_move_roundtrip_white() {
        let mapper = DefaultActionMapper;
        let perspective = Color::White;
        let mut seen = vec![false; BOARD_MOVE_COUNT];

        for from_idx in 0u8..81 {
            for to_idx in 0u8..81 {
                if from_idx == to_idx { continue; }
                for promote in [false, true] {
                    let mv = Move::Board {
                        from: Square::new_unchecked(from_idx),
                        to: Square::new_unchecked(to_idx),
                        promote,
                    };
                    let encoded = <DefaultActionMapper as ActionMapper>::encode(&mapper, mv, perspective).unwrap();
                    assert!(encoded < BOARD_MOVE_COUNT, "index {} out of board range", encoded);
                    assert!(!seen[encoded], "collision at index {} (white perspective)", encoded);
                    seen[encoded] = true;
                    let decoded = <DefaultActionMapper as ActionMapper>::decode(&mapper, encoded, perspective).unwrap();
                    assert_eq!(decoded, mv, "white perspective roundtrip failed");
                }
            }
        }
        assert!(seen.iter().all(|&v| v), "not all board indices covered (white)");
    }

    /// Exhaustive drop move roundtrip under White perspective with collision check.
    #[test]
    fn test_exhaustive_drop_move_roundtrip_white() {
        let m = mapper();
        let perspective = Color::White;
        let mut seen = HashSet::new();

        for to_idx in 0u8..81 {
            for &piece_type in &HandPieceType::ALL {
                let to = Square::new_unchecked(to_idx);
                let mv = Move::Drop { to, piece_type };
                let idx = trait_encode(&m, mv, perspective);

                assert!(
                    idx >= BOARD_MOVE_COUNT && idx < ACTION_SPACE_SIZE,
                    "idx={idx} out of drop range (white perspective)"
                );
                assert!(
                    seen.insert(idx),
                    "duplicate index {idx} for to={to_idx} piece={piece_type:?} (white perspective)"
                );

                let decoded = trait_decode(&m, idx, perspective).expect("decode failed (white)");
                assert_eq!(
                    decoded, mv,
                    "roundtrip failed for to={to_idx} piece={piece_type:?} (white perspective)"
                );
            }
        }

        assert_eq!(seen.len(), DROP_MOVE_COUNT, "not all drop indices covered (white)");
    }
}
