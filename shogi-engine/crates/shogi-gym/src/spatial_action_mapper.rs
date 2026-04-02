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

    /// Check if a board move is a knight move, and if so return slot (0=left, 1=right).
    ///
    /// In perspective space, knights always move |dr|=2, |dc|=1. The "left/right"
    /// classification is based on dc sign relative to dr sign:
    ///   dc < 0 when dr < 0 (or dc > 0 when dr > 0) → slot 0 ("left")
    ///   dc > 0 when dr < 0 (or dc < 0 when dr > 0) → slot 1 ("right")
    ///
    /// This normalization ensures encode/decode roundtrip regardless of whether
    /// dr is -2 (Black perspective) or +2 (White perspective after flip).
    fn knight_slot(from_row: i8, from_col: i8, to_row: i8, to_col: i8) -> Option<usize> {
        let dr = to_row - from_row;
        let dc = to_col - from_col;

        if dr.unsigned_abs() != 2 || dc.unsigned_abs() != 1 {
            return None;
        }

        // Normalize dc relative to dr: "same sign as dr" → left, "opposite sign" → right.
        // This is invariant under the 180° perspective flip which negates both dr and dc.
        let same_sign = (dr > 0 && dc > 0) || (dr < 0 && dc < 0);

        if same_sign {
            Some(0) // "left" (dc same sign as dr)
        } else {
            Some(1) // "right" (dc opposite sign to dr)
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
            let knight_side = knight_idx / 2; // 0=left (dc same sign as dr), 1=right (opposite)
            let promote = (knight_idx % 2) == 1;

            let from_sq = Square::new_unchecked(square_idx as u8);
            let from_row = from_sq.row() as i8;
            let from_col = from_sq.col() as i8;

            // Forward is always dr=-2 in perspective space (same convention as
            // DIRECTIONS[0]=N). The encoder applies the perspective flip to
            // from/to squares BEFORE computing dr, so in perspective space
            // the knight always moves toward decreasing row.
            let dr: i8 = -2;
            let dc: i8 = if knight_side == 0 { -1 } else { 1 };
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
        let m = mapper();
        let from = Square::new_unchecked(40);
        let to_row = 40i8 / 9 - 4;
        let to_col = 40i8 % 9;
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
        let to = Square::new_unchecked(11);
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

    #[test]
    fn test_corner_square_roundtrips() {
        let m = mapper();
        // Corner squares: 0 (row 0, col 0), 8 (row 0, col 8), 72 (row 8, col 0), 80 (row 8, col 8)
        for &corner in &[0u8, 8, 72, 80] {
            let from = Square::new_unchecked(corner);
            let from_row = from.row() as i8;
            let from_col = from.col() as i8;

            // Test all 8 directions at distance 1 from each corner
            for (dir_idx, (dr, dc)) in DIRECTIONS.iter().enumerate() {
                let to_row = from_row + dr;
                let to_col = from_col + dc;
                if to_row < 0 || to_row > 8 || to_col < 0 || to_col > 8 {
                    continue; // Off-board, skip
                }
                let to = Square::from_row_col(to_row as u8, to_col as u8).unwrap();
                let mv = Move::Board { from, to, promote: false };
                let idx = trait_encode(&m, mv, Color::Black);
                let decoded = trait_decode(&m, idx, Color::Black).expect("decode failed");
                assert_eq!(
                    decoded, mv,
                    "Corner {} dir {} roundtrip failed", corner, dir_idx
                );
            }
        }
    }

    #[test]
    fn test_knight_from_edge_column() {
        let m = mapper();
        // Knight at col 0: can only jump to (row-2, col+1), not (row-2, col-1)
        let from = Square::from_row_col(4, 0).unwrap();
        let to_right = Square::from_row_col(2, 1).unwrap();
        let mv = Move::Board { from, to: to_right, promote: false };
        let idx = trait_encode(&m, mv, Color::Black);
        let decoded = trait_decode(&m, idx, Color::Black).expect("decode failed");
        assert_eq!(decoded, mv, "Knight from col 0 should encode/decode correctly");

        // Knight at col 8: can only jump to (row-2, col-1), not (row-2, col+1)
        let from = Square::from_row_col(4, 8).unwrap();
        let to_left = Square::from_row_col(2, 7).unwrap();
        let mv = Move::Board { from, to: to_left, promote: false };
        let idx = trait_encode(&m, mv, Color::Black);
        let decoded = trait_decode(&m, idx, Color::Black).expect("decode failed");
        assert_eq!(decoded, mv, "Knight from col 8 should encode/decode correctly");
    }

    #[test]
    fn test_decode_knight_off_board() {
        let m = mapper();
        // Knight on row 0 col 0: both knight destinations (row -2, col ±1) are off-board
        // Slot 128 = left knight no promote from sq 0
        let idx = 0 * SPATIAL_MOVE_TYPES + 128;
        assert!(trait_decode(&m, idx, Color::Black).is_err(), "Knight off-board should fail");

        // Slot 130 = right knight no promote from sq 0
        let idx = 0 * SPATIAL_MOVE_TYPES + 130;
        assert!(trait_decode(&m, idx, Color::Black).is_err(), "Knight off-board should fail");
    }

    #[test]
    fn test_no_index_collisions_drops_white() {
        let m = mapper();
        let mut seen = HashSet::new();
        for to_idx in 0u8..81 {
            for &pt in &HandPieceType::ALL {
                let mv = Move::Drop { to: Square::new_unchecked(to_idx), piece_type: pt };
                let idx = trait_encode(&m, mv, Color::White);
                assert!(seen.insert(idx), "White collision at index {} for drop to={} piece={:?}", idx, to_idx, pt);
            }
        }
        assert_eq!(seen.len(), 81 * 7);
    }

    #[test]
    fn test_knight_move_roundtrip_white() {
        let m = mapper();
        // White knight at absolute (6,4) moves forward to (4,3) and (4,5)
        let from = Square::from_row_col(6, 4).unwrap();
        let to_left = Square::from_row_col(4, 3).unwrap();
        let to_right = Square::from_row_col(4, 5).unwrap();
        for (to, side) in [(to_left, "left"), (to_right, "right")] {
            for promote in [false, true] {
                let mv = Move::Board { from, to, promote };
                let idx = trait_encode(&m, mv, Color::White);
                let decoded = trait_decode(&m, idx, Color::White).expect("decode failed");
                assert_eq!(decoded, mv, "White knight roundtrip failed: {}, promote={}", side, promote);
            }
        }
    }

    /// Verify HandPieceType::index() is the inverse of HandPieceType::ALL[idx].
    /// The drop encoding relies on this invariant: encode uses piece_type.index(),
    /// decode uses HandPieceType::ALL[piece_idx]. If shogi-core ever reorders
    /// the enum variants, this test will catch it before drop encoding silently breaks.
    #[test]
    fn test_hand_piece_type_index_stability() {
        for (expected_idx, &hpt) in HandPieceType::ALL.iter().enumerate() {
            assert_eq!(
                hpt.index(), expected_idx,
                "HandPieceType::{:?}.index() should be {}, got {}",
                hpt, expected_idx, hpt.index()
            );
            assert_eq!(
                HandPieceType::ALL[hpt.index()], hpt,
                "HandPieceType::ALL[{:?}.index()] should round-trip to {:?}",
                hpt, hpt
            );
        }
        assert_eq!(HandPieceType::ALL.len(), 7, "Should have exactly 7 droppable piece types");
    }

    /// Round-trip all legal moves from the starting position.
    #[test]
    fn test_startpos_legal_moves_roundtrip() {
        let m = mapper();
        let mut state = shogi_core::GameState::new();
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
