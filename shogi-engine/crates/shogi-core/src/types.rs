use std::fmt;

// ---------------------------------------------------------------------------
// ShogiError — declared first so other types can reference it
// ---------------------------------------------------------------------------

/// Top-level error type for the shogi-core crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShogiError {
    InvalidSfen(String),
    IllegalMove(Move),
    InvalidSquare(u8),
    GameOver(GameResult),
}

impl fmt::Display for ShogiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShogiError::InvalidSfen(s) => write!(f, "invalid SFEN: {}", s),
            ShogiError::IllegalMove(m) => write!(f, "illegal move: {:?}", m),
            ShogiError::InvalidSquare(idx) => write!(f, "invalid square index: {}", idx),
            ShogiError::GameOver(result) => write!(f, "game is over: {:?}", result),
        }
    }
}

impl std::error::Error for ShogiError {}

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Color {
    Black,
    White,
}

impl Color {
    pub fn opponent(self) -> Color {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

// ---------------------------------------------------------------------------
// PieceType
// ---------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceType {
    Pawn   = 1,
    Lance  = 2,
    Knight = 3,
    Silver = 4,
    Gold   = 5,
    Bishop = 6,
    Rook   = 7,
    King   = 8,
}

impl PieceType {
    pub const COUNT: usize = 8;

    pub fn from_u8(val: u8) -> Option<PieceType> {
        match val {
            1 => Some(PieceType::Pawn),
            2 => Some(PieceType::Lance),
            3 => Some(PieceType::Knight),
            4 => Some(PieceType::Silver),
            5 => Some(PieceType::Gold),
            6 => Some(PieceType::Bishop),
            7 => Some(PieceType::Rook),
            8 => Some(PieceType::King),
            _ => None,
        }
    }

    pub fn can_promote(self) -> bool {
        matches!(
            self,
            PieceType::Pawn
                | PieceType::Lance
                | PieceType::Knight
                | PieceType::Silver
                | PieceType::Bishop
                | PieceType::Rook
        )
    }
}

// ---------------------------------------------------------------------------
// HandPieceType
// ---------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HandPieceType {
    Pawn   = 1,
    Lance  = 2,
    Knight = 3,
    Silver = 4,
    Gold   = 5,
    Bishop = 6,
    Rook   = 7,
}

impl HandPieceType {
    pub const COUNT: usize = 7;

    pub const ALL: [HandPieceType; 7] = [
        HandPieceType::Pawn,
        HandPieceType::Lance,
        HandPieceType::Knight,
        HandPieceType::Silver,
        HandPieceType::Gold,
        HandPieceType::Bishop,
        HandPieceType::Rook,
    ];

    pub fn to_piece_type(self) -> PieceType {
        match self {
            HandPieceType::Pawn   => PieceType::Pawn,
            HandPieceType::Lance  => PieceType::Lance,
            HandPieceType::Knight => PieceType::Knight,
            HandPieceType::Silver => PieceType::Silver,
            HandPieceType::Gold   => PieceType::Gold,
            HandPieceType::Bishop => PieceType::Bishop,
            HandPieceType::Rook   => PieceType::Rook,
        }
    }

    pub fn from_piece_type(pt: PieceType) -> Option<HandPieceType> {
        match pt {
            PieceType::Pawn   => Some(HandPieceType::Pawn),
            PieceType::Lance  => Some(HandPieceType::Lance),
            PieceType::Knight => Some(HandPieceType::Knight),
            PieceType::Silver => Some(HandPieceType::Silver),
            PieceType::Gold   => Some(HandPieceType::Gold),
            PieceType::Bishop => Some(HandPieceType::Bishop),
            PieceType::Rook   => Some(HandPieceType::Rook),
            PieceType::King   => None,
        }
    }

    /// Zero-based index suitable for array indexing (0..7).
    pub fn index(self) -> usize {
        (self as u8 - 1) as usize
    }
}

// ---------------------------------------------------------------------------
// Square
// ---------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Square(u8);

impl Square {
    pub const NUM_SQUARES: usize = 81;

    pub fn new(index: u8) -> Result<Square, ShogiError> {
        if index < 81 {
            Ok(Square(index))
        } else {
            Err(ShogiError::InvalidSquare(index))
        }
    }

    pub fn from_row_col(row: u8, col: u8) -> Result<Square, ShogiError> {
        if row < 9 && col < 9 {
            Ok(Square(row * 9 + col))
        } else {
            // Report whichever coordinate is bad (row takes priority)
            let bad = if row >= 9 { row } else { col };
            Err(ShogiError::InvalidSquare(bad))
        }
    }

    /// Construct a Square without bounds checking. Only a `debug_assert` guards it.
    #[inline]
    pub fn new_unchecked(index: u8) -> Square {
        debug_assert!(index < 81, "Square index {} is out of range", index);
        Square(index)
    }

    #[inline]
    pub fn row(self) -> u8 {
        self.0 / 9
    }

    #[inline]
    pub fn col(self) -> u8 {
        self.0 % 9
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// 180-degree rotation of the board.
    #[inline]
    pub fn flip(self) -> Square {
        Square(80 - self.0)
    }

    /// Apply a signed delta to the raw index, returning `None` if out of range.
    pub fn offset(self, delta: i8) -> Option<Square> {
        let new_idx = self.0 as i16 + delta as i16;
        if (0..81).contains(&new_idx) {
            Some(Square(new_idx as u8))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Move {
    Board {
        from: Square,
        to: Square,
        promote: bool,
    },
    Drop {
        to: Square,
        piece_type: HandPieceType,
    },
}

// ---------------------------------------------------------------------------
// GameResult
// ---------------------------------------------------------------------------

/// All decisive results use `winner` for semantic consistency.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GameResult {
    InProgress,
    Checkmate { winner: Color },
    Impasse { winner: Option<Color> },
    Repetition,
    PerpetualCheck { winner: Color },
    MaxMoves,
}

impl GameResult {
    pub fn is_terminal(self) -> bool {
        !matches!(self, GameResult::InProgress)
    }

    pub fn is_truncation(self) -> bool {
        matches!(self, GameResult::MaxMoves)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Color ---------------------------------------------------------------

    #[test]
    fn test_color_opponent() {
        assert_eq!(Color::Black.opponent(), Color::White);
        assert_eq!(Color::White.opponent(), Color::Black);
    }

    #[test]
    fn test_color_opponent_roundtrip() {
        assert_eq!(Color::Black.opponent().opponent(), Color::Black);
        assert_eq!(Color::White.opponent().opponent(), Color::White);
    }

    // -- PieceType -----------------------------------------------------------

    #[test]
    fn test_piece_type_from_u8_valid() {
        assert_eq!(PieceType::from_u8(1), Some(PieceType::Pawn));
        assert_eq!(PieceType::from_u8(2), Some(PieceType::Lance));
        assert_eq!(PieceType::from_u8(3), Some(PieceType::Knight));
        assert_eq!(PieceType::from_u8(4), Some(PieceType::Silver));
        assert_eq!(PieceType::from_u8(5), Some(PieceType::Gold));
        assert_eq!(PieceType::from_u8(6), Some(PieceType::Bishop));
        assert_eq!(PieceType::from_u8(7), Some(PieceType::Rook));
        assert_eq!(PieceType::from_u8(8), Some(PieceType::King));
    }

    #[test]
    fn test_piece_type_from_u8_invalid() {
        assert_eq!(PieceType::from_u8(0), None);
        assert_eq!(PieceType::from_u8(9), None);
        assert_eq!(PieceType::from_u8(255), None);
    }

    // -- HandPieceType -------------------------------------------------------

    #[test]
    fn test_hand_piece_type_roundtrip() {
        for &hpt in &HandPieceType::ALL {
            let pt = hpt.to_piece_type();
            let back = HandPieceType::from_piece_type(pt);
            assert_eq!(back, Some(hpt));
        }
    }

    #[test]
    fn test_king_not_hand_piece() {
        assert_eq!(HandPieceType::from_piece_type(PieceType::King), None);
    }

    #[test]
    fn test_hand_piece_index() {
        // Indices must be 0..=6, each unique
        let mut seen = [false; 7];
        for &hpt in &HandPieceType::ALL {
            let idx = hpt.index();
            assert!(idx < 7, "index {} out of range for {:?}", idx, hpt);
            assert!(!seen[idx], "duplicate index {} for {:?}", idx, hpt);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&v| v), "not all indices 0-6 covered");
    }

    // -- can_promote ---------------------------------------------------------

    #[test]
    fn test_can_promote() {
        assert!(PieceType::Pawn.can_promote());
        assert!(PieceType::Lance.can_promote());
        assert!(PieceType::Knight.can_promote());
        assert!(PieceType::Silver.can_promote());
        assert!(PieceType::Bishop.can_promote());
        assert!(PieceType::Rook.can_promote());
        assert!(!PieceType::Gold.can_promote());
        assert!(!PieceType::King.can_promote());
    }

    // -- Square --------------------------------------------------------------

    #[test]
    fn test_square_row_col() {
        let sq = Square::from_row_col(0, 0).unwrap();
        assert_eq!(sq.row(), 0);
        assert_eq!(sq.col(), 0);

        let sq = Square::from_row_col(8, 8).unwrap();
        assert_eq!(sq.row(), 8);
        assert_eq!(sq.col(), 8);

        let sq = Square::from_row_col(3, 5).unwrap();
        assert_eq!(sq.row(), 3);
        assert_eq!(sq.col(), 5);
        assert_eq!(sq.index(), 3 * 9 + 5);
    }

    #[test]
    fn test_square_flip() {
        let sq = Square::new(0).unwrap();
        assert_eq!(sq.flip().index(), 80);

        let sq = Square::new(80).unwrap();
        assert_eq!(sq.flip().index(), 0);

        let sq = Square::new(40).unwrap();
        assert_eq!(sq.flip().index(), 40);
    }

    #[test]
    fn test_square_flip_roundtrip() {
        for i in 0u8..81 {
            let sq = Square::new(i).unwrap();
            assert_eq!(sq.flip().flip(), sq);
        }
    }

    #[test]
    fn test_square_invalid() {
        assert!(Square::new(81).is_err());
        assert!(Square::new(255).is_err());
        assert!(Square::from_row_col(9, 0).is_err());
        assert!(Square::from_row_col(0, 9).is_err());
    }

    #[test]
    fn test_square_offset() {
        let sq = Square::new(40).unwrap();
        assert_eq!(sq.offset(0), Some(sq));
        assert_eq!(sq.offset(1), Some(Square::new(41).unwrap()));
        assert_eq!(sq.offset(-1), Some(Square::new(39).unwrap()));

        // Boundary: index 0, delta -1 => None
        let sq0 = Square::new(0).unwrap();
        assert_eq!(sq0.offset(-1), None);

        // Boundary: index 80, delta +1 => None
        let sq80 = Square::new(80).unwrap();
        assert_eq!(sq80.offset(1), None);
    }

    // -- GameResult ----------------------------------------------------------

    #[test]
    fn test_game_result_is_terminal() {
        assert!(!GameResult::InProgress.is_terminal());
        assert!(GameResult::Checkmate { winner: Color::Black }.is_terminal());
        assert!(GameResult::Impasse { winner: None }.is_terminal());
        assert!(GameResult::Repetition.is_terminal());
        assert!(GameResult::PerpetualCheck { winner: Color::White }.is_terminal());
        assert!(GameResult::MaxMoves.is_terminal());
    }

    #[test]
    fn test_game_result_is_truncation() {
        assert!(!GameResult::InProgress.is_truncation());
        assert!(!GameResult::Checkmate { winner: Color::Black }.is_truncation());
        assert!(!GameResult::Repetition.is_truncation());
        assert!(GameResult::MaxMoves.is_truncation());
    }

    #[test]
    fn test_game_result_is_truncation_exhaustive() {
        // Ensure ALL non-MaxMoves variants return false
        assert!(!GameResult::PerpetualCheck { winner: Color::Black }.is_truncation());
        assert!(!GameResult::PerpetualCheck { winner: Color::White }.is_truncation());
        assert!(!GameResult::Impasse { winner: None }.is_truncation());
        assert!(!GameResult::Impasse { winner: Some(Color::Black) }.is_truncation());
        assert!(!GameResult::Impasse { winner: Some(Color::White) }.is_truncation());
        assert!(!GameResult::Checkmate { winner: Color::White }.is_truncation());
    }

    // -- ShogiError Display ---------------------------------------------------

    #[test]
    fn test_shogi_error_display() {
        let e1 = ShogiError::InvalidSfen("bad data".into());
        assert_eq!(format!("{}", e1), "invalid SFEN: bad data");

        let e2 = ShogiError::InvalidSquare(99);
        assert_eq!(format!("{}", e2), "invalid square index: 99");

        let e3 = ShogiError::GameOver(GameResult::MaxMoves);
        let s = format!("{}", e3);
        assert!(s.contains("game is over"), "GameOver display should contain 'game is over': {}", s);

        let e4 = ShogiError::IllegalMove(Move::Drop {
            to: Square::new(40).unwrap(),
            piece_type: HandPieceType::Pawn,
        });
        let s = format!("{}", e4);
        assert!(s.contains("illegal move"), "IllegalMove display: {}", s);
    }

    #[test]
    fn test_shogi_error_is_std_error() {
        // Verify ShogiError implements std::error::Error
        let e: Box<dyn std::error::Error> = Box::new(ShogiError::InvalidSquare(99));
        assert!(e.to_string().contains("99"));
    }

    // -- Move: compile-time check that Drop has no King ----------------------
    // There is no King variant in HandPieceType, so this is enforced by the
    // type system at compile time. The test below simply verifies construction.
    #[test]
    fn test_move_drop_no_king() {
        // This must compile — if HandPieceType had a King variant we couldn't
        // accidentally use it here.
        let _m = Move::Drop {
            to: Square::new(40).unwrap(),
            piece_type: HandPieceType::Rook,
        };
    }

    // ===================================================================
    // Gap #5: Exhaustive Square boundary tests
    // ===================================================================

    /// Exhaustive roundtrip: from_row_col → row()/col() for all 81 valid squares.
    #[test]
    fn test_square_from_row_col_exhaustive_roundtrip() {
        for row in 0u8..9 {
            for col in 0u8..9 {
                let sq = Square::from_row_col(row, col)
                    .unwrap_or_else(|_| panic!("from_row_col({},{}) should succeed", row, col));
                assert_eq!(sq.row(), row, "row mismatch for ({},{})", row, col);
                assert_eq!(sq.col(), col, "col mismatch for ({},{})", row, col);
                assert_eq!(sq.index(), (row as usize) * 9 + (col as usize));
            }
        }
    }

    /// All invalid row/col pairs are rejected.
    #[test]
    fn test_square_from_row_col_invalid_exhaustive() {
        // Row 9+ with any col
        for col in 0u8..20 {
            assert!(Square::from_row_col(9, col).is_err());
            assert!(Square::from_row_col(10, col).is_err());
            assert!(Square::from_row_col(255, col).is_err());
        }
        // Valid row, col 9+
        for row in 0u8..9 {
            assert!(Square::from_row_col(row, 9).is_err());
            assert!(Square::from_row_col(row, 10).is_err());
            assert!(Square::from_row_col(row, 255).is_err());
        }
    }

    /// Square::new accepts 0..80 and rejects 81..255.
    #[test]
    fn test_square_new_boundary() {
        // Valid: 0 through 80
        for i in 0u8..81 {
            assert!(Square::new(i).is_ok(), "Square::new({}) should be Ok", i);
        }
        // Invalid: 81 through 255
        for i in 81u8..=255 {
            assert!(Square::new(i).is_err(), "Square::new({}) should be Err", i);
        }
    }

    /// Square::new(idx).index() == idx for all valid indices.
    #[test]
    fn test_square_index_roundtrip() {
        for i in 0u8..81 {
            let sq = Square::new(i).unwrap();
            assert_eq!(sq.index(), i as usize);
        }
    }

    /// Square::offset at the boundaries of the board.
    #[test]
    fn test_square_offset_all_corners() {
        // Top-left corner (0,0)
        let tl = Square::from_row_col(0, 0).unwrap();
        assert_eq!(tl.offset(-1), None, "top-left offset -1 should be None");
        assert_eq!(tl.offset(-9), None, "top-left offset -9 should be None");
        assert!(tl.offset(1).is_some(), "top-left offset +1 should be Some");
        assert!(tl.offset(9).is_some(), "top-left offset +9 should be Some");

        // Bottom-right corner (8,8)
        let br = Square::from_row_col(8, 8).unwrap();
        assert_eq!(br.offset(1), None, "bottom-right offset +1 should be None");
        assert_eq!(br.offset(9), None, "bottom-right offset +9 should be None");
        assert!(br.offset(-1).is_some());
        assert!(br.offset(-9).is_some());

        // Top-right corner (0,8)
        let tr = Square::from_row_col(0, 8).unwrap();
        assert_eq!(tr.offset(-1), Some(Square::from_row_col(0, 7).unwrap()));
        assert_eq!(tr.offset(1), Some(Square::from_row_col(1, 0).unwrap()));

        // Bottom-left corner (8,0)
        let bl = Square::from_row_col(8, 0).unwrap();
        assert_eq!(bl.offset(-1), Some(Square::from_row_col(7, 8).unwrap()));
        assert_eq!(bl.offset(9), None);
    }

    /// PieceType::from_u8 covers the full u8 range.
    #[test]
    fn test_piece_type_from_u8_full_range() {
        let mut valid_count = 0;
        for v in 0u8..=255 {
            match PieceType::from_u8(v) {
                Some(_) => valid_count += 1,
                None => {}
            }
        }
        assert_eq!(valid_count, PieceType::COUNT, "should have exactly 8 valid PieceType values");
    }

    /// HandPieceType::ALL has exactly COUNT elements, all unique.
    #[test]
    fn test_hand_piece_type_all_unique() {
        let mut seen = std::collections::HashSet::new();
        for &hpt in &HandPieceType::ALL {
            assert!(seen.insert(hpt), "duplicate in HandPieceType::ALL: {:?}", hpt);
        }
        assert_eq!(seen.len(), HandPieceType::COUNT);
    }

    /// GameResult variant exhaustiveness: every variant has correct is_terminal/is_truncation.
    #[test]
    fn test_game_result_variants_complete() {
        let variants = [
            (GameResult::InProgress, false, false),
            (GameResult::Checkmate { winner: Color::Black }, true, false),
            (GameResult::Checkmate { winner: Color::White }, true, false),
            (GameResult::Impasse { winner: None }, true, false),
            (GameResult::Impasse { winner: Some(Color::Black) }, true, false),
            (GameResult::Impasse { winner: Some(Color::White) }, true, false),
            (GameResult::Repetition, true, false),
            (GameResult::PerpetualCheck { winner: Color::Black }, true, false),
            (GameResult::PerpetualCheck { winner: Color::White }, true, false),
            (GameResult::MaxMoves, true, true),
        ];
        for (result, expected_terminal, expected_truncation) in variants {
            assert_eq!(
                result.is_terminal(), expected_terminal,
                "{:?}.is_terminal() should be {}", result, expected_terminal
            );
            assert_eq!(
                result.is_truncation(), expected_truncation,
                "{:?}.is_truncation() should be {}", result, expected_truncation
            );
        }
    }
}
