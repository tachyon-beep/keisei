//! Position — board state, hands, Zobrist hash, and derived queries.

use std::fmt;

use crate::piece::Piece;
use crate::types::{Color, HandPieceType, PieceType, Square};
use crate::zobrist::ZOBRIST;

// ---------------------------------------------------------------------------
// Hands type alias
// ---------------------------------------------------------------------------

/// Per-player hand counts: `[color_index][hpt.index()]`.
pub type Hands = [[u8; HandPieceType::COUNT]; 2];

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

/// Complete board position: board array, hands, side-to-move, and Zobrist hash.
#[derive(Clone, PartialEq, Eq)]
pub struct Position {
    /// Piece occupancy. 0 = empty, non-zero = `Piece::to_u8()`.
    pub board: [u8; Square::NUM_SQUARES],
    /// Captured pieces available to drop.
    pub hands: Hands,
    /// The player whose turn it is.
    pub current_player: Color,
    /// Incremental Zobrist hash of the full position.
    pub hash: u64,
}

impl Position {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// An empty board with no pieces, Black to move, and hash = 0.
    pub fn empty() -> Position {
        Position {
            board: [0u8; Square::NUM_SQUARES],
            hands: [[0u8; HandPieceType::COUNT]; 2],
            current_player: Color::Black,
            hash: 0,
        }
    }

    /// Standard Shogi starting position.
    ///
    /// Layout (row 0 = top = White's back rank, row 8 = bottom = Black's back rank):
    ///
    /// ```text
    /// row 0: W-L  W-N  W-S  W-G  W-K  W-G  W-S  W-N  W-L   (col 0..8)
    /// row 1:      W-B                   W-R
    /// row 2: W-P  W-P  W-P  W-P  W-P  W-P  W-P  W-P  W-P
    /// row 3-5: (empty)
    /// row 6: B-P  B-P  B-P  B-P  B-P  B-P  B-P  B-P  B-P
    /// row 7:      B-R                   B-B
    /// row 8: B-L  B-N  B-S  B-G  B-K  B-G  B-S  B-N  B-L
    /// ```
    pub fn startpos() -> Position {
        let mut pos = Position::empty();

        // Helper closure — places a piece and skips hashing (done at end).
        let mut place = |row: u8, col: u8, pt: PieceType, color: Color| {
            let sq = Square::from_row_col(row, col).expect("startpos: bad square");
            let piece = Piece::new(pt, color, false);
            pos.board[sq.index()] = piece.to_u8();
        };

        // --- Row 0: White back rank ---
        let back_rank = [
            PieceType::Lance,
            PieceType::Knight,
            PieceType::Silver,
            PieceType::Gold,
            PieceType::King,
            PieceType::Gold,
            PieceType::Silver,
            PieceType::Knight,
            PieceType::Lance,
        ];
        for (col, &pt) in back_rank.iter().enumerate() {
            place(0, col as u8, pt, Color::White);
        }

        // --- Row 1: White Rook (col 1) and Bishop (col 7) ---
        // SFEN row 1: "1r5b1" => Rook at col 1, Bishop at col 7
        place(1, 1, PieceType::Rook,   Color::White);
        place(1, 7, PieceType::Bishop, Color::White);

        // --- Row 2: White Pawns ---
        for col in 0u8..9 {
            place(2, col, PieceType::Pawn, Color::White);
        }

        // --- Row 6: Black Pawns ---
        for col in 0u8..9 {
            place(6, col, PieceType::Pawn, Color::Black);
        }

        // --- Row 7: Black Bishop (col 1) and Rook (col 7) ---
        // SFEN row 7: "1B5R1" => Bishop at col 1, Rook at col 7
        place(7, 1, PieceType::Bishop, Color::Black);
        place(7, 7, PieceType::Rook,   Color::Black);

        // --- Row 8: Black back rank ---
        for (col, &pt) in back_rank.iter().enumerate() {
            place(8, col as u8, pt, Color::Black);
        }

        // Compute hash from scratch and store it.
        pos.hash = pos.compute_hash();
        pos
    }

    // -----------------------------------------------------------------------
    // Board accessors
    // -----------------------------------------------------------------------

    /// Return the piece on `sq`, or `None` if the square is empty.
    #[inline]
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        Piece::from_u8(self.board[sq.index()])
    }

    /// Place `piece` on `sq` (overwrites whatever was there).
    #[inline]
    pub fn set_piece(&mut self, sq: Square, piece: Piece) {
        self.board[sq.index()] = piece.to_u8();
    }

    /// Remove any piece from `sq`.
    #[inline]
    pub fn clear_square(&mut self, sq: Square) {
        self.board[sq.index()] = 0;
    }

    // -----------------------------------------------------------------------
    // Hand accessors
    // -----------------------------------------------------------------------

    /// Return how many pieces of type `hpt` player `color` holds in hand.
    #[inline]
    pub fn hand_count(&self, color: Color, hpt: HandPieceType) -> u8 {
        self.hands[color as usize][hpt.index()]
    }

    /// Set the hand count for `color`'s `hpt` to `count`.
    #[inline]
    pub fn set_hand_count(&mut self, color: Color, hpt: HandPieceType, count: u8) {
        self.hands[color as usize][hpt.index()] = count;
    }

    // -----------------------------------------------------------------------
    // Zobrist hash
    // -----------------------------------------------------------------------

    /// Compute the full Zobrist hash from scratch.
    ///
    /// XORs in every piece on the board, every non-zero hand count, and the
    /// side-to-move token when it is White's turn.
    pub fn compute_hash(&self) -> u64 {
        let z = &*ZOBRIST;
        let mut h: u64 = 0;

        // Board pieces.
        for (idx, &raw) in self.board.iter().enumerate() {
            if raw != 0 {
                let sq = Square::new_unchecked(idx as u8);
                // SAFETY: raw != 0 so from_u8 returns Some.
                let piece = Piece::from_u8(raw).expect("non-zero board byte must decode");
                h ^= z.hash_piece_at(sq, piece);
            }
        }

        // Hand counts (count 0 contributes nothing per the table design).
        for color in [Color::Black, Color::White] {
            for &hpt in &HandPieceType::ALL {
                let count = self.hand_count(color, hpt);
                if count > 0 {
                    h ^= z.hash_hand(color, hpt, count);
                }
            }
        }

        // Side to move.
        if self.current_player == Color::White {
            h ^= z.side_to_move;
        }

        h
    }

    // -----------------------------------------------------------------------
    // King location
    // -----------------------------------------------------------------------

    /// Find the square of `color`'s King by linear scan.
    pub fn find_king(&self, color: Color) -> Option<Square> {
        let king_piece = Piece::new(PieceType::King, color, false);
        let target = king_piece.to_u8();
        for (idx, &raw) in self.board.iter().enumerate() {
            if raw == target {
                return Some(Square::new_unchecked(idx as u8));
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Position [{:?} to move, hash={:#018x}]", self.current_player, self.hash)?;

        // Board — 9 rows × 9 columns
        for row in 0u8..9 {
            write!(f, "  ")?;
            for col in 0u8..9 {
                let sq = Square::new_unchecked(row * 9 + col);
                match self.piece_at(sq) {
                    None => write!(f, "  .  ")?,
                    Some(p) => {
                        let color_char = if p.color() == Color::Black { 'b' } else { 'w' };
                        let prom = if p.is_promoted() { '+' } else { ' ' };
                        let pt_str = match p.piece_type() {
                            PieceType::Pawn   => "P",
                            PieceType::Lance  => "L",
                            PieceType::Knight => "N",
                            PieceType::Silver => "S",
                            PieceType::Gold   => "G",
                            PieceType::Bishop => "B",
                            PieceType::Rook   => "R",
                            PieceType::King   => "K",
                        };
                        write!(f, "{}{}{} ", color_char, pt_str, prom)?;
                    }
                }
            }
            writeln!(f)?;
        }

        // Hands
        write!(f, "  Black hand:")?;
        for &hpt in &HandPieceType::ALL {
            let n = self.hand_count(Color::Black, hpt);
            if n > 0 {
                write!(f, " {:?}×{}", hpt, n)?;
            }
        }
        writeln!(f)?;

        write!(f, "  White hand:")?;
        for &hpt in &HandPieceType::ALL {
            let n = self.hand_count(Color::White, hpt);
            if n > 0 {
                write!(f, " {:?}×{}", hpt, n)?;
            }
        }
        writeln!(f)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos_piece_count() {
        let pos = Position::startpos();
        let count = pos.board.iter().filter(|&&b| b != 0).count();
        assert_eq!(count, 40, "startpos must have exactly 40 pieces on the board");
    }

    #[test]
    fn test_startpos_kings() {
        let pos = Position::startpos();

        let black_king_sq = pos.find_king(Color::Black).expect("black king not found");
        let expected_black = Square::from_row_col(8, 4).unwrap();
        assert_eq!(black_king_sq, expected_black, "Black king should be at (8,4)");

        let white_king_sq = pos.find_king(Color::White).expect("white king not found");
        let expected_white = Square::from_row_col(0, 4).unwrap();
        assert_eq!(white_king_sq, expected_white, "White king should be at (0,4)");
    }

    #[test]
    fn test_startpos_pawns() {
        let pos = Position::startpos();

        // Black pawns on row 6, all 9 columns.
        for col in 0u8..9 {
            let sq = Square::from_row_col(6, col).unwrap();
            let piece = pos.piece_at(sq).expect(&format!("no Black pawn at (6,{})", col));
            assert_eq!(piece.piece_type(), PieceType::Pawn);
            assert_eq!(piece.color(), Color::Black);
            assert!(!piece.is_promoted());
        }

        // White pawns on row 2, all 9 columns.
        for col in 0u8..9 {
            let sq = Square::from_row_col(2, col).unwrap();
            let piece = pos.piece_at(sq).expect(&format!("no White pawn at (2,{})", col));
            assert_eq!(piece.piece_type(), PieceType::Pawn);
            assert_eq!(piece.color(), Color::White);
            assert!(!piece.is_promoted());
        }
    }

    #[test]
    fn test_startpos_hash_nonzero() {
        let pos = Position::startpos();
        assert_ne!(pos.hash, 0, "startpos hash must be non-zero");
    }

    #[test]
    fn test_hash_recomputation_matches() {
        let pos = Position::startpos();
        assert_eq!(
            pos.hash,
            pos.compute_hash(),
            "stored hash must equal freshly computed hash"
        );
    }

    #[test]
    fn test_startpos_black_to_move() {
        let pos = Position::startpos();
        assert_eq!(pos.current_player, Color::Black);
    }

    #[test]
    fn test_empty_hands() {
        let pos = Position::startpos();
        for color in [Color::Black, Color::White] {
            for &hpt in &HandPieceType::ALL {
                assert_eq!(
                    pos.hand_count(color, hpt),
                    0,
                    "startpos hand count for {:?}/{:?} must be 0",
                    color,
                    hpt
                );
            }
        }
    }
}
