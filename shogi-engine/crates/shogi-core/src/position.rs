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

    // -----------------------------------------------------------------------
    // Hash sensitivity: board mutation must change hash
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_hash_changes_on_board_mutation() {
        let pos = Position::startpos();
        let orig_hash = pos.hash;

        // Move a pawn off the board — hash must change
        let mut mutated = pos.clone();
        let sq = Square::from_row_col(6, 0).unwrap();
        mutated.clear_square(sq);
        mutated.hash = mutated.compute_hash();
        assert_ne!(
            orig_hash, mutated.hash,
            "removing a piece must change the hash"
        );
    }

    #[test]
    fn test_compute_hash_changes_with_side_to_move() {
        let pos = Position::startpos();
        assert_eq!(pos.current_player, Color::Black);

        let mut flipped = pos.clone();
        flipped.current_player = Color::White;
        flipped.hash = flipped.compute_hash();
        assert_ne!(
            pos.hash, flipped.hash,
            "flipping side to move must change the hash (White XOR branch)"
        );
    }

    #[test]
    fn test_compute_hash_changes_on_hand_mutation() {
        let pos = Position::startpos();

        let mut mutated = pos.clone();
        mutated.set_hand_count(Color::Black, HandPieceType::Pawn, 1);
        mutated.hash = mutated.compute_hash();
        assert_ne!(
            pos.hash, mutated.hash,
            "adding a hand piece must change the hash"
        );
    }

    // -----------------------------------------------------------------------
    // set_piece / clear_square do NOT modify hash (caller's job)
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_clear_do_not_modify_hash() {
        let mut pos = Position::startpos();
        let hash_before = pos.hash;

        let sq = Square::from_row_col(6, 0).unwrap();
        pos.clear_square(sq);
        assert_eq!(
            pos.hash, hash_before,
            "clear_square must not change hash (caller updates it)"
        );

        let pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        pos.set_piece(sq, pawn);
        assert_eq!(
            pos.hash, hash_before,
            "set_piece must not change hash (caller updates it)"
        );
    }

    // -----------------------------------------------------------------------
    // find_king returns None when king is absent
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_king_returns_none_when_absent() {
        let pos = Position::empty();
        assert_eq!(pos.find_king(Color::Black), None);
        assert_eq!(pos.find_king(Color::White), None);
    }

    // -----------------------------------------------------------------------
    // hand_count / set_hand_count roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_hand_count_set_roundtrip() {
        let mut pos = Position::empty();
        for &color in &[Color::Black, Color::White] {
            for &hpt in &HandPieceType::ALL {
                pos.set_hand_count(color, hpt, 5);
                assert_eq!(
                    pos.hand_count(color, hpt), 5,
                    "hand_count roundtrip failed for {:?}/{:?}",
                    color, hpt
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Debug formatting doesn't panic
    // -----------------------------------------------------------------------

    #[test]
    fn test_debug_formatting() {
        let pos = Position::startpos();
        let debug_str = format!("{:?}", pos);
        assert!(debug_str.contains("Position"), "Debug output should contain 'Position'");
        assert!(debug_str.contains("Black"), "Debug output should mention a color");
    }

    // ===================================================================
    // Gap #4: Mutation edge case tests
    // ===================================================================

    /// set_piece on an occupied square overwrites silently.
    #[test]
    fn test_set_piece_overwrites_occupied_square() {
        let mut pos = Position::startpos();
        let sq = Square::from_row_col(6, 0).unwrap(); // Black pawn
        let original = pos.piece_at(sq).unwrap();
        assert_eq!(original.piece_type(), PieceType::Pawn);
        assert_eq!(original.color(), Color::Black);

        // Overwrite with a White rook
        let new_piece = Piece::new(PieceType::Rook, Color::White, false);
        pos.set_piece(sq, new_piece);
        assert_eq!(pos.piece_at(sq), Some(new_piece));
    }

    /// clear_square on an already-empty square is a no-op.
    #[test]
    fn test_clear_empty_square_is_noop() {
        let mut pos = Position::startpos();
        let sq = Square::from_row_col(4, 4).unwrap(); // center, empty at start
        assert!(pos.piece_at(sq).is_none());

        let board_before = pos.board;
        pos.clear_square(sq);
        assert_eq!(pos.board, board_before, "clearing an empty square should not change the board");
    }

    /// set_piece followed by clear_square restores the square to empty.
    #[test]
    fn test_set_then_clear_restores_empty() {
        let mut pos = Position::empty();
        let sq = Square::from_row_col(4, 4).unwrap();
        let piece = Piece::new(PieceType::Bishop, Color::Black, false);

        pos.set_piece(sq, piece);
        assert_eq!(pos.piece_at(sq), Some(piece));

        pos.clear_square(sq);
        assert!(pos.piece_at(sq).is_none());
    }

    /// set_hand_count with max plausible value (18 pawns) works correctly.
    #[test]
    fn test_set_hand_count_max_value() {
        let mut pos = Position::empty();
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 18);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Pawn), 18);
    }

    /// set_hand_count to 0 and back up.
    #[test]
    fn test_set_hand_count_zero_and_back() {
        let mut pos = Position::empty();
        pos.set_hand_count(Color::Black, HandPieceType::Rook, 2);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Rook), 2);

        pos.set_hand_count(Color::Black, HandPieceType::Rook, 0);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Rook), 0);

        pos.set_hand_count(Color::Black, HandPieceType::Rook, 1);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Rook), 1);
    }

    /// find_king exhaustive: every board square can hold a king.
    #[test]
    fn test_find_king_every_square() {
        for idx in 0u8..81 {
            let sq = Square::new(idx).unwrap();
            let mut pos = Position::empty();
            pos.set_piece(sq, Piece::new(PieceType::King, Color::Black, false));
            assert_eq!(pos.find_king(Color::Black), Some(sq), "find_king failed for square index {}", idx);
        }
    }

    /// piece_at returns correct value for all piece types on all squares.
    #[test]
    fn test_piece_at_all_types() {
        let types = [
            PieceType::Pawn, PieceType::Lance, PieceType::Knight,
            PieceType::Silver, PieceType::Gold, PieceType::Bishop,
            PieceType::Rook, PieceType::King,
        ];
        for &pt in &types {
            for &color in &[Color::Black, Color::White] {
                for &promoted in &[false, true] {
                    if promoted && !pt.can_promote() {
                        continue;
                    }
                    let piece = Piece::new(pt, color, promoted);
                    let mut pos = Position::empty();
                    let sq = Square::from_row_col(4, 4).unwrap();
                    pos.set_piece(sq, piece);
                    let read_back = pos.piece_at(sq).unwrap();
                    assert_eq!(read_back.piece_type(), pt);
                    assert_eq!(read_back.color(), color);
                    assert_eq!(read_back.is_promoted(), promoted);
                }
            }
        }
    }

    /// compute_hash is different for positions that differ only in hand counts.
    #[test]
    fn test_compute_hash_sensitive_to_different_hand_piece_types() {
        let mut pos1 = Position::empty();
        pos1.set_hand_count(Color::Black, HandPieceType::Pawn, 1);
        pos1.hash = pos1.compute_hash();

        let mut pos2 = Position::empty();
        pos2.set_hand_count(Color::Black, HandPieceType::Lance, 1);
        pos2.hash = pos2.compute_hash();

        assert_ne!(
            pos1.hash, pos2.hash,
            "Different hand piece types should produce different hashes"
        );
    }
}
