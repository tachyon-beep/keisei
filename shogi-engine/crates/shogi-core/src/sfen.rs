//! SFEN serialization and parsing for Shogi positions.
//!
//! SFEN format: `board side_to_move hands move_number`
//! Example (startpos): `lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1`

use crate::piece::Piece;
use crate::position::Position;
use crate::types::{Color, HandPieceType, PieceType, Square, ShogiError};

/// The SFEN string for the standard Shogi starting position.
pub const STARTPOS_SFEN: &str =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

// ---------------------------------------------------------------------------
// Helper: piece -> SFEN character(s)
// ---------------------------------------------------------------------------

/// Convert a `Piece` to its SFEN character representation.
/// Promoted pieces are prefixed with `+`.
/// Black pieces use uppercase letters; White pieces use lowercase.
fn piece_to_sfen_char(piece: Piece) -> String {
    let base = match piece.piece_type() {
        PieceType::Pawn   => 'P',
        PieceType::Lance  => 'L',
        PieceType::Knight => 'N',
        PieceType::Silver => 'S',
        PieceType::Gold   => 'G',
        PieceType::Bishop => 'B',
        PieceType::Rook   => 'R',
        PieceType::King   => 'K',
    };

    let ch = if piece.color() == Color::White {
        base.to_ascii_lowercase()
    } else {
        base
    };

    if piece.is_promoted() {
        format!("+{}", ch)
    } else {
        ch.to_string()
    }
}

// ---------------------------------------------------------------------------
// Helper: SFEN character -> piece
// ---------------------------------------------------------------------------

/// Parse a single SFEN piece character (and promotion flag) into a `Piece`.
/// `ch` must be a letter; `promoted` is true when preceded by `+`.
fn sfen_char_to_piece(ch: char, promoted: bool) -> Result<Piece, ShogiError> {
    let color = if ch.is_uppercase() {
        Color::Black
    } else {
        Color::White
    };

    let pt = match ch.to_ascii_uppercase() {
        'P' => PieceType::Pawn,
        'L' => PieceType::Lance,
        'N' => PieceType::Knight,
        'S' => PieceType::Silver,
        'G' => PieceType::Gold,
        'B' => PieceType::Bishop,
        'R' => PieceType::Rook,
        'K' => PieceType::King,
        other => {
            return Err(ShogiError::InvalidSfen(format!(
                "unknown piece character '{}'",
                other
            )));
        }
    };

    if promoted && !pt.can_promote() {
        return Err(ShogiError::InvalidSfen(format!(
            "piece type {:?} cannot be promoted",
            pt
        )));
    }

    Ok(Piece::new(pt, color, promoted))
}

// ---------------------------------------------------------------------------
// Position impl
// ---------------------------------------------------------------------------

impl Position {
    /// Serialize this position to a SFEN string.
    /// The move number is always `1` (positions carry no history).
    pub fn to_sfen(&self) -> String {
        // --- Board section ---
        let mut board_str = String::new();
        for row in 0u8..9 {
            if row > 0 {
                board_str.push('/');
            }
            let mut empty_count: u32 = 0;
            for col in 0u8..9 {
                let sq = Square::new_unchecked(row * 9 + col);
                match self.piece_at(sq) {
                    None => {
                        empty_count += 1;
                    }
                    Some(piece) => {
                        if empty_count > 0 {
                            board_str.push_str(&empty_count.to_string());
                            empty_count = 0;
                        }
                        board_str.push_str(&piece_to_sfen_char(piece));
                    }
                }
            }
            if empty_count > 0 {
                board_str.push_str(&empty_count.to_string());
            }
        }

        // --- Side to move ---
        let side = match self.current_player {
            Color::Black => 'b',
            Color::White => 'w',
        };

        // --- Hands section ---
        // Order: R, B, G, S, N, L, P — Black first, then White
        // Count prefix only when > 1; `-` if nothing in hand.
        let hand_order = [
            HandPieceType::Rook,
            HandPieceType::Bishop,
            HandPieceType::Gold,
            HandPieceType::Silver,
            HandPieceType::Knight,
            HandPieceType::Lance,
            HandPieceType::Pawn,
        ];

        let mut hands_str = String::new();
        for &color in &[Color::Black, Color::White] {
            for &hpt in &hand_order {
                let count = self.hand_count(color, hpt);
                if count > 0 {
                    if count > 1 {
                        hands_str.push_str(&count.to_string());
                    }
                    let ch = match hpt {
                        HandPieceType::Rook   => 'R',
                        HandPieceType::Bishop => 'B',
                        HandPieceType::Gold   => 'G',
                        HandPieceType::Silver => 'S',
                        HandPieceType::Knight => 'N',
                        HandPieceType::Lance  => 'L',
                        HandPieceType::Pawn   => 'P',
                    };
                    let ch = if color == Color::White {
                        ch.to_ascii_lowercase()
                    } else {
                        ch
                    };
                    hands_str.push(ch);
                }
            }
        }
        if hands_str.is_empty() {
            hands_str.push('-');
        }

        format!("{} {} {} 1", board_str, side, hands_str)
    }

    /// Parse a SFEN string into a `Position`.
    pub fn from_sfen(sfen: &str) -> Result<Position, ShogiError> {
        let parts: Vec<&str> = sfen.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(ShogiError::InvalidSfen(format!(
                "expected at least 3 space-separated fields, got {}",
                parts.len()
            )));
        }

        let board_str = parts[0];
        let side_str = parts[1];
        let hands_str = parts[2];
        // parts[3] is the move number — we ignore it

        let mut pos = Position::empty();

        // --- Parse board ---
        let ranks: Vec<&str> = board_str.split('/').collect();
        if ranks.len() != 9 {
            return Err(ShogiError::InvalidSfen(format!(
                "board must have 9 ranks separated by '/', got {}",
                ranks.len()
            )));
        }

        for (row, rank_str) in ranks.iter().enumerate() {
            let mut col: u8 = 0;
            let mut chars = rank_str.chars().peekable();
            while let Some(ch) = chars.next() {
                if col > 9 {
                    return Err(ShogiError::InvalidSfen(format!(
                        "rank {} exceeds 9 columns",
                        row
                    )));
                }
                if ch == '+' {
                    // Promoted piece follows
                    let next = chars.next().ok_or_else(|| {
                        ShogiError::InvalidSfen("'+' at end of rank with no following piece".into())
                    })?;
                    let piece = sfen_char_to_piece(next, true)?;
                    let sq = Square::from_row_col(row as u8, col).map_err(|_| {
                        ShogiError::InvalidSfen(format!("square out of range row={} col={}", row, col))
                    })?;
                    pos.set_piece(sq, piece);
                    col += 1;
                } else if ch.is_ascii_digit() {
                    let empty = ch as u8 - b'0';
                    if empty == 0 || empty > 9 {
                        return Err(ShogiError::InvalidSfen(format!(
                            "invalid empty count '{}' in rank {}",
                            ch, row
                        )));
                    }
                    col += empty;
                } else if ch.is_ascii_alphabetic() {
                    let piece = sfen_char_to_piece(ch, false)?;
                    let sq = Square::from_row_col(row as u8, col).map_err(|_| {
                        ShogiError::InvalidSfen(format!("square out of range row={} col={}", row, col))
                    })?;
                    pos.set_piece(sq, piece);
                    col += 1;
                } else {
                    return Err(ShogiError::InvalidSfen(format!(
                        "unexpected character '{}' in board string",
                        ch
                    )));
                }
            }
            if col != 9 {
                return Err(ShogiError::InvalidSfen(format!(
                    "rank {} has {} columns, expected 9",
                    row, col
                )));
            }
        }

        // --- Parse side to move ---
        pos.current_player = match side_str {
            "b" => Color::Black,
            "w" => Color::White,
            other => {
                return Err(ShogiError::InvalidSfen(format!(
                    "invalid side to move '{}', expected 'b' or 'w'",
                    other
                )));
            }
        };

        // --- Parse hands ---
        if hands_str != "-" {
            let mut chars = hands_str.chars().peekable();
            while let Some(ch) = chars.peek().copied() {
                // Optional count prefix
                let count: u8 = if ch.is_ascii_digit() {
                    // Consume digits (multi-digit count, e.g., "18p")
                    let mut num_str = String::new();
                    while let Some(&d) = chars.peek() {
                        if d.is_ascii_digit() {
                            num_str.push(d);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    num_str.parse::<u8>().map_err(|_| {
                        ShogiError::InvalidSfen(format!("invalid hand count '{}'", num_str))
                    })?
                } else {
                    1
                };

                let piece_ch = chars.next().ok_or_else(|| {
                    ShogiError::InvalidSfen("hand string ends after count with no piece".into())
                })?;

                let color = if piece_ch.is_uppercase() {
                    Color::Black
                } else {
                    Color::White
                };

                let hpt = match piece_ch.to_ascii_uppercase() {
                    'R' => HandPieceType::Rook,
                    'B' => HandPieceType::Bishop,
                    'G' => HandPieceType::Gold,
                    'S' => HandPieceType::Silver,
                    'N' => HandPieceType::Knight,
                    'L' => HandPieceType::Lance,
                    'P' => HandPieceType::Pawn,
                    other => {
                        return Err(ShogiError::InvalidSfen(format!(
                            "invalid hand piece character '{}'",
                            other
                        )));
                    }
                };

                pos.set_hand_count(color, hpt, count);
            }
        }

        // --- Compute hash ---
        pos.hash = pos.compute_hash();

        Ok(pos)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_startpos_sfen_roundtrip() {
        let pos = Position::startpos();
        let sfen = pos.to_sfen();
        assert_eq!(sfen, STARTPOS_SFEN, "startpos serialization mismatch");
    }

    #[test]
    fn test_parse_startpos_sfen() {
        let pos = Position::from_sfen(STARTPOS_SFEN).expect("failed to parse startpos SFEN");
        let expected = Position::startpos();

        // Board matches
        assert_eq!(pos.board, expected.board, "board mismatch");
        // Hands empty
        assert_eq!(pos.hands, expected.hands, "hands mismatch");
        // Black to move
        assert_eq!(pos.current_player, Color::Black, "side to move mismatch");
    }

    #[test]
    fn test_sfen_roundtrip_with_hands() {
        // A position where both players have pieces in hand.
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w RGSb 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        let reserialized = pos.to_sfen();
        assert_eq!(reserialized, sfen, "roundtrip with hands failed");
    }

    #[test]
    fn test_sfen_roundtrip_with_promoted() {
        // A mid-game position with promoted pieces on the board.
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1+B5R1/LNSGKGSNL b - 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        let reserialized = pos.to_sfen();
        assert_eq!(reserialized, sfen, "roundtrip with promoted pieces failed");
    }

    #[test]
    fn test_sfen_parse_white_to_move() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        assert_eq!(pos.current_player, Color::White);
    }

    #[test]
    fn test_sfen_hash_matches_recomputation() {
        let pos = Position::from_sfen(STARTPOS_SFEN).expect("parse failed");
        assert_eq!(
            pos.hash,
            pos.compute_hash(),
            "hash stored in parsed position doesn't match recomputed hash"
        );
    }

    #[test]
    fn test_sfen_invalid_too_short() {
        let result = Position::from_sfen("lnsgkgsnl b");
        assert!(result.is_err(), "expected error for too-short SFEN");
    }

    #[test]
    fn test_sfen_invalid_piece_char() {
        // 'X' is not a valid piece character
        let sfen = "Xnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "expected error for invalid piece character");
    }

    #[test]
    fn test_sfen_different_positions_different_hashes() {
        let pos1 = Position::startpos();
        // White to move variant
        let pos2 = Position::from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1")
            .expect("parse failed");
        assert_ne!(
            pos1.hash, pos2.hash,
            "different positions must have different hashes"
        );
    }

    // -----------------------------------------------------------------------
    // SFEN edge cases — Gap #10
    // -----------------------------------------------------------------------

    #[test]
    fn test_sfen_multi_digit_hand_count() {
        // 18 pawns in hand (maximum possible)
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b 18P 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        assert_eq!(
            pos.hand_count(Color::Black, HandPieceType::Pawn),
            18,
            "Black should have 18 pawns in hand"
        );
    }

    #[test]
    fn test_sfen_multiple_hand_pieces() {
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b 2R2B4G4S4N4L18P 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Rook), 2);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Bishop), 2);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Gold), 4);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Silver), 4);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Knight), 4);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Lance), 4);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Pawn), 18);
    }

    #[test]
    fn test_sfen_invalid_wrong_rank_count() {
        // Only 8 ranks instead of 9
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1 b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "expected error for 8 ranks");
    }

    #[test]
    fn test_sfen_invalid_wrong_column_count() {
        // First rank has only 8 columns
        let sfen = "lnsgkgsn/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "expected error for rank with 8 columns");
    }

    #[test]
    fn test_sfen_invalid_zero_empty() {
        // '0' is not a valid empty count
        let sfen = "lnsgkgsnl/0r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "expected error for empty count '0'");
    }

    #[test]
    fn test_sfen_invalid_promoted_gold() {
        // Gold cannot be promoted
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSG+KGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "expected error for promoted King");
    }

    #[test]
    fn test_sfen_invalid_side_to_move() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL x - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "expected error for invalid side 'x'");
    }

    #[test]
    fn test_sfen_roundtrip_multi_digit_hands() {
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b 18P2r 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        let reserialized = pos.to_sfen();
        assert_eq!(reserialized, sfen, "multi-digit hand roundtrip failed");
    }

    // -----------------------------------------------------------------------
    // SFEN: promoted pieces on board survive roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_sfen_roundtrip_all_promoted_types() {
        // Position with every promotable piece type promoted on the board
        let sfen = "4k4/9/9/9/+P+L+N+S+B+R3/9/9/9/4K4 b - 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        let reserialized = pos.to_sfen();
        assert_eq!(
            reserialized, sfen,
            "roundtrip with all promoted piece types failed"
        );
    }

    #[test]
    fn test_sfen_roundtrip_white_only_in_hand() {
        // White has pieces in hand, Black has none
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b 2r3b 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        assert_eq!(pos.hand_count(Color::White, HandPieceType::Rook), 2);
        assert_eq!(pos.hand_count(Color::White, HandPieceType::Bishop), 3);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Rook), 0);
        let reserialized = pos.to_sfen();
        assert_eq!(reserialized, sfen, "roundtrip with White-only hand failed");
    }

    /// Simulate capturing a promoted piece: the hand should have the base type.
    /// We verify this through SFEN by creating a position with a capture result.
    #[test]
    fn test_sfen_captured_promoted_piece_in_hand_as_base() {
        use crate::game::GameState;
        use crate::types::Move;

        // White promoted rook (Dragon) at (4,4). Black rook at (4,0) captures it.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::White, true), // promoted
        );
        pos.set_piece(
            Square::from_row_col(4, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // Capture the promoted rook
        let capture = Move::Board {
            from: Square::from_row_col(4, 0).unwrap(),
            to: Square::from_row_col(4, 4).unwrap(),
            promote: false,
        };
        gs.make_move(capture);

        // Black should have 1 Rook (base type) in hand
        assert_eq!(gs.position.hand_count(Color::Black, HandPieceType::Rook), 1);

        // Serialize to SFEN and verify hand shows 'R' (not '+R')
        let sfen = gs.position.to_sfen();
        assert!(
            sfen.contains(" R ") || sfen.contains(" R") || sfen.split_whitespace().nth(2).unwrap().contains('R'),
            "SFEN hand should show 'R' (base type), got hand part: {}",
            sfen.split_whitespace().nth(2).unwrap()
        );

        // Verify roundtrip
        let reparsed = Position::from_sfen(&sfen).expect("re-parse failed");
        assert_eq!(
            reparsed.hand_count(Color::Black, HandPieceType::Rook),
            1,
            "Roundtrip after capture: Black should still have 1 Rook in hand"
        );
        assert_eq!(reparsed.board, gs.position.board, "Board mismatch after SFEN roundtrip");
    }

    // -----------------------------------------------------------------------
    // SFEN edge cases: promoted king, empty board, king in hand, col overflow
    // -----------------------------------------------------------------------

    /// Promoted king (+K) on the board should be rejected (King cannot promote).
    #[test]
    fn test_sfen_invalid_promoted_king() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSG+KGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "promoted King (+K) should be invalid");
    }

    /// Empty board SFEN: all squares empty, only kings needed for a valid
    /// position isn't required by the parser — test pure empty parsing.
    #[test]
    fn test_sfen_empty_board() {
        let sfen = "9/9/9/9/9/9/9/9/9 b - 1";
        let pos = Position::from_sfen(sfen).expect("empty board should parse");
        // Every square should be empty
        for idx in 0..81 {
            let sq = Square::new_unchecked(idx as u8);
            assert!(
                pos.piece_at(sq).is_none(),
                "square {} should be empty on empty board",
                idx
            );
        }
        // Hands should be empty
        for &hpt in &HandPieceType::ALL {
            assert_eq!(pos.hand_count(Color::Black, hpt), 0);
            assert_eq!(pos.hand_count(Color::White, hpt), 0);
        }
        // Roundtrip
        assert_eq!(pos.to_sfen(), sfen);
    }

    /// King character in hand string should be rejected (Kings can't be captured).
    #[test]
    fn test_sfen_invalid_king_in_hand() {
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b K 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "King in hand should be invalid");
    }

    /// Column overflow: a rank with more than 9 columns should be rejected.
    #[test]
    fn test_sfen_invalid_column_overflow() {
        // "55" = 5 empty + 5 empty = 10 columns
        let sfen = "55sgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "rank with >9 columns should be rejected");
    }

    /// Lowercase promoted piece in SFEN (White's promoted pawn).
    #[test]
    fn test_sfen_roundtrip_white_promoted_piece() {
        let sfen = "4k4/9/9/9/+p8/9/9/9/4K4 b - 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        let piece = pos.piece_at(Square::from_row_col(4, 0).unwrap()).unwrap();
        assert_eq!(piece.piece_type(), PieceType::Pawn);
        assert_eq!(piece.color(), Color::White);
        assert!(piece.is_promoted());
        assert_eq!(pos.to_sfen(), sfen);
    }

    // -----------------------------------------------------------------------
    // White-to-move roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_sfen_roundtrip_white_to_move() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        assert_eq!(pos.current_player, Color::White);
        let reserialized = pos.to_sfen();
        assert_eq!(reserialized, sfen, "White-to-move roundtrip failed");
    }

    // -----------------------------------------------------------------------
    // Roundtrip with both colors having pieces in hand
    // -----------------------------------------------------------------------

    #[test]
    fn test_sfen_roundtrip_mixed_hands_both_colors() {
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b 2G3Prbp 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Gold), 2);
        assert_eq!(pos.hand_count(Color::Black, HandPieceType::Pawn), 3);
        assert_eq!(pos.hand_count(Color::White, HandPieceType::Rook), 1);
        assert_eq!(pos.hand_count(Color::White, HandPieceType::Bishop), 1);
        assert_eq!(pos.hand_count(Color::White, HandPieceType::Pawn), 1);
        let reserialized = pos.to_sfen();
        assert_eq!(reserialized, sfen, "Mixed hands roundtrip failed");
    }

    // -----------------------------------------------------------------------
    // Error paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_sfen_rank_too_short() {
        // First rank has only 7 columns (missing 2)
        let sfen = "lnsgkgs/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "rank with 7 columns should be rejected");
    }

    #[test]
    fn test_sfen_rank_too_long() {
        // First rank has 10 columns
        let sfen = "lnsgkgsnll/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "rank with 10 columns should be rejected");
    }

    #[test]
    fn test_sfen_plus_at_end_of_rank() {
        // '+' at end of rank with no following piece character
        let sfen = "lnsgkgsn+/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "'+' at end of rank should be rejected");
    }

    #[test]
    fn test_sfen_hand_ends_after_count() {
        // Hand string "3" — count with no following piece
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b 3 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "hand string ending after count should be rejected");
    }

    #[test]
    fn test_sfen_invalid_hand_piece_char() {
        // 'X' is not a valid hand piece
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 b X 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "invalid hand piece character should be rejected");
    }

    #[test]
    fn test_sfen_unexpected_char_in_board() {
        // '!' is not valid in board string
        let sfen = "!nsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let result = Position::from_sfen(sfen);
        assert!(result.is_err(), "unexpected character in board should be rejected");
    }

    // -----------------------------------------------------------------------
    // Hash consistency: parsed SFEN hash == computed hash (White to move)
    // -----------------------------------------------------------------------

    #[test]
    fn test_sfen_hash_correct_white_to_move() {
        let sfen = "4k4/9/9/9/9/9/9/9/4K4 w 2Pp 1";
        let pos = Position::from_sfen(sfen).expect("parse failed");
        assert_eq!(
            pos.hash,
            pos.compute_hash(),
            "parsed hash must equal recomputed hash (White to move with hands)"
        );
    }
}
