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
}
