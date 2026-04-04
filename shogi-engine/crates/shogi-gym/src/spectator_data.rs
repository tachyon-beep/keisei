use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use shogi_core::{Color, GameResult, GameState, HandPieceType, Move, Piece, PieceType, Position, Square};

/// Single-letter piece abbreviation for Hodges notation.
pub fn piece_char(pt: PieceType) -> char {
    match pt {
        PieceType::King   => 'K',
        PieceType::Rook   => 'R',
        PieceType::Bishop => 'B',
        PieceType::Gold   => 'G',
        PieceType::Silver => 'S',
        PieceType::Knight => 'N',
        PieceType::Lance  => 'L',
        PieceType::Pawn   => 'P',
    }
}

/// Format a square as Hodges coordinates: file (1-9) + rank (a-i).
/// Example: row=6, col=4 → file=5, rank='g' → "5g"
pub fn square_notation(sq: Square) -> String {
    let file = 9 - sq.col();
    let rank = (b'a' + sq.row()) as char;
    format!("{}{}", file, rank)
}

/// Check if a square is in the promotion zone for the given color.
/// Black: rows 0-2 (ranks a-c). White: rows 6-8 (ranks g-i).
pub fn in_promotion_zone(sq: Square, color: Color) -> bool {
    match color {
        Color::Black => sq.row() <= 2,
        Color::White => sq.row() >= 6,
    }
}

/// Check if a piece could promote on this move (but hasn't necessarily chosen to).
/// True when: piece type can promote, piece is not already promoted,
/// and either source or destination is in the promotion zone.
pub fn could_promote(piece: Piece, from: Square, to: Square) -> bool {
    piece.piece_type().can_promote()
        && !piece.is_promoted()
        && (in_promotion_zone(from, piece.color()) || in_promotion_zone(to, piece.color()))
}

pub fn piece_type_name(pt: PieceType) -> &'static str {
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

pub fn color_name(c: Color) -> &'static str {
    match c {
        Color::Black => "black",
        Color::White => "white",
    }
}

pub fn game_result_str(r: &GameResult) -> &'static str {
    match r {
        GameResult::InProgress        => "in_progress",
        GameResult::Checkmate { .. }  => "checkmate",
        GameResult::Repetition        => "repetition",
        GameResult::PerpetualCheck { .. } => "perpetual_check",
        GameResult::Impasse { .. }    => "impasse",
        GameResult::MaxMoves          => "max_moves",
    }
}

/// Encode a drop piece char: P, L, N, S, G, B, R
pub fn hand_piece_char(hpt: HandPieceType) -> char {
    piece_char(hpt.to_piece_type())
}

/// Check if promotion is forced for this piece reaching the destination.
/// Pawn/Lance on last rank, Knight on last two ranks.
fn is_forced_promotion(pt: PieceType, to: Square, color: Color) -> bool {
    let dest_row = to.row();
    match color {
        Color::Black => match pt {
            PieceType::Pawn | PieceType::Lance => dest_row == 0,
            PieceType::Knight => dest_row <= 1,
            _ => false,
        },
        Color::White => match pt {
            PieceType::Pawn | PieceType::Lance => dest_row == 8,
            PieceType::Knight => dest_row >= 7,
            _ => false,
        },
    }
}

/// Build Hodges notation string from a Move, a Position, and the legal moves list.
///
/// Board: `"P-7f"`, `"Bx3c"`, `"Nx7c+"`, `"S-4d="`, `"+R-5a"`, `"G6g-5h"`
/// Drop:  `"P*5e"`
pub fn move_notation(mv: Move, position: &Position, legal_moves: &[Move]) -> String {
    match mv {
        Move::Board { from, to, promote } => {
            let piece = position.piece_at(from)
                .expect("move_notation: no piece at source square");
            let pt = piece.piece_type();
            let color = piece.color();
            let promoted = piece.is_promoted();

            // Piece prefix: "+R" if promoted, "R" if not
            let prefix = if promoted {
                format!("+{}", piece_char(pt))
            } else {
                format!("{}", piece_char(pt))
            };

            // Disambiguation: check if another legal board move by same piece type
            // (and same promoted status) targets the same destination
            let disambig = if pt == PieceType::King {
                String::new()
            } else {
                let ambiguous = legal_moves.iter().any(|other| {
                    if let Move::Board { from: of, to: ot, .. } = other {
                        *ot == to && *of != from && {
                            if let Some(other_piece) = position.piece_at(*of) {
                                other_piece.piece_type() == pt
                                    && other_piece.is_promoted() == promoted
                            } else {
                                false
                            }
                        }
                    } else {
                        false
                    }
                });
                if ambiguous {
                    square_notation(from)
                } else {
                    String::new()
                }
            };

            // Capture or move separator
            let sep = if position.piece_at(to).is_some() { "x" } else { "-" };

            // Destination
            let dest = square_notation(to);

            // Promotion suffix
            let suffix = if promote || is_forced_promotion(pt, to, color) {
                "+"
            } else if could_promote(piece, from, to) {
                "="
            } else {
                ""
            };

            format!("{}{}{}{}{}", prefix, disambig, sep, dest, suffix)
        }
        Move::Drop { to, piece_type } => {
            format!("{}*{}", piece_char(piece_type.to_piece_type()), square_notation(to))
        }
    }
}

/// Build a spectator-format Python dict from a GameState.
/// Omits move_history (caller supplies it if available).
pub fn build_spectator_dict(py: Python<'_>, game: &GameState) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);

    // -- board: list of 81 elements (None or piece dict) --
    let board_list = PyList::empty(py);
    for idx in 0..81usize {
        let sq = Square::new_unchecked(idx as u8);
        match game.position.piece_at(sq) {
            None => board_list.append(py.None())?,
            Some(piece) => {
                let pd = PyDict::new(py);
                pd.set_item("type", piece_type_name(piece.piece_type()))?;
                pd.set_item("color", color_name(piece.color()))?;
                pd.set_item("promoted", piece.is_promoted())?;
                pd.set_item("row", sq.row() as i64)?;
                pd.set_item("col", sq.col() as i64)?;
                board_list.append(pd)?;
            }
        }
    }
    d.set_item("board", board_list)?;

    // -- hands --
    let hands_dict = PyDict::new(py);
    for &color in &[Color::Black, Color::White] {
        let hand_dict = PyDict::new(py);
        for &hpt in &HandPieceType::ALL {
            let count = game.position.hand_count(color, hpt) as i64;
            hand_dict.set_item(piece_type_name(hpt.to_piece_type()), count)?;
        }
        hands_dict.set_item(color_name(color), hand_dict)?;
    }
    d.set_item("hands", hands_dict)?;

    // -- scalar fields --
    d.set_item("current_player", color_name(game.position.current_player))?;
    d.set_item("ply", game.ply as i64)?;
    d.set_item("is_over", game.result.is_terminal())?;
    d.set_item("result", game_result_str(&game.result))?;
    d.set_item("sfen", game.position.to_sfen())?;
    d.set_item("in_check", game.is_in_check())?;

    Ok(d.into())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::{Piece, Position};

    #[test]
    fn test_piece_type_name_all() {
        assert_eq!(piece_type_name(PieceType::Pawn), "pawn");
        assert_eq!(piece_type_name(PieceType::Lance), "lance");
        assert_eq!(piece_type_name(PieceType::Knight), "knight");
        assert_eq!(piece_type_name(PieceType::Silver), "silver");
        assert_eq!(piece_type_name(PieceType::Gold), "gold");
        assert_eq!(piece_type_name(PieceType::Bishop), "bishop");
        assert_eq!(piece_type_name(PieceType::Rook), "rook");
        assert_eq!(piece_type_name(PieceType::King), "king");
    }

    #[test]
    fn test_color_name_all() {
        assert_eq!(color_name(Color::Black), "black");
        assert_eq!(color_name(Color::White), "white");
    }

    #[test]
    fn test_game_result_str_all() {
        assert_eq!(game_result_str(&GameResult::InProgress), "in_progress");
        assert_eq!(
            game_result_str(&GameResult::Checkmate { winner: Color::Black }),
            "checkmate"
        );
        assert_eq!(game_result_str(&GameResult::Repetition), "repetition");
        assert_eq!(
            game_result_str(&GameResult::PerpetualCheck { winner: Color::White }),
            "perpetual_check"
        );
        assert_eq!(
            game_result_str(&GameResult::Impasse { winner: None }),
            "impasse"
        );
        assert_eq!(
            game_result_str(&GameResult::Impasse { winner: Some(Color::Black) }),
            "impasse"
        );
        assert_eq!(game_result_str(&GameResult::MaxMoves), "max_moves");
    }

    #[test]
    fn test_piece_char_all() {
        assert_eq!(piece_char(PieceType::King), 'K');
        assert_eq!(piece_char(PieceType::Rook), 'R');
        assert_eq!(piece_char(PieceType::Bishop), 'B');
        assert_eq!(piece_char(PieceType::Gold), 'G');
        assert_eq!(piece_char(PieceType::Silver), 'S');
        assert_eq!(piece_char(PieceType::Knight), 'N');
        assert_eq!(piece_char(PieceType::Lance), 'L');
        assert_eq!(piece_char(PieceType::Pawn), 'P');
    }

    #[test]
    fn test_square_notation_corners_and_center() {
        // Top-right (row=0, col=0): file=9-0=9, rank='a'+0='a' → "9a"
        assert_eq!(square_notation(Square::from_row_col(0, 0).unwrap()), "9a");
        // Bottom-left (row=8, col=8): file=9-8=1, rank='a'+8='i' → "1i"
        assert_eq!(square_notation(Square::from_row_col(8, 8).unwrap()), "1i");
        // Center (row=4, col=4): file=9-4=5, rank='a'+4='e' → "5e"
        assert_eq!(square_notation(Square::from_row_col(4, 4).unwrap()), "5e");
        // Top-left (row=0, col=8): file=9-8=1, rank='a'+0='a' → "1a"
        assert_eq!(square_notation(Square::from_row_col(0, 8).unwrap()), "1a");
        // Bottom-right (row=8, col=0): file=9-0=9, rank='a'+8='i' → "9i"
        assert_eq!(square_notation(Square::from_row_col(8, 0).unwrap()), "9i");
    }

    #[test]
    fn test_hand_piece_char_delegates_to_piece_char() {
        for &hpt in &HandPieceType::ALL {
            assert_eq!(
                hand_piece_char(hpt),
                piece_char(hpt.to_piece_type()),
                "hand_piece_char and piece_char disagree for {:?}",
                hpt
            );
        }
    }

    #[test]
    fn test_in_promotion_zone_black() {
        // Black promotion zone: rows 0, 1, 2 (ranks a, b, c)
        let col = 4;
        assert!(in_promotion_zone(Square::from_row_col(0, col).unwrap(), Color::Black));
        assert!(in_promotion_zone(Square::from_row_col(1, col).unwrap(), Color::Black));
        assert!(in_promotion_zone(Square::from_row_col(2, col).unwrap(), Color::Black));
        // Row 3 is NOT in zone
        assert!(!in_promotion_zone(Square::from_row_col(3, col).unwrap(), Color::Black));
        // Nor is row 8
        assert!(!in_promotion_zone(Square::from_row_col(8, col).unwrap(), Color::Black));
    }

    #[test]
    fn test_in_promotion_zone_white() {
        // White promotion zone: rows 6, 7, 8 (ranks g, h, i)
        let col = 4;
        assert!(in_promotion_zone(Square::from_row_col(6, col).unwrap(), Color::White));
        assert!(in_promotion_zone(Square::from_row_col(7, col).unwrap(), Color::White));
        assert!(in_promotion_zone(Square::from_row_col(8, col).unwrap(), Color::White));
        // Row 5 is NOT in zone
        assert!(!in_promotion_zone(Square::from_row_col(5, col).unwrap(), Color::White));
        // Nor is row 0
        assert!(!in_promotion_zone(Square::from_row_col(0, col).unwrap(), Color::White));
    }

    #[test]
    fn test_could_promote_all_conditions() {
        let from_outside = Square::from_row_col(5, 4).unwrap(); // row 5 = rank f
        let to_inside = Square::from_row_col(2, 4).unwrap();    // row 2 = rank c (Black zone)
        let from_inside = Square::from_row_col(2, 4).unwrap();
        let to_outside = Square::from_row_col(5, 4).unwrap();

        let silver_black = Piece::new(PieceType::Silver, Color::Black, false);
        let gold_black = Piece::new(PieceType::Gold, Color::Black, false);
        let promoted_silver = Piece::new(PieceType::Silver, Color::Black, true);
        let king_black = Piece::new(PieceType::King, Color::Black, false);

        // Silver moving INTO zone: can promote
        assert!(could_promote(silver_black, from_outside, to_inside));
        // Silver moving OUT OF zone: can promote
        assert!(could_promote(silver_black, from_inside, to_outside));
        // Gold: cannot promote (type doesn't allow)
        assert!(!could_promote(gold_black, from_outside, to_inside));
        // King: cannot promote
        assert!(!could_promote(king_black, from_outside, to_inside));
        // Already promoted silver: cannot promote again
        assert!(!could_promote(promoted_silver, from_outside, to_inside));
        // Silver moving entirely outside zone: cannot promote
        let other_outside = Square::from_row_col(6, 4).unwrap();
        assert!(!could_promote(silver_black, from_outside, other_outside));
    }

    // -----------------------------------------------------------------------
    // move_notation tests — Hodges format
    // -----------------------------------------------------------------------

    fn position_with_pieces(pieces: &[(Square, Piece)]) -> Position {
        let mut pos = Position::empty();
        for &(sq, piece) in pieces {
            pos.set_piece(sq, piece);
        }
        pos
    }

    #[test]
    fn test_notation_simple_move() {
        let from = Square::from_row_col(6, 2).unwrap();
        let to = Square::from_row_col(5, 2).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-7f");
    }

    #[test]
    fn test_notation_capture() {
        let from = Square::from_row_col(7, 1).unwrap();
        let to = Square::from_row_col(2, 6).unwrap();
        let bishop = Piece::new(PieceType::Bishop, Color::Black, false);
        let enemy_pawn = Piece::new(PieceType::Pawn, Color::White, false);
        let pos = position_with_pieces(&[(from, bishop), (to, enemy_pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "Bx3c=");
    }

    #[test]
    fn test_notation_promotion() {
        let from = Square::from_row_col(3, 1).unwrap();
        let to = Square::from_row_col(1, 2).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::Black, false);
        let enemy = Piece::new(PieceType::Gold, Color::White, false);
        let pos = position_with_pieces(&[(from, knight), (to, enemy)]);
        let mv = Move::Board { from, to, promote: true };
        assert_eq!(move_notation(mv, &pos, &[mv]), "Nx7b+");
    }

    #[test]
    fn test_notation_declined_promotion() {
        let from = Square::from_row_col(3, 5).unwrap();
        let to = Square::from_row_col(2, 5).unwrap();
        let silver = Piece::new(PieceType::Silver, Color::Black, false);
        let pos = position_with_pieces(&[(from, silver)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "S-4c=");
    }

    #[test]
    fn test_notation_white_declined_promotion() {
        // White silver at 6f (row=5, col=3) moves into White zone at 6g (row=6, col=3)
        let from = Square::from_row_col(5, 3).unwrap();
        let to = Square::from_row_col(6, 3).unwrap();
        let silver = Piece::new(PieceType::Silver, Color::White, false);
        let pos = position_with_pieces(&[(from, silver)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "S-6g=");
    }

    #[test]
    fn test_notation_promoted_piece_moving() {
        let from = Square::from_row_col(0, 4).unwrap();
        let to = Square::from_row_col(1, 4).unwrap();
        let dragon = Piece::new(PieceType::Rook, Color::Black, true);
        let pos = position_with_pieces(&[(from, dragon)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "+R-5b");
    }

    #[test]
    fn test_notation_drop() {
        let to = Square::from_row_col(4, 4).unwrap();
        let pos = Position::empty();
        let mv = Move::Drop { to, piece_type: HandPieceType::Pawn };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P*5e");
    }

    #[test]
    fn test_notation_drop_all_piece_types() {
        let to = Square::from_row_col(4, 4).unwrap();
        let pos = Position::empty();
        let expected = [
            (HandPieceType::Pawn,   "P*5e"),
            (HandPieceType::Lance,  "L*5e"),
            (HandPieceType::Knight, "N*5e"),
            (HandPieceType::Silver, "S*5e"),
            (HandPieceType::Gold,   "G*5e"),
            (HandPieceType::Bishop, "B*5e"),
            (HandPieceType::Rook,   "R*5e"),
        ];
        for (hpt, exp) in &expected {
            let mv = Move::Drop { to, piece_type: *hpt };
            assert_eq!(move_notation(mv, &pos, &[mv]), *exp, "Drop for {:?}", hpt);
        }
    }

    #[test]
    fn test_notation_disambiguation() {
        let from1 = Square::from_row_col(6, 3).unwrap();
        let from2 = Square::from_row_col(6, 5).unwrap();
        let to = Square::from_row_col(5, 4).unwrap();
        let gold = Piece::new(PieceType::Gold, Color::Black, false);
        let pos = position_with_pieces(&[(from1, gold), (from2, gold)]);
        let mv1 = Move::Board { from: from1, to, promote: false };
        let mv2 = Move::Board { from: from2, to, promote: false };
        let legal = vec![mv1, mv2];
        assert_eq!(move_notation(mv1, &pos, &legal), "G6g-5f");
        assert_eq!(move_notation(mv2, &pos, &legal), "G4g-5f");
    }

    #[test]
    fn test_notation_no_disambiguation_single_piece() {
        let from = Square::from_row_col(6, 3).unwrap();
        let to = Square::from_row_col(7, 4).unwrap();
        let gold = Piece::new(PieceType::Gold, Color::Black, false);
        let pos = position_with_pieces(&[(from, gold)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "G-5h");
    }

    #[test]
    fn test_notation_king_never_disambiguated() {
        let from = Square::from_row_col(8, 4).unwrap();
        let to = Square::from_row_col(7, 4).unwrap();
        let king = Piece::new(PieceType::King, Color::Black, false);
        let pos = position_with_pieces(&[(from, king)]);
        let mv = Move::Board { from, to, promote: false };
        let fake = Move::Board {
            from: Square::from_row_col(7, 3).unwrap(),
            to,
            promote: false,
        };
        assert_eq!(move_notation(mv, &pos, &[mv, fake]), "K-5h");
    }

    #[test]
    fn test_notation_forced_promotion_pawn_last_rank() {
        let from = Square::from_row_col(1, 2).unwrap();
        let to = Square::from_row_col(0, 2).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        let mv = Move::Board { from, to, promote: true };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-7a+");
        // Guard: forced promotion even if promote=false
        let mv_bad = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv_bad, &pos, &[mv_bad]), "P-7a+");
    }

    #[test]
    fn test_notation_forced_promotion_knight_last_two_ranks() {
        let from = Square::from_row_col(3, 2).unwrap();
        let to = Square::from_row_col(1, 1).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::Black, false);
        let pos = position_with_pieces(&[(from, knight)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "N-8b+");
    }

    #[test]
    fn test_notation_forced_promotion_lance_last_rank() {
        let from = Square::from_row_col(1, 4).unwrap();
        let to = Square::from_row_col(0, 4).unwrap();
        let lance = Piece::new(PieceType::Lance, Color::Black, false);
        let pos = position_with_pieces(&[(from, lance)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "L-5a+");
    }

    #[test]
    fn test_notation_white_forced_promotion() {
        let from = Square::from_row_col(7, 6).unwrap();
        let to = Square::from_row_col(8, 6).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::White, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-3i+");
    }

    #[test]
    fn test_notation_white_knight_forced_promotion() {
        let from = Square::from_row_col(6, 6).unwrap();
        let to = Square::from_row_col(8, 7).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::White, false);
        let pos = position_with_pieces(&[(from, knight)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "N-2i+");
    }

    #[test]
    fn test_notation_boundary_top_right_king() {
        let from = Square::from_row_col(0, 0).unwrap();
        let to = Square::from_row_col(1, 0).unwrap();
        let king = Piece::new(PieceType::King, Color::Black, false);
        let pos = position_with_pieces(&[(from, king)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "K-9b");
    }

    #[test]
    fn test_notation_boundary_bottom_left_king() {
        let from = Square::from_row_col(8, 8).unwrap();
        let to = Square::from_row_col(7, 8).unwrap();
        let king = Piece::new(PieceType::King, Color::White, false);
        let pos = position_with_pieces(&[(from, king)]);
        let mv = Move::Board { from: from, to: to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "K-1h");
    }

    #[test]
    fn test_notation_boundary_drop_corner() {
        let drop_sq = Square::from_row_col(0, 8).unwrap();
        let pos = Position::empty();
        let mv = Move::Drop { to: drop_sq, piece_type: HandPieceType::Pawn };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P*1a");
    }
}
