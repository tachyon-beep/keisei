use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use shogi_core::{Color, GameResult, GameState, HandPieceType, Move, PieceType, Square};

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
    match hpt {
        HandPieceType::Pawn   => 'P',
        HandPieceType::Lance  => 'L',
        HandPieceType::Knight => 'N',
        HandPieceType::Silver => 'S',
        HandPieceType::Gold   => 'G',
        HandPieceType::Bishop => 'B',
        HandPieceType::Rook   => 'R',
    }
}

/// Build move notation string from a Move.
/// Board: `"9a→8a+"` format: `{9-from_col}{chr(a+from_row)}→{9-to_col}{chr(a+to_row)}{+ if promote}`
/// Drop:  `"P*5e"` format: `{piece_char}*{9-to_col}{chr(a+to_row)}`
pub fn move_notation(mv: Move) -> String {
    match mv {
        Move::Board { from, to, promote } => {
            let from_col_shogi = 9 - from.col();
            let from_row_char = (b'a' + from.row()) as char;
            let to_col_shogi = 9 - to.col();
            let to_row_char = (b'a' + to.row()) as char;
            let promo_str = if promote { "+" } else { "" };
            format!(
                "{}{}→{}{}{}",
                from_col_shogi, from_row_char,
                to_col_shogi, to_row_char,
                promo_str
            )
        }
        Move::Drop { to, piece_type } => {
            let piece_char = hand_piece_char(piece_type);
            let to_col_shogi = 9 - to.col();
            let to_row_char = (b'a' + to.row()) as char;
            format!("{}*{}{}", piece_char, to_col_shogi, to_row_char)
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
}
