use numpy::{PyArray3, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use shogi_core::GameState;

use crate::action_mapper::{ActionMapper, DefaultActionMapper};
use crate::observation::{DefaultObservationGenerator, ObservationGenerator, BUFFER_LEN, NUM_CHANNELS};
use crate::spectator_data::{build_spectator_dict, color_name, move_notation};

// ---------------------------------------------------------------------------
// SpectatorEnv
// ---------------------------------------------------------------------------

/// Single-game environment for spectator/display use.
///
/// Key differences from VecEnv:
/// - Returns rich Python dicts (acceptable — not on the hot path)
/// - Does NOT auto-reset on game end — stays ended until explicitly `reset()`
/// - Provides `to_dict()` for JSON serialization (Streamlit dashboard)
/// - `legal_actions()` returns list of valid action indices
#[pyclass]
pub struct SpectatorEnv {
    game: GameState,
    max_ply: u32,
    mapper: DefaultActionMapper,
    obs_gen: DefaultObservationGenerator,
    move_history: Vec<(usize, String)>,  // (action_index, move_notation)
}

#[pymethods]
impl SpectatorEnv {
    /// Create a new SpectatorEnv.
    ///
    /// Args:
    ///     max_ply: Maximum number of plies before the game ends (default 500).
    #[new]
    #[pyo3(signature = (max_ply = 500))]
    pub fn new(max_ply: u32) -> Self {
        SpectatorEnv {
            game: GameState::with_max_ply(max_ply),
            max_ply,
            mapper: DefaultActionMapper,
            obs_gen: DefaultObservationGenerator::new(),
            move_history: Vec::new(),
        }
    }

    /// Create a SpectatorEnv from a SFEN string.
    ///
    /// Args:
    ///     sfen: SFEN position string.
    ///     max_ply: Maximum plies before truncation (default 500).
    ///
    /// Raises ValueError if the SFEN is invalid.
    #[staticmethod]
    #[pyo3(signature = (sfen, max_ply = None))]
    pub fn from_sfen(sfen: &str, max_ply: Option<u32>) -> PyResult<Self> {
        let max_ply = max_ply.unwrap_or(500);
        let game = GameState::from_sfen(sfen, max_ply)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid SFEN: {e}")))?;
        Ok(SpectatorEnv {
            game,
            max_ply,
            mapper: DefaultActionMapper,
            obs_gen: DefaultObservationGenerator::new(),
            move_history: Vec::new(),
        })
    }

    /// Reset game to startpos, clear move history, return state dict.
    pub fn reset(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.game = GameState::with_max_ply(self.max_ply);
        self.move_history.clear();
        self.to_dict(py)
    }

    /// Apply an action to the game.
    ///
    /// Raises RuntimeError if the game is already over.
    /// Returns the new state dict.
    pub fn step(&mut self, py: Python<'_>, action: usize) -> PyResult<Py<PyDict>> {
        if self.game.result.is_terminal() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot step: game is already over. Call reset() to start a new game.",
            ));
        }

        let perspective = self.game.position.current_player;
        let mv = <DefaultActionMapper as ActionMapper>::decode(&self.mapper, action, perspective)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        let legal_moves = self.game.legal_moves();
        if !legal_moves.contains(&mv) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "action index {} is not legal for current position",
                action
            )));
        }

        let notation = move_notation(mv, &self.game.position, &legal_moves);
        self.move_history.push((action, notation));

        self.game.make_move(mv);
        self.game.check_termination();

        self.to_dict(py)
    }

    /// Return current state as a rich Python dict suitable for JSON serialization.
    ///
    /// Keys:
    /// - `board`: list of 81 elements, each None or `{"type": str, "color": str, "promoted": bool, "row": int, "col": int}`
    /// - `hands`: `{"black": {"pawn": N, ...}, "white": {...}}`
    /// - `current_player`: "black" or "white"
    /// - `ply`: int
    /// - `is_over`: bool
    /// - `result`: "in_progress" / "checkmate" / "repetition" / "perpetual_check" / "impasse" / "max_moves"
    /// - `sfen`: str
    /// - `in_check`: bool
    /// - `move_history`: list of `{"action": int, "notation": str}`
    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d_bound = build_spectator_dict(py, &self.game)?;
        let d = d_bound.bind(py);

        // Append move_history (SpectatorEnv-only)
        let history_list = PyList::empty(py);
        for (action_idx, notation) in &self.move_history {
            let hd = PyDict::new(py);
            hd.set_item("action", *action_idx as i64)?;
            hd.set_item("notation", notation.as_str())?;
            history_list.append(hd)?;
        }
        d.set_item("move_history", history_list)?;

        Ok(d_bound)
    }

    /// Serialize current position to SFEN string.
    pub fn to_sfen(&self) -> String {
        self.game.position.to_sfen()
    }

    /// Return the observation as a shaped (46, 9, 9) numpy array.
    ///
    /// The observation is generated from the current player's perspective,
    /// consistent with VecEnv observation format.
    pub fn get_observation<'py>(&self, py: Python<'py>) -> PyResult<Py<PyArray3<f32>>> {
        let mut buffer = vec![0.0_f32; BUFFER_LEN];
        let perspective = self.game.position.current_player;
        self.obs_gen.generate(&self.game, perspective, &mut buffer);
        let array = buffer.to_pyarray(py);
        let shaped = array
            .reshape([NUM_CHANNELS, 9, 9])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(shaped.unbind())
    }

    /// Return a list of legal action indices for the current position.
    pub fn legal_actions(&mut self) -> Vec<usize> {
        let perspective = self.game.position.current_player;
        let moves = self.game.legal_moves();
        moves
            .into_iter()
            .map(|mv| self.mapper.encode(mv, perspective)
                .expect("legal move must be encodable"))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Property getters
    // -----------------------------------------------------------------------

    /// Whether the game has ended.
    #[getter]
    pub fn is_over(&self) -> bool {
        self.game.result.is_terminal()
    }

    /// Current player as a string: "black" or "white".
    #[getter]
    pub fn current_player(&self) -> &str {
        color_name(self.game.position.current_player)
    }

    /// Current ply count.
    #[getter]
    pub fn ply(&self) -> u32 {
        self.game.ply
    }

    /// Total number of actions in the action space (13527).
    #[getter]
    pub fn action_space_size(&self) -> usize {
        self.mapper.action_space_size()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    use shogi_core::{GameState, Color, HandPieceType, Move, Piece, PieceType, Position, Square};
    use crate::action_mapper::ActionMapper;
    use crate::spectator_data::{hand_piece_char, move_notation};

    // Helper: create a position with specific pieces placed.
    // Starts from empty board, places pieces, sets current_player.
    fn position_with_pieces(pieces: &[(Square, Piece)]) -> Position {
        let mut pos = Position::empty();
        for &(sq, piece) in pieces {
            pos.set_piece(sq, piece);
        }
        pos
    }

    // -----------------------------------------------------------------------
    // move_notation tests — Hodges format
    // -----------------------------------------------------------------------

    #[test]
    fn test_notation_simple_move() {
        // Black pawn at 7g (row=6, col=2) moves to 7f (row=5, col=2)
        let from = Square::from_row_col(6, 2).unwrap();
        let to = Square::from_row_col(5, 2).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-7f");
    }

    #[test]
    fn test_notation_capture() {
        // Black bishop at 8h captures White pawn at 3c
        let from = Square::from_row_col(7, 1).unwrap(); // 8h
        let to = Square::from_row_col(2, 6).unwrap();   // 3c
        let bishop = Piece::new(PieceType::Bishop, Color::Black, false);
        let enemy_pawn = Piece::new(PieceType::Pawn, Color::White, false);
        let pos = position_with_pieces(&[(from, bishop), (to, enemy_pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "Bx3c=");
    }

    #[test]
    fn test_notation_promotion() {
        // Black knight at 8d (row=3, col=1) captures on 7b (row=1, col=2), promotes
        let from = Square::from_row_col(3, 1).unwrap(); // 8d
        let to = Square::from_row_col(1, 2).unwrap();   // 7b
        let knight = Piece::new(PieceType::Knight, Color::Black, false);
        let enemy = Piece::new(PieceType::Gold, Color::White, false);
        let pos = position_with_pieces(&[(from, knight), (to, enemy)]);
        let mv = Move::Board { from, to, promote: true };
        assert_eq!(move_notation(mv, &pos, &[mv]), "Nx7b+");
    }

    #[test]
    fn test_notation_declined_promotion() {
        // Black silver at 4d (row=3, col=5) moves to 4c (row=2, col=5), declines
        let from = Square::from_row_col(3, 5).unwrap(); // 4d
        let to = Square::from_row_col(2, 5).unwrap();   // 4c
        let silver = Piece::new(PieceType::Silver, Color::Black, false);
        let pos = position_with_pieces(&[(from, silver)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "S-4c=");
    }

    #[test]
    fn test_notation_promoted_piece_moving() {
        // Promoted rook (dragon) at 5a (row=0, col=4) moves to 5b (row=1, col=4)
        let from = Square::from_row_col(0, 4).unwrap(); // 5a
        let to = Square::from_row_col(1, 4).unwrap();   // 5b
        let dragon = Piece::new(PieceType::Rook, Color::Black, true);
        let pos = position_with_pieces(&[(from, dragon)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "+R-5b");
    }

    #[test]
    fn test_notation_drop() {
        // Drop pawn at 5e
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
        // Two Black golds can both reach 5f (row=5, col=4) by moving forward.
        let from1 = Square::from_row_col(6, 3).unwrap(); // 6g
        let from2 = Square::from_row_col(6, 5).unwrap(); // 4g
        let to = Square::from_row_col(5, 4).unwrap();    // 5f
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
        // Only one gold can reach 5h — no origin needed
        let from = Square::from_row_col(6, 3).unwrap(); // 6g
        let to = Square::from_row_col(7, 4).unwrap();   // 5h
        let gold = Piece::new(PieceType::Gold, Color::Black, false);
        let pos = position_with_pieces(&[(from, gold)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "G-5h");
    }

    #[test]
    fn test_notation_king_never_disambiguated() {
        // Even if we pass a contrived legal_moves list with two "king" moves,
        // king should never get origin coordinates (only one king per side).
        let from = Square::from_row_col(8, 4).unwrap(); // 5i
        let to = Square::from_row_col(7, 4).unwrap();   // 5h
        let king = Piece::new(PieceType::King, Color::Black, false);
        let pos = position_with_pieces(&[(from, king)]);
        let mv = Move::Board { from, to, promote: false };
        // Pass a fake second king move — should still not disambiguate
        let fake = Move::Board {
            from: Square::from_row_col(7, 3).unwrap(),
            to,
            promote: false,
        };
        assert_eq!(move_notation(mv, &pos, &[mv, fake]), "K-5h");
    }

    #[test]
    fn test_notation_forced_promotion_pawn_last_rank() {
        // Black pawn at 7b (row=1, col=2) moves to 7a (row=0, col=2).
        let from = Square::from_row_col(1, 2).unwrap();
        let to = Square::from_row_col(0, 2).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        let mv = Move::Board { from, to, promote: true };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-7a+");
        // Verify the guard: if engine erroneously passes promote=false,
        // we still emit "+" for forced promotion
        let mv_bad = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv_bad, &pos, &[mv_bad]), "P-7a+");
    }

    #[test]
    fn test_notation_forced_promotion_knight_last_two_ranks() {
        // Black knight at 7d (row=3, col=2) moves to 8b (row=1, col=1).
        let from = Square::from_row_col(3, 2).unwrap();
        let to = Square::from_row_col(1, 1).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::Black, false);
        let pos = position_with_pieces(&[(from, knight)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "N-8b+");
    }

    #[test]
    fn test_notation_forced_promotion_lance_last_rank() {
        // Black lance at 5b (row=1, col=4) moves to 5a (row=0, col=4).
        let from = Square::from_row_col(1, 4).unwrap();
        let to = Square::from_row_col(0, 4).unwrap();
        let lance = Piece::new(PieceType::Lance, Color::Black, false);
        let pos = position_with_pieces(&[(from, lance)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "L-5a+");
    }

    #[test]
    fn test_notation_white_forced_promotion() {
        // White pawn at 3h (row=7, col=6) moves to 3i (row=8, col=6).
        let from = Square::from_row_col(7, 6).unwrap();
        let to = Square::from_row_col(8, 6).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::White, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-3i+");
    }

    #[test]
    fn test_notation_white_knight_forced_promotion() {
        // White knight at 3g (row=6, col=6) moves to 2i (row=8, col=7).
        let from = Square::from_row_col(6, 6).unwrap();
        let to = Square::from_row_col(8, 7).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::White, false);
        let pos = position_with_pieces(&[(from, knight)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "N-2i+");
    }

    #[test]
    fn test_notation_boundary_squares() {
        // 9a (row=0, col=0) → 9b (row=1, col=0): King move
        let from = Square::from_row_col(0, 0).unwrap();
        let to = Square::from_row_col(1, 0).unwrap();
        let king = Piece::new(PieceType::King, Color::Black, false);
        let pos = position_with_pieces(&[(from, king)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "K-9b");

        // 1i (row=8, col=8) → 1h (row=7, col=8): King move
        let from2 = Square::from_row_col(8, 8).unwrap();
        let to2 = Square::from_row_col(7, 8).unwrap();
        let king2 = Piece::new(PieceType::King, Color::White, false);
        let pos2 = position_with_pieces(&[(from2, king2)]);
        let mv2 = Move::Board { from: from2, to: to2, promote: false };
        assert_eq!(move_notation(mv2, &pos2, &[mv2]), "K-1h");

        // Drop at corner 1a
        let drop_sq = Square::from_row_col(0, 8).unwrap();
        let pos3 = Position::empty();
        let mv3 = Move::Drop { to: drop_sq, piece_type: HandPieceType::Pawn };
        assert_eq!(move_notation(mv3, &pos3, &[mv3]), "P*1a");
    }

    // -----------------------------------------------------------------------
    // SpectatorEnv internal logic tests (no Python needed)
    // -----------------------------------------------------------------------

    /// Test that a new SpectatorEnv starts at ply 0 and is not over.
    #[test]
    fn test_spectator_env_initial_state() {
        // We can't call #[new] directly without Python, but we can test
        // the GameState that backs it.
        let game = GameState::with_max_ply(500);
        assert_eq!(game.ply, 0);
        assert!(!game.result.is_terminal());
        assert_eq!(game.position.current_player, Color::Black);
    }

    /// Test from_sfen with valid startpos.
    #[test]
    fn test_spectator_env_from_sfen_valid() {
        let sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";
        let game = GameState::from_sfen(sfen, 500).expect("valid SFEN should parse");
        assert_eq!(game.ply, 0);
        assert_eq!(game.position.current_player, Color::Black);
    }

    /// Test from_sfen with invalid string.
    #[test]
    fn test_spectator_env_from_sfen_invalid() {
        let result = GameState::from_sfen("garbage sfen string", 500);
        assert!(result.is_err(), "Invalid SFEN should return error");
    }

    /// Test that stepping increments ply and flips player.
    #[test]
    fn test_spectator_env_step_increments_ply() {
        let mut game = GameState::with_max_ply(500);
        let mapper = DefaultActionMapper;

        let legal = game.legal_moves();
        assert!(!legal.is_empty());
        let mv = legal[0];
        let action = mapper.encode(mv, game.position.current_player).unwrap();

        // Decode and apply
        let perspective = game.position.current_player;
        let decoded = <DefaultActionMapper as ActionMapper>::decode(&mapper, action, perspective)
            .expect("decode should succeed");
        game.make_move(decoded);

        assert_eq!(game.ply, 1);
        assert_eq!(game.position.current_player, Color::White);
    }

    #[test]
    fn test_spectator_step_rejects_illegal_drop() {
        let mut env = SpectatorEnv::new(500);
        let mapper = DefaultActionMapper;
        let illegal_drop = Move::Drop {
            to: Square::from_row_col(4, 4).unwrap(),
            piece_type: HandPieceType::Pawn,
        };
        let action = mapper
            .encode(illegal_drop, Color::Black)
            .expect("drop should be encodable");

        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let err = env.step(py, action).expect_err("illegal drop must error");
            assert!(
                err.to_string().contains("not legal"),
                "unexpected error: {}",
                err
            );
        });
    }

    /// Test that legal_actions at startpos returns 30 entries.
    #[test]
    fn test_spectator_env_legal_actions_count() {
        let mut game = GameState::with_max_ply(500);
        let mapper = DefaultActionMapper;
        let perspective = game.position.current_player;

        let legal = game.legal_moves();
        let actions: Vec<usize> = legal
            .iter()
            .map(|mv| mapper.encode(*mv, perspective).unwrap())
            .collect();

        assert_eq!(actions.len(), 30, "Startpos should have 30 legal actions");

        // All action indices should be unique
        let unique: std::collections::HashSet<usize> = actions.iter().copied().collect();
        assert_eq!(unique.len(), 30, "All 30 action indices should be unique");
    }

    /// After a game reaches terminal state, verify is_terminal() returns true.
    #[test]
    fn test_spectator_env_game_over_detection() {
        // Use max_ply=0 to immediately end the game
        let mut game = GameState::with_max_ply(0);
        game.check_termination();
        assert!(game.result.is_terminal(), "Game with max_ply=0 should be over");
    }

    /// Test hand_piece_char covers all types correctly.
    #[test]
    fn test_hand_piece_char_all_types() {
        assert_eq!(hand_piece_char(HandPieceType::Pawn), 'P');
        assert_eq!(hand_piece_char(HandPieceType::Lance), 'L');
        assert_eq!(hand_piece_char(HandPieceType::Knight), 'N');
        assert_eq!(hand_piece_char(HandPieceType::Silver), 'S');
        assert_eq!(hand_piece_char(HandPieceType::Gold), 'G');
        assert_eq!(hand_piece_char(HandPieceType::Bishop), 'B');
        assert_eq!(hand_piece_char(HandPieceType::Rook), 'R');
    }

}
