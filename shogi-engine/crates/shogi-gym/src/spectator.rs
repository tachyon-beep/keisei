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

        let notation = move_notation(mv);
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
    use shogi_core::{GameState, Color, HandPieceType, Move, Square};
    use crate::action_mapper::ActionMapper;
    use crate::spectator_data::{hand_piece_char, move_notation};

    // -----------------------------------------------------------------------
    // move_notation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_move_notation_board_move() {
        // Board move: from (6,4) to (5,4), no promotion
        // from: col=4 → 9-4=5, row=6 → 'g'. So "5g"
        // to: col=4 → 9-4=5, row=5 → 'f'. So "5f"
        let mv = Move::Board {
            from: Square::from_row_col(6, 4).unwrap(),
            to: Square::from_row_col(5, 4).unwrap(),
            promote: false,
        };
        let notation = move_notation(mv);
        assert_eq!(notation, "5g→5f", "Board move notation mismatch");
    }

    #[test]
    fn test_move_notation_board_move_with_promotion() {
        let mv = Move::Board {
            from: Square::from_row_col(1, 2).unwrap(),
            to: Square::from_row_col(0, 2).unwrap(),
            promote: true,
        };
        let notation = move_notation(mv);
        assert_eq!(notation, "7b→7a+", "Promoting move should end with '+'");
    }

    #[test]
    fn test_move_notation_drop() {
        // Drop pawn at (4,4): piece='P', col=4 → 9-4=5, row=4 → 'e'
        let mv = Move::Drop {
            to: Square::from_row_col(4, 4).unwrap(),
            piece_type: HandPieceType::Pawn,
        };
        let notation = move_notation(mv);
        assert_eq!(notation, "P*5e", "Drop notation mismatch");
    }

    #[test]
    fn test_move_notation_drop_all_piece_types() {
        let to = Square::from_row_col(4, 4).unwrap();
        let expected = [
            (HandPieceType::Pawn, "P*5e"),
            (HandPieceType::Lance, "L*5e"),
            (HandPieceType::Knight, "N*5e"),
            (HandPieceType::Silver, "S*5e"),
            (HandPieceType::Gold, "G*5e"),
            (HandPieceType::Bishop, "B*5e"),
            (HandPieceType::Rook, "R*5e"),
        ];
        for (hpt, exp) in &expected {
            let mv = Move::Drop { to, piece_type: *hpt };
            assert_eq!(move_notation(mv), *exp, "Drop notation for {:?}", hpt);
        }
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

    #[test]
    fn test_move_notation_boundary_squares() {
        // Top-right corner: row=0, col=0 → "9a"
        // Bottom-left corner: row=8, col=8 → "1i"
        let mv_top_right = Move::Board {
            from: Square::from_row_col(0, 0).unwrap(),
            to: Square::from_row_col(1, 0).unwrap(),
            promote: false,
        };
        let notation = move_notation(mv_top_right);
        assert_eq!(notation, "9a→9b", "Top-right corner notation mismatch: got {}", notation);

        let mv_bottom_left = Move::Board {
            from: Square::from_row_col(8, 8).unwrap(),
            to: Square::from_row_col(7, 8).unwrap(),
            promote: false,
        };
        let notation = move_notation(mv_bottom_left);
        assert_eq!(notation, "1i→1h", "Bottom-left corner notation mismatch: got {}", notation);

        // Drop at corner
        let mv_drop_corner = Move::Drop {
            to: Square::from_row_col(0, 8).unwrap(),
            piece_type: HandPieceType::Pawn,
        };
        let notation = move_notation(mv_drop_corner);
        assert_eq!(notation, "P*1a", "Drop at corner notation mismatch: got {}", notation);
    }
}
