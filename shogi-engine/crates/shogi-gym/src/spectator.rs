use numpy::{PyArray3, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use shogi_core::GameState;

use crate::action_mapper::{ActionMapper, DefaultActionMapper, ACTION_SPACE_SIZE};
use crate::observation::{DefaultObservationGenerator, ObservationGenerator, BUFFER_LEN, NUM_CHANNELS};
use crate::spatial_action_mapper::{SpatialActionMapper, SPATIAL_ACTION_SPACE_SIZE};
use crate::spectator_data::{build_spectator_dict, color_name, move_notation};

// ---------------------------------------------------------------------------
// ActionMode dispatch (mirrors vec_env.rs)
// ---------------------------------------------------------------------------

enum SpectatorActionMode {
    Default(DefaultActionMapper),
    Spatial(SpatialActionMapper),
}

impl SpectatorActionMode {
    fn action_space_size(&self) -> usize {
        match self {
            SpectatorActionMode::Default(_) => ACTION_SPACE_SIZE,
            SpectatorActionMode::Spatial(_) => SPATIAL_ACTION_SPACE_SIZE,
        }
    }

    fn encode(&self, mv: shogi_core::Move, perspective: shogi_core::Color) -> Result<usize, String> {
        match self {
            SpectatorActionMode::Default(m) => <DefaultActionMapper as ActionMapper>::encode(m, mv, perspective),
            SpectatorActionMode::Spatial(m) => <SpatialActionMapper as ActionMapper>::encode(m, mv, perspective),
        }
    }

    fn decode(&self, idx: usize, perspective: shogi_core::Color) -> Result<shogi_core::Move, String> {
        match self {
            SpectatorActionMode::Default(m) => <DefaultActionMapper as ActionMapper>::decode(m, idx, perspective),
            SpectatorActionMode::Spatial(m) => <SpatialActionMapper as ActionMapper>::decode(m, idx, perspective),
        }
    }
}

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
    mapper: SpectatorActionMode,
    obs_gen: DefaultObservationGenerator,
    move_history: Vec<(usize, String)>,  // (action_index, move_notation)
}

#[pymethods]
impl SpectatorEnv {
    /// Create a new SpectatorEnv.
    ///
    /// Args:
    ///     max_ply: Maximum number of plies before the game ends (default 500).
    ///     action_mode: "default" (13527 actions) or "spatial" (11259, matches CNN policy head).
    #[new]
    #[pyo3(signature = (max_ply = 500, action_mode = "default"))]
    pub fn new(max_ply: u32, action_mode: &str) -> PyResult<Self> {
        let mapper = match action_mode {
            "default" => SpectatorActionMode::Default(DefaultActionMapper),
            "spatial" => SpectatorActionMode::Spatial(SpatialActionMapper::new()),
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown action_mode '{}'. Valid: 'default', 'spatial'", other)
            )),
        };
        Ok(SpectatorEnv {
            game: GameState::with_max_ply(max_ply),
            max_ply,
            mapper,
            obs_gen: DefaultObservationGenerator::new(),
            move_history: Vec::new(),
        })
    }

    /// Create a SpectatorEnv from a SFEN string.
    ///
    /// Args:
    ///     sfen: SFEN position string.
    ///     max_ply: Maximum plies before truncation (default 500).
    ///
    /// Raises ValueError if the SFEN is invalid.
    #[staticmethod]
    #[pyo3(signature = (sfen, max_ply = None, action_mode = "default"))]
    pub fn from_sfen(sfen: &str, max_ply: Option<u32>, action_mode: &str) -> PyResult<Self> {
        let max_ply = max_ply.unwrap_or(500);
        let mapper = match action_mode {
            "default" => SpectatorActionMode::Default(DefaultActionMapper),
            "spatial" => SpectatorActionMode::Spatial(SpatialActionMapper::new()),
            other => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown action_mode '{}'. Valid: 'default', 'spatial'", other)
            )),
        };
        let game = GameState::from_sfen(sfen, max_ply)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("invalid SFEN: {e}")))?;
        Ok(SpectatorEnv {
            game,
            max_ply,
            mapper,
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
        let mv = self.mapper.decode(action, perspective)
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

    /// Return all legal moves at the current position with their USI strings.
    /// Read-only; no state mutation. Order matches `legal_actions()`.
    pub fn legal_moves_with_usi(&mut self) -> Vec<(usize, String)> {
        use crate::spectator_data::move_usi;
        let perspective = self.game.position.current_player;
        let moves = self.game.legal_moves();
        moves
            .into_iter()
            .map(|mv| {
                let idx = self.mapper.encode(mv, perspective)
                    .expect("legal move must be encodable");
                (idx, move_usi(mv))
            })
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

    /// Total number of actions in the action space.
    /// 13527 for "default", 11259 for "spatial".
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
    use crate::action_mapper::ActionMapper;
    use pyo3::Python;
    use shogi_core::{Color, GameState, HandPieceType, Move, Square};

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
        let mut env = SpectatorEnv {
            game: GameState::with_max_ply(500),
            max_ply: 500,
            mapper: SpectatorActionMode::Default(DefaultActionMapper),
            obs_gen: DefaultObservationGenerator::new(),
            move_history: Vec::new(),
        };
        let illegal_drop = Move::Drop {
            to: Square::from_row_col(4, 4).unwrap(),
            piece_type: HandPieceType::Pawn,
        };
        let action = DefaultActionMapper
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

}
