use numpy::{PyArray3, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use shogi_core::{GameState, HandPieceType, Move};

use crate::action_mapper::{ActionMapper, DefaultActionMapper};
use crate::observation::{DefaultObservationGenerator, ObservationGenerator, BUFFER_LEN, NUM_CHANNELS};
use crate::spectator_data::{build_spectator_dict, color_name};

// ---------------------------------------------------------------------------
// Helper functions (spectator-only)
// ---------------------------------------------------------------------------

/// Encode a drop piece char: P, L, N, S, G, B, R
fn hand_piece_char(hpt: HandPieceType) -> char {
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
fn move_notation(mv: Move) -> String {
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
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

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
            .map(|mv| self.mapper.encode(mv, perspective))
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
