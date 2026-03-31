//! VecEnv — vectorized batch environment for N Shogi games.
//!
//! Pre-allocates all output buffers at construction time and writes in-place
//! each step. Uses a two-phase step contract:
//!   Phase 1: decode + validate all N actions (read-only, no state mutation)
//!   Phase 2: apply all moves with GIL released via `py.allow_threads()`
//!
//! Auto-resets terminated/truncated games, saving terminal observations in a
//! separate buffer before resetting.

use crate::action_mapper::{ActionMapper, DefaultActionMapper, ACTION_SPACE_SIZE};
use crate::observation::{
    DefaultObservationGenerator, ObservationGenerator, BUFFER_LEN, NUM_CHANNELS,
};
use crate::spectator_data::build_spectator_dict;
use crate::step_result::{ResetResult, StepMetadata, StepResult, TerminationReason};

use numpy::{PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use shogi_core::{Color, GameResult, GameState, HandPieceType, Move, MoveList};
use std::sync::atomic::{AtomicU64, Ordering};

/// Minimum number of environments to use rayon parallel iteration.
const PARALLEL_THRESHOLD: usize = 64;

/// Wrapper that asserts a raw pointer may be sent across threads.
///
/// # Safety
/// The caller must guarantee that accesses through this pointer are
/// non-overlapping across threads (e.g. each thread uses a disjoint index).
#[derive(Copy, Clone)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    #[inline]
    unsafe fn offset(&self, count: usize) -> *mut T {
        unsafe { self.0.add(count) }
    }
}

// ---------------------------------------------------------------------------
// Reward computation
// ---------------------------------------------------------------------------

/// Compute the scalar reward for the player who just moved.
///
/// `last_mover` is the color of the player who made the move (i.e.
/// `game.position.current_player.opponent()` after `make_move`, since
/// `make_move` flips the current player).
fn compute_reward(result: &GameResult, last_mover: Color) -> f32 {
    match result {
        GameResult::Checkmate { winner } | GameResult::PerpetualCheck { winner } => {
            if *winner == last_mover {
                1.0
            } else {
                -1.0
            }
        }
        GameResult::Impasse {
            winner: Some(winner),
        } => {
            if *winner == last_mover {
                1.0
            } else {
                -1.0
            }
        }
        GameResult::Impasse { winner: None }
        | GameResult::Repetition
        | GameResult::MaxMoves
        | GameResult::InProgress => 0.0,
    }
}

// ---------------------------------------------------------------------------
// VecEnv
// ---------------------------------------------------------------------------

#[pyclass]
pub struct VecEnv {
    games: Vec<GameState>,
    num_envs: usize,
    max_ply: u32,

    // Rust-owned flat buffers written in-place each step
    obs_buffer: Vec<f32>,          // N * BUFFER_LEN
    legal_mask_buffer: Vec<bool>,  // N * ACTION_SPACE_SIZE
    reward_buffer: Vec<f32>,       // N
    terminated_buffer: Vec<bool>,  // N
    truncated_buffer: Vec<bool>,   // N
    captured_buffer: Vec<u8>,      // N (metadata)
    term_reason_buffer: Vec<u8>,   // N (metadata)
    ply_buffer: Vec<u16>,          // N (metadata)
    terminal_obs_buffer: Vec<f32>,   // N * BUFFER_LEN
    current_players_buffer: Vec<u8>, // N (0=Black, 1=White)

    mapper: DefaultActionMapper,
    obs_gen: DefaultObservationGenerator,

    // Episode tracking counters (atomic for rayon safety)
    episodes_completed: AtomicU64,
    episodes_drawn: AtomicU64,     // Repetition or Impasse(None)
    episodes_truncated: AtomicU64, // MaxMoves
    total_episode_ply: AtomicU64,  // sum of ply at episode end (for mean_episode_length)
}

impl VecEnv {
    /// Write observation and legal mask for environment `i` into the main buffers.
    fn write_obs_and_mask(&mut self, i: usize) {
        let perspective = self.games[i].position.current_player;

        // Write observation
        let obs_start = i * BUFFER_LEN;
        let obs_slice = &mut self.obs_buffer[obs_start..obs_start + BUFFER_LEN];
        self.obs_gen.generate(&self.games[i], perspective, obs_slice);

        // Write legal mask
        let mask_start = i * ACTION_SPACE_SIZE;
        let mask_slice = &mut self.legal_mask_buffer[mask_start..mask_start + ACTION_SPACE_SIZE];
        mask_slice.fill(false);
        let mut move_list = MoveList::new();
        self.games[i].generate_legal_moves_into(&mut move_list);
        for mv in move_list.iter() {
            let idx = self.mapper.encode(*mv, perspective);
            mask_slice[idx] = true;
        }
    }

}

#[pymethods]
impl VecEnv {
    /// Create a new VecEnv with `num_envs` parallel games.
    #[new]
    #[pyo3(signature = (num_envs = 512, max_ply = 500))]
    pub fn new(num_envs: usize, max_ply: u32) -> Self {
        let games: Vec<GameState> = (0..num_envs)
            .map(|_| GameState::with_max_ply(max_ply))
            .collect();

        VecEnv {
            games,
            num_envs,
            max_ply,
            obs_buffer: vec![0.0; num_envs * BUFFER_LEN],
            legal_mask_buffer: vec![false; num_envs * ACTION_SPACE_SIZE],
            reward_buffer: vec![0.0; num_envs],
            terminated_buffer: vec![false; num_envs],
            truncated_buffer: vec![false; num_envs],
            captured_buffer: vec![255; num_envs],
            term_reason_buffer: vec![0; num_envs],
            ply_buffer: vec![0; num_envs],
            terminal_obs_buffer: vec![0.0; num_envs * BUFFER_LEN],
            current_players_buffer: vec![0; num_envs],
            mapper: DefaultActionMapper,
            obs_gen: DefaultObservationGenerator::new(),
            episodes_completed: AtomicU64::new(0),
            episodes_drawn: AtomicU64::new(0),
            episodes_truncated: AtomicU64::new(0),
            total_episode_ply: AtomicU64::new(0),
        }
    }

    /// Reset all games to the starting position.
    ///
    /// Returns a `ResetResult` with initial observations and legal masks.
    pub fn reset(&mut self, py: Python<'_>) -> PyResult<ResetResult> {
        // Reset all games
        for i in 0..self.num_envs {
            self.games[i] = GameState::with_max_ply(self.max_ply);
        }

        // Write initial obs + legal masks for all games
        for i in 0..self.num_envs {
            self.write_obs_and_mask(i);
        }

        // Build numpy arrays
        let obs_array = self.obs_buffer.to_pyarray(py);
        let obs_4d = obs_array
            .reshape([self.num_envs, NUM_CHANNELS, 9, 9])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mask_array = self.legal_mask_buffer.to_pyarray(py);
        let mask_2d = mask_array
            .reshape([self.num_envs, ACTION_SPACE_SIZE])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(ResetResult {
            observations: obs_4d.unbind(),
            legal_masks: mask_2d.unbind(),
        })
    }

    /// Step all environments with the given actions.
    ///
    /// Two-phase contract:
    ///   Phase 1: Decode and validate all N actions (no state mutation).
    ///   Phase 2: Apply all moves with GIL released.
    pub fn step(&mut self, py: Python<'_>, actions: Vec<i64>) -> PyResult<StepResult> {
        // --- Phase 1: Validate (read-only) ---

        if actions.len() != self.num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} actions, got {}",
                self.num_envs,
                actions.len()
            )));
        }

        // Decode all actions and validate against legal masks
        let mut decoded_moves: Vec<Move> = Vec::with_capacity(self.num_envs);
        for (i, action) in actions.iter().enumerate() {
            let action_idx = *action as usize;
            let perspective = self.games[i].position.current_player;

            let mv = <DefaultActionMapper as ActionMapper>::decode(
                &self.mapper,
                action_idx,
                perspective,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "env {}: invalid action index {}: {}",
                    i, action_idx, e
                ))
            })?;

            // Check legal mask
            let mask_start = i * ACTION_SPACE_SIZE;
            if !self.legal_mask_buffer[mask_start + action_idx] {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "env {}: action index {} is not legal",
                    i, action_idx
                )));
            }

            decoded_moves.push(mv);
        }

        // --- Phase 2: Apply (GIL released) ---

        py.allow_threads(|| {
            let num_envs = self.num_envs;
            let max_ply = self.max_ply;

            // Extract raw pointers for non-overlapping parallel access.
            // SAFETY: each index `i` in 0..num_envs accesses only its own
            // non-overlapping slice of each buffer and its own games[i].
            let games_ptr = SendPtr(self.games.as_mut_ptr());
            let obs_ptr = SendPtr(self.obs_buffer.as_mut_ptr());
            let mask_ptr = SendPtr(self.legal_mask_buffer.as_mut_ptr());
            let terminal_obs_ptr = SendPtr(self.terminal_obs_buffer.as_mut_ptr());
            let reward_ptr = SendPtr(self.reward_buffer.as_mut_ptr());
            let terminated_ptr = SendPtr(self.terminated_buffer.as_mut_ptr());
            let truncated_ptr = SendPtr(self.truncated_buffer.as_mut_ptr());
            let captured_ptr = SendPtr(self.captured_buffer.as_mut_ptr());
            let term_reason_ptr = SendPtr(self.term_reason_buffer.as_mut_ptr());
            let ply_ptr = SendPtr(self.ply_buffer.as_mut_ptr());
            let current_players_ptr = SendPtr(self.current_players_buffer.as_mut_ptr());

            // Episode counters (atomic — safe for parallel access)
            let ep_completed = &self.episodes_completed;
            let ep_drawn = &self.episodes_drawn;
            let ep_truncated = &self.episodes_truncated;
            let ep_total_ply = &self.total_episode_ply;

            // Local copies — both are stateless. Shared refs used in closure.
            let mapper = DefaultActionMapper;
            let obs_gen = DefaultObservationGenerator::new();
            let mapper_ref = &mapper;
            let obs_gen_ref = &obs_gen;
            let decoded = &decoded_moves;

            let process_env = move |i: usize| {
                // SAFETY: each `i` accesses non-overlapping memory regions.
                unsafe {
                    let game = &mut *games_ptr.offset(i);
                    let mv = decoded[i];

                    // Apply move
                    let undo_info = game.make_move(mv);

                    // last_mover is the player who just moved (current_player has flipped)
                    let last_mover = game.position.current_player.opponent();

                    // Check termination
                    game.check_termination();

                    let result = game.result;
                    let terminated = result.is_terminal() && !result.is_truncation();
                    let truncated = result.is_truncation();

                    // Set scalar metadata buffers
                    *terminated_ptr.offset(i) = terminated;
                    *truncated_ptr.offset(i) = truncated;
                    *reward_ptr.offset(i) = compute_reward(&result, last_mover);
                    *term_reason_ptr.offset(i) = TerminationReason::from_game_result(result) as u8;
                    *ply_ptr.offset(i) = game.ply as u16;

                    // Captured piece metadata
                    if let Some(captured_piece) = undo_info.captured {
                        let pt = captured_piece.piece_type();
                        match HandPieceType::from_piece_type(pt) {
                            Some(hpt) => *captured_ptr.offset(i) = hpt.index() as u8,
                            None => *captured_ptr.offset(i) = 255,
                        }
                    } else {
                        *captured_ptr.offset(i) = 255;
                    }

                    if terminated || truncated {
                        // Update episode counters
                        ep_completed.fetch_add(1, Ordering::Relaxed);
                        ep_total_ply.fetch_add(game.ply as u64, Ordering::Relaxed);
                        match result {
                            GameResult::Repetition
                            | GameResult::Impasse { winner: None } => {
                                ep_drawn.fetch_add(1, Ordering::Relaxed);
                            }
                            GameResult::MaxMoves => {
                                ep_truncated.fetch_add(1, Ordering::Relaxed);
                            }
                            _ => {}
                        }

                        // Save terminal observation
                        let term_obs_slice = std::slice::from_raw_parts_mut(
                            terminal_obs_ptr.offset(i * BUFFER_LEN),
                            BUFFER_LEN,
                        );
                        let perspective = game.position.current_player;
                        obs_gen_ref.generate(game, perspective, term_obs_slice);

                        // Auto-reset
                        *game = GameState::with_max_ply(max_ply);
                    }

                    // Write obs+mask for current game state (reset or continuing)
                    let perspective = game.position.current_player;

                    let obs_slice = std::slice::from_raw_parts_mut(
                        obs_ptr.offset(i * BUFFER_LEN),
                        BUFFER_LEN,
                    );
                    obs_gen_ref.generate(game, perspective, obs_slice);

                    let mask_slice = std::slice::from_raw_parts_mut(
                        mask_ptr.offset(i * ACTION_SPACE_SIZE),
                        ACTION_SPACE_SIZE,
                    );
                    mask_slice.fill(false);
                    let mut move_list = MoveList::new();
                    game.generate_legal_moves_into(&mut move_list);
                    for legal_mv in move_list.iter() {
                        let idx = mapper_ref.encode(*legal_mv, perspective);
                        mask_slice[idx] = true;
                    }

                    // Record current player (after potential reset)
                    *current_players_ptr.offset(i) = match game.position.current_player {
                        Color::Black => 0,
                        Color::White => 1,
                    };
                }
            };

            if num_envs >= PARALLEL_THRESHOLD {
                (0..num_envs).into_par_iter().for_each(process_env);
            } else {
                (0..num_envs).for_each(process_env);
            }
        });

        // --- Build Python result (GIL held) ---

        let obs_array = self.obs_buffer.to_pyarray(py);
        let obs_4d = obs_array
            .reshape([self.num_envs, NUM_CHANNELS, 9, 9])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mask_array = self.legal_mask_buffer.to_pyarray(py);
        let mask_2d = mask_array
            .reshape([self.num_envs, ACTION_SPACE_SIZE])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let rewards = self.reward_buffer.to_pyarray(py);
        let terminated = self.terminated_buffer.to_pyarray(py);
        let truncated = self.truncated_buffer.to_pyarray(py);

        let terminal_obs_array = self.terminal_obs_buffer.to_pyarray(py);
        let terminal_obs_4d = terminal_obs_array
            .reshape([self.num_envs, NUM_CHANNELS, 9, 9])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let current_players = self.current_players_buffer.to_pyarray(py);

        let captured_arr = self.captured_buffer.to_pyarray(py);
        let term_reason_arr = self.term_reason_buffer.to_pyarray(py);
        let ply_arr = self.ply_buffer.to_pyarray(py);

        let metadata = Py::new(
            py,
            StepMetadata {
                captured_piece: captured_arr.unbind(),
                termination_reason: term_reason_arr.unbind(),
                ply_count: ply_arr.unbind(),
            },
        )?;

        Ok(StepResult {
            observations: obs_4d.unbind(),
            legal_masks: mask_2d.unbind(),
            rewards: rewards.unbind(),
            terminated: terminated.unbind(),
            truncated: truncated.unbind(),
            terminal_observations: terminal_obs_4d.unbind(),
            current_players: current_players.unbind(),
            step_metadata: metadata,
        })
    }

    /// Total number of actions in the action space.
    #[getter]
    pub fn action_space_size(&self) -> usize {
        ACTION_SPACE_SIZE
    }

    /// Number of observation channels.
    #[getter]
    pub fn observation_channels(&self) -> usize {
        NUM_CHANNELS
    }

    /// Number of parallel environments.
    #[getter]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Total episodes completed since construction (or last reset_stats).
    #[getter]
    pub fn episodes_completed(&self) -> u64 {
        self.episodes_completed.load(Ordering::Relaxed)
    }

    /// Episodes ending in a draw (Repetition or Impasse with no winner).
    #[getter]
    pub fn episodes_drawn(&self) -> u64 {
        self.episodes_drawn.load(Ordering::Relaxed)
    }

    /// Episodes truncated by max_ply limit.
    #[getter]
    pub fn episodes_truncated(&self) -> u64 {
        self.episodes_truncated.load(Ordering::Relaxed)
    }

    /// Draw rate: episodes_drawn / episodes_completed (0.0 if none completed).
    #[getter]
    pub fn draw_rate(&self) -> f64 {
        let completed = self.episodes_completed.load(Ordering::Relaxed);
        if completed == 0 {
            0.0
        } else {
            self.episodes_drawn.load(Ordering::Relaxed) as f64 / completed as f64
        }
    }

    /// Mean episode length across all completed episodes since last reset_stats().
    #[getter]
    pub fn mean_episode_length(&self) -> f64 {
        let completed = self.episodes_completed.load(Ordering::Relaxed);
        if completed == 0 {
            0.0
        } else {
            self.total_episode_ply.load(Ordering::Relaxed) as f64 / completed as f64
        }
    }

    /// Fraction of completed episodes that were truncated (hit max_ply).
    #[getter]
    pub fn truncation_rate(&self) -> f64 {
        let completed = self.episodes_completed.load(Ordering::Relaxed);
        if completed == 0 {
            0.0
        } else {
            self.episodes_truncated.load(Ordering::Relaxed) as f64 / completed as f64
        }
    }

    /// Reset episode counters to zero.
    pub fn reset_stats(&self) {
        self.episodes_completed.store(0, Ordering::Relaxed);
        self.episodes_drawn.store(0, Ordering::Relaxed);
        self.episodes_truncated.store(0, Ordering::Relaxed);
        self.total_episode_ply.store(0, Ordering::Relaxed);
    }

    /// Get the SFEN string for a single game by index.
    ///
    /// Raises IndexError if game_id >= num_envs.
    pub fn get_sfen(&self, game_id: usize) -> PyResult<String> {
        if game_id >= self.num_envs {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "game_id {} out of range for {} environments",
                game_id, self.num_envs
            )));
        }
        Ok(self.games[game_id].position.to_sfen())
    }

    /// Get SFEN strings for all games.
    pub fn get_sfens(&self) -> Vec<String> {
        self.games.iter().map(|g| g.position.to_sfen()).collect()
    }

    /// Return spectator-format dicts for all games.
    pub fn get_spectator_data(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut result = Vec::with_capacity(self.num_envs);
        for game in &self.games {
            result.push(build_spectator_dict(py, game)?);
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::Color;

    // -----------------------------------------------------------------------
    // compute_reward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reward_checkmate_winner_is_last_mover() {
        let result = GameResult::Checkmate {
            winner: Color::Black,
        };
        assert_eq!(compute_reward(&result, Color::Black), 1.0);
    }

    #[test]
    fn test_reward_checkmate_winner_is_not_last_mover() {
        let result = GameResult::Checkmate {
            winner: Color::White,
        };
        assert_eq!(compute_reward(&result, Color::Black), -1.0);
    }

    #[test]
    fn test_reward_perpetual_check_winner_is_last_mover() {
        let result = GameResult::PerpetualCheck {
            winner: Color::White,
        };
        assert_eq!(compute_reward(&result, Color::White), 1.0);
    }

    #[test]
    fn test_reward_perpetual_check_winner_is_not_last_mover() {
        let result = GameResult::PerpetualCheck {
            winner: Color::Black,
        };
        assert_eq!(compute_reward(&result, Color::White), -1.0);
    }

    #[test]
    fn test_reward_impasse_with_winner_is_last_mover() {
        let result = GameResult::Impasse {
            winner: Some(Color::Black),
        };
        assert_eq!(compute_reward(&result, Color::Black), 1.0);
    }

    #[test]
    fn test_reward_impasse_with_winner_is_not_last_mover() {
        let result = GameResult::Impasse {
            winner: Some(Color::White),
        };
        assert_eq!(compute_reward(&result, Color::Black), -1.0);
    }

    #[test]
    fn test_reward_impasse_draw() {
        let result = GameResult::Impasse { winner: None };
        assert_eq!(compute_reward(&result, Color::Black), 0.0);
    }

    #[test]
    fn test_reward_repetition() {
        let result = GameResult::Repetition;
        assert_eq!(compute_reward(&result, Color::Black), 0.0);
        assert_eq!(compute_reward(&result, Color::White), 0.0);
    }

    #[test]
    fn test_reward_max_moves() {
        let result = GameResult::MaxMoves;
        assert_eq!(compute_reward(&result, Color::Black), 0.0);
        assert_eq!(compute_reward(&result, Color::White), 0.0);
    }

    #[test]
    fn test_reward_in_progress() {
        let result = GameResult::InProgress;
        assert_eq!(compute_reward(&result, Color::Black), 0.0);
        assert_eq!(compute_reward(&result, Color::White), 0.0);
    }

    // -----------------------------------------------------------------------
    // VecEnv construction tests (no Python needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_buffer_sizes() {
        let env = VecEnv::new(4, 100);
        assert_eq!(env.num_envs, 4);
        assert_eq!(env.max_ply, 100);
        assert_eq!(env.obs_buffer.len(), 4 * BUFFER_LEN);
        assert_eq!(env.legal_mask_buffer.len(), 4 * ACTION_SPACE_SIZE);
        assert_eq!(env.reward_buffer.len(), 4);
        assert_eq!(env.terminated_buffer.len(), 4);
        assert_eq!(env.truncated_buffer.len(), 4);
        assert_eq!(env.captured_buffer.len(), 4);
        assert_eq!(env.term_reason_buffer.len(), 4);
        assert_eq!(env.ply_buffer.len(), 4);
        assert_eq!(env.terminal_obs_buffer.len(), 4 * BUFFER_LEN);
        assert_eq!(env.games.len(), 4);
    }

    #[test]
    fn test_write_obs_and_mask_startpos() {
        let mut env = VecEnv::new(2, 500);
        env.write_obs_and_mask(0);

        // At startpos, Black moves first. There should be legal moves.
        let mask_start = 0;
        let mask_end = ACTION_SPACE_SIZE;
        let legal_count: usize = env.legal_mask_buffer[mask_start..mask_end]
            .iter()
            .filter(|&&x| x)
            .count();
        assert!(
            legal_count > 0,
            "startpos should have legal moves, got {}",
            legal_count
        );

        // Standard Shogi startpos has 30 legal moves
        assert_eq!(legal_count, 30, "startpos should have exactly 30 legal moves");
    }

    #[test]
    fn test_default_metadata_buffers() {
        let env = VecEnv::new(3, 500);
        for i in 0..3 {
            assert_eq!(env.captured_buffer[i], 255);
            assert_eq!(env.term_reason_buffer[i], 0);
            assert_eq!(env.ply_buffer[i], 0);
        }
    }

    // -----------------------------------------------------------------------
    // VecEnv: legal mask matches legal_moves after a move
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_obs_and_mask_after_move() {
        let mut env = VecEnv::new(1, 500);

        // Get Black's first legal move and apply it
        let first_move = env.games[0].legal_moves()[0];
        env.games[0].make_move(first_move);

        // Now it's White's turn. Write obs and mask.
        env.write_obs_and_mask(0);

        // Count legal moves from the mask
        let mask_start = 0;
        let mask_end = ACTION_SPACE_SIZE;
        let mask_legal_count: usize = env.legal_mask_buffer[mask_start..mask_end]
            .iter()
            .filter(|&&x| x)
            .count();

        // Compare with legal_moves()
        let legal_moves = env.games[0].legal_moves();
        assert_eq!(
            mask_legal_count,
            legal_moves.len(),
            "After one move, legal mask count ({}) should match legal_moves count ({})",
            mask_legal_count,
            legal_moves.len()
        );

        // White's response should also be 30 moves (symmetric position after one pawn push)
        assert_eq!(
            legal_moves.len(), 30,
            "White's first move should also have 30 options"
        );
    }

    #[test]
    fn test_legal_mask_encodes_correct_action_indices() {
        let mut env = VecEnv::new(1, 500);
        env.write_obs_and_mask(0);

        let perspective = env.games[0].position.current_player;
        let legal_moves = env.games[0].legal_moves();

        // Every legal move's encoded action index should be true in the mask
        for mv in &legal_moves {
            let idx = env.mapper.encode(*mv, perspective);
            assert!(
                env.legal_mask_buffer[idx],
                "Legal move {:?} encoded to index {} but mask is false",
                mv,
                idx
            );
        }

        // Count of true bits should equal number of legal moves
        let true_count: usize = env.legal_mask_buffer[..ACTION_SPACE_SIZE]
            .iter()
            .filter(|&&x| x)
            .count();
        assert_eq!(
            true_count,
            legal_moves.len(),
            "Mask has {} true bits but {} legal moves",
            true_count,
            legal_moves.len()
        );
    }

    #[test]
    fn test_observation_buffer_nonzero_after_write() {
        let mut env = VecEnv::new(1, 500);
        env.write_obs_and_mask(0);

        // The observation buffer should have non-zero values (pieces on board)
        let obs_slice = &env.obs_buffer[..BUFFER_LEN];
        let nonzero_count = obs_slice.iter().filter(|&&x| x != 0.0).count();
        assert!(
            nonzero_count > 0,
            "Observation buffer should have non-zero values after write"
        );

        // Channel 42 (player indicator) should be all 1.0 for Black
        let ch42_start = 42 * 81;
        for i in 0..81 {
            assert_eq!(
                obs_slice[ch42_start + i], 1.0,
                "Channel 42 should be 1.0 for Black at startpos"
            );
        }
    }

    // -----------------------------------------------------------------------
    // VecEnv: manual step simulation (no Python needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_manual_step_simulation() {
        let mut env = VecEnv::new(1, 500);
        env.write_obs_and_mask(0);

        // Simulate 10 moves: pick first legal action each time
        for step in 0..10 {
            let legal_moves = env.games[0].legal_moves();
            assert!(!legal_moves.is_empty(), "No legal moves at step {}", step);

            let mv = legal_moves[0];
            let _undo_info = env.games[0].make_move(mv);
            let last_mover = env.games[0].position.current_player.opponent();

            env.games[0].check_termination();

            let result = env.games[0].result;
            let reward = compute_reward(&result, last_mover);

            if result.is_terminal() {
                // Reward should be non-zero for decisive results
                // (may be 0.0 for draws/truncation)
                break;
            } else {
                assert_eq!(reward, 0.0, "In-progress reward should be 0.0");
            }

            // Write next obs and mask
            env.write_obs_and_mask(0);

            // Verify mask is consistent
            let new_legal = env.games[0].legal_moves();
            let mask_count: usize = env.legal_mask_buffer[..ACTION_SPACE_SIZE]
                .iter()
                .filter(|&&x| x)
                .count();
            assert_eq!(
                mask_count,
                new_legal.len(),
                "Mask count mismatch at step {}",
                step
            );
        }
    }

    #[test]
    fn test_max_ply_termination() {
        // Create env with max_ply=2 — game should end after 2 moves
        let mut env = VecEnv::new(1, 2);
        env.write_obs_and_mask(0);

        // Step 1
        let mv1 = env.games[0].legal_moves()[0];
        env.games[0].make_move(mv1);
        env.games[0].check_termination();
        assert_eq!(env.games[0].result, GameResult::InProgress);

        // Step 2
        let mv2 = env.games[0].legal_moves()[0];
        env.games[0].make_move(mv2);
        env.games[0].check_termination();
        assert_eq!(env.games[0].result, GameResult::MaxMoves);

        let reward = compute_reward(&env.games[0].result, env.games[0].position.current_player.opponent());
        assert_eq!(reward, 0.0, "MaxMoves should give 0.0 reward");
    }
}
