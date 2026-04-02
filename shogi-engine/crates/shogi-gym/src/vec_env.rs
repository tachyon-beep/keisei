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
use crate::katago_observation::{
    KataGoObservationGenerator, KATAGO_BUFFER_LEN, KATAGO_NUM_CHANNELS,
};
use crate::observation::{
    DefaultObservationGenerator, ObservationGenerator, BUFFER_LEN, NUM_CHANNELS,
};
use crate::spatial_action_mapper::{SpatialActionMapper, SPATIAL_ACTION_SPACE_SIZE};
use crate::spectator_data::build_spectator_dict;
use crate::step_result::{ResetResult, StepMetadata, StepResult, TerminationReason};

use numpy::{PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use shogi_core::{Color, GameResult, GameState, HandPieceType, Move, MoveList};
use shogi_core::rules::material_balance;
use std::sync::atomic::{AtomicU64, Ordering};

// Compile-time assertion: observation generators and action mappers must be ZSTs.
// The rayon closure reconstructs these per-thread via ::new(). If a future generator
// acquires state, this assertion will fail, forcing explicit resolution of the
// sharing design (e.g., Arc<dyn Trait>).
const _: () = assert!(std::mem::size_of::<DefaultObservationGenerator>() == 0);
const _: () = assert!(std::mem::size_of::<KataGoObservationGenerator>() == 0);
const _: () = assert!(std::mem::size_of::<DefaultActionMapper>() == 0);
const _: () = assert!(std::mem::size_of::<SpatialActionMapper>() == 0);

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
// Mode enums for dynamic dispatch
// ---------------------------------------------------------------------------
//
// COUPLING NOTE: Adding a new observation or action mode requires changes in
// FOUR places:
//   1. The `ObsMode` / `ActionMode` enum (here) — add variant + trait dispatch
//   2. The `ObsModeTag` / `ActionModeTag` enum — add Copy variant
//   3. The rayon closure's `match obs_tag { ... }` / `match act_tag { ... }`
//   4. The `VecEnv::new()` string-to-enum match in the constructor
// The ZST compile-time assertions below guard against generators with state,
// but they do not guard against forgetting a match arm.

enum ObsMode {
    Default(DefaultObservationGenerator),
    KataGo(KataGoObservationGenerator),
}

impl ObsMode {
    fn channels(&self) -> usize {
        match self {
            ObsMode::Default(_) => NUM_CHANNELS,
            ObsMode::KataGo(_) => KATAGO_NUM_CHANNELS,
        }
    }

    fn buffer_len(&self) -> usize {
        match self {
            ObsMode::Default(_) => BUFFER_LEN,
            ObsMode::KataGo(_) => KATAGO_BUFFER_LEN,
        }
    }

    fn generate(&self, state: &GameState, perspective: Color, buffer: &mut [f32]) {
        match self {
            ObsMode::Default(g) => g.generate(state, perspective, buffer),
            ObsMode::KataGo(g) => g.generate(state, perspective, buffer),
        }
    }
}

enum ActionMode {
    Default(DefaultActionMapper),
    Spatial(SpatialActionMapper),
}

impl ActionMode {
    fn action_space_size(&self) -> usize {
        match self {
            ActionMode::Default(_) => ACTION_SPACE_SIZE,
            ActionMode::Spatial(_) => SPATIAL_ACTION_SPACE_SIZE,
        }
    }

    fn encode(&self, mv: Move, perspective: Color) -> usize {
        match self {
            ActionMode::Default(m) => <DefaultActionMapper as ActionMapper>::encode(m, mv, perspective),
            ActionMode::Spatial(m) => <SpatialActionMapper as ActionMapper>::encode(m, mv, perspective),
        }
    }

    fn decode(&self, idx: usize, perspective: Color) -> Result<Move, String> {
        match self {
            ActionMode::Default(m) => <DefaultActionMapper as ActionMapper>::decode(m, idx, perspective),
            ActionMode::Spatial(m) => <SpatialActionMapper as ActionMapper>::decode(m, idx, perspective),
        }
    }
}

/// Tag enums for rayon closure dispatch (Copy + Send safe).
/// Used inside parallel closures where we can't borrow the enum wrappers.
/// NOTE: Assumes all generators are stateless ZSTs. If a future generator
/// acquires state, per-thread reconstruction would create independent copies.
#[derive(Copy, Clone)]
enum ObsModeTag { Default, KataGo }
#[derive(Copy, Clone)]
enum ActionModeTag { Default, Spatial }

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
    material_balance_buffer: Vec<i32>, // N (per-step material balance)
    terminal_obs_buffer: Vec<f32>,   // N * BUFFER_LEN
    current_players_buffer: Vec<u8>, // N (0=Black, 1=White)

    mapper: ActionMode,
    obs_gen: ObsMode,
    obs_buffer_len: usize,    // cached: obs_mode.buffer_len()
    action_space: usize,       // cached: action_mode.action_space_size()
    num_channels: usize,       // cached: obs_mode.channels()

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

        let obs_start = i * self.obs_buffer_len;
        let obs_slice = &mut self.obs_buffer[obs_start..obs_start + self.obs_buffer_len];
        self.obs_gen.generate(&self.games[i], perspective, obs_slice);

        let mask_start = i * self.action_space;
        let mask_slice = &mut self.legal_mask_buffer[mask_start..mask_start + self.action_space];
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
    #[pyo3(signature = (num_envs = 512, max_ply = 500, observation_mode = "default", action_mode = "default"))]
    pub fn new(num_envs: usize, max_ply: u32, observation_mode: &str, action_mode: &str) -> PyResult<Self> {
        let obs_mode = match observation_mode {
            "default" => ObsMode::Default(DefaultObservationGenerator::new()),
            "katago" => ObsMode::KataGo(KataGoObservationGenerator::new()),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown observation_mode '{}'. Valid: 'default', 'katago'", observation_mode)
            )),
        };

        let action_mode_enum = match action_mode {
            "default" => ActionMode::Default(DefaultActionMapper),
            "spatial" => ActionMode::Spatial(SpatialActionMapper),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown action_mode '{}'. Valid: 'default', 'spatial'", action_mode)
            )),
        };

        let obs_buf_len = obs_mode.buffer_len();
        let act_space = action_mode_enum.action_space_size();
        let channels = obs_mode.channels();

        let games: Vec<GameState> = (0..num_envs)
            .map(|_| GameState::with_max_ply(max_ply))
            .collect();

        Ok(VecEnv {
            games,
            num_envs,
            max_ply,
            obs_buffer: vec![0.0; num_envs * obs_buf_len],
            legal_mask_buffer: vec![false; num_envs * act_space],
            reward_buffer: vec![0.0; num_envs],
            terminated_buffer: vec![false; num_envs],
            truncated_buffer: vec![false; num_envs],
            captured_buffer: vec![255; num_envs],
            term_reason_buffer: vec![0; num_envs],
            ply_buffer: vec![0; num_envs],
            material_balance_buffer: vec![0; num_envs],
            terminal_obs_buffer: vec![0.0; num_envs * obs_buf_len],
            current_players_buffer: vec![0; num_envs],
            mapper: action_mode_enum,
            obs_gen: obs_mode,
            obs_buffer_len: obs_buf_len,
            action_space: act_space,
            num_channels: channels,
            episodes_completed: AtomicU64::new(0),
            episodes_drawn: AtomicU64::new(0),
            episodes_truncated: AtomicU64::new(0),
            total_episode_ply: AtomicU64::new(0),
        })
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
            .reshape([self.num_envs, self.num_channels, 9, 9])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mask_array = self.legal_mask_buffer.to_pyarray(py);
        let mask_2d = mask_array
            .reshape([self.num_envs, self.action_space])
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
            if *action < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "env {}: negative action index {}", i, *action
                )));
            }
            let action_idx = *action as usize;
            let perspective = self.games[i].position.current_player;

            let mv = self.mapper.decode(action_idx, perspective)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "env {}: invalid action index {}: {}",
                        i, action_idx, e
                    ))
                })?;

            // Check legal mask
            let mask_start = i * self.action_space;
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
            let obs_buf_len = self.obs_buffer_len;
            let act_space = self.action_space;

            // Tag enums for static dispatch inside the closure (Copy + Send safe)
            let obs_tag = match &self.obs_gen {
                ObsMode::Default(_) => ObsModeTag::Default,
                ObsMode::KataGo(_) => ObsModeTag::KataGo,
            };
            let act_tag = match &self.mapper {
                ActionMode::Default(_) => ActionModeTag::Default,
                ActionMode::Spatial(_) => ActionModeTag::Spatial,
            };

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
            let material_balance_ptr = SendPtr(self.material_balance_buffer.as_mut_ptr());
            let current_players_ptr = SendPtr(self.current_players_buffer.as_mut_ptr());

            // Episode counters (atomic — safe for parallel access)
            let ep_completed = &self.episodes_completed;
            let ep_drawn = &self.episodes_drawn;
            let ep_truncated = &self.episodes_truncated;
            let ep_total_ply = &self.total_episode_ply;

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

                    // Material balance from last_mover's perspective (every step).
                    // PERSPECTIVE CONVENTION: last_mover is the player who just moved.
                    // The Python training loop stores the pre-step observation (from the
                    // player who was about to move, i.e., last_mover) alongside this
                    // score target. The perspectives match.
                    *material_balance_ptr.offset(i) = material_balance(
                        &game.position, last_mover,
                    );

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

                        // Save terminal observation.
                        // NOTE: perspective is current_player AFTER the terminal move,
                        // which is the OPPONENT of the player who caused termination.
                        // This is correct for RL value bootstrapping (the "next state"
                        // perspective), but training code that indexes terminal_observations
                        // by the player who received rewards[i] must be aware they are
                        // seeing the opposite player's frame of reference.
                        let term_obs_slice = std::slice::from_raw_parts_mut(
                            terminal_obs_ptr.offset(i * obs_buf_len),
                            obs_buf_len,
                        );
                        let perspective = game.position.current_player;
                        match obs_tag {
                            ObsModeTag::Default => DefaultObservationGenerator::new()
                                .generate(game, perspective, term_obs_slice),
                            ObsModeTag::KataGo => KataGoObservationGenerator::new()
                                .generate(game, perspective, term_obs_slice),
                        }

                        // Auto-reset
                        *game = GameState::with_max_ply(max_ply);
                    }

                    // Write obs+mask for current game state (reset or continuing)
                    let perspective = game.position.current_player;

                    let obs_slice = std::slice::from_raw_parts_mut(
                        obs_ptr.offset(i * obs_buf_len),
                        obs_buf_len,
                    );
                    match obs_tag {
                        ObsModeTag::Default => DefaultObservationGenerator::new()
                            .generate(game, perspective, obs_slice),
                        ObsModeTag::KataGo => KataGoObservationGenerator::new()
                            .generate(game, perspective, obs_slice),
                    }

                    let mask_slice = std::slice::from_raw_parts_mut(
                        mask_ptr.offset(i * act_space),
                        act_space,
                    );
                    mask_slice.fill(false);
                    let mut move_list = MoveList::new();
                    game.generate_legal_moves_into(&mut move_list);
                    for legal_mv in move_list.iter() {
                        let idx = match act_tag {
                            ActionModeTag::Default => DefaultActionMapper.encode(*legal_mv, perspective),
                            ActionModeTag::Spatial => SpatialActionMapper.encode(*legal_mv, perspective),
                        };
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
            .reshape([self.num_envs, self.num_channels, 9, 9])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mask_array = self.legal_mask_buffer.to_pyarray(py);
        let mask_2d = mask_array
            .reshape([self.num_envs, self.action_space])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let rewards = self.reward_buffer.to_pyarray(py);
        let terminated = self.terminated_buffer.to_pyarray(py);
        let truncated = self.truncated_buffer.to_pyarray(py);

        let terminal_obs_array = self.terminal_obs_buffer.to_pyarray(py);
        let terminal_obs_4d = terminal_obs_array
            .reshape([self.num_envs, self.num_channels, 9, 9])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let current_players = self.current_players_buffer.to_pyarray(py);

        let captured_arr = self.captured_buffer.to_pyarray(py);
        let term_reason_arr = self.term_reason_buffer.to_pyarray(py);
        let ply_arr = self.ply_buffer.to_pyarray(py);
        let material_arr = self.material_balance_buffer.to_pyarray(py);

        let metadata = Py::new(
            py,
            StepMetadata {
                captured_piece: captured_arr.unbind(),
                termination_reason: term_reason_arr.unbind(),
                ply_count: ply_arr.unbind(),
                material_balance: material_arr.unbind(),
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
        self.action_space
    }

    /// Number of observation channels.
    #[getter]
    pub fn observation_channels(&self) -> usize {
        self.num_channels
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
    use crate::spatial_action_mapper::{SpatialActionMapper, SPATIAL_ACTION_SPACE_SIZE};
    use crate::action_mapper::ActionMapper;

    /// Test-only constructor that builds a default-mode VecEnv without PyO3.
    fn make_env(num_envs: usize, max_ply: u32) -> VecEnv {
        let obs_mode = ObsMode::Default(DefaultObservationGenerator::new());
        let action_mode = ActionMode::Default(DefaultActionMapper);
        let obs_buf_len = obs_mode.buffer_len();
        let act_space = action_mode.action_space_size();
        let channels = obs_mode.channels();

        let games: Vec<GameState> = (0..num_envs)
            .map(|_| GameState::with_max_ply(max_ply))
            .collect();

        VecEnv {
            games,
            num_envs,
            max_ply,
            obs_buffer: vec![0.0; num_envs * obs_buf_len],
            legal_mask_buffer: vec![false; num_envs * act_space],
            reward_buffer: vec![0.0; num_envs],
            terminated_buffer: vec![false; num_envs],
            truncated_buffer: vec![false; num_envs],
            captured_buffer: vec![255; num_envs],
            term_reason_buffer: vec![0; num_envs],
            ply_buffer: vec![0; num_envs],
            material_balance_buffer: vec![0; num_envs],
            terminal_obs_buffer: vec![0.0; num_envs * obs_buf_len],
            current_players_buffer: vec![0; num_envs],
            mapper: action_mode,
            obs_gen: obs_mode,
            obs_buffer_len: obs_buf_len,
            action_space: act_space,
            num_channels: channels,
            episodes_completed: AtomicU64::new(0),
            episodes_drawn: AtomicU64::new(0),
            episodes_truncated: AtomicU64::new(0),
            total_episode_ply: AtomicU64::new(0),
        }
    }

    /// Test-only constructor that builds a VecEnv with specified modes, without PyO3.
    fn make_env_with_modes(
        num_envs: usize,
        max_ply: u32,
        obs_mode: ObsMode,
        action_mode: ActionMode,
    ) -> VecEnv {
        let obs_buf_len = obs_mode.buffer_len();
        let act_space = action_mode.action_space_size();
        let channels = obs_mode.channels();

        let games: Vec<GameState> = (0..num_envs)
            .map(|_| GameState::with_max_ply(max_ply))
            .collect();

        VecEnv {
            games,
            num_envs,
            max_ply,
            obs_buffer: vec![0.0; num_envs * obs_buf_len],
            legal_mask_buffer: vec![false; num_envs * act_space],
            reward_buffer: vec![0.0; num_envs],
            terminated_buffer: vec![false; num_envs],
            truncated_buffer: vec![false; num_envs],
            captured_buffer: vec![255; num_envs],
            term_reason_buffer: vec![0; num_envs],
            ply_buffer: vec![0; num_envs],
            material_balance_buffer: vec![0; num_envs],
            terminal_obs_buffer: vec![0.0; num_envs * obs_buf_len],
            current_players_buffer: vec![0; num_envs],
            mapper: action_mode,
            obs_gen: obs_mode,
            obs_buffer_len: obs_buf_len,
            action_space: act_space,
            num_channels: channels,
            episodes_completed: AtomicU64::new(0),
            episodes_drawn: AtomicU64::new(0),
            episodes_truncated: AtomicU64::new(0),
            total_episode_ply: AtomicU64::new(0),
        }
    }

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
        let env = make_env(4, 100);
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
        assert_eq!(env.material_balance_buffer.len(), 4);
        assert_eq!(env.terminal_obs_buffer.len(), 4 * BUFFER_LEN);
        assert_eq!(env.games.len(), 4);
    }

    #[test]
    fn test_write_obs_and_mask_startpos() {
        let mut env = make_env(2, 500);
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
        let env = make_env(3, 500);
        for i in 0..3 {
            assert_eq!(env.captured_buffer[i], 255);
            assert_eq!(env.term_reason_buffer[i], 0);
            assert_eq!(env.ply_buffer[i], 0);
            assert_eq!(env.material_balance_buffer[i], 0);
        }
    }

    // -----------------------------------------------------------------------
    // VecEnv: legal mask matches legal_moves after a move
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_obs_and_mask_after_move() {
        let mut env = make_env(1, 500);

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
        let mut env = make_env(1, 500);
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
        let mut env = make_env(1, 500);
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
        let mut env = make_env(1, 500);
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
        let mut env = make_env(1, 2);
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

    // -----------------------------------------------------------------------
    // Episode counter and auto-reset simulation (no Python needed)
    // -----------------------------------------------------------------------

    #[test]
    fn test_episode_counters_after_max_ply() {
        let mut env = make_env(1, 2);
        env.write_obs_and_mask(0);

        // Drive game to terminal via max_ply = 2
        let mv1 = env.games[0].legal_moves()[0];
        env.games[0].make_move(mv1);
        env.games[0].check_termination();

        let mv2 = env.games[0].legal_moves()[0];
        env.games[0].make_move(mv2);
        env.games[0].check_termination();

        assert_eq!(env.games[0].result, GameResult::MaxMoves);

        // Simulate the counter updates that step() would do
        let result = env.games[0].result;
        let truncated = result.is_truncation();
        assert!(truncated, "MaxMoves should be a truncation");

        env.episodes_completed.fetch_add(1, Ordering::Relaxed);
        env.total_episode_ply.fetch_add(env.games[0].ply as u64, Ordering::Relaxed);
        if truncated {
            env.episodes_truncated.fetch_add(1, Ordering::Relaxed);
        }

        assert_eq!(env.episodes_completed.load(Ordering::Relaxed), 1);
        assert_eq!(env.episodes_truncated.load(Ordering::Relaxed), 1);
        assert_eq!(env.total_episode_ply.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_auto_reset_produces_startpos_obs() {
        let mut env = make_env(1, 2);
        env.write_obs_and_mask(0);

        // Snapshot startpos observation
        let startpos_obs: Vec<f32> = env.obs_buffer[..BUFFER_LEN].to_vec();
        let startpos_mask_count: usize = env.legal_mask_buffer[..ACTION_SPACE_SIZE]
            .iter()
            .filter(|&&x| x)
            .count();

        // Play 2 moves to terminate
        let mv1 = env.games[0].legal_moves()[0];
        env.games[0].make_move(mv1);
        env.write_obs_and_mask(0);
        let mv2 = env.games[0].legal_moves()[0];
        env.games[0].make_move(mv2);
        env.games[0].check_termination();
        assert_eq!(env.games[0].result, GameResult::MaxMoves);

        // Save terminal obs
        let terminal_obs: Vec<f32> = {
            let mut buf = vec![0.0f32; BUFFER_LEN];
            let perspective = env.games[0].position.current_player;
            env.obs_gen.generate(&env.games[0], perspective, &mut buf);
            buf
        };

        // Auto-reset
        env.games[0] = GameState::with_max_ply(2);
        env.write_obs_and_mask(0);

        // After reset, obs should match startpos again
        let post_reset_obs: Vec<f32> = env.obs_buffer[..BUFFER_LEN].to_vec();
        assert_eq!(
            post_reset_obs, startpos_obs,
            "After auto-reset, observation should match startpos"
        );

        // Legal mask count should be 30 again
        let post_reset_mask_count: usize = env.legal_mask_buffer[..ACTION_SPACE_SIZE]
            .iter()
            .filter(|&&x| x)
            .count();
        assert_eq!(
            post_reset_mask_count, startpos_mask_count,
            "After auto-reset, legal move count should match startpos (30)"
        );

        // Terminal obs should differ from startpos (position changed after 2 moves)
        assert_ne!(
            terminal_obs, startpos_obs,
            "Terminal observation should differ from startpos"
        );
    }

    // -----------------------------------------------------------------------
    // reset_stats clears counters
    // -----------------------------------------------------------------------

    #[test]
    fn test_reset_stats() {
        let env = make_env(1, 500);
        env.episodes_completed.fetch_add(10, Ordering::Relaxed);
        env.episodes_drawn.fetch_add(3, Ordering::Relaxed);
        env.episodes_truncated.fetch_add(2, Ordering::Relaxed);
        env.total_episode_ply.fetch_add(1000, Ordering::Relaxed);

        env.reset_stats();

        assert_eq!(env.episodes_completed.load(Ordering::Relaxed), 0);
        assert_eq!(env.episodes_drawn.load(Ordering::Relaxed), 0);
        assert_eq!(env.episodes_truncated.load(Ordering::Relaxed), 0);
        assert_eq!(env.total_episode_ply.load(Ordering::Relaxed), 0);
    }

    // -----------------------------------------------------------------------
    // current_players_buffer correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_players_buffer_after_move() {
        let mut env = make_env(2, 500);

        // At startpos, both should be Black (0)
        env.current_players_buffer[0] = match env.games[0].position.current_player {
            Color::Black => 0,
            Color::White => 1,
        };
        env.current_players_buffer[1] = match env.games[1].position.current_player {
            Color::Black => 0,
            Color::White => 1,
        };
        assert_eq!(env.current_players_buffer[0], 0, "Game 0 should start as Black");
        assert_eq!(env.current_players_buffer[1], 0, "Game 1 should start as Black");

        // Make a move in game 0 only
        let mv = env.games[0].legal_moves()[0];
        env.games[0].make_move(mv);
        env.current_players_buffer[0] = match env.games[0].position.current_player {
            Color::Black => 0,
            Color::White => 1,
        };

        assert_eq!(env.current_players_buffer[0], 1, "Game 0 should be White after one move");
        assert_eq!(env.current_players_buffer[1], 0, "Game 1 should still be Black");
    }

    // -----------------------------------------------------------------------
    // compute_reward: draw variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_reward_draw_variants_all_zero() {
        // Repetition, Impasse(None), MaxMoves, InProgress — all 0.0
        let draws = [
            GameResult::Repetition,
            GameResult::Impasse { winner: None },
            GameResult::MaxMoves,
            GameResult::InProgress,
        ];
        for result in &draws {
            assert_eq!(
                compute_reward(result, Color::Black), 0.0,
                "{:?} should give 0.0 reward for Black",
                result
            );
            assert_eq!(
                compute_reward(result, Color::White), 0.0,
                "{:?} should give 0.0 reward for White",
                result
            );
        }
    }

    // -----------------------------------------------------------------------
    // Multi-env buffer isolation
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_env_obs_isolation() {
        let mut env = make_env(3, 500);

        // Write obs for all envs
        for i in 0..3 {
            env.write_obs_and_mask(i);
        }

        // All three should have identical observations at startpos
        let obs0 = &env.obs_buffer[0..BUFFER_LEN];
        let obs1 = &env.obs_buffer[BUFFER_LEN..2 * BUFFER_LEN];
        let obs2 = &env.obs_buffer[2 * BUFFER_LEN..3 * BUFFER_LEN];
        assert_eq!(obs0, obs1, "Env 0 and 1 should have same obs at startpos");
        assert_eq!(obs1, obs2, "Env 1 and 2 should have same obs at startpos");

        // Make a move in env 1 only
        let mv = env.games[1].legal_moves()[0];
        env.games[1].make_move(mv);
        env.write_obs_and_mask(1);

        // Now env 1 should differ from env 0
        let obs0_after = &env.obs_buffer[0..BUFFER_LEN];
        let obs1_after = &env.obs_buffer[BUFFER_LEN..2 * BUFFER_LEN];
        assert_ne!(obs0_after, obs1_after, "After a move, env 1 obs should differ from env 0");
    }

    // -----------------------------------------------------------------------
    // VecEnv at scale: buffer construction above PARALLEL_THRESHOLD
    // -----------------------------------------------------------------------

    #[test]
    fn test_large_vecenv_buffer_construction() {
        let n = 128; // above PARALLEL_THRESHOLD (64)
        let env = make_env(n, 100);
        assert_eq!(env.num_envs, n);
        assert_eq!(env.games.len(), n);
        assert_eq!(env.obs_buffer.len(), n * BUFFER_LEN);
        assert_eq!(env.legal_mask_buffer.len(), n * ACTION_SPACE_SIZE);
        assert_eq!(env.reward_buffer.len(), n);
        assert_eq!(env.terminated_buffer.len(), n);
        assert_eq!(env.truncated_buffer.len(), n);
        assert_eq!(env.terminal_obs_buffer.len(), n * BUFFER_LEN);
        assert_eq!(env.current_players_buffer.len(), n);
        assert_eq!(env.material_balance_buffer.len(), n);
    }

    #[test]
    fn test_large_vecenv_obs_mask_consistency() {
        let n = 128;
        let mut env = make_env(n, 500);

        // Write obs and mask for all envs
        for i in 0..n {
            env.write_obs_and_mask(i);
        }

        // All envs start at startpos — obs should be identical across all
        let obs_ref = &env.obs_buffer[0..BUFFER_LEN];
        for i in 1..n {
            let obs_i = &env.obs_buffer[i * BUFFER_LEN..(i + 1) * BUFFER_LEN];
            assert_eq!(
                obs_ref, obs_i,
                "Env {} obs should match env 0 at startpos", i
            );
        }

        // All envs should have exactly 30 legal moves at startpos
        for i in 0..n {
            let mask_start = i * ACTION_SPACE_SIZE;
            let mask_end = mask_start + ACTION_SPACE_SIZE;
            let legal_count: usize = env.legal_mask_buffer[mask_start..mask_end]
                .iter()
                .filter(|&&x| x)
                .count();
            assert_eq!(
                legal_count, 30,
                "Env {} should have 30 legal moves at startpos, got {}", i, legal_count
            );
        }
    }

    #[test]
    fn test_large_vecenv_obs_isolation_after_move() {
        let n = 128;
        let mut env = make_env(n, 500);

        // Write initial obs for all
        for i in 0..n {
            env.write_obs_and_mask(i);
        }

        // Make a move in env 0 only
        let mv = env.games[0].legal_moves()[0];
        env.games[0].make_move(mv);
        env.write_obs_and_mask(0);

        // Env 0 should differ from env 1 (which is still at startpos)
        let obs0 = &env.obs_buffer[0..BUFFER_LEN];
        let obs1 = &env.obs_buffer[BUFFER_LEN..2 * BUFFER_LEN];
        assert_ne!(obs0, obs1, "Env 0 obs should differ after a move");

        // Env 1 should still match env 127 (both at startpos)
        let obs_last = &env.obs_buffer[127 * BUFFER_LEN..128 * BUFFER_LEN];
        assert_eq!(obs1, obs_last, "Unmoved envs should still have matching obs");
    }

    #[test]
    fn test_large_vecenv_episode_counters_after_max_ply() {
        let n = 128;
        let env = make_env(n, 0); // max_ply=0 → immediate truncation

        // Verify the episode counters are at zero before any step
        assert_eq!(
            env.episodes_completed.load(Ordering::Relaxed), 0,
            "episodes_completed should start at 0"
        );
        assert_eq!(
            env.episodes_truncated.load(Ordering::Relaxed), 0,
            "episodes_truncated should start at 0"
        );
        assert_eq!(
            env.episodes_drawn.load(Ordering::Relaxed), 0,
            "episodes_drawn should start at 0"
        );
    }

    // -----------------------------------------------------------------------
    // Test gap remediation: H5 — spatial encoder mask
    // -----------------------------------------------------------------------

    #[test]
    fn test_write_legal_mask_into_spatial_startpos() {
        let mut gs = GameState::with_max_ply(500);
        let mapper = SpatialActionMapper;
        let perspective = Color::Black;

        // Use write_legal_mask_into with spatial encoder — the whole point of H5
        // is to verify the debug_assert!(idx < mask.len()) doesn't fire.
        let encode_fn = |mv: Move| -> usize {
            <SpatialActionMapper as ActionMapper>::encode(&mapper, mv, perspective)
        };

        let mut mask = vec![false; SPATIAL_ACTION_SPACE_SIZE];
        gs.write_legal_mask_into(&mut mask, &encode_fn);

        let true_count = mask.iter().filter(|&&x| x).count();
        assert_eq!(
            true_count, 30,
            "Startpos spatial mask should have exactly 30 true bits, got {}",
            true_count
        );

        // All encoded indices must be in bounds and present in mask
        let legal = gs.legal_moves();
        for mv in &legal {
            let idx = encode_fn(*mv);
            assert!(
                idx < SPATIAL_ACTION_SPACE_SIZE,
                "Spatial index {} out of bounds for move {:?}",
                idx, mv
            );
            assert!(mask[idx], "Legal move {:?} not set in spatial mask at index {}", mv, idx);
        }
    }

    // -----------------------------------------------------------------------
    // Test gap remediation: C2 + M3 — draw_rate
    // -----------------------------------------------------------------------

    #[test]
    fn test_draw_rate_zero_before_any_episodes() {
        let env = make_env(4, 500);
        assert_eq!(
            env.draw_rate(), 0.0,
            "draw_rate should be 0.0 when no episodes have completed"
        );
    }

    #[test]
    fn test_draw_rate_after_draws() {
        let env = make_env(4, 500);

        // Simulate: 3 episodes completed, 2 of which were draws
        env.episodes_completed.store(3, Ordering::Relaxed);
        env.episodes_drawn.store(2, Ordering::Relaxed);

        let rate = env.draw_rate();
        assert!(
            (rate - 2.0 / 3.0).abs() < 1e-10,
            "draw_rate should be 2/3 ≈ 0.6667, got {}",
            rate
        );
    }

    #[test]
    fn test_draw_rate_after_no_draws() {
        let env = make_env(4, 500);

        // Simulate: 5 episodes completed, 0 draws
        env.episodes_completed.store(5, Ordering::Relaxed);
        env.episodes_drawn.store(0, Ordering::Relaxed);

        assert_eq!(
            env.draw_rate(), 0.0,
            "draw_rate should be 0.0 when no episodes were draws"
        );
    }

    // -----------------------------------------------------------------------
    // Test gap remediation: H1 — KataGo and Spatial mode buffer shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_katago_mode_obs_shape() {
        use crate::katago_observation::{KataGoObservationGenerator, KATAGO_NUM_CHANNELS, KATAGO_BUFFER_LEN};

        let n = 2;
        let env = make_env_with_modes(
            n, 500,
            ObsMode::KataGo(KataGoObservationGenerator::new()),
            ActionMode::Default(DefaultActionMapper),
        );

        assert_eq!(env.num_channels, KATAGO_NUM_CHANNELS, "KataGo obs should have {} channels", KATAGO_NUM_CHANNELS);
        assert_eq!(env.obs_buffer.len(), n * KATAGO_BUFFER_LEN, "obs buffer length mismatch for KataGo mode");
        assert_eq!(env.action_space, ACTION_SPACE_SIZE, "action space should be default");
    }

    #[test]
    fn test_spatial_mode_mask_size() {
        let n = 2;
        let env = make_env_with_modes(
            n, 500,
            ObsMode::Default(DefaultObservationGenerator::new()),
            ActionMode::Spatial(SpatialActionMapper),
        );

        assert_eq!(env.action_space, SPATIAL_ACTION_SPACE_SIZE, "action space should be spatial (11,259)");
        assert_eq!(
            env.legal_mask_buffer.len(), n * SPATIAL_ACTION_SPACE_SIZE,
            "legal mask buffer length mismatch for spatial mode"
        );
        assert_eq!(env.num_channels, NUM_CHANNELS, "obs channels should be default");
    }

    #[test]
    fn test_katago_spatial_obs_and_mask_write() {
        use crate::katago_observation::{KataGoObservationGenerator, KATAGO_NUM_CHANNELS};

        let n = 1;
        let mut env = make_env_with_modes(
            n, 500,
            ObsMode::KataGo(KataGoObservationGenerator::new()),
            ActionMode::Spatial(SpatialActionMapper),
        );

        assert_eq!(env.num_channels, KATAGO_NUM_CHANNELS);
        assert_eq!(env.action_space, SPATIAL_ACTION_SPACE_SIZE);

        // Write obs and mask — should not panic
        env.write_obs_and_mask(0);

        // Observation buffer should have some non-zero values (startpos has pieces)
        let obs_nonzero = env.obs_buffer.iter().any(|&v| v != 0.0);
        assert!(obs_nonzero, "KataGo obs for startpos should have non-zero values");

        // Mask should have exactly 30 true bits (startpos legal moves)
        let mask_count = env.legal_mask_buffer.iter().filter(|&&x| x).count();
        assert_eq!(mask_count, 30, "Spatial mask for startpos should have 30 true bits");
    }

    // -----------------------------------------------------------------------
    // Test gap remediation: H2 — material_balance value correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_material_balance_startpos_is_zero() {
        let mut env = make_env(1, 500);
        env.write_obs_and_mask(0);

        // At startpos, both sides have equal material → balance = 0
        assert_eq!(
            env.material_balance_buffer[0], 0,
            "Material balance at startpos should be 0, got {}",
            env.material_balance_buffer[0]
        );
    }

    #[test]
    fn test_material_balance_sign_convention_after_move() {
        let mut env = make_env(1, 500);

        // Make a non-capture move: material balance stays 0
        let legal = env.games[0].legal_moves();
        let mv = legal[0]; // first legal move (a pawn push, no capture)
        env.games[0].make_move(mv);
        env.write_obs_and_mask(0);

        // Still equal material after a non-capture move
        assert_eq!(
            env.material_balance_buffer[0], 0,
            "Material balance after non-capture move should still be 0, got {}",
            env.material_balance_buffer[0]
        );
    }
}
