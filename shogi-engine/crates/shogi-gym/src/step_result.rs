use pyo3::prelude::*;
use pyo3::Py;
use numpy::{PyArray1, PyArray2, PyArray4};
use shogi_core::GameResult;

/// Codes indicating why an episode terminated.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum TerminationReason {
    NotTerminated  = 0,
    Checkmate      = 1,
    Repetition     = 2,
    PerpetualCheck = 3,
    Impasse        = 4,
    MaxMoves       = 5,
}

impl TerminationReason {
    /// Map a `shogi_core::GameResult` to a `TerminationReason` code.
    pub fn from_game_result(result: GameResult) -> Self {
        match result {
            GameResult::InProgress                 => TerminationReason::NotTerminated,
            GameResult::Checkmate { .. }           => TerminationReason::Checkmate,
            GameResult::Repetition                 => TerminationReason::Repetition,
            GameResult::PerpetualCheck { .. }      => TerminationReason::PerpetualCheck,
            GameResult::Impasse { .. }             => TerminationReason::Impasse,
            GameResult::MaxMoves                   => TerminationReason::MaxMoves,
        }
    }
}

/// Per-step metadata stored as parallel arrays across the N environments.
#[pyclass]
pub struct StepMetadata {
    /// Hand-piece type captured this step (0-6), or 255 if no capture.
    #[pyo3(get)]
    pub captured_piece: Py<PyArray1<u8>>,
    /// `TerminationReason` code for each environment.
    #[pyo3(get)]
    pub termination_reason: Py<PyArray1<u8>>,
    /// Ply count at the time of this step, per environment.
    #[pyo3(get)]
    pub ply_count: Py<PyArray1<u16>>,
}

/// Output of a VecEnv `step()` call.
#[pyclass]
pub struct StepResult {
    /// Board observations — shape `(N, C, 9, 9)`.
    #[pyo3(get)]
    pub observations: Py<PyArray4<f32>>,
    /// Legal-move mask — shape `(N, A)`.
    #[pyo3(get)]
    pub legal_masks: Py<PyArray2<bool>>,
    /// Per-environment scalar rewards — shape `(N,)`.
    #[pyo3(get)]
    pub rewards: Py<PyArray1<f32>>,
    /// Whether each environment terminated this step — shape `(N,)`.
    #[pyo3(get)]
    pub terminated: Py<PyArray1<bool>>,
    /// Whether each environment was truncated this step — shape `(N,)`.
    #[pyo3(get)]
    pub truncated: Py<PyArray1<bool>>,
    /// Structured per-step metadata.
    #[pyo3(get)]
    pub step_metadata: Py<StepMetadata>,
}

/// Output of a VecEnv `reset()` call.
#[pyclass]
pub struct ResetResult {
    /// Initial board observations — shape `(N, C, 9, 9)`.
    #[pyo3(get)]
    pub observations: Py<PyArray4<f32>>,
    /// Legal-move mask for the first step — shape `(N, A)`.
    #[pyo3(get)]
    pub legal_masks: Py<PyArray2<bool>>,
}
