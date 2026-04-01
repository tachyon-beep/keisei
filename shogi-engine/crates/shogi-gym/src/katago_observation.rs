// shogi-engine/crates/shogi-gym/src/katago_observation.rs

//! KataGoObservationGenerator — 50-channel observation tensor.
//!
//! Channels 0-43:  Same as DefaultObservationGenerator (piece planes, hand, player, ply)
//! Channels 44-47: Repetition count (4 binary planes: 1×, 2×, 3×, 4×)
//! Channel  48:    Check indicator (1.0 if current player is in check)
//! Channel  49:    Reserved (zeros)

use pyo3::prelude::*;
use shogi_core::{Color, GameState, HandPieceType, PieceType, Square};

use crate::observation::ObservationGenerator;

pub const KATAGO_NUM_CHANNELS: usize = 50;
pub const KATAGO_NUM_SQUARES: usize = 81;
pub const KATAGO_BUFFER_LEN: usize = KATAGO_NUM_CHANNELS * KATAGO_NUM_SQUARES;

const HAND_MAX_COUNTS: [f32; HandPieceType::COUNT] = [18.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0];

#[inline]
fn unpromoted_channel(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn   => 0,
        PieceType::Lance  => 1,
        PieceType::Knight => 2,
        PieceType::Silver => 3,
        PieceType::Gold   => 4,
        PieceType::Bishop => 5,
        PieceType::Rook   => 6,
        PieceType::King   => 7,
    }
}

#[inline]
fn promoted_channel(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn   => 0,
        PieceType::Lance  => 1,
        PieceType::Knight => 2,
        PieceType::Silver => 3,
        PieceType::Bishop => 4,
        PieceType::Rook   => 5,
        PieceType::Gold | PieceType::King => panic!("piece type {:?} cannot be promoted", pt),
    }
}

#[pyclass]
pub struct KataGoObservationGenerator;

impl KataGoObservationGenerator {
    pub fn new() -> Self {
        KataGoObservationGenerator
    }
}

impl Default for KataGoObservationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ObservationGenerator for KataGoObservationGenerator {
    fn channels(&self) -> usize {
        KATAGO_NUM_CHANNELS
    }

    fn generate(&self, state: &GameState, perspective: Color, buffer: &mut [f32]) {
        assert_eq!(
            buffer.len(),
            KATAGO_BUFFER_LEN,
            "buffer must have length {} (got {})",
            KATAGO_BUFFER_LEN,
            buffer.len()
        );

        buffer.fill(0.0);

        let pos = &state.position;
        let opponent = perspective.opponent();
        let flip = perspective == Color::White;

        // --- Channels 0-27: Board piece planes (identical to Default) ---
        for idx in 0..KATAGO_NUM_SQUARES {
            let sq = Square::new_unchecked(idx as u8);
            if let Some(piece) = pos.piece_at(sq) {
                let piece_color = piece.color();
                let pt = piece.piece_type();
                let promoted = piece.is_promoted();
                let out_sq = if flip { 80 - idx } else { idx };

                if piece_color == perspective {
                    let ch = if promoted {
                        8 + promoted_channel(pt)
                    } else {
                        unpromoted_channel(pt)
                    };
                    buffer[ch * KATAGO_NUM_SQUARES + out_sq] = 1.0;
                } else {
                    let ch = if promoted {
                        22 + promoted_channel(pt)
                    } else {
                        14 + unpromoted_channel(pt)
                    };
                    buffer[ch * KATAGO_NUM_SQUARES + out_sq] = 1.0;
                }
            }
        }

        // --- Channels 28-34: Current player's hand counts ---
        for &hpt in &HandPieceType::ALL {
            let count = pos.hand_count(perspective, hpt) as f32;
            let max_count = HAND_MAX_COUNTS[hpt.index()];
            let normalized = count / max_count;
            let ch = 28 + hpt.index();
            let start = ch * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(normalized);
        }

        // --- Channels 35-41: Opponent's hand counts ---
        for &hpt in &HandPieceType::ALL {
            let count = pos.hand_count(opponent, hpt) as f32;
            let max_count = HAND_MAX_COUNTS[hpt.index()];
            let normalized = count / max_count;
            let ch = 35 + hpt.index();
            let start = ch * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(normalized);
        }

        // --- Channel 42: Player indicator ---
        let player_indicator = if perspective == Color::Black { 1.0_f32 } else { 0.0_f32 };
        let start = 42 * KATAGO_NUM_SQUARES;
        buffer[start..start + KATAGO_NUM_SQUARES].fill(player_indicator);

        // --- Channel 43: Move count ---
        let move_count = if state.max_ply == 0 {
            0.0_f32
        } else {
            state.ply as f32 / state.max_ply as f32
        };
        let start = 43 * KATAGO_NUM_SQUARES;
        buffer[start..start + KATAGO_NUM_SQUARES].fill(move_count);

        // --- Channels 44-47: Repetition count (binary planes) ---
        let current_hash = pos.hash;
        let raw_count = state.repetition_map.get(&current_hash).copied().unwrap_or(0);
        // raw_count starts at 1 for the initial position (see GameState::from_position
        // which does repetition_map.insert(hash, 1)). After each return to the same
        // position, make_move increments it. So:
        //   raw_count=1 → first occurrence (no prior repetitions)
        //   raw_count=2 → seen once before
        //   raw_count=3 → seen twice before
        //   raw_count=4 → seen three times before (sennichite)
        //
        // Subtract 1 to get "number of prior repetitions":
        let prior_reps = raw_count.saturating_sub(1);
        // Channel 44 = 1 prior rep, channel 45 = 2, channel 46 = 3, channel 47 = 4+
        if prior_reps >= 1 && prior_reps <= 3 {
            let ch = 44 + (prior_reps as usize - 1);
            let start = ch * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(1.0);
        } else if prior_reps >= 4 {
            let start = 47 * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(1.0);
        }
        // prior_reps == 0: all repetition channels stay 0.0 (first occurrence)

        // --- Channel 48: Check indicator ---
        // Note: is_in_check() checks the *current player's* king, not the perspective's.
        // In normal self-play, perspective == state.position.current_player, so this is
        // correct. If they ever differ, channel 48 reflects the side-to-move's check status.
        if state.is_in_check() {
            let start = 48 * KATAGO_NUM_SQUARES;
            buffer[start..start + KATAGO_NUM_SQUARES].fill(1.0);
        }

        // Channel 49: Reserved (already zeroed)
    }
}

#[pymethods]
impl KataGoObservationGenerator {
    #[new]
    pub fn py_new() -> Self {
        KataGoObservationGenerator::new()
    }

    #[getter]
    pub fn channels(&self) -> usize {
        KATAGO_NUM_CHANNELS
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::GameState;

    fn make_gen() -> KataGoObservationGenerator {
        KataGoObservationGenerator::new()
    }

    fn make_buffer() -> Vec<f32> {
        vec![0.0_f32; KATAGO_BUFFER_LEN]
    }

    #[test]
    fn test_katago_channels() {
        let obs_gen = make_gen();
        assert_eq!(
            <KataGoObservationGenerator as ObservationGenerator>::channels(&obs_gen),
            50
        );
    }

    #[test]
    fn test_katago_buffer_length() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);
        assert_eq!(buf.len(), 50 * 81);
    }

    #[test]
    fn test_katago_startpos_pieces_match_default() {
        // Channels 0-27 should be identical to DefaultObservationGenerator
        use crate::observation::{DefaultObservationGenerator, BUFFER_LEN};

        let katago_gen = make_gen();
        let default_gen = DefaultObservationGenerator::new();
        let state = GameState::new();

        let mut katago_buf = make_buffer();
        let mut default_buf = vec![0.0_f32; BUFFER_LEN];

        katago_gen.generate(&state, Color::Black, &mut katago_buf);
        default_gen.generate(&state, Color::Black, &mut default_buf);

        // Compare channels 0-27 (piece planes)
        for ch in 0..28 {
            for sq in 0..81 {
                assert_eq!(
                    katago_buf[ch * 81 + sq],
                    default_buf[ch * 81 + sq],
                    "Mismatch at ch={}, sq={} (piece planes)",
                    ch, sq
                );
            }
        }

        // Compare channels 28-43 (hand + meta)
        for ch in 28..44 {
            for sq in 0..81 {
                assert_eq!(
                    katago_buf[ch * 81 + sq],
                    default_buf[ch * 81 + sq],
                    "Mismatch at ch={}, sq={} (hand/meta planes)",
                    ch, sq
                );
            }
        }
    }

    #[test]
    fn test_katago_no_nan() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);
        for (i, val) in buf.iter().enumerate() {
            assert!(!val.is_nan(), "NaN at buffer position {}", i);
        }
    }

    #[test]
    fn test_katago_reserved_channel_49_zero() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);
        let start = 49 * 81;
        for i in 0..81 {
            assert_eq!(buf[start + i], 0.0, "ch49[{}] should be 0.0", i);
        }
    }

    // -------------------------------------------------------------------
    // Repetition channel tests (Task 3)
    // -------------------------------------------------------------------

    /// Helper: create a state with a known repetition count by directly
    /// setting the repetition_map. This avoids fragile move sequences
    /// that depend on exact legal move generator behavior.
    fn make_state_with_repetition_count(count: u8) -> GameState {
        let mut state = GameState::new();
        let hash = state.position.hash;
        // repetition_map starts at 1 for initial position.
        // We set it to count+1 so prior_reps = (count+1)-1 = count.
        state.repetition_map.insert(hash, count + 1);
        state
    }

    #[test]
    fn test_repetition_channels_zero_at_startpos() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        for ch in 44..=47 {
            let start = ch * 81;
            for sq in 0..81 {
                assert_eq!(
                    buf[start + sq], 0.0,
                    "ch{}[{}] should be 0.0 at startpos (no prior repetitions)",
                    ch, sq
                );
            }
        }
    }

    #[test]
    fn test_repetition_channel_44_after_one_repeat() {
        let obs_gen = make_gen();
        let state = make_state_with_repetition_count(1);
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        let start44 = 44 * 81;
        for sq in 0..81 {
            assert_eq!(buf[start44 + sq], 1.0, "ch44[{}] should be 1.0 after 1 repetition", sq);
        }

        for ch in 45..=47 {
            let start = ch * 81;
            assert_eq!(buf[start], 0.0, "ch{}[0] should be 0.0 after 1 repetition", ch);
        }
    }

    #[test]
    fn test_repetition_channel_45_after_two_repeats() {
        let obs_gen = make_gen();
        let state = make_state_with_repetition_count(2);
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        let start45 = 45 * 81;
        assert_eq!(buf[start45], 1.0, "ch45 should be 1.0 after 2 repetitions");

        let start44 = 44 * 81;
        assert_eq!(buf[start44], 0.0, "ch44 should be 0.0 after 2 repetitions");
    }

    #[test]
    fn test_repetition_channel_46_after_three_repeats() {
        let obs_gen = make_gen();
        let state = make_state_with_repetition_count(3);
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        let start46 = 46 * 81;
        assert_eq!(buf[start46], 1.0, "ch46 should be 1.0 after 3 repetitions");
    }

    #[test]
    fn test_repetition_channel_47_after_four_repeats() {
        let obs_gen = make_gen();
        let state = make_state_with_repetition_count(4);
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        let start47 = 47 * 81;
        assert_eq!(buf[start47], 1.0, "ch47 should be 1.0 after 4+ repetitions");

        for ch in 44..=46 {
            assert_eq!(buf[ch * 81], 0.0, "ch{} should be 0.0 after 4 repetitions", ch);
        }
    }

    #[test]
    fn test_repetition_channels_mutually_exclusive() {
        let obs_gen = make_gen();

        for reps in 0..=4u8 {
            let state = make_state_with_repetition_count(reps);
            let mut buf = make_buffer();
            obs_gen.generate(&state, Color::Black, &mut buf);

            let mut active_channels = Vec::new();
            for ch in 44..=47 {
                if buf[ch * 81] == 1.0 {
                    active_channels.push(ch);
                }
            }

            if reps == 0 {
                assert!(
                    active_channels.is_empty(),
                    "No repetition channel should be active at reps=0, got {:?}",
                    active_channels
                );
            } else {
                assert_eq!(
                    active_channels.len(), 1,
                    "Exactly one repetition channel should be active at reps={}, got {:?}",
                    reps, active_channels
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // Check indicator channel tests (Task 4)
    // -------------------------------------------------------------------

    #[test]
    fn test_check_channel_not_in_check() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        let start48 = 48 * 81;
        for sq in 0..81 {
            assert_eq!(buf[start48 + sq], 0.0, "ch48[{}] should be 0.0 when not in check", sq);
        }
    }

    #[test]
    fn test_check_channel_in_check() {
        use shogi_core::{Piece, PieceType, Position, Square};

        let obs_gen = make_gen();

        // Black king in check from White rook on same file
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let state = GameState::from_position(pos, 500);
        assert!(state.is_in_check(), "Black king should be in check from White rook");

        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        let start48 = 48 * 81;
        for sq in 0..81 {
            assert_eq!(buf[start48 + sq], 1.0, "ch48[{}] should be 1.0 when in check", sq);
        }
    }

    #[test]
    fn test_check_channel_white_perspective() {
        use shogi_core::{Piece, PieceType, Position, Square};

        let obs_gen = make_gen();

        // White king in check from Black rook
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        pos.current_player = Color::White;
        pos.hash = pos.compute_hash();

        let state = GameState::from_position(pos, 500);
        assert!(state.is_in_check(), "White king should be in check");

        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::White, &mut buf);

        let start48 = 48 * 81;
        for sq in 0..81 {
            assert_eq!(buf[start48 + sq], 1.0, "ch48[{}] should be 1.0 for White in check", sq);
        }
    }
}
