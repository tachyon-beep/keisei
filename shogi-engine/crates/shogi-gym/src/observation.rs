//! ObservationGenerator trait and 46-channel DefaultObservationGenerator.
//!
//! The 46-channel layout mirrors Keisei's Python observation space:
//!
//!  Ch  0-7:  Current player's unpromoted pieces (Pawn, Lance, Knight, Silver, Gold, Bishop, Rook, King)
//!  Ch  8-13: Current player's promoted pieces   (+Pawn, +Lance, +Knight, +Silver, +Bishop, +Rook)
//!  Ch 14-21: Opponent's unpromoted pieces        (same order)
//!  Ch 22-27: Opponent's promoted pieces          (same order)
//!  Ch 28-34: Current player's hand counts        (7 types, normalized, constant plane)
//!  Ch 35-41: Opponent's hand counts              (same)
//!  Ch 42:    Player indicator (1.0 = Black, 0.0 = White), constant plane
//!  Ch 43:    Move count (ply / max_ply), constant plane
//!  Ch 44-45: Reserved (zeros)

use pyo3::prelude::*;
use shogi_core::{Color, GameState, HandPieceType, PieceType, Square};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Total number of channels in the observation tensor.
pub const NUM_CHANNELS: usize = 46;
/// Total number of squares on the board.
pub const NUM_SQUARES: usize = 81;
/// Flat buffer length: NUM_CHANNELS * NUM_SQUARES.
pub const BUFFER_LEN: usize = NUM_CHANNELS * NUM_SQUARES;

/// Maximum hand counts per piece type for normalization.
/// Order: [Pawn=18, Lance=4, Knight=4, Silver=4, Gold=4, Bishop=2, Rook=2]
/// Indices match HandPieceType::index() (0-based, same order as HandPieceType::ALL).
const HAND_MAX_COUNTS: [f32; HandPieceType::COUNT] = [18.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0];

// ---------------------------------------------------------------------------
// Channel mapping helpers
// ---------------------------------------------------------------------------

/// Map an unpromoted PieceType to its channel offset (0..=7) in the
/// current-player unpromoted block (ch 0-7) or opponent unpromoted block (ch 14-21).
///
/// Mapping: Pawn→0, Lance→1, Knight→2, Silver→3, Gold→4, Bishop→5, Rook→6, King→7
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

/// Map a promotable PieceType to its channel offset (0..=5) in the
/// current-player promoted block (ch 8-13) or opponent promoted block (ch 22-27).
///
/// Mapping: +Pawn→0, +Lance→1, +Knight→2, +Silver→3, +Bishop→4, +Rook→5
/// (Gold and King cannot be promoted and should never appear here.)
#[inline]
fn promoted_channel(pt: PieceType) -> usize {
    match pt {
        PieceType::Pawn   => 0,
        PieceType::Lance  => 1,
        PieceType::Knight => 2,
        PieceType::Silver => 3,
        PieceType::Bishop => 4,
        PieceType::Rook   => 5,
        _ => panic!("piece type {:?} cannot be promoted", pt),
    }
}

// ---------------------------------------------------------------------------
// ObservationGenerator trait
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub trait ObservationGenerator: Send + Sync {
    /// Fill `buffer` with the observation for `state` from `perspective`'s point of view.
    ///
    /// `buffer` must have length `self.channels() * 81`.
    fn generate(&self, state: &GameState, perspective: Color, buffer: &mut [f32]);

    /// Number of channels in the observation.
    fn channels(&self) -> usize;
}

// ---------------------------------------------------------------------------
// DefaultObservationGenerator
// ---------------------------------------------------------------------------

/// 46-channel observation generator matching Keisei's Python layout.
#[pyclass]
pub struct DefaultObservationGenerator;

impl DefaultObservationGenerator {
    pub fn new() -> Self {
        DefaultObservationGenerator
    }
}

impl Default for DefaultObservationGenerator {
    fn default() -> Self {
        DefaultObservationGenerator::new()
    }
}

impl ObservationGenerator for DefaultObservationGenerator {
    fn channels(&self) -> usize {
        NUM_CHANNELS
    }

    fn generate(&self, state: &GameState, perspective: Color, buffer: &mut [f32]) {
        assert_eq!(
            buffer.len(),
            BUFFER_LEN,
            "buffer must have length {} (got {})",
            BUFFER_LEN,
            buffer.len()
        );

        // Zero out the entire buffer first (handles reserved channels 44-45 and
        // any squares left empty by piece placement).
        buffer.fill(0.0);

        let pos = &state.position;
        let opponent = perspective.opponent();

        // Whether to flip square indices (White's perspective).
        let flip = perspective == Color::White;

        // -----------------------------------------------------------------------
        // Channels 0-27: Board piece planes
        // -----------------------------------------------------------------------
        for idx in 0..NUM_SQUARES {
            let sq = Square::new_unchecked(idx as u8);
            if let Some(piece) = pos.piece_at(sq) {
                let piece_color = piece.color();
                let pt = piece.piece_type();
                let promoted = piece.is_promoted();

                // Determine the output square index (may be flipped for White's POV).
                let out_sq = if flip { 80 - idx } else { idx };

                if piece_color == perspective {
                    // Current player's piece
                    let ch = if promoted {
                        // Promoted: channels 8-13
                        8 + promoted_channel(pt)
                    } else {
                        // Unpromoted: channels 0-7
                        unpromoted_channel(pt)
                    };
                    buffer[ch * NUM_SQUARES + out_sq] = 1.0;
                } else {
                    // Opponent's piece
                    let ch = if promoted {
                        // Promoted: channels 22-27
                        22 + promoted_channel(pt)
                    } else {
                        // Unpromoted: channels 14-21
                        14 + unpromoted_channel(pt)
                    };
                    buffer[ch * NUM_SQUARES + out_sq] = 1.0;
                }
            }
        }

        // -----------------------------------------------------------------------
        // Channels 28-34: Current player's hand counts (normalized, constant planes)
        // -----------------------------------------------------------------------
        for &hpt in &HandPieceType::ALL {
            let count = pos.hand_count(perspective, hpt) as f32;
            let max_count = HAND_MAX_COUNTS[hpt.index()];
            let normalized = count / max_count;
            let ch = 28 + hpt.index();
            // Fill all 81 squares with the same normalized value.
            let start = ch * NUM_SQUARES;
            buffer[start..start + NUM_SQUARES].fill(normalized);
        }

        // -----------------------------------------------------------------------
        // Channels 35-41: Opponent's hand counts (normalized, constant planes)
        // -----------------------------------------------------------------------
        for &hpt in &HandPieceType::ALL {
            let count = pos.hand_count(opponent, hpt) as f32;
            let max_count = HAND_MAX_COUNTS[hpt.index()];
            let normalized = count / max_count;
            let ch = 35 + hpt.index();
            let start = ch * NUM_SQUARES;
            buffer[start..start + NUM_SQUARES].fill(normalized);
        }

        // -----------------------------------------------------------------------
        // Channel 42: Player indicator (1.0 = Black perspective, 0.0 = White)
        // -----------------------------------------------------------------------
        let player_indicator = if perspective == Color::Black { 1.0_f32 } else { 0.0_f32 };
        let start = 42 * NUM_SQUARES;
        buffer[start..start + NUM_SQUARES].fill(player_indicator);

        // -----------------------------------------------------------------------
        // Channel 43: Move count (ply / max_ply), constant plane
        // -----------------------------------------------------------------------
        let move_count = if state.max_ply == 0 {
            0.0_f32
        } else {
            state.ply as f32 / state.max_ply as f32
        };
        let start = 43 * NUM_SQUARES;
        buffer[start..start + NUM_SQUARES].fill(move_count);

        // Channels 44-45 are already zeroed by the initial fill(0.0).
    }
}

#[pymethods]
impl DefaultObservationGenerator {
    #[new]
    pub fn py_new() -> Self {
        DefaultObservationGenerator::new()
    }

    #[getter]
    pub fn channels(&self) -> usize {
        NUM_CHANNELS
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use shogi_core::GameState;

    fn make_gen() -> DefaultObservationGenerator {
        DefaultObservationGenerator::new()
    }

    fn make_buffer() -> Vec<f32> {
        vec![0.0_f32; BUFFER_LEN]
    }

    // -----------------------------------------------------------------------
    // Basic structural tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_channels() {
        let obs_gen = make_gen();
        assert_eq!(<DefaultObservationGenerator as ObservationGenerator>::channels(&obs_gen), 46);
    }

    #[test]
    fn test_startpos_observation_shape() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);
        assert_eq!(buf.len(), 3726);
    }

    // -----------------------------------------------------------------------
    // Board piece plane tests
    // -----------------------------------------------------------------------

    /// At startpos from Black's perspective:
    /// - Black's king (current player) is at row 8, col 4 → index 76 → channel 7
    /// - White's king (opponent) is at row 0, col 4 → index 4 → channel 21
    #[test]
    fn test_startpos_black_perspective_pieces() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        // Black's king: channel 7, square 76
        assert_eq!(buf[7 * NUM_SQUARES + 76], 1.0, "Black king not found at ch7, sq76");

        // White's king (opponent): channel 21, square 4
        assert_eq!(buf[21 * NUM_SQUARES + 4], 1.0, "White king not found at ch21, sq4");

        // Sanity: no king in wrong channels
        assert_eq!(buf[7 * NUM_SQUARES + 4], 0.0, "Should be no king in ch7, sq4");
        assert_eq!(buf[21 * NUM_SQUARES + 76], 0.0, "Should be no king in ch21, sq76");
    }

    /// At startpos from White's perspective (board is flipped 180°):
    /// - White's king (current player) flipped: 80-4=76 → channel 7
    /// - Black's king (opponent) flipped: 80-76=4 → channel 21
    #[test]
    fn test_startpos_white_perspective_flipped() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::White, &mut buf);

        // White's king (current): channel 7, flipped square = 80 - 4 = 76
        assert_eq!(buf[7 * NUM_SQUARES + 76], 1.0, "White king not found at ch7, sq76 (flipped)");

        // Black's king (opponent): channel 21, flipped square = 80 - 76 = 4
        assert_eq!(buf[21 * NUM_SQUARES + 4], 1.0, "Black king not found at ch21, sq4 (flipped)");
    }

    // -----------------------------------------------------------------------
    // Player indicator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_player_indicator_black() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        let start = 42 * NUM_SQUARES;
        for i in 0..NUM_SQUARES {
            assert_eq!(buf[start + i], 1.0, "ch42[{}] should be 1.0 for Black", i);
        }
    }

    #[test]
    fn test_player_indicator_white() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::White, &mut buf);

        let start = 42 * NUM_SQUARES;
        for i in 0..NUM_SQUARES {
            assert_eq!(buf[start + i], 0.0, "ch42[{}] should be 0.0 for White", i);
        }
    }

    // -----------------------------------------------------------------------
    // Move count normalization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_move_count_normalization() {
        let obs_gen = make_gen();

        // ply=0 → channel 43 all 0.0
        let state0 = GameState::new(); // ply=0, max_ply=500
        let mut buf0 = make_buffer();
        obs_gen.generate(&state0, Color::Black, &mut buf0);
        let start = 43 * NUM_SQUARES;
        for i in 0..NUM_SQUARES {
            assert_eq!(buf0[start + i], 0.0, "ch43[{}] should be 0.0 at ply=0", i);
        }

        // ply=1, max_ply=100 → channel 43 all 0.01
        let mut state1 = GameState::with_max_ply(100);
        // Advance by one legal move to get ply=1
        let moves = state1.legal_moves();
        assert!(!moves.is_empty(), "startpos should have legal moves");
        state1.make_move(moves[0]);
        assert_eq!(state1.ply, 1);

        let mut buf1 = make_buffer();
        obs_gen.generate(&state1, Color::Black, &mut buf1);
        for i in 0..NUM_SQUARES {
            let expected = 1.0_f32 / 100.0_f32;
            let actual = buf1[start + i];
            assert!(
                (actual - expected).abs() < 1e-6,
                "ch43[{}] expected {}, got {} at ply=1/max_ply=100",
                i, expected, actual
            );
        }
    }

    // -----------------------------------------------------------------------
    // Reserved channels test
    // -----------------------------------------------------------------------

    #[test]
    fn test_reserved_channels_zero() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        // Channel 44
        let start44 = 44 * NUM_SQUARES;
        for i in 0..NUM_SQUARES {
            assert_eq!(buf[start44 + i], 0.0, "ch44[{}] should be 0.0", i);
        }

        // Channel 45
        let start45 = 45 * NUM_SQUARES;
        for i in 0..NUM_SQUARES {
            assert_eq!(buf[start45 + i], 0.0, "ch45[{}] should be 0.0", i);
        }
    }

    // -----------------------------------------------------------------------
    // Hand normalization test
    // -----------------------------------------------------------------------

    #[test]
    fn test_hand_normalization() {
        let obs_gen = make_gen();
        let state = GameState::new();
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        // At startpos, all hands are empty → channels 28-41 should all be 0.0
        for ch in 28..=41 {
            let start = ch * NUM_SQUARES;
            for i in 0..NUM_SQUARES {
                assert_eq!(
                    buf[start + i], 0.0,
                    "ch{}[{}] should be 0.0 at startpos (hand normalization)",
                    ch, i
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Non-empty hand observation — Gap #15
    // -----------------------------------------------------------------------

    /// After a capture, the capturing side should have non-zero hand channels.
    #[test]
    fn test_hand_normalization_with_pieces_in_hand() {
        use shogi_core::{HandPieceType, Piece, PieceType, Position, Square};

        let obs_gen = make_gen();

        // Build a position where Black has pawns in hand.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 9);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let state = GameState::from_position(pos, 500);
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        // Channel 28 = current player's Pawn hand count.
        // HandPieceType::Pawn.index() = 0, so channel = 28 + 0 = 28.
        // Normalized: 9 / 18.0 = 0.5
        let ch28_start = 28 * NUM_SQUARES;
        let expected = 9.0_f32 / 18.0_f32;
        for i in 0..NUM_SQUARES {
            assert!(
                (buf[ch28_start + i] - expected).abs() < 1e-6,
                "ch28[{}] should be {} with 9 pawns in hand, got {}",
                i,
                expected,
                buf[ch28_start + i]
            );
        }

        // Other hand channels should still be 0.0 for current player
        for hpt_idx in 1..HandPieceType::COUNT {
            let ch = 28 + hpt_idx;
            let start = ch * NUM_SQUARES;
            assert_eq!(
                buf[start], 0.0,
                "ch{}[0] should be 0.0 (no pieces of this type in hand)",
                ch
            );
        }
    }

    /// Opponent's hand pieces should appear in channels 35-41.
    #[test]
    fn test_opponent_hand_channels() {
        use shogi_core::{HandPieceType, Piece, PieceType, Position, Square};

        let obs_gen = make_gen();

        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // Give White (opponent from Black's perspective) 2 rooks in hand
        pos.set_hand_count(Color::White, HandPieceType::Rook, 2);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let state = GameState::from_position(pos, 500);
        let mut buf = make_buffer();
        obs_gen.generate(&state, Color::Black, &mut buf);

        // Channel 35 + HandPieceType::Rook.index() = 35 + 6 = 41
        // (Rook as u8 = 7, index = 7 - 1 = 6)
        // Normalized: 2 / 2.0 = 1.0
        let ch = 35 + HandPieceType::Rook.index(); // 41
        let ch_start = ch * NUM_SQUARES;
        for i in 0..NUM_SQUARES {
            assert!(
                (buf[ch_start + i] - 1.0).abs() < 1e-6,
                "ch{}[{}] should be 1.0 with 2 rooks in opponent's hand, got {}",
                ch,
                i,
                buf[ch_start + i]
            );
        }
    }
}
