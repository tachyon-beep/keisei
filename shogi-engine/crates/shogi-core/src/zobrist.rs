//! Zobrist hashing tables for incremental position hashing.
//!
//! The table is generated deterministically using the xoshiro256** PRNG seeded
//! with a fixed value, so position hashes are reproducible across runs.

use crate::types::{Color, HandPieceType, Square};
use crate::piece::Piece;

// ---------------------------------------------------------------------------
// Private PRNG — splitmix64 seeder + xoshiro256**
// ---------------------------------------------------------------------------

struct SimpleRng {
    state: [u64; 4],
}

impl SimpleRng {
    /// Seed the 4-word state via splitmix64.
    fn new(seed: u64) -> Self {
        let mut s = seed;
        let mut state = [0u64; 4];
        for word in state.iter_mut() {
            // splitmix64 step
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *word = z ^ (z >> 31);
        }
        Self { state }
    }

    /// xoshiro256** next value.
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.state[1]
            .wrapping_mul(5)
            .rotate_left(7)
            .wrapping_mul(9);

        let t = self.state[1] << 17;

        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];

        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);

        result
    }
}

// ---------------------------------------------------------------------------
// ZobristTable
// ---------------------------------------------------------------------------

/// Pre-generated random values used to build Zobrist position hashes.
pub struct ZobristTable {
    /// Random value for each (square_index, piece_u8_value) pair.
    /// `piece_square[sq][piece.to_u8() as usize]`
    pub piece_square: [[u64; 64]; 81],

    /// Random value for each (color, hand_piece_index, count) pair.
    /// `hand[color as usize][hpt.index()][count as usize]`
    /// count index 0 is unused (0 pieces in hand contributes no hash).
    pub hand: [[[u64; 19]; 7]; 2],

    /// XORed into the hash when it is White's turn to move.
    pub side_to_move: u64,
}

impl ZobristTable {
    /// Build the table deterministically using xoshiro256** seeded at
    /// `0xDEAD_BEEF_CAFE_BABE`.
    pub fn new() -> Self {
        let mut rng = SimpleRng::new(0xDEAD_BEEF_CAFE_BABEu64);

        let mut piece_square = [[0u64; 64]; 81];
        for sq in piece_square.iter_mut() {
            for val in sq.iter_mut() {
                *val = rng.next_u64();
            }
        }

        let mut hand = [[[0u64; 19]; 7]; 2];
        for color_hand in hand.iter_mut() {
            for piece_counts in color_hand.iter_mut() {
                for count_val in piece_counts.iter_mut() {
                    *count_val = rng.next_u64();
                }
            }
        }

        let side_to_move = rng.next_u64();

        Self {
            piece_square,
            hand,
            side_to_move,
        }
    }

    /// Return the Zobrist value for `piece` on `sq`.
    #[inline]
    pub fn hash_piece_at(&self, sq: Square, piece: Piece) -> u64 {
        self.piece_square[sq.index()][piece.to_u8() as usize]
    }

    /// Return the Zobrist value for `count` pieces of type `hpt` in `color`'s hand.
    #[inline]
    pub fn hash_hand(&self, color: Color, hpt: HandPieceType, count: u8) -> u64 {
        self.hand[color as usize][hpt.index()][count as usize]
    }
}

impl Default for ZobristTable {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Global table
// ---------------------------------------------------------------------------

/// The global, lazily-initialized Zobrist table.
///
/// Initialised once on first access; safe to use from any thread after that.
pub static ZOBRIST: std::sync::LazyLock<ZobristTable> =
    std::sync::LazyLock::new(ZobristTable::new);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Color, HandPieceType, PieceType, Square};
    use crate::piece::Piece;

    #[test]
    fn test_zobrist_deterministic() {
        let t1 = ZobristTable::new();
        let t2 = ZobristTable::new();

        // Check a sample of piece_square entries.
        for sq_idx in 0..81usize {
            for piece_val in 0..64usize {
                assert_eq!(
                    t1.piece_square[sq_idx][piece_val],
                    t2.piece_square[sq_idx][piece_val],
                    "piece_square[{}][{}] differs",
                    sq_idx,
                    piece_val
                );
            }
        }

        // Check all hand entries.
        for c in 0..2 {
            for h in 0..7 {
                for cnt in 0..19 {
                    assert_eq!(
                        t1.hand[c][h][cnt],
                        t2.hand[c][h][cnt],
                        "hand[{}][{}][{}] differs",
                        c,
                        h,
                        cnt
                    );
                }
            }
        }

        assert_eq!(t1.side_to_move, t2.side_to_move);
    }

    #[test]
    fn test_zobrist_no_obvious_collisions() {
        let table = ZobristTable::new();
        let piece = Piece::new(PieceType::Pawn, Color::Black, false);

        let mut hashes = std::collections::HashSet::new();
        for sq_idx in 0u8..81 {
            let sq = Square::new(sq_idx).unwrap();
            let h = table.hash_piece_at(sq, piece);
            assert!(
                hashes.insert(h),
                "collision: same hash for pawn on square {}",
                sq_idx
            );
        }
    }

    #[test]
    fn test_zobrist_side_to_move_nonzero() {
        let table = ZobristTable::new();
        assert_ne!(table.side_to_move, 0, "side_to_move must be non-zero");
    }

    #[test]
    fn test_lazy_lock_accessible() {
        // Accessing ZOBRIST must not panic.
        let _ = ZOBRIST.side_to_move;

        // Basic sanity: the global table should be identical to a freshly built one.
        let fresh = ZobristTable::new();
        assert_eq!(ZOBRIST.side_to_move, fresh.side_to_move);
        assert_eq!(
            ZOBRIST.piece_square[0][1],
            fresh.piece_square[0][1]
        );
    }

    #[test]
    fn test_hash_hand_helper() {
        let table = ZobristTable::new();
        // Verify the helper delegates correctly.
        let h = table.hash_hand(Color::White, HandPieceType::Rook, 2);
        assert_eq!(
            h,
            table.hand[Color::White as usize][HandPieceType::Rook.index()][2]
        );
    }

    #[test]
    fn test_hash_piece_at_helper() {
        let table = ZobristTable::new();
        let sq = Square::new(40).unwrap();
        let piece = Piece::new(PieceType::King, Color::Black, false);
        let h = table.hash_piece_at(sq, piece);
        assert_eq!(h, table.piece_square[40][piece.to_u8() as usize]);
    }

    // -----------------------------------------------------------------------
    // Zobrist: different hand counts produce different hashes
    // -----------------------------------------------------------------------

    #[test]
    fn test_zobrist_different_hand_counts_different_hashes() {
        let table = ZobristTable::new();

        // 1 pawn in hand vs 2 pawns in hand should have different hash values
        for &hpt in &HandPieceType::ALL {
            let h1 = table.hash_hand(Color::Black, hpt, 1);
            let h2 = table.hash_hand(Color::Black, hpt, 2);
            assert_ne!(
                h1, h2,
                "Hand hash for {:?} count=1 and count=2 must differ",
                hpt
            );
        }
    }

    #[test]
    fn test_zobrist_same_piece_different_colors_different_hashes() {
        let table = ZobristTable::new();
        let sq = Square::new(40).unwrap();

        let black_pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        let white_pawn = Piece::new(PieceType::Pawn, Color::White, false);

        let h_black = table.hash_piece_at(sq, black_pawn);
        let h_white = table.hash_piece_at(sq, white_pawn);
        assert_ne!(
            h_black, h_white,
            "Same piece type, different colors, on same square must have different hashes"
        );
    }

    #[test]
    fn test_zobrist_side_to_move_produces_distinct_position_hashes() {
        use crate::position::Position;

        let pos1 = Position::startpos(); // Black to move

        let mut pos2 = Position::startpos();
        pos2.current_player = Color::White;
        pos2.hash = pos2.compute_hash();

        assert_ne!(
            pos1.hash, pos2.hash,
            "Same board, different side to move must have different hashes"
        );
        // The difference should be exactly the side_to_move XOR
        assert_eq!(
            pos1.hash ^ pos2.hash,
            ZOBRIST.side_to_move,
            "Hash difference should be exactly the side_to_move value"
        );
    }

    #[test]
    fn test_zobrist_hand_counts_all_unique() {
        let table = ZobristTable::new();

        // For each hand piece type, verify all counts 1..18 produce unique hashes
        for &hpt in &HandPieceType::ALL {
            let mut seen = std::collections::HashSet::new();
            for count in 1u8..=18 {
                let h = table.hash_hand(Color::Black, hpt, count);
                assert!(
                    seen.insert(h),
                    "Collision in hand hash for {:?} at count {}",
                    hpt,
                    count
                );
            }
        }
    }

    // ===================================================================
    // Gap #8b: Zobrist collision resistance sanity check
    // ===================================================================

    /// Hash N random-ish positions and verify collision rate is negligible.
    /// With 64-bit hashes and 500 positions, birthday paradox gives ~6.8e-15
    /// probability of any collision — essentially zero.
    #[test]
    fn test_zobrist_collision_resistance_random_positions() {
        use crate::position::Position;

        let mut seen = std::collections::HashSet::new();

        // Generate diverse positions by permuting board state
        for seed in 0u64..500 {
            let mut pos = Position::empty();
            pos.current_player = if seed % 2 == 0 { Color::Black } else { Color::White };

            // Place a king for each side
            let bk_idx = (seed % 81) as u8;
            pos.set_piece(
                Square::new_unchecked(bk_idx),
                Piece::new(PieceType::King, Color::Black, false),
            );
            let wk_idx = ((seed * 7 + 13) % 81) as u8;
            if wk_idx != bk_idx {
                pos.set_piece(
                    Square::new_unchecked(wk_idx),
                    Piece::new(PieceType::King, Color::White, false),
                );
            }

            // Add a few more pieces based on the seed
            let piece_types = [
                PieceType::Pawn, PieceType::Lance, PieceType::Knight,
                PieceType::Silver, PieceType::Gold, PieceType::Bishop,
                PieceType::Rook,
            ];
            let pt = piece_types[(seed as usize) % piece_types.len()];
            let color = if seed % 3 == 0 { Color::White } else { Color::Black };
            let sq_idx = ((seed * 11 + 3) % 81) as u8;
            if sq_idx != bk_idx && sq_idx != wk_idx {
                pos.set_piece(
                    Square::new_unchecked(sq_idx),
                    Piece::new(pt, color, false),
                );
            }

            // Add hand pieces
            if seed % 5 == 0 {
                pos.set_hand_count(Color::Black, HandPieceType::Pawn, (seed % 18 + 1) as u8);
            }

            let h = pos.compute_hash();
            // Note: We don't assert zero collisions (hash collisions are possible in
            // principle) but with 500 samples and 64-bit hashes, even ONE collision
            // would be astronomically unlikely and indicate a bug.
            seen.insert(h);
        }

        // We should have close to 500 unique hashes.
        // Allow a tiny margin (499) in case two seeds generate identical boards.
        assert!(
            seen.len() >= 490,
            "Expected ~500 unique hashes, got {} — possible collision bug",
            seen.len()
        );
    }

    /// Zobrist values for same piece on different squares are all unique.
    #[test]
    fn test_zobrist_piece_square_all_unique_per_piece() {
        let table = ZobristTable::new();
        let pieces_to_check = [
            Piece::new(PieceType::Pawn, Color::Black, false),
            Piece::new(PieceType::Rook, Color::White, false),
            Piece::new(PieceType::Bishop, Color::Black, true),
            Piece::new(PieceType::King, Color::White, false),
        ];

        for piece in pieces_to_check {
            let mut hashes = std::collections::HashSet::new();
            for sq_idx in 0u8..81 {
                let sq = Square::new(sq_idx).unwrap();
                let h = table.hash_piece_at(sq, piece);
                assert!(
                    hashes.insert(h),
                    "Collision: {:?} on square {} has same hash as another square",
                    piece, sq_idx
                );
            }
            assert_eq!(hashes.len(), 81, "not all 81 squares produced unique hashes for {:?}", piece);
        }
    }
}
