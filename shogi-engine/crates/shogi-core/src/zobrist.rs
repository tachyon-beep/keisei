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
}
