//! MoveList — fixed-capacity, stack-allocated move buffer for zero-allocation move generation.

use crate::types::Move;

// ---------------------------------------------------------------------------
// Capacity
// ---------------------------------------------------------------------------

/// Maximum number of moves that can be stored in a `MoveList`.
pub const MOVELIST_CAPACITY: usize = 1024;

// ---------------------------------------------------------------------------
// MoveList
// ---------------------------------------------------------------------------

/// A fixed-capacity, stack-allocated buffer for moves.
///
/// Intended for hot-path move generation where heap allocation must be avoided.
/// All elements are stored as `MaybeUninit<Move>`; only indices `[0, len)` are
/// initialized and safe to read.
pub struct MoveList {
    moves: [std::mem::MaybeUninit<Move>; MOVELIST_CAPACITY],
    len: usize,
}

impl MoveList {
    /// Create a new, empty `MoveList`.
    pub fn new() -> MoveList {
        // SAFETY: An array of MaybeUninit is always safe to initialize with
        // `uninit()` — we promise not to read past `self.len`.
        MoveList {
            moves: unsafe { std::mem::MaybeUninit::uninit().assume_init() },
            len: 0,
        }
    }

    /// Reset the list to empty (does not drop elements — `Move` is `Copy`).
    #[inline]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Append a move. Panics in debug builds if the list is full.
    #[inline]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < MOVELIST_CAPACITY, "MoveList capacity exceeded");
        self.moves[self.len].write(mv);
        self.len += 1;
    }

    /// Return the number of moves currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if no moves are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the move at `index`. Panics in debug builds if out of bounds.
    ///
    /// # Safety
    /// The caller must ensure `index < self.len()`.
    #[inline]
    pub fn get(&self, index: usize) -> Move {
        debug_assert!(index < self.len, "MoveList index out of bounds");
        // SAFETY: index < len, so this slot was written by `push`.
        unsafe { self.moves[index].assume_init() }
    }

    /// Return the initialized elements as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: moves[0..len] are all initialized by `push`.
        unsafe {
            std::slice::from_raw_parts(
                self.moves.as_ptr() as *const Move,
                self.len,
            )
        }
    }

    /// Iterate over the stored moves.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.as_slice().iter()
    }

    /// Return the maximum number of moves that can be stored.
    #[inline]
    pub fn capacity() -> usize {
        MOVELIST_CAPACITY
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Move, Square};

    fn dummy_move(idx: u8) -> Move {
        Move::Board {
            from: Square::new_unchecked(idx),
            to: Square::new_unchecked(idx + 1),
            promote: false,
        }
    }

    #[test]
    fn test_movelist_push_and_read() {
        let mut ml = MoveList::new();
        let mv = dummy_move(0);
        ml.push(mv);
        assert_eq!(ml.len(), 1);
        assert_eq!(ml.get(0), mv);
    }

    #[test]
    fn test_movelist_clear() {
        let mut ml = MoveList::new();
        ml.push(dummy_move(0));
        ml.push(dummy_move(2));
        assert_eq!(ml.len(), 2);
        ml.clear();
        assert!(ml.is_empty());
        assert_eq!(ml.len(), 0);
    }

    #[test]
    fn test_movelist_capacity() {
        assert_eq!(MoveList::capacity(), 1024);
    }

    #[test]
    fn test_movelist_as_slice() {
        let mut ml = MoveList::new();
        for i in 0u8..10 {
            ml.push(dummy_move(i * 2));
        }
        let s = ml.as_slice();
        assert_eq!(s.len(), 10);
        for (i, mv) in s.iter().enumerate() {
            assert_eq!(*mv, dummy_move(i as u8 * 2));
        }
    }

    // -----------------------------------------------------------------------
    // Capacity boundary — Gap #14
    // -----------------------------------------------------------------------

    /// Push exactly MOVELIST_CAPACITY moves and verify all are readable.
    #[test]
    fn test_movelist_full_capacity() {
        let mut ml = MoveList::new();
        // Push 1024 moves (using wrapping indices within valid square range)
        for i in 0..MOVELIST_CAPACITY {
            let from_idx = (i % 80) as u8;
            let to_idx = ((i % 80) + 1) as u8;
            ml.push(Move::Board {
                from: Square::new_unchecked(from_idx),
                to: Square::new_unchecked(to_idx),
                promote: i % 2 == 0,
            });
        }
        assert_eq!(ml.len(), MOVELIST_CAPACITY);
        assert!(!ml.is_empty());

        // Verify first and last elements
        let first = ml.get(0);
        assert_eq!(first, Move::Board {
            from: Square::new_unchecked(0),
            to: Square::new_unchecked(1),
            promote: true,
        });

        let last = ml.get(MOVELIST_CAPACITY - 1);
        let last_i = MOVELIST_CAPACITY - 1;
        assert_eq!(last, Move::Board {
            from: Square::new_unchecked((last_i % 80) as u8),
            to: Square::new_unchecked(((last_i % 80) + 1) as u8),
            promote: last_i % 2 == 0,
        });
    }

    /// Verify that iter() returns all pushed elements.
    #[test]
    fn test_movelist_iter_count() {
        let mut ml = MoveList::new();
        for i in 0u8..50 {
            ml.push(dummy_move(i % 40 * 2));
        }
        assert_eq!(ml.iter().count(), 50);
    }

    /// Verify that clear + push works correctly (reuse after clear).
    #[test]
    fn test_movelist_reuse_after_clear() {
        let mut ml = MoveList::new();
        ml.push(dummy_move(0));
        ml.push(dummy_move(2));
        ml.clear();
        ml.push(dummy_move(10));
        assert_eq!(ml.len(), 1);
        assert_eq!(ml.get(0), dummy_move(10));
    }
}
