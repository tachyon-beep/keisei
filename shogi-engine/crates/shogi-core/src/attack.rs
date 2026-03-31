//! Attack map computation — ground-truth oracle for which squares each player attacks.
//!
//! `compute_attack_map` performs a full board scan and ray-casting pass.
//! All incremental update logic elsewhere in the engine is regression-tested
//! against this function.

use crate::piece::Piece;
use crate::position::Position;
use crate::types::{Color, PieceType, Square};

// ---------------------------------------------------------------------------
// Direction constants (flat 9×9 board, row-major, stride = 9)
// ---------------------------------------------------------------------------

pub const UP: i8 = -9;
pub const DOWN: i8 = 9;
pub const LEFT: i8 = -1;
pub const RIGHT: i8 = 1;
pub const UP_LEFT: i8 = -10;
pub const UP_RIGHT: i8 = -8;
pub const DOWN_LEFT: i8 = 8;
pub const DOWN_RIGHT: i8 = 10;

// ---------------------------------------------------------------------------
// AttackMap type
// ---------------------------------------------------------------------------

/// `attack_map[color_index][square_index]` = number of times `color` attacks
/// that square.
pub type AttackMap = [[u8; Square::NUM_SQUARES]; 2];

// ---------------------------------------------------------------------------
// would_wrap_file
// ---------------------------------------------------------------------------

/// Returns `true` if stepping by `delta` from `from_sq` would either:
/// - go out of bounds (index < 0 or >= 81), or
/// - wrap around a file edge (column changes by more than 1).
///
/// This prevents horizontal ray-casting from silently crossing file boundaries.
#[inline]
pub fn would_wrap_file(from_sq: Square, delta: i8) -> bool {
    let new_idx = from_sq.index() as i16 + delta as i16;
    if new_idx < 0 || new_idx >= 81 {
        return true;
    }
    let new_col = new_idx as usize % 9;
    let old_col = from_sq.col() as usize;
    // Column should change by at most 1
    let col_diff = (new_col as i32 - old_col as i32).abs();
    col_diff > 1
}

// ---------------------------------------------------------------------------
// piece_attack_dirs
// ---------------------------------------------------------------------------

/// Returns `(step_directions, slide_directions)` for a piece.
///
/// - Step directions are single-square moves.
/// - Slide directions are ray-cast (repeated steps until blocked or out of bounds).
/// - Knight is excluded — it requires special handling via `compute_knight_attacks`.
pub fn piece_attack_dirs(
    piece_type: PieceType,
    color: Color,
    promoted: bool,
) -> (Vec<i8>, Vec<i8>) {
    let forward = if color == Color::Black { UP } else { DOWN };
    let backward = if color == Color::Black { DOWN } else { UP };
    let fwd_left = if color == Color::Black { UP_LEFT } else { DOWN_RIGHT };
    let fwd_right = if color == Color::Black { UP_RIGHT } else { DOWN_LEFT };
    let bwd_left = if color == Color::Black { DOWN_LEFT } else { UP_RIGHT };
    let bwd_right = if color == Color::Black { DOWN_RIGHT } else { UP_LEFT };

    // Gold movement: fwd, fwd_left, fwd_right, left, right, backward
    let gold_steps = vec![forward, fwd_left, fwd_right, LEFT, RIGHT, backward];

    if promoted {
        return match piece_type {
            // Promoted Pawn / Lance / Knight / Silver → Gold movement
            PieceType::Pawn | PieceType::Lance | PieceType::Knight | PieceType::Silver => {
                (gold_steps, vec![])
            }
            // Horse (promoted Bishop): diagonal slides + orthogonal steps
            PieceType::Bishop => (
                vec![UP, DOWN, LEFT, RIGHT],
                vec![UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT],
            ),
            // Dragon (promoted Rook): orthogonal slides + diagonal steps
            PieceType::Rook => (
                vec![UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT],
                vec![UP, DOWN, LEFT, RIGHT],
            ),
            // Gold and King cannot be promoted
            PieceType::Gold | PieceType::King => unreachable!(
                "Gold and King cannot be promoted"
            ),
        };
    }

    match piece_type {
        PieceType::Pawn => (vec![forward], vec![]),
        PieceType::Lance => (vec![], vec![forward]),
        PieceType::Knight => (vec![], vec![]), // handled separately
        PieceType::Silver => (
            vec![forward, fwd_left, fwd_right, bwd_left, bwd_right],
            vec![],
        ),
        PieceType::Gold => (gold_steps, vec![]),
        PieceType::Bishop => (vec![], vec![UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT]),
        PieceType::Rook => (vec![], vec![UP, DOWN, LEFT, RIGHT]),
        PieceType::King => (
            vec![UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT],
            vec![],
        ),
    }
}

// ---------------------------------------------------------------------------
// compute_knight_attacks
// ---------------------------------------------------------------------------

/// Returns all squares that a knight on `sq` of color `color` attacks.
///
/// - Black knight: jumps to (row-2, col±1)
/// - White knight: jumps to (row+2, col±1)
///
/// Out-of-bounds targets are silently filtered.
pub fn compute_knight_attacks(sq: Square, color: Color) -> Vec<Square> {
    let row = sq.row() as i16;
    let col = sq.col() as i16;

    let target_row = if color == Color::Black {
        row - 2
    } else {
        row + 2
    };

    let mut targets = Vec::with_capacity(2);
    for &dc in &[-1i16, 1i16] {
        let tc = col + dc;
        if target_row >= 0 && target_row < 9 && tc >= 0 && tc < 9 {
            targets.push(
                Square::from_row_col(target_row as u8, tc as u8)
                    .expect("knight target must be valid"),
            );
        }
    }
    targets
}

// ---------------------------------------------------------------------------
// compute_attack_map
// ---------------------------------------------------------------------------

/// Compute an attack map from scratch by scanning every square in `pos`.
///
/// For each piece found:
/// 1. If it is an unpromoted knight, use `compute_knight_attacks`.
/// 2. Otherwise, obtain `(steps, slides)` from `piece_attack_dirs`.
///    - For each step direction: add 1 to the attacked square's count if the
///      step does not wrap a file and stays in bounds.
///    - For each slide direction: ray-cast, incrementing counts, stopping when
///      a piece is reached (the occupied square IS attacked) or the ray leaves
///      the board.
pub fn compute_attack_map(pos: &Position) -> AttackMap {
    let mut attack_map: AttackMap = [[0u8; Square::NUM_SQUARES]; 2];

    for i in 0..Square::NUM_SQUARES {
        let sq = Square::new_unchecked(i as u8);
        let piece: Piece = match pos.piece_at(sq) {
            Some(p) => p,
            None => continue,
        };

        let color = piece.color();
        let color_idx = color as usize;
        let pt = piece.piece_type();
        let promoted = piece.is_promoted();

        // Knights need special handling
        if pt == PieceType::Knight && !promoted {
            for target in compute_knight_attacks(sq, color) {
                attack_map[color_idx][target.index()] =
                    attack_map[color_idx][target.index()].saturating_add(1);
            }
            continue;
        }

        let (steps, slides) = piece_attack_dirs(pt, color, promoted);

        // Step attacks
        for delta in steps {
            if !would_wrap_file(sq, delta) {
                if let Some(target) = sq.offset(delta) {
                    attack_map[color_idx][target.index()] =
                        attack_map[color_idx][target.index()].saturating_add(1);
                }
            }
        }

        // Slide attacks (ray-cast)
        for delta in slides {
            let mut cur = sq;
            loop {
                if would_wrap_file(cur, delta) {
                    break;
                }
                match cur.offset(delta) {
                    None => break,
                    Some(next) => {
                        attack_map[color_idx][next.index()] =
                            attack_map[color_idx][next.index()].saturating_add(1);
                        // Stop ray if we hit an occupied square
                        if pos.piece_at(next).is_some() {
                            break;
                        }
                        cur = next;
                    }
                }
            }
        }
    }

    attack_map
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::Piece;
    use crate::position::Position;
    use crate::types::{Color, PieceType, Square};

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn lone_piece_pos(row: u8, col: u8, pt: PieceType, color: Color, promoted: bool) -> Position {
        let mut pos = Position::empty();
        let sq = Square::from_row_col(row, col).unwrap();
        pos.set_piece(sq, Piece::new(pt, color, promoted));
        pos
    }

    // -----------------------------------------------------------------------
    // startpos: Black pawns attack row 5, White pawns attack row 3
    // -----------------------------------------------------------------------

    #[test]
    fn test_startpos_attack_map_pawn_squares() {
        let pos = Position::startpos();
        let map = compute_attack_map(&pos);

        // Black pawns are on row 6; they attack one step UP (row 5).
        for col in 0u8..9 {
            let target = Square::from_row_col(5, col).unwrap();
            assert!(
                map[Color::Black as usize][target.index()] >= 1,
                "Black should attack (5,{})",
                col
            );
        }

        // White pawns are on row 2; they attack one step DOWN (row 3).
        for col in 0u8..9 {
            let target = Square::from_row_col(3, col).unwrap();
            assert!(
                map[Color::White as usize][target.index()] >= 1,
                "White should attack (3,{})",
                col
            );
        }
    }

    // -----------------------------------------------------------------------
    // Lone king at center attacks all 8 adjacent squares
    // -----------------------------------------------------------------------

    #[test]
    fn test_attack_map_king_attacks_8_squares() {
        let pos = lone_piece_pos(4, 4, PieceType::King, Color::Black, false);
        let map = compute_attack_map(&pos);

        let center = Square::from_row_col(4, 4).unwrap();
        let directions = [UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT];

        let mut attacked_count = 0usize;
        for delta in directions {
            if let Some(target) = center.offset(delta) {
                assert!(
                    map[Color::Black as usize][target.index()] >= 1,
                    "King at (4,4) should attack square at delta {}",
                    delta
                );
                attacked_count += 1;
            }
        }
        assert_eq!(attacked_count, 8, "King at center should attack exactly 8 squares");
    }

    // -----------------------------------------------------------------------
    // Lone rook at (4,4) attacks entire row and column
    // -----------------------------------------------------------------------

    #[test]
    fn test_attack_map_rook_slides() {
        let pos = lone_piece_pos(4, 4, PieceType::Rook, Color::Black, false);
        let map = compute_attack_map(&pos);

        // Entire row 4 (except self)
        for col in 0u8..9 {
            if col == 4 {
                continue;
            }
            let sq = Square::from_row_col(4, col).unwrap();
            assert!(
                map[Color::Black as usize][sq.index()] >= 1,
                "Rook at (4,4) should attack (4,{})",
                col
            );
        }

        // Entire column 4 (except self)
        for row in 0u8..9 {
            if row == 4 {
                continue;
            }
            let sq = Square::from_row_col(row, 4).unwrap();
            assert!(
                map[Color::Black as usize][sq.index()] >= 1,
                "Rook at (4,4) should attack ({},4)",
                row
            );
        }
    }

    // -----------------------------------------------------------------------
    // Rook blocked: rook at (4,4) blocked at (4,6) should not reach (4,7+)
    // -----------------------------------------------------------------------

    #[test]
    fn test_attack_map_rook_blocked_by_piece() {
        let mut pos = Position::empty();
        // Place rook at (4,4)
        let rook_sq = Square::from_row_col(4, 4).unwrap();
        pos.set_piece(rook_sq, Piece::new(PieceType::Rook, Color::Black, false));
        // Place a blocking piece at (4,6)
        let blocker_sq = Square::from_row_col(4, 6).unwrap();
        pos.set_piece(blocker_sq, Piece::new(PieceType::Pawn, Color::White, false));

        let map = compute_attack_map(&pos);

        // Rook should attack (4,5) and (4,6) (the blocker's square IS attacked)
        let sq_5 = Square::from_row_col(4, 5).unwrap();
        let sq_6 = Square::from_row_col(4, 6).unwrap();
        assert!(
            map[Color::Black as usize][sq_5.index()] >= 1,
            "Rook should attack (4,5)"
        );
        assert!(
            map[Color::Black as usize][sq_6.index()] >= 1,
            "Rook should attack (4,6) — the blocking square"
        );

        // Rook should NOT attack (4,7) or (4,8)
        let sq_7 = Square::from_row_col(4, 7).unwrap();
        let sq_8 = Square::from_row_col(4, 8).unwrap();
        assert_eq!(
            map[Color::Black as usize][sq_7.index()],
            0,
            "Rook should NOT attack (4,7) — blocked"
        );
        assert_eq!(
            map[Color::Black as usize][sq_8.index()],
            0,
            "Rook should NOT attack (4,8) — blocked"
        );
    }

    // -----------------------------------------------------------------------
    // Knight at (4,4) attacks only (2,3) and (2,5)
    // -----------------------------------------------------------------------

    #[test]
    fn test_attack_map_knight_jumps() {
        let pos = lone_piece_pos(4, 4, PieceType::Knight, Color::Black, false);
        let map = compute_attack_map(&pos);

        let t1 = Square::from_row_col(2, 3).unwrap();
        let t2 = Square::from_row_col(2, 5).unwrap();

        assert!(
            map[Color::Black as usize][t1.index()] >= 1,
            "Black knight at (4,4) should attack (2,3)"
        );
        assert!(
            map[Color::Black as usize][t2.index()] >= 1,
            "Black knight at (4,4) should attack (2,5)"
        );

        // All other squares should be 0
        for i in 0..Square::NUM_SQUARES {
            let sq = Square::new_unchecked(i as u8);
            if sq == t1 || sq == t2 {
                continue;
            }
            assert_eq!(
                map[Color::Black as usize][i],
                0,
                "Black knight at (4,4) should NOT attack square {}",
                i
            );
        }
    }

    // -----------------------------------------------------------------------
    // Promoted bishop (Horse) at center has orthogonal step attacks
    // -----------------------------------------------------------------------

    #[test]
    fn test_attack_map_promoted_bishop_has_orthogonal_steps() {
        let pos = lone_piece_pos(4, 4, PieceType::Bishop, Color::Black, true);
        let map = compute_attack_map(&pos);

        // Orthogonal steps: one square in each cardinal direction
        let center = Square::from_row_col(4, 4).unwrap();
        for delta in [UP, DOWN, LEFT, RIGHT] {
            let target = center.offset(delta).expect("center +orthogonal must be in bounds");
            assert!(
                map[Color::Black as usize][target.index()] >= 1,
                "Horse at (4,4) should have orthogonal step attack at delta {}",
                delta
            );
        }

        // Also has diagonal slides (check one diagonal square beyond center)
        let diag = Square::from_row_col(3, 3).unwrap(); // UP_LEFT from center
        assert!(
            map[Color::Black as usize][diag.index()] >= 1,
            "Horse at (4,4) should attack (3,3) via diagonal slide"
        );
    }

    // -----------------------------------------------------------------------
    // Black lance at (4,4) slides toward row 0, not backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_attack_map_lance_slides_forward_only() {
        let pos = lone_piece_pos(4, 4, PieceType::Lance, Color::Black, false);
        let map = compute_attack_map(&pos);

        // Should attack rows 3, 2, 1, 0 in column 4
        for row in 0u8..4 {
            let sq = Square::from_row_col(row, 4).unwrap();
            assert!(
                map[Color::Black as usize][sq.index()] >= 1,
                "Black lance at (4,4) should attack ({},4)",
                row
            );
        }

        // Should NOT attack anything in column 4 below row 4
        for row in 5u8..9 {
            let sq = Square::from_row_col(row, 4).unwrap();
            assert_eq!(
                map[Color::Black as usize][sq.index()],
                0,
                "Black lance at (4,4) should NOT attack ({},4)",
                row
            );
        }
    }

    // -----------------------------------------------------------------------
    // Two rooks attacking the same square → count = 2
    // -----------------------------------------------------------------------

    #[test]
    fn test_attack_count_multiple_attackers() {
        let mut pos = Position::empty();
        // Two Black rooks: one on row 4 col 0, one on row 0 col 4
        let r1 = Square::from_row_col(4, 0).unwrap();
        let r2 = Square::from_row_col(0, 4).unwrap();
        pos.set_piece(r1, Piece::new(PieceType::Rook, Color::Black, false));
        pos.set_piece(r2, Piece::new(PieceType::Rook, Color::Black, false));

        let map = compute_attack_map(&pos);

        // Square (4,4) is on row 4 (attacked by r1) AND on col 4 (attacked by r2)
        let target = Square::from_row_col(4, 4).unwrap();
        assert_eq!(
            map[Color::Black as usize][target.index()],
            2,
            "Square (4,4) should be attacked by both rooks (count=2)"
        );
    }
}
