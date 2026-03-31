//! Pseudo-legal move generation.
//!
//! Generates all structurally valid moves for a given color without checking:
//! - King safety (legal filtering happens in game.rs)
//! - Nifu / double-pawn (checked in game.rs)
//! - Uchi-fu-zume (checked in rules.rs)

use crate::attack::{compute_knight_attacks, piece_attack_dirs, would_wrap_file};
use crate::piece::Piece;
use crate::position::Position;
use crate::types::{Color, HandPieceType, Move, PieceType, Square};

// ---------------------------------------------------------------------------
// Promotion-zone helpers
// ---------------------------------------------------------------------------

/// Returns true if `row` is inside the promotion zone for `color`.
///
/// - Black: rows 0–2 (ranks 1–3 from Black's perspective)
/// - White: rows 6–8 (ranks 7–9 from White's perspective)
#[inline]
pub fn in_promotion_zone(row: u8, color: Color) -> bool {
    match color {
        Color::Black => row <= 2,
        Color::White => row >= 6,
    }
}

/// Returns true when a piece MUST promote (it would have no legal moves otherwise).
///
/// - Pawn / Lance: last rank (Black: row 0, White: row 8)
/// - Knight: last two ranks (Black: row <= 1, White: row >= 7)
/// - All other pieces: false
#[inline]
pub fn must_promote(piece_type: PieceType, to_row: u8, color: Color) -> bool {
    match piece_type {
        PieceType::Pawn | PieceType::Lance => match color {
            Color::Black => to_row == 0,
            Color::White => to_row == 8,
        },
        PieceType::Knight => match color {
            Color::Black => to_row <= 1,
            Color::White => to_row >= 7,
        },
        _ => false,
    }
}

/// Same semantics as `must_promote` but accepts a `HandPieceType` (used for
/// dead-drop detection — a drop that immediately leaves the piece with no moves).
#[inline]
pub fn is_dead_drop(piece_type: HandPieceType, to_row: u8, color: Color) -> bool {
    match piece_type {
        HandPieceType::Pawn | HandPieceType::Lance => match color {
            Color::Black => to_row == 0,
            Color::White => to_row == 8,
        },
        HandPieceType::Knight => match color {
            Color::Black => to_row <= 1,
            Color::White => to_row >= 7,
        },
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Push board move(s) for a `from → to` pair, applying promotion rules.
///
/// Rules:
/// 1. If `must_promote`: only emit the promoted version.
/// 2. Else if `can_promote` and (to or from is in the promotion zone):
///    emit BOTH the non-promoting and promoting versions.
/// 3. Otherwise: emit the non-promoting version only.
///
/// Pieces that are already promoted cannot promote again.
#[inline]
fn add_board_move_with_promotion(
    from: Square,
    to: Square,
    piece_type: PieceType,
    already_promoted: bool,
    color: Color,
    moves: &mut Vec<Move>,
) {
    let from_row = from.row();
    let to_row = to.row();

    // Already-promoted pieces and Gold/King: no promotion logic.
    if already_promoted || !piece_type.can_promote() {
        moves.push(Move::Board { from, to, promote: false });
        return;
    }

    // Unpromoted, promotable piece.
    if must_promote(piece_type, to_row, color) {
        moves.push(Move::Board { from, to, promote: true });
    } else if in_promotion_zone(from_row, color) || in_promotion_zone(to_row, color) {
        // Optional promotion — emit both choices.
        moves.push(Move::Board { from, to, promote: false });
        moves.push(Move::Board { from, to, promote: true });
    } else {
        moves.push(Move::Board { from, to, promote: false });
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Generate all pseudo-legal board moves for `color`.
///
/// Does NOT check king safety. Caller is responsible for filtering illegal moves.
pub fn generate_pseudo_legal_board_moves(pos: &Position, color: Color, moves: &mut Vec<Move>) {
    for idx in 0..Square::NUM_SQUARES {
        let from = Square::new_unchecked(idx as u8);
        let piece: Piece = match pos.piece_at(from) {
            Some(p) if p.color() == color => p,
            _ => continue,
        };

        let pt = piece.piece_type();
        let promoted = piece.is_promoted();

        // Knights need special handling (they jump, not slide/step).
        if pt == PieceType::Knight && !promoted {
            for to in compute_knight_attacks(from, color) {
                // Skip squares occupied by own pieces.
                if let Some(target_piece) = pos.piece_at(to) {
                    if target_piece.color() == color {
                        continue;
                    }
                }
                add_board_move_with_promotion(from, to, pt, promoted, color, moves);
            }
            continue;
        }

        let (steps, slides) = piece_attack_dirs(pt, color, promoted);

        // Single-step moves.
        for delta in steps {
            if would_wrap_file(from, delta) {
                continue;
            }
            let to = match from.offset(delta) {
                Some(sq) => sq,
                None => continue,
            };
            // Skip squares occupied by own pieces.
            if let Some(target_piece) = pos.piece_at(to) {
                if target_piece.color() == color {
                    continue;
                }
            }
            add_board_move_with_promotion(from, to, pt, promoted, color, moves);
        }

        // Sliding moves (ray-cast until blocked or out of bounds).
        for delta in slides {
            let mut cur = from;
            loop {
                if would_wrap_file(cur, delta) {
                    break;
                }
                let to = match cur.offset(delta) {
                    Some(sq) => sq,
                    None => break,
                };
                if let Some(target_piece) = pos.piece_at(to) {
                    // Can capture enemy, but ray stops here regardless.
                    if target_piece.color() != color {
                        add_board_move_with_promotion(from, to, pt, promoted, color, moves);
                    }
                    break;
                }
                // Empty square — always legal to move there.
                add_board_move_with_promotion(from, to, pt, promoted, color, moves);
                cur = to;
            }
        }
    }
}

/// Generate all pseudo-legal drop moves for `color`.
///
/// Does NOT check nifu or uchi-fu-zume. Caller is responsible for those.
pub fn generate_pseudo_legal_drops(pos: &Position, color: Color, moves: &mut Vec<Move>) {
    for &hpt in &HandPieceType::ALL {
        if pos.hand_count(color, hpt) == 0 {
            continue;
        }
        for idx in 0..Square::NUM_SQUARES {
            let to = Square::new_unchecked(idx as u8);
            // Drops are only on empty squares.
            if pos.piece_at(to).is_some() {
                continue;
            }
            // Skip dead drops (piece would have no subsequent moves).
            if is_dead_drop(hpt, to.row(), color) {
                continue;
            }
            moves.push(Move::Drop { to, piece_type: hpt });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::Piece;
    use crate::position::Position;
    use crate::types::{Color, HandPieceType, Move, PieceType, Square};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn startpos_moves_for(color: Color) -> Vec<Move> {
        let pos = Position::startpos();
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, color, &mut moves);
        moves
    }

    fn lone_piece_pos(row: u8, col: u8, pt: PieceType, color: Color, promoted: bool) -> Position {
        let mut pos = Position::empty();
        let sq = Square::from_row_col(row, col).unwrap();
        pos.set_piece(sq, Piece::new(pt, color, promoted));
        pos
    }

    // -----------------------------------------------------------------------
    // test_startpos_board_moves_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_startpos_board_moves_count() {
        let moves = startpos_moves_for(Color::Black);

        // Count pawn pushes: Black pawns are on row 6, each can push to row 5
        // (optional promotion does not apply there — row 5 is not in zone for Black).
        let pawn_pushes = moves
            .iter()
            .filter(|m| {
                if let Move::Board { from, to, promote: false } = m {
                    from.row() == 6 && to.row() == 5
                } else {
                    false
                }
            })
            .count();

        assert!(
            pawn_pushes >= 9,
            "Expected >= 9 pawn pushes, got {}",
            pawn_pushes
        );
        assert!(
            moves.len() <= 50,
            "Expected <= 50 total moves from startpos, got {}",
            moves.len()
        );
    }

    // -----------------------------------------------------------------------
    // test_knight_forward_direction
    // -----------------------------------------------------------------------

    #[test]
    fn test_knight_forward_direction() {
        // Place knight on row 4 so its targets (row 2) are in the promotion zone —
        // each target square therefore produces 2 moves (promote=false and promote=true).
        let pos = lone_piece_pos(4, 4, PieceType::Knight, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        let t1 = Square::from_row_col(2, 3).unwrap();
        let t2 = Square::from_row_col(2, 5).unwrap();

        // Collect unique destination squares.
        let mut unique_targets: Vec<Square> = moves
            .iter()
            .filter_map(|m| {
                if let Move::Board { to, .. } = m {
                    Some(*to)
                } else {
                    None
                }
            })
            .collect();
        unique_targets.dedup_by_key(|s| s.index());
        // After dedup the two distinct target squares must be present.
        assert!(
            unique_targets.contains(&t1),
            "Black knight at (4,4) should target (2,3)"
        );
        assert!(
            unique_targets.contains(&t2),
            "Black knight at (4,4) should target (2,5)"
        );

        // Row 2 is in Black's promotion zone → each target generates 2 moves.
        // Total: 4 moves (2 per target square).
        assert_eq!(
            moves.len(),
            4,
            "Black knight at (4,4) with both targets in promotion zone should have 4 moves, got {:?}",
            moves
        );
    }

    // -----------------------------------------------------------------------
    // test_forced_promotion
    // -----------------------------------------------------------------------

    #[test]
    fn test_forced_promotion() {
        // Black pawn on row 1 must promote when moving to row 0.
        let pos = lone_piece_pos(1, 4, PieceType::Pawn, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        // Should produce exactly 1 move (promote=true only).
        assert_eq!(
            moves.len(),
            1,
            "Expected exactly 1 move (forced promotion), got {:?}",
            moves
        );
        match moves[0] {
            Move::Board { promote: true, .. } => {}
            _ => panic!("Expected a promoting move, got {:?}", moves[0]),
        }
    }

    // -----------------------------------------------------------------------
    // test_optional_promotion
    // -----------------------------------------------------------------------

    #[test]
    fn test_optional_promotion() {
        // Black pawn on row 3 moves to row 2 (inside promotion zone for Black).
        // Should produce 2 moves: promote=false and promote=true.
        let pos = lone_piece_pos(3, 4, PieceType::Pawn, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        assert_eq!(
            moves.len(),
            2,
            "Expected exactly 2 moves (optional promotion), got {:?}",
            moves
        );

        let has_non_promote = moves.iter().any(|m| matches!(m, Move::Board { promote: false, .. }));
        let has_promote = moves.iter().any(|m| matches!(m, Move::Board { promote: true, .. }));
        assert!(has_non_promote, "Missing non-promoting move");
        assert!(has_promote, "Missing promoting move");
    }

    // -----------------------------------------------------------------------
    // test_dead_drop_prevention
    // -----------------------------------------------------------------------

    #[test]
    fn test_dead_drop_prevention() {
        // Pawn
        assert!(is_dead_drop(HandPieceType::Pawn, 0, Color::Black), "Pawn drop to row 0 is dead for Black");
        assert!(!is_dead_drop(HandPieceType::Pawn, 1, Color::Black), "Pawn drop to row 1 is not dead for Black");
        assert!(is_dead_drop(HandPieceType::Pawn, 8, Color::White), "Pawn drop to row 8 is dead for White");
        assert!(!is_dead_drop(HandPieceType::Pawn, 7, Color::White), "Pawn drop to row 7 is not dead for White");

        // Lance
        assert!(is_dead_drop(HandPieceType::Lance, 0, Color::Black), "Lance drop to row 0 is dead for Black");
        assert!(!is_dead_drop(HandPieceType::Lance, 1, Color::Black), "Lance drop to row 1 is not dead for Black");

        // Knight
        assert!(is_dead_drop(HandPieceType::Knight, 0, Color::Black), "Knight drop to row 0 is dead for Black");
        assert!(is_dead_drop(HandPieceType::Knight, 1, Color::Black), "Knight drop to row 1 is dead for Black");
        assert!(!is_dead_drop(HandPieceType::Knight, 2, Color::Black), "Knight drop to row 2 is not dead for Black");
        assert!(is_dead_drop(HandPieceType::Knight, 8, Color::White), "Knight drop to row 8 is dead for White");
        assert!(is_dead_drop(HandPieceType::Knight, 7, Color::White), "Knight drop to row 7 is dead for White");
        assert!(!is_dead_drop(HandPieceType::Knight, 6, Color::White), "Knight drop to row 6 is not dead for White");

        // Gold — never a dead drop
        assert!(!is_dead_drop(HandPieceType::Gold, 0, Color::Black), "Gold is never a dead drop");
        assert!(!is_dead_drop(HandPieceType::Gold, 8, Color::White), "Gold is never a dead drop");

        // Silver — never a dead drop
        assert!(!is_dead_drop(HandPieceType::Silver, 0, Color::Black), "Silver is never a dead drop");
    }

    // -----------------------------------------------------------------------
    // test_drops_only_on_empty_squares
    // -----------------------------------------------------------------------

    #[test]
    fn test_drops_only_on_empty_squares() {
        // Give Black a rook in hand; use startpos so the board is not empty.
        let mut pos = Position::startpos();
        pos.set_hand_count(Color::Black, HandPieceType::Rook, 1);

        let mut moves = Vec::new();
        generate_pseudo_legal_drops(&pos, Color::Black, &mut moves);

        for m in &moves {
            if let Move::Drop { to, .. } = m {
                assert!(
                    pos.piece_at(*to).is_none(),
                    "Drop target {:?} must be empty",
                    to
                );
            }
        }
    }
}
