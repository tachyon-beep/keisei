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
                if let Some(target_piece) = pos.piece_at(to)
                    && target_piece.color() == color {
                        continue;
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
            if let Some(target_piece) = pos.piece_at(to)
                && target_piece.color() == color {
                    continue;
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

    // -----------------------------------------------------------------------
    // White perspective movegen — Gap #7
    // -----------------------------------------------------------------------

    /// White pawn moves DOWN (forward for White).
    #[test]
    fn test_white_pawn_moves_forward() {
        let pos = lone_piece_pos(2, 4, PieceType::Pawn, Color::White, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::White, &mut moves);

        // White pawn at row 2 → moves to row 3 (DOWN).
        // Row 3 is NOT in White's promotion zone (rows 6-8), so no promotion variant.
        assert_eq!(moves.len(), 1, "White pawn at (2,4) should have 1 move");
        match moves[0] {
            Move::Board { from, to, promote: false } => {
                assert_eq!(from.row(), 2);
                assert_eq!(to.row(), 3);
                assert_eq!(to.col(), 4);
            }
            _ => panic!("Expected non-promoting board move, got {:?}", moves[0]),
        }
    }

    /// White pawn approaching promotion zone gets optional promotion.
    #[test]
    fn test_white_pawn_optional_promotion() {
        // White pawn at row 5 → moves to row 6 (inside White's promotion zone)
        let pos = lone_piece_pos(5, 4, PieceType::Pawn, Color::White, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::White, &mut moves);

        assert_eq!(
            moves.len(), 2,
            "White pawn at (5,4) should have 2 moves (promote + non-promote)"
        );
        let has_promote = moves.iter().any(|m| matches!(m, Move::Board { promote: true, .. }));
        let has_non_promote = moves.iter().any(|m| matches!(m, Move::Board { promote: false, .. }));
        assert!(has_promote, "Missing promotion move");
        assert!(has_non_promote, "Missing non-promotion move");
    }

    /// White pawn at row 7 must promote when moving to row 8.
    #[test]
    fn test_white_pawn_forced_promotion() {
        let pos = lone_piece_pos(7, 4, PieceType::Pawn, Color::White, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::White, &mut moves);

        assert_eq!(moves.len(), 1, "White pawn at (7,4) should have 1 forced promotion");
        match moves[0] {
            Move::Board { promote: true, .. } => {}
            _ => panic!("Expected forced promotion move"),
        }
    }

    /// White knight jumps DOWN 2 rows.
    #[test]
    fn test_white_knight_forward_direction() {
        let pos = lone_piece_pos(4, 4, PieceType::Knight, Color::White, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::White, &mut moves);

        // White knight at (4,4) targets (6,3) and (6,5).
        // Row 6 is in White's promotion zone → each target gets 2 moves.
        let t1 = Square::from_row_col(6, 3).unwrap();
        let t2 = Square::from_row_col(6, 5).unwrap();

        let targets: Vec<Square> = moves
            .iter()
            .filter_map(|m| if let Move::Board { to, .. } = m { Some(*to) } else { None })
            .collect();
        assert!(targets.contains(&t1), "White knight should target (6,3)");
        assert!(targets.contains(&t2), "White knight should target (6,5)");
        assert_eq!(moves.len(), 4, "White knight at (4,4) with both targets in promotion zone should have 4 moves");
    }

    /// White startpos opening moves count.
    #[test]
    fn test_startpos_white_board_moves() {
        let pos = Position::startpos();
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::White, &mut moves);

        // White pawns on row 2, each pushes to row 3 (outside promo zone).
        let pawn_pushes = moves
            .iter()
            .filter(|m| matches!(m, Move::Board { from, to, promote: false } if from.row() == 2 && to.row() == 3))
            .count();
        assert_eq!(pawn_pushes, 9, "White should have 9 pawn pushes from row 2");
    }

    // -----------------------------------------------------------------------
    // Sliding piece move generation — Gap #8
    // -----------------------------------------------------------------------

    /// Lone rook on empty board at center should attack 16 squares
    /// (4 in each cardinal direction from center of 9×9).
    #[test]
    fn test_lone_rook_move_count() {
        // Rook at (4,4) on empty board: 4 squares in each of 4 directions = 16 targets.
        // None are in Black's promotion zone (rows 0-2), except targets at rows 0-2 col 4:
        //   row 0-2 = 3 targets in promotion zone.
        // From (4,4) to rows 0-2: optional promotion → each produces 2 moves.
        // From (4,4) to rows 3, 5-8 and cols 0-3, 5-8: no promotion → 1 move each.
        // Row targets: 0,1,2 (3 promo zone), 3,5,6,7,8 (5 normal) = 8 total row targets.
        // Col targets: 0,1,2,3,5,6,7,8 = 8 total col targets, none in promo zone for Black.
        // Promo moves: 3 targets × 2 = 6
        // Normal moves: 5 row targets × 1 + 8 col targets × 1 = 13
        // Total: 6 + 13 = 19
        let pos = lone_piece_pos(4, 4, PieceType::Rook, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        assert_eq!(
            moves.len(), 19,
            "Lone rook at (4,4) should have 19 moves (including promotion variants)"
        );
    }

    /// Lone bishop on empty board at center.
    #[test]
    fn test_lone_bishop_move_count() {
        // Bishop at (4,4) on empty board.
        // Diagonals from (4,4):
        //   UP_LEFT: (3,3), (2,2), (1,1), (0,0) → 4 targets
        //   UP_RIGHT: (3,5), (2,6), (1,7), (0,8) → 4 targets
        //   DOWN_LEFT: (5,3), (6,2), (7,1), (8,0) → 4 targets
        //   DOWN_RIGHT: (5,5), (6,6), (7,7), (8,8) → 4 targets
        // Total: 16 unique target squares.
        // Promotion zone for Black = rows 0-2.
        // In promo zone: (2,2), (1,1), (0,0), (2,6), (1,7), (0,8) = 6 targets
        // Also: from-square (4,4) is NOT in promo zone.
        // So each of those 6 targets gets 2 moves (promote + non-promote).
        // Remaining 10 targets get 1 move each.
        // Total: 6*2 + 10*1 = 22
        let pos = lone_piece_pos(4, 4, PieceType::Bishop, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        assert_eq!(
            moves.len(), 22,
            "Lone bishop at (4,4) should have 22 moves (including promotion variants)"
        );
    }

    /// Lone lance slides only forward (UP for Black).
    #[test]
    fn test_lone_lance_move_count() {
        // Black lance at (4,4) slides UP only: (3,4), (2,4), (1,4), (0,4) = 4 targets.
        // Promo zone targets (rows 0-2): (2,4), (1,4), (0,4) = 3
        //   (0,4) = forced promotion → 1 move
        //   (1,4) and (2,4) = optional promotion → 2 moves each
        // Non-promo: (3,4) → 1 move
        // Total: 1 + 2 + 2 + 1 = 6
        let pos = lone_piece_pos(4, 4, PieceType::Lance, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        assert_eq!(
            moves.len(), 6,
            "Lone lance at (4,4) should have 6 moves (including promotion variants)"
        );
    }

    // -----------------------------------------------------------------------
    // Sliding piece blocked by own piece — no self-capture
    // -----------------------------------------------------------------------

    /// Rook blocked by own pawn: ray stops before the friendly piece.
    #[test]
    fn test_rook_blocked_by_own_piece() {
        let mut pos = Position::empty();
        // Black rook at (4,4)
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        // Black pawn at (4,6) — blocks rook's rightward ray
        pos.set_piece(
            Square::from_row_col(4, 6).unwrap(),
            Piece::new(PieceType::Pawn, Color::Black, false),
        );
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        // Rook should NOT have any move to (4,6) or beyond (4,7), (4,8)
        let rook_moves: Vec<_> = moves
            .iter()
            .filter(|m| {
                if let Move::Board { from, .. } = m {
                    *from == Square::from_row_col(4, 4).unwrap()
                } else {
                    false
                }
            })
            .collect();

        for m in &rook_moves {
            if let Move::Board { to, .. } = m {
                assert_ne!(
                    *to,
                    Square::from_row_col(4, 6).unwrap(),
                    "Rook should not capture own pawn at (4,6)"
                );
                assert_ne!(
                    *to,
                    Square::from_row_col(4, 7).unwrap(),
                    "Rook ray should not pass through own pawn to (4,7)"
                );
                assert_ne!(
                    *to,
                    Square::from_row_col(4, 8).unwrap(),
                    "Rook ray should not pass through own pawn to (4,8)"
                );
            }
        }

        // Rook should still reach (4,5) — one square before own pawn
        let reaches_45 = rook_moves.iter().any(|m| {
            matches!(m, Move::Board { to, .. } if *to == Square::from_row_col(4, 5).unwrap())
        });
        assert!(reaches_45, "Rook should be able to reach (4,5)");
    }

    /// Bishop blocked by own piece: diagonal ray stops.
    #[test]
    fn test_bishop_blocked_by_own_piece() {
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Bishop, Color::Black, false),
        );
        // Own piece at (2,2) blocks the UP_LEFT diagonal
        pos.set_piece(
            Square::from_row_col(2, 2).unwrap(),
            Piece::new(PieceType::Gold, Color::Black, false),
        );
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        let bishop_moves: Vec<_> = moves
            .iter()
            .filter(|m| {
                if let Move::Board { from, .. } = m {
                    *from == Square::from_row_col(4, 4).unwrap()
                } else {
                    false
                }
            })
            .collect();

        // Bishop should reach (3,3) but NOT (2,2), (1,1), or (0,0)
        let reaches_33 = bishop_moves.iter().any(|m| {
            matches!(m, Move::Board { to, .. } if *to == Square::from_row_col(3, 3).unwrap())
        });
        assert!(reaches_33, "Bishop should reach (3,3) before blocker");

        let reaches_22 = bishop_moves.iter().any(|m| {
            matches!(m, Move::Board { to, .. } if *to == Square::from_row_col(2, 2).unwrap())
        });
        assert!(!reaches_22, "Bishop should NOT capture own piece at (2,2)");
    }

    // -----------------------------------------------------------------------
    // Edge-square movegen: Silver, Gold, Horse, Dragon
    // -----------------------------------------------------------------------

    /// Silver at corner (0,0) for Black. Silver steps:
    /// forward(UP), fwd_left(UP_LEFT), fwd_right(UP_RIGHT),
    /// bwd_left(DOWN_LEFT), bwd_right(DOWN_RIGHT).
    /// From (0,0): UP=OOB, UP_LEFT=OOB/wrap, UP_RIGHT=OOB/wrap,
    /// DOWN_LEFT=wraps(col -1), DOWN_RIGHT=(1,1).
    /// Only (1,1) should be reachable.
    #[test]
    fn test_silver_at_corner_0_0() {
        let pos = lone_piece_pos(0, 0, PieceType::Silver, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        let targets: Vec<Square> = moves
            .iter()
            .filter_map(|m| if let Move::Board { to, .. } = m { Some(*to) } else { None })
            .collect();

        // (1,1) is the only reachable square
        assert!(
            targets.contains(&Square::from_row_col(1, 1).unwrap()),
            "Silver at (0,0) should reach (1,1) via DOWN_RIGHT"
        );

        // Should NOT wrap to any square on col 8
        for t in &targets {
            assert_ne!(t.col(), 8, "Silver at (0,0) should not wrap to col 8");
        }

        // Unique target squares: only (1,1)
        let unique: std::collections::HashSet<usize> = targets.iter().map(|s| s.index()).collect();
        assert_eq!(unique.len(), 1, "Silver at (0,0) should have exactly 1 target square, got {:?}", targets);
    }

    /// Silver at corner (0,8) for Black.
    /// From (0,8): DOWN_RIGHT wraps to col 0 — should be blocked.
    /// DOWN_LEFT=(1,7) is the only valid target.
    #[test]
    fn test_silver_at_corner_0_8() {
        let pos = lone_piece_pos(0, 8, PieceType::Silver, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        let targets: Vec<Square> = moves
            .iter()
            .filter_map(|m| if let Move::Board { to, .. } = m { Some(*to) } else { None })
            .collect();

        let unique: std::collections::HashSet<usize> = targets.iter().map(|s| s.index()).collect();
        assert_eq!(unique.len(), 1, "Silver at (0,8) should have exactly 1 target, got {:?}", targets);
        assert!(targets.contains(&Square::from_row_col(1, 7).unwrap()));
    }

    /// Gold at corner (0,0) for Black.
    /// Gold steps: forward(UP), fwd_left(UP_LEFT), fwd_right(UP_RIGHT),
    /// LEFT, RIGHT, backward(DOWN).
    /// From (0,0): UP=OOB, UP_LEFT=OOB, UP_RIGHT=OOB, LEFT=wrap, RIGHT=(0,1), DOWN=(1,0).
    /// Reachable: (0,1), (1,0).
    #[test]
    fn test_gold_at_corner_0_0() {
        let pos = lone_piece_pos(0, 0, PieceType::Gold, Color::Black, false);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        let targets: Vec<Square> = moves
            .iter()
            .filter_map(|m| if let Move::Board { to, .. } = m { Some(*to) } else { None })
            .collect();

        let unique: std::collections::HashSet<usize> = targets.iter().map(|s| s.index()).collect();
        assert_eq!(unique.len(), 2, "Gold at (0,0) should have 2 targets, got {:?}", targets);
        assert!(targets.contains(&Square::from_row_col(0, 1).unwrap()));
        assert!(targets.contains(&Square::from_row_col(1, 0).unwrap()));
    }

    /// Horse (promoted Bishop) at corner (0,0).
    /// Slides: all 4 diagonals (only DOWN_RIGHT = (1,1)..(8,8) works from (0,0)).
    /// Steps: UP, DOWN, LEFT, RIGHT → only DOWN=(1,0) and RIGHT=(0,1).
    /// Targets: (0,1), (1,0) from steps + (1,1),(2,2),...,(8,8) from slide.
    #[test]
    fn test_horse_at_corner_0_0() {
        let pos = lone_piece_pos(0, 0, PieceType::Bishop, Color::Black, true);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        let targets: std::collections::HashSet<usize> = moves
            .iter()
            .filter_map(|m| if let Move::Board { to, .. } = m { Some(to.index()) } else { None })
            .collect();

        // Steps: (0,1) and (1,0)
        assert!(targets.contains(&Square::from_row_col(0, 1).unwrap().index()), "Horse step to (0,1)");
        assert!(targets.contains(&Square::from_row_col(1, 0).unwrap().index()), "Horse step to (1,0)");
        // Diagonal slide: (1,1) through (8,8)
        for i in 1..=8 {
            assert!(
                targets.contains(&Square::from_row_col(i, i).unwrap().index()),
                "Horse diagonal slide to ({},{})", i, i
            );
        }
        // Total unique targets: 2 steps + 8 diagonal = 10
        assert_eq!(targets.len(), 10, "Horse at (0,0) should have 10 unique targets");
    }

    /// Dragon (promoted Rook) at corner (8,8).
    /// Slides: UP=(7,8)..(0,8), LEFT=(8,7)..(8,0). DOWN and RIGHT = OOB.
    /// Steps: UP_LEFT=(7,7), UP_RIGHT=wrap, DOWN_LEFT=wrap, DOWN_RIGHT=OOB.
    /// Just UP_LEFT from (8,8) → (7,7).
    #[test]
    fn test_dragon_at_corner_8_8() {
        let pos = lone_piece_pos(8, 8, PieceType::Rook, Color::Black, true);
        let mut moves = Vec::new();
        generate_pseudo_legal_board_moves(&pos, Color::Black, &mut moves);

        let targets: std::collections::HashSet<usize> = moves
            .iter()
            .filter_map(|m| if let Move::Board { to, .. } = m { Some(to.index()) } else { None })
            .collect();

        // Diagonal step: only (7,7)
        assert!(targets.contains(&Square::from_row_col(7, 7).unwrap().index()), "Dragon step to (7,7)");
        // Orthogonal slides: UP col 8 (rows 0-7) + LEFT row 8 (cols 0-7)
        for r in 0..8 {
            assert!(
                targets.contains(&Square::from_row_col(r, 8).unwrap().index()),
                "Dragon UP slide to ({},8)", r
            );
        }
        for c in 0..8 {
            assert!(
                targets.contains(&Square::from_row_col(8, c).unwrap().index()),
                "Dragon LEFT slide to (8,{})", c
            );
        }
        // Total: 1 step + 8 up + 8 left = 17
        assert_eq!(targets.len(), 17, "Dragon at (8,8) should have 17 unique targets");
    }

    /// Drop generation with zero hand pieces should produce no drops.
    #[test]
    fn test_no_drops_with_empty_hand() {
        let pos = Position::startpos(); // All pieces on board, nothing in hand
        let mut moves = Vec::new();
        generate_pseudo_legal_drops(&pos, Color::Black, &mut moves);
        assert!(moves.is_empty(), "No drops should be generated with empty hand");
    }
}
