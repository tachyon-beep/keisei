//! Special Shogi rules: uchi-fu-zume, sennichite, and impasse (24-point rule).

use crate::attack::compute_attack_map;
use crate::game::GameState;
use crate::piece::Piece;
use crate::position::Position;
use crate::types::{Color, GameResult, HandPieceType, PieceType, Square};
use crate::zobrist::ZOBRIST;

// ---------------------------------------------------------------------------
// Uchi-fu-zume (pawn-drop checkmate)
// ---------------------------------------------------------------------------

/// Returns true if dropping a pawn at `to` by `color` would deliver inescapable
/// checkmate, making the drop illegal.
///
/// This function temporarily simulates the drop on the game state, checks for
/// checkmate, and always undoes the simulation before returning.
pub fn is_uchi_fu_zume(game: &mut GameState, to: Square, color: Color) -> bool {
    let z = &*ZOBRIST;
    let opponent = color.opponent();

    // --- Save state for undo ---
    let prev_hash = game.position.hash;
    let prev_attack_map = game.attack_map;
    let prev_board_byte = game.position.board[to.index()];

    // The square must be empty for a drop.
    debug_assert_eq!(prev_board_byte, 0, "is_uchi_fu_zume: target square not empty");

    // --- Simulate the pawn drop ---
    let pawn = Piece::new(PieceType::Pawn, color, false);
    game.position.set_piece(to, pawn);
    game.position.hash ^= z.hash_piece_at(to, pawn);

    // We do NOT decrement hand count or flip side-to-move — we only need to
    // check whether this board position delivers inescapable check.

    // Recompute attack map with the pawn placed.
    game.attack_map = compute_attack_map(&game.position);

    // --- Check if the dropped pawn gives check to the opponent's king ---
    let opp_king_sq = match game.position.find_king(opponent) {
        Some(sq) => sq,
        None => {
            // No opponent king — cannot be uchi-fu-zume.
            // Undo and return false.
            game.position.board[to.index()] = prev_board_byte;
            game.position.hash = prev_hash;
            game.attack_map = prev_attack_map;
            return false;
        }
    };

    let in_check = game.attack_map[color as usize][opp_king_sq.index()] > 0;
    if !in_check {
        // The pawn drop does not give check — not uchi-fu-zume.
        game.position.board[to.index()] = prev_board_byte;
        game.position.hash = prev_hash;
        game.attack_map = prev_attack_map;
        return false;
    }

    // --- The pawn gives check. Verify the opponent has no escape. ---
    let is_mate = !opponent_can_escape(game, to, color, opponent, opp_king_sq);

    // --- Always undo the simulation ---
    game.position.board[to.index()] = prev_board_byte;
    game.position.hash = prev_hash;
    game.attack_map = prev_attack_map;

    is_mate
}

/// Check whether the opponent can escape from check delivered by a pawn drop.
///
/// The opponent can escape by:
/// 1. Moving the king to an adjacent safe square.
/// 2. Capturing the dropped pawn with a non-pinned piece.
fn opponent_can_escape(
    game: &mut GameState,
    pawn_sq: Square,
    dropper: Color,
    opponent: Color,
    king_sq: Square,
) -> bool {
    // 1. King escape: check all ≤8 adjacent squares.
    let king_row = king_sq.row() as i8;
    let king_col = king_sq.col() as i8;
    for dr in -1i8..=1 {
        for dc in -1i8..=1 {
            if dr == 0 && dc == 0 {
                continue;
            }
            let nr = king_row + dr;
            let nc = king_col + dc;
            if nr < 0 || nr >= 9 || nc < 0 || nc >= 9 {
                continue;
            }
            let adj_sq = Square::from_row_col(nr as u8, nc as u8).unwrap();

            // Can't move to a square occupied by own piece.
            if let Some(p) = game.position.piece_at(adj_sq) {
                if p.color() == opponent {
                    continue;
                }
            }

            // Can't move to a square attacked by the dropper.
            if game.attack_map[dropper as usize][adj_sq.index()] > 0 {
                continue;
            }

            // This square is a valid king escape.
            return true;
        }
    }

    // 2. Capture the dropped pawn: find any opponent piece that attacks the pawn square.
    //    For each candidate, simulate the capture and check if the opponent's king
    //    would still be safe (pin detection).
    for i in 0..Square::NUM_SQUARES {
        let sq = Square::new_unchecked(i as u8);
        let piece = match game.position.piece_at(sq) {
            Some(p) if p.color() == opponent => p,
            _ => continue,
        };

        // The king's own capture is already handled above (king escape).
        if piece.piece_type() == PieceType::King {
            continue;
        }

        // Check if this piece attacks the pawn square.
        // We do this by checking if the piece can reach pawn_sq.
        if !piece_attacks_square(&game.position, sq, piece, pawn_sq) {
            continue;
        }

        // Simulate the capture: move this piece to pawn_sq.
        let saved_from = game.position.board[sq.index()];
        let saved_to = game.position.board[pawn_sq.index()];

        game.position.board[sq.index()] = 0;
        game.position.set_piece(pawn_sq, piece);

        // Recompute attack map after the simulated capture.
        let capture_attack_map = compute_attack_map(&game.position);

        // Find opponent's king (it hasn't moved).
        let king_safe = capture_attack_map[dropper as usize][king_sq.index()] == 0;

        // Restore.
        game.position.board[sq.index()] = saved_from;
        game.position.board[pawn_sq.index()] = saved_to;

        if king_safe {
            return true;
        }
    }

    false
}

/// Check if `piece` at `from` attacks `target` square.
///
/// This is a simplified check used for uchi-fu-zume pin detection.
fn piece_attacks_square(pos: &Position, from: Square, piece: Piece, target: Square) -> bool {
    use crate::attack::{compute_knight_attacks, piece_attack_dirs, would_wrap_file};

    let pt = piece.piece_type();
    let color = piece.color();
    let promoted = piece.is_promoted();

    // Knight special case.
    if pt == PieceType::Knight && !promoted {
        return compute_knight_attacks(from, color).contains(&target);
    }

    let (steps, slides) = piece_attack_dirs(pt, color, promoted);

    // Step attacks.
    for delta in &steps {
        if !would_wrap_file(from, *delta) {
            if let Some(sq) = from.offset(*delta) {
                if sq == target {
                    return true;
                }
            }
        }
    }

    // Slide attacks.
    for delta in &slides {
        let mut cur = from;
        loop {
            if would_wrap_file(cur, *delta) {
                break;
            }
            match cur.offset(*delta) {
                None => break,
                Some(next) => {
                    if next == target {
                        return true;
                    }
                    // Blocked by a piece before reaching target.
                    if pos.piece_at(next).is_some() {
                        break;
                    }
                    cur = next;
                }
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Sennichite (fourfold repetition) and perpetual check
// ---------------------------------------------------------------------------

/// Check for fourfold repetition and perpetual check.
///
/// Returns:
/// - `Some(GameResult::Repetition)` if the position has been repeated 4 times
///   and neither side was continuously giving check.
/// - `Some(GameResult::PerpetualCheck { winner })` if one side was always
///   giving check at matching plies.
/// - `None` if the repetition count is below 4.
pub fn check_sennichite(game: &GameState) -> Option<GameResult> {
    let current_hash = game.position.hash;
    let count = game.repetition_map.get(&current_hash).copied().unwrap_or(0);

    if count < 4 {
        return None;
    }

    // Fourfold repetition detected. Walk through hash_history to find matching plies.
    // Also consider the current position (not yet in history).
    let mut matching_plies: Vec<usize> = Vec::new();
    for (ply, &h) in game.hash_history.iter().enumerate() {
        if h == current_hash {
            matching_plies.push(ply);
        }
    }
    // The current position (ply = hash_history.len()) is also a match.
    // check_history doesn't cover the current ply yet, so we compute it.

    // Check if all matching plies had the side-to-move in check.
    // At each matching ply, check_history[ply] tells us if the side-to-move was in check.
    // If the side-to-move was in check, it means the OPPONENT was giving check.
    if matching_plies.is_empty() {
        // Only the current position has this hash repeated — shouldn't reach count >= 4
        // with no history matches, but be defensive.
        return Some(GameResult::Repetition);
    }

    let all_checks = matching_plies.iter().all(|&ply| {
        ply < game.check_history.len() && game.check_history[ply]
    });

    if all_checks {
        // Determine who was giving check: at matching plies, the side-to-move was
        // in check, meaning the opponent of the side-to-move was giving check.
        // Since positions with the same hash have the same side-to-move, the
        // checker is consistent across all matching plies.
        //
        // The current position has the same side-to-move. The opponent of the
        // current side-to-move was the one perpetually checking.
        let checking_side = game.position.current_player.opponent();
        let winner = checking_side.opponent(); // the victim wins
        return Some(GameResult::PerpetualCheck { winner });
    }

    Some(GameResult::Repetition)
}

// ---------------------------------------------------------------------------
// Impasse (CSA 24-point rule)
// ---------------------------------------------------------------------------

/// Check for impasse under the CSA 24-point rule.
///
/// Both kings must have entered the opponent's camp, and both players must have
/// at least 10 pieces in the promotion zone. Then piece values are tallied:
/// Rook/Bishop = 5, all others (except King) = 1. Both board pieces and hand
/// pieces count toward the score.
pub fn check_impasse(game: &GameState) -> Option<GameResult> {
    let pos = &game.position;

    // 1. Find both kings.
    let black_king_sq = pos.find_king(Color::Black)?;
    let white_king_sq = pos.find_king(Color::White)?;

    // 2. Both kings must have entered the opponent's camp.
    // Black's king must be in White's camp (rows 0-2).
    // White's king must be in Black's camp (rows 6-8).
    if black_king_sq.row() > 2 {
        return None;
    }
    if white_king_sq.row() < 6 {
        return None;
    }

    // 3. Count pieces in promotion zone for each color (including king).
    let black_count = count_pieces_in_promotion_zone(pos, Color::Black);
    let white_count = count_pieces_in_promotion_zone(pos, Color::White);

    if black_count < 10 || white_count < 10 {
        return None;
    }

    // 4. Compute impasse scores.
    let black_score = compute_impasse_score(pos, Color::Black);
    let white_score = compute_impasse_score(pos, Color::White);

    // 5. Determine result.
    if black_score >= 24 && white_score >= 24 {
        Some(GameResult::Impasse { winner: None })
    } else if black_score >= 24 {
        Some(GameResult::Impasse {
            winner: Some(Color::Black),
        })
    } else if white_score >= 24 {
        Some(GameResult::Impasse {
            winner: Some(Color::White),
        })
    } else {
        None
    }
}

/// Count pieces of `color` in the opponent's promotion zone (including king).
///
/// - Black's promotion zone: rows 0-2 (White's camp)
/// - White's promotion zone: rows 6-8 (Black's camp)
pub fn count_pieces_in_promotion_zone(pos: &Position, color: Color) -> u8 {
    let mut count = 0u8;
    for i in 0..Square::NUM_SQUARES {
        let sq = Square::new_unchecked(i as u8);
        if let Some(piece) = pos.piece_at(sq) {
            if piece.color() != color {
                continue;
            }
            let row = sq.row();
            let in_zone = match color {
                Color::Black => row <= 2,
                Color::White => row >= 6,
            };
            if in_zone {
                count += 1;
            }
        }
    }
    count
}

/// Compute the impasse score for `color`.
///
/// Rook/Bishop (including promoted forms) = 5 points, all other pieces
/// (except King) = 1 point. Counts both board pieces (anywhere on the board)
/// and hand pieces.
pub fn compute_impasse_score(pos: &Position, color: Color) -> u8 {
    let mut score = 0u8;

    // Board pieces.
    for i in 0..Square::NUM_SQUARES {
        let sq = Square::new_unchecked(i as u8);
        if let Some(piece) = pos.piece_at(sq) {
            if piece.color() != color {
                continue;
            }
            let base_type = piece.piece_type();
            if base_type == PieceType::King {
                continue;
            }
            score += piece_impasse_value(base_type);
        }
    }

    // Hand pieces.
    for &hpt in &HandPieceType::ALL {
        let count = pos.hand_count(color, hpt);
        if count > 0 {
            score += count * piece_impasse_value(hpt.to_piece_type());
        }
    }

    score
}

/// Impasse point value for a base piece type.
fn piece_impasse_value(pt: PieceType) -> u8 {
    match pt {
        PieceType::Rook | PieceType::Bishop => 5,
        PieceType::King => 0,
        _ => 1,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameState;
    use crate::position::Position;
    use crate::types::{Color, Square};

    #[test]
    fn test_impasse_score_starting() {
        let pos = Position::startpos();
        // Each side: 2G + 2S + 2N + 2L + 9P = 18 * 1pt + 1B(5) + 1R(5) = 28
        // Wait: 2G(1) + 2S(1) + 2N(1) + 2L(1) + 9P(1) = 17 * 1 = 17 + 5 + 5 = 27
        let black = compute_impasse_score(&pos, Color::Black);
        let white = compute_impasse_score(&pos, Color::White);
        assert_eq!(black, 27, "Black starting impasse score should be 27");
        assert_eq!(white, 27, "White starting impasse score should be 27");
    }

    #[test]
    fn test_sennichite_not_triggered_below_4() {
        let gs = GameState::new();
        assert_eq!(
            check_sennichite(&gs),
            None,
            "New game should not trigger sennichite"
        );
    }

    #[test]
    fn test_impasse_requires_both_kings_entered() {
        let gs = GameState::new();
        assert_eq!(
            check_impasse(&gs),
            None,
            "Starting position should not trigger impasse"
        );
    }

    #[test]
    fn test_piece_attacks_square_pawn() {
        // Black pawn at (5,4) should attack (4,4).
        let mut pos = Position::empty();
        let sq = Square::from_row_col(5, 4).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        pos.set_piece(sq, pawn);

        let target = Square::from_row_col(4, 4).unwrap();
        assert!(piece_attacks_square(&pos, sq, pawn, target));

        let non_target = Square::from_row_col(6, 4).unwrap();
        assert!(!piece_attacks_square(&pos, sq, pawn, non_target));
    }

    #[test]
    fn test_piece_attacks_square_rook_blocked() {
        // Rook at (4,0), blocker at (4,3), should not attack (4,5).
        let mut pos = Position::empty();
        let rook_sq = Square::from_row_col(4, 0).unwrap();
        let rook = Piece::new(PieceType::Rook, Color::Black, false);
        pos.set_piece(rook_sq, rook);
        let blocker_sq = Square::from_row_col(4, 3).unwrap();
        pos.set_piece(blocker_sq, Piece::new(PieceType::Pawn, Color::White, false));

        let target_blocked = Square::from_row_col(4, 5).unwrap();
        assert!(!piece_attacks_square(&pos, rook_sq, rook, target_blocked));

        // But should attack (4,2).
        let target_ok = Square::from_row_col(4, 2).unwrap();
        assert!(piece_attacks_square(&pos, rook_sq, rook, target_ok));
    }

    #[test]
    fn test_count_pieces_in_promotion_zone_startpos() {
        let pos = Position::startpos();
        // In starting position, White has all pieces in rows 0-2 (their own territory,
        // not Black's promotion zone). Black's promotion zone is rows 0-2, and
        // Black has 0 pieces there initially.
        let black_in_zone = count_pieces_in_promotion_zone(&pos, Color::Black);
        assert_eq!(black_in_zone, 0, "Black has no pieces in promotion zone at start");

        let white_in_zone = count_pieces_in_promotion_zone(&pos, Color::White);
        assert_eq!(white_in_zone, 0, "White has no pieces in promotion zone at start");
    }
}
