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
            if !(0..9).contains(&nr) || !(0..9).contains(&nc) {
                continue;
            }
            let adj_sq = Square::from_row_col(nr as u8, nc as u8).unwrap();

            // Can't move to a square occupied by own piece.
            if let Some(p) = game.position.piece_at(adj_sq)
                && p.color() == opponent {
                    continue;
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
        if !would_wrap_file(from, *delta)
            && let Some(sq) = from.offset(*delta)
                && sq == target {
                    return true;
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

    // -----------------------------------------------------------------------
    // Uchi-fu-zume (pawn-drop checkmate) — Gap #1
    // -----------------------------------------------------------------------

    /// Test that dropping a pawn to deliver inescapable checkmate is detected.
    ///
    /// Position (Black to move):
    ///   - White king at (0,0), cornered
    ///   - Black gold at (2,1) — far enough to not attack the White king directly
    ///     but after a pawn drop at (1,0), the gold at (2,1) covers (1,0), (1,1)
    ///   - Black lance at (8,0) — covers column 0, preventing White king from
    ///     going to (1,0) after the pawn is dropped (lance attacks up through col 0)
    ///   - Black rook at (0,8) — covers row 0, attacking (0,1) so king can't go right
    ///   - Black king at (8,8)
    ///   - Black has a pawn in hand
    ///
    /// Dropping pawn at (1,0): pawn attacks (0,0) = check.
    /// King escapes? (0,1) attacked by rook. (1,0) occupied by pawn.
    /// (1,1) attacked by gold at (2,1) (gold moves: forward, fwd-left, fwd-right,
    ///   left, right, backward — for White's opponent (Black) gold at (2,1),
    ///   forward=UP so (1,1) is fwd-right).
    /// Can White capture pawn at (1,0)? Only the king could, but (1,0) is the
    ///   pawn itself and also covered by lance.
    /// → Checkmate via pawn drop → uchi-fu-zume.
    #[test]
    fn test_uchi_fu_zume_positive() {
        use crate::types::HandPieceType;

        let mut pos = Position::empty();
        // White king cornered at (0,0)
        pos.set_piece(
            Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // Black king far away
        pos.set_piece(
            Square::from_row_col(8, 8).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // Black rook at (0,8) — attacks all of row 0 including (0,1)
        pos.set_piece(
            Square::from_row_col(0, 8).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        // Black gold at (2,1) — after pawn drop, covers (1,1) via step attack
        pos.set_piece(
            Square::from_row_col(2, 1).unwrap(),
            Piece::new(PieceType::Gold, Color::Black, false),
        );
        // Black lance at (8,0) — slides up column 0, covers (1,0) after pawn drop
        // (ray blocked by pawn at (1,0), but the pawn itself is "protected")
        pos.set_piece(
            Square::from_row_col(8, 0).unwrap(),
            Piece::new(PieceType::Lance, Color::Black, false),
        );
        // Black pawn in hand
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 1);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // Drop pawn at (1, 0): Black pawn attacks UP → (0, 0) = White king.
        let drop_sq = Square::from_row_col(1, 0).unwrap();
        assert!(
            is_uchi_fu_zume(&mut gs, drop_sq, Color::Black),
            "Pawn drop at (1,0) should be uchi-fu-zume"
        );
    }

    /// Test that a pawn drop giving check is NOT uchi-fu-zume when the king can escape.
    #[test]
    fn test_uchi_fu_zume_negative_king_escapes() {
        use crate::types::HandPieceType;

        let mut pos = Position::empty();
        // White king at (0,4) — center of back rank, has escape squares
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // Black king far away
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // Black has a pawn in hand
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 1);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // Dropping pawn at (1,4) gives check on White king at (0,4),
        // but king can escape to (0,3), (0,5), (1,3), (1,5), etc.
        let drop_sq = Square::from_row_col(1, 4).unwrap();
        assert!(
            !is_uchi_fu_zume(&mut gs, drop_sq, Color::Black),
            "Pawn drop at (1,4) should NOT be uchi-fu-zume — king can escape"
        );
    }

    /// Test that a pawn drop NOT giving check is NOT uchi-fu-zume.
    #[test]
    fn test_uchi_fu_zume_negative_no_check() {
        use crate::types::HandPieceType;

        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(8, 8).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 1);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // Drop pawn at (4,4) — nowhere near the White king → not check → not uchi-fu-zume.
        let drop_sq = Square::from_row_col(4, 4).unwrap();
        assert!(
            !is_uchi_fu_zume(&mut gs, drop_sq, Color::Black),
            "Pawn drop far from king should not be uchi-fu-zume"
        );
    }

    // -----------------------------------------------------------------------
    // Sennichite (fourfold repetition) — Gap #2
    // -----------------------------------------------------------------------

    /// Test 4-fold repetition detection via shuttling kings back and forth.
    ///
    /// Minimal position: Black king at (8,4), White king at (0,4).
    /// Black moves king (8,4)→(7,4), White moves king (0,4)→(1,4),
    /// Black moves king (7,4)→(8,4), White moves king (1,4)→(0,4).
    /// Repeat this cycle — after 3 full cycles (12 half-moves) the starting
    /// position will have appeared 4 times.
    #[test]
    fn test_sennichite_fourfold_repetition() {
        use crate::types::Move;

        // Minimal two-king position.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        let bk_down = Move::Board {
            from: Square::from_row_col(8, 4).unwrap(),
            to: Square::from_row_col(7, 4).unwrap(),
            promote: false,
        };
        let bk_up = Move::Board {
            from: Square::from_row_col(7, 4).unwrap(),
            to: Square::from_row_col(8, 4).unwrap(),
            promote: false,
        };
        let wk_down = Move::Board {
            from: Square::from_row_col(0, 4).unwrap(),
            to: Square::from_row_col(1, 4).unwrap(),
            promote: false,
        };
        let wk_up = Move::Board {
            from: Square::from_row_col(1, 4).unwrap(),
            to: Square::from_row_col(0, 4).unwrap(),
            promote: false,
        };

        // Cycle: Black down, White down, Black up, White up → back to start
        // Need 3 full cycles for the start position to appear 4 times
        // (1 initial + 3 returns = 4)
        for _ in 0..3 {
            gs.make_move(bk_down); // Black king to (7,4)
            gs.make_move(wk_down); // White king to (1,4)
            gs.make_move(bk_up);   // Black king to (8,4)
            gs.make_move(wk_up);   // White king to (0,4) → start position repeated
        }

        // Now the start position hash should have count >= 4
        let result = check_sennichite(&gs);
        assert!(
            result.is_some(),
            "Should detect fourfold repetition after 3 cycles"
        );
        assert_eq!(
            result.unwrap(),
            GameResult::Repetition,
            "Simple king shuttle should be Repetition, not PerpetualCheck"
        );
    }

    /// Test that 3-fold repetition does NOT trigger sennichite.
    #[test]
    fn test_sennichite_not_triggered_at_threefold() {
        use crate::types::Move;

        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        let bk_down = Move::Board {
            from: Square::from_row_col(8, 4).unwrap(),
            to: Square::from_row_col(7, 4).unwrap(),
            promote: false,
        };
        let bk_up = Move::Board {
            from: Square::from_row_col(7, 4).unwrap(),
            to: Square::from_row_col(8, 4).unwrap(),
            promote: false,
        };
        let wk_down = Move::Board {
            from: Square::from_row_col(0, 4).unwrap(),
            to: Square::from_row_col(1, 4).unwrap(),
            promote: false,
        };
        let wk_up = Move::Board {
            from: Square::from_row_col(1, 4).unwrap(),
            to: Square::from_row_col(0, 4).unwrap(),
            promote: false,
        };

        // Only 2 full cycles → position appears 3 times (1 initial + 2 returns)
        for _ in 0..2 {
            gs.make_move(bk_down);
            gs.make_move(wk_down);
            gs.make_move(bk_up);
            gs.make_move(wk_up);
        }

        assert_eq!(
            check_sennichite(&gs),
            None,
            "3-fold repetition should NOT trigger sennichite (need 4)"
        );
    }

    // -----------------------------------------------------------------------
    // Perpetual check — Gap #3
    // -----------------------------------------------------------------------

    /// Test perpetual check detection.
    ///
    /// White king at (0,0). Black rook shuttles between (0,8) and (1,8),
    /// giving check from row 0 and not-check from row 1. But we need the
    /// check to happen at the SAME position hash for it to count as
    /// perpetual check. Instead, we use a setup where Black continuously
    /// gives check at matching positions.
    ///
    /// Simpler approach: Black rook at (0,8) checks White king at (0,0).
    /// White king has to move to (1,0). Then Black rook goes to (1,8) to
    /// check again. White king goes back to (0,0). Black rook goes to (0,8).
    /// This creates a cycle where Black is always giving check when it's
    /// White's turn (the side-to-move is always in check at matching positions).
    #[test]
    fn test_perpetual_check_detection() {
        use crate::types::Move;

        let mut pos = Position::empty();
        // White king at (0,0)
        pos.set_piece(
            Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // Black king far away at (8,8)
        pos.set_piece(
            Square::from_row_col(8, 8).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // Black rook at (0,8) — gives check along row 0 to White king
        pos.set_piece(
            Square::from_row_col(0, 8).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        pos.current_player = Color::White;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // White is in check from the rook. White must escape.
        assert!(gs.is_in_check(), "White king should be in check initially");

        // Cycle:
        // 1. White king (0,0)→(1,0) — escapes check
        // 2. Black rook (0,8)→(1,8) — gives check along row 1
        // 3. White king (1,0)→(0,0) — escapes check
        // 4. Black rook (1,8)→(0,8) — gives check along row 0 → back to start
        let wk_escape = Move::Board {
            from: Square::from_row_col(0, 0).unwrap(),
            to: Square::from_row_col(1, 0).unwrap(),
            promote: false,
        };
        let br_chase_1 = Move::Board {
            from: Square::from_row_col(0, 8).unwrap(),
            to: Square::from_row_col(1, 8).unwrap(),
            promote: false,
        };
        let wk_return = Move::Board {
            from: Square::from_row_col(1, 0).unwrap(),
            to: Square::from_row_col(0, 0).unwrap(),
            promote: false,
        };
        let br_chase_2 = Move::Board {
            from: Square::from_row_col(1, 8).unwrap(),
            to: Square::from_row_col(0, 8).unwrap(),
            promote: false,
        };

        // 3 full cycles → the start position (White to move, in check) appears 4 times
        for _ in 0..3 {
            gs.make_move(wk_escape);
            gs.make_move(br_chase_1);
            gs.make_move(wk_return);
            gs.make_move(br_chase_2);
        }

        let result = check_sennichite(&gs);
        assert!(result.is_some(), "Should detect repetition with perpetual check");
        match result.unwrap() {
            GameResult::PerpetualCheck { winner } => {
                // The side giving check (Black) loses; the victim (White) wins.
                assert_eq!(
                    winner,
                    Color::White,
                    "White (the victim of perpetual check) should win"
                );
            }
            other => panic!(
                "Expected PerpetualCheck, got {:?}",
                other
            ),
        }
    }

    // -----------------------------------------------------------------------
    // Impasse (CSA 24-point rule) — Gap #4
    // -----------------------------------------------------------------------

    /// Test impasse scoring with pieces in hand.
    #[test]
    fn test_impasse_score_with_hand_pieces() {
        let mut pos = Position::startpos();
        // Give Black 2 pawns in hand (each worth 1 point)
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 2);
        let score = compute_impasse_score(&pos, Color::Black);
        // Starting: 27. +2 pawns in hand = 29.
        assert_eq!(score, 29, "Black should have 29 points with 2 extra pawns in hand");
    }

    /// Test impasse scoring counts promoted pieces at their base value.
    #[test]
    fn test_impasse_score_promoted_piece_value() {
        let mut pos = Position::empty();
        // Promoted rook (Dragon) on board — still worth 5 points
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, true),
        );
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        let score = compute_impasse_score(&pos, Color::Black);
        assert_eq!(score, 5, "Promoted rook should still be worth 5 points");
    }

    /// Test that check_impasse triggers when both kings have entered and
    /// both players have 10+ pieces in the opponent's promotion zone.
    #[test]
    fn test_impasse_triggers_correctly() {
        // We need:
        //   - Black king in rows 0-2 (White's camp)
        //   - White king in rows 6-8 (Black's camp)
        //   - Both have 10+ pieces in their promotion zone
        //   - Score determines winner
        let mut pos = Position::empty();

        // Black king at (0,4) — in White's camp
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // White king at (8,4) — in Black's camp
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );

        // Place 10 Black pieces in rows 0-2 (Black's promotion zone)
        // King already counts as 1. Need 9 more.
        let black_squares: [(u8, u8); 9] = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 5),
            (0, 6), (0, 7), (0, 8), (1, 0),
        ];
        for &(r, c) in &black_squares {
            pos.set_piece(
                Square::from_row_col(r, c).unwrap(),
                Piece::new(PieceType::Pawn, Color::Black, false),
            );
        }
        // Black score: King(0) + 9 Pawns(1 each) = 9 from board.
        // Give Black rook+bishop in hand: 5+5=10 → total 19 < 24.
        // Actually, let's give enough: 9 pawns + rook(5) + bishop(5) board pieces
        // Replace two pawns with rook and bishop.
        pos.set_piece(
            Square::from_row_col(1, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::Bishop, Color::Black, false),
        );
        // Now Black in zone: King + Bishop(5) + Rook(5) + 7 Pawns(1 each) = 17
        // Plus give hand pieces: 2 Gold + 2 Silver + 2 Knight + 2 Lance = 8 → total 25 >= 24
        pos.set_hand_count(Color::Black, HandPieceType::Gold, 2);
        pos.set_hand_count(Color::Black, HandPieceType::Silver, 2);
        pos.set_hand_count(Color::Black, HandPieceType::Knight, 2);
        pos.set_hand_count(Color::Black, HandPieceType::Lance, 2);
        // Black board score: 5+5+7 = 17, hand: 8 = total 25

        // Place 10 White pieces in rows 6-8 (White's promotion zone)
        let white_squares: [(u8, u8); 9] = [
            (8, 0), (8, 1), (8, 2), (8, 3), (8, 5),
            (8, 6), (8, 7), (8, 8), (7, 0),
        ];
        for &(r, c) in &white_squares {
            pos.set_piece(
                Square::from_row_col(r, c).unwrap(),
                Piece::new(PieceType::Pawn, Color::White, false),
            );
        }
        // White score: 9 pawns = 9, need more
        pos.set_hand_count(Color::White, HandPieceType::Rook, 1);
        pos.set_hand_count(Color::White, HandPieceType::Bishop, 1);
        pos.set_hand_count(Color::White, HandPieceType::Gold, 2);
        pos.set_hand_count(Color::White, HandPieceType::Silver, 2);
        pos.set_hand_count(Color::White, HandPieceType::Knight, 2);
        pos.set_hand_count(Color::White, HandPieceType::Lance, 2);
        // White board: 9, hand: 5+5+2+2+2+2 = 18, total = 27

        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let gs = GameState::from_position(pos, 500);

        // Verify piece counts in zone
        let black_in_zone = count_pieces_in_promotion_zone(&gs.position, Color::Black);
        assert!(black_in_zone >= 10, "Black needs 10+ pieces in zone, got {}", black_in_zone);
        let white_in_zone = count_pieces_in_promotion_zone(&gs.position, Color::White);
        assert!(white_in_zone >= 10, "White needs 10+ pieces in zone, got {}", white_in_zone);

        let result = check_impasse(&gs);
        assert!(result.is_some(), "Impasse should be triggered");
        // Both have >= 24, so it's a draw
        match result.unwrap() {
            GameResult::Impasse { winner } => {
                // Both >= 24 → draw
                assert_eq!(winner, None, "Both sides >= 24 should be a draw");
            }
            other => panic!("Expected Impasse, got {:?}", other),
        }
    }

    /// Test impasse where only one side reaches 24 points.
    #[test]
    fn test_impasse_one_sided_winner() {
        let mut pos = Position::empty();

        // Black king at (0,4) — in White's camp
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // White king at (8,4) — in Black's camp
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );

        // 10 Black pieces in rows 0-2 with high value
        for c in 0u8..9 {
            if c == 4 { continue; } // King is already there
            pos.set_piece(
                Square::from_row_col(0, c).unwrap(),
                Piece::new(PieceType::Gold, Color::Black, false),
            );
        }
        // Row 1: rook and bishop
        pos.set_piece(
            Square::from_row_col(1, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(1, 1).unwrap(),
            Piece::new(PieceType::Bishop, Color::Black, false),
        );
        // Black in zone: King + 8 Gold(1 each) + Rook(5) + Bishop(5) = 18
        // Plus hand: give 7 more → total 25
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 7);
        // Black total: 18 + 7 = 25 >= 24 ✓

        // 10 White pieces in rows 6-8 with LOW value (< 24)
        for c in 0u8..9 {
            if c == 4 { continue; }
            pos.set_piece(
                Square::from_row_col(8, c).unwrap(),
                Piece::new(PieceType::Pawn, Color::White, false),
            );
        }
        pos.set_piece(
            Square::from_row_col(7, 0).unwrap(),
            Piece::new(PieceType::Pawn, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(7, 1).unwrap(),
            Piece::new(PieceType::Pawn, Color::White, false),
        );
        // White in zone: 10 pawns = 10 < 24

        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();
        let gs = GameState::from_position(pos, 500);

        let result = check_impasse(&gs);
        assert!(result.is_some(), "Impasse should trigger");
        match result.unwrap() {
            GameResult::Impasse { winner: Some(Color::Black) } => {}
            other => panic!("Expected Impasse with Black winner, got {:?}", other),
        }
    }
}
