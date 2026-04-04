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
// Material balance (for training score head targets)
// ---------------------------------------------------------------------------

/// Material value of a piece for training score head targets.
/// Distinct from `piece_impasse_value()` which uses simplified counts for adjudication.
/// Standard computer Shogi values; promoted pieces use their promoted worth.
pub fn piece_value(pt: PieceType, promoted: bool) -> i32 {
    match (pt, promoted) {
        (PieceType::Pawn, false) => 1,
        (PieceType::Pawn, true) => 7,     // Tokin
        (PieceType::Lance, false) => 3,
        (PieceType::Lance, true) => 6,
        (PieceType::Knight, false) => 4,
        (PieceType::Knight, true) => 6,
        (PieceType::Silver, false) => 5,
        (PieceType::Silver, true) => 6,
        (PieceType::Gold, _) => 6,         // Gold cannot promote; defensive fallback
        (PieceType::Bishop, false) => 8,
        (PieceType::Bishop, true) => 10,   // Horse
        (PieceType::Rook, false) => 10,
        (PieceType::Rook, true) => 12,     // Dragon
        (PieceType::King, _) => 0,         // King excluded: never captured, adds same to both
    }
}

/// Compute material balance from `perspective`'s point of view.
/// Positive = perspective has more material. Counts board pieces + hand pieces.
///
/// Takes `&Position` (not `&GameState`), matching the `compute_impasse_score` pattern.
pub fn material_balance(pos: &Position, perspective: Color) -> i32 {
    let opponent = perspective.opponent();
    let mut balance: i32 = 0;

    // Board pieces
    for sq_idx in 0..Square::NUM_SQUARES {
        let sq = Square::new_unchecked(sq_idx as u8);
        if let Some(piece) = pos.piece_at(sq) {
            if piece.piece_type() == PieceType::King {
                continue;
            }
            let value = piece_value(piece.piece_type(), piece.is_promoted());
            if piece.color() == perspective {
                balance += value;
            } else {
                balance -= value;
            }
        }
    }

    // Hand pieces (never promoted)
    for &hpt in &HandPieceType::ALL {
        let pt = hpt.to_piece_type();
        let value = piece_value(pt, false);
        let own = pos.hand_count(perspective, hpt) as i32;
        let opp = pos.hand_count(opponent, hpt) as i32;
        balance += value * own;
        balance -= value * opp;
    }

    balance
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
    // piece_value — direct tests for all combinations
    // -----------------------------------------------------------------------

    /// All (PieceType, promoted) combinations return the expected values.
    #[test]
    fn test_piece_value_all_combinations() {
        // Unpromoted
        assert_eq!(piece_value(PieceType::Pawn, false), 1, "Pawn unpromoted");
        assert_eq!(piece_value(PieceType::Lance, false), 3, "Lance unpromoted");
        assert_eq!(piece_value(PieceType::Knight, false), 4, "Knight unpromoted");
        assert_eq!(piece_value(PieceType::Silver, false), 5, "Silver unpromoted");
        assert_eq!(piece_value(PieceType::Gold, false), 6, "Gold");
        assert_eq!(piece_value(PieceType::Bishop, false), 8, "Bishop unpromoted");
        assert_eq!(piece_value(PieceType::Rook, false), 10, "Rook unpromoted");
        assert_eq!(piece_value(PieceType::King, false), 0, "King unpromoted");

        // Promoted
        assert_eq!(piece_value(PieceType::Pawn, true), 7, "Tokin (promoted pawn)");
        assert_eq!(piece_value(PieceType::Lance, true), 6, "Promoted lance");
        assert_eq!(piece_value(PieceType::Knight, true), 6, "Promoted knight");
        assert_eq!(piece_value(PieceType::Silver, true), 6, "Promoted silver");
        // Gold cannot promote in standard Shogi; this arm is a defensive fallback,
        // intentionally equal to the unpromoted value.
        assert_eq!(piece_value(PieceType::Gold, true), 6, "Gold (promoted arm — defensive fallback)");
        assert_eq!(piece_value(PieceType::Bishop, true), 10, "Horse (promoted bishop)");
        assert_eq!(piece_value(PieceType::Rook, true), 12, "Dragon (promoted rook)");
        assert_eq!(piece_value(PieceType::King, true), 0, "King promoted arm");
    }

    /// Promoted pieces are worth strictly more than unpromoted for all promotable types.
    /// Gold is excluded because it cannot promote in standard Shogi.
    /// King is excluded because it has value 0 in both states.
    #[test]
    fn test_piece_value_promotion_increases_value() {
        for &pt in &[PieceType::Pawn, PieceType::Lance, PieceType::Knight,
                     PieceType::Silver, PieceType::Bishop, PieceType::Rook] {
            assert!(
                piece_value(pt, true) > piece_value(pt, false),
                "{:?}: promoted value should exceed unpromoted",
                pt
            );
        }
    }

    // -----------------------------------------------------------------------
    // material_balance — direct tests
    // -----------------------------------------------------------------------

    /// Starting position is perfectly symmetric: balance = 0 for both sides.
    #[test]
    fn test_material_balance_startpos_is_zero() {
        let pos = Position::startpos();
        let black_balance = material_balance(&pos, Color::Black);
        let white_balance = material_balance(&pos, Color::White);
        assert_eq!(black_balance, 0, "Black material balance at startpos should be 0");
        assert_eq!(white_balance, 0, "White material balance at startpos should be 0");
    }

    /// Perspective antisymmetry: balance(pos, Black) == -balance(pos, White).
    /// Tested on a genuinely asymmetric position (startpos trivially passes as 0 == -0).
    #[test]
    fn test_material_balance_perspective_negation() {
        // Use an asymmetric position so the property is non-trivially exercised
        let mut pos2 = Position::empty();
        pos2.set_piece(Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false));
        pos2.set_piece(Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false));
        // Black has an extra rook
        pos2.set_piece(Square::from_row_col(4, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false));
        pos2.hash = pos2.compute_hash();
        assert_eq!(
            material_balance(&pos2, Color::Black),
            -material_balance(&pos2, Color::White),
            "material_balance must negate for asymmetric position"
        );
    }

    /// Black advantage: Black has extra rook (+10), White has no compensating material.
    #[test]
    fn test_material_balance_black_has_extra_rook() {
        let mut pos = Position::empty();
        pos.set_piece(Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false));
        pos.set_piece(Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false));
        pos.set_piece(Square::from_row_col(4, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false));
        pos.hash = pos.compute_hash();

        let bal = material_balance(&pos, Color::Black);
        assert_eq!(bal, piece_value(PieceType::Rook, false),
            "Black with extra rook should have balance = rook value");
    }

    /// Hand pieces count toward material balance.
    #[test]
    fn test_material_balance_hand_pieces_counted() {
        let mut pos = Position::empty();
        pos.set_piece(Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false));
        pos.set_piece(Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false));
        // Give Black a gold in hand
        pos.set_hand_count(Color::Black, HandPieceType::Gold, 1);
        pos.hash = pos.compute_hash();

        let bal = material_balance(&pos, Color::Black);
        assert_eq!(bal, piece_value(PieceType::Gold, false),
            "Gold in hand should contribute its value to material balance");
    }

    /// Promoted pieces use promoted value, not base value.
    #[test]
    fn test_material_balance_promoted_piece_uses_promoted_value() {
        let mut pos = Position::empty();
        pos.set_piece(Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false));
        pos.set_piece(Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false));
        // Black has a Dragon (promoted rook) on board
        pos.set_piece(Square::from_row_col(4, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, true));
        pos.hash = pos.compute_hash();

        let bal = material_balance(&pos, Color::Black);
        assert_eq!(bal, piece_value(PieceType::Rook, true),
            "Dragon (promoted rook) should be valued at promoted rook value (12), not 10");
    }

    /// King is excluded from material balance.
    #[test]
    fn test_material_balance_king_excluded() {
        let mut pos = Position::empty();
        pos.set_piece(Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false));
        pos.set_piece(Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false));
        pos.hash = pos.compute_hash();

        // Only kings on board — balance must be 0 (kings excluded)
        assert_eq!(material_balance(&pos, Color::Black), 0,
            "Kings-only position should have balance 0");
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

    // -----------------------------------------------------------------------
    // Impasse winner cases (one side >= 24, other < 24)
    // -----------------------------------------------------------------------

    /// Helper: build a position where both kings have entered the opponent's camp
    /// and both sides have 10+ pieces in their promotion zone, with configurable
    /// piece counts so we can tune the score to any value.
    ///
    /// Black king at (0,4), White king at (8,4).
    /// Black gets `black_pawns` pawns in rows 0-2 plus its king (so black_count = 1 + black_pawns).
    /// White gets `white_pawns` pawns in rows 6-8 plus its king (so white_count = 1 + white_pawns).
    fn make_impasse_position(
        black_pawns: u8,
        black_hand_rooks: u8, // rooks in hand for Black (5 impasse pts each)
        white_pawns: u8,
        white_hand_rooks: u8, // rooks in hand for White (5 impasse pts each)
    ) -> Position {
        // 3 rows x 9 cols minus 1 king square = 26 available squares per side
        assert!(black_pawns <= 26, "Cannot place {} Black pawns (max 26)", black_pawns);
        assert!(white_pawns <= 26, "Cannot place {} White pawns (max 26)", white_pawns);

        let mut pos = Position::empty();
        // Kings already in opponent's camp
        pos.set_piece(Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false));
        pos.set_piece(Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false));

        // Black pawns scattered across rows 0-2 (avoiding col 4 where king is)
        let mut black_placed = 0u8;
        'outer_b: for r in 0u8..3 {
            for c in 0u8..9 {
                if r == 0 && c == 4 { continue; } // king's square
                if black_placed >= black_pawns { break 'outer_b; }
                pos.set_piece(Square::from_row_col(r, c).unwrap(),
                    Piece::new(PieceType::Pawn, Color::Black, false));
                black_placed += 1;
            }
        }
        assert_eq!(black_placed, black_pawns,
            "Failed to place all Black pawns: placed {}, wanted {}", black_placed, black_pawns);

        // White pawns scattered across rows 6-8 (avoiding col 4 where king is)
        let mut white_placed = 0u8;
        'outer_w: for r in 6u8..9 {
            for c in 0u8..9 {
                if r == 8 && c == 4 { continue; } // king's square
                if white_placed >= white_pawns { break 'outer_w; }
                pos.set_piece(Square::from_row_col(r, c).unwrap(),
                    Piece::new(PieceType::Pawn, Color::White, false));
                white_placed += 1;
            }
        }
        assert_eq!(white_placed, white_pawns,
            "Failed to place all White pawns: placed {}, wanted {}", white_placed, white_pawns);

        // Extra rooks in hand to tune the score
        pos.set_hand_count(Color::Black, HandPieceType::Rook, black_hand_rooks);
        pos.set_hand_count(Color::White, HandPieceType::Rook, white_hand_rooks);

        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();
        pos
    }

    /// Black >= 24, White < 24 → Black wins.
    ///
    /// Black: King + 9 pawns (board) = 9 pts + 3 rooks in hand (15 pts) = 24 pts.
    /// White: King + 9 pawns = 9 pts < 24. White count in zone = 10 (king+9 pawns).
    /// But wait — check_impasse requires both sides to have 10+ pieces in zone.
    /// King itself counts as 1, so 9 pawns + king = 10 for both. Good.
    #[test]
    fn test_check_impasse_black_wins() {
        // Black: 9 board pawns + 3 hand rooks = 9 + 15 = 24
        // White: 9 board pawns + 0 hand = 9 < 24
        let pos = make_impasse_position(9, 3, 9, 0);

        // Verify zone counts
        let black_in_zone = count_pieces_in_promotion_zone(&pos, Color::Black);
        let white_in_zone = count_pieces_in_promotion_zone(&pos, Color::White);
        assert!(black_in_zone >= 10,
            "Black needs 10+ pieces in zone, got {}", black_in_zone);
        assert!(white_in_zone >= 10,
            "White needs 10+ pieces in zone, got {}", white_in_zone);

        let black_score = compute_impasse_score(&pos, Color::Black);
        let white_score = compute_impasse_score(&pos, Color::White);
        assert!(black_score >= 24,
            "Black score should be >= 24, got {}", black_score);
        assert!(white_score < 24,
            "White score should be < 24, got {}", white_score);

        let gs = GameState::from_position(pos, 500);
        let result = check_impasse(&gs);
        assert!(result.is_some(), "Impasse should trigger");
        match result.unwrap() {
            GameResult::Impasse { winner: Some(Color::Black) } => {}
            other => panic!("Expected Impasse {{ winner: Some(Black) }}, got {:?}", other),
        }
    }

    /// White >= 24, Black < 24 → White wins.
    ///
    /// White: King + 9 pawns + 3 hand rooks = 9 + 15 = 24.
    /// Black: King + 9 pawns = 9 < 24.
    #[test]
    fn test_check_impasse_white_wins() {
        // Black: 9 board pawns + 0 hand = 9 < 24
        // White: 9 board pawns + 3 hand rooks = 9 + 15 = 24
        let pos = make_impasse_position(9, 0, 9, 3);

        // Verify zone counts (symmetric with Black-wins test)
        let black_in_zone = count_pieces_in_promotion_zone(&pos, Color::Black);
        let white_in_zone = count_pieces_in_promotion_zone(&pos, Color::White);
        assert!(black_in_zone >= 10,
            "Black needs 10+ pieces in zone, got {}", black_in_zone);
        assert!(white_in_zone >= 10,
            "White needs 10+ pieces in zone, got {}", white_in_zone);

        let black_score = compute_impasse_score(&pos, Color::Black);
        let white_score = compute_impasse_score(&pos, Color::White);
        assert!(white_score >= 24,
            "White score should be >= 24, got {}", white_score);
        assert!(black_score < 24,
            "Black score should be < 24, got {}", black_score);

        let gs = GameState::from_position(pos, 500);
        let result = check_impasse(&gs);
        assert!(result.is_some(), "Impasse should trigger");
        match result.unwrap() {
            GameResult::Impasse { winner: Some(Color::White) } => {}
            other => panic!("Expected Impasse {{ winner: Some(White) }}, got {:?}", other),
        }
    }

    /// Neither side reaches the 24-point score threshold → check_impasse returns
    /// None even though both kings have entered and both have 10+ pieces in zone.
    #[test]
    fn test_check_impasse_neither_reaches_score_threshold_returns_none() {
        // Both sides: 9 pawns = 9 pts each, well below 24
        let pos = make_impasse_position(9, 0, 9, 0);

        let black_score = compute_impasse_score(&pos, Color::Black);
        let white_score = compute_impasse_score(&pos, Color::White);
        assert!(black_score < 24, "Black score should be < 24, got {}", black_score);
        assert!(white_score < 24, "White score should be < 24, got {}", white_score);

        let gs = GameState::from_position(pos, 500);
        assert_eq!(check_impasse(&gs), None,
            "Neither side >= 24 should not trigger impasse");
    }

    // -----------------------------------------------------------------------
    // Uchi-fu-zume: pinned defender cannot capture the dropped pawn
    // -----------------------------------------------------------------------

    /// A piece that appears to capture the dropped pawn is pinned to its king.
    /// The drop should still be uchi-fu-zume since the pin prevents the capture.
    ///
    /// Position:
    ///   - White king at (0,0)
    ///   - White gold at (1,0) — could capture a pawn dropped at (1,0)... but see below
    ///   Wait — the gold is at (1,0) and the pawn would be dropped at (1,0)? No.
    ///   Let me set this up properly.
    ///
    ///   - White king at (0,0)
    ///   - White silver at (0,1) — pinned by Black rook at (0,8) along row 0
    ///   - Black drops pawn at (1,0), giving check to White king
    ///   - White silver at (0,1) attacks (1,0) via backward-left, but moving it
    ///     would expose the king to the rook on row 0 → pinned
    ///   - King can't go to (1,0) (occupied by pawn), (0,1) (own piece),
    ///     or (1,1) (must be covered by Black)
    ///   → Uchi-fu-zume
    #[test]
    fn test_uchi_fu_zume_pinned_defender() {
        use crate::types::HandPieceType;

        let mut pos = Position::empty();
        // White king cornered at (0,0)
        pos.set_piece(
            Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // White silver at (0,1) — pinned to king by Black rook along row 0
        // Silver attacks (1,0) normally (bwd_left for White = UP_LEFT = row-1,col-1
        // from (0,1)? No, for White silver: forward=DOWN.
        // bwd_left = UP_RIGHT = (row-1, col+1) from (0,1) = (-1,2) OOB.
        // bwd_right = UP_LEFT = (row-1, col-1) = (-1,0) OOB.
        // Hmm, White silver can't reach (1,0) from (0,1).
        //
        // Let me use a White gold at (0,1) instead.
        // White gold forward=DOWN. Directions: DOWN, DOWN_LEFT, DOWN_RIGHT, LEFT, RIGHT, UP.
        // From (0,1): DOWN=(1,1), DOWN_LEFT=(1,0)✓, DOWN_RIGHT=(1,2), LEFT=(0,0)=king,
        // RIGHT=(0,2), UP=(-1,1)=OOB.
        // Gold at (0,1) attacks (1,0) ✓. Gold is pinned by rook on row 0.
        pos.set_piece(
            Square::from_row_col(0, 1).unwrap(),
            Piece::new(PieceType::Gold, Color::White, false),
        );
        // Black rook at (0,8) — pins the gold along row 0
        pos.set_piece(
            Square::from_row_col(0, 8).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        // Black gold at (2, 1) — covers (1,1) so king can't escape there
        // Black gold forward=UP. From (2,1): UP=(1,1)✓.
        pos.set_piece(
            Square::from_row_col(2, 1).unwrap(),
            Piece::new(PieceType::Gold, Color::Black, false),
        );
        // Black lance at (8,0) — covers column 0, protects the dropped pawn
        pos.set_piece(
            Square::from_row_col(8, 0).unwrap(),
            Piece::new(PieceType::Lance, Color::Black, false),
        );
        // Black king far away
        pos.set_piece(
            Square::from_row_col(8, 8).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 1);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // Drop pawn at (1,0): attacks (0,0) = check on White king.
        // White gold at (0,1) attacks (1,0) but is pinned by rook on row 0.
        // King can't go to (0,1) — own gold. (1,0) — the pawn. (1,1) — covered by Black gold.
        let drop_sq = Square::from_row_col(1, 0).unwrap();
        assert!(
            is_uchi_fu_zume(&mut gs, drop_sq, Color::Black),
            "Pawn drop at (1,0) should be uchi-fu-zume: gold at (0,1) is pinned"
        );
    }

    /// White drops a pawn to deliver uchi-fu-zume (test with White as the dropper).
    #[test]
    fn test_uchi_fu_zume_white_as_dropper() {
        use crate::types::HandPieceType;

        let mut pos = Position::empty();
        // Black king cornered at (8,8)
        pos.set_piece(
            Square::from_row_col(8, 8).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // White king far away
        pos.set_piece(
            Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // White rook at (8,0) — covers row 8, attacks (8,7) blocking king escape
        pos.set_piece(
            Square::from_row_col(8, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        // White gold at (6,7) — covers (7,7) via forward (DOWN for White = toward row 8)
        // White gold forward=DOWN. From (6,7): DOWN=(7,7)✓, DOWN_LEFT=(7,6),
        // DOWN_RIGHT=(7,8)✓. Also covers (7,8).
        pos.set_piece(
            Square::from_row_col(6, 7).unwrap(),
            Piece::new(PieceType::Gold, Color::White, false),
        );
        // White lance at (0,8) — slides down column 8
        // Wait: White lance slides DOWN (forward for White). From (0,8) slides
        // through (1,8)...(7,8). (7,8) is covered by gold already.
        // The lance protects the dropped pawn at (7,8).
        pos.set_piece(
            Square::from_row_col(0, 8).unwrap(),
            Piece::new(PieceType::Lance, Color::White, false),
        );
        pos.set_hand_count(Color::White, HandPieceType::Pawn, 1);
        pos.current_player = Color::White;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // White pawn dropped at (7,8): White pawn attacks DOWN = (8,8) = Black king.
        // King escapes: (8,7) attacked by rook. (7,7) attacked by gold.
        // (7,8) = the pawn.
        let drop_sq = Square::from_row_col(7, 8).unwrap();
        assert!(
            is_uchi_fu_zume(&mut gs, drop_sq, Color::White),
            "White pawn drop at (7,8) should be uchi-fu-zume"
        );
    }

    /// King can directly capture the dropped pawn — NOT uchi-fu-zume.
    #[test]
    fn test_uchi_fu_zume_king_captures_pawn() {
        use crate::types::HandPieceType;

        let mut pos = Position::empty();
        // White king at (0,4)
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // Black king far away
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 1);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // Drop pawn at (1,4): gives check. King at (0,4) can capture at (1,4)
        // if (1,4) is not defended by other Black pieces.
        let drop_sq = Square::from_row_col(1, 4).unwrap();
        assert!(
            !is_uchi_fu_zume(&mut gs, drop_sq, Color::Black),
            "King can capture the dropped pawn — not uchi-fu-zume"
        );
    }

    // -----------------------------------------------------------------------
    // Perpetual check: White perpetually checks Black
    // -----------------------------------------------------------------------

    /// Symmetric perpetual check test: White checks Black repeatedly.
    /// This verifies winner attribution works correctly when roles are reversed.
    #[test]
    fn test_perpetual_check_white_checks_black() {
        use crate::types::Move;

        let mut pos = Position::empty();
        // Black king at (8,8)
        pos.set_piece(
            Square::from_row_col(8, 8).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // White king far away at (0,0)
        pos.set_piece(
            Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // White rook at (8,0) — gives check along row 8 to Black king
        pos.set_piece(
            Square::from_row_col(8, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        // Black is in check from the White rook.
        assert!(gs.is_in_check(), "Black king should be in check initially");

        // Cycle:
        // 1. Black king (8,8)→(7,8) — escapes check
        // 2. White rook (8,0)→(7,0) — gives check along row 7
        // 3. Black king (7,8)→(8,8) — escapes check
        // 4. White rook (7,0)→(8,0) — gives check along row 8 → back to start
        let bk_escape = Move::Board {
            from: Square::from_row_col(8, 8).unwrap(),
            to: Square::from_row_col(7, 8).unwrap(),
            promote: false,
        };
        let wr_chase_1 = Move::Board {
            from: Square::from_row_col(8, 0).unwrap(),
            to: Square::from_row_col(7, 0).unwrap(),
            promote: false,
        };
        let bk_return = Move::Board {
            from: Square::from_row_col(7, 8).unwrap(),
            to: Square::from_row_col(8, 8).unwrap(),
            promote: false,
        };
        let wr_chase_2 = Move::Board {
            from: Square::from_row_col(7, 0).unwrap(),
            to: Square::from_row_col(8, 0).unwrap(),
            promote: false,
        };

        for _ in 0..3 {
            gs.make_move(bk_escape);
            gs.make_move(wr_chase_1);
            gs.make_move(bk_return);
            gs.make_move(wr_chase_2);
        }

        let result = check_sennichite(&gs);
        assert!(result.is_some(), "Should detect perpetual check (White checking Black)");
        match result.unwrap() {
            GameResult::PerpetualCheck { winner } => {
                // White is the checker; Black (the victim) wins.
                assert_eq!(
                    winner,
                    Color::Black,
                    "Black (the victim of White's perpetual check) should win"
                );
            }
            other => panic!("Expected PerpetualCheck, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Impasse scoring: promoted pawn (Tokin) and promoted bishop (Horse)
    // -----------------------------------------------------------------------

    #[test]
    fn test_impasse_score_promoted_pawn_worth_1() {
        let mut pos = Position::empty();
        // Promoted pawn (Tokin) — worth 1 point (base type is Pawn)
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Pawn, Color::Black, true),
        );
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        let score = compute_impasse_score(&pos, Color::Black);
        assert_eq!(score, 1, "Promoted pawn (Tokin) should be worth 1 point, not 5");
    }

    #[test]
    fn test_impasse_score_promoted_bishop_worth_5() {
        let mut pos = Position::empty();
        // Promoted bishop (Horse) — worth 5 points (base type is Bishop)
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Bishop, Color::Black, true),
        );
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        let score = compute_impasse_score(&pos, Color::Black);
        assert_eq!(score, 5, "Promoted bishop (Horse) should be worth 5 points");
    }

    #[test]
    fn test_impasse_score_mixed_promoted_and_unpromoted() {
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        // Unpromoted rook: 5
        pos.set_piece(
            Square::from_row_col(4, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::Black, false),
        );
        // Promoted bishop (Horse): 5
        pos.set_piece(
            Square::from_row_col(4, 1).unwrap(),
            Piece::new(PieceType::Bishop, Color::Black, true),
        );
        // Promoted pawn (Tokin): 1
        pos.set_piece(
            Square::from_row_col(4, 2).unwrap(),
            Piece::new(PieceType::Pawn, Color::Black, true),
        );
        // Unpromoted gold: 1
        pos.set_piece(
            Square::from_row_col(4, 3).unwrap(),
            Piece::new(PieceType::Gold, Color::Black, false),
        );
        // Silver in hand: 1
        pos.set_hand_count(Color::Black, HandPieceType::Silver, 1);

        let score = compute_impasse_score(&pos, Color::Black);
        assert_eq!(score, 13, "Mixed score: Rook(5) + Horse(5) + Tokin(1) + Gold(1) + Silver-hand(1) = 13");
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

    /// Asymmetric impasse: White reaches 24 but Black doesn't.
    #[test]
    fn test_impasse_one_sided_white_wins() {
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

        // 10 Black pieces in zone with LOW value
        for c in 0u8..9 {
            if c == 4 { continue; }
            pos.set_piece(
                Square::from_row_col(0, c).unwrap(),
                Piece::new(PieceType::Pawn, Color::Black, false),
            );
        }
        pos.set_piece(
            Square::from_row_col(1, 0).unwrap(),
            Piece::new(PieceType::Pawn, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(1, 1).unwrap(),
            Piece::new(PieceType::Pawn, Color::Black, false),
        );
        // Black: 10 pawns in zone = 10 points < 24

        // 10 White pieces in zone with HIGH value
        for c in 0u8..9 {
            if c == 4 { continue; }
            pos.set_piece(
                Square::from_row_col(8, c).unwrap(),
                Piece::new(PieceType::Gold, Color::White, false),
            );
        }
        pos.set_piece(
            Square::from_row_col(7, 0).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(7, 1).unwrap(),
            Piece::new(PieceType::Bishop, Color::White, false),
        );
        // White in zone: King + 8 Gold(1) + Rook(5) + Bishop(5) = 18
        pos.set_hand_count(Color::White, HandPieceType::Pawn, 7);
        // White total: 18 + 7 = 25 >= 24 ✓

        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();
        let gs = GameState::from_position(pos, 500);

        let result = check_impasse(&gs);
        assert!(result.is_some(), "Impasse should trigger");
        match result.unwrap() {
            GameResult::Impasse { winner: Some(Color::White) } => {}
            other => panic!("Expected Impasse with White winner, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // piece_attacks_square: coverage for all piece types
    // -----------------------------------------------------------------------

    #[test]
    fn test_piece_attacks_square_knight() {
        let pos = Position::empty();
        let sq = Square::from_row_col(4, 4).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::Black, false);
        // Black knight at (4,4) should attack (2,3) and (2,5)
        let target1 = Square::from_row_col(2, 3).unwrap();
        let target2 = Square::from_row_col(2, 5).unwrap();
        assert!(piece_attacks_square(&pos, sq, knight, target1), "Knight should attack (2,3)");
        assert!(piece_attacks_square(&pos, sq, knight, target2), "Knight should attack (2,5)");
        // Should NOT attack (3,4)
        let non_target = Square::from_row_col(3, 4).unwrap();
        assert!(!piece_attacks_square(&pos, sq, knight, non_target));
    }

    #[test]
    fn test_piece_attacks_square_white_knight() {
        let pos = Position::empty();
        let sq = Square::from_row_col(4, 4).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::White, false);
        // White knight at (4,4) jumps DOWN: targets (6,3) and (6,5)
        let target1 = Square::from_row_col(6, 3).unwrap();
        let target2 = Square::from_row_col(6, 5).unwrap();
        assert!(piece_attacks_square(&pos, sq, knight, target1));
        assert!(piece_attacks_square(&pos, sq, knight, target2));
    }

    #[test]
    fn test_piece_attacks_square_lance() {
        let mut pos = Position::empty();
        let sq = Square::from_row_col(6, 4).unwrap();
        let lance = Piece::new(PieceType::Lance, Color::Black, false);
        // Black lance slides UP from (6,4) — should attack (5,4), (4,4), etc.
        let target = Square::from_row_col(3, 4).unwrap();
        assert!(piece_attacks_square(&pos, sq, lance, target));

        // Place blocker at (4,4) — should NOT attack (3,4) anymore
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Pawn, Color::White, false),
        );
        assert!(!piece_attacks_square(&pos, sq, lance, target), "Lance should be blocked");
        // But should still attack (4,4) — the blocker itself
        let blocker_sq = Square::from_row_col(4, 4).unwrap();
        assert!(piece_attacks_square(&pos, sq, lance, blocker_sq));
    }

    #[test]
    fn test_piece_attacks_square_silver() {
        let pos = Position::empty();
        let sq = Square::from_row_col(4, 4).unwrap();
        let silver = Piece::new(PieceType::Silver, Color::Black, false);
        // Black silver attacks: forward(3,4), fwd-left(3,3), fwd-right(3,5),
        // back-left(5,3), back-right(5,5)
        assert!(piece_attacks_square(&pos, sq, silver, Square::from_row_col(3, 4).unwrap()));
        assert!(piece_attacks_square(&pos, sq, silver, Square::from_row_col(3, 3).unwrap()));
        assert!(piece_attacks_square(&pos, sq, silver, Square::from_row_col(3, 5).unwrap()));
        assert!(piece_attacks_square(&pos, sq, silver, Square::from_row_col(5, 3).unwrap()));
        assert!(piece_attacks_square(&pos, sq, silver, Square::from_row_col(5, 5).unwrap()));
        // Should NOT attack directly left/right/backward-center
        assert!(!piece_attacks_square(&pos, sq, silver, Square::from_row_col(4, 3).unwrap()));
        assert!(!piece_attacks_square(&pos, sq, silver, Square::from_row_col(5, 4).unwrap()));
    }

    #[test]
    fn test_piece_attacks_square_gold() {
        let pos = Position::empty();
        let sq = Square::from_row_col(4, 4).unwrap();
        let gold = Piece::new(PieceType::Gold, Color::Black, false);
        // Black gold attacks: forward(3,4), fwd-left(3,3), fwd-right(3,5),
        // left(4,3), right(4,5), backward(5,4)
        assert!(piece_attacks_square(&pos, sq, gold, Square::from_row_col(3, 4).unwrap()));
        assert!(piece_attacks_square(&pos, sq, gold, Square::from_row_col(3, 3).unwrap()));
        assert!(piece_attacks_square(&pos, sq, gold, Square::from_row_col(4, 3).unwrap()));
        assert!(piece_attacks_square(&pos, sq, gold, Square::from_row_col(5, 4).unwrap()));
        // Should NOT attack diagonally backward
        assert!(!piece_attacks_square(&pos, sq, gold, Square::from_row_col(5, 3).unwrap()));
        assert!(!piece_attacks_square(&pos, sq, gold, Square::from_row_col(5, 5).unwrap()));
    }

    #[test]
    fn test_piece_attacks_square_bishop_diagonal() {
        let pos = Position::empty();
        let sq = Square::from_row_col(4, 4).unwrap();
        let bishop = Piece::new(PieceType::Bishop, Color::Black, false);
        // Bishop slides diagonally
        assert!(piece_attacks_square(&pos, sq, bishop, Square::from_row_col(2, 2).unwrap()));
        assert!(piece_attacks_square(&pos, sq, bishop, Square::from_row_col(6, 6).unwrap()));
        // Should NOT attack orthogonally
        assert!(!piece_attacks_square(&pos, sq, bishop, Square::from_row_col(4, 6).unwrap()));
    }

    #[test]
    fn test_piece_attacks_square_promoted_rook() {
        // Dragon (promoted rook) = rook slides + king-like diagonal steps
        let pos = Position::empty();
        let sq = Square::from_row_col(4, 4).unwrap();
        let dragon = Piece::new(PieceType::Rook, Color::Black, true);
        // Rook slides
        assert!(piece_attacks_square(&pos, sq, dragon, Square::from_row_col(4, 8).unwrap()));
        assert!(piece_attacks_square(&pos, sq, dragon, Square::from_row_col(0, 4).unwrap()));
        // Diagonal steps (king-like)
        assert!(piece_attacks_square(&pos, sq, dragon, Square::from_row_col(3, 3).unwrap()));
        assert!(piece_attacks_square(&pos, sq, dragon, Square::from_row_col(5, 5).unwrap()));
        // NOT diagonal slide (2 squares diagonal)
        assert!(!piece_attacks_square(&pos, sq, dragon, Square::from_row_col(2, 2).unwrap()));
    }

    #[test]
    fn test_piece_attacks_square_promoted_bishop() {
        // Horse (promoted bishop) = bishop slides + king-like orthogonal steps
        let pos = Position::empty();
        let sq = Square::from_row_col(4, 4).unwrap();
        let horse = Piece::new(PieceType::Bishop, Color::Black, true);
        // Bishop slides
        assert!(piece_attacks_square(&pos, sq, horse, Square::from_row_col(2, 2).unwrap()));
        // Orthogonal steps (king-like)
        assert!(piece_attacks_square(&pos, sq, horse, Square::from_row_col(3, 4).unwrap()));
        assert!(piece_attacks_square(&pos, sq, horse, Square::from_row_col(4, 5).unwrap()));
        // NOT orthogonal slide (2 squares)
        assert!(!piece_attacks_square(&pos, sq, horse, Square::from_row_col(2, 4).unwrap()));
    }

    // -----------------------------------------------------------------------
    // Impasse: one king entered, one not — should NOT trigger
    // -----------------------------------------------------------------------

    /// Impasse requires BOTH kings to have entered the opponent's camp.
    /// Here only Black's king has entered; White's king stays home.
    #[test]
    fn test_check_impasse_only_one_king_entered_returns_none() {
        let mut pos = Position::empty();
        // Black king entered White's camp (row 0)
        pos.set_piece(Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false));
        // White king has NOT entered — still in own territory (row 0 = White's home)
        pos.set_piece(Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::King, Color::White, false));

        // Give Black 9 pawns in zone + 3 hand rooks = 24 pts (would qualify)
        for c in [0u8, 1, 2, 3, 5, 6, 7, 8] {
            pos.set_piece(Square::from_row_col(1, c).unwrap(),
                Piece::new(PieceType::Pawn, Color::Black, false));
        }
        pos.set_piece(Square::from_row_col(2, 0).unwrap(),
            Piece::new(PieceType::Pawn, Color::Black, false));
        pos.set_hand_count(Color::Black, HandPieceType::Rook, 3);

        // White has no pieces in Black's camp at all
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let gs = GameState::from_position(pos, 500);
        assert_eq!(check_impasse(&gs), None,
            "Impasse must not trigger when only one king has entered opponent's camp");
    }
}
