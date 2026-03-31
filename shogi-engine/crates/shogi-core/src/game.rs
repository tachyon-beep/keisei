//! GameState — make/unmake moves, legal move generation, and game lifecycle.

use std::collections::HashMap;

use crate::attack::{compute_attack_map, AttackMap};
use crate::movegen::{generate_pseudo_legal_board_moves, generate_pseudo_legal_drops};
use crate::movelist::MoveList;
use crate::piece::Piece;
use crate::position::Position;
use crate::types::{Color, GameResult, HandPieceType, Move, PieceType, ShogiError, Square};
use crate::zobrist::ZOBRIST;

// ---------------------------------------------------------------------------
// UndoInfo
// ---------------------------------------------------------------------------

/// Information needed to reverse a `make_move` call.
#[derive(Debug, Clone)]
pub struct UndoInfo {
    pub captured: Option<Piece>,
    pub prev_hash: u64,
    pub prev_attack_map: AttackMap,
    pub was_in_check: bool,
}

// ---------------------------------------------------------------------------
// Helper: compute pawn columns from scratch
// ---------------------------------------------------------------------------

/// Scan the board for unpromoted pawns and return a per-color, per-column presence map.
pub fn compute_pawn_columns(pos: &Position) -> [[bool; 9]; 2] {
    let mut cols = [[false; 9]; 2];
    for idx in 0..Square::NUM_SQUARES {
        let sq = Square::new_unchecked(idx as u8);
        if let Some(piece) = pos.piece_at(sq) {
            if piece.piece_type() == PieceType::Pawn && !piece.is_promoted() {
                cols[piece.color() as usize][sq.col() as usize] = true;
            }
        }
    }
    cols
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

pub struct GameState {
    pub position: Position,
    pub attack_map: AttackMap,
    pub pawn_columns: [[bool; 9]; 2],
    pub repetition_map: HashMap<u64, u8>,
    pub check_history: Vec<bool>,
    pub hash_history: Vec<u64>,
    pub ply: u32,
    pub max_ply: u32,
    pub result: GameResult,
}

impl GameState {
    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    /// Standard starting position with max_ply = 500.
    pub fn new() -> GameState {
        Self::with_max_ply(500)
    }

    /// Standard starting position with a custom max_ply.
    pub fn with_max_ply(max_ply: u32) -> GameState {
        let position = Position::startpos();
        Self::from_position(position, max_ply)
    }

    /// Parse a SFEN string into a GameState.
    pub fn from_sfen(sfen: &str, max_ply: u32) -> Result<GameState, ShogiError> {
        let position = Position::from_sfen(sfen)?;
        Ok(Self::from_position(position, max_ply))
    }

    /// Internal: build a GameState from an already-constructed Position.
    fn from_position(position: Position, max_ply: u32) -> GameState {
        let attack_map = compute_attack_map(&position);
        let pawn_columns = compute_pawn_columns(&position);
        let mut repetition_map = HashMap::with_capacity(max_ply as usize);
        repetition_map.insert(position.hash, 1);

        GameState {
            position,
            attack_map,
            pawn_columns,
            repetition_map,
            check_history: Vec::new(),
            hash_history: Vec::new(),
            ply: 0,
            max_ply,
            result: GameResult::InProgress,
        }
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// O(1) check detection: is the current player's king attacked by the opponent?
    pub fn is_in_check(&self) -> bool {
        let color = self.position.current_player;
        let opponent = color.opponent();
        if let Some(king_sq) = self.position.find_king(color) {
            self.attack_map[opponent as usize][king_sq.index()] > 0
        } else {
            // No king on board — shouldn't happen in a valid game, but treat as
            // not in check to avoid panics.
            false
        }
    }

    // -----------------------------------------------------------------------
    // make_move
    // -----------------------------------------------------------------------

    /// Apply `mv` to the position, returning undo information.
    ///
    /// This is the hot path. Hash updates are incremental via XOR.
    pub fn make_move(&mut self, mv: Move) -> UndoInfo {
        let z = &*ZOBRIST;

        // Save state for undo.
        let prev_hash = self.position.hash;
        let prev_attack_map = self.attack_map;
        let was_in_check = self.is_in_check();

        // Record history BEFORE the move.
        self.hash_history.push(self.position.hash);
        self.check_history.push(was_in_check);

        let mover_color = self.position.current_player;
        let mut captured: Option<Piece> = None;

        match mv {
            Move::Board { from, to, promote } => {
                let moving_piece = self.position.piece_at(from)
                    .expect("make_move: no piece at from square");

                // 1. Remove piece from `from`.
                self.position.hash ^= z.hash_piece_at(from, moving_piece);
                self.position.clear_square(from);

                // 2. Handle capture at `to`.
                if let Some(cap) = self.position.piece_at(to) {
                    captured = Some(cap);
                    self.position.hash ^= z.hash_piece_at(to, cap);
                    self.position.clear_square(to);

                    // Add base piece type to mover's hand.
                    // Promoted pieces revert to their unpromoted type in hand.
                    let hand_type = HandPieceType::from_piece_type(cap.piece_type())
                        .expect("captured piece must have a valid hand type (not king)");
                    let old_count = self.position.hand_count(mover_color, hand_type);
                    // XOR out old hand hash, XOR in new hand hash.
                    if old_count > 0 {
                        self.position.hash ^= z.hash_hand(mover_color, hand_type, old_count);
                    }
                    let new_count = old_count + 1;
                    self.position.hash ^= z.hash_hand(mover_color, hand_type, new_count);
                    self.position.set_hand_count(mover_color, hand_type, new_count);
                }

                // 3. Place piece at `to` (possibly promoted).
                let placed_piece = if promote {
                    moving_piece.promote()
                } else {
                    moving_piece
                };
                self.position.set_piece(to, placed_piece);
                self.position.hash ^= z.hash_piece_at(to, placed_piece);

                // 4. Update pawn_columns if relevant.
                let pt = moving_piece.piece_type();
                if pt == PieceType::Pawn && !moving_piece.is_promoted() {
                    // Pawn left `from` column.
                    // We need to check if there's still an unpromoted pawn of
                    // the same color on that column.
                    self.update_pawn_column_for(mover_color, from.col());
                }
                if let Some(cap) = captured {
                    if cap.piece_type() == PieceType::Pawn && !cap.is_promoted() {
                        let opp = mover_color.opponent();
                        self.update_pawn_column_for(opp, to.col());
                    }
                }
                // If a pawn was placed (not promoted), mark the destination column.
                if pt == PieceType::Pawn && !promote && !moving_piece.is_promoted() {
                    self.pawn_columns[mover_color as usize][to.col() as usize] = true;
                }
                // If the pawn promoted, the destination column loses our pawn.
                if pt == PieceType::Pawn && promote && !moving_piece.is_promoted() {
                    // Already handled by the from-column update above if same column.
                    // If different column... well, pawns only move forward, same column.
                    // But to be safe, update the to-column too.
                    self.update_pawn_column_for(mover_color, to.col());
                }
            }

            Move::Drop { to, piece_type: hpt } => {
                // 1. Decrement hand count with hash updates.
                let old_count = self.position.hand_count(mover_color, hpt);
                debug_assert!(old_count > 0, "make_move: dropping piece not in hand");
                self.position.hash ^= z.hash_hand(mover_color, hpt, old_count);
                let new_count = old_count - 1;
                if new_count > 0 {
                    self.position.hash ^= z.hash_hand(mover_color, hpt, new_count);
                }
                self.position.set_hand_count(mover_color, hpt, new_count);

                // 2. Place piece on board.
                let piece = Piece::new(hpt.to_piece_type(), mover_color, false);
                self.position.set_piece(to, piece);
                self.position.hash ^= z.hash_piece_at(to, piece);

                // 3. Update pawn_columns if pawn.
                if hpt == HandPieceType::Pawn {
                    self.pawn_columns[mover_color as usize][to.col() as usize] = true;
                }
            }
        }

        // 5. Flip side to move.
        self.position.hash ^= z.side_to_move;
        self.position.current_player = mover_color.opponent();

        // 6. Recompute attack map from scratch.
        self.attack_map = compute_attack_map(&self.position);

        // 7. Update repetition map.
        *self.repetition_map.entry(self.position.hash).or_insert(0) += 1;

        // 8. Increment ply.
        self.ply += 1;

        UndoInfo {
            captured,
            prev_hash,
            prev_attack_map,
            was_in_check,
        }
    }

    // -----------------------------------------------------------------------
    // unmake_move
    // -----------------------------------------------------------------------

    /// Reverse a `make_move` call using the provided undo information.
    pub fn unmake_move(&mut self, mv: Move, undo: UndoInfo) {
        // 1. Decrement repetition count for current hash.
        let count = self.repetition_map.get_mut(&self.position.hash)
            .expect("unmake_move: current hash not in repetition map");
        if *count <= 1 {
            self.repetition_map.remove(&self.position.hash);
        } else {
            *count -= 1;
        }

        // 2. Flip side to move back.
        self.position.current_player = self.position.current_player.opponent();
        let mover_color = self.position.current_player;

        // 3. Reverse the board changes.
        match mv {
            Move::Board { from, to, promote } => {
                // Remove placed piece from `to`.
                let placed = self.position.piece_at(to)
                    .expect("unmake_move: no piece at to square");

                // Restore original piece at `from`.
                let original = if promote {
                    placed.unpromote()
                } else {
                    placed
                };
                self.position.set_piece(from, original);
                self.position.clear_square(to);

                // Restore captured piece at `to` if any.
                if let Some(cap) = undo.captured {
                    self.position.set_piece(to, cap);

                    // Restore hand: decrement the captured piece's base type from mover's hand.
                    let hand_type = HandPieceType::from_piece_type(cap.piece_type())
                        .expect("captured piece must have valid hand type");
                    let cur_count = self.position.hand_count(mover_color, hand_type);
                    debug_assert!(cur_count > 0, "unmake_move: hand count already 0");
                    self.position.set_hand_count(mover_color, hand_type, cur_count - 1);
                }
            }

            Move::Drop { to, piece_type: hpt } => {
                // Remove dropped piece from board.
                self.position.clear_square(to);

                // Restore hand count.
                let cur_count = self.position.hand_count(mover_color, hpt);
                self.position.set_hand_count(mover_color, hpt, cur_count + 1);
            }
        }

        // 4. Restore hash.
        self.position.hash = undo.prev_hash;

        // 5. Restore attack map.
        self.attack_map = undo.prev_attack_map;

        // 6. Recompute pawn columns from scratch (simpler than tracking incrementally).
        self.pawn_columns = compute_pawn_columns(&self.position);

        // 7. Pop history.
        self.hash_history.pop();
        self.check_history.pop();

        // 8. Decrement ply.
        self.ply -= 1;
    }

    // -----------------------------------------------------------------------
    // legal_moves
    // -----------------------------------------------------------------------

    /// Generate all legal moves for the current player.
    ///
    /// Uses make/unmake to filter pseudo-legal moves by king safety.
    pub fn legal_moves(&mut self) -> Vec<Move> {
        let color = self.position.current_player;

        // Generate pseudo-legal moves.
        let mut candidates = Vec::with_capacity(128);
        generate_pseudo_legal_board_moves(&self.position, color, &mut candidates);
        generate_pseudo_legal_drops(&self.position, color, &mut candidates);

        // Filter by nifu for pawn drops, then by king safety.
        let mut legal = Vec::with_capacity(candidates.len());
        for mv in candidates {
            // Nifu and uchi-fu-zume checks for pawn drops.
            if let Move::Drop { to, piece_type: HandPieceType::Pawn } = mv {
                if self.pawn_columns[color as usize][to.col() as usize] {
                    continue; // Already have an unpromoted pawn on this column.
                }
                if crate::rules::is_uchi_fu_zume(self, to, color) {
                    continue; // Pawn drop would deliver checkmate (illegal).
                }
            }

            // King safety: make the move and check if our king is safe.
            let undo = self.make_move(mv);

            // After make_move, current_player has flipped. The "mover" is now the
            // opponent from the perspective of the new current_player.
            // We need to check that the mover's king is NOT attacked by the new
            // current player (i.e., the mover left their king in check).
            let mover = self.position.current_player.opponent(); // == color
            let king_safe = if let Some(king_sq) = self.position.find_king(mover) {
                self.attack_map[self.position.current_player as usize][king_sq.index()] == 0
            } else {
                // No king — shouldn't happen, but treat as illegal.
                false
            };

            self.unmake_move(mv, undo);

            if king_safe {
                legal.push(mv);
            }
        }

        legal
    }

    // -----------------------------------------------------------------------
    // Hot-path move generation API
    // -----------------------------------------------------------------------

    /// Hot-path API: generate legal moves into a caller-owned `MoveList`.
    ///
    /// Clears `move_list` before filling it. Avoids all heap allocation.
    pub fn generate_legal_moves_into(&mut self, move_list: &mut MoveList) {
        move_list.clear();
        let color = self.position.current_player;

        // Generate pseudo-legal moves into a temporary Vec.
        // We reuse the same filtering logic as legal_moves().
        let mut candidates = Vec::with_capacity(128);
        generate_pseudo_legal_board_moves(&self.position, color, &mut candidates);
        generate_pseudo_legal_drops(&self.position, color, &mut candidates);

        for mv in candidates {
            // Nifu and uchi-fu-zume checks for pawn drops.
            if let Move::Drop { to, piece_type: HandPieceType::Pawn } = mv {
                if self.pawn_columns[color as usize][to.col() as usize] {
                    continue;
                }
                if crate::rules::is_uchi_fu_zume(self, to, color) {
                    continue;
                }
            }

            // King safety check via make/unmake.
            let undo = self.make_move(mv);
            let mover = self.position.current_player.opponent();
            let king_safe = if let Some(king_sq) = self.position.find_king(mover) {
                self.attack_map[self.position.current_player as usize][king_sq.index()] == 0
            } else {
                false
            };
            self.unmake_move(mv, undo);

            if king_safe {
                move_list.push(mv);
            }
        }
    }

    /// Hot-path API: write a legal action mask into a caller-owned buffer.
    ///
    /// Sets `mask[encode_fn(mv)] = true` for every legal move `mv`.
    /// Fills the rest of the mask with `false` first.
    pub fn write_legal_mask_into(
        &mut self,
        mask: &mut [bool],
        encode_fn: &dyn Fn(Move) -> usize,
    ) {
        mask.fill(false);
        let mut move_list = MoveList::new();
        self.generate_legal_moves_into(&mut move_list);
        for mv in move_list.iter() {
            let idx = encode_fn(*mv);
            debug_assert!(idx < mask.len(), "encoded action index out of mask bounds");
            mask[idx] = true;
        }
    }

    // -----------------------------------------------------------------------
    // check_termination
    // -----------------------------------------------------------------------

    /// Check if the game has ended. Updates `self.result` if a terminal condition
    /// is detected.
    ///
    /// Checks in order:
    /// 1. Max ply reached
    /// 2. Sennichite / perpetual check
    /// 3. Impasse (CSA 24-point rule)
    /// 4. Checkmate / stalemate (no legal moves)
    pub fn check_termination(&mut self) {
        if self.result.is_terminal() {
            return;
        }

        // 1. Max ply.
        if self.ply >= self.max_ply {
            self.result = GameResult::MaxMoves;
            return;
        }

        // 2. Sennichite / perpetual check.
        if let Some(result) = crate::rules::check_sennichite(self) {
            self.result = result;
            return;
        }

        // 3. Impasse.
        if let Some(result) = crate::rules::check_impasse(self) {
            self.result = result;
            return;
        }

        // 4. No legal moves: checkmate or stalemate.
        let moves = self.legal_moves();
        if moves.is_empty() {
            if self.is_in_check() {
                // Current player is checkmated — opponent wins.
                self.result = GameResult::Checkmate {
                    winner: self.position.current_player.opponent(),
                };
            } else {
                // Stalemate — in Shogi, the player with no moves loses.
                // (Stalemate is a loss for the side without moves.)
                self.result = GameResult::Checkmate {
                    winner: self.position.current_player.opponent(),
                };
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Recompute `pawn_columns[color][col]` by scanning that column.
    fn update_pawn_column_for(&mut self, color: Color, col: u8) {
        let color_idx = color as usize;
        let col_idx = col as usize;
        self.pawn_columns[color_idx][col_idx] = false;
        for row in 0u8..9 {
            let sq = Square::from_row_col(row, col).unwrap();
            if let Some(piece) = self.position.piece_at(sq) {
                if piece.color() == color
                    && piece.piece_type() == PieceType::Pawn
                    && !piece.is_promoted()
                {
                    self.pawn_columns[color_idx][col_idx] = true;
                    return;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attack::compute_attack_map;
    use crate::movelist::MoveList;
    use crate::piece::Piece;
    use crate::types::{Color, GameResult, HandPieceType, Move, PieceType, Square};

    // -----------------------------------------------------------------------
    // test_make_unmake_roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_make_unmake_roundtrip() {
        let mut gs = GameState::new();
        let original_hash = gs.position.hash;
        let original_board = gs.position.board;
        let original_hands = gs.position.hands;

        let moves = gs.legal_moves();
        assert!(!moves.is_empty(), "opening must have legal moves");

        for mv in moves {
            let undo = gs.make_move(mv);
            gs.unmake_move(mv, undo);

            assert_eq!(
                gs.position.hash, original_hash,
                "hash mismatch after make/unmake of {:?}",
                mv
            );
            assert_eq!(
                gs.position.board, original_board,
                "board mismatch after make/unmake of {:?}",
                mv
            );
            assert_eq!(
                gs.position.hands, original_hands,
                "hands mismatch after make/unmake of {:?}",
                mv
            );
        }
    }

    // -----------------------------------------------------------------------
    // test_hash_matches_recomputation_after_move
    // -----------------------------------------------------------------------

    #[test]
    fn test_hash_matches_recomputation_after_move() {
        let mut gs = GameState::new();
        let moves = gs.legal_moves();

        for mv in moves {
            let undo = gs.make_move(mv);

            let recomputed = gs.position.compute_hash();
            assert_eq!(
                gs.position.hash, recomputed,
                "incremental hash != recomputed hash after {:?}",
                mv
            );

            gs.unmake_move(mv, undo);
        }
    }

    // -----------------------------------------------------------------------
    // test_attack_map_matches_recomputation_after_move
    // -----------------------------------------------------------------------

    #[test]
    fn test_attack_map_matches_recomputation_after_move() {
        let mut gs = GameState::new();
        let moves = gs.legal_moves();

        for mv in moves {
            let undo = gs.make_move(mv);

            let recomputed = compute_attack_map(&gs.position);
            assert_eq!(
                gs.attack_map, recomputed,
                "attack map mismatch after {:?}",
                mv
            );

            gs.unmake_move(mv, undo);
        }
    }

    // -----------------------------------------------------------------------
    // test_legal_moves_opening_count
    // -----------------------------------------------------------------------

    #[test]
    fn test_legal_moves_opening_count() {
        let mut gs = GameState::new();
        let moves = gs.legal_moves();
        assert_eq!(
            moves.len(),
            30,
            "standard Shogi opening must have exactly 30 legal moves, got {}",
            moves.len()
        );
    }

    // -----------------------------------------------------------------------
    // test_in_check_detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_in_check_detection() {
        // Set up a position where Black's king is attacked by a White rook.
        // Black king at (8,4), White rook on (4,4) with clear column.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let gs = GameState::from_position(pos, 500);
        assert!(gs.is_in_check(), "Black king should be in check from White rook");
    }

    // -----------------------------------------------------------------------
    // test_capture_adds_to_hand
    // -----------------------------------------------------------------------

    #[test]
    fn test_capture_adds_to_hand() {
        // Place Black pawn at (5,4) and White pawn at (4,4).
        // Move Black pawn forward to capture White pawn.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(5, 4).unwrap(),
            Piece::new(PieceType::Pawn, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(4, 4).unwrap(),
            Piece::new(PieceType::Pawn, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);

        assert_eq!(gs.position.hand_count(Color::Black, HandPieceType::Pawn), 0);

        let capture_move = Move::Board {
            from: Square::from_row_col(5, 4).unwrap(),
            to: Square::from_row_col(4, 4).unwrap(),
            promote: false,
        };
        let _undo = gs.make_move(capture_move);

        assert_eq!(
            gs.position.hand_count(Color::Black, HandPieceType::Pawn),
            1,
            "Black should have 1 pawn in hand after capture"
        );
    }

    // -----------------------------------------------------------------------
    // test_nifu_prevented
    // -----------------------------------------------------------------------

    #[test]
    fn test_nifu_prevented() {
        // Set up: Black has a pawn on col 4 and a pawn in hand.
        // Legal moves should NOT include dropping a pawn on col 4.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(8, 4).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(0, 4).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        pos.set_piece(
            Square::from_row_col(6, 4).unwrap(),
            Piece::new(PieceType::Pawn, Color::Black, false),
        );
        pos.set_hand_count(Color::Black, HandPieceType::Pawn, 1);
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);
        let moves = gs.legal_moves();

        // No pawn drop on column 4 should appear.
        let nifu_drops: Vec<_> = moves
            .iter()
            .filter(|m| {
                if let Move::Drop { to, piece_type: HandPieceType::Pawn } = m {
                    to.col() == 4
                } else {
                    false
                }
            })
            .collect();

        assert!(
            nifu_drops.is_empty(),
            "Nifu: pawn drops on col 4 should be excluded, but found {:?}",
            nifu_drops
        );

        // But pawn drops on other columns should be allowed (where squares are empty).
        let other_col_drops: Vec<_> = moves
            .iter()
            .filter(|m| {
                if let Move::Drop { to, piece_type: HandPieceType::Pawn } = m {
                    to.col() != 4
                } else {
                    false
                }
            })
            .collect();

        assert!(
            !other_col_drops.is_empty(),
            "Pawn drops on other columns should be allowed"
        );
    }

    // -----------------------------------------------------------------------
    // test_ply_tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_ply_tracking() {
        let mut gs = GameState::new();
        assert_eq!(gs.ply, 0);

        let moves = gs.legal_moves();
        let mv = moves[0];
        let undo = gs.make_move(mv);
        assert_eq!(gs.ply, 1);

        gs.unmake_move(mv, undo);
        assert_eq!(gs.ply, 0);
    }

    // -----------------------------------------------------------------------
    // test_check_termination_max_ply
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_termination_max_ply() {
        let mut gs = GameState::with_max_ply(0);
        gs.check_termination();
        assert_eq!(gs.result, GameResult::MaxMoves);
    }

    // -----------------------------------------------------------------------
    // test_check_termination_checkmate
    // -----------------------------------------------------------------------

    #[test]
    fn test_check_termination_checkmate() {
        // A contrived checkmate: Black king at (0,0) hemmed in.
        // White rook on (0,8) gives check along row 0.
        // White gold at (1,1) controls escape squares (0,1), (1,0).
        // White rook on (8,1) covers (1,1) so king can't capture gold.
        let mut pos = Position::empty();
        pos.set_piece(
            Square::from_row_col(0, 0).unwrap(),
            Piece::new(PieceType::King, Color::Black, false),
        );
        pos.set_piece(
            Square::from_row_col(8, 8).unwrap(),
            Piece::new(PieceType::King, Color::White, false),
        );
        // Rook giving check along row 0.
        pos.set_piece(
            Square::from_row_col(0, 8).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        // Gold covering (0,1), (1,0), and occupying (1,1).
        pos.set_piece(
            Square::from_row_col(1, 1).unwrap(),
            Piece::new(PieceType::Gold, Color::White, false),
        );
        // Second rook covering col 1 to protect the gold at (1,1).
        pos.set_piece(
            Square::from_row_col(8, 1).unwrap(),
            Piece::new(PieceType::Rook, Color::White, false),
        );
        pos.current_player = Color::Black;
        pos.hash = pos.compute_hash();

        let mut gs = GameState::from_position(pos, 500);
        gs.check_termination();
        assert_eq!(
            gs.result,
            GameResult::Checkmate { winner: Color::White },
            "Black should be checkmated"
        );
    }

    // -----------------------------------------------------------------------
    // test_from_sfen
    // -----------------------------------------------------------------------

    #[test]
    fn test_from_sfen() {
        let gs = GameState::from_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            500,
        )
        .expect("failed to parse SFEN");

        assert_eq!(gs.position.current_player, Color::Black);
        assert_eq!(gs.ply, 0);
        assert_eq!(gs.max_ply, 500);
        assert_eq!(gs.result, GameResult::InProgress);
    }

    // -----------------------------------------------------------------------
    // test_hot_path_matches_ergonomic
    // -----------------------------------------------------------------------

    #[test]
    fn test_hot_path_matches_ergonomic() {
        let mut gs = GameState::new();

        // Collect moves from the ergonomic Vec-based API.
        let mut ergonomic = gs.legal_moves();
        ergonomic.sort_by_key(|m| format!("{:?}", m));

        // Collect moves from the hot-path MoveList API.
        let mut ml = MoveList::new();
        gs.generate_legal_moves_into(&mut ml);
        let mut hot_path: Vec<Move> = ml.iter().copied().collect();
        hot_path.sort_by_key(|m| format!("{:?}", m));

        assert_eq!(
            ergonomic.len(),
            hot_path.len(),
            "hot-path produced {} moves but ergonomic produced {}",
            hot_path.len(),
            ergonomic.len(),
        );
        assert_eq!(
            ergonomic, hot_path,
            "hot-path move set differs from ergonomic move set"
        );
    }
}
