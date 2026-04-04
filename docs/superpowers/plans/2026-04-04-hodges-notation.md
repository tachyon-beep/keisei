# Hodges Notation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace coordinate notation (`9g→9f`) with standard Hodges shogi notation (`P-7f`) in the Rust engine's move notation function.

**Architecture:** Modify `move_notation()` in `spectator_data.rs` to accept `&Position` and `&[Move]`, enabling piece lookup, capture detection, disambiguation, and promotion suffix logic. Update callers in `vec_env.rs` and `spectator.rs` to pass game state and legal moves.

**Tech Stack:** Rust (shogi-core, shogi-gym), JavaScript (WebUI moveRows.js), Vitest

**Spec:** `docs/superpowers/specs/2026-04-04-hodges-notation-design.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `shogi-engine/crates/shogi-gym/src/spectator_data.rs` | Modify | Rewrite `move_notation()`, add helpers, consolidate `hand_piece_char` |
| `shogi-engine/crates/shogi-gym/src/vec_env.rs` | Modify | Update notation recording to pass position and legal moves |
| `shogi-engine/crates/shogi-gym/src/spectator.rs` | Modify | Update `step()` to pass position and legal moves to `move_notation()` |
| `webui/src/lib/moveRows.js` | Modify | Update toggle label from `'coord'` to `'western'` |
| `webui/src/lib/MoveLog.svelte` | Modify | Update toggle button text |
| `webui/src/lib/moveRows.test.js` | Modify | Add `toJapanese` integration tests for Hodges notation |

---

### Task 1: Add helper functions and `piece_char`

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/spectator_data.rs:1-47`

- [ ] **Step 1: Write failing tests for `piece_char` and `square_notation`**

Add these tests at the bottom of the `mod tests` block in `spectator_data.rs` (before the closing `}`):

```rust
    #[test]
    fn test_piece_char_all() {
        assert_eq!(piece_char(PieceType::King), 'K');
        assert_eq!(piece_char(PieceType::Rook), 'R');
        assert_eq!(piece_char(PieceType::Bishop), 'B');
        assert_eq!(piece_char(PieceType::Gold), 'G');
        assert_eq!(piece_char(PieceType::Silver), 'S');
        assert_eq!(piece_char(PieceType::Knight), 'N');
        assert_eq!(piece_char(PieceType::Lance), 'L');
        assert_eq!(piece_char(PieceType::Pawn), 'P');
    }

    #[test]
    fn test_square_notation_corners_and_center() {
        // Top-right (row=0, col=0): file=9-0=9, rank='a'+0='a' → "9a"
        assert_eq!(square_notation(Square::from_row_col(0, 0).unwrap()), "9a");
        // Bottom-left (row=8, col=8): file=9-8=1, rank='a'+8='i' → "1i"
        assert_eq!(square_notation(Square::from_row_col(8, 8).unwrap()), "1i");
        // Center (row=4, col=4): file=9-4=5, rank='a'+4='e' → "5e"
        assert_eq!(square_notation(Square::from_row_col(4, 4).unwrap()), "5e");
        // Top-left (row=0, col=8): file=9-8=1, rank='a'+0='a' → "1a"
        assert_eq!(square_notation(Square::from_row_col(0, 8).unwrap()), "1a");
        // Bottom-right (row=8, col=0): file=9-0=9, rank='a'+8='i' → "9i"
        assert_eq!(square_notation(Square::from_row_col(8, 0).unwrap()), "9i");
    }

    #[test]
    fn test_hand_piece_char_delegates_to_piece_char() {
        for &hpt in &HandPieceType::ALL {
            assert_eq!(
                hand_piece_char(hpt),
                piece_char(hpt.to_piece_type()),
                "hand_piece_char and piece_char disagree for {:?}",
                hpt
            );
        }
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/john/keisei/shogi-engine && cargo test -p shogi-gym -- test_piece_char_all test_square_notation test_hand_piece_char_delegates 2>&1 | tail -20`

Expected: compilation errors — `piece_char` and `square_notation` not found.

- [ ] **Step 3: Implement `piece_char` and `square_notation`, consolidate `hand_piece_char`**

Add these functions to `spectator_data.rs` after the existing `use` statements and before `piece_type_name`:

```rust
/// Single-letter piece abbreviation for Hodges notation.
pub fn piece_char(pt: PieceType) -> char {
    match pt {
        PieceType::King   => 'K',
        PieceType::Rook   => 'R',
        PieceType::Bishop => 'B',
        PieceType::Gold   => 'G',
        PieceType::Silver => 'S',
        PieceType::Knight => 'N',
        PieceType::Lance  => 'L',
        PieceType::Pawn   => 'P',
    }
}

/// Format a square as Hodges coordinates: file (1-9) + rank (a-i).
/// Example: row=6, col=4 → file=5, rank='g' → "5g"
pub fn square_notation(sq: Square) -> String {
    let file = 9 - sq.col();
    let rank = (b'a' + sq.row()) as char;
    format!("{}{}", file, rank)
}
```

Then replace the existing `hand_piece_char` body to delegate:

```rust
/// Encode a drop piece char: P, L, N, S, G, B, R
pub fn hand_piece_char(hpt: HandPieceType) -> char {
    piece_char(hpt.to_piece_type())
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/john/keisei/shogi-engine && cargo test -p shogi-gym -- test_piece_char_all test_square_notation test_hand_piece_char_delegates 2>&1 | tail -20`

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/john/keisei && git add shogi-engine/crates/shogi-gym/src/spectator_data.rs
git commit -m "feat(notation): add piece_char, square_notation helpers for Hodges notation"
```

---

### Task 2: Add promotion zone and disambiguation helpers

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/spectator_data.rs`

- [ ] **Step 1: Write failing tests for `in_promotion_zone` and `could_promote`**

Add to `mod tests`:

```rust
    use crate::spectator_data::{in_promotion_zone, could_promote};

    #[test]
    fn test_in_promotion_zone_black() {
        // Black promotion zone: rows 0, 1, 2 (ranks a, b, c)
        let col = 4;
        assert!(in_promotion_zone(Square::from_row_col(0, col).unwrap(), Color::Black));
        assert!(in_promotion_zone(Square::from_row_col(1, col).unwrap(), Color::Black));
        assert!(in_promotion_zone(Square::from_row_col(2, col).unwrap(), Color::Black));
        // Row 3 is NOT in zone
        assert!(!in_promotion_zone(Square::from_row_col(3, col).unwrap(), Color::Black));
        // Nor is row 8
        assert!(!in_promotion_zone(Square::from_row_col(8, col).unwrap(), Color::Black));
    }

    #[test]
    fn test_in_promotion_zone_white() {
        // White promotion zone: rows 6, 7, 8 (ranks g, h, i)
        let col = 4;
        assert!(in_promotion_zone(Square::from_row_col(6, col).unwrap(), Color::White));
        assert!(in_promotion_zone(Square::from_row_col(7, col).unwrap(), Color::White));
        assert!(in_promotion_zone(Square::from_row_col(8, col).unwrap(), Color::White));
        // Row 5 is NOT in zone
        assert!(!in_promotion_zone(Square::from_row_col(5, col).unwrap(), Color::White));
        // Nor is row 0
        assert!(!in_promotion_zone(Square::from_row_col(0, col).unwrap(), Color::White));
    }

    #[test]
    fn test_could_promote_all_conditions() {
        let from_outside = Square::from_row_col(5, 4).unwrap(); // row 5 = rank f
        let to_inside = Square::from_row_col(2, 4).unwrap();    // row 2 = rank c (Black zone)
        let from_inside = Square::from_row_col(2, 4).unwrap();
        let to_outside = Square::from_row_col(5, 4).unwrap();

        let silver_black = Piece::new(PieceType::Silver, Color::Black, false);
        let gold_black = Piece::new(PieceType::Gold, Color::Black, false);
        let promoted_silver = Piece::new(PieceType::Silver, Color::Black, true);
        let king_black = Piece::new(PieceType::King, Color::Black, false);

        // Silver moving INTO zone: can promote
        assert!(could_promote(silver_black, from_outside, to_inside));
        // Silver moving OUT OF zone: can promote
        assert!(could_promote(silver_black, from_inside, to_outside));
        // Gold: cannot promote (type doesn't allow)
        assert!(!could_promote(gold_black, from_outside, to_inside));
        // King: cannot promote
        assert!(!could_promote(king_black, from_outside, to_inside));
        // Already promoted silver: cannot promote again
        assert!(!could_promote(promoted_silver, from_outside, to_inside));
        // Silver moving entirely outside zone: cannot promote
        let other_outside = Square::from_row_col(6, 4).unwrap();
        assert!(!could_promote(silver_black, from_outside, other_outside));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/john/keisei/shogi-engine && cargo test -p shogi-gym -- test_in_promotion_zone test_could_promote 2>&1 | tail -20`

Expected: compilation errors — `in_promotion_zone` and `could_promote` not found.

- [ ] **Step 3: Implement the helpers**

Add to `spectator_data.rs` after `square_notation`, before `move_notation`. Also add `use shogi_core::Piece;` to the imports at the top (alongside the existing `use shogi_core::{...}`):

Update the import line:
```rust
use shogi_core::{Color, GameResult, GameState, HandPieceType, Move, Piece, PieceType, Position, Square};
```

Add the functions:

```rust
/// Check if a square is in the promotion zone for the given color.
/// Black: rows 0-2 (ranks a-c). White: rows 6-8 (ranks g-i).
pub fn in_promotion_zone(sq: Square, color: Color) -> bool {
    match color {
        Color::Black => sq.row() <= 2,
        Color::White => sq.row() >= 6,
    }
}

/// Check if a piece could promote on this move (but hasn't necessarily chosen to).
/// True when: piece type can promote, piece is not already promoted,
/// and either source or destination is in the promotion zone.
pub fn could_promote(piece: Piece, from: Square, to: Square) -> bool {
    piece.piece_type().can_promote()
        && !piece.is_promoted()
        && (in_promotion_zone(from, piece.color()) || in_promotion_zone(to, piece.color()))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/john/keisei/shogi-engine && cargo test -p shogi-gym -- test_in_promotion_zone test_could_promote 2>&1 | tail -20`

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/john/keisei && git add shogi-engine/crates/shogi-gym/src/spectator_data.rs
git commit -m "feat(notation): add promotion zone and could_promote helpers"
```

---

### Task 3: Rewrite `move_notation` with Hodges format

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/spectator_data.rs:49-74` (replace `move_notation`)

- [ ] **Step 1: Write failing tests for new `move_notation` signature**

Replace ALL existing `move_notation` tests in `spectator.rs` (lines 207-410 — the tests in the `mod tests` block of `spectator.rs` that reference `move_notation`). Read the file first, then replace the existing test functions (`test_move_notation_board_move`, `test_move_notation_board_move_with_promotion`, `test_move_notation_drop`, `test_move_notation_drop_all_piece_types`, `test_move_notation_boundary_squares`) with:

```rust
    use shogi_core::{Piece, Position};
    use crate::spectator_data::{move_notation, hand_piece_char};

    // Helper: create a position with specific pieces placed.
    // Starts from empty board, places pieces, sets current_player.
    fn position_with_pieces(pieces: &[(Square, Piece)]) -> Position {
        let mut pos = Position::empty();
        for &(sq, piece) in pieces {
            pos.set_piece(sq, piece);
        }
        pos
    }

    // -----------------------------------------------------------------------
    // move_notation tests — Hodges format
    // -----------------------------------------------------------------------

    #[test]
    fn test_notation_simple_move() {
        // Black pawn at 7g (row=6, col=2) moves to 7f (row=5, col=2)
        let from = Square::from_row_col(6, 2).unwrap();
        let to = Square::from_row_col(5, 2).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-7f");
    }

    #[test]
    fn test_notation_capture() {
        // Black bishop at 8h captures White pawn at 3c
        let from = Square::from_row_col(7, 1).unwrap(); // 8h
        let to = Square::from_row_col(2, 6).unwrap();   // 3c
        let bishop = Piece::new(PieceType::Bishop, Color::Black, false);
        let enemy_pawn = Piece::new(PieceType::Pawn, Color::White, false);
        let pos = position_with_pieces(&[(from, bishop), (to, enemy_pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "Bx3c=");
    }

    #[test]
    fn test_notation_promotion() {
        // Black knight at 8d (row=3, col=1) captures on 7b (row=1, col=2), promotes
        let from = Square::from_row_col(3, 1).unwrap(); // 8d
        let to = Square::from_row_col(1, 2).unwrap();   // 7b
        let knight = Piece::new(PieceType::Knight, Color::Black, false);
        let enemy = Piece::new(PieceType::Gold, Color::White, false);
        let pos = position_with_pieces(&[(from, knight), (to, enemy)]);
        let mv = Move::Board { from, to, promote: true };
        assert_eq!(move_notation(mv, &pos, &[mv]), "Nx7b+");
    }

    #[test]
    fn test_notation_declined_promotion() {
        // Black silver at 4d (row=3, col=5) moves to 4c (row=2, col=5), declines
        let from = Square::from_row_col(3, 5).unwrap(); // 4d
        let to = Square::from_row_col(2, 5).unwrap();   // 4c
        let silver = Piece::new(PieceType::Silver, Color::Black, false);
        let pos = position_with_pieces(&[(from, silver)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "S-4c=");
    }

    #[test]
    fn test_notation_promoted_piece_moving() {
        // Promoted rook (dragon) at 5a (row=0, col=4) moves to 5b (row=1, col=4)
        let from = Square::from_row_col(0, 4).unwrap(); // 5a
        let to = Square::from_row_col(1, 4).unwrap();   // 5b
        let dragon = Piece::new(PieceType::Rook, Color::Black, true);
        let pos = position_with_pieces(&[(from, dragon)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "+R-5b");
    }

    #[test]
    fn test_notation_drop() {
        // Drop pawn at 5e
        let to = Square::from_row_col(4, 4).unwrap();
        let pos = Position::empty();
        let mv = Move::Drop { to, piece_type: HandPieceType::Pawn };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P*5e");
    }

    #[test]
    fn test_notation_drop_all_piece_types() {
        let to = Square::from_row_col(4, 4).unwrap();
        let pos = Position::empty();
        let expected = [
            (HandPieceType::Pawn,   "P*5e"),
            (HandPieceType::Lance,  "L*5e"),
            (HandPieceType::Knight, "N*5e"),
            (HandPieceType::Silver, "S*5e"),
            (HandPieceType::Gold,   "G*5e"),
            (HandPieceType::Bishop, "B*5e"),
            (HandPieceType::Rook,   "R*5e"),
        ];
        for (hpt, exp) in &expected {
            let mv = Move::Drop { to, piece_type: *hpt };
            assert_eq!(move_notation(mv, &pos, &[mv]), *exp, "Drop for {:?}", hpt);
        }
    }

    #[test]
    fn test_notation_disambiguation() {
        // Two Black golds can both reach 5f (row=5, col=4) by moving forward.
        // Gold at 6g (row=6, col=3) moves diag-forward-right to 5f.
        // Gold at 4g (row=6, col=5) moves diag-forward-left to 5f.
        // (Gold can move forward, diag-forward, sideways, or straight back.)
        let from1 = Square::from_row_col(6, 3).unwrap(); // 6g
        let from2 = Square::from_row_col(6, 5).unwrap(); // 4g
        let to = Square::from_row_col(5, 4).unwrap();    // 5f
        let gold = Piece::new(PieceType::Gold, Color::Black, false);
        let pos = position_with_pieces(&[(from1, gold), (from2, gold)]);
        let mv1 = Move::Board { from: from1, to, promote: false };
        let mv2 = Move::Board { from: from2, to, promote: false };
        let legal = vec![mv1, mv2];
        assert_eq!(move_notation(mv1, &pos, &legal), "G6g-5f");
        assert_eq!(move_notation(mv2, &pos, &legal), "G4g-5f");
    }

    #[test]
    fn test_notation_no_disambiguation_single_piece() {
        // Only one gold can reach 5h — no origin needed
        let from = Square::from_row_col(6, 3).unwrap(); // 6g
        let to = Square::from_row_col(7, 4).unwrap();   // 5h
        let gold = Piece::new(PieceType::Gold, Color::Black, false);
        let pos = position_with_pieces(&[(from, gold)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "G-5h");
    }

    #[test]
    fn test_notation_king_never_disambiguated() {
        // Even if we pass a contrived legal_moves list with two "king" moves,
        // king should never get origin coordinates (only one king per side).
        let from = Square::from_row_col(8, 4).unwrap(); // 5i
        let to = Square::from_row_col(7, 4).unwrap();   // 5h
        let king = Piece::new(PieceType::King, Color::Black, false);
        let pos = position_with_pieces(&[(from, king)]);
        let mv = Move::Board { from, to, promote: false };
        // Pass a fake second king move — should still not disambiguate
        let fake = Move::Board {
            from: Square::from_row_col(7, 3).unwrap(),
            to,
            promote: false,
        };
        assert_eq!(move_notation(mv, &pos, &[mv, fake]), "K-5h");
    }

    #[test]
    fn test_notation_forced_promotion_pawn_last_rank() {
        // Black pawn at 7b (row=1, col=2) moves to 7a (row=0, col=2).
        // Promotion is forced — must show "+", never "=".
        let from = Square::from_row_col(1, 2).unwrap();
        let to = Square::from_row_col(0, 2).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::Black, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        // Even if promote flag were false, forced promotion means "+"
        let mv = Move::Board { from, to, promote: true };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-7a+");
        // Verify the guard: if engine erroneously passes promote=false,
        // we still emit "+" for forced promotion
        let mv_bad = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv_bad, &pos, &[mv_bad]), "P-7a+");
    }

    #[test]
    fn test_notation_forced_promotion_knight_last_two_ranks() {
        // Black knight at 7d (row=3, col=2) moves to 8b (row=1, col=1).
        // Knight reaching row 0 or 1 for Black = forced promotion.
        let from = Square::from_row_col(3, 2).unwrap();
        let to = Square::from_row_col(1, 1).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::Black, false);
        let pos = position_with_pieces(&[(from, knight)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "N-8b+");
    }

    #[test]
    fn test_notation_forced_promotion_lance_last_rank() {
        // Black lance at 5b (row=1, col=4) moves to 5a (row=0, col=4).
        let from = Square::from_row_col(1, 4).unwrap();
        let to = Square::from_row_col(0, 4).unwrap();
        let lance = Piece::new(PieceType::Lance, Color::Black, false);
        let pos = position_with_pieces(&[(from, lance)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "L-5a+");
    }

    #[test]
    fn test_notation_white_forced_promotion() {
        // White pawn at 3h (row=7, col=6) moves to 3i (row=8, col=6).
        let from = Square::from_row_col(7, 6).unwrap();
        let to = Square::from_row_col(8, 6).unwrap();
        let pawn = Piece::new(PieceType::Pawn, Color::White, false);
        let pos = position_with_pieces(&[(from, pawn)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "P-3i+");
    }

    #[test]
    fn test_notation_white_knight_forced_promotion() {
        // White knight at 3g (row=6, col=6) moves to 2i (row=8, col=7).
        // Row 8 for White = forced promotion for knight (rows 7-8).
        let from = Square::from_row_col(6, 6).unwrap();
        let to = Square::from_row_col(8, 7).unwrap();
        let knight = Piece::new(PieceType::Knight, Color::White, false);
        let pos = position_with_pieces(&[(from, knight)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "N-2i+");
    }

    #[test]
    fn test_notation_boundary_squares() {
        // 9a (row=0, col=0) → 9b (row=1, col=0): King move
        let from = Square::from_row_col(0, 0).unwrap();
        let to = Square::from_row_col(1, 0).unwrap();
        let king = Piece::new(PieceType::King, Color::Black, false);
        let pos = position_with_pieces(&[(from, king)]);
        let mv = Move::Board { from, to, promote: false };
        assert_eq!(move_notation(mv, &pos, &[mv]), "K-9b");

        // 1i (row=8, col=8) → 1h (row=7, col=8): King move
        let from2 = Square::from_row_col(8, 8).unwrap();
        let to2 = Square::from_row_col(7, 8).unwrap();
        let king2 = Piece::new(PieceType::King, Color::White, false);
        let pos2 = position_with_pieces(&[(from2, king2)]);
        let mv2 = Move::Board { from: from2, to: to2, promote: false };
        assert_eq!(move_notation(mv2, &pos2, &[mv2]), "K-1h");

        // Drop at corner 1a
        let drop_sq = Square::from_row_col(0, 8).unwrap();
        let pos3 = Position::empty();
        let mv3 = Move::Drop { to: drop_sq, piece_type: HandPieceType::Pawn };
        assert_eq!(move_notation(mv3, &pos3, &[mv3]), "P*1a");
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/john/keisei/shogi-engine && cargo test -p shogi-gym -- test_notation_ 2>&1 | tail -30`

Expected: failures — `move_notation` signature mismatch (still takes only `Move`).

- [ ] **Step 3: Check if `Position::empty()` exists**

Run: `cd /home/john/keisei/shogi-engine && grep -n 'fn empty' crates/shogi-core/src/position.rs`

If it doesn't exist, add it to `position.rs` inside `impl Position`:

```rust
    /// Create an empty position (no pieces, Black to move).
    pub fn empty() -> Position {
        Position {
            board: [0u8; Square::NUM_SQUARES],
            hands: [[0u8; HandPieceType::COUNT]; 2],
            current_player: Color::Black,
            hash: 0,
        }
    }
```

- [ ] **Step 4: Rewrite `move_notation` with Hodges format**

Replace the entire `move_notation` function (lines 49-74 of `spectator_data.rs`) with:

```rust
/// Check if promotion is forced for this piece reaching the destination.
/// Pawn/Lance on last rank, Knight on last two ranks.
fn is_forced_promotion(pt: PieceType, to: Square, color: Color) -> bool {
    let dest_row = to.row();
    match color {
        Color::Black => match pt {
            PieceType::Pawn | PieceType::Lance => dest_row == 0,
            PieceType::Knight => dest_row <= 1,
            _ => false,
        },
        Color::White => match pt {
            PieceType::Pawn | PieceType::Lance => dest_row == 8,
            PieceType::Knight => dest_row >= 7,
            _ => false,
        },
    }
}

/// Build Hodges notation string from a Move, a Position, and the legal moves list.
///
/// Board: `"P-7f"`, `"Bx3c"`, `"Nx7c+"`, `"S-4d="`, `"+R-5a"`, `"G6g-5h"`
/// Drop:  `"P*5e"`
pub fn move_notation(mv: Move, position: &Position, legal_moves: &[Move]) -> String {
    match mv {
        Move::Board { from, to, promote } => {
            let piece = position.piece_at(from)
                .expect("move_notation: no piece at source square");
            let pt = piece.piece_type();
            let color = piece.color();
            let promoted = piece.is_promoted();

            // Piece prefix: "+R" if promoted, "R" if not
            let prefix = if promoted {
                format!("+{}", piece_char(pt))
            } else {
                format!("{}", piece_char(pt))
            };

            // Disambiguation: check if another legal board move by same piece type
            // (and same promoted status) targets the same destination
            let disambig = if pt == PieceType::King {
                String::new()
            } else {
                let ambiguous = legal_moves.iter().any(|other| {
                    if let Move::Board { from: of, to: ot, .. } = other {
                        *ot == to && *of != from && {
                            if let Some(other_piece) = position.piece_at(*of) {
                                other_piece.piece_type() == pt
                                    && other_piece.is_promoted() == promoted
                            } else {
                                false
                            }
                        }
                    } else {
                        false
                    }
                });
                if ambiguous {
                    square_notation(from)
                } else {
                    String::new()
                }
            };

            // Capture or move separator
            let sep = if position.piece_at(to).is_some() { "x" } else { "-" };

            // Destination
            let dest = square_notation(to);

            // Promotion suffix
            let suffix = if promote || is_forced_promotion(pt, to, color) {
                "+"
            } else if could_promote(piece, from, to) {
                "="
            } else {
                ""
            };

            format!("{}{}{}{}{}", prefix, disambig, sep, dest, suffix)
        }
        Move::Drop { to, piece_type } => {
            format!("{}*{}", piece_char(piece_type.to_piece_type()), square_notation(to))
        }
    }
}
```

- [ ] **Step 5: Run all tests**

Run: `cd /home/john/keisei/shogi-engine && cargo test -p shogi-gym -- test_notation_ 2>&1 | tail -40`

Expected: all notation tests pass.

- [ ] **Step 6: Do NOT commit yet** — the crate won't compile until Tasks 4 and 5 update callers.

---

### Task 4: Update VecEnv caller to pass position and legal moves

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/vec_env.rs:676-681`

- [ ] **Step 1: Update the notation recording loop**

In `vec_env.rs`, replace lines 676-681:

```rust
        // Record move notations before apply (single-threaded, before state mutation)
        for (i, mv) in decoded_moves.iter().enumerate() {
            let action_idx = actions[i] as usize;
            let notation = move_notation(*mv);
            self.move_histories[i].push((action_idx, notation));
        }
```

With:

```rust
        // Record move notations before apply (single-threaded, before state mutation).
        // Generate legal moves per-env for disambiguation. The mutable borrow for
        // generate_legal_moves_into must end before we borrow position immutably.
        let mut move_list = MoveList::new();
        for (i, mv) in decoded_moves.iter().enumerate() {
            let action_idx = actions[i] as usize;
            self.games[i].generate_legal_moves_into(&mut move_list);
            let notation = move_notation(*mv, &self.games[i].position, move_list.as_slice());
            self.move_histories[i].push((action_idx, notation));
        }
```

- [ ] **Step 2: Add MoveList to imports**

In `vec_env.rs`, find the `use shogi_core::` import line and add `MoveList` if not already present. It should look like:

```rust
use shogi_core::{Color, GameResult, GameState, Move, MoveList, Square};
```

(Keep whatever other items are already imported, just add `MoveList`.)

- [ ] **Step 3: Build to check compilation**

Run: `cd /home/john/keisei/shogi-engine && cargo build -p shogi-gym 2>&1 | tail -20`

Expected: compiles successfully.

- [ ] **Step 4: Run full test suite**

Run: `cd /home/john/keisei/shogi-engine && cargo test -p shogi-gym 2>&1 | tail -30`

Expected: all tests pass.

- [ ] **Step 5: Do NOT commit yet** — wait for Task 5 to complete so all callers compile together.

---

### Task 5: Update SpectatorEnv caller

**Files:**
- Modify: `shogi-engine/crates/shogi-gym/src/spectator.rs:88-100`

- [ ] **Step 1: Update SpectatorEnv::step()**

In `spectator.rs`, the `step()` method (around line 81) already computes `legal_moves` at line 91. Replace lines 99-100:

```rust
        let notation = move_notation(mv);
        self.move_history.push((action, notation));
```

With:

```rust
        let notation = move_notation(mv, &self.game.position, &legal_moves);
        self.move_history.push((action, notation));
```

This works because `legal_moves` is already a `Vec<Move>` computed at line 91, and `self.game.position` is available since `make_move` hasn't been called yet at this point.

- [ ] **Step 2: Build and test**

Run: `cd /home/john/keisei/shogi-engine && cargo test -p shogi-gym 2>&1 | tail -20`

Expected: all tests pass.

- [ ] **Step 3: Commit Tasks 3+4+5 together** (signature change + both caller updates must be atomic)

```bash
cd /home/john/keisei && git add shogi-engine/crates/shogi-gym/src/spectator_data.rs shogi-engine/crates/shogi-gym/src/spectator.rs shogi-engine/crates/shogi-gym/src/vec_env.rs shogi-engine/crates/shogi-core/src/position.rs
git commit -m "feat(notation): rewrite move_notation to produce Hodges shogi notation

Replaces coordinate notation (9g→9f) with standard Western Hodges
notation (P-7f). Includes piece prefix, capture/move separator,
promotion suffixes (+/=), forced promotion guard, and disambiguation."
```

---

### Task 6: Update WebUI labels and add toJapanese integration tests

**Files:**
- Modify: `webui/src/lib/moveRows.js:27`
- Modify: `webui/src/lib/MoveLog.svelte:12,29`
- Modify: `webui/src/lib/moveRows.test.js`

- [ ] **Step 1: Write failing tests for toJapanese with Hodges notation**

Add a new `describe` block at the end of `moveRows.test.js`. First, export `toJapanese` — it's currently a private function. In `moveRows.js`, change `function toJapanese` to `export function toJapanese`:

In `moveRows.js` line 46, change:
```javascript
function toJapanese(notation) {
```
To:
```javascript
export function toJapanese(notation) {
```

Then add to `moveRows.test.js`:

```javascript
import { parseMoves, buildMoveRows, toJapanese } from './moveRows.js'

describe('toJapanese', () => {
  it('converts simple Hodges move', () => {
    expect(toJapanese('P-7f')).toBe('P-７六')
  })

  it('converts capture notation', () => {
    expect(toJapanese('Bx3c')).toBe('Bx３三')
  })

  it('converts promoted piece move', () => {
    expect(toJapanese('+R-5a')).toBe('+R-５一')
  })

  it('converts drop notation', () => {
    expect(toJapanese('P*5e')).toBe('P*５五')
  })

  it('converts promotion suffix', () => {
    expect(toJapanese('Nx7c+')).toBe('Nx７三+')
  })

  it('converts declined promotion suffix', () => {
    expect(toJapanese('S-4d=')).toBe('S-４四=')
  })

  it('converts disambiguated move', () => {
    expect(toJapanese('G6g-5h')).toBe('G６七-５八')
  })

  it('returns empty string for empty input', () => {
    expect(toJapanese('')).toBe('')
  })
})
```

Also update the existing import at line 2 to include `toJapanese`:
```javascript
import { parseMoves, buildMoveRows, toJapanese } from './moveRows.js'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/john/keisei/webui && npx vitest run src/lib/moveRows.test.js 2>&1 | tail -30`

Expected: `toJapanese` tests fail (not exported yet, or import fails).

- [ ] **Step 3: Export toJapanese and run tests**

Make the change in `moveRows.js` (line 46) as described in Step 1, then:

Run: `cd /home/john/keisei/webui && npx vitest run src/lib/moveRows.test.js 2>&1 | tail -30`

Expected: all tests pass (the regex `/([1-9])([a-i])/g` correctly matches coordinates in Hodges format).

- [ ] **Step 4: Update toggle labels**

In `moveRows.js` line 27, change:
```javascript
  const fmt = style === 'japanese' ? toJapanese : (s) => s
```
No change needed here — `'coord'` was the old name but it's only compared as `=== 'japanese'`.

In `MoveLog.svelte`, update the toggle state and button text. Change line 9:
```javascript
  let notationStyle = 'western'
```

Change line 12:
```javascript
    notationStyle = notationStyle === 'western' ? 'japanese' : 'western'
```

Change line 29:
```javascript
      {notationStyle === 'western' ? '漢' : 'W'}
```

- [ ] **Step 5: Run WebUI tests**

Run: `cd /home/john/keisei/webui && npx vitest run 2>&1 | tail -20`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/john/keisei && git add webui/src/lib/moveRows.js webui/src/lib/moveRows.test.js webui/src/lib/MoveLog.svelte
git commit -m "feat(notation): update WebUI for Hodges notation labels and add toJapanese tests"
```

---

### Task 7: Full integration build and test

**Files:** None (verification only)

- [ ] **Step 1: Run full Rust test suite**

Run: `cd /home/john/keisei/shogi-engine && cargo test 2>&1 | tail -20`

Expected: all tests pass.

- [ ] **Step 2: Run full WebUI test suite**

Run: `cd /home/john/keisei/webui && npx vitest run 2>&1 | tail -20`

Expected: all tests pass.

- [ ] **Step 3: Build the Python wheel to confirm Rust compilation**

Run: `cd /home/john/keisei && uv run maturin develop --manifest-path shogi-engine/Cargo.toml --release 2>&1 | tail -10`

Expected: wheel builds successfully.

- [ ] **Step 4: Run Python tests**

Run: `cd /home/john/keisei && uv run pytest -x -q 2>&1 | tail -20`

Expected: all tests pass (Python tests use the engine's notation output — any that assert on old coordinate format will need updating here).

- [ ] **Step 5: Verify in browser (manual)**

Start the WebUI and confirm:
- Move log shows Hodges notation (e.g., `P-7f` not `7g→7f`)
- Toggle switches between Western and Japanese
- Japanese mode converts coordinates correctly (e.g., `P-７六`)

- [ ] **Step 6: Final commit if any fixups were needed**

```bash
cd /home/john/keisei && git add -u
git commit -m "fix(notation): integration fixups for Hodges notation"
```
