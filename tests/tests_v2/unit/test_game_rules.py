"""Tests for the Shogi game rules engine using real ShogiGame objects (no mocks).

Covers: initial position, legal move generation, check detection, checkmate,
piece movement rules, promotion zone, and board boundaries.
"""

import pytest

from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece, MoveTuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _board_moves_to_set(legal_moves):
    """Extract (from_r, from_c, to_r, to_c, promote) tuples for board moves only."""
    return {m for m in legal_moves if m[0] is not None}


def _destinations_from(legal_moves, from_r, from_c):
    """Return set of (to_r, to_c) for board moves originating from (from_r, from_c)."""
    return {
        (m[2], m[3])
        for m in legal_moves
        if m[0] == from_r and m[1] == from_c
    }


def _move_exists(legal_moves, from_r, from_c, to_r, to_c, promote=None):
    """Check whether a specific board move exists in the legal moves list.

    If *promote* is None, any promotion flag is accepted.
    """
    for m in legal_moves:
        if m[0] == from_r and m[1] == from_c and m[2] == to_r and m[3] == to_c:
            if promote is None or m[4] == promote:
                return True
    return False


# ===========================================================================
# 1. Initial Position (~5 tests)
# ===========================================================================

class TestInitialPosition:
    """Verify the standard starting position is set up correctly."""

    def test_initial_position_black_moves_first(self):
        game = ShogiGame()
        assert game.current_player == Color.BLACK

    def test_initial_position_game_not_over(self):
        game = ShogiGame()
        assert game.game_over is False
        assert game.winner is None
        assert game.termination_reason is None

    def test_initial_position_piece_placement(self):
        """Check key squares for correct piece type and color."""
        game = ShogiGame()
        # Black king at (8, 4)
        king_b = game.get_piece(8, 4)
        assert king_b is not None
        assert king_b.type == PieceType.KING
        assert king_b.color == Color.BLACK

        # White king at (0, 4)
        king_w = game.get_piece(0, 4)
        assert king_w is not None
        assert king_w.type == PieceType.KING
        assert king_w.color == Color.WHITE

        # Black rook at (7, 7)
        rook_b = game.get_piece(7, 7)
        assert rook_b is not None
        assert rook_b.type == PieceType.ROOK
        assert rook_b.color == Color.BLACK

        # White bishop at (1, 7)
        bishop_w = game.get_piece(1, 7)
        assert bishop_w is not None
        assert bishop_w.type == PieceType.BISHOP
        assert bishop_w.color == Color.WHITE

        # Black pawn at (6, 0)
        pawn_b = game.get_piece(6, 0)
        assert pawn_b is not None
        assert pawn_b.type == PieceType.PAWN
        assert pawn_b.color == Color.BLACK

    def test_initial_position_has_legal_moves(self):
        game = ShogiGame()
        legal = game.get_legal_moves()
        assert len(legal) > 0
        # Black should have at least 20 moves (9 pawn pushes + piece moves)
        assert len(legal) > 20

    def test_initial_position_kings_found(self):
        game = ShogiGame()
        black_king = game.find_king(Color.BLACK)
        white_king = game.find_king(Color.WHITE)
        assert black_king == (8, 4)
        assert white_king == (0, 4)


# ===========================================================================
# 2. Legal Move Generation (~8 tests)
# ===========================================================================

class TestLegalMoveGeneration:
    """Verify individual piece movement patterns using custom SFEN positions."""

    def test_pawn_moves_forward_one_square(self):
        """A Black pawn on an open board should move one square forward (row-1)."""
        # Lone Black pawn at e5 (row 4, col 4), kings required for legality
        game = ShogiGame.from_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        # Pawn at (4,4) should be able to move to (3,4)
        assert _move_exists(legal, 4, 4, 3, 4)
        # Pawn should NOT move sideways or backward
        assert not _move_exists(legal, 4, 4, 5, 4)
        assert not _move_exists(legal, 4, 4, 4, 3)
        assert not _move_exists(legal, 4, 4, 4, 5)

    def test_knight_jumps_correctly(self):
        """A Black knight should jump 2-forward-1-side (and nothing else from center)."""
        # Black knight at (4, 4), kings for legality
        game = ShogiGame.from_sfen("4k4/9/9/9/4N4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Knight at row 4 for Black moves forward (row-2) so targets row 2, cols 3 and 5
        assert (2, 3) in dests
        assert (2, 5) in dests
        # Should NOT include adjacent squares
        assert (3, 4) not in dests
        assert (5, 4) not in dests

    def test_silver_moves_diagonally_and_forward(self):
        """Silver moves forward, forward-diagonal, and backward-diagonal."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4S4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Forward (3,4), forward-diagonal (3,3) and (3,5), backward-diagonal (5,3) and (5,5)
        assert (3, 4) in dests
        assert (3, 3) in dests
        assert (3, 5) in dests
        assert (5, 3) in dests
        assert (5, 5) in dests
        # Silver does NOT move straight backward or sideways
        assert (5, 4) not in dests
        assert (4, 3) not in dests
        assert (4, 5) not in dests

    def test_gold_moves_adjacent_except_backward_diagonal(self):
        """Gold moves in all directions except backward-diagonal."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4G4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Forward, forward-diagonals, sideways, straight back
        assert (3, 4) in dests  # forward
        assert (3, 3) in dests  # forward-left
        assert (3, 5) in dests  # forward-right
        assert (4, 3) in dests  # left
        assert (4, 5) in dests  # right
        assert (5, 4) in dests  # backward
        # Gold does NOT move backward-diagonal
        assert (5, 3) not in dests
        assert (5, 5) not in dests

    def test_king_moves_one_step_any_direction(self):
        """King moves one step in all 8 directions."""
        game = ShogiGame.from_sfen("9/9/9/9/4K4/9/9/9/4k4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        expected = {(3, 3), (3, 4), (3, 5), (4, 3), (4, 5), (5, 3), (5, 4), (5, 5)}
        assert expected.issubset(dests)

    def test_bishop_slides_diagonally(self):
        """A bishop on an open board slides along all four diagonals."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4B4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Should include all diagonal squares reachable
        assert (3, 3) in dests
        assert (2, 2) in dests
        assert (1, 1) in dests
        assert (0, 0) in dests
        assert (3, 5) in dests
        assert (5, 3) in dests
        assert (5, 5) in dests
        # Should NOT include orthogonal squares
        assert (3, 4) not in dests
        assert (4, 3) not in dests

    def test_rook_slides_orthogonally(self):
        """A rook on an open board slides along all four orthogonal directions."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Should include orthogonal squares
        assert (3, 4) in dests
        assert (0, 4) in dests  # up to row 0 (blocked by White king at 0,4 -> capture)
        assert (5, 4) in dests
        assert (4, 0) in dests
        assert (4, 8) in dests
        # Should NOT include diagonal squares
        assert (3, 3) not in dests
        assert (5, 5) not in dests

    def test_lance_slides_forward(self):
        """A Black lance slides forward (decreasing row) until blocked."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4L4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Lance slides forward from row 4 toward row 0
        assert (3, 4) in dests
        assert (2, 4) in dests
        assert (1, 4) in dests
        # Row 0 col 4 has the White king -- should be capturable
        assert (0, 4) in dests
        # Lance should NOT move backward or sideways
        assert (5, 4) not in dests
        assert (4, 3) not in dests


# ===========================================================================
# 3. Check Detection (~5 tests)
# ===========================================================================

class TestCheckDetection:
    """Verify check detection and its impact on legal moves."""

    def test_not_in_check_at_start(self):
        game = ShogiGame()
        assert game.is_in_check(Color.BLACK) is False
        assert game.is_in_check(Color.WHITE) is False

    def test_check_from_rook_on_same_file(self):
        """A rook giving check on the same file should be detected."""
        # White king at (0,4), Black rook at (4,4) -- rook attacks king along file
        game = ShogiGame.from_sfen("4k4/9/9/9/4R4/9/9/9/4K4 b - 1")
        assert game.is_in_check(Color.WHITE) is True

    def test_check_from_bishop_on_diagonal(self):
        """A bishop giving check on a diagonal should be detected."""
        # White king at (0,0), Black bishop on diagonal at (4,4)
        game = ShogiGame.from_sfen("k8/9/9/9/4B4/9/9/9/4K4 b - 1")
        assert game.is_in_check(Color.WHITE) is True

    def test_moving_into_check_not_allowed(self):
        """Moves that would leave the king in check must not appear in legal moves."""
        # Black king at (8,4), White rook at (0,3) -- file 3 is attacked
        # King should not be able to move to (7,3) because that is attacked by the rook
        game = ShogiGame.from_sfen("3r1k3/9/9/9/9/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        # King at (8,4) should NOT be able to move to (7,3) -- attacked by rook on col 3
        assert not _move_exists(legal, 8, 4, 7, 3)

    def test_blocking_check_is_legal(self):
        """A piece that can block a check should appear in legal moves."""
        # Black king at (8,4), White rook at (0,4) giving check on file,
        # Black rook at (8,0) can interpose by moving to e.g. (4,4) or any square on col 4
        # Actually, let's use a simpler setup: Black gold can block
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/4r4/3GK4 b - 1")
        # White rook at (7,4) attacks Black king at (8,4) -- Black is in check
        assert game.is_in_check(Color.BLACK) is True
        legal = game.get_legal_moves()
        # Gold at (8,3) should be able to capture/block at (7,4) if reachable
        # Gold moves: forward (7,3), forward-diag (7,2) and (7,4) -- (7,4) captures rook
        assert _move_exists(legal, 8, 3, 7, 4)


# ===========================================================================
# 4. Checkmate (~5 tests)
# ===========================================================================

class TestCheckmate:
    """Verify checkmate and stalemate detection."""

    def test_checkmate_sets_game_over(self):
        """A known checkmate position should immediately set game_over=True."""
        # White king at (0,0). Black gold at (1,1) gives check on the diagonal.
        # Black rook at (2,0) covers file 0. Black bishop at (3,3) protects the
        # gold so the king cannot capture it.
        # Escape squares: (0,1) covered by gold, (1,0) covered by gold+rook.
        # Gold at (1,1) is protected by bishop at (3,3), so capture is not legal.
        game = ShogiGame.from_sfen("k8/1G7/R8/3B5/9/9/9/9/4K4 w - 1")
        assert game.game_over is True

    def test_checkmate_winner_is_attacker(self):
        """The winner should be the player who delivered checkmate."""
        game = ShogiGame.from_sfen("k8/1G7/R8/3B5/9/9/9/9/4K4 w - 1")
        assert game.winner == Color.BLACK

    def test_checkmate_termination_reason(self):
        """Termination reason for checkmate should be 'Tsumi'."""
        game = ShogiGame.from_sfen("k8/1G7/R8/3B5/9/9/9/9/4K4 w - 1")
        assert game.termination_reason == "Tsumi"

    def test_stalemate_game_over_no_winner(self):
        """A position with no legal moves but NOT in check is stalemate (draw)."""
        # White king at (0,8). Adjacent squares: (0,7), (1,7), (1,8).
        # Black gold at (1,6) covers (0,7) via forward-diagonal.
        # Black gold at (2,7) covers (1,8) via forward-diagonal and (1,7) via forward.
        # White king is NOT in check at (0,8) -- no Black piece attacks that square.
        # White has no other pieces and no pieces in hand, so no legal moves.
        game = ShogiGame.from_sfen("8k/6G2/7G1/9/9/9/9/9/4K4 w - 1")
        assert game.game_over is True
        assert game.winner is None
        assert game.termination_reason == "stalemate"

    def test_checkmate_via_make_move(self):
        """Delivering checkmate via make_move should end the game."""
        # White king at (0,0). Black gold at (2,1) and Black rook at (2,0),
        # Black bishop at (3,3). Black to move.
        # Gold moves from (2,1) to (1,1), giving check on the diagonal.
        # After the move: king at (0,0) in check from gold at (1,1).
        # Escape (0,1): covered by gold at (1,1) forward-diagonal.
        # Escape (1,0): covered by gold at (1,1) sideways and rook at (2,0) sliding.
        # Capture gold at (1,1): protected by bishop at (3,3).
        # Result: checkmate.
        game = ShogiGame.from_sfen("k8/9/RG7/3B5/9/9/9/9/4K4 b - 1")
        assert game.game_over is False
        obs, reward, done, info = game.make_move((2, 1, 1, 1, False))
        assert done is True
        assert game.game_over is True
        assert game.winner == Color.BLACK
        assert game.termination_reason == "Tsumi"
        assert reward == 1.0


# ===========================================================================
# 5. Piece Movement Rules (~5 tests)
# ===========================================================================

class TestPieceMovementRules:
    """Verify movement rules for promoted and restricted pieces."""

    def test_promoted_rook_dragon_moves(self):
        """Promoted rook (dragon) slides orthogonally AND moves one square diagonally."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4+R4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Orthogonal sliding
        assert (0, 4) in dests  # capture White king or slide to row 0
        assert (4, 0) in dests
        assert (4, 8) in dests
        assert (8, 4) not in dests  # blocked by own king at (8,4)
        # Diagonal one-step
        assert (3, 3) in dests
        assert (3, 5) in dests
        assert (5, 3) in dests
        assert (5, 5) in dests
        # Diagonal should NOT slide beyond one step
        assert (2, 2) not in dests
        assert (2, 6) not in dests

    def test_promoted_bishop_horse_moves(self):
        """Promoted bishop (horse) slides diagonally AND moves one square orthogonally."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4+B4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Diagonal sliding
        assert (0, 0) in dests
        assert (3, 3) in dests
        assert (3, 5) in dests
        assert (5, 3) in dests
        assert (5, 5) in dests
        # Orthogonal one-step
        assert (3, 4) in dests
        assert (5, 4) in dests
        assert (4, 3) in dests
        assert (4, 5) in dests
        # Orthogonal should NOT slide beyond one step
        assert (2, 4) not in dests
        assert (4, 2) not in dests

    def test_promoted_pawn_moves_like_gold(self):
        """A promoted pawn (tokin) should move like a gold general."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4+P4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Gold-like moves for Black: forward, forward-diag, sideways, straight back
        assert (3, 4) in dests  # forward
        assert (3, 3) in dests  # forward-left
        assert (3, 5) in dests  # forward-right
        assert (4, 3) in dests  # left
        assert (4, 5) in dests  # right
        assert (5, 4) in dests  # backward
        # NOT backward-diagonal
        assert (5, 3) not in dests
        assert (5, 5) not in dests

    def test_knight_cannot_move_backward(self):
        """A Black knight should not be able to jump backward."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4N4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        dests = _destinations_from(legal, 4, 4)
        # Backward jumps for Black would be row 6
        assert (6, 3) not in dests
        assert (6, 5) not in dests

    def test_pawn_cannot_move_backward(self):
        """A Black pawn should not be able to move backward (increasing row)."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4P4/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        assert not _move_exists(legal, 4, 4, 5, 4)
        assert not _move_exists(legal, 4, 4, 5, 3)
        assert not _move_exists(legal, 4, 4, 5, 5)


# ===========================================================================
# 6. Promotion Zone (~5 tests)
# ===========================================================================

class TestPromotionZone:
    """Verify promotion zone rules and mandatory/optional promotion."""

    def test_promotion_zone_black_rows_0_to_2(self):
        game = ShogiGame()
        for row in range(3):
            assert game.is_in_promotion_zone(row, Color.BLACK) is True
        for row in range(3, 9):
            assert game.is_in_promotion_zone(row, Color.BLACK) is False

    def test_promotion_zone_white_rows_6_to_8(self):
        game = ShogiGame()
        for row in range(6, 9):
            assert game.is_in_promotion_zone(row, Color.WHITE) is True
        for row in range(6):
            assert game.is_in_promotion_zone(row, Color.WHITE) is False

    def test_piece_entering_promotion_zone_can_promote(self):
        """A pawn moving into the promotion zone should have both promote=True
        and promote=False options (unless it MUST promote)."""
        # Black pawn at row 3 col 4, moving to row 2 (entering promotion zone)
        game = ShogiGame.from_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        # Move from (3,4) to (2,4) -- entering promotion zone
        has_promote_true = _move_exists(legal, 3, 4, 2, 4, promote=True)
        has_promote_false = _move_exists(legal, 3, 4, 2, 4, promote=False)
        assert has_promote_true, "Should be able to promote when entering promotion zone"
        assert has_promote_false, "Should also be able to decline promotion at row 2"

    def test_pawn_must_promote_on_last_rank(self):
        """A Black pawn reaching row 0 (last rank) must promote -- only promote=True."""
        # Black pawn at row 1 col 4, moving to row 0 (last rank)
        game = ShogiGame.from_sfen("4k4/4P4/9/9/9/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        # The pawn at (1,4) should only have promote=True for move to (0,4)
        has_promote_true = _move_exists(legal, 1, 4, 0, 4, promote=True)
        has_promote_false = _move_exists(legal, 1, 4, 0, 4, promote=False)
        assert has_promote_true, "Pawn must be able to promote on last rank"
        assert not has_promote_false, "Pawn must NOT have non-promote option on last rank"

    def test_lance_must_promote_on_last_rank(self):
        """A Black lance reaching row 0 (last rank) must promote -- only promote=True."""
        # Black lance at row 1, col 0. White king away from col 0.
        game = ShogiGame.from_sfen("4k4/L8/9/9/9/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()
        has_promote_true = _move_exists(legal, 1, 0, 0, 0, promote=True)
        has_promote_false = _move_exists(legal, 1, 0, 0, 0, promote=False)
        assert has_promote_true, "Lance must be able to promote on last rank"
        assert not has_promote_false, "Lance must NOT have non-promote option on last rank"


# ===========================================================================
# 7. Board Boundaries (~2 tests)
# ===========================================================================

class TestBoardBoundaries:
    """Verify is_on_board for valid and invalid coordinates."""

    def test_valid_coordinates(self):
        game = ShogiGame()
        for r in range(9):
            for c in range(9):
                assert game.is_on_board(r, c) is True

    def test_invalid_coordinates(self):
        game = ShogiGame()
        assert game.is_on_board(-1, 0) is False
        assert game.is_on_board(9, 0) is False
        assert game.is_on_board(0, -1) is False
        assert game.is_on_board(0, 9) is False
        assert game.is_on_board(-1, -1) is False
        assert game.is_on_board(9, 9) is False
