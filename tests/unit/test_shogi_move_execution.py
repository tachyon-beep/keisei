"""Tests for keisei.shogi.shogi_move_execution module.

Covers the three main functions:
  - apply_move_to_board_state: board moves, drops, captures, promotions
  - apply_move_to_game: player switching, move count, simulation mode
  - revert_last_applied_move: full state restoration
"""

import copy

import pytest

from keisei.shogi.shogi_core_definitions import (
    Color,
    MoveApplicationResult,
    MoveTuple,
    Piece,
    PieceType,
    get_unpromoted_types,
)
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_move_execution import (
    apply_move_to_board_state,
    apply_move_to_game,
    revert_last_applied_move,
)


@pytest.fixture
def new_game() -> ShogiGame:
    """A freshly initialized ShogiGame in the standard starting position."""
    return ShogiGame()


# ---------------------------------------------------------------------------
# Board move basics
# ---------------------------------------------------------------------------


class TestBoardMoveBasics:
    """Basic board-to-board move mechanics."""

    def test_move_black_pawn_forward(self, new_game: ShogiGame) -> None:
        """Moving Black pawn from (6,0) to (5,0) clears the source and places
        the pawn at the destination."""
        move: MoveTuple = (6, 0, 5, 0, False)
        result = apply_move_to_board_state(
            new_game.board, new_game.hands, move, Color.BLACK
        )

        assert new_game.board[6][0] is None, "Source square should be empty"
        dest_piece = new_game.board[5][0]
        assert dest_piece is not None
        assert dest_piece.type == PieceType.PAWN
        assert dest_piece.color == Color.BLACK

    def test_move_result_no_capture_no_promotion(self, new_game: ShogiGame) -> None:
        """A simple pawn advance returns no capture and no promotion."""
        move: MoveTuple = (6, 0, 5, 0, False)
        result = apply_move_to_board_state(
            new_game.board, new_game.hands, move, Color.BLACK
        )

        assert result.captured_piece_type is None
        assert result.was_promotion is False

    def test_move_white_pawn_forward(self, new_game: ShogiGame) -> None:
        """White pawn at (2,0) moves forward to (3,0) (increasing row index)."""
        move: MoveTuple = (2, 0, 3, 0, False)
        result = apply_move_to_board_state(
            new_game.board, new_game.hands, move, Color.WHITE
        )

        assert new_game.board[2][0] is None
        dest_piece = new_game.board[3][0]
        assert dest_piece is not None
        assert dest_piece.type == PieceType.PAWN
        assert dest_piece.color == Color.WHITE

    def test_move_rook_across_board(self) -> None:
        """Move Black rook from (7,7) along a column on a mostly empty board."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/7R1/4K4 b - 1")
        move: MoveTuple = (7, 7, 0, 7, False)  # Rook to top of board
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        assert game.board[7][7] is None
        assert game.board[0][7] is not None
        assert game.board[0][7].type == PieceType.ROOK
        assert result.captured_piece_type is None

    def test_capture_opponent_piece(self) -> None:
        """Capturing an opponent piece adds the unpromoted type to the
        capturing player's hand."""
        # Black pawn at (4,4), White pawn at (3,4)
        game = ShogiGame.from_sfen("4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1")
        move: MoveTuple = (4, 4, 3, 4, False)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        assert result.captured_piece_type == PieceType.PAWN
        assert game.board[4][4] is None
        dest = game.board[3][4]
        assert dest is not None
        assert dest.color == Color.BLACK
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 1


# ---------------------------------------------------------------------------
# Drop moves
# ---------------------------------------------------------------------------


class TestDropMoves:
    """Dropping pieces from hand onto the board."""

    def test_drop_pawn_from_hand(self) -> None:
        """Drop a Pawn from Black's hand onto an empty square."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 1

        move: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        assert game.board[4][4] is not None
        assert game.board[4][4].type == PieceType.PAWN
        assert game.board[4][4].color == Color.BLACK
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 0
        assert result.captured_piece_type is None
        assert result.was_promotion is False

    def test_drop_rook_from_hand(self) -> None:
        """Drop a Rook from White's hand."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 w r 1")
        assert game.hands[Color.WHITE.value][PieceType.ROOK] == 1

        move: MoveTuple = (None, None, 5, 5, PieceType.ROOK)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.WHITE
        )

        assert game.board[5][5] is not None
        assert game.board[5][5].type == PieceType.ROOK
        assert game.board[5][5].color == Color.WHITE
        assert game.hands[Color.WHITE.value][PieceType.ROOK] == 0

    def test_drop_decrements_hand_count_multiple(self) -> None:
        """When hand has multiple pieces, dropping one decrements the count by
        exactly one."""
        # "2P" means Black has 2 Pawns in hand
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b 2P 1")
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 2

        move: MoveTuple = (None, None, 5, 0, PieceType.PAWN)
        apply_move_to_board_state(game.board, game.hands, move, Color.BLACK)

        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 1

    def test_drop_gold_from_hand(self) -> None:
        """Drop a Gold from Black's hand."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b G 1")
        move: MoveTuple = (None, None, 6, 4, PieceType.GOLD)
        apply_move_to_board_state(game.board, game.hands, move, Color.BLACK)

        placed = game.board[6][4]
        assert placed is not None
        assert placed.type == PieceType.GOLD
        assert placed.color == Color.BLACK
        assert game.hands[Color.BLACK.value][PieceType.GOLD] == 0


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


class TestPromotion:
    """Promotion flag handling during board moves."""

    def test_promote_pawn_entering_promotion_zone(self) -> None:
        """A Black pawn moving into the promotion zone (row < 3) with
        promote=True becomes PROMOTED_PAWN."""
        # Black pawn at row 3, moving to row 2 (entering promotion zone)
        game = ShogiGame.from_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        move: MoveTuple = (3, 4, 2, 4, True)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        assert result.was_promotion is True
        dest = game.board[2][4]
        assert dest is not None
        assert dest.type == PieceType.PROMOTED_PAWN
        assert dest.color == Color.BLACK

    def test_decline_promotion(self) -> None:
        """A pawn entering the promotion zone with promote=False stays
        unpromoted."""
        game = ShogiGame.from_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        move: MoveTuple = (3, 4, 2, 4, False)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        assert result.was_promotion is False
        dest = game.board[2][4]
        assert dest is not None
        assert dest.type == PieceType.PAWN

    def test_promote_silver(self) -> None:
        """Silver general promotes to PROMOTED_SILVER."""
        game = ShogiGame.from_sfen("4k4/9/9/4S4/9/9/9/9/4K4 b - 1")
        move: MoveTuple = (3, 4, 2, 4, True)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        assert result.was_promotion is True
        assert game.board[2][4].type == PieceType.PROMOTED_SILVER

    def test_promote_rook_to_dragon(self) -> None:
        """Rook promotes to PROMOTED_ROOK (Dragon)."""
        game = ShogiGame.from_sfen("4k4/9/9/4R4/9/9/9/9/4K4 b - 1")
        move: MoveTuple = (3, 4, 0, 4, True)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        assert result.was_promotion is True
        assert game.board[0][4].type == PieceType.PROMOTED_ROOK


# ---------------------------------------------------------------------------
# Capture with demotion (promoted piece captured goes to hand as base type)
# ---------------------------------------------------------------------------


class TestCaptureDemotion:
    """Capturing promoted pieces demotes them when added to the hand."""

    def test_capture_promoted_rook_gets_rook_in_hand(self) -> None:
        """Capturing a PROMOTED_ROOK gives the capturing player an unpromoted
        ROOK in hand, not a PROMOTED_ROOK."""
        # Black gold at (4,4), White promoted rook at (3,4)
        game = ShogiGame.from_sfen("4k4/9/9/4+r4/4G4/9/9/9/4K4 b - 1")
        move: MoveTuple = (4, 4, 3, 4, False)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        # The captured piece type returned should be the BASE type
        assert result.captured_piece_type == PieceType.ROOK
        assert game.hands[Color.BLACK.value][PieceType.ROOK] == 1

    def test_capture_promoted_pawn_gets_pawn_in_hand(self) -> None:
        """Capturing a PROMOTED_PAWN (Tokin) gives the base PAWN in hand."""
        # Black rook at (5,0), White tokin at (4,0)
        game = ShogiGame.from_sfen("4k4/9/9/9/+p8/R8/9/9/4K4 b - 1")
        move: MoveTuple = (5, 0, 4, 0, False)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.BLACK
        )

        assert result.captured_piece_type == PieceType.PAWN
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 1

    def test_capture_promoted_bishop_gets_bishop_in_hand(self) -> None:
        """Capturing a PROMOTED_BISHOP (Horse) gives the base BISHOP in hand."""
        # White lance at (5,0), Black promoted bishop at (6,0)
        game = ShogiGame.from_sfen("4k4/9/9/9/9/l8/+B8/9/4K4 w - 1")
        move: MoveTuple = (5, 0, 6, 0, False)
        result = apply_move_to_board_state(
            game.board, game.hands, move, Color.WHITE
        )

        assert result.captured_piece_type == PieceType.BISHOP
        assert game.hands[Color.WHITE.value][PieceType.BISHOP] == 1


# ---------------------------------------------------------------------------
# apply_move_to_game integration
# ---------------------------------------------------------------------------


class TestApplyMoveToGame:
    """Tests for apply_move_to_game (player switching, move count)."""

    def test_switches_current_player_black_to_white(
        self, new_game: ShogiGame
    ) -> None:
        """After apply_move_to_game, Black's turn becomes White's turn."""
        assert new_game.current_player == Color.BLACK
        apply_move_to_game(new_game)
        assert new_game.current_player == Color.WHITE

    def test_switches_current_player_white_to_black(self) -> None:
        """After apply_move_to_game on White's turn, it becomes Black's turn."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 w - 1")
        assert game.current_player == Color.WHITE
        apply_move_to_game(game)
        assert game.current_player == Color.BLACK

    def test_increments_move_count_normal(self, new_game: ShogiGame) -> None:
        """In normal (non-simulation) mode, move_count is incremented."""
        initial_count = new_game.move_count
        apply_move_to_game(new_game, is_simulation=False)
        assert new_game.move_count == initial_count + 1

    def test_does_not_increment_move_count_simulation(
        self, new_game: ShogiGame
    ) -> None:
        """In simulation mode, move_count is NOT incremented."""
        initial_count = new_game.move_count
        apply_move_to_game(new_game, is_simulation=True)
        assert new_game.move_count == initial_count


# ---------------------------------------------------------------------------
# revert_last_applied_move
# ---------------------------------------------------------------------------


class TestRevertLastAppliedMove:
    """Tests for reverting moves and restoring game state."""

    def test_revert_simple_pawn_move(self, new_game: ShogiGame) -> None:
        """After reverting a simple pawn move, the board is identical to the
        original state."""
        original_board = [row[:] for row in new_game.board]
        original_hands = {k: v.copy() for k, v in new_game.hands.items()}
        original_player = new_game.current_player
        original_count = new_game.move_count

        # Apply a move
        move: MoveTuple = (6, 0, 5, 0, False)
        apply_move_to_board_state(
            new_game.board, new_game.hands, move, Color.BLACK
        )
        apply_move_to_game(new_game)

        # Verify the move was applied
        assert new_game.board[5][0] is not None
        assert new_game.current_player == Color.WHITE

        # Revert
        revert_last_applied_move(
            new_game, original_board, original_hands, original_player, original_count
        )

        assert new_game.board[6][0] is not None
        assert new_game.board[6][0].type == PieceType.PAWN
        assert new_game.board[5][0] is None
        assert new_game.current_player == Color.BLACK
        assert new_game.move_count == original_count

    def test_revert_capture_restores_captured_piece(self) -> None:
        """After reverting a capture, the captured piece reappears and the
        hand count reverts."""
        game = ShogiGame.from_sfen("4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1")
        original_board = [row[:] for row in game.board]
        original_hands = {k: v.copy() for k, v in game.hands.items()}
        original_player = game.current_player
        original_count = game.move_count

        # Black pawn captures White pawn
        move: MoveTuple = (4, 4, 3, 4, False)
        apply_move_to_board_state(game.board, game.hands, move, Color.BLACK)
        apply_move_to_game(game)

        # Verify capture happened
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 1

        # Revert
        revert_last_applied_move(
            game, original_board, original_hands, original_player, original_count
        )

        # White pawn should be back at (3,4)
        assert game.board[3][4] is not None
        assert game.board[3][4].color == Color.WHITE
        assert game.board[3][4].type == PieceType.PAWN
        # Black pawn should be back at (4,4)
        assert game.board[4][4] is not None
        assert game.board[4][4].color == Color.BLACK
        # Hand should be empty again
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 0

    def test_revert_promotion_restores_unpromoted_piece(self) -> None:
        """After reverting a promotion, the piece type is restored to the
        unpromoted version."""
        game = ShogiGame.from_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        original_board = [row[:] for row in game.board]
        original_hands = {k: v.copy() for k, v in game.hands.items()}
        original_player = game.current_player
        original_count = game.move_count

        # Promote pawn
        move: MoveTuple = (3, 4, 2, 4, True)
        apply_move_to_board_state(game.board, game.hands, move, Color.BLACK)
        apply_move_to_game(game)

        # Verify promotion
        assert game.board[2][4].type == PieceType.PROMOTED_PAWN

        # Revert
        revert_last_applied_move(
            game, original_board, original_hands, original_player, original_count
        )

        # Piece should be back at (3,4) as an unpromoted pawn
        assert game.board[3][4] is not None
        assert game.board[3][4].type == PieceType.PAWN
        assert game.board[2][4] is None

    def test_revert_resets_game_over_state(self) -> None:
        """Reverting resets game_over, winner, and termination_reason."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")

        original_board = [row[:] for row in game.board]
        original_hands = {k: v.copy() for k, v in game.hands.items()}
        original_player = game.current_player
        original_count = game.move_count

        # Manually set game-over state to simulate the scenario
        game.game_over = True
        game.winner = Color.BLACK
        game.termination_reason = "Tsumi"

        revert_last_applied_move(
            game, original_board, original_hands, original_player, original_count
        )

        assert game.game_over is False
        assert game.winner is None
        assert game.termination_reason is None

    def test_revert_drop_restores_hand_and_clears_board(self) -> None:
        """After reverting a drop, the piece returns to hand and the board
        square is cleared."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        original_board = [row[:] for row in game.board]
        original_hands = {k: v.copy() for k, v in game.hands.items()}
        original_player = game.current_player
        original_count = game.move_count

        # Drop pawn
        move: MoveTuple = (None, None, 5, 4, PieceType.PAWN)
        apply_move_to_board_state(game.board, game.hands, move, Color.BLACK)
        apply_move_to_game(game)

        # Verify drop
        assert game.board[5][4] is not None
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 0

        # Revert
        revert_last_applied_move(
            game, original_board, original_hands, original_player, original_count
        )

        # Board square should be empty again, hand should have the pawn back
        assert game.board[5][4] is None
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 1
        assert game.current_player == Color.BLACK


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Edge cases and error paths in apply_move_to_board_state."""

    def test_error_no_piece_at_source(self) -> None:
        """Raises ValueError if no piece exists at the source square."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        move: MoveTuple = (5, 5, 4, 5, False)
        with pytest.raises(ValueError, match="No piece at source square"):
            apply_move_to_board_state(
                game.board, game.hands, move, Color.BLACK
            )

    def test_error_move_opponents_piece(self) -> None:
        """Raises ValueError if trying to move an opponent's piece."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/4p4/9/4K4 b - 1")
        # White pawn at (6,4), but current player is Black
        move: MoveTuple = (6, 4, 5, 4, False)
        with pytest.raises(ValueError, match="belongs to"):
            apply_move_to_board_state(
                game.board, game.hands, move, Color.BLACK
            )

    def test_error_capture_own_piece(self) -> None:
        """Raises ValueError when trying to capture one's own piece."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4P4/4P4/9/9/4K4 b - 1")
        # Black pawn at (5,4) trying to move to (4,4) where another Black pawn sits
        move: MoveTuple = (5, 4, 4, 4, False)
        with pytest.raises(ValueError, match="Cannot capture own piece"):
            apply_move_to_board_state(
                game.board, game.hands, move, Color.BLACK
            )

    def test_error_drop_piece_type_missing_from_hand_dict(self) -> None:
        """Raises ValueError if the piece type key is completely absent from
        the hand dictionary (not merely zero-count)."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b - 1")
        # Remove PAWN key entirely from Black's hand dict to trigger the check
        del game.hands[Color.BLACK.value][PieceType.PAWN]
        move: MoveTuple = (None, None, 5, 5, PieceType.PAWN)
        with pytest.raises(ValueError, match="not in hand"):
            apply_move_to_board_state(
                game.board, game.hands, move, Color.BLACK
            )
