"""Tests for move execution, captures, promotions, drops, and undo using real ShogiGame objects.

Covers five categories:
1. Basic move execution
2. Captures
3. Promotions
4. Drops
5. Undo
"""

import numpy as np
import pytest

from keisei.shogi.shogi_core_definitions import Color, MoveTuple, Piece, PieceType
from keisei.shogi.shogi_game import ShogiGame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def new_game() -> ShogiGame:
    """A fresh ShogiGame in the standard initial position."""
    return ShogiGame()


# ---------------------------------------------------------------------------
# 1. Basic move execution
# ---------------------------------------------------------------------------


class TestBasicMoveExecution:
    """Tests for fundamental move mechanics on the initial board."""

    def test_pawn_advance_changes_board_state(self, new_game: ShogiGame):
        """Moving a pawn forward removes it from source and places it at dest."""
        # Black's pawn at (6, 0) should move to (5, 0).
        move: MoveTuple = (6, 0, 5, 0, False)
        legal = new_game.get_legal_moves()
        assert move in legal, "Pawn advance should be a legal opening move"

        new_game.make_move(move)

        assert new_game.get_piece(6, 0) is None, "Source square should be empty after move"
        dest_piece = new_game.get_piece(5, 0)
        assert dest_piece is not None, "Destination square should contain a piece"
        assert dest_piece.type == PieceType.PAWN
        assert dest_piece.color == Color.BLACK

    def test_move_increments_move_count(self, new_game: ShogiGame):
        """Each make_move call should increment move_count by 1."""
        assert new_game.move_count == 0
        move: MoveTuple = (6, 4, 5, 4, False)  # Black pawn e3-e4
        new_game.make_move(move)
        assert new_game.move_count == 1

    def test_move_switches_current_player(self, new_game: ShogiGame):
        """After Black moves, current_player should switch to White."""
        assert new_game.current_player == Color.BLACK
        move: MoveTuple = (6, 4, 5, 4, False)
        new_game.make_move(move)
        assert new_game.current_player == Color.WHITE

    def test_make_move_returns_correct_tuple_types(self, new_game: ShogiGame):
        """make_move should return (obs, reward, done, info) with correct types."""
        move: MoveTuple = (6, 4, 5, 4, False)
        result = new_game.make_move(move)

        assert isinstance(result, tuple), "make_move should return a tuple"
        assert len(result) == 4, "Return tuple should have 4 elements"
        obs, reward, done, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_observation_shape(self, new_game: ShogiGame):
        """Observation returned by make_move should be (46, 9, 9)."""
        move: MoveTuple = (6, 4, 5, 4, False)
        obs, _, _, _ = new_game.make_move(move)
        assert obs.shape == (46, 9, 9), f"Expected (46, 9, 9), got {obs.shape}"


# ---------------------------------------------------------------------------
# 2. Captures
# ---------------------------------------------------------------------------


class TestCaptures:
    """Tests for piece captures and hand management."""

    def test_capture_removes_piece_and_adds_to_hand(self):
        """Capturing an opponent piece removes it from the board and adds
        the base type to the capturer's hand."""
        # SFEN: Black rook at (5,4), White pawn at (4,4), kings present.
        game = ShogiGame.from_sfen("4k4/9/9/9/4p4/4R4/9/9/4K4 b - 1")
        capture_move: MoveTuple = (5, 4, 4, 4, False)
        legal = game.get_legal_moves()
        assert capture_move in legal, "Rook should be able to capture the pawn"

        game.make_move(capture_move)

        # White pawn is gone from (4,4); Black rook is now there.
        piece = game.get_piece(4, 4)
        assert piece is not None
        assert piece.type == PieceType.ROOK
        assert piece.color == Color.BLACK

        hand = game.get_pieces_in_hand(Color.BLACK)
        assert hand[PieceType.PAWN] >= 1, "Captured pawn should be in Black's hand"

    def test_capture_promoted_piece_adds_unpromoted_to_hand(self):
        """Capturing a promoted piece adds the unpromoted base type to hand."""
        # White promoted pawn (tokin) at (4,4), Black rook at (5,4).
        game = ShogiGame.from_sfen("4k4/9/9/9/4+p4/4R4/9/9/4K4 b - 1")
        capture_move: MoveTuple = (5, 4, 4, 4, False)
        legal = game.get_legal_moves()
        assert capture_move in legal

        game.make_move(capture_move)

        hand = game.get_pieces_in_hand(Color.BLACK)
        assert hand[PieceType.PAWN] >= 1, (
            "Capturing a promoted pawn should add PAWN (not PROMOTED_PAWN) to hand"
        )

    def test_captured_piece_in_hand_via_get_pieces_in_hand(self):
        """get_pieces_in_hand reflects captured pieces correctly."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4p4/4R4/9/9/4K4 b - 1")
        hand_before = game.get_pieces_in_hand(Color.BLACK)
        pawn_count_before = hand_before.get(PieceType.PAWN, 0)

        game.make_move((5, 4, 4, 4, False))

        hand_after = game.get_pieces_in_hand(Color.BLACK)
        assert hand_after[PieceType.PAWN] == pawn_count_before + 1

    def test_multiple_captures_accumulate_in_hand(self):
        """Successive captures accumulate pieces in the capturer's hand."""
        # Black rook at (5,4). White pawns at (4,4) and (3,4). Kings present.
        game = ShogiGame.from_sfen("4k4/9/9/4p4/4p4/4R4/9/9/4K4 b - 1")

        # First capture: rook takes pawn at (4,4)
        game.make_move((5, 4, 4, 4, False))
        assert game.get_pieces_in_hand(Color.BLACK)[PieceType.PAWN] == 1

        # White must move -- move the king.
        white_legal = game.get_legal_moves()
        # Find a king move for White.
        white_king_move = None
        for m in white_legal:
            if m[0] is not None:
                piece = game.get_piece(m[0], m[1])
                if piece and piece.type == PieceType.KING:
                    white_king_move = m
                    break
        assert white_king_move is not None, "White king should have a legal move"
        game.make_move(white_king_move)

        # Second capture: rook takes pawn at (3,4)
        game.make_move((4, 4, 3, 4, False))
        assert game.get_pieces_in_hand(Color.BLACK)[PieceType.PAWN] == 2

    def test_rook_captures_pawn(self):
        """A rook can capture a pawn (testing capture with a major piece)."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4p4/4R4/9/9/4K4 b - 1")
        move: MoveTuple = (5, 4, 4, 4, False)
        legal = game.get_legal_moves()
        assert move in legal

        game.make_move(move)

        piece = game.get_piece(4, 4)
        assert piece is not None
        assert piece.type == PieceType.ROOK
        assert piece.color == Color.BLACK


# ---------------------------------------------------------------------------
# 3. Promotions
# ---------------------------------------------------------------------------


class TestPromotions:
    """Tests for piece promotion mechanics."""

    def test_pawn_promotes_when_entering_promotion_zone(self):
        """A pawn moving into the promotion zone with promote=True becomes
        PROMOTED_PAWN."""
        # Black pawn at (3,4) about to enter promotion zone (row 2).
        # Black king at (8,4), White king at (0,4).
        game = ShogiGame.from_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        promote_move: MoveTuple = (3, 4, 2, 4, True)
        legal = game.get_legal_moves()
        assert promote_move in legal, "Pawn should be able to promote entering zone"

        game.make_move(promote_move)

        piece = game.get_piece(2, 4)
        assert piece is not None
        assert piece.type == PieceType.PROMOTED_PAWN
        assert piece.color == Color.BLACK

    def test_pawn_declines_promotion_entering_zone(self):
        """A pawn entering promotion zone with promote=False stays as PAWN
        (as long as it is not on the last rank where promotion is forced)."""
        # Black pawn at (3,4) moving to (2,4) -- row 2 is promotion zone but
        # not last rank, so declining is allowed.
        game = ShogiGame.from_sfen("4k4/9/9/4P4/9/9/9/9/4K4 b - 1")
        no_promote_move: MoveTuple = (3, 4, 2, 4, False)
        legal = game.get_legal_moves()
        assert no_promote_move in legal, "Declining promotion should be legal on row 2"

        game.make_move(no_promote_move)

        piece = game.get_piece(2, 4)
        assert piece is not None
        assert piece.type == PieceType.PAWN

    def test_piece_leaving_promotion_zone_can_promote(self):
        """A piece already in the promotion zone can promote when moving
        within or out of the zone."""
        # Black silver at (2,4) (inside promotion zone for Black) moving to (3,3).
        # Moving OUT of the promotion zone -- promotion should be offered.
        game = ShogiGame.from_sfen("4k4/9/4S4/9/9/9/9/9/4K4 b - 1")
        promote_move: MoveTuple = (2, 4, 3, 3, True)
        legal = game.get_legal_moves()
        assert promote_move in legal, "Silver leaving promotion zone should be able to promote"

        game.make_move(promote_move)

        piece = game.get_piece(3, 3)
        assert piece is not None
        assert piece.type == PieceType.PROMOTED_SILVER

    def test_pawn_on_last_rank_must_promote(self):
        """A pawn reaching the last rank (row 0 for Black) must promote --
        only promote=True should appear in legal moves for that destination."""
        # Black pawn at (1,4) about to reach row 0 (last rank for Black).
        game = ShogiGame.from_sfen("4k4/4P4/9/9/9/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()

        # Look for all moves from (1,4) to (0,4).
        pawn_to_last = [m for m in legal if m[:4] == (1, 4, 0, 4)]
        assert len(pawn_to_last) >= 1, "Pawn should have at least one move to last rank"
        # Every such move must have promote=True.
        for m in pawn_to_last:
            assert m[4] is True, (
                f"Pawn on last rank must promote, but found move with promote={m[4]}"
            )

    def test_knight_on_last_two_ranks_must_promote(self):
        """A knight reaching the last two ranks (rows 0-1 for Black) must
        promote -- only promote=True should be in legal moves."""
        # Black knight at (3,4). Knight moves to (1,3) or (1,5) -- both in
        # the must-promote zone for Black.
        game = ShogiGame.from_sfen("4k4/9/9/4N4/9/9/9/9/4K4 b - 1")
        legal = game.get_legal_moves()

        # Knight at (3,4) can jump to (1,3) and (1,5).
        knight_moves_to_rank_1 = [
            m for m in legal if m[0] == 3 and m[1] == 4 and m[2] == 1
        ]
        assert len(knight_moves_to_rank_1) > 0, "Knight should have moves to rank 1"
        for m in knight_moves_to_rank_1:
            assert m[4] is True, (
                f"Knight on second-to-last rank must promote, got promote={m[4]}"
            )


# ---------------------------------------------------------------------------
# 4. Drops
# ---------------------------------------------------------------------------


class TestDrops:
    """Tests for piece drop mechanics."""

    def test_drop_places_piece_on_empty_square(self):
        """Dropping a piece from hand places it on the target empty square."""
        # Black has a pawn in hand, no pawns on column 4 for Black.
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        drop_move: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
        legal = game.get_legal_moves()
        assert drop_move in legal, "Drop of pawn in hand on empty square should be legal"

        game.make_move(drop_move)

        piece = game.get_piece(4, 4)
        assert piece is not None, "Dropped piece should appear on the board"
        assert piece.type == PieceType.PAWN
        assert piece.color == Color.BLACK

    def test_drop_decrements_hand_count(self):
        """Dropping a piece should decrement its count in the player's hand."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b P 1")
        hand_before = game.get_pieces_in_hand(Color.BLACK)
        assert hand_before[PieceType.PAWN] == 1

        drop_move: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
        game.make_move(drop_move)

        hand_after = game.get_pieces_in_hand(Color.BLACK)
        assert hand_after[PieceType.PAWN] == 0

    def test_drop_move_in_legal_moves_when_piece_in_hand(self):
        """A drop move tuple should appear in legal_moves when the piece is
        available in hand and the target square is empty and legal."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b G 1")
        legal = game.get_legal_moves()

        # Gold in hand -- should be droppable on many empty squares.
        gold_drops = [
            m for m in legal if m[0] is None and m[4] == PieceType.GOLD
        ]
        assert len(gold_drops) > 0, "Gold drops should be in legal moves"

        # Specific check: dropping on (4,4) should be legal.
        assert (None, None, 4, 4, PieceType.GOLD) in legal

    def test_cannot_drop_on_occupied_square(self):
        """A drop on an occupied square should not appear in legal moves."""
        # Black has a pawn in hand. Square (4,4) is occupied by a White pawn.
        game = ShogiGame.from_sfen("4k4/9/9/9/4p4/9/9/9/4K4 b P 1")
        legal = game.get_legal_moves()

        drop_on_occupied: MoveTuple = (None, None, 4, 4, PieceType.PAWN)
        assert drop_on_occupied not in legal, "Cannot drop on an occupied square"

    def test_dropped_piece_belongs_to_current_player(self):
        """A dropped piece should belong to the player who dropped it."""
        game = ShogiGame.from_sfen("4k4/9/9/9/9/9/9/9/4K4 b R 1")
        drop_move: MoveTuple = (None, None, 4, 4, PieceType.ROOK)
        game.make_move(drop_move)

        piece = game.get_piece(4, 4)
        assert piece is not None
        assert piece.color == Color.BLACK, "Dropped piece should belong to Black"


# ---------------------------------------------------------------------------
# 5. Undo
# ---------------------------------------------------------------------------


class TestUndo:
    """Tests for undo_move reverting game state."""

    def test_undo_restores_board_state(self, new_game: ShogiGame):
        """undo_move should restore the board to its state before the move."""
        original_piece = new_game.get_piece(6, 4)
        assert original_piece is not None

        move: MoveTuple = (6, 4, 5, 4, False)
        new_game.make_move(move)

        # Verify the move happened.
        assert new_game.get_piece(6, 4) is None
        assert new_game.get_piece(5, 4) is not None

        new_game.undo_move()

        # Board should be restored.
        restored_piece = new_game.get_piece(6, 4)
        assert restored_piece is not None
        assert restored_piece.type == PieceType.PAWN
        assert restored_piece.color == Color.BLACK
        assert new_game.get_piece(5, 4) is None

    def test_undo_restores_current_player(self, new_game: ShogiGame):
        """undo_move should restore current_player to the player who made
        the undone move."""
        assert new_game.current_player == Color.BLACK
        new_game.make_move((6, 4, 5, 4, False))
        assert new_game.current_player == Color.WHITE

        new_game.undo_move()
        assert new_game.current_player == Color.BLACK

    def test_undo_restores_move_count(self, new_game: ShogiGame):
        """undo_move should decrement move_count back to its prior value."""
        assert new_game.move_count == 0
        new_game.make_move((6, 4, 5, 4, False))
        assert new_game.move_count == 1

        new_game.undo_move()
        assert new_game.move_count == 0

    def test_undo_after_capture_restores_captured_piece(self):
        """After undoing a capture, the captured piece should be back on the
        board and removed from the capturer's hand."""
        game = ShogiGame.from_sfen("4k4/9/9/9/4p4/4R4/9/9/4K4 b - 1")

        # Verify the White pawn is at (4,4) before capture.
        target = game.get_piece(4, 4)
        assert target is not None
        assert target.type == PieceType.PAWN
        assert target.color == Color.WHITE

        game.make_move((5, 4, 4, 4, False))

        # After capture: rook at (4,4), pawn in Black's hand.
        assert game.get_piece(4, 4).type == PieceType.ROOK
        assert game.get_pieces_in_hand(Color.BLACK)[PieceType.PAWN] >= 1

        game.undo_move()

        # After undo: White pawn restored at (4,4), rook back at (5,4).
        restored_target = game.get_piece(4, 4)
        assert restored_target is not None
        assert restored_target.type == PieceType.PAWN
        assert restored_target.color == Color.WHITE

        rook = game.get_piece(5, 4)
        assert rook is not None
        assert rook.type == PieceType.ROOK
        assert rook.color == Color.BLACK

        hand = game.get_pieces_in_hand(Color.BLACK)
        assert hand[PieceType.PAWN] == 0, "Captured pawn should be removed from hand after undo"

    def test_multiple_undo_restores_to_original(self, new_game: ShogiGame):
        """Undoing all moves should restore the game to the initial position."""
        # Save the initial board string for comparison.
        initial_board_str = new_game.to_string()
        initial_move_count = new_game.move_count
        initial_player = new_game.current_player

        # Play two moves.
        new_game.make_move((6, 4, 5, 4, False))  # Black pawn
        new_game.make_move((2, 4, 3, 4, False))  # White pawn

        assert new_game.move_count == 2
        assert new_game.current_player == Color.BLACK

        # Undo both.
        new_game.undo_move()
        new_game.undo_move()

        assert new_game.to_string() == initial_board_str
        assert new_game.move_count == initial_move_count
        assert new_game.current_player == initial_player
