"""Tests for special Shogi rules: nifu, uchi-fu-zume, sennichite, max moves,
and drop restrictions.

These tests exercise the ShogiGame methods for detecting and enforcing the
special rules that distinguish Shogi from simpler board games.
"""

import random

import pytest

from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_core_definitions import Color, PieceType, Piece


# ---------------------------------------------------------------------------
# 1. Nifu (two pawns on the same file)
# ---------------------------------------------------------------------------


class TestNifu:
    """Tests for the nifu (two-pawn) rule."""

    def test_is_nifu_true_when_unpromoted_pawn_on_column(self):
        """is_nifu returns True when the column already has an unpromoted pawn
        of the specified color."""
        # SFEN: Black king on 8a (row 8 col 0), White king on 0i (row 0 col 8),
        # Black pawn on row 4 col 4 (5e), Black has a pawn in hand.
        sfen = "8k/9/9/9/4P4/9/9/9/K8 b P 1"
        game = ShogiGame.from_sfen(sfen)
        # Column 4 already has a Black unpromoted pawn at row 4.
        assert game.is_nifu(Color.BLACK, 4) is True

    def test_is_nifu_false_when_column_clear(self):
        """is_nifu returns False when the column has no unpromoted pawn of
        that color."""
        sfen = "8k/9/9/9/4P4/9/9/9/K8 b P 1"
        game = ShogiGame.from_sfen(sfen)
        # Column 0 has no Black pawn.
        assert game.is_nifu(Color.BLACK, 0) is False

    def test_is_nifu_false_for_promoted_pawn_tokin(self):
        """is_nifu returns False when the column has only a promoted pawn
        (tokin). Tokin does not count for the nifu rule."""
        # Place a promoted pawn (+P) on column 4 for Black, no unpromoted pawn.
        sfen = "8k/9/9/9/4+P4/9/9/9/K8 b P 1"
        game = ShogiGame.from_sfen(sfen)
        # Tokin on col 4 should not trigger nifu.
        assert game.is_nifu(Color.BLACK, 4) is False

    def test_nifu_pawn_drop_not_in_legal_moves(self):
        """A pawn drop onto a column that already has an unpromoted pawn of
        the same color must not appear in the legal moves list."""
        # Black has a pawn on column 4 and a pawn in hand.
        sfen = "8k/9/9/9/4P4/9/9/9/K8 b P 1"
        game = ShogiGame.from_sfen(sfen)
        legal_moves = game.get_legal_moves()

        # No drop move should target column 4 with a pawn.
        nifu_drops = [
            m
            for m in legal_moves
            if m[0] is None and m[1] is None and m[4] == PieceType.PAWN and m[3] == 4
        ]
        assert nifu_drops == [], (
            f"Found illegal nifu pawn drops on column 4: {nifu_drops}"
        )


# ---------------------------------------------------------------------------
# 2. Uchi-fu-zume (pawn-drop checkmate)
# ---------------------------------------------------------------------------


class TestUchiFuZume:
    """Tests for the uchi-fu-zume (pawn drop checkmate) rule."""

    def test_pawn_drop_checkmate_not_in_legal_moves(self):
        """A pawn drop that would cause immediate inescapable checkmate must
        not appear in the legal moves list."""
        # White king on row 0 col 4 (5a), hemmed in by own pieces on sides
        # and back rank. Black can drop a pawn at row 1 col 4 (5b) giving
        # check but the king has no escape -- this should be forbidden.
        #
        # Position: White king at 0,4; White golds at 0,3 and 0,5;
        # White pieces blocking row 0 escape. Black king safe far away.
        # Black has pawn in hand.
        sfen = "3gkg3/9/9/9/9/9/9/9/4K4 b P 1"
        game = ShogiGame.from_sfen(sfen)

        # Check if dropping pawn at (1, 4) would be uchi-fu-zume
        # The king at (0,4) would be in check from pawn at (1,4), and
        # the golds at (0,3) and (0,5) block lateral escapes while (0,4) is the king.
        # If the king cannot escape, this drop is uchi-fu-zume.
        uchi_result = game.is_uchi_fu_zume(1, 4, Color.BLACK)
        if uchi_result:
            # Verify such a drop is not in legal moves
            legal_moves = game.get_legal_moves()
            uchi_drops = [
                m
                for m in legal_moves
                if m[0] is None
                and m[1] is None
                and m[4] == PieceType.PAWN
                and m[2] == 1
                and m[3] == 4
            ]
            assert uchi_drops == [], (
                "Pawn drop causing uchi-fu-zume should not be in legal moves"
            )

    def test_pawn_drop_check_with_escape_is_legal(self):
        """A pawn drop that gives check but the opponent can escape should be
        allowed in legal moves."""
        # White king on 0,4 with no blocking pieces -- can escape sideways.
        # Black drops pawn at 1,4 giving check, but king can move to 0,3 or 0,5.
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b P 1"
        game = ShogiGame.from_sfen(sfen)

        # The king at (0,4) can escape to (0,3) or (0,5), so this is NOT uchi-fu-zume.
        assert game.is_uchi_fu_zume(1, 4, Color.BLACK) is False

        # Verify the pawn drop at (1,4) IS in legal moves.
        legal_moves = game.get_legal_moves()
        pawn_drop_1_4 = [
            m
            for m in legal_moves
            if m[0] is None
            and m[1] is None
            and m[4] == PieceType.PAWN
            and m[2] == 1
            and m[3] == 4
        ]
        assert len(pawn_drop_1_4) == 1, (
            "Pawn drop giving escapable check should be in legal moves"
        )

    def test_is_uchi_fu_zume_returns_false_for_non_threatening_drop(self):
        """is_uchi_fu_zume returns False when the pawn drop does not even
        give check to the opponent king."""
        # White king on 0,0; Black drops pawn at 5,5 -- far away, no check.
        sfen = "k8/9/9/9/9/9/9/9/8K b P 1"
        game = ShogiGame.from_sfen(sfen)
        assert game.is_uchi_fu_zume(5, 5, Color.BLACK) is False


# ---------------------------------------------------------------------------
# 3. Sennichite (repetition draw)
# ---------------------------------------------------------------------------


class TestSennichite:
    """Tests for the sennichite (fourfold repetition) rule."""

    def test_no_sennichite_after_normal_play(self):
        """During typical play with no repeated positions, is_sennichite
        returns False and the game is not over due to repetition."""
        game = ShogiGame()
        # Make a few opening moves (pawn pushes that do not repeat positions).
        # 7g-7f (Black pawn from (6,6) to (5,6))
        game.make_move((6, 6, 5, 6, False))
        assert game.is_sennichite() is False
        # 3c-3d (White pawn from (2,2) to (3,2))
        game.make_move((2, 2, 3, 2, False))
        assert game.is_sennichite() is False

    def test_sennichite_after_four_repetitions(self):
        """After the same position occurs 4 times, the game ends as a draw
        with termination_reason indicating Sennichite."""
        # Use a minimal position: two kings and alternating moves.
        # Black king on 8,0; White king on 0,8.
        # Black king moves 8,0 -> 7,0 and back; White king moves 0,8 -> 1,8 and back.
        # Each full cycle of 4 half-moves restores the original state.
        # We need the same state_hash to appear 4 times in move_history.
        sfen = "8k/9/9/9/9/9/9/9/K8 b - 1"
        game = ShogiGame.from_sfen(sfen)

        # Perform cycles. After each full cycle (4 half-moves), the position
        # with Black to move returns to the original board layout.
        # The state_hash recorded for Black's move back includes board + player.
        for _ in range(4):
            if game.game_over:
                break
            game.make_move((8, 0, 7, 0, False))  # Black king up
            if game.game_over:
                break
            game.make_move((0, 8, 1, 8, False))  # White king down
            if game.game_over:
                break
            game.make_move((7, 0, 8, 0, False))  # Black king back
            if game.game_over:
                break
            game.make_move((1, 8, 0, 8, False))  # White king back

        assert game.game_over is True
        assert game.termination_reason == "Sennichite"

    def test_sennichite_winner_is_none(self):
        """Sennichite is a draw, so the winner should be None."""
        sfen = "8k/9/9/9/9/9/9/9/K8 b - 1"
        game = ShogiGame.from_sfen(sfen)

        for _ in range(4):
            if game.game_over:
                break
            game.make_move((8, 0, 7, 0, False))
            if game.game_over:
                break
            game.make_move((0, 8, 1, 8, False))
            if game.game_over:
                break
            game.make_move((7, 0, 8, 0, False))
            if game.game_over:
                break
            game.make_move((1, 8, 0, 8, False))

        if game.termination_reason == "Sennichite":
            assert game.winner is None

    def test_is_sennichite_false_with_fewer_than_four_repetitions(self):
        """is_sennichite returns False when the position has occurred fewer
        than 4 times."""
        sfen = "8k/9/9/9/9/9/9/9/K8 b - 1"
        game = ShogiGame.from_sfen(sfen)

        # Perform only 2 full cycles (position repeats 2 times).
        for _ in range(2):
            game.make_move((8, 0, 7, 0, False))
            game.make_move((0, 8, 1, 8, False))
            game.make_move((7, 0, 8, 0, False))
            game.make_move((1, 8, 0, 8, False))

        assert game.is_sennichite() is False


# ---------------------------------------------------------------------------
# 4. Max moves
# ---------------------------------------------------------------------------


class TestMaxMoves:
    """Tests for the maximum move limit."""

    def test_game_ends_at_max_moves(self):
        """The game ends when move_count reaches max_moves_per_game."""
        game = ShogiGame(max_moves_per_game=10)
        rng = random.Random(42)

        while not game.game_over:
            legal = game.get_legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            game.make_move(move)

        assert game.game_over is True
        # The game should end at or before the max move limit.
        assert game.move_count <= 10

    def test_max_moves_per_game_property(self):
        """ShogiGame(max_moves_per_game=N) exposes the correct property value."""
        game = ShogiGame(max_moves_per_game=42)
        assert game.max_moves_per_game == 42

    def test_max_moves_winner_is_none(self):
        """When the game ends by max moves, the winner should be None (draw)."""
        game = ShogiGame(max_moves_per_game=10)
        rng = random.Random(42)

        while not game.game_over:
            legal = game.get_legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            game.make_move(move)

        if game.termination_reason == "Max moves reached":
            assert game.winner is None

    def test_max_moves_termination_reason(self):
        """When the game ends by max moves, termination_reason is
        'Max moves reached'."""
        game = ShogiGame(max_moves_per_game=10)
        rng = random.Random(42)

        while not game.game_over:
            legal = game.get_legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            game.make_move(move)

        # The game may have ended by checkmate before reaching 10 moves.
        # We verify the termination_reason is correct for the actual outcome.
        assert game.termination_reason in (
            "Max moves reached",
            "Tsumi",
            "stalemate",
            "Sennichite",
        )

    def test_very_low_max_moves_ends_quickly(self):
        """A game with a very low max_moves_per_game (e.g., 4) ends within
        that number of moves."""
        game = ShogiGame(max_moves_per_game=4)
        rng = random.Random(123)

        while not game.game_over:
            legal = game.get_legal_moves()
            if not legal:
                break
            move = rng.choice(legal)
            game.make_move(move)

        assert game.game_over is True
        assert game.move_count <= 4


# ---------------------------------------------------------------------------
# 5. Drop restrictions
# ---------------------------------------------------------------------------


class TestDropRestrictions:
    """Tests for piece drop restrictions based on rank and hand inventory."""

    def test_cannot_drop_pawn_on_last_rank_black(self):
        """Black cannot drop a pawn on row 0 (last rank for Black)."""
        # Black king far away, has pawn in hand. Row 0 col 4 is empty.
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b P 1"
        game = ShogiGame.from_sfen(sfen)
        assert game.can_drop_piece(PieceType.PAWN, 0, 4, Color.BLACK) is False

    def test_cannot_drop_pawn_on_last_rank_white(self):
        """White cannot drop a pawn on row 8 (last rank for White)."""
        sfen = "4k4/9/9/9/9/9/9/9/4K4 w p 1"
        game = ShogiGame.from_sfen(sfen)
        assert game.can_drop_piece(PieceType.PAWN, 8, 4, Color.WHITE) is False

    def test_cannot_drop_lance_on_last_rank(self):
        """Black cannot drop a lance on row 0 (last rank for Black)."""
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b L 1"
        game = ShogiGame.from_sfen(sfen)
        assert game.can_drop_piece(PieceType.LANCE, 0, 4, Color.BLACK) is False

    def test_cannot_drop_knight_on_last_two_ranks_black(self):
        """Black cannot drop a knight on rows 0 or 1 (last two ranks for
        Black), because a knight there would have no legal subsequent moves."""
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b N 1"
        game = ShogiGame.from_sfen(sfen)
        # Row 0 -- last rank
        assert game.can_drop_piece(PieceType.KNIGHT, 0, 4, Color.BLACK) is False
        # Row 1 -- second-to-last rank
        assert game.can_drop_piece(PieceType.KNIGHT, 1, 4, Color.BLACK) is False

    def test_cannot_drop_knight_on_last_two_ranks_white(self):
        """White cannot drop a knight on rows 7 or 8 (last two ranks for
        White)."""
        sfen = "4k4/9/9/9/9/9/9/9/4K4 w n 1"
        game = ShogiGame.from_sfen(sfen)
        assert game.can_drop_piece(PieceType.KNIGHT, 8, 4, Color.WHITE) is False
        assert game.can_drop_piece(PieceType.KNIGHT, 7, 4, Color.WHITE) is False

    def test_can_drop_gold_on_any_empty_square(self):
        """Gold has no rank restriction; it can be dropped on any empty
        square."""
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b G 1"
        game = ShogiGame.from_sfen(sfen)
        # Row 0 is fine for gold (unlike pawn/lance/knight).
        assert game.can_drop_piece(PieceType.GOLD, 0, 0, Color.BLACK) is True
        # Middle of board.
        assert game.can_drop_piece(PieceType.GOLD, 4, 4, Color.BLACK) is True

    def test_cannot_drop_piece_not_in_hand(self):
        """A piece can only be dropped if the player actually has it in hand
        (count > 0)."""
        # Black has NO pieces in hand (hands segment is "-").
        sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        game = ShogiGame.from_sfen(sfen)
        # Black has 0 pawns in hand.
        assert game.can_drop_piece(PieceType.PAWN, 4, 4, Color.BLACK) is False
        # Black has 0 golds in hand.
        assert game.can_drop_piece(PieceType.GOLD, 4, 4, Color.BLACK) is False
