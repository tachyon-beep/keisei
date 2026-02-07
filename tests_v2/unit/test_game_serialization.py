"""Tests for SFEN roundtrip serialization and observation generation using real ShogiGame objects."""

import numpy as np
import pytest

from keisei.shogi.shogi_core_definitions import Color, Piece, PieceType
from keisei.shogi.shogi_game import ShogiGame

# The canonical initial SFEN for standard Shogi
INITIAL_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"


# ---------------------------------------------------------------------------
# SFEN roundtrip tests
# ---------------------------------------------------------------------------


class TestSFENRoundtrip:
    """Tests for SFEN serialization and deserialization."""

    def test_initial_position_to_sfen(self):
        """A freshly constructed game's SFEN matches the known initial position string."""
        game = ShogiGame()
        assert game.to_sfen() == INITIAL_SFEN

    def test_initial_position_to_sfen_string_alias(self):
        """to_sfen_string() returns the same result as to_sfen()."""
        game = ShogiGame()
        assert game.to_sfen_string() == game.to_sfen()

    def test_from_sfen_initial_position(self):
        """from_sfen with the initial SFEN produces a correct initial position."""
        game = ShogiGame.from_sfen(INITIAL_SFEN)
        assert game.current_player == Color.BLACK
        assert game.move_count == 0
        assert not game.game_over
        assert game.winner is None
        # Verify a few known pieces
        assert game.get_piece(0, 4) == Piece(PieceType.KING, Color.WHITE)
        assert game.get_piece(8, 4) == Piece(PieceType.KING, Color.BLACK)
        assert game.get_piece(6, 0) == Piece(PieceType.PAWN, Color.BLACK)
        assert game.get_piece(2, 0) == Piece(PieceType.PAWN, Color.WHITE)
        assert game.get_piece(7, 7) == Piece(PieceType.ROOK, Color.BLACK)
        assert game.get_piece(1, 1) == Piece(PieceType.ROOK, Color.WHITE)

    def test_roundtrip_initial_position(self):
        """Roundtrip: from_sfen(game.to_sfen()) produces an identical SFEN."""
        original = ShogiGame()
        sfen = original.to_sfen()
        restored = ShogiGame.from_sfen(sfen)
        assert restored.to_sfen() == sfen

    def test_roundtrip_custom_sfen_with_pieces_in_hand(self):
        """A custom SFEN with pieces in hand roundtrips correctly."""
        custom_sfen = "4k4/9/9/9/9/9/9/9/4K4 b Pp 1"
        game = ShogiGame.from_sfen(custom_sfen)
        # Black should have a Pawn in hand, White should have a Pawn in hand
        assert game.hands[Color.BLACK.value][PieceType.PAWN] == 1
        assert game.hands[Color.WHITE.value][PieceType.PAWN] == 1
        assert game.to_sfen() == custom_sfen

    def test_roundtrip_sfen_with_promoted_pieces(self):
        """A custom SFEN with promoted pieces on the board roundtrips correctly."""
        # Promoted pawn (+P) for Black at row 0, col 0 and kings
        promoted_sfen = "+P3k4/9/9/9/9/9/9/9/4K4 b - 1"
        game = ShogiGame.from_sfen(promoted_sfen)
        piece = game.get_piece(0, 0)
        assert piece is not None
        assert piece.type == PieceType.PROMOTED_PAWN
        assert piece.color == Color.BLACK
        assert game.to_sfen() == promoted_sfen

    def test_from_sfen_preserves_current_player_black(self):
        """from_sfen correctly preserves Black as current player."""
        sfen_black = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        game = ShogiGame.from_sfen(sfen_black)
        assert game.current_player == Color.BLACK

    def test_from_sfen_preserves_current_player_white(self):
        """from_sfen correctly preserves White as current player."""
        sfen_white = "4k4/9/9/9/9/9/9/9/4K4 w - 1"
        game = ShogiGame.from_sfen(sfen_white)
        assert game.current_player == Color.WHITE


# ---------------------------------------------------------------------------
# Observation generation tests
# ---------------------------------------------------------------------------


class TestObservationGeneration:
    """Tests for get_observation / get_state output shape, dtype, and content."""

    def test_observation_shape(self):
        """get_observation returns an ndarray of shape (46, 9, 9)."""
        game = ShogiGame()
        obs = game.get_observation()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (46, 9, 9)

    def test_observation_dtype(self):
        """Observation dtype is float32."""
        game = ShogiGame()
        obs = game.get_observation()
        assert obs.dtype == np.float32

    def test_initial_position_has_nonzero_piece_channels(self):
        """Initial position has non-zero values in the piece channels (0-27)."""
        game = ShogiGame()
        obs = game.get_observation()
        piece_channels = obs[:28, :, :]
        assert np.any(piece_channels != 0.0), "Piece channels should have non-zero values"

    def test_player_indicator_channel_black_turn(self):
        """Channel 42 (current player indicator) is all 1.0 when it is Black's turn."""
        game = ShogiGame()  # Black moves first
        obs = game.get_observation()
        indicator = obs[42, :, :]
        assert np.all(indicator == 1.0), (
            "Player indicator channel should be all 1.0 for Black's turn"
        )

    def test_player_indicator_channel_white_turn(self):
        """Channel 42 (current player indicator) is all 0.0 when it is White's turn."""
        sfen_white = "4k4/9/9/9/9/9/9/9/4K4 w - 1"
        game = ShogiGame.from_sfen(sfen_white)
        obs = game.get_observation()
        indicator = obs[42, :, :]
        assert np.all(indicator == 0.0), (
            "Player indicator channel should be all 0.0 for White's turn"
        )

    def test_observation_changes_after_move(self):
        """After making a move, the observation is not identical to the previous one."""
        game = ShogiGame()
        obs_before = game.get_observation().copy()
        legal_moves = game.get_legal_moves()
        assert len(legal_moves) > 0, "There should be legal moves at the start"
        game.make_move(legal_moves[0])
        obs_after = game.get_observation()
        assert not np.array_equal(obs_before, obs_after), (
            "Observation should change after a move"
        )

    def test_get_state_is_alias_for_get_observation(self):
        """get_state returns the same result as get_observation."""
        game = ShogiGame()
        obs = game.get_observation()
        state = game.get_state()
        assert np.array_equal(obs, state)


# ---------------------------------------------------------------------------
# State hash tests
# ---------------------------------------------------------------------------


class TestBoardStateHash:
    """Tests for get_board_state_hash consistency and uniqueness."""

    def test_same_position_same_hash(self):
        """Two games at the same position produce the same hash."""
        game1 = ShogiGame()
        game2 = ShogiGame()
        assert game1.get_board_state_hash() == game2.get_board_state_hash()

    def test_different_positions_different_hashes(self):
        """After making a move, the hash differs from the initial position."""
        game = ShogiGame()
        hash_before = game.get_board_state_hash()
        legal_moves = game.get_legal_moves()
        game.make_move(legal_moves[0])
        hash_after = game.get_board_state_hash()
        assert hash_before != hash_after, "Hash should change after a move"

    def test_hash_includes_player_to_move(self):
        """Same board layout but different current player produces a different hash."""
        sfen_black = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        sfen_white = "4k4/9/9/9/9/9/9/9/4K4 w - 1"
        game_b = ShogiGame.from_sfen(sfen_black)
        game_w = ShogiGame.from_sfen(sfen_white)
        assert game_b.get_board_state_hash() != game_w.get_board_state_hash(), (
            "Hash should differ when the player to move is different"
        )

    def test_hash_after_reset_matches_initial(self):
        """After reset, the hash matches the initial position hash."""
        game = ShogiGame()
        initial_hash = game.get_board_state_hash()
        legal_moves = game.get_legal_moves()
        game.make_move(legal_moves[0])
        assert game.get_board_state_hash() != initial_hash
        game.reset()
        assert game.get_board_state_hash() == initial_hash
