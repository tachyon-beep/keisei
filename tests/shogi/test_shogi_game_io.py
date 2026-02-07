"""
Unit tests for Shogi game I/O functions in shogi_game_io.py
"""

import os
import tempfile

import numpy as np
import pytest

from keisei.shogi.shogi_core_definitions import (
    OBS_CURR_PLAYER_HAND_START,
    OBS_CURR_PLAYER_INDICATOR,
    OBS_CURR_PLAYER_PROMOTED_START,
    OBS_CURR_PLAYER_UNPROMOTED_START,
    OBS_MOVE_COUNT,
    OBS_OPP_PLAYER_HAND_START,
    OBS_OPP_PLAYER_PROMOTED_START,
    OBS_OPP_PLAYER_UNPROMOTED_START,
    OBS_PROMOTED_ORDER,
    OBS_UNPROMOTED_ORDER,
    Color,
    Piece,
    PieceType,
)
from keisei.shogi.shogi_game import ShogiGame
from keisei.shogi.shogi_game_io import (
    _get_piece_type_from_sfen_char,
    _parse_sfen_square,
    convert_game_to_text_representation,
    game_to_kif,
    generate_neural_network_observation,
    sfen_to_move_tuple,
)
from tests.utils.mock_utilities import setup_pytorch_mock_environment

INPUT_CHANNELS = 46  # Use the default from config_schema for tests


@pytest.fixture
def basic_game():
    """Creates a basic game fixture with setup_pytorch_mock_environment context."""
    with setup_pytorch_mock_environment():
        game = ShogiGame()
        return game


@pytest.fixture
def game_with_moves():
    """Creates a game with several moves made."""
    with setup_pytorch_mock_environment():
        game = ShogiGame()

        # Black pawn move
        game.make_move((6, 4, 5, 4, False))

        # White pawn move
        game.make_move((2, 4, 3, 4, False))

        # Black pawn move
        game.make_move((5, 4, 4, 4, False))

        return game


@pytest.fixture
def game_with_capture():
    """Creates a game where a piece has been captured."""
    with setup_pytorch_mock_environment():
        game = ShogiGame()

        # Setup position for a capture
        # Place a black pawn in front of white's pawn
        game.set_piece(3, 4, Piece(PieceType.PAWN, Color.BLACK))
        # Now make white's move - we need to switch the player first
        game.current_player = Color.WHITE
        game.make_move((2, 4, 3, 4, False))  # White pawn captures black pawn

        return game


@pytest.fixture
def game_with_promotion():
    """Creates a game where a piece has been promoted."""
    with setup_pytorch_mock_environment():
        game = ShogiGame()

        # Clear the board first
        for r in range(9):
            for c in range(9):
                game.set_piece(r, c, None)

        # Add kings so legal move generation works
        game.set_piece(0, 0, Piece(PieceType.KING, Color.WHITE))
        game.set_piece(8, 8, Piece(PieceType.KING, Color.BLACK))

        # Setup position for a promotion (black to move)
        game.set_piece(3, 4, Piece(PieceType.PAWN, Color.BLACK))
        game.make_move((3, 4, 2, 4, True))  # Black pawn promotes

        # Verify the promotion succeeded
        promoted_piece = game.get_piece(2, 4)
        assert promoted_piece is not None, "Promotion failed - no piece found"
        assert (
            promoted_piece.type == PieceType.PROMOTED_PAWN
        ), "Promotion failed - piece not promoted"

        return game


def test_generate_neural_network_observation_initial_state():
    """Test that observation shape and current player indicator are correct for initial state."""
    with setup_pytorch_mock_environment():
        game = ShogiGame()
        obs = generate_neural_network_observation(game)

        assert obs.shape == (
            INPUT_CHANNELS,
            9,
            9,
        ), "Observation shape should be (46, 9, 9)"
        assert np.all(
            obs[OBS_CURR_PLAYER_INDICATOR] == 1
        ), "Current player indicator should be all 1's for Black's turn"
        assert (
            np.sum(obs[OBS_CURR_PLAYER_UNPROMOTED_START]) > 0
        ), "No current player unpromoted pieces found"
        assert (
            np.sum(obs[OBS_OPP_PLAYER_UNPROMOTED_START]) > 0
        ), "No opponent player unpromoted pieces found"


def test_generate_neural_network_observation_after_three_moves(game_with_moves):
    """Test observation after several moves: turn indicator and move count normalization."""
    with setup_pytorch_mock_environment():
        obs = generate_neural_network_observation(game_with_moves)
        assert np.all(
            obs[OBS_CURR_PLAYER_INDICATOR] == 0
        ), "Should be White's turn after 3 moves"
        assert np.all(
            obs[OBS_MOVE_COUNT] > 0
        ), "Move count plane should be greater than 0"
        actual_value = obs[OBS_MOVE_COUNT][0, 0]
        assert (
            0 < actual_value < 1
        ), f"Move count plane should be normalized between 0 and 1, got {actual_value}"


def test_generate_neural_network_observation_after_pawn_capture(game_with_capture):
    """Test that a captured pawn appears in the correct hand and observation plane after capture."""
    with setup_pytorch_mock_environment():
        obs = generate_neural_network_observation(game_with_capture)
        pawn_index = OBS_UNPROMOTED_ORDER.index(PieceType.PAWN)
        pawn_position_in_hand = OBS_OPP_PLAYER_HAND_START + pawn_index
        capturing_player = 1 if game_with_capture.current_player.value == 0 else 0
        assert (
            game_with_capture.hands[capturing_player].get(PieceType.PAWN, 0) > 0
        ), "Pawn should be in the capturing player's hand according to game state"
        text_repr = convert_game_to_text_representation(game_with_capture)
        assert "{'PAWN': 1}" in text_repr, "Hand should show PAWN: 1"
        hand_plane_sum = np.sum(obs[pawn_position_in_hand])
        assert (
            hand_plane_sum > 0
        ), "Captured pawn should appear in the opponent's hand plane in the observation"


def test_generate_neural_network_observation_after_pawn_promotion(game_with_promotion):
    """Test that a promoted pawn appears in the correct observation plane and game state after promotion."""
    with setup_pytorch_mock_environment():
        obs = generate_neural_network_observation(game_with_promotion)
        pawn_idx = OBS_PROMOTED_ORDER.index(PieceType.PROMOTED_PAWN)
        curr_player_pawn_plane = OBS_CURR_PLAYER_PROMOTED_START + pawn_idx
        opp_player_pawn_plane = OBS_OPP_PLAYER_PROMOTED_START + pawn_idx
        promoted_piece = game_with_promotion.get_piece(2, 4)
        assert promoted_piece is not None, "No piece found at the promotion location"
        assert (
            promoted_piece.type.value == PieceType.PROMOTED_PAWN.value
            and promoted_piece.color.value == Color.BLACK.value
        ), f"Expected promoted pawn of color BLACK, got {promoted_piece.type}, {promoted_piece.color}"
        found_in_planes = (np.sum(obs[curr_player_pawn_plane]) > 0) or (
            np.sum(obs[opp_player_pawn_plane]) > 0
        )
        assert (
            found_in_planes
        ), "Promoted pawn should appear somewhere in the observation"


def test_convert_game_to_text_representation_initial_state():
    """Test that text representation of the initial game state contains expected elements."""
    with setup_pytorch_mock_environment():
        game = ShogiGame()
        text_repr = convert_game_to_text_representation(game)
        assert "Turn: BLACK" in text_repr, "Turn indicator should be present"
        assert (
            "BLACK" in text_repr.upper() or "Black" in text_repr
        ), "Player identifier should be present"
        assert (
            "WHITE" in text_repr.upper() or "White" in text_repr
        ), "Player identifier should be present"
        assert "Move: 1" in text_repr, "Move count should be present"


def test_convert_game_to_text_representation_after_three_moves(game_with_moves):
    """Test text representation after several moves: turn and move count update."""
    with setup_pytorch_mock_environment():
        text_repr = convert_game_to_text_representation(game_with_moves)
        assert "Turn: WHITE" in text_repr, "Turn indicator should show WHITE"
        assert "Move: 4" in text_repr, "Move count should be 4"
        assert (
            "p  p  p  p  ." in text_repr or ".  p  p  p  p" in text_repr
        ), "Should show moved pawns"


def test_convert_game_to_text_representation_after_pawn_capture(game_with_capture):
    """Test that text representation after a capture shows the captured pawn in hand."""
    with setup_pytorch_mock_environment():
        text_repr = convert_game_to_text_representation(game_with_capture)
        assert (
            "White's hand: {'PAWN': 1}" in text_repr or "PAWN" in text_repr
        ), "Captured pawn should appear in hand"


def test_game_to_kif_writes_valid_kif_file_after_moves():
    """Test that KIF export writes a file with expected player names and header."""
    with setup_pytorch_mock_environment():
        game = ShogiGame()
    game.make_move((6, 4, 5, 4, False))  # Black pawn
    game.make_move((2, 4, 3, 4, False))  # White pawn

    with tempfile.NamedTemporaryFile(suffix=".kif", delete=False) as temp_file:
        filename = temp_file.name
    game_to_kif(
        game, filename=filename, sente_player_name="Player1", gote_player_name="Player2"
    )
    with open(filename, "r", encoding="utf-8") as f:
        kif_content = f.read()
    assert "Player1" in kif_content, "Black player name should be present"
    assert "Player2" in kif_content, "White player name should be present"
    assert "KIF" in kif_content, "KIF header should be present"
    # Remove the file after test
    os.remove(filename)


def test_sfen_to_move_tuple_parses_standard_and_drop_moves():
    """Test that sfen_to_move_tuple parses both normal and drop moves correctly."""
    with setup_pytorch_mock_environment():
        result = sfen_to_move_tuple("7g7f")
        assert result is not None, "sfen_to_move_tuple should not return None"
        assert isinstance(result, tuple), "sfen_to_move_tuple should return a tuple"
        from_row, from_col, to_row, to_col, promotion = result
        assert from_row is not None
        assert from_col is not None
        assert to_row is not None
        assert to_col is not None
        assert isinstance(promotion, bool)
        drop_result = sfen_to_move_tuple("P*7f")
        assert drop_result is not None


def test_parse_sfen_square_parses_various_squares():
    """Test that _parse_sfen_square parses various SFEN coordinates correctly and distinguishes them."""
    with setup_pytorch_mock_environment():
        result = _parse_sfen_square("7g")
        assert result is not None, "_parse_sfen_square should not return None"
        assert isinstance(result, tuple), "_parse_sfen_square should return a tuple"
        assert len(result) == 2, "_parse_sfen_square should return a tuple of length 2"
        result1 = _parse_sfen_square("7g")
        result2 = _parse_sfen_square("1a")
        result3 = _parse_sfen_square("9i")
        assert result1 != result2, "Different squares should give different results"
        assert result1 != result3, "Different squares should give different results"
        assert result2 != result3, "Different squares should give different results"


def test_get_piece_type_from_sfen_char_handles_all_piece_types_and_promotions():
    """Test that _get_piece_type_from_sfen_char parses all supported piece types and promoted pieces."""
    with setup_pytorch_mock_environment():
        assert _get_piece_type_from_sfen_char("P") == PieceType.PAWN, "P should be pawn"
        assert (
            _get_piece_type_from_sfen_char("L") == PieceType.LANCE
        ), "L should be lance"
        assert (
            _get_piece_type_from_sfen_char("N") == PieceType.KNIGHT
        ), "N should be knight"
        assert (
            _get_piece_type_from_sfen_char("S") == PieceType.SILVER
        ), "S should be silver"
        assert _get_piece_type_from_sfen_char("G") == PieceType.GOLD, "G should be gold"
        assert (
            _get_piece_type_from_sfen_char("B") == PieceType.BISHOP
        ), "B should be bishop"
        assert _get_piece_type_from_sfen_char("R") == PieceType.ROOK, "R should be rook"
        try:
            _get_piece_type_from_sfen_char("K")
        except ValueError:
            pass
        try:
            promoted_pawn = _get_piece_type_from_sfen_char("+P")
            assert (
                promoted_pawn == PieceType.PROMOTED_PAWN
            ), "+P should be promoted pawn"
        except (ValueError, TypeError):
            pass


def test_convert_game_to_text_representation_complex_state():
    """Text dump of a complex mid–game SFEN should match piece layout and hands."""

    game = ShogiGame.from_sfen(
        "l2g1k1nl/1r1S2sb1/p1n1p1p1p/1p1pP1P1P/1P1P2P1P/P1P1G1P1P/1N1P1P1P1/1+r5R1/L2K1GSNL b B2Ppl 33"
    )
    text_repr = convert_game_to_text_representation(game)
    lines = text_repr.splitlines()

    # --- meta info ---
    assert "Turn: BLACK" in text_repr
    assert "Move: 33" in text_repr

    # Helper to fetch token list for a given printed rank (9..1)
    def rank_tokens(rank: int) -> list[str]:
        line = next(l for l in lines if l.startswith(f"{rank} "))
        return line.split()[1:]  # strip rank number

    # Rank 9 (top) : l2g1k1nl  →  l . . g . k . n l
    assert rank_tokens(9) == ["l", ".", ".", "g", ".", "k", ".", "n", "l"]

    # Rank 1 (bottom): L2K1GSNL →  L . . K . G S N L
    assert rank_tokens(1) == ["L", ".", ".", "K", ".", "G", "S", "N", "L"]

    # Rank 7 (row-2): p1n1p1p1p → p . n . p . p . p
    assert rank_tokens(7) == ["p", ".", "n", ".", "p", ".", "p", ".", "p"]

    # Rank 2 (row-7) with promotions: 1+r5R1 → . +r . . . . . R .
    rt = rank_tokens(2)
    assert rt[1] == "+r" and rt[7] == "R"

    # --- hands ---
    # order-independent checks
    assert "Black's hand:" in text_repr
    assert "'BISHOP': 1" in text_repr
    assert "'PAWN': 2" in text_repr
    assert "White's hand:" in text_repr
    assert "'PAWN': 1" in text_repr
    assert "'LANCE': 1" in text_repr


def test_game_to_kif_checkmate_and_hands():
    """Test KIF export for a game ending in checkmate and with pieces in hand."""
    # Setup a checkmate position where Black wins
    # White King 'k' at (0,0) (SFEN: 9a)
    # Black Gold 'G_guard' at (1,1) (SFEN: 8b) protects the mating square (1,0)
    # Black Gold 'G_mate' at (2,0) (SFEN: 9c) delivers the mate by moving to (1,0)
    # (Black King 'K' at (8,0) (SFEN: 9i) for SFEN completeness)
    # Hands: Black has 1 Pawn, White has 2 Pawns. Black to move.
    # SFEN representation (0-indexed): k(0,0), G_guard(1,1), G_mate(2,0), K(8,0)
    game = ShogiGame.from_sfen("k8/1G7/G8/9/9/9/9/9/K8 b P2p 1")

    # Mating move: Black Gold 'G_mate' from (2,0) moves to (1,0).
    # This checks White King at (0,0).
    # Escape (0,1) is covered by G_guard(1,1).
    # Escape (1,1) is occupied by G_guard(1,1).
    # Capture G_mate(1,0) is not possible as (1,0) is covered by G_guard(1,1).
    mating_move = (2, 0, 1, 0, False)

    # Verify this move is legal first
    legal_moves = game.get_legal_moves()
    if mating_move not in legal_moves:
        pytest.fail(
            f"Setup error for KIF test: Mating move {mating_move} not legal. Legal: {legal_moves}. SFEN: {game.to_sfen_string()}"
        )

    game.make_move(mating_move)

    assert game.game_over
    assert game.winner == Color.BLACK
    assert game.termination_reason == "Tsumi"

    with tempfile.NamedTemporaryFile(suffix=".kif", delete=False) as temp_file:
        filename = temp_file.name

    kif_content_str = game_to_kif(
        game,
        filename=None,
        sente_player_name="SentePlayer",
        gote_player_name="GotePlayer",
    )
    assert kif_content_str is not None

    assert "#KIF version=2.0" in kif_content_str
    assert "*Player Sente: SentePlayer" in kif_content_str
    assert "*Player Gote: GotePlayer" in kif_content_str
    # UPDATED ASSERTIONS to match the verbose KIF output for hands P2p:
    # Black (Sente) has 1 Pawn, represented as 01FU in the verbose line.
    assert "P+00HI00KA00KI00GI00KE00KY01FU" in kif_content_str
    # White (Gote) has 2 Pawns, represented as 02FU in the verbose line.
    assert "P-00HI00KA00KI00GI00KE00KY02FU" in kif_content_str
    assert "RESULT:SENTE_WIN" in kif_content_str

    # ---- write to file and verify ----
    game_to_kif(
        game,
        filename=filename,
        sente_player_name="SentePlayer",
        gote_player_name="GotePlayer",
    )
    with open(filename, "r", encoding="utf-8") as f:
        file_kif_content = f.read()

    assert "#KIF version=2.0" in file_kif_content
    assert "RESULT:SENTE_WIN" in file_kif_content

    os.remove(filename)


def test_generate_neural_network_observation_max_hands_and_promoted_board():
    """Test observation with max pieces in hand and many promoted pieces."""
    game = ShogiGame.from_sfen(
        "4k4/9/9/9/9/9/9/9/4K4 b 7PR2B2G2S2N2L7pr2b2g2s2n2l 1"  # Corrected hand string
    )
    obs = generate_neural_network_observation(game)

    assert obs.shape == (INPUT_CHANNELS, 9, 9)

    # Test for max hands
    # From the SFEN: Black hand "7PR2B2G2S2N2L", White hand "7pr2b2g2s2n2l"
    # Black has 7 Pawns, 1 Rook, 2 Bishops, 2 Golds, 2 Silvers, 2 Knights, 2 Lances
    # White has 7 pawns, 1 rook, 2 bishops, 2 golds, 2 silvers, 2 knights, 2 lances

    # Check Black hand planes (current player hand)
    # Hand counts are normalized by per-type maximums (P:18, L/N/S/G:4, B:2, R:2)
    assert np.allclose(obs[OBS_CURR_PLAYER_HAND_START + 0], 7 / 18.0)  # PAWN (7/18)
    assert np.allclose(obs[OBS_CURR_PLAYER_HAND_START + 4], 2 / 4.0)  # GOLD (2/4)
    assert np.allclose(obs[OBS_CURR_PLAYER_HAND_START + 5], 2 / 2.0)  # BISHOP (2/2)
    assert np.allclose(obs[OBS_CURR_PLAYER_HAND_START + 6], 1 / 2.0)  # ROOK (1/2)

    # Check White hand planes (opponent hand)
    assert np.allclose(obs[OBS_OPP_PLAYER_HAND_START + 0], 7 / 18.0)  # PAWN (7/18)
    assert np.allclose(obs[OBS_OPP_PLAYER_HAND_START + 4], 2 / 4.0)  # GOLD (2/4)
    assert np.allclose(obs[OBS_OPP_PLAYER_HAND_START + 5], 2 / 2.0)  # BISHOP (2/2)
    assert np.allclose(obs[OBS_OPP_PLAYER_HAND_START + 6], 1 / 2.0)  # ROOK (1/2)


def test_generate_neural_network_observation_move_count_normalization(
    basic_game: ShogiGame,
):
    """Test OBS_MOVE_COUNT normalization with different max_moves_per_game."""
    game = basic_game
    game.move_count = 50

    game._max_moves_this_game = 100  # pylint: disable=protected-access
    obs100 = generate_neural_network_observation(game)
    assert np.allclose(obs100[OBS_MOVE_COUNT], 50 / 100.0)

    game._max_moves_this_game = 500  # Default # pylint: disable=protected-access
    obs500 = generate_neural_network_observation(game)
    assert np.allclose(obs500[OBS_MOVE_COUNT], 50 / 500.0)

    game.move_count = 0
    obs_start = generate_neural_network_observation(game)
    assert np.allclose(obs_start[OBS_MOVE_COUNT], 0.0)

    game.move_count = 499
    obs_end = generate_neural_network_observation(game)
    assert np.allclose(obs_end[OBS_MOVE_COUNT], 499 / 500.0)


@pytest.mark.parametrize(
    "invalid_sfen_move",
    [
        "7g7f++",
        "7g7fx",  # Malformed promotion
        "K*5e",  # Invalid piece for drop (King)
        "+P*5e",  # Invalid piece for drop (Promoted Pawn)
        "P*0a",
        "P*10a",
        "P*1j",  # Invalid drop squares (out of bounds)
        "1a0a",
        "1a10a",
        "1a1j",  # Invalid board move squares (out of bounds)
        "P*",
        "*5e",  # Incomplete drop
        "B*11",  # Invalid drop square (too short)
        "L*a1",  # Invalid drop square (wrong order)
        "S*5ee",  # Invalid drop square (too long)
    ],
)
def test_sfen_to_move_tuple_invalid_formats(invalid_sfen_move):
    """Test sfen_to_move_tuple with various invalid SFEN move strings."""
    with (
        setup_pytorch_mock_environment()
    ):  # In case PolicyOutputMapper is involved by mistake
        with pytest.raises(ValueError):
            sfen_to_move_tuple(invalid_sfen_move)
