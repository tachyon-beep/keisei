"""Tests for SFEN-to-board_state conversion."""

import pytest

pytestmark = pytest.mark.unit


class TestSfenToBoardState:
    """Tests for sfen_to_board_state()."""

    STARTPOS = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"

    def test_starting_position_dimensions(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert len(result["board"]) == 9
        assert all(len(row) == 9 for row in result["board"])

    def test_starting_position_current_player(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert result["current_player"] == "black"

    def test_starting_position_move_count(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert result["move_count"] == 1

    def test_starting_position_not_game_over(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert result["game_over"] is False
        assert result["winner"] is None

    def test_starting_position_empty_hands(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        assert result["black_hand"] == {}
        assert result["white_hand"] == {}

    def test_starting_position_black_pieces_row9(self):
        """Row 9 (index 8): LNSGKGSNL — all black pieces."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        row = result["board"][8]
        expected_types = [
            "lance", "knight", "silver", "gold", "king",
            "gold", "silver", "knight", "lance",
        ]
        for col, expected_type in enumerate(expected_types):
            assert row[col] is not None, f"col {col} should not be empty"
            assert row[col]["type"] == expected_type
            assert row[col]["color"] == "black"
            assert row[col]["promoted"] is False

    def test_starting_position_white_pieces_row1(self):
        """Row 1 (index 0): lnsgkgsnl — all white pieces."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        row = result["board"][0]
        expected_types = [
            "lance", "knight", "silver", "gold", "king",
            "gold", "silver", "knight", "lance",
        ]
        for col, expected_type in enumerate(expected_types):
            assert row[col] is not None, f"col {col} should not be empty"
            assert row[col]["type"] == expected_type
            assert row[col]["color"] == "white"
            assert row[col]["promoted"] is False

    def test_starting_position_empty_middle_rows(self):
        """Rows 4-6 (indices 3-5) should be all None."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        for r in range(3, 6):
            assert all(cell is None for cell in result["board"][r]), f"row {r}"

    def test_starting_position_bishops_and_rooks(self):
        """Row 2 (index 1): white bishop at col 7, rook at col 1."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        result = sfen_to_board_state(self.STARTPOS)
        # White rook at row 1, col 1
        assert result["board"][1][1]["type"] == "rook"
        assert result["board"][1][1]["color"] == "white"
        # White bishop at row 1, col 7
        assert result["board"][1][7]["type"] == "bishop"
        assert result["board"][1][7]["color"] == "white"
        # Black bishop at row 7, col 1
        assert result["board"][7][1]["type"] == "bishop"
        assert result["board"][7][1]["color"] == "black"
        # Black rook at row 7, col 7
        assert result["board"][7][7]["type"] == "rook"
        assert result["board"][7][7]["color"] == "black"

    def test_promoted_pieces(self):
        """SFEN with +R (dragon) and +B (horse)."""
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "lnsgkgsnl/1+R5+b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
        result = sfen_to_board_state(sfen)
        # Black promoted rook at row 1, col 1
        assert result["board"][1][1]["type"] == "promoted_rook"
        assert result["board"][1][1]["color"] == "black"
        assert result["board"][1][1]["promoted"] is True
        # White promoted bishop at row 1, col 7
        assert result["board"][1][7]["type"] == "promoted_bishop"
        assert result["board"][1][7]["color"] == "white"
        assert result["board"][1][7]["promoted"] is True

    def test_white_turn(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1"
        result = sfen_to_board_state(sfen)
        assert result["current_player"] == "white"

    def test_hand_pieces_multiple_types(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "4k4/9/9/9/9/9/9/9/4K4 b 2P1L3Nrb 10"
        result = sfen_to_board_state(sfen)
        assert result["black_hand"] == {"pawn": 2, "lance": 1, "knight": 3}
        assert result["white_hand"] == {"rook": 1, "bishop": 1}

    def test_hand_single_piece_no_count(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "4k4/9/9/9/9/9/9/9/4K4 b Pp 1"
        result = sfen_to_board_state(sfen)
        assert result["black_hand"] == {"pawn": 1}
        assert result["white_hand"] == {"pawn": 1}

    def test_empty_hand_dash(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 1"
        result = sfen_to_board_state(sfen)
        assert result["black_hand"] == {}
        assert result["white_hand"] == {}

    def test_move_count_parsed(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        sfen = "4k4/9/9/9/9/9/9/9/4K4 b - 42"
        result = sfen_to_board_state(sfen)
        assert result["move_count"] == 42


class TestSfenErrors:
    """Tests for SFEN parser error handling."""

    def test_too_few_fields(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="at least 3 fields"):
            sfen_to_board_state("lnsgkgsnl/9/9/9/9/9/9/9/LNSGKGSNL b")

    def test_wrong_row_count(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="9 rows"):
            sfen_to_board_state("4k4/9/9/9/9/9/9/4K4 b - 1")

    def test_row_too_short(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="8 squares"):
            sfen_to_board_state("4k3/9/9/9/9/9/9/9/4K4 b - 1")

    def test_unknown_piece_char(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="Unknown piece character 'x'"):
            sfen_to_board_state("4k3x/9/9/9/9/9/9/9/4K4 b - 1")

    def test_invalid_turn(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="turn must be 'b' or 'w'"):
            sfen_to_board_state("4k4/9/9/9/9/9/9/9/4K4 x - 1")

    def test_unknown_hand_piece(self):
        from keisei.webui.sfen_utils import sfen_to_board_state

        with pytest.raises(ValueError, match="Unknown hand piece"):
            sfen_to_board_state("4k4/9/9/9/9/9/9/9/4K4 b X 1")
