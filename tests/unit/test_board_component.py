"""Unit tests for the board component Python wrapper."""

import pytest

pytestmark = pytest.mark.unit


class TestBoardComponentImport:
    """Board component wrapper is importable and has expected API."""

    def test_shogi_board_function_exists(self):
        """shogi_board function is importable."""
        from keisei.webui.board_component import shogi_board

        assert callable(shogi_board)

    def test_shogi_board_accepts_expected_args(self):
        """shogi_board accepts board_state, heatmap, square_actions, piece_images, selected_square, key."""
        import inspect

        from keisei.webui.board_component import shogi_board

        sig = inspect.signature(shogi_board)
        param_names = list(sig.parameters.keys())
        assert "board_state" in param_names
        assert "heatmap" in param_names
        assert "square_actions" in param_names
        assert "piece_images" in param_names
        assert "selected_square" in param_names
        assert "key" in param_names
