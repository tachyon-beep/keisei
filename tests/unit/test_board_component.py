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


class TestBoardComponentHTML:
    """Verify the JS frontend HTML file has the expected structure."""

    def _read_index(self):
        from pathlib import Path

        index = (
            Path(__file__).parent.parent.parent
            / "keisei"
            / "webui"
            / "board_component"
            / "frontend"
            / "index.html"
        )
        return index.read_text()

    def test_frontend_index_exists(self):
        """The frontend index.html file exists."""
        content = self._read_index()
        assert len(content) > 100

    def test_frontend_uses_grid_role(self):
        """The frontend HTML uses role='grid', not role='table'."""
        content = self._read_index()
        assert 'role="grid"' in content or "role='grid'" in content
        assert 'role="table"' not in content

    def test_frontend_has_no_cdn_dependency(self):
        """The frontend does not load JS from external CDNs."""
        content = self._read_index()
        assert "cdn.jsdelivr.net" not in content
        assert "unpkg.com" not in content

    def test_frontend_has_setComponentValue(self):
        """The frontend calls Streamlit.setComponentValue for interaction events."""
        content = self._read_index()
        assert "setComponentValue" in content

    def test_frontend_has_roving_tabindex(self):
        """The frontend implements roving tabindex pattern."""
        content = self._read_index()
        # The JS builds tabindex values dynamically; check for the logic
        assert "tabindex" in content
        assert "moveFocusTo" in content

    def test_frontend_has_aria_selected(self):
        """The frontend sets aria-selected on gridcells."""
        content = self._read_index()
        assert "aria-selected" in content

    def test_frontend_has_aria_rowcount(self):
        """The frontend declares grid dimensions for screen readers."""
        content = self._read_index()
        assert "aria-rowcount" in content
        assert "aria-colcount" in content
