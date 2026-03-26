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


class TestBoardComponentFrontend:
    """Verify the JS frontend files have the expected structure."""

    def _read_js(self):
        from pathlib import Path

        js_path = (
            Path(__file__).parent.parent.parent
            / "keisei"
            / "webui"
            / "board_component"
            / "frontend"
            / "board.js"
        )
        return js_path.read_text()

    def _read_css(self):
        from pathlib import Path

        css_path = (
            Path(__file__).parent.parent.parent
            / "keisei"
            / "webui"
            / "board_component"
            / "frontend"
            / "board.css"
        )
        return css_path.read_text()

    def test_frontend_js_exists(self):
        """The frontend board.js file exists and is substantial."""
        content = self._read_js()
        assert len(content) > 100

    def test_frontend_uses_grid_role(self):
        """The JS generates role='grid', not role='table'."""
        content = self._read_js()
        assert 'role="grid"' in content or "role='grid'" in content

    def test_frontend_has_no_cdn_dependency(self):
        """The frontend does not load JS from external CDNs."""
        content = self._read_js()
        assert "cdn.jsdelivr.net" not in content
        assert "unpkg.com" not in content

    def test_frontend_uses_v2_api(self):
        """The JS uses the v2 component API (setTriggerValue, not setComponentValue)."""
        content = self._read_js()
        assert "setTriggerValue" in content
        assert "export default function" in content

    def test_frontend_has_roving_tabindex(self):
        """The frontend implements roving tabindex pattern."""
        content = self._read_js()
        assert "tabindex" in content
        assert "moveFocusTo" in content

    def test_frontend_has_aria_selected(self):
        """The frontend sets aria-selected on gridcells."""
        content = self._read_js()
        assert "aria-selected" in content

    def test_frontend_has_aria_rowcount(self):
        """The frontend declares grid dimensions for screen readers."""
        content = self._read_js()
        assert "aria-rowcount" in content
        assert "aria-colcount" in content

    def test_css_has_selection_styles(self):
        """The CSS includes selection and focus indicator styles."""
        content = self._read_css()
        assert ".selected" in content
        assert "#2a52b0" in content
        assert "#666" in content
