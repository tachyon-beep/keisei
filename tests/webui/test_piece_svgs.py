"""Tests for Shogi piece SVG assets and the SVG loading cache."""

import pytest
from pathlib import Path

try:
    import streamlit  # noqa: F401

    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False

IMAGES_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "keisei"
    / "webui"
    / "static"
    / "images"
)

EXPECTED_PIECES = [
    "king",
    "rook",
    "bishop",
    "gold",
    "silver",
    "knight",
    "lance",
    "pawn",
    "promoted_rook",
    "promoted_bishop",
    "promoted_silver",
    "promoted_knight",
    "promoted_lance",
    "promoted_pawn",
]
COLORS = ["black", "white"]


@pytest.mark.unit
class TestPieceSVGs:
    def test_images_directory_exists(self):
        assert IMAGES_DIR.is_dir(), f"Missing directory: {IMAGES_DIR}"

    @pytest.mark.parametrize("piece", EXPECTED_PIECES)
    @pytest.mark.parametrize("color", COLORS)
    def test_svg_file_exists(self, piece, color):
        path = IMAGES_DIR / f"{piece}_{color}.svg"
        assert path.exists(), f"Missing SVG: {path.name}"

    @pytest.mark.parametrize("piece", EXPECTED_PIECES)
    @pytest.mark.parametrize("color", COLORS)
    def test_svg_is_valid_xml(self, piece, color):
        import xml.etree.ElementTree as ET

        path = IMAGES_DIR / f"{piece}_{color}.svg"
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag.endswith("svg"), f"{path.name} root is not <svg>"

    @pytest.mark.parametrize("piece", EXPECTED_PIECES)
    @pytest.mark.parametrize("color", COLORS)
    def test_svg_has_viewbox(self, piece, color):
        import xml.etree.ElementTree as ET

        path = IMAGES_DIR / f"{piece}_{color}.svg"
        tree = ET.parse(path)
        root = tree.getroot()
        assert "viewBox" in root.attrib, f"{path.name} missing viewBox"

    def test_total_file_count(self):
        svgs = list(IMAGES_DIR.glob("*.svg"))
        assert len(svgs) == 28, f"Expected 28 SVGs, found {len(svgs)}"

    @pytest.mark.skipif(not _HAS_STREAMLIT, reason="streamlit not installed")
    def test_svg_cache_loads(self):
        """Verify the app's SVG loading function populates the cache."""
        from keisei.webui.streamlit_app import _PIECE_SVG_CACHE, _load_piece_svgs

        _PIECE_SVG_CACHE.clear()
        _load_piece_svgs()
        assert len(_PIECE_SVG_CACHE) == 28, f"Cache has {len(_PIECE_SVG_CACHE)} entries"
        for piece in EXPECTED_PIECES:
            for color in COLORS:
                key = f"{piece}_{color}"
                assert key in _PIECE_SVG_CACHE, f"Missing cache key: {key}"
                assert _PIECE_SVG_CACHE[key].startswith("data:image/svg+xml;base64,")


@pytest.mark.unit
class TestSampleState:
    def test_sample_state_passes_envelope_validation(self):
        import json
        from keisei.webui.view_contracts import validate_envelope

        sample_path = (
            Path(__file__).resolve().parent.parent.parent
            / "keisei"
            / "webui"
            / "sample_state.json"
        )
        with open(sample_path) as f:
            data = json.load(f)
        errors = validate_envelope(data)
        assert errors == [], f"Validation errors: {errors}"
