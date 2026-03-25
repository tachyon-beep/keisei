"""Generate SVG Shogi piece images for the Streamlit training dashboard.

Creates 28 SVGs (14 piece types x 2 colors) using a pentagonal Shogi
piece shape with kanji characters.  White (gote) pieces are rotated 180°.

Run:
    python -m keisei.webui.piece_svg_generator
"""

from pathlib import Path
from typing import Dict, List, Tuple

_OUTPUT_DIR = Path(__file__).parent / "static" / "images"

# (svg_stem, kanji, is_promoted)
_PIECE_CATALOG: List[Tuple[str, str, bool]] = [
    ("king", "王", False),
    ("rook", "飛", False),
    ("bishop", "角", False),
    ("gold", "金", False),
    ("silver", "銀", False),
    ("knight", "桂", False),
    ("lance", "香", False),
    ("pawn", "歩", False),
    ("promoted_rook", "龍", True),
    ("promoted_bishop", "馬", True),
    ("promoted_silver", "全", True),
    ("promoted_knight", "圭", True),
    ("promoted_lance", "杏", True),
    ("promoted_pawn", "と", True),
]

# SVG dimensions
_WIDTH = 40
_HEIGHT = 44
_CX = _WIDTH / 2  # 20
_CY = _HEIGHT / 2  # 22

# Pentagonal piece path: pointed top, widening shoulders, flat bottom
_PIECE_PATH = "M20 2 L36 14 L32 42 L8 42 L4 14 Z"

_FILL = "#f5deb3"
_STROKE = "#000000"
_STROKE_WIDTH = "1.5"
_KANJI_COLOR = "#000000"
_PROMOTED_COLOR = "#cc0000"
_FONT_SIZE = "18"


def _generate_svg(kanji: str, is_promoted: bool, is_white: bool) -> str:
    """Generate a single piece SVG as a string."""
    fill_color = _PROMOTED_COLOR if is_promoted else _KANJI_COLOR
    transform = f'transform="rotate(180, {_CX}, {_CY})"' if is_white else ""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {_HEIGHT}" width="{_WIDTH}" height="{_HEIGHT}">
  <g {transform}>
    <path d="{_PIECE_PATH}" fill="{_FILL}" stroke="{_STROKE}" stroke-width="{_STROKE_WIDTH}"/>
    <text x="{_CX}" y="30" text-anchor="middle" dominant-baseline="central"
          font-size="{_FONT_SIZE}" font-family="serif" fill="{fill_color}">{kanji}</text>
  </g>
</svg>
"""


def generate_all() -> Dict[str, str]:
    """Generate all 28 piece SVGs, returning {filename_stem: svg_content}."""
    pieces: Dict[str, str] = {}
    for stem, kanji, is_promoted in _PIECE_CATALOG:
        for color in ("black", "white"):
            key = f"{stem}_{color}"
            pieces[key] = _generate_svg(kanji, is_promoted, color == "white")
    return pieces


def write_all(output_dir: Path | None = None) -> int:
    """Write all piece SVGs to disk. Returns number of files written."""
    dest = output_dir or _OUTPUT_DIR
    dest.mkdir(parents=True, exist_ok=True)

    pieces = generate_all()
    for name, svg in pieces.items():
        (dest / f"{name}.svg").write_text(svg, encoding="utf-8")

    return len(pieces)


if __name__ == "__main__":
    count = write_all()
    print(f"Generated {count} SVG files in {_OUTPUT_DIR}")
