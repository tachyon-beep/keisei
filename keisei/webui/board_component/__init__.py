"""Custom Streamlit component: interactive shogi board with click-to-inspect.

Uses the ``st.components.v2`` API with inline HTML/CSS/JS (no iframe, no npm).
The JS frontend (board.js) calls ``component.setTriggerValue("selection", ...)``
for click/keyboard events and ``component.setStateValue("board_focused", ...)``
for focus tracking.  On the Python side, the v2 ``BidiComponentResult`` exposes
these as ``result.selection`` and ``result.board_focused`` attributes (accessed
via ``getattr`` in ``render_game_tab``).

See: docs/superpowers/specs/2026-03-26-board-interactivity-design.md, Section 4.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v2 as components

_logger = logging.getLogger(__name__)

# Load JS from file to keep the Python module clean
_JS_PATH = Path(__file__).parent / "frontend" / "board.js"
_CSS_PATH = Path(__file__).parent / "frontend" / "board.css"

if not _JS_PATH.exists():
    raise FileNotFoundError(f"Board component JS not found: {_JS_PATH}")
if not _CSS_PATH.exists():
    _logger.warning("Board component CSS not found: %s — styling may be missing", _CSS_PATH)

_JS = _JS_PATH.read_text()
_CSS = _CSS_PATH.read_text() if _CSS_PATH.exists() else ""

_component_func = components.component(
    "shogi_board",
    html='<div id="board-root" style="width:100%;text-align:center;"></div>',
    css=_CSS,
    js=_JS,
    isolate_styles=False,
)


def shogi_board(
    board_state: Dict[str, Any],
    heatmap: Optional[List[List[float]]] = None,
    square_actions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    piece_images: Optional[Dict[str, str]] = None,
    selected_square: Optional[Dict[str, int]] = None,
    key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Render the shogi board with click-to-inspect support.

    Parameters
    ----------
    board_state : dict
        Board state from the envelope (9x9 grid, hands, current player, etc.)
    heatmap : list[list[float]] | None
        Raw 9x9 probability sums for heatmap overlay. Log-scale normalization
        is done client-side.
    square_actions : dict | None
        Per-square top-3 actions keyed by ``"r,c"`` (0-indexed).
    piece_images : dict | None
        Mapping of piece type keys (e.g. ``"pawn_black"``) to base64 data URIs.
    selected_square : dict | None
        Currently selected square ``{"row": int, "col": int}`` from session
        state — used to initialize selection on mount.
    key : str | None
        Streamlit widget key for this component instance.

    Returns
    -------
    dict | None
        The ``selection`` trigger value when a square is selected/deselected,
        or None if no interaction occurred. Shape:
        - ``{"row": int, "col": int, "type": "select"}`` — square selected
        - ``{"type": "deselect"}`` — selection cleared
    """
    result = _component_func(
        key=key,
        data={
            "board_state": board_state,
            "heatmap": heatmap,
            "square_actions": square_actions or {},
            "piece_images": piece_images or {},
            "selected_square": selected_square,
        },
        default={"board_focused": False},
        on_selection_change=lambda: None,
        on_board_focused_change=lambda: None,
    )

    return result
