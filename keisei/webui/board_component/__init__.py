"""Custom Streamlit component: interactive shogi board with click-to-inspect.

Uses ``declare_component`` with a vanilla HTML/JS frontend (no npm build).
The component sends click, keyboard selection, and focus events back to
Python via ``Streamlit.setComponentValue()``.

See: docs/superpowers/specs/2026-03-26-board-interactivity-design.md, Section 4.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit.components.v1 as components

_FRONTEND_DIR = str(Path(__file__).parent / "frontend")

_component_func = components.declare_component("shogi_board", path=_FRONTEND_DIR)


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
        state — used to initialize roving tabindex at the selected cell on mount.
    key : str | None
        Streamlit widget key for this component instance.

    Returns
    -------
    dict | None
        Interaction event dict, or None if no interaction occurred:
        - ``{"row": int, "col": int, "type": "select"}`` — square selected
        - ``{"type": "deselect"}`` — selection cleared
        - ``{"type": "focus"}`` — board gained focus
        - ``{"type": "blur"}`` — board lost focus

    Note: Streamlit replays the last ``setComponentValue`` on every fragment
    re-run. The Python side should handle focus/blur events idempotently
    (compare against current session state before setting).
    """
    return _component_func(
        board_state=board_state,
        heatmap=heatmap,
        square_actions=square_actions or {},
        piece_images=piece_images or {},
        selected_square=selected_square,
        key=key,
        default=None,
    )
