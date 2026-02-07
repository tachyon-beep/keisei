"""
Streamlit dashboard for Keisei training visualization.

Standalone app — imports NO keisei code. Reads JSON state from a file
written atomically by the training thread.

Usage:
    streamlit run keisei/webui/streamlit_app.py -- --state-file path/to/state.json
    streamlit run keisei/webui/streamlit_app.py   # demo mode with sample data
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# SVG piece cache — loaded once at import time
# ---------------------------------------------------------------------------
_IMAGES_DIR = Path(__file__).parent / "static" / "images"
_PIECE_SVG_CACHE: Dict[str, str] = {}


def _load_piece_svgs() -> None:
    """Pre-load all SVG piece images as base64 data URIs."""
    if _PIECE_SVG_CACHE:
        return  # Already loaded
    if not _IMAGES_DIR.exists():
        return
    for svg_file in _IMAGES_DIR.glob("*.svg"):
        raw = svg_file.read_bytes()
        b64 = base64.b64encode(raw).decode()
        _PIECE_SVG_CACHE[svg_file.stem] = f"data:image/svg+xml;base64,{b64}"


_load_piece_svgs()

# Piece type to SVG filename stem mapping
_PROMOTED_TYPES = {
    "promoted_pawn",
    "promoted_lance",
    "promoted_knight",
    "promoted_silver",
    "promoted_bishop",
    "promoted_rook",
}


def _piece_image_key(piece: Dict[str, Any]) -> str:
    """Map a piece dict to its SVG filename stem (e.g. 'pawn_black')."""
    ptype = piece["type"]
    color = piece["color"]
    if piece.get("promoted") and ptype not in _PROMOTED_TYPES:
        ptype = f"promoted_{ptype}"
    return f"{ptype}_{color}"


# ---------------------------------------------------------------------------
# State loading
# ---------------------------------------------------------------------------

_SAMPLE_STATE_PATH = Path(__file__).parent / "sample_state.json"


def load_state(state_file: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load state from JSON file, with fallback to sample data."""
    path = Path(state_file) if state_file else None

    if path and path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    # Demo mode: use sample state
    if _SAMPLE_STATE_PATH.exists():
        with open(_SAMPLE_STATE_PATH) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------


def render_board(board_state: Dict[str, Any]) -> None:
    """Render the shogi board as an HTML table with SVG pieces."""
    board = board_state.get("board", [])
    if not board:
        st.info("No board data available")
        return

    cell_size = 48
    html_rows = []
    # Column headers (9 to 1, standard shogi notation)
    header = "<tr><td></td>"
    for c in range(9, 0, -1):
        header += f'<td style="text-align:center;font-weight:bold;font-size:12px;">{c}</td>'
    header += "</tr>"
    html_rows.append(header)

    row_labels = "abcdefghi"
    for r, row in enumerate(board):
        tr = f'<td style="font-weight:bold;font-size:12px;padding:2px 4px;">{row_labels[r]}</td>'
        for c in range(9):
            piece = row[c]
            bg = "#f5deb3" if (r + c) % 2 == 0 else "#deb887"
            cell_content = ""
            if piece is not None:
                key = _piece_image_key(piece)
                svg_uri = _PIECE_SVG_CACHE.get(key, "")
                if svg_uri:
                    cell_content = f'<img src="{svg_uri}" width="{cell_size - 8}" height="{cell_size - 8}">'
                else:
                    # Fallback: text label
                    label = piece["type"][0].upper()
                    if piece["promoted"]:
                        label = "+" + label
                    color = "#000" if piece["color"] == "black" else "#c00"
                    cell_content = f'<span style="color:{color};font-weight:bold;">{label}</span>'
            tr += (
                f'<td style="width:{cell_size}px;height:{cell_size}px;'
                f'background:{bg};text-align:center;vertical-align:middle;'
                f'border:1px solid #8b7355;">{cell_content}</td>'
            )
        html_rows.append(f"<tr>{tr}</tr>")

    table_html = (
        '<table style="border-collapse:collapse;margin:auto;">'
        + "".join(html_rows)
        + "</table>"
    )
    # Add enough height for the board plus padding
    st.components.v1.html(table_html, height=cell_size * 10 + 40)


def render_hands(board_state: Dict[str, Any]) -> None:
    """Render captured pieces (hands) for both players."""
    black_hand = board_state.get("black_hand", {})
    white_hand = board_state.get("white_hand", {})

    def _format_hand(hand: Dict[str, int], label: str) -> str:
        if not hand:
            return f"**{label}:** (empty)"
        pieces = ", ".join(f"{k.capitalize()}: {v}" for k, v in sorted(hand.items()))
        return f"**{label}:** {pieces}"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(_format_hand(white_hand, "White (Gote) hand"))
    with col2:
        st.markdown(_format_hand(black_hand, "Black (Sente) hand"))


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------


def render_training_charts(metrics: Dict[str, Any]) -> None:
    """Render training metric charts in a 2-column grid."""
    curves = metrics.get("learning_curves", {})
    if not curves:
        st.info("No training data yet")
        return

    chart_configs = [
        ("Policy Loss", "policy_losses"),
        ("Value Loss", "value_losses"),
        ("Entropy", "entropies"),
        ("KL Divergence", "kl_divergences"),
        ("Clip Fraction", "clip_fractions"),
        ("Episode Length", "episode_lengths"),
        ("Episode Reward", "episode_rewards"),
    ]

    col1, col2 = st.columns(2)
    for i, (title, key) in enumerate(chart_configs):
        data = curves.get(key, [])
        if not data:
            continue
        target = col1 if i % 2 == 0 else col2
        with target:
            st.caption(title)
            st.line_chart(data, height=150)


def render_win_rate_chart(metrics: Dict[str, Any]) -> None:
    """Render win rate trends."""
    history = metrics.get("win_rates_history", [])
    if not history:
        return

    import pandas as pd

    df = pd.DataFrame(history)
    rename = {
        "win_rate_black": "Black %",
        "win_rate_white": "White %",
        "win_rate_draw": "Draw %",
    }
    df = df.rename(columns=rename)
    st.caption("Win Rate Trends")
    st.line_chart(df[list(rename.values())], height=200)


# ---------------------------------------------------------------------------
# Game status
# ---------------------------------------------------------------------------


def render_game_status(board_state: Dict[str, Any], step_info: Optional[Dict]) -> None:
    """Render current game status and recent moves."""
    col1, col2 = st.columns(2)
    with col1:
        player = board_state.get("current_player", "?").capitalize()
        move_count = board_state.get("move_count", 0)
        game_over = board_state.get("game_over", False)
        winner = board_state.get("winner")

        if game_over:
            if winner:
                st.success(f"Game over — {winner.capitalize()} wins! (Move {move_count})")
            else:
                st.warning(f"Game over — Draw (Move {move_count})")
        else:
            st.info(f"Move {move_count} — {player} to play")

    with col2:
        if step_info:
            moves = step_info.get("move_log", [])
            if moves:
                st.caption("Recent Moves")
                # Show most recent first, up to 10
                for move in reversed(moves[-10:]):
                    st.text(move)
            else:
                st.caption("No moves yet")


def render_buffer_bar(buffer_info: Optional[Dict]) -> None:
    """Render experience buffer fill level."""
    if not buffer_info:
        return
    size = buffer_info.get("size", 0)
    capacity = buffer_info.get("capacity", 1)
    ratio = size / capacity if capacity > 0 else 0
    st.caption(f"Experience Buffer ({size}/{capacity})")
    st.progress(min(ratio, 1.0))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Keisei Training Dashboard",
        page_icon="♟",
        layout="wide",
    )

    # Parse --state-file from Streamlit's extra args
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", default=None, help="Path to state JSON file")
    args, _ = parser.parse_known_args()

    state = load_state(args.state_file)

    if state is None:
        st.error("No state data available. Waiting for training to start...")
        time.sleep(2)
        st.rerun()
        return

    # Stale detection
    ts = state.get("timestamp", 0)
    age = time.time() - ts
    if age > 30:
        st.warning(f"Training data is {int(age)}s old — training may be paused or stopped.")

    metrics = state.get("metrics", {})
    board_state = state.get("board_state")
    step_info = state.get("step_info")
    buffer_info = state.get("buffer_info")

    # --- Header metrics ---
    st.title("Keisei Training Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Timestep", f"{metrics.get('global_timestep', 0):,}")
    with c2:
        st.metric("Episodes", f"{metrics.get('total_episodes', 0):,}")
    with c3:
        total = metrics.get("total_episodes", 0)
        bw = metrics.get("black_wins", 0)
        wr = f"{bw / total * 100:.1f}%" if total > 0 else "—"
        st.metric("Black Win Rate", wr)
    with c4:
        st.metric("Speed", f"{state.get('speed', 0):.0f} steps/s")

    # --- Processing indicator ---
    if metrics.get("processing"):
        st.info("PPO update in progress...")

    # --- Main content ---
    if board_state:
        board_col, chart_col = st.columns([2, 3])
        with board_col:
            st.subheader("Board")
            render_board(board_state)
            render_hands(board_state)
        with chart_col:
            st.subheader("Training Curves")
            render_training_charts(metrics)
    else:
        st.subheader("Training Curves")
        render_training_charts(metrics)

    # --- Lower panels ---
    lower1, lower2 = st.columns(2)
    with lower1:
        if board_state:
            render_game_status(board_state, step_info)
        if step_info:
            st.caption("Move Statistics")
            sc, gc = step_info.get("sente_capture_count", 0), step_info.get("gote_capture_count", 0)
            sd, gd = step_info.get("sente_drop_count", 0), step_info.get("gote_drop_count", 0)
            sp, gp = step_info.get("sente_promo_count", 0), step_info.get("gote_promo_count", 0)
            st.text(f"Captures: Black {sc} / White {gc}")
            st.text(f"Drops:    Black {sd} / White {gd}")
            st.text(f"Promos:   Black {sp} / White {gp}")
    with lower2:
        render_win_rate_chart(metrics)
        render_buffer_bar(buffer_info)

    # --- Auto-refresh ---
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    main()
