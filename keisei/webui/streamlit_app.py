"""
Streamlit dashboard for Keisei training visualization.

Reads JSON state from a file written atomically by the training thread.
All state access goes through ``EnvelopeParser`` — rendering functions
never access raw envelope keys directly.

Usage:
    streamlit run keisei/webui/streamlit_app.py -- --state-file path/to/state.json
    streamlit run keisei/webui/streamlit_app.py   # demo mode with sample data
"""

import argparse
import base64
import json
import math
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from keisei.webui.board_component import shogi_board
from keisei.webui.envelope_parser import EnvelopeParser

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
    """Load state from JSON file, with fallback to sample data.

    When *state_file* is explicitly provided but the file does not exist yet
    (e.g. training hasn't written its first snapshot), ``None`` is returned so
    the caller can display a "waiting" message rather than misleading demo data.
    Demo / sample data is only used when no *state_file* was requested at all.
    """
    path = Path(state_file) if state_file else None

    if path and path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    # An explicit state-file was given but not (yet) present — do not fall
    # back to demo data as that would display fabricated metrics.
    if path is not None:
        return None

    # Demo mode (no --state-file): use sample state
    if _SAMPLE_STATE_PATH.exists():
        with open(_SAMPLE_STATE_PATH) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------


def _piece_aria_label(piece: Optional[Dict[str, Any]]) -> str:
    """Build an accessible label for a board cell's contents."""
    if piece is None:
        return "empty square"
    ptype = piece["type"].replace("_", " ")
    return f"{piece['color']} {ptype}"


def _compute_heatmap_overlay(
    heatmap: List[List[float]],
) -> List[List[float]]:
    """Normalize a 9x9 probability heatmap to [0, 1] using log scale.

    Log scale makes the full distribution visible when action
    probabilities are extremely peaked (common in Shogi PPO).
    """
    epsilon = 1e-10
    flat = [v for row in heatmap for v in row if v > epsilon]
    if not flat:
        return [[0.0] * 9 for _ in range(9)]
    log_min = math.log(min(flat) + epsilon)
    log_max = math.log(max(flat) + epsilon)
    log_range = log_max - log_min if log_max > log_min else 1.0

    result = []
    for row in heatmap:
        out_row = []
        for v in row:
            if v <= epsilon:
                out_row.append(0.0)
            else:
                norm = (math.log(v + epsilon) - log_min) / log_range
                out_row.append(max(0.0, min(1.0, norm)))
        result.append(out_row)
    return result


def render_board(
    board_state: Dict[str, Any],
    heatmap: Optional[List[List[float]]] = None,
) -> None:
    """Render the shogi board as a semantic HTML table with SVG pieces.

    Uses ``role="table"`` with proper ``<th>`` headers, ``alt`` text on
    piece images, and ``aria-label`` on each cell.  Row labels use
    numeric 1-9 (standard Shogi notation), not alphabetic.

    When *heatmap* is provided (9x9 raw probability sums), a log-scaled
    orange overlay is rendered on each square.
    """
    board = board_state.get("board", [])
    if not board:
        st.info("No board data available")
        return

    move_count = board_state.get("move_count", 0)
    current_player = board_state.get("current_player", "unknown")
    cell_size = 48

    # Pre-compute heatmap overlay if provided
    overlay = _compute_heatmap_overlay(heatmap) if heatmap else None

    # --- Column headers ---
    header_cells = '<th scope="col" aria-hidden="true"></th>'
    for file_num in range(9, 0, -1):
        header_cells += (
            f'<th scope="col" style="text-align:center;'
            f'font-weight:bold;font-size:12px;">{file_num}</th>'
        )

    # --- Board rows (rank 1-9) ---
    body_rows = []
    for r, row in enumerate(board):
        rank = r + 1
        row_header = (
            f'<th scope="row" style="font-weight:bold;'
            f'font-size:12px;padding:2px 4px;">{rank}</th>'
        )
        cells = ""
        for c in range(9):
            piece = row[c]
            file_num = 9 - c
            bg = "#f5deb3" if (r + c) % 2 == 0 else "#deb887"
            aria = _piece_aria_label(piece)
            cell_label = f"{file_num}-{rank}: {aria}"

            # Promotion zone tint: rows 1-3 (White), rows 7-9 (Black)
            zone_tint = ""
            if rank <= 3:
                zone_tint = (
                    "background-image:linear-gradient("
                    "rgba(100,149,237,0.08),rgba(100,149,237,0.08));"
                )
            elif rank >= 7:
                zone_tint = (
                    "background-image:linear-gradient("
                    "rgba(220,80,80,0.08),rgba(220,80,80,0.08));"
                )

            # Promotion zone boundary: thicker borders at ranks 3/6
            border_bottom = "1px solid #8b7355"
            if rank == 3 or rank == 6:
                border_bottom = "2.5px solid #6b5335"

            # Heatmap overlay: blend orange over the cell bg
            heat_style = ""
            if overlay and overlay[r][c] > 0.01:
                alpha = overlay[r][c] * 0.5
                heat_style = (
                    f"box-shadow:inset 0 0 0 100px " f"rgba(0,210,140,{alpha:.2f});"
                )

            cell_content = ""
            if piece is not None:
                key = _piece_image_key(piece)
                svg_uri = _PIECE_SVG_CACHE.get(key, "")
                if svg_uri:
                    img_sz = cell_size - 8
                    cell_content = (
                        f'<img src="{svg_uri}" width="{img_sz}"'
                        f' height="{img_sz}"'
                        f' alt="{aria}">'
                    )
                else:
                    # Fallback: text label (contrast-safe)
                    label = piece["type"][0].upper()
                    if piece.get("promoted"):
                        label = "+" + label
                    txt_color = "#000" if piece["color"] == "black" else "#8b0000"
                    cell_content = (
                        f'<span style="color:{txt_color};'
                        f'font-weight:bold;">{label}</span>'
                    )
            cells += (
                f'<td tabindex="0" data-row="{r}" data-col="{c}"'
                f' aria-label="{cell_label}"'
                f' style="width:{cell_size}px;height:{cell_size}px;'
                f"background:{bg};text-align:center;"
                f"vertical-align:middle;"
                f"border:1px solid #8b7355;"
                f"border-bottom:{border_bottom};"
                f"{zone_tint}"
                f'{heat_style}">'
                f"{cell_content}</td>"
            )
        body_rows.append(f"<tr>{row_header}{cells}</tr>")

    caption = (
        f"Shogi board position, move {move_count}, "
        f"{current_player.capitalize()} to play. "
        f"Black (Sente) plays from bottom, "
        f"White (Gote) from top."
    )

    # Dark theme + keyboard navigation + focus styling
    board_css_and_js = """
    <style>
    @media (prefers-color-scheme: dark) {
      table[role="table"] th { color: #e0e0e0; }
    }
    td[tabindex="0"]:focus {
      outline: 3px solid #4169e1;
      outline-offset: -3px;
      z-index: 1;
      position: relative;
    }
    </style>
    <script>
    document.addEventListener('DOMContentLoaded', function() {
      var cells = document.querySelectorAll('td[tabindex="0"]');
      cells.forEach(function(cell) {
        cell.addEventListener('keydown', function(e) {
          var row = parseInt(cell.dataset.row);
          var col = parseInt(cell.dataset.col);
          var nextRow = row, nextCol = col;
          if (e.key === 'ArrowUp') nextRow = Math.max(0, row - 1);
          else if (e.key === 'ArrowDown') nextRow = Math.min(8, row + 1);
          else if (e.key === 'ArrowLeft') nextCol = Math.max(0, col - 1);
          else if (e.key === 'ArrowRight') nextCol = Math.min(8, col + 1);
          else return;
          e.preventDefault();
          var next = document.querySelector(
            'td[data-row="' + nextRow + '"][data-col="' + nextCol + '"]'
          );
          if (next) next.focus();
        });
      });
    });
    </script>
    """

    table_html = (
        f"{board_css_and_js}"
        f'<table id="shogi-board" role="table"'
        f' style="border-collapse:collapse;margin:auto;"'
        f' aria-label="{caption}">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f'<tbody>{"".join(body_rows)}</tbody>'
        f"</table>"
    )
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
    """Render training metric charts in a 2-column grid.

    Each chart is 220px tall with the current (latest) value annotated.
    Learning rate is shown as an inline metric rather than a chart when
    constant across the sample window.
    """
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
        ("Episode Reward", "episode_rewards"),
    ]

    col1, col2 = st.columns(2)
    for i, (title, key) in enumerate(chart_configs):
        data = curves.get(key, [])
        if not data:
            continue
        target = col1 if i % 2 == 0 else col2
        with target:
            current = data[-1]
            st.caption(f"{title}  ({current:.4g})")
            st.line_chart(data, height=220)

    # Learning rate: inline metric if constant, chart if varying
    lr_data = curves.get("learning_rates", [])
    if lr_data:
        lr_set = set(lr_data)
        if len(lr_set) <= 1:
            st.caption(f"Learning Rate: {lr_data[-1]:.2e}")
        else:
            with col1:
                current = lr_data[-1]
                st.caption(f"Learning Rate  ({current:.2e})")
                st.line_chart(lr_data, height=220)


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


def render_game_status(board_state: Dict[str, Any]) -> None:
    """Render current game status badge with aria-live for updates."""
    player = board_state.get("current_player", "?").capitalize()
    move_count = board_state.get("move_count", 0)
    game_over = board_state.get("game_over", False)
    winner = board_state.get("winner")

    if game_over:
        if winner:
            status_text = (
                f"Game over \u2014 {winner.capitalize()} wins! " f"(Move {move_count})"
            )
            st.success(status_text)
        else:
            status_text = f"Game over \u2014 Draw (Move {move_count})"
            st.warning(status_text)
    else:
        status_text = f"Move {move_count} \u2014 {player} to play"
        st.info(status_text)

    # Scoped aria-live region for screen reader announcements
    st.markdown(
        f'<div role="status" aria-live="polite" '
        f'style="position:absolute;width:1px;height:1px;'
        f'overflow:hidden;clip:rect(0,0,0,0);">'
        f"{status_text}</div>",
        unsafe_allow_html=True,
    )


def render_move_log(step_info: Optional[Dict]) -> None:
    """Render the move log in chronological order with auto-scroll.

    Uses an HTML container with overflow-y and JavaScript to auto-scroll
    to the bottom on each render, so the most recent move is visible.
    """
    if not step_info:
        return
    moves = step_info.get("move_log", [])
    if not moves:
        st.caption("No moves yet")
        return

    st.caption(f"Move Log ({len(moves)} moves)")
    # Build numbered move list (chronological, 1-indexed)
    lines = []
    for i, move in enumerate(moves, 1):
        lines.append(f"{i:>3}. {move}")
    text = "\n".join(lines)

    # Scrollable container, auto-scrolls to bottom
    html = (
        f'<div id="movelog" style="font-family:monospace;'
        f"font-size:13px;max-height:300px;overflow-y:auto;"
        f'padding:8px;background:#f9f9f9;border-radius:4px;">'
        f'<pre style="margin:0;">{text}</pre></div>'
        f"<script>var el=document.getElementById('movelog');"
        f"el.scrollTop=el.scrollHeight;</script>"
    )
    st.components.v1.html(html, height=min(len(moves) * 20 + 40, 340))


# ---------------------------------------------------------------------------
# Policy insight renderers
# ---------------------------------------------------------------------------


def render_position_assessment(insight: Dict[str, Any]) -> None:
    """Render the Position Assessment group: V(s) + confidence."""
    st.caption("Position Assessment")

    # Value estimate with directional label
    v = insight.get("value_estimate", 0.0)
    if v > 0.05:
        label = f"V(s): +{v:.3f} (Black +)"
        color = "#006400"
    elif v < -0.05:
        label = f"V(s): {v:.3f} (White +)"
        color = "#8b0000"
    else:
        label = f"V(s): {v:.3f} (Even)"
        color = "#555555"
    st.markdown(
        f'<span style="font-size:18px;font-weight:bold;'
        f'color:{color};">{label}</span>',
        unsafe_allow_html=True,
    )

    # Confidence meter (entropy-derived)
    entropy = insight.get("action_entropy", 0.0)
    # Normalize: max entropy for 13527 actions is ln(13527) ~ 9.51
    max_entropy = 9.51
    confidence = max(0.0, 1.0 - entropy / max_entropy)
    conf_pct = int(confidence * 100)
    conf_label = (
        "Focused"
        if confidence > 0.7
        else ("Moderate" if confidence > 0.3 else "Uncertain")
    )
    st.progress(confidence, text=f"Confidence: {conf_pct}% ({conf_label})")


def render_top_actions(insight: Dict[str, Any]) -> None:
    """Render the Action Distribution group: top-K horizontal bars."""
    st.caption("Action Distribution")
    top_actions = insight.get("top_actions", [])
    if not top_actions:
        st.text("No action data")
        return

    max_prob = top_actions[0]["prob"] if top_actions else 1.0
    lines = []
    for act in top_actions:
        action_str = act["action"]
        prob = act["prob"]
        pct = prob * 100
        # Bar width proportional to max action
        bar_w = int((prob / max_prob) * 120) if max_prob > 0 else 0
        bar = (
            f'<span style="display:inline-block;width:{bar_w}px;'
            f"height:12px;background:#007850;border-radius:2px;"
            f'vertical-align:middle;"></span>'
        )
        lines.append(
            f'<span style="font-family:monospace;font-size:13px;">'
            f"{action_str:<8s} {pct:5.1f}%  {bar}</span>"
        )
    html = "<br>".join(lines)
    st.markdown(html, unsafe_allow_html=True)


def render_policy_insight_panel(
    insight: Optional[Dict[str, Any]],
) -> None:
    """Render the full policy insight right panel."""
    if not insight:
        return

    render_position_assessment(insight)
    st.divider()
    render_top_actions(insight)


def render_selected_square_panel(
    selected: Dict[str, Any],
    board_state: Dict[str, Any],
    square_actions: Dict[str, List[Dict[str, Any]]],
    heatmap: Optional[List[List[float]]],
    insight_available: bool,
) -> None:
    """Render the selected square detail panel.

    Shows top-3 actions targeting the selected square with probability bars.
    Inserted between Action Distribution and Game Status in the Game tab.
    Only visible when a square is selected.
    """
    row = selected["row"]
    col = selected["col"]
    file_num = 9 - col
    rank = row + 1
    square_label = f"{file_num}{rank}"

    st.caption(f"Square {square_label} actions")

    if not insight_available:
        st.text(
            "Policy insight not available. Enable in config to see action breakdown."
        )
        return

    # Piece context from board_state
    board = board_state.get("board", [])
    if 0 <= row < len(board) and 0 <= col < len(board[row]):
        piece = board[row][col]
        if piece:
            st.text(
                f"Contains: {piece['color'].capitalize()} "
                f"{piece['type'].replace('_', ' ').title()}"
            )

    # Heatmap probability sum for this square
    if heatmap and 0 <= row < len(heatmap) and 0 <= col < len(heatmap[row]):
        prob_sum = heatmap[row][col]
        st.text(f"Probability mass: {prob_sum * 100:.2f}%")

    key = f"{row},{col}"
    actions = square_actions.get(key, [])

    if not actions:
        st.text("No actions target this square")
    else:
        # Render probability bars (same style as render_top_actions)
        max_prob = actions[0]["prob"] if actions else 1.0
        lines = []
        for act in actions:
            action_str = act["action"]
            prob = act["prob"]
            pct = prob * 100
            bar_w = int((prob / max_prob) * 120) if max_prob > 0 else 0
            bar = (
                f'<span style="display:inline-block;width:{bar_w}px;'
                f"height:12px;background:#007850;border-radius:2px;"
                f'vertical-align:middle;"></span>'
            )
            lines.append(
                f'<span style="font-family:monospace;font-size:13px;">'
                f"{action_str:<8s} {pct:5.1f}%  {bar}</span>"
            )
        html = "<br>".join(lines)
        st.markdown(html, unsafe_allow_html=True)
        st.caption("Probabilities are global (share of all possible moves)")

    # Screen reader announcement (WCAG 4.1.3)
    # Gate on selection change — only announce when the selected square
    # differs from the last announcement. Without this gate, the aria-live
    # div fires on every 2s fragment re-render, spamming screen reader users.
    announce_key = f"{row},{col}"
    if st.session_state.get("last_announced_square") != announce_key:
        st.session_state.last_announced_square = announce_key
        if actions:
            announce = (
                f"Selected {square_label}. Top action: "
                f"{actions[0]['action']} {actions[0]['prob']*100:.1f}%."
            )
        else:
            announce = f"Selected {square_label}. No actions target this square."

        st.markdown(
            f'<div role="status" aria-live="polite" '
            f'style="position:absolute;width:1px;height:1px;'
            f'overflow:hidden;clip:rect(0,0,0,0);">'
            f"{announce}</div>",
            unsafe_allow_html=True,
        )


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


_VIEW_LABELS: Dict[str, str] = {
    "league": "League / Elo Rankings",
    "lineage": "Model Lineage",
    "skill_differential": "Skill Differential",
    "model_profile": "Model Profile",
}


def render_lineage_panel(env: EnvelopeParser) -> None:
    """Render the model lineage panel when lineage data is available."""
    lineage = env.lineage
    if lineage is None:
        return

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Generation", lineage.get("generation", 0))
    with c2:
        rating = lineage.get("latest_rating")
        st.metric("Elo Rating", f"{rating:.0f}" if rating is not None else "\u2014")
    with c3:
        st.metric("Events", lineage.get("event_count", 0))

    # Current model info
    model_id = lineage.get("model_id", "\u2014")
    parent_id = lineage.get("parent_id", "\u2014")
    run_name = lineage.get("run_name", "\u2014")
    st.text(f"Model:  {model_id}")
    st.text(f"Parent: {parent_id}")
    st.text(f"Run:    {run_name}")

    # Ancestor chain
    ancestors = lineage.get("ancestor_chain", [])
    if ancestors:
        with st.expander(f"Ancestor chain ({len(ancestors)} models)", expanded=False):
            for i, ancestor_id in enumerate(ancestors):
                prefix = (
                    "\u2514\u2500\u2500 "
                    if i == len(ancestors) - 1
                    else "\u251c\u2500\u2500 "
                )
                st.text(f"{prefix}{ancestor_id}")

    # Recent events
    recent = lineage.get("recent_events", [])
    if recent:
        with st.expander(f"Recent events ({len(recent)})", expanded=False):
            for event in reversed(recent):
                ts = event.get("emitted_at", "?")
                etype = event.get("event_type", "?")
                mid = event.get("model_id", "?")
                st.text(f"{ts}  {etype}  {mid}")


def render_optional_view_placeholders(env: EnvelopeParser) -> None:
    """Show informative placeholders for missing optional views."""
    missing = env.missing_optional_views()
    if not missing:
        return
    with st.expander("Upcoming views (not yet available)", expanded=False):
        for view in missing:
            label = _VIEW_LABELS.get(view, view)
            health = env.view_health(view)
            st.caption(f"{label}  ({health})")


def render_stale_warning(env: EnvelopeParser) -> None:
    """Show a warning banner when the snapshot is stale."""
    if env.is_stale():
        age = int(env.age_seconds())
        st.warning(f"Training data is {age}s old — training may be paused or stopped.")


def _render_header_metrics(env: EnvelopeParser) -> None:
    """Render the always-visible header metrics row above the tabs."""
    metrics = env.metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Timestep", f"{metrics.get('global_timestep', 0):,}")
    with c2:
        st.metric("Episodes", f"{metrics.get('total_episodes', 0):,}")
    with c3:
        total = metrics.get("total_episodes", 0)
        bw = metrics.get("black_wins", 0)
        wr = f"{bw / total * 100:.1f}%" if total > 0 else "\u2014"
        st.metric("Black Win Rate", wr)
    with c4:
        st.metric("Speed", f"{env.speed:.0f} steps/s")

    if metrics.get("processing"):
        st.info("PPO update in progress...")


def render_metrics_tab(env: EnvelopeParser) -> None:
    """Render the Metrics tab: training curves, win rates, buffer."""
    # Clear board focus when not on Game tab (prevents stuck focus-pause)
    st.session_state.board_focused = False
    metrics = env.metrics
    model_info = env.model_info

    render_training_charts(metrics)

    # Summary row: win rates, gradient norm, buffer
    summary1, summary2 = st.columns(2)
    with summary1:
        render_win_rate_chart(metrics)
    with summary2:
        grad_norm = model_info.get("gradient_norm")
        if grad_norm is not None:
            st.metric("Gradient Norm", f"{grad_norm:.4f}")
        render_buffer_bar(env.buffer_info)


def render_game_tab(env: EnvelopeParser) -> None:
    """Render the Game tab: board, policy insight, game status, moves."""
    board_state = env.board_state
    step_info = env.step_info
    metrics = env.metrics
    insight = env.policy_insight

    if not board_state:
        st.info("Waiting for first episode...")
        return

    # Selected square invalidation: clear when move counter changes
    prev_move = st.session_state.get("last_move_count")
    curr_move = board_state.get("move_count", 0)
    if prev_move is not None and prev_move != curr_move:
        st.session_state.selected_square = None
    st.session_state.last_move_count = curr_move

    # Skip-to-board link (accessibility: Phase 4, item 30)
    st.markdown(
        '<a href="#shogi-board" style="font-size:0;position:absolute;">'
        "Skip to board</a>",
        unsafe_allow_html=True,
    )

    # Heatmap overlay — controlled by sidebar toggle
    heatmap = None
    if st.session_state.get("show_heatmap", False) and insight:
        heatmap = insight.get("action_heatmap")

    # Square actions for the component and panel
    square_actions = insight.get("square_actions", {}) if insight else {}

    board_col, insight_col = st.columns([2, 3])
    with board_col:
        # Interactive board component (v2 API) with fallback
        try:
            board_event = shogi_board(
                board_state=board_state,
                heatmap=heatmap,
                square_actions=square_actions,
                piece_images=_PIECE_SVG_CACHE,
                selected_square=st.session_state.get("selected_square"),
                key="main_board",
            )
        except Exception:
            # Fallback to non-interactive board rendering
            render_board(board_state, heatmap=heatmap)
            board_event = None

        # Process events from the v2 component
        if board_event is not None:
            # Selection trigger (transient — fires once per click)
            selection = getattr(board_event, "selection", None)
            if selection:
                event_type = selection.get("type")
                if event_type == "select":
                    st.session_state.selected_square = {
                        "row": selection["row"],
                        "col": selection["col"],
                    }
                elif event_type == "deselect":
                    st.session_state.selected_square = None
                    st.session_state.last_announced_square = None

            # Focus state (persistent)
            focused = getattr(board_event, "board_focused", None)
            if focused is not None:
                st.session_state.board_focused = bool(focused)
                if focused:
                    st.session_state.board_focus_timestamp = time.time()

        render_hands(board_state)

    with insight_col:
        render_policy_insight_panel(insight)

        # Selected square detail panel
        selected = st.session_state.get("selected_square")
        if selected:
            st.divider()
            render_selected_square_panel(
                selected=selected,
                board_state=board_state,
                square_actions=square_actions,
                heatmap=heatmap,
                insight_available=insight is not None,
            )

        render_game_status(board_state)
        render_move_log(step_info)
        if step_info:
            st.caption("Move Statistics")
            sc = step_info.get("sente_capture_count", 0)
            gc = step_info.get("gote_capture_count", 0)
            sd = step_info.get("sente_drop_count", 0)
            gd = step_info.get("gote_drop_count", 0)
            sp = step_info.get("sente_promo_count", 0)
            gp = step_info.get("gote_promo_count", 0)
            st.text(f"Captures: Black {sc} / White {gc}")
            st.text(f"Drops:    Black {sd} / White {gd}")
            st.text(f"Promos:   Black {sp} / White {gp}")

        # Hot squares (text when no heatmap data)
        if not insight:
            hot_squares = metrics.get("hot_squares", [])
            if hot_squares:
                st.caption("Hot Squares: " + ", ".join(str(s) for s in hot_squares))


def render_lineage_tab(env: EnvelopeParser) -> None:
    """Render the Lineage tab: generation tree, Elo, events."""
    st.session_state.board_focused = False
    render_lineage_panel(env)


def _render_dashboard_content(env: EnvelopeParser) -> None:
    """Render header metrics + tab-based content from a parsed envelope."""
    _render_header_metrics(env)

    # Build tab list — Lineage only shown when data is available
    tab_names = ["Metrics", "Game"]
    if env.has_view("lineage"):
        tab_names.append("Lineage")

    tabs = st.tabs(tab_names)
    with tabs[0]:
        render_metrics_tab(env)
    with tabs[1]:
        render_game_tab(env)
    if len(tabs) > 2:
        with tabs[2]:
            render_lineage_tab(env)


def main() -> None:
    st.set_page_config(
        page_title="Keisei Training Dashboard",
        page_icon="\u265f",
        layout="wide",
    )

    # Parse --state-file from Streamlit's extra args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--state-file", default=None, help="Path to state JSON file"
    )
    args, _ = arg_parser.parse_known_args()
    state_file: Optional[str] = args.state_file

    # --- Sidebar controls (outside the fragment — rendered once) ---
    with st.sidebar:
        st.toggle("Auto-refresh", value=True, key="auto_refresh")
        st.toggle("Show heatmap", value=False, key="show_heatmap")

    st.title("Keisei Training Dashboard")

    # --- Live data fragment ---
    # The fragment always runs on a 2-second timer.  When paused, it
    # renders from the cached last_state instead of reloading from disk.
    # This avoids the decorator-evaluation-time pitfall: run_every is
    # captured once at definition time, so we cannot change it dynamically.
    @st.fragment(run_every=timedelta(seconds=2))
    def _live_data_section() -> None:
        # Timeout fallback: auto-clear stuck board_focused after 30s
        if st.session_state.get("board_focused", False):
            last_focus_time = st.session_state.get("board_focus_timestamp", 0)
            if time.time() - last_focus_time > 30:
                st.session_state.board_focused = False

        paused_by_toggle = not st.session_state.get("auto_refresh", True)
        paused_by_focus = st.session_state.get("board_focused", False)

        if paused_by_toggle or paused_by_focus:
            # Paused — render from cached state, skip disk read
            cached = st.session_state.get("last_state")
            if cached is None:
                st.info("Paused. No cached state available.")
                return
            env = EnvelopeParser(cached)
            if paused_by_focus:
                st.info("Paused — inspecting board")
            render_stale_warning(env)
            _render_dashboard_content(env)
        else:
            state = load_state(state_file)
            if state is None:
                st.error("No state data available. Waiting for training to start...")
                return

            env = EnvelopeParser(state)
            st.session_state.last_state = state

            # Clear selection and focus when board disappears between episodes
            if env.board_state is None:
                st.session_state.selected_square = None
                st.session_state.last_move_count = None

            render_stale_warning(env)
            _render_dashboard_content(env)

    _live_data_section()

    # Export button outside the fragment (fragments can't write to sidebar)
    current = st.session_state.get("last_state")
    if current is not None:
        with st.sidebar:
            st.download_button(
                label="Export state",
                data=json.dumps(current, indent=2),
                file_name="keisei_state.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
