# Live Spectator Boards Design

**Date:** 2026-03-29
**Issue:** keisei-6b03a469b2
**Status:** Draft

## Overview

Add live game board rendering to the Ladder tab in the Streamlit dashboard. Displays up to 3 simultaneous spectated matches: one primary board (large, with move log and hand pieces) and two secondary boards (compact, with promotion buttons). Reads match state from the existing ladder state file (``.keisei_ladder/state.json``).

## Goals

- Show spectated matches as rendered Shogi boards in the Ladder tab
- Primary board shows the "most interesting" match (highest combined Elo) by default
- Users can manually pin a match as primary; pin persists across game completions
- Promotion API supports future integrations (e.g., Twitch voting)
- Auto-rotation: when a match completes, it disappears and the next match fills the slot

## Non-Goals

- Policy insight (heatmaps, top actions, V(s)) — future enhancement
- Changes to the scheduler's broadcast format
- New Streamlit components or custom JS
- Interactive board features (click-to-select squares)

## Architecture

### New File: `keisei/webui/sfen_utils.py`

Pure utility module with a single public function:

```python
def sfen_to_board_state(sfen: str) -> Dict[str, Any]
```

**Input:** Standard SFEN string, e.g., ``lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1``

**Output:** Same dict shape as ``extract_board_state()`` produces:

```python
{
    "board": [[Optional[{"type": str, "color": str, "promoted": bool}], ...], ...],  # 9x9
    "current_player": "black" | "white",
    "move_count": int,
    "game_over": False,
    "winner": None,
    "black_hand": {"pawn": 2, "lance": 1, ...},
    "white_hand": {"knight": 1, ...},
}
```

**SFEN piece mapping:**
- Uppercase = black (sente): K, R, B, G, S, N, L, P
- Lowercase = white (gote): k, r, b, g, s, n, l, p
- ``+`` prefix = promoted (e.g., ``+R`` = promoted rook / dragon)
- Digits = empty squares
- ``/`` = row separator

**Hand parsing:** ``2P1L`` → ``{"pawn": 2, "lance": 1}``. ``-`` = empty hand.

**Error handling:** Raises ``ValueError`` on malformed SFEN with a descriptive message.

### New Functions in `keisei/webui/streamlit_app.py`

#### `select_display_matches(matches, pinned_match_id)`

```python
def select_display_matches(
    matches: List[Dict[str, Any]],
    pinned_match_id: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """Select primary and secondary matches for display.

    Returns (primary, secondaries, effective_pin) where effective_pin
    is the pinned_match_id if still valid, or None if cleared.
    """
```

**Logic:**
1. Filter to spectated matches with a valid ``sfen`` field
2. If ``pinned_match_id`` is set and that match is in the filtered list, it becomes primary
3. Otherwise, primary = match with highest ``model_a.elo + model_b.elo``
4. Secondaries = up to 2 remaining matches, sorted by combined Elo descending
5. Returns ``effective_pin`` so the caller can clear stale pins from session state

#### `set_primary_match(match_id)`

```python
def set_primary_match(match_id: Optional[str]) -> None:
    """Pin a match as primary, or pass None to reset to auto-selection.

    This is the promotion API — the same entry point that UI buttons
    and future integrations (e.g., Twitch voting) use.
    """
    st.session_state["ladder_primary_match_id"] = match_id
```

#### `render_primary_board(match)`

Renders the primary board panel inside a Streamlit container:
- Player names and Elo ratings (header row)
- ``render_board(board_state, cell_size=40)`` — the existing static board renderer
- Hand pieces for both players (single compact line each)
- Move log (last 10 moves, ~5 visible lines, monospace)
- Status line: move count + current player

Calls ``sfen_to_board_state(match["sfen"])`` to convert the SFEN string.

#### `render_secondary_board(match)`

Renders a compact secondary board:
- Player names and Elo ratings (smaller text)
- ``render_board(board_state, cell_size=32)``
- Status line: move count + current player
- ``st.button("Watch", on_click=set_primary_match, args=(match["match_id"],))``

No hand pieces, no move log — minimal chrome.

### Changes to `render_ladder_tab()`

After the existing leaderboard table, add a "Live Games" section:

```python
st.markdown("---")
st.subheader("Live Games")

matches = ladder_state.get("matches", [])
pinned = st.session_state.get("ladder_primary_match_id")
primary, secondaries, effective_pin = select_display_matches(matches, pinned)

if effective_pin != pinned:
    st.session_state["ladder_primary_match_id"] = effective_pin

if primary is None:
    st.info("No live games — waiting for spectated matches")
else:
    if secondaries:
        primary_col, secondary_col = st.columns([3, 2])
    else:
        primary_col = st.container()
        secondary_col = None

    with primary_col:
        render_primary_board(primary)

    if secondary_col is not None:
        with secondary_col:
            for match in secondaries:
                render_secondary_board(match)
            if st.session_state.get("ladder_primary_match_id"):
                st.button(
                    "↻ Auto-select primary",
                    on_click=set_primary_match,
                    args=(None,),
                )
```

## Layout

```
┌─────────────────────────────────────────────────────┐
│  Existing Leaderboard Table (unchanged)             │
├─────────────────────────────────────────────────────┤
│  Live Games                                         │
│  ┌──────────────────────┬────────────────────────┐  │
│  │  PRIMARY (60%)       │  SECONDARY 1           │  │
│  │                      │  ┌──────────────────┐  │  │
│  │  Player A  1687  vs  │  │ Board (32px)     │  │  │
│  │  Player B  1623      │  │ Move 31 · White  │  │  │
│  │                      │  │ [Watch]          │  │  │
│  │  ┌────────────────┐  │  └──────────────────┘  │  │
│  │  │ Board (40px)   │  │                        │  │
│  │  │                │  │  SECONDARY 2           │  │
│  │  └────────────────┘  │  ┌──────────────────┐  │  │
│  │  ☗ Hand: P×2, L×1   │  │ Board (32px)     │  │  │
│  │  ☖ Hand: N×1        │  │ Move 17 · Black  │  │  │
│  │  Move log (last 10) │  │ [Watch]          │  │  │
│  │  Move 48 · Black    │  └──────────────────┘  │  │
│  │                      │                        │  │
│  │                      │  [↻ Auto-select]       │  │
│  └──────────────────────┴────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**Edge cases:**
- 0 spectated matches → ``st.info`` message, no columns
- 1 match → primary only, full-width container, no secondary column
- 2 matches → primary + 1 secondary
- 3 matches → primary + 2 secondaries (max spectated slots)

## Data Flow

```
ContinuousMatchScheduler
    │ _publish_state() every move (spectated slots)
    ▼
.keisei_ladder/state.json
    │ read by Streamlit every 2s (existing refresh)
    ▼
render_ladder_tab(ladder_state)
    │ ladder_state["matches"] filtered to spectated
    ▼
select_display_matches(matches, pinned_id)
    │ returns (primary, secondaries)
    ▼
sfen_to_board_state(match["sfen"])
    │ SFEN string → board_state dict
    ▼
render_board(board_state, cell_size=40|32)
    │ existing static HTML board renderer
    ▼
Browser
```

## Testing

### `tests/unit/test_sfen_utils.py`

| Test | Description |
|------|-------------|
| Starting position | All 40 pieces in correct positions |
| Mid-game with promotions | ``+R``, ``+B`` etc. map to ``promoted: True`` |
| Hand pieces — multiple types | ``2P1L3N`` parses correctly for both players |
| Hand pieces — empty | ``-`` → empty dicts |
| Hand pieces — single piece | ``P`` → ``{"pawn": 1}`` |
| Current player — black | ``b`` in SFEN → ``"black"`` |
| Current player — white | ``w`` in SFEN → ``"white"`` |
| Move count | Fourth field parsed as int |
| Invalid SFEN — too few fields | Raises ``ValueError`` |
| Invalid SFEN — bad board rows | Raises ``ValueError`` |
| Invalid SFEN — unknown piece | Raises ``ValueError`` |

### `tests/unit/test_select_display_matches.py`

| Test | Description |
|------|-------------|
| Empty matches | Returns ``(None, [], None)`` |
| Single match | Returns ``(match, [], None)`` |
| Three matches, no pin | Primary = highest combined Elo |
| Three matches, valid pin | Pinned match is primary |
| Stale pin (match gone) | Falls back to auto, clears pin |
| Non-spectated matches filtered | Only ``spectated: True`` with ``sfen`` included |
| Two matches | Primary + 1 secondary |

## Files Changed

| File | Change |
|------|--------|
| ``keisei/webui/sfen_utils.py`` | **New** — SFEN parser |
| ``keisei/webui/streamlit_app.py`` | Add 4 functions + extend ``render_ladder_tab()`` |
| ``tests/unit/test_sfen_utils.py`` | **New** — SFEN parser tests |
| ``tests/unit/test_select_display_matches.py`` | **New** — match selection tests |
