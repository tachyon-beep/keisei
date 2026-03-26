# Streamlit Dashboard v2.1 — Board Interactivity Design

**Status**: Reviewed (incorporating UX specialist feedback)
**Date**: 2026-03-26
**Parent**: [Streamlit Dashboard v2 Design](../../designs/streamlit-dashboard-v2.md) (Sections 6.3, 6.4, 8 Phase v2.1)

---

## 1. Overview

This spec covers the five features deferred from the v2 dashboard release to v2.1:

1. **StepManager observation cache** — push wiring for live policy insight during training
2. **Per-square action data** — state contract extension for square-level action breakdown
3. **Custom board component** — bidirectional Streamlit component replacing `st.components.v1.html()`
4. **Selected square detail panel** — UI panel showing actions targeting a clicked square
5. **Focus-pause behavior** — auto-refresh pauses when the board has keyboard/mouse focus

### Scope Boundary

This spec does NOT cover:
- Changes to the PPO agent or training loop
- New policy insight extraction logic (already implemented in `state_snapshot.py`)
- Board visual design changes (SVG pieces, promotion zones — completed in v2)

---

## 2. StepManager Observation Cache

### Problem

`extract_policy_insight()` in `state_snapshot.py` needs a current observation tensor to run a forward pass through the PPO agent. Currently this only works in demo mode (via `sample_state.json`). During live training, no observation is available to the snapshot builder.

### Design

**Push model** (per v2 design doc Section 2.2): `StepManager` pushes the current observation into a stash attribute after each step. The snapshot builder reads it via an explicit parameter.

#### StepManager Changes (`keisei/training/step_manager.py`)

Add an attribute initialized in `__init__`:

```python
self._latest_obs_for_snapshot: Optional[torch.Tensor] = None
```

At the end of `execute_step()`, after a successful step produces `next_obs_tensor`, stash a detached clone:

```python
self._latest_obs_for_snapshot = step_result.next_obs_tensor.detach().clone()
```

The `detach().clone()` is essential: a bare reference would alias a tensor that may be modified in-place by environment reset or the next step.

Clear on episode reset (in `_clear_episode_counters()`):

```python
self._latest_obs_for_snapshot = None
```

#### build_snapshot Changes (`keisei/webui/state_snapshot.py`)

Add optional parameter:

```python
def build_snapshot(
    trainer,
    speed: float,
    pending_updates: dict,
    latest_observation: Optional[torch.Tensor] = None,
) -> dict:
```

When `latest_observation` is not None and policy insight is enabled, call `extract_policy_insight()` with it.

#### StreamlitManager Changes (`keisei/webui/streamlit_manager.py`)

In `_write_if_due()`, pass the observation:

```python
obs = trainer.step_manager._latest_obs_for_snapshot
snapshot = build_snapshot(trainer, speed, pending_updates, latest_observation=obs)
```

### Threading Safety

Policy insight extraction is gated on the snapshot write cycle (every ~500ms at 2 Hz). The observation is a detached clone — no aliasing risk. Extraction is skipped when `metrics.processing == True` (PPO update in progress) to avoid reading partially-updated weights.

---

## 3. State Contract Extension

### New Types (`keisei/webui/view_contracts.py`)

```python
class SquareAction(TypedDict):
    """A single action targeting a specific board square."""
    action: str   # USI notation, e.g. "7g7f" or "P*5e"
    prob: float   # probability in [0, 1]
```

Extend `PolicyInsight`:

```python
class PolicyInsight(TypedDict, total=False):
    action_heatmap: list[list[float]]
    top_actions: list[TopAction]
    value_estimate: float
    action_entropy: float
    # NEW in v2.1:
    square_actions: dict[str, list[SquareAction]]  # "r,c" -> top-3 actions
```

### Producer Logic (`extract_policy_insight`)

For each of the 13,527 actions, determine the destination square (already done for the heatmap). Group actions by destination. For each destination square with at least one action above the threshold (prob > 0.001), keep the top-3 by probability. Key format: `"r,c"` where r and c are 0-indexed board coordinates (e.g., `"0,4"` for rank 1, file 5).

### Size Bound

Worst case: 81 squares x 3 actions x ~40 bytes each = ~10 KB. Typical case with a focused policy: 10-20 populated squares, ~1-2 KB. This is negligible relative to the existing state file (~5-15 KB).

### Coordinate Conversion Reference

A single authoritative mapping used by both Python and JS:

| row, col (0-indexed) | Square (Shogi notation) | Example |
|----------------------|------------------------|---------|
| row=0, col=0 | 9-1 (file 9, rank 1) | Top-left (White's lance) |
| row=0, col=4 | 5-1 (file 5, rank 1) | Top-center (White's king) |
| row=8, col=8 | 1-9 (file 1, rank 9) | Bottom-right (Black's lance) |

**Formula**: `file = 9 - col`, `rank = row + 1`. Both Python and JS must use this single conversion.

### Schema Version

No bump required. `square_actions` is an optional field addition (patch-level per the versioning policy in `view_contracts.py`). Old consumers ignore unknown keys.

---

## 4. Custom Board Component

### Architecture

Replace the current `st.components.v1.html()` board rendering with a proper bidirectional custom Streamlit component using `declare_component`.

**No npm/Node.js build step.** The frontend is a single vanilla HTML/JS file served directly by Streamlit's static file serving. This matches the existing codebase's zero-JS-toolchain approach.

### File Structure

```
keisei/webui/
├── board_component/
│   ├── __init__.py              # Python: declare_component + wrapper function
│   └── frontend/
│       └── index.html           # Vanilla JS: board renderer + interaction
├── streamlit_app.py             # Updated: calls shogi_board() instead of render_board()
└── ...
```

### Python Wrapper (`board_component/__init__.py`)

```python
import streamlit.components.v1 as components
from pathlib import Path

_component_func = components.declare_component(
    "shogi_board",
    path=str(Path(__file__).parent / "frontend"),
)

def shogi_board(
    board_state: dict,
    heatmap: list[list[float]] | None = None,
    square_actions: dict[str, list[dict]] | None = None,
    piece_images: dict[str, str] | None = None,
    key: str | None = None,
) -> dict | None:
    """Render the shogi board with click-to-inspect support.

    Returns a dict with interaction state:
        {"row": int, "col": int, "type": "select"} — square selected
        {"type": "deselect"} — selection cleared
        {"type": "focus"} — board gained focus
        {"type": "blur"} — board lost focus
        None — no interaction this render cycle
    """
    return _component_func(
        board_state=board_state,
        heatmap=heatmap,
        square_actions=square_actions or {},
        piece_images=piece_images or {},
        key=key,
        default=None,
    )
```

### JS Frontend (`board_component/frontend/index.html`)

Single HTML file containing:

1. **Streamlit component bootstrap**: Load `streamlit-component-lib.js` from Streamlit's CDN, call `Streamlit.setFrameHeight()` after render.

2. **Board renderer**: Port the existing `render_board()` HTML generation to JS. Receives `board_state`, `heatmap`, `piece_images` (base64 SVG data URIs) as component args via `Streamlit.RENDER_EVENT`. Renders the same semantic HTML table with:
   - Promotion zone tints (rows 1-3, 7-9)
   - Rank 3/6 thicker borders
   - Heatmap overlay (log-scaled orange)
   - SVG piece images
   - `role="grid"` with `role="gridcell"` on each `<td>` (see ARIA section below)
   - ARIA labels on each cell, updated on every `RENDER_EVENT`
   - Grid `aria-label` updated on each render: `"Shogi board, move {N}, {player} to play"`

3. **Click handler**: On cell click, toggle selection state. If clicking the already-selected square, deselect. Call `Streamlit.setComponentValue({row, col, type: "select"})` or `{type: "deselect"}`.

4. **Keyboard handler**:
   - **Arrow keys**: Navigate between cells (existing logic), updating roving tabindex
   - **Enter/Space**: Toggle selection — if the focused cell is already selected, deselect it; if not, select it. This mirrors the click toggle behavior exactly.
   - **Escape**: Deselect the current selection (if any)
   - **Tab/Shift-Tab**: Exit the grid immediately (do not cycle through cells). This prevents the board from being a Tab key trap. Arrow keys are the sole within-grid navigation mechanism, per the standard grid widget contract.

5. **Selection highlight**: The selected square gets a distinct visual treatment from the focus indicator:
   - **Focus-only** (navigating with arrows): `2px dashed #888` outline — muted, indicates cursor position
   - **Selected** (clicked or Enter/Space): `3px solid #2a52b0` outline + `rgba(42,82,176,0.15)` semi-transparent fill — distinct from both focus ring and heatmap overlay (orange)
   - **Both focused and selected** (same cell): shows the selected style (solid outline + fill)
   - **Contrast note**: `#2a52b0` achieves 3.1:1 against the lighter wheat cell (`#f5deb3`), meeting WCAG 1.4.11 non-text contrast. The previous `#4169e1` was 2.9:1, failing 1.4.11.

6. **Focus tracking**: `focusin` / `focusout` on the table element, with gating:
   - Maintain a JS-side `boardFocused` boolean. Only send `{type: "focus"}` via `setComponentValue` when transitioning from `false` → `true` (i.e., focus enters the grid from outside). Do NOT send on every cell-to-cell arrow navigation.
   - On `focusout`, debounce by 100ms before sending `{type: "blur"}`. This prevents false blur→focus cycles when arrow keys move focus between cells. Only send blur if no cell in the grid has focus after the debounce.

7. **Iframe height**: Set a fixed frame height matching the board dimensions (`cell_size * 10 + 40`) rather than calling `setFrameHeight()` with computed height on every render. This prevents page scroll jitter during re-renders.

### Piece Image Passing

The current `_PIECE_SVG_CACHE` dict (base64 data URIs keyed by piece type like `"pawn_black"`) is passed as a component arg `piece_images`. The JS frontend uses these URIs directly in `<img>` tags. This avoids duplicating SVG loading logic in JS.

### Component Return Value Protocol

The component communicates back to Python via `Streamlit.setComponentValue()`. The return value is a dict with a `type` field:

| `type` | Fields | Meaning |
|--------|--------|---------|
| `"select"` | `row`, `col` | Square selected (0-indexed) |
| `"deselect"` | — | Selection cleared |
| `"focus"` | — | Board gained keyboard/mouse focus |
| `"blur"` | — | Board lost focus |

Python reads this as the return value of `shogi_board()`.

**Note on event timing**: Only the most recent interaction per render cycle is processed. If the user clicks multiple squares faster than the 2s fragment cycle, intermediate selections are intentionally dropped. The user sees the final selection reflected on the next render. This is correct for a read-only analysis tool — the user is interested in inspecting the current selection, not replaying all clicks.

### ARIA Role and Roving Tabindex

The board switches from `role="table"` to `role="grid"` now that it has interactive cells. This was explicitly deferred in v2 (Decision #9) pending click-to-inspect. Each `<td>` gets `role="gridcell"`.

**Roving tabindex** (required by the grid contract): Only one cell in the grid has `tabindex="0"` at a time; all others have `tabindex="-1"`. On mount, initialize `tabindex="0"` on cell (0,0), or on the pre-existing `selected_square` if one is passed as a component arg. When the user arrows between cells, shift `tabindex="0"` to the newly-focused cell and set `tabindex="-1"` on the previous one. This ensures Tab enters the grid at the last-focused cell and Tab exits immediately (only one cell is in the tab sequence).

---

## 5. Selected Square Detail Panel

### Location

Inserted in the Game tab right column between "Action Distribution" and "Game Status" — only visible when a square is selected. No layout shifts in the existing sections; the panel appears/disappears as a new section.

### Renderer (`streamlit_app.py`)

New function `render_selected_square_panel()`:

```python
def render_selected_square_panel(
    selected: dict,          # {"row": int, "col": int}
    square_actions: dict,    # from PolicyInsight
    heatmap: list[list[float]] | None,
) -> None:
```

**Content:**

1. **Header**: "Square 76 actions" (compact Shogi notation without dash, avoiding collision with move notation like "7-6" which looks like "from 7 to 6"). Conversion: file = 9 - col, rank = row + 1, display as `f"{file}{rank}"`.
2. **Probability sum**: The heatmap value for this square (raw, not log-scaled), shown as a percentage of total policy mass
3. **Action list**: Top-3 actions from `square_actions["r,c"]`, rendered as horizontal probability bars (same visual style as the global top-actions)
4. **Sub-caption**: "Probabilities are global (share of all possible moves)" — prevents misreading percentages as relative to the selected square
5. **Empty states** (two variants):
   - When `square_actions` key is absent or empty for this square: "No actions target this square"
   - When `insight` is None (policy data unavailable): "Policy insight not available. Enable in config to see action breakdown."
6. **Piece context**: If the square contains a piece, show "Contains: Black Pawn" (from board_state) for context

**Screen reader announcement** (WCAG 4.1.3 — Status Messages):

Add a scoped `aria-live="polite"` off-screen div in `render_selected_square_panel()`, following the existing pattern from `render_game_status()`. When the panel renders with new content, announce:
- With actions: "Selected 76. Top action: 7g7f 23.1%."
- Empty: "Selected 76. No actions target this square."

This ensures keyboard users who select via Enter/Space receive confirmation without needing to navigate to the panel.

### Integration in `render_game_tab()`

```python
with insight_col:
    render_policy_insight_panel(insight)
    # NEW: selected square detail
    if selected_square and insight:
        square_actions = insight.get("square_actions", {})
        render_selected_square_panel(selected_square, square_actions, heatmap)
    render_game_status(board_state)
    render_move_log(step_info)
    # ... move statistics, hot squares ...
```

---

## 6. Selected Square Invalidation

### Trigger

When `board_state.move_count` changes between fragment render cycles, the selected square is stale (it referred to a position that no longer exists).

### Implementation

At the top of the fragment's render function, before rendering:

```python
# Clear selection when board disappears (between episodes)
if board_state is None:
    st.session_state.selected_square = None
    st.session_state.last_move_count = None
else:
    # Clear selection when position changes (new move or new episode)
    prev_move = st.session_state.get("last_move_count")
    curr_move = board_state.get("move_count", 0)
    if prev_move is not None and prev_move != curr_move:
        st.session_state.selected_square = None
    st.session_state.last_move_count = curr_move
```

This handles three transitions:
- **Move played**: `move_count` changes → clear selection
- **Episode boundary**: `board_state` becomes None → clear selection
- **New episode**: `move_count` resets to 0 (differs from previous) → clear selection

---

## 7. Focus-Pause Behavior

### Mechanism

When the board component reports focus, the fragment skips state reload and renders from cached state — the same mechanism as the manual pause toggle.

### Implementation

In the fragment, after reading the board component's return value:

```python
board_result = shogi_board(...)

# Track focus state from component
if board_result and board_result.get("type") == "focus":
    st.session_state.board_focused = True
elif board_result and board_result.get("type") == "blur":
    st.session_state.board_focused = False
```

**Stuck-state prevention**: The board component only sends `blur` when the user actively moves focus out of the grid. If the user navigates to a different tab or closes the browser tab, no `blur` event fires — `board_focused` gets stuck `True`. To prevent this:

```python
# At the top of the fragment, before rendering tab content:
# If we're not rendering the Game tab, the board is not focused
if active_tab_index != GAME_TAB_INDEX:
    st.session_state.board_focused = False
```

Since `st.tabs()` doesn't expose the active tab index directly, track it via `st.session_state.active_tab_index` set by a tab-selection callback, or clear `board_focused` at the start of every non-Game-tab render path (Metrics tab and Lineage tab both set `board_focused = False`).

### Pause Precedence

**Manual toggle takes precedence over focus-pause.** The combined logic:

```python
paused_by_toggle = not st.session_state.get("auto_refresh", True)
paused_by_focus = st.session_state.get("board_focused", False)

if paused_by_toggle or paused_by_focus:
    cached = st.session_state.get("last_state")
    # ... render from cache ...
```

When `auto_refresh` is toggled back on while the board is focused, auto-refresh remains paused (focus-pause still active). The status indicator distinguishes the two states so the user understands why.

### Status Indicator

Placed in the **header row above the tabs** (always visible regardless of sidebar state), using `st.info()` at the top of the fragment's paused render branch:

- Manual toggle off: `st.info("Paused")`
- Board focused (auto-refresh on): `st.info("Paused — inspecting board")`
- Both: `st.info("Paused — inspecting board")`

### Focus Debounce

The JS frontend debounces `focusout` by 100ms before sending a `blur` event. This prevents false blur→focus cycles when the user tabs between board cells (focus leaves one cell and enters another within the same table). Without debounce, each arrow-key press would trigger blur→focus→blur→focus. See also Section 4 item 6 for the JS-side `boardFocused` gating.

---

## 8. Migration: render_board() to shogi_board()

### What Changes

The current `render_board()` function in `streamlit_app.py` (lines 144-311) generates an HTML string and renders it via `st.components.v1.html()`. This is replaced by the `shogi_board()` component call.

### What Moves

The HTML/CSS/JS generation logic moves from Python string templates to the JS frontend (`index.html`). The board visual design (cell sizes, colors, promotion zones, heatmap overlay, piece images) is preserved exactly.

### What Stays

- `_load_piece_svgs()` and `_PIECE_SVG_CACHE` stay in `streamlit_app.py` — they produce the base64 data URIs passed to the component
- `_compute_heatmap_overlay()` moves to JS (the log-scale normalization is simple arithmetic, better done client-side to avoid recomputing on each Python render)
- `_piece_aria_label()` moves to JS
- `render_hands()` stays as-is (it uses native Streamlit widgets, not the board component)

### Backward Compatibility

The old `render_board()` function is removed. No external consumers — it's only called from `render_game_tab()`.

---

## 9. Session State Keys

| Key | Type | Purpose |
|-----|------|---------|
| `selected_square` | `dict \| None` | `{"row": int, "col": int}` or None |
| `last_move_count` | `int \| None` | Previous move_count for invalidation |
| `board_focused` | `bool` | Whether board component has focus |
| `auto_refresh` | `bool` | Manual pause toggle (existing) |
| `show_heatmap` | `bool` | Heatmap overlay toggle (existing) |
| `last_state` | `dict \| None` | Cached state for pause rendering (existing) |

---

## 10. File Change Summary

| File | Change | Description |
|------|--------|-------------|
| `keisei/training/step_manager.py` | Minor | Add `_latest_obs_for_snapshot` attribute, stash after each step |
| `keisei/webui/state_snapshot.py` | Extend | Add `latest_observation` parameter to `build_snapshot()`, pass to `extract_policy_insight()` |
| `keisei/webui/streamlit_manager.py` | Minor | Pass observation to `build_snapshot()` |
| `keisei/webui/view_contracts.py` | Extend | Add `SquareAction` TypedDict, add `square_actions` to `PolicyInsight` |
| `keisei/webui/board_component/__init__.py` | New | Python wrapper for custom component |
| `keisei/webui/board_component/frontend/index.html` | New | Vanilla JS board renderer with interaction |
| `keisei/webui/streamlit_app.py` | Modify | Replace `render_board()` with `shogi_board()`, add selected square panel, focus-pause logic, invalidation |
| `keisei/webui/envelope_parser.py` | No change | `policy_insight` property already returns `square_actions` (it returns the full dict) |
| `keisei/webui/sample_state.json` | Extend | Add sample `square_actions` data for demo mode |

---

## 11. Testing Strategy

### Unit Tests

- `test_square_action_extraction`: Verify `extract_policy_insight()` produces correct `square_actions` — top-3 per square, threshold filtering, correct key format
- `test_selected_square_invalidation`: Verify `selected_square` clears when `move_count` changes
- `test_obs_cache_lifecycle`: Verify `_latest_obs_for_snapshot` is set after step, cleared on episode reset

### Integration Tests

- `test_board_component_render`: Verify `shogi_board()` returns None when no interaction, returns selection dict on simulated click
- `test_focus_pause_integration`: Verify fragment renders from cache when `board_focused` is True
- `test_live_policy_insight`: Verify end-to-end: StepManager stashes obs → build_snapshot receives it → PolicyInsight includes square_actions

### Manual Testing

- Demo mode: `streamlit run keisei/webui/streamlit_app.py` — verify click-to-inspect works with sample data
- Live training: verify policy insight updates during training, focus-pause works, invalidation clears stale selections

---

## 12. Resolved Design Decisions

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Per-square data approach | Top-3 per destination square (hybrid) | Bounded size (~10KB worst case), directly usable, no client-side action mapping needed |
| 2 | Component architecture | `declare_component` with vanilla HTML (no npm) | Official bidirectional API without adding build toolchain dependency |
| 3 | Detail panel placement | Insert below Action Distribution | Static layout — no shifts when no square selected. Matches design doc's "separate panel" language |
| 4 | Focus-pause scope | Pause entire fragment | When inspecting a square, the entire state should be frozen — right panel data corresponds to board position |
| 5 | ARIA role | Switch to `role="grid"` with roving tabindex | Interactive cells require grid contract. Roving tabindex ensures only one cell in tab sequence. |
| 6 | Focus debounce | 100ms on blur, gated focusin | Prevents false blur→focus cycles; avoids flooding setComponentValue on arrow navigation |
| 7 | Focus vs. selection visual | Dashed muted outline (focus) vs. solid blue + fill (selected) | Must be independently recognizable when user navigates away from selected cell |
| 8 | Selection highlight color | `#2a52b0` (not `#4169e1`) | 3.1:1 against wheat background, meeting WCAG 1.4.11 non-text contrast |
| 9 | Panel header notation | "Square 76 actions" (no dash) | Dash format "7-6" collides with move notation. Compact format matches standard Shogi square references |
| 10 | Focus-pause stuck state | Clear `board_focused` on non-Game-tab render | Prevents auto-refresh from permanently stopping when user leaves Game tab |
| 11 | Pause precedence | Manual toggle takes precedence; focus-pause additive | Both contribute to pause state; status indicator distinguishes the reason |
| 12 | Tab behavior in grid | Tab/Shift-Tab exit immediately | Prevents 81-cell Tab trap; arrow keys are the within-grid navigation mechanism |
| 13 | Iframe height | Fixed height, not computed per render | Prevents page scroll jitter during re-renders |
