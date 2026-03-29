# Recent Match Results Feed Design

**Date:** 2026-03-30
**Issue:** keisei-c7bee70c23
**Status:** Draft

## Overview

Add a scrolling feed of recent match results to the Ladder tab, between the Elo rating history chart and the Live Games section. Each result shows winner/loser with color coding and Elo deltas.

## Goals

- Display last 20 completed matches with color-coded outcomes
- Newest results first
- Green for winner, red for loser, neutral for draws

## Non-Goals

- Click-to-replay (future enhancement)
- Filtering or search
- Persistent storage (reads from existing `recent_results` in ladder state)

## Architecture

### New Functions in `keisei/webui/streamlit_app.py`

#### `format_match_result(result: Dict[str, Any]) -> str`

Pure function that converts a match result dict into a Streamlit-markdown string.

**Win format:**
```
:green[**Model A**] (1523) beat :red[**Model B**] (1487) — +12 / -12 Elo, 142 moves
```

**Draw format:**
```
**Model A** (1500) drew **Model B** (1500) — +0 / -0 Elo, 200 moves
```

**Logic:**
1. Extract `model_a`, `model_b`, `winner`, `elo_delta_a`, `elo_delta_b`, `move_count`
2. Compute display Elo: current Elo is `leaderboard_elo`, but we don't have that here — use deltas to show change direction only. Format deltas as `+12 / -12`.
3. If `winner == model_a`: model_a green, model_b red
4. If `winner == model_b`: model_b green, model_a red
5. If `winner == "draw"`: both neutral (no color wrap)
6. Verb: "beat" for wins, "drew" for draws

**Edge cases:**
- Missing `winner` field: treat as draw
- Missing `move_count`: show "? moves"
- Missing model names: show "?"

#### `render_recent_results_section(recent_results: List[Dict[str, Any]]) -> None`

Renders the full section:

```python
def render_recent_results_section(recent_results: List[Dict[str, Any]]) -> None:
    if not recent_results:
        st.caption("No completed matches yet")
        return
    st.subheader("Recent Results")
    for result in reversed(recent_results):
        st.markdown(format_match_result(result))
```

### Changes to `render_ladder_tab()`

Insert between the rating history chart and the Live Games section:

```python
    # --- Recent Results feed ---
    if recent_results:
        st.markdown("---")
        render_recent_results_section(recent_results)
```

The `recent_results` variable is already available in `render_ladder_tab()` (extracted at line ~1179).

## Testing

### `TestFormatMatchResult` in `tests/unit/test_ladder_dashboard.py`

| Test | Description |
|------|-------------|
| Win by model_a | model_a green, model_b red, verb "beat" |
| Win by model_b | model_b green, model_a red, verb "beat" |
| Draw | Both neutral, verb "drew" |
| Delta formatting | Positive shows +, negative shows - |
| Missing winner | Treated as draw |
| Missing move_count | Shows "? moves" |

## Files Changed

| File | Change |
|------|--------|
| `keisei/webui/streamlit_app.py` | Add `format_match_result()`, `render_recent_results_section()`, extend `render_ladder_tab()` |
| `tests/unit/test_ladder_dashboard.py` | Add `TestFormatMatchResult` class |
