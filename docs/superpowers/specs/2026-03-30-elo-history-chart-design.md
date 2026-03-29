# Elo Rating History Chart Design

**Date:** 2026-03-30
**Issue:** keisei-754ca3852f
**Status:** Draft

## Overview

Add a line chart showing Elo rating progression over time for the top 10 models in the ladder. Each model is a separate line. X-axis is match sequence number, Y-axis is Elo rating. Placed in the Ladder tab between the leaderboard table and the Live Games section.

## Goals

- Show how models' Elo ratings evolve over the course of a ladder session
- Persist match history so the chart survives scheduler restarts
- Top 10 models by current Elo — readable without filtering controls

## Non-Goals

- User-selectable model filtering (future enhancement)
- SQLite or database storage (future refactor)
- Timestamp-based X-axis (uneven spacing; match sequence is cleaner)

## Architecture

### 1. Match Log (Scheduler Side)

**File:** `keisei/evaluation/scheduler.py`

Add a `_append_match_log(result: Dict[str, Any])` method to `ContinuousMatchScheduler` that appends one JSON line to a JSONL file after each completed match.

**Log path:** `self._state_path.parent / "match_log.jsonl"` (defaults to `.keisei_ladder/match_log.jsonl`)

**Log entry format:** Same shape as `recent_results` entries (already constructed at line 532-541):

```json
{"model_a": "ckpt_200", "model_b": "ckpt_180", "winner": "ckpt_200", "elo_delta_a": 12.1, "elo_delta_b": -12.1, "move_count": 45, "reason": "checkmate", "timestamp": 1711756800.0}
```

**Integration point:** Called at line ~542, right after `self._recent_results.append(match_result)`:

```python
self._recent_results.append(match_result)
self._append_match_log(match_result)
```

**Implementation:**
```python
def _append_match_log(self, result: Dict[str, Any]) -> None:
    log_path = self._state_path.parent / "match_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(result) + "\n")
```

No locking — the scheduler is the only writer. Directory creation is already handled by `_publish_state()`.

### 2. Timeline Builder (`webui/elo_chart.py`)

**New file:** `keisei/webui/elo_chart.py`

Pure function:

```python
def build_elo_timelines(
    log_path: Path,
    top_n: int = 10,
    leaderboard: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[float]]
```

**Logic:**
1. Read JSONL file line by line. If file missing or empty, return `{}`.
2. Track current Elo per model, starting each at 1500.0 (initial rating).
3. After each match, update both participants' Elo using the deltas.
4. Build per-model rating lists: each list has one entry per match in the log. Models not involved in a match get their previous rating forward-filled.
5. Determine top N: if `leaderboard` provided, use top N names from it. Otherwise, use top N by final computed Elo.
6. Return `{model_name: [elo_at_match_0, elo_at_match_1, ...]}` — only for the top N models.

**Output shape for `st.line_chart()`:** The returned dict is directly passable — Streamlit interprets dict keys as column names and list values as Y-axis data.

**Edge cases:**
- Log file doesn't exist: return `{}`
- Log file exists but empty: return `{}`
- Fewer than N models: return all of them
- Malformed JSON line: skip it (log a warning), continue

### 3. Chart Rendering

**File:** `keisei/webui/streamlit_app.py`

New function `render_elo_history_section(ladder_state, log_path)`:

```python
def render_elo_history_section(
    ladder_state: Dict[str, Any],
    log_path: Path,
) -> None:
    leaderboard = ladder_state.get("leaderboard", [])
    timelines = build_elo_timelines(log_path, top_n=10, leaderboard=leaderboard)
    if not timelines:
        st.caption("Rating history will appear after matches complete")
        return
    st.subheader("Rating History")
    st.line_chart(timelines)
    st.caption(f"Top {len(timelines)} models by Elo")
```

**Integration in `render_ladder_tab()`:** Called between the timestamp caption and the Live Games divider. The log path is derived from the ladder state file path (passed through from `main()`).

### 4. Passing the Log Path

The Streamlit app already knows the ladder state file path (from `--ladder-state-file` CLI arg or the default). The log path is derived:

```python
log_path = Path(ladder_state_file or _DEFAULT_LADDER_STATE_PATH).parent / "match_log.jsonl"
```

`render_ladder_tab()` signature changes to accept the log path:

```python
def render_ladder_tab(ladder_state: Optional[Dict[str, Any]], log_path: Optional[Path] = None) -> None:
```

Default `None` means the chart section is skipped (backwards compatible).

## Data Flow

```
ContinuousMatchScheduler._run_match()
    │ match completes, Elo updated
    ▼
_append_match_log(result)
    │ append one JSON line
    ▼
.keisei_ladder/match_log.jsonl
    │ read by Streamlit every 2s refresh
    ▼
build_elo_timelines(log_path, top_n=10, leaderboard)
    │ reconstruct per-model Elo series
    ▼
st.line_chart(timelines)
    ▼
Browser
```

## Testing

### `tests/unit/test_elo_chart.py`

| Test | Description |
|------|-------------|
| Empty/missing log | Returns `{}` |
| Single match | Two models, each with 2 data points (1500 + delta) |
| Multiple matches | Correct cumulative tracking |
| Forward-fill | Non-participating models carry previous rating |
| Top-N filtering | Only top N by final Elo returned |
| Top-N with leaderboard | Uses leaderboard names for filtering |
| Malformed line skipped | Bad JSON line doesn't crash, rest of log parsed |

### No tests for:
- `_append_match_log()` — one-liner file append
- `render_elo_history_section()` — thin Streamlit wrapper

## Files Changed

| File | Change |
|------|--------|
| `keisei/evaluation/scheduler.py` | Add `_append_match_log()` method, call it after match completion |
| `keisei/webui/elo_chart.py` | **New** — `build_elo_timelines()` pure function |
| `keisei/webui/streamlit_app.py` | Add `render_elo_history_section()`, extend `render_ladder_tab()` signature and body, pass log path from `main()` |
| `tests/unit/test_elo_chart.py` | **New** — timeline builder tests |
