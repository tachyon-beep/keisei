# Elo Rating History Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a JSONL match log to the scheduler and a line chart in the Ladder tab showing Elo rating progression for the top 10 models.

**Architecture:** Scheduler appends one JSON line per match to `.keisei_ladder/match_log.jsonl`. New `elo_chart.py` module reads the log and builds per-model rating timelines. Chart renders in the Ladder tab between leaderboard and Live Games.

**Tech Stack:** Python 3.13, Streamlit `st.line_chart()`, JSONL file I/O

**Spec:** `docs/superpowers/specs/2026-03-30-elo-history-chart-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `keisei/evaluation/scheduler.py` | Modify | Add `_append_match_log()`, call after match completion |
| `keisei/webui/elo_chart.py` | Create | `build_elo_timelines()` pure function |
| `keisei/webui/streamlit_app.py` | Modify | Add `render_elo_history_section()`, thread log path through rendering chain |
| `tests/unit/test_elo_chart.py` | Create | Timeline builder unit tests |

---

### Task 1: Timeline Builder — Core Tests and Implementation

**Files:**
- Create: `tests/unit/test_elo_chart.py`
- Create: `keisei/webui/elo_chart.py`

- [ ] **Step 1: Write core tests**

```python
"""Tests for Elo rating timeline builder."""

import json
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _write_log(entries, path):
    """Write match result entries to a JSONL file."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _make_result(model_a, model_b, delta_a, delta_b, ts=1.0):
    return {
        "model_a": model_a,
        "model_b": model_b,
        "winner": model_a if delta_a > 0 else model_b,
        "elo_delta_a": delta_a,
        "elo_delta_b": delta_b,
        "move_count": 50,
        "reason": "checkmate",
        "timestamp": ts,
    }


class TestBuildEloTimelines:
    """Tests for build_elo_timelines()."""

    def test_missing_file_returns_empty(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        result = build_elo_timelines(tmp_path / "nonexistent.jsonl")
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "empty.jsonl"
        log.write_text("")
        result = build_elo_timelines(log)
        assert result == {}

    def test_single_match(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        _write_log([_make_result("A", "B", 12.0, -12.0)], log)
        result = build_elo_timelines(log)
        assert "A" in result
        assert "B" in result
        # Initial 1500 + delta after 1 match
        assert result["A"] == [1500.0, 1512.0]
        assert result["B"] == [1500.0, 1488.0]

    def test_multiple_matches_cumulative(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        _write_log(
            [
                _make_result("A", "B", 12.0, -12.0, ts=1.0),
                _make_result("A", "B", 10.0, -10.0, ts=2.0),
            ],
            log,
        )
        result = build_elo_timelines(log)
        assert result["A"] == [1500.0, 1512.0, 1522.0]
        assert result["B"] == [1500.0, 1488.0, 1478.0]

    def test_forward_fill(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        _write_log(
            [
                _make_result("A", "B", 12.0, -12.0, ts=1.0),
                _make_result("A", "C", 8.0, -8.0, ts=2.0),
            ],
            log,
        )
        result = build_elo_timelines(log)
        # B didn't play in match 2, should carry forward
        assert result["B"] == [1500.0, 1488.0, 1488.0]
        # C appears in match 2 — starts at 1500 for match 1, then delta
        assert result["C"] == [1500.0, 1500.0, 1492.0]

    def test_top_n_filtering(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        # Create 12 models, each beats the next
        entries = []
        for i in range(12):
            name = f"model_{i:02d}"
            entries.append(_make_result(name, "baseline", float(i + 1), float(-(i + 1)), ts=float(i)))
        _write_log(entries, log)
        result = build_elo_timelines(log, top_n=3)
        assert len(result) == 3
        # Top 3 by final Elo: model_11, model_10, model_09
        assert "model_11" in result
        assert "model_10" in result
        assert "model_09" in result

    def test_top_n_with_leaderboard(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        _write_log(
            [
                _make_result("A", "B", 12.0, -12.0),
                _make_result("C", "B", 5.0, -5.0),
            ],
            log,
        )
        leaderboard = [
            {"name": "B", "elo": 1483.0},
            {"name": "C", "elo": 1505.0},
        ]
        result = build_elo_timelines(log, top_n=2, leaderboard=leaderboard)
        # Leaderboard says top 2 are B and C (not A)
        assert "B" in result
        assert "C" in result
        assert "A" not in result

    def test_malformed_line_skipped(self, tmp_path):
        from keisei.webui.elo_chart import build_elo_timelines

        log = tmp_path / "log.jsonl"
        with open(log, "w") as f:
            f.write(json.dumps(_make_result("A", "B", 12.0, -12.0)) + "\n")
            f.write("this is not valid json\n")
            f.write(json.dumps(_make_result("A", "B", 10.0, -10.0)) + "\n")
        result = build_elo_timelines(log)
        # Should have parsed 2 valid matches
        assert result["A"] == [1500.0, 1512.0, 1522.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_elo_chart.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'keisei.webui.elo_chart'`

- [ ] **Step 3: Implement build_elo_timelines**

```python
"""Build Elo rating timelines from the match log for charting.

Reads a JSONL match log and reconstructs per-model rating progression
over time, suitable for rendering with ``st.line_chart()``.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_INITIAL_ELO = 1500.0


def build_elo_timelines(
    log_path: Path,
    top_n: int = 10,
    leaderboard: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[float]]:
    """Read a JSONL match log and build per-model Elo timelines.

    Args:
        log_path: Path to the match_log.jsonl file.
        top_n: Number of top models to include in the output.
        leaderboard: If provided, use these model names (in order) to
            determine top-N instead of computing from the log.

    Returns:
        Dict mapping model name to a list of Elo values, one per match
        in the log. Models not involved in a match get their previous
        rating forward-filled. Returns ``{}`` if the log is missing or empty.
    """
    entries = _read_log(log_path)
    if not entries:
        return {}

    # Track current Elo per model
    current_elo: Dict[str, float] = {}
    # Build timeline: one snapshot per match
    snapshots: List[Dict[str, float]] = []

    for entry in entries:
        model_a = entry.get("model_a", "")
        model_b = entry.get("model_b", "")
        delta_a = entry.get("elo_delta_a", 0.0)
        delta_b = entry.get("elo_delta_b", 0.0)

        # Initialize unseen models
        if model_a not in current_elo:
            current_elo[model_a] = _INITIAL_ELO
        if model_b not in current_elo:
            current_elo[model_b] = _INITIAL_ELO

        # Apply deltas
        current_elo[model_a] += delta_a
        current_elo[model_b] += delta_b

        # Snapshot all known models (forward-fill)
        snapshots.append(dict(current_elo))

    # Determine which models to include
    if leaderboard:
        top_names = [e.get("name", "") for e in leaderboard[:top_n]]
        # Only include names that actually appear in the log
        top_names = [n for n in top_names if n in current_elo]
    else:
        # Sort by final Elo descending
        sorted_models = sorted(current_elo.items(), key=lambda x: x[1], reverse=True)
        top_names = [name for name, _ in sorted_models[:top_n]]

    if not top_names:
        return {}

    # Build output: initial value + one entry per match
    all_models = set()
    for snap in snapshots:
        all_models.update(snap.keys())

    result: Dict[str, List[float]] = {}
    for name in top_names:
        timeline = [_INITIAL_ELO]
        for snap in snapshots:
            timeline.append(snap.get(name, timeline[-1]))
        result[name] = timeline

    return result


def _read_log(log_path: Path) -> List[Dict[str, Any]]:
    """Read and parse a JSONL match log, skipping malformed lines."""
    if not log_path.exists():
        return []

    entries: List[Dict[str, Any]] = []
    with open(log_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line %d in %s", line_num, log_path)
    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_elo_chart.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/webui/elo_chart.py tests/unit/test_elo_chart.py
git commit -m "feat(webui): add Elo timeline builder from JSONL match log"
```

---

### Task 2: Scheduler Match Log

**Files:**
- Modify: `keisei/evaluation/scheduler.py`

- [ ] **Step 1: Add _append_match_log method**

Add the following method to `ContinuousMatchScheduler`, after the `_build_state_snapshot` method (around line 308):

```python
    def _append_match_log(self, result: Dict[str, Any]) -> None:
        """Append a match result to the persistent JSONL log."""
        log_path = self._state_path.parent / "match_log.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")
```

- [ ] **Step 2: Call _append_match_log after match completion**

In `_run_match()`, add one line after line 542 (`self._recent_results.append(match_result)`):

```python
            self._recent_results.append(match_result)
            self._append_match_log(match_result)
            self._recent_results = self._recent_results[-50:]
```

- [ ] **Step 3: Verify the scheduler still imports cleanly**

Run: `uv run python -c "from keisei.evaluation.scheduler import ContinuousMatchScheduler; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add keisei/evaluation/scheduler.py
git commit -m "feat(scheduler): append match results to persistent JSONL log"
```

---

### Task 3: Chart Rendering and Tab Integration

**Files:**
- Modify: `keisei/webui/streamlit_app.py`

- [ ] **Step 1: Add render_elo_history_section function**

Add this function after the `render_secondary_board()` function and before `render_ladder_tab()`. Also add `from pathlib import Path` to imports if not already present.

```python
def render_elo_history_section(
    ladder_state: Dict[str, Any],
    log_path: Path,
) -> None:
    """Render the Elo rating history line chart."""
    from keisei.webui.elo_chart import build_elo_timelines

    leaderboard = ladder_state.get("leaderboard", [])
    timelines = build_elo_timelines(log_path, top_n=10, leaderboard=leaderboard)
    if not timelines:
        st.caption("Rating history will appear after matches complete")
        return
    st.subheader("Rating History")
    st.line_chart(timelines)
    st.caption(f"Top {len(timelines)} models by Elo")
```

- [ ] **Step 2: Update render_ladder_tab signature to accept log_path**

Change the function signature from:

```python
def render_ladder_tab(ladder_state: Optional[Dict[str, Any]]) -> None:
```

to:

```python
def render_ladder_tab(
    ladder_state: Optional[Dict[str, Any]],
    log_path: Optional[Path] = None,
) -> None:
```

- [ ] **Step 3: Add the chart section to render_ladder_tab**

Insert the following between the timestamp caption (line ~1298) and the Live Games section (line ~1300), replacing the existing `# --- Live Games section ---` comment block:

```python
    # --- Rating History chart ---
    if log_path is not None:
        st.markdown("---")
        render_elo_history_section(ladder_state, log_path)

    # --- Live Games section ---
```

- [ ] **Step 4: Thread log_path through _render_dashboard_content**

Update `_render_dashboard_content` signature from:

```python
def _render_dashboard_content(
    env: EnvelopeParser,
    ladder_state: Optional[Dict[str, Any]] = None,
) -> None:
```

to:

```python
def _render_dashboard_content(
    env: EnvelopeParser,
    ladder_state: Optional[Dict[str, Any]] = None,
    ladder_log_path: Optional[Path] = None,
) -> None:
```

And update the `render_ladder_tab` call inside it from:

```python
            render_ladder_tab(ladder_state)
```

to:

```python
            render_ladder_tab(ladder_state, log_path=ladder_log_path)
```

- [ ] **Step 5: Compute log_path in main() and pass it through**

In `_live_data_section()` inside `main()`, compute the log path once and pass it to `_render_dashboard_content`. After the line that defines `ladder_state_file` (around line 1387), the log path is:

```python
    # Derive match log path from ladder state file location
    _ladder_dir = Path(ladder_state_file).parent if ladder_state_file else _DEFAULT_LADDER_STATE_PATH.parent
    _match_log_path = _ladder_dir / "match_log.jsonl"
```

Add these two lines after `ladder_state_file` is defined (around line 1387), inside `main()` but before `_live_data_section`.

Then update both calls to `_render_dashboard_content` inside `_live_data_section()`:

From:
```python
            _render_dashboard_content(env, ladder_state=ladder)
```

To:
```python
            _render_dashboard_content(env, ladder_state=ladder, ladder_log_path=_match_log_path)
```

There are two call sites — one in the paused branch (line ~1434) and one in the active branch (line ~1457). Update both.

- [ ] **Step 6: Verify imports and run existing tests**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py tests/unit/test_elo_chart.py -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add keisei/webui/streamlit_app.py
git commit -m "feat(webui): add Elo rating history chart to Ladder tab"
```

---

### Task 4: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run linting**

Run: `uv run flake8 keisei/webui/elo_chart.py keisei/webui/streamlit_app.py keisei/evaluation/scheduler.py --max-line-length=120`
Expected: No errors

- [ ] **Step 2: Run type checking on new file**

Run: `uv run mypy keisei/webui/elo_chart.py --ignore-missing-imports`
Expected: No new errors (pre-existing errors in other files are OK)

- [ ] **Step 3: Run formatting**

Run: `uv run black --check keisei/webui/elo_chart.py keisei/webui/streamlit_app.py keisei/evaluation/scheduler.py`
Expected: Already formatted (or run `uv run black` to fix)

- [ ] **Step 4: Run all related tests**

Run: `uv run pytest tests/unit/test_elo_chart.py tests/unit/test_ladder_dashboard.py tests/unit/test_sfen_utils.py -v`
Expected: All tests PASS

- [ ] **Step 5: Final commit if any formatting changes**

```bash
git add -u
git commit -m "style: format Elo history chart code"
```
