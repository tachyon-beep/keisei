# Recent Match Results Feed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a color-coded feed of last 20 match results to the Ladder tab, between the Elo chart and Live Games.

**Architecture:** One pure formatting function (`format_match_result`) + one rendering function (`render_recent_results_section`), both in `streamlit_app.py`. Data comes from existing `recent_results` in the ladder state — no new data sources.

**Tech Stack:** Python 3.13, Streamlit markdown with `:green[]` / `:red[]` color syntax

**Spec:** `docs/superpowers/specs/2026-03-30-recent-results-feed-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `keisei/webui/streamlit_app.py` | Modify | Add `format_match_result()`, `render_recent_results_section()`, extend `render_ladder_tab()` |
| `tests/unit/test_ladder_dashboard.py` | Modify | Add `TestFormatMatchResult` class |

---

### Task 1: format_match_result — Tests and Implementation

**Files:**
- Modify: `tests/unit/test_ladder_dashboard.py`
- Modify: `keisei/webui/streamlit_app.py`

- [ ] **Step 1: Write tests**

Add to the end of `tests/unit/test_ladder_dashboard.py`:

```python
class TestFormatMatchResult:
    """Tests for format_match_result helper."""

    def _make_result(self, winner="model_a", delta_a=12.0, delta_b=-12.0, moves=142):
        return {
            "model_a": "model_a",
            "model_b": "model_b",
            "winner": winner,
            "elo_delta_a": delta_a,
            "elo_delta_b": delta_b,
            "move_count": moves,
            "reason": "checkmate",
            "timestamp": 1.0,
        }

    def test_win_by_model_a(self):
        from keisei.webui.streamlit_app import format_match_result

        result = format_match_result(self._make_result(winner="model_a"))
        assert ":green[**model_a**]" in result
        assert ":red[**model_b**]" in result
        assert "beat" in result
        assert "+12" in result
        assert "-12" in result
        assert "142 moves" in result

    def test_win_by_model_b(self):
        from keisei.webui.streamlit_app import format_match_result

        result = format_match_result(self._make_result(winner="model_b", delta_a=-8.0, delta_b=8.0))
        assert ":red[**model_a**]" in result
        assert ":green[**model_b**]" in result
        assert "lost to" in result

    def test_draw(self):
        from keisei.webui.streamlit_app import format_match_result

        result = format_match_result(self._make_result(winner="draw", delta_a=0.0, delta_b=0.0))
        assert ":green[" not in result
        assert ":red[" not in result
        assert "drew" in result
        assert "**model_a**" in result
        assert "**model_b**" in result

    def test_delta_formatting(self):
        from keisei.webui.streamlit_app import format_match_result

        result = format_match_result(self._make_result(delta_a=12.5, delta_b=-12.5))
        assert "+12.5" in result or "+13" in result  # rounded is fine

    def test_missing_winner_treated_as_draw(self):
        from keisei.webui.streamlit_app import format_match_result

        r = self._make_result()
        del r["winner"]
        result = format_match_result(r)
        assert "drew" in result

    def test_missing_move_count(self):
        from keisei.webui.streamlit_app import format_match_result

        r = self._make_result()
        del r["move_count"]
        result = format_match_result(r)
        assert "? moves" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py::TestFormatMatchResult -v`
Expected: FAIL — `ImportError: cannot import name 'format_match_result'`

- [ ] **Step 3: Implement format_match_result**

Add this function to `keisei/webui/streamlit_app.py`, after the `format_elo_arrow()` function and before `select_display_matches()`:

```python
def format_match_result(result: Dict[str, Any]) -> str:
    """Format a match result as a color-coded Streamlit markdown string."""
    model_a = result.get("model_a", "?")
    model_b = result.get("model_b", "?")
    winner = result.get("winner")
    delta_a = result.get("elo_delta_a", 0.0)
    delta_b = result.get("elo_delta_b", 0.0)
    moves = result.get("move_count")
    moves_str = f"{moves} moves" if moves is not None else "? moves"

    da = f"+{delta_a:.0f}" if delta_a >= 0 else f"{delta_a:.0f}"
    db = f"+{delta_b:.0f}" if delta_b >= 0 else f"{delta_b:.0f}"

    if winner == model_a:
        return (
            f":green[**{model_a}**] beat :red[**{model_b}**]"
            f" \u2014 {da} / {db} Elo, {moves_str}"
        )
    elif winner == model_b:
        return (
            f":red[**{model_a}**] lost to :green[**{model_b}**]"
            f" \u2014 {da} / {db} Elo, {moves_str}"
        )
    else:
        return (
            f"**{model_a}** drew **{model_b}**"
            f" \u2014 {da} / {db} Elo, {moves_str}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py::TestFormatMatchResult -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/webui/streamlit_app.py tests/unit/test_ladder_dashboard.py
git commit -m "feat(webui): add format_match_result with color-coded output"
```

---

### Task 2: Render Section and Tab Integration

**Files:**
- Modify: `keisei/webui/streamlit_app.py`

- [ ] **Step 1: Add render_recent_results_section**

Add this function after `format_match_result()` and before `select_display_matches()`:

```python
def render_recent_results_section(recent_results: List[Dict[str, Any]]) -> None:
    """Render a feed of recent match results with color-coded outcomes."""
    if not recent_results:
        st.caption("No completed matches yet")
        return
    st.subheader("Recent Results")
    for result in reversed(recent_results):
        st.markdown(format_match_result(result))
```

- [ ] **Step 2: Add section to render_ladder_tab**

Find the comment `# --- Live Games section ---` (around line 1325). Insert BEFORE it:

```python
    # --- Recent Results feed ---
    if recent_results:
        st.markdown("---")
        render_recent_results_section(recent_results)
```

The `recent_results` variable is already extracted at the top of `render_ladder_tab()`.

- [ ] **Step 3: Run all ladder dashboard tests**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run full related test suite**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py tests/unit/test_elo_chart.py tests/unit/test_sfen_utils.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/webui/streamlit_app.py
git commit -m "feat(webui): add recent match results feed to Ladder tab"
```

---

### Task 3: Final Verification

**Files:** None (verification only)

- [ ] **Step 1: Run linting**

Run: `uv run flake8 keisei/webui/streamlit_app.py --max-line-length=120`
Expected: No errors

- [ ] **Step 2: Run formatting check**

Run: `uv run black --check keisei/webui/streamlit_app.py`
Expected: Already formatted (or run `uv run black` to fix)

- [ ] **Step 3: Final test run**

Run: `uv run pytest tests/unit/test_ladder_dashboard.py tests/unit/test_elo_chart.py tests/unit/test_sfen_utils.py -q`
Expected: All tests PASS

- [ ] **Step 4: Commit if any formatting changes**

```bash
git add -u
git commit -m "style: format recent results feed code"
```
