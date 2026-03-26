# Board Interactivity v2.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add click-to-inspect board interactivity to the Streamlit training dashboard, enabling per-square policy action inspection with a custom bidirectional component.

**Architecture:** Replace the one-directional `st.components.v1.html()` board with a `declare_component` custom component that sends click/focus events back to Python. Extend the state contract with per-square action data. Add focus-pause and selection invalidation to the fragment lifecycle.

**Tech Stack:** Python 3.13, Streamlit >= 1.55 (`declare_component`), vanilla HTML/JS (no npm), PyTorch (observation caching)

**Spec:** `docs/superpowers/specs/2026-03-26-board-interactivity-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `keisei/webui/board_component/__init__.py` | Python wrapper: `declare_component` + `shogi_board()` function |
| `keisei/webui/board_component/frontend/index.html` | Vanilla JS board renderer with click/keyboard/focus interaction |
| `tests/unit/test_board_component.py` | Unit tests for the Python wrapper |
| `tests/unit/test_square_actions.py` | Unit tests for per-square action extraction |

### Modified Files

| File | Change |
|------|--------|
| `keisei/webui/view_contracts.py` | Add `SquareAction` TypedDict, `square_actions` field to `PolicyInsight` |
| `keisei/webui/state_snapshot.py` | Add `square_actions` computation to `extract_policy_insight()` |
| `keisei/training/step_manager.py` | Already done — `_latest_obs_for_snapshot` (np.ndarray) and `_latest_legal_mask_for_snapshot` (tensor) exist |
| `keisei/webui/streamlit_manager.py` | No change needed (already reads `_latest_obs_for_snapshot` via `_build_training_view`) |
| `keisei/webui/streamlit_app.py` | Replace `render_board()` with `shogi_board()`, add selected square panel, invalidation, focus-pause |
| `keisei/webui/sample_state.json` | Add `square_actions` sample data to existing `policy_insight` |
| `tests/unit/test_step_manager.py` | Add obs cache lifecycle tests |
| `tests/integration/test_webui_state_snapshot.py` | Add square_actions integration test |

---

## Task 1: StepManager Observation Cache Tests (Implementation Already Done)

**Status:** The `_latest_obs_for_snapshot` and `_latest_legal_mask_for_snapshot` attributes already exist in `step_manager.py` (added previously). Key facts about the current implementation:
- `_latest_obs_for_snapshot` is a **numpy array** (`np.ndarray`), NOT a tensor — set via `self._latest_obs_for_snapshot = episode_state.current_obs` at line 269
- `_latest_legal_mask_for_snapshot` is a **tensor** — set at line 270
- Both initialized to `None` in `__init__` (lines 98-99)
- Both are NOT cleared in `_clear_episode_counters()` (they persist across episodes)
- `_build_training_view()` in `state_snapshot.py:284-295` already reads both via `getattr` and passes them to `extract_policy_insight()`

**No implementation changes needed. This task adds only the lifecycle tests.**

**Files:**
- Test: `tests/unit/test_step_manager.py`

- [ ] **Step 1: Write lifecycle tests for obs and legal mask cache**

Add to `tests/unit/test_step_manager.py` (use the existing `_noop_logger` at line 65 — do NOT redefine it):

```python
class TestObsSnapshotCache:
    """_latest_obs_for_snapshot and _latest_legal_mask_for_snapshot lifecycle."""

    def test_initial_values_are_none(self):
        """Both snapshot caches start as None."""
        sm = _make_step_manager()
        assert sm._latest_obs_for_snapshot is None
        assert sm._latest_legal_mask_for_snapshot is None

    def test_stashed_after_successful_step(self):
        """After a successful execute_step, both caches are populated."""
        sm = _make_step_manager()
        sm.game.get_legal_moves.return_value = [(0, 0, 2, 2, False)]
        sm.policy_mapper.get_legal_mask.return_value = torch.zeros(13527)
        sm.agent.select_action.return_value = ((0, 0, 2, 2, False), 42, -0.5, 0.3)
        sm.game.make_move.return_value = (_make_obs(), 0.0, False, {})
        sm.game.current_player = MagicMock()
        sm.game.current_player.name = "BLACK"
        sm.game.current_player.value = 0

        episode = _make_episode_state()
        result = sm.execute_step(episode, global_timestep=1, logger_func=_noop_logger)

        assert result.success
        # Obs cache is a numpy array (stashed from episode_state.current_obs)
        assert sm._latest_obs_for_snapshot is not None
        assert isinstance(sm._latest_obs_for_snapshot, np.ndarray)
        # Legal mask cache is a tensor
        assert sm._latest_legal_mask_for_snapshot is not None
        assert isinstance(sm._latest_legal_mask_for_snapshot, torch.Tensor)

    def test_obs_is_current_observation(self):
        """The stashed obs matches the episode's current_obs (numpy array)."""
        sm = _make_step_manager()
        sm.game.get_legal_moves.return_value = [(0, 0, 2, 2, False)]
        sm.policy_mapper.get_legal_mask.return_value = torch.zeros(13527)
        sm.agent.select_action.return_value = ((0, 0, 2, 2, False), 42, -0.5, 0.3)
        sm.game.make_move.return_value = (_make_obs(), 0.0, False, {})
        sm.game.current_player = MagicMock()
        sm.game.current_player.name = "BLACK"
        sm.game.current_player.value = 0

        episode = _make_episode_state()
        sm.execute_step(episode, global_timestep=1, logger_func=_noop_logger)

        np.testing.assert_array_equal(
            sm._latest_obs_for_snapshot, episode.current_obs
        )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/unit/test_step_manager.py::TestObsSnapshotCache -v`
Expected: All 3 PASS (implementation already exists)

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_step_manager.py
git commit -m "test(training): add lifecycle tests for observation/legal_mask snapshot cache"
```

---

## Task 2: State Contract Extension — SquareAction

**Files:**
- Modify: `keisei/webui/view_contracts.py:183-205`
- Test: `tests/unit/test_square_actions.py` (new)

- [ ] **Step 1: Write failing test for SquareAction type**

Create `tests/unit/test_square_actions.py`:

```python
"""Unit tests for per-square action data in the state contract."""

import pytest

pytestmark = pytest.mark.unit


class TestSquareActionContract:
    """SquareAction and square_actions field in PolicyInsight."""

    def test_square_action_typeddict_exists(self):
        """SquareAction TypedDict is importable."""
        from keisei.webui.view_contracts import SquareAction

        # TypedDict instances are just dicts
        sa: SquareAction = {"action": "7g7f", "prob": 0.23}
        assert sa["action"] == "7g7f"
        assert sa["prob"] == 0.23

    def test_policy_insight_allows_square_actions(self):
        """PolicyInsight accepts square_actions field."""
        from keisei.webui.view_contracts import PolicyInsight

        pi: PolicyInsight = {
            "action_heatmap": [[0.0] * 9 for _ in range(9)],
            "top_actions": [],
            "value_estimate": 0.0,
            "action_entropy": 0.0,
            "square_actions": {
                "5,2": [{"action": "7g7f", "prob": 0.23}],
            },
        }
        assert "5,2" in pi["square_actions"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_square_actions.py -v`
Expected: FAIL — `SquareAction` not found in view_contracts

- [ ] **Step 3: Add SquareAction to view_contracts.py**

In `keisei/webui/view_contracts.py`, after the `TopAction` TypedDict (line ~188), add:

```python
class SquareAction(TypedDict):
    """A single action targeting a specific board square."""

    action: str  # USI notation, e.g. "7g7f" or "P*5e"
    prob: float  # probability in [0, 1]
```

In the `PolicyInsight` TypedDict, add after `action_entropy`:

```python
    square_actions: Dict[str, List["SquareAction"]]  # "r,c" -> top-3 actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_square_actions.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/webui/view_contracts.py tests/unit/test_square_actions.py
git commit -m "feat(webui): add SquareAction type and square_actions to PolicyInsight"
```

---

## Task 3: Per-Square Action Extraction Logic

**Files:**
- Modify: `keisei/webui/state_snapshot.py:130-224` (extract_policy_insight)
- Test: `tests/unit/test_square_actions.py`

- [ ] **Step 1: Write failing tests for square_actions extraction**

Add to `tests/unit/test_square_actions.py`:

```python
from unittest.mock import MagicMock
import numpy as np
import torch


class TestSquareActionExtraction:
    """extract_policy_insight produces correct square_actions."""

    def _make_mock_agent_and_mapper(self):
        """Create mocks that produce a known probability distribution."""
        agent = MagicMock()
        agent.device = torch.device("cpu")
        agent.scaler = None

        # Model that returns known logits
        model = MagicMock()
        model.training = False
        # 13527 logits, all zero (uniform) except a few
        logits = torch.zeros(1, 13527)
        logits[0, 0] = 5.0   # High prob action targeting (2, 2)
        logits[0, 1] = 3.0   # Medium prob action targeting (2, 2)
        logits[0, 2] = 2.0   # Lower prob action targeting (4, 4)
        model.return_value = (logits, torch.tensor([[0.5]]))
        model.eval = MagicMock()
        model.train = MagicMock()
        agent.model = model

        # Policy mapper with known move destinations
        mapper = MagicMock()
        idx_to_move = [(0, 0, 2, 2, False)] * 13527  # All point to (2,2) by default
        idx_to_move[0] = (0, 0, 2, 2, False)  # Action 0 -> dest (2,2)
        idx_to_move[1] = (1, 1, 2, 2, False)  # Action 1 -> dest (2,2)
        idx_to_move[2] = (3, 3, 4, 4, False)  # Action 2 -> dest (4,4)
        mapper.idx_to_move = idx_to_move
        mapper.action_idx_to_usi_move = lambda idx: f"act{idx}"

        return agent, mapper

    def test_square_actions_present_in_result(self):
        """extract_policy_insight returns square_actions key."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = self._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        # legal_mask: all True (all actions legal) — matches the current signature
        legal_mask = torch.ones(13527, dtype=torch.bool)

        result = extract_policy_insight(agent, obs, mapper, top_k=5, legal_mask=legal_mask)
        assert result is not None
        assert "square_actions" in result

    def test_square_actions_top3_per_square(self):
        """Each square gets at most 3 actions."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = self._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        legal_mask = torch.ones(13527, dtype=torch.bool)

        result = extract_policy_insight(agent, obs, mapper, top_k=5, legal_mask=legal_mask)
        sa = result["square_actions"]
        for key, actions in sa.items():
            assert len(actions) <= 3

    def test_square_actions_key_format(self):
        """Keys use 'r,c' format with 0-indexed coordinates."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = self._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        legal_mask = torch.ones(13527, dtype=torch.bool)

        result = extract_policy_insight(agent, obs, mapper, top_k=5, legal_mask=legal_mask)
        sa = result["square_actions"]
        for key in sa:
            parts = key.split(",")
            assert len(parts) == 2
            r, c = int(parts[0]), int(parts[1])
            assert 0 <= r < 9
            assert 0 <= c < 9

    def test_square_actions_sorted_by_prob(self):
        """Actions within each square are sorted descending by probability."""
        from keisei.webui.state_snapshot import extract_policy_insight

        agent, mapper = self._make_mock_agent_and_mapper()
        obs = np.zeros((46, 9, 9), dtype=np.float32)
        legal_mask = torch.ones(13527, dtype=torch.bool)

        result = extract_policy_insight(agent, obs, mapper, top_k=5, legal_mask=legal_mask)
        sa = result["square_actions"]
        for key, actions in sa.items():
            probs = [a["prob"] for a in actions]
            assert probs == sorted(probs, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_square_actions.py::TestSquareActionExtraction -v`
Expected: FAIL — `square_actions` not in result

- [ ] **Step 3: Add square_actions computation to extract_policy_insight**

In `keisei/webui/state_snapshot.py`, inside `extract_policy_insight()`, after the heatmap loop (after line ~200 `heatmap[to_r][to_c] += p`) and before the top-K section, add:

```python
        # Per-square action breakdown: top-3 actions per destination square
        from collections import defaultdict
        square_action_candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)
```

Then, inside the existing heatmap loop, extend it to also collect per-square candidates. Modify the loop body (lines ~187-200) to:

```python
        square_action_candidates: dict = {}
        for idx in range(len(idx_to_move)):
            p = float(probs_np[idx])
            if p < 1e-8:
                continue
            move = idx_to_move[idx]
            to_r, to_c = move[2], move[3]
            if (
                isinstance(to_r, int)
                and isinstance(to_c, int)
                and 0 <= to_r < 9
                and 0 <= to_c < 9
            ):
                heatmap[to_r][to_c] += p
                # Collect for per-square breakdown (threshold: prob > 0.001)
                if p > 0.001:
                    key = f"{to_r},{to_c}"
                    if key not in square_action_candidates:
                        square_action_candidates[key] = []
                    square_action_candidates[key].append((p, idx))
```

After the loop, build the square_actions dict:

```python
        # Top-3 per square, sorted by probability descending
        square_actions: dict[str, list[dict]] = {}
        for key, candidates in square_action_candidates.items():
            candidates.sort(reverse=True)  # Sort by probability (first element)
            top3 = candidates[:3]
            actions = []
            for prob, idx in top3:
                try:
                    usi = policy_mapper.action_idx_to_usi_move(int(idx))
                except (IndexError, ValueError):
                    usi = f"idx:{idx}"
                actions.append({"action": usi, "prob": prob})
            square_actions[key] = actions
```

In the return dict, add `square_actions`:

```python
        return {
            "action_heatmap": heatmap,
            "top_actions": top_actions,
            "value_estimate": value_estimate,
            "action_entropy": action_entropy,
            "square_actions": square_actions,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_square_actions.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing policy insight tests to check for regressions**

Run: `pytest tests/integration/test_webui_state_snapshot.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/webui/state_snapshot.py tests/unit/test_square_actions.py
git commit -m "feat(webui): add per-square action extraction to policy insight"
```

---

## Task 4: Update Sample State JSON

**Files:**
- Modify: `keisei/webui/sample_state.json`

- [ ] **Step 1: Add square_actions to sample policy_insight data**

In `keisei/webui/sample_state.json`, find the `"policy_insight"` object. Add a `"square_actions"` field with realistic sample data. The existing `policy_insight` has an `action_heatmap` and `top_actions`. Add after `action_entropy`:

```json
"square_actions": {
  "5,2": [
    {"action": "7g7f", "prob": 0.231},
    {"action": "6h7g", "prob": 0.042}
  ],
  "2,6": [
    {"action": "3c3d", "prob": 0.084},
    {"action": "2b3c", "prob": 0.015}
  ],
  "6,2": [
    {"action": "7g7f", "prob": 0.231}
  ],
  "3,5": [
    {"action": "4e3d", "prob": 0.058},
    {"action": "5d4e", "prob": 0.041},
    {"action": "3c3d", "prob": 0.012}
  ]
}
```

- [ ] **Step 2: Validate JSON is still parseable**

Run: `python -c "import json; json.load(open('keisei/webui/sample_state.json'))"`
Expected: No error

- [ ] **Step 3: Commit**

```bash
git add keisei/webui/sample_state.json
git commit -m "feat(webui): add square_actions sample data for demo mode"
```

---

## Task 5: Board Component Python Wrapper

**Files:**
- Create: `keisei/webui/board_component/__init__.py`
- Create: `keisei/webui/board_component/frontend/index.html` (placeholder)
- Test: `tests/unit/test_board_component.py` (new)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p keisei/webui/board_component/frontend
```

- [ ] **Step 2: Write failing test for the Python wrapper**

Create `tests/unit/test_board_component.py`:

```python
"""Unit tests for the board component Python wrapper."""

import pytest

pytestmark = pytest.mark.unit


class TestBoardComponentImport:
    """Board component wrapper is importable and has expected API."""

    def test_shogi_board_function_exists(self):
        """shogi_board function is importable."""
        from keisei.webui.board_component import shogi_board
        assert callable(shogi_board)

    def test_shogi_board_accepts_expected_args(self):
        """shogi_board accepts board_state, heatmap, square_actions, piece_images, selected_square, key."""
        import inspect
        from keisei.webui.board_component import shogi_board

        sig = inspect.signature(shogi_board)
        param_names = list(sig.parameters.keys())
        assert "board_state" in param_names
        assert "heatmap" in param_names
        assert "square_actions" in param_names
        assert "piece_images" in param_names
        assert "selected_square" in param_names
        assert "key" in param_names
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/unit/test_board_component.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Create the Python wrapper**

Create `keisei/webui/board_component/__init__.py`:

```python
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
```

Create a minimal placeholder `keisei/webui/board_component/frontend/index.html`:

```html
<!DOCTYPE html>
<html>
<head><title>Shogi Board Component</title></head>
<body>
<div id="root">Loading board component...</div>
<script>
// Load streamlit-component-lib from Streamlit's own static serving (no CDN dependency).
// Streamlit serves this at a relative path when using declare_component with a local path.
// The actual script tag is: <script src="streamlit-component-lib.js"> which Streamlit
// rewrites to the correct URL at serve time.
// Placeholder — full implementation in Task 6
</script>
<script src="streamlit-component-lib.js"></script>
</body>
</html>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_board_component.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/webui/board_component/ tests/unit/test_board_component.py
git commit -m "feat(webui): add board component Python wrapper with declare_component"
```

---

## Task 6: Board Component JS Frontend

**Files:**
- Modify: `keisei/webui/board_component/frontend/index.html`

This is the largest task. The JS frontend renders the board and handles all interaction. No automated test for this task — it's pure frontend JS tested via manual demo mode.

- [ ] **Step 1: Write the full index.html**

Replace the placeholder `keisei/webui/board_component/frontend/index.html` with the full implementation. The file should contain:

**HTML structure:**
- `<div id="root">` container
- Board renders as `<table role="grid" aria-label="...">` with `<th>` headers
- Each cell: `<td role="gridcell" tabindex="-1" data-row="N" data-col="N" aria-label="...">`

**CSS (in `<style>` tag):**
- Cell base styles: 48px x 48px, border, background colors (`#f5deb3` / `#deb887` alternating)
- Promotion zone tints: rows 0-2 blue tint, rows 6-8 red tint
- Rank 3/6 thicker borders (2.5px)
- Focus indicator: `td[role="gridcell"]:focus { outline: 2px dashed #888; outline-offset: -2px; }`
- Selected indicator: `.selected { outline: 3px solid #2a52b0; outline-offset: -3px; background-color: rgba(42,82,176,0.15) !important; }`
- Heatmap overlay: `box-shadow: inset 0 0 0 100px rgba(255,140,0,ALPHA)`
- Dark theme: `@media (prefers-color-scheme: dark) { table[role="grid"] th { color: #e0e0e0; } }` — note: must use `role="grid"` selector, NOT `role="table"` (the old `render_board()` used `role="table"`)

**JS (in `<script>` tag):**

1. **Streamlit component lifecycle:**
   ```javascript
   Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
   Streamlit.setComponentReady();
   ```

2. **`onRender(event)` function:**
   - Read `event.detail.args` for `board_state`, `heatmap`, `square_actions`, `piece_images`
   - Build the HTML table with all visual features
   - Apply heatmap overlay using log-scale normalization (port `_compute_heatmap_overlay` logic)
   - Apply selection highlight if a cell is selected
   - Set up event listeners
   - Update grid `aria-label` on each render: `"Shogi board position, move {N}, {player} to play. Black (Sente) plays from bottom, White (Gote) from top."` — preserves orientation guidance from the original `render_board()` caption
   - Call `Streamlit.setFrameHeight(FIXED_HEIGHT)` with fixed height (48 * 10 + 40 = 520)

3. **Roving tabindex:**
   - Track `currentFocusRow` and `currentFocusCol`
   - On mount: if `args.selected_square` exists (passed from Python session state), initialize focus at that cell; otherwise default to (0,0)
   - Only that cell gets `tabindex="0"`; all others get `tabindex="-1"`
   - On arrow key navigation, shift `tabindex` values

4. **Click handler:**
   ```javascript
   function onCellClick(row, col) {
     if (selectedRow === row && selectedCol === col) {
       selectedRow = null; selectedCol = null;
       Streamlit.setComponentValue({type: "deselect"});
     } else {
       selectedRow = row; selectedCol = col;
       Streamlit.setComponentValue({row: row, col: col, type: "select"});
     }
     renderBoard();  // Re-render to update selection highlight
   }
   ```

5. **Keyboard handler:**
   ```javascript
   // Arrow keys: navigate, update roving tabindex
   // Enter/Space: toggle select (same as click)
   // Escape: deselect
   // Tab: do nothing (let browser exit the grid)
   ```

6. **Focus tracking:**
   ```javascript
   let boardFocused = false;
   let blurTimeout = null;

   table.addEventListener('focusin', function() {
     if (!boardFocused) {
       boardFocused = true;
       Streamlit.setComponentValue({type: "focus"});
     }
     clearTimeout(blurTimeout);
   });

   table.addEventListener('focusout', function() {
     clearTimeout(blurTimeout);
     blurTimeout = setTimeout(function() {
       boardFocused = false;
       Streamlit.setComponentValue({type: "blur"});
     }, 100);
   });
   ```

7. **Board rendering function:**
   - Port the Python `render_board()` HTML generation to JS
   - Use `piece_images[key]` for SVG data URIs (key format: `"{type}_{color}"`, e.g., `"pawn_black"`)
   - Build aria-labels: `"{file}-{rank}: {piece description}"` where file = 9 - col, rank = row + 1. **Note:** Cell aria-labels use "7-1" geographic format; announcement text (Task 7) uses compact "76" — this is intentional per spec Decision #9. Do not "fix" the mismatch.

- [ ] **Step 2: Test manually in demo mode**

Run: `streamlit run keisei/webui/streamlit_app.py`

Verify in browser:
- Board renders with pieces
- Arrow keys navigate between cells
- Click selects/deselects a cell (blue highlight + fill)
- Focus indicator is dashed and distinct from selection
- Tab exits the board grid
- Heatmap overlay visible when toggled

Note: The app won't use the component yet (render_board still calls html()). To test the component standalone, temporarily add a test script or modify the Game tab to call `shogi_board()`. This integration happens in Task 8.

- [ ] **Step 3: Write automated smoke test for board HTML structure**

Add to `tests/unit/test_board_component.py`:

```python
class TestBoardComponentHTML:
    """Verify the JS frontend HTML file has the expected structure."""

    def test_frontend_index_exists(self):
        """The frontend index.html file exists."""
        from pathlib import Path
        index = Path(__file__).parent.parent.parent / "keisei" / "webui" / "board_component" / "frontend" / "index.html"
        assert index.exists()

    def test_frontend_uses_grid_role(self):
        """The frontend HTML uses role='grid', not role='table'."""
        from pathlib import Path
        index = Path(__file__).parent.parent.parent / "keisei" / "webui" / "board_component" / "frontend" / "index.html"
        content = index.read_text()
        assert 'role="grid"' in content or "role='grid'" in content
        assert 'role="table"' not in content

    def test_frontend_has_no_cdn_dependency(self):
        """The frontend does not load JS from external CDNs."""
        from pathlib import Path
        index = Path(__file__).parent.parent.parent / "keisei" / "webui" / "board_component" / "frontend" / "index.html"
        content = index.read_text()
        assert "cdn.jsdelivr.net" not in content
        assert "unpkg.com" not in content

    def test_frontend_has_setComponentValue(self):
        """The frontend calls Streamlit.setComponentValue for interaction events."""
        from pathlib import Path
        index = Path(__file__).parent.parent.parent / "keisei" / "webui" / "board_component" / "frontend" / "index.html"
        content = index.read_text()
        assert "setComponentValue" in content

    def test_frontend_has_roving_tabindex(self):
        """The frontend implements roving tabindex (tabindex=-1 pattern)."""
        from pathlib import Path
        index = Path(__file__).parent.parent.parent / "keisei" / "webui" / "board_component" / "frontend" / "index.html"
        content = index.read_text()
        assert 'tabindex="-1"' in content or "tabindex=\"-1\"" in content
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_board_component.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/webui/board_component/frontend/index.html tests/unit/test_board_component.py
git commit -m "feat(webui): implement board component JS frontend with interaction"
```

---

## Task 7: Selected Square Detail Panel

**Files:**
- Modify: `keisei/webui/streamlit_app.py`

- [ ] **Step 1: Write `render_selected_square_panel()` function**

Add to `keisei/webui/streamlit_app.py`, after `render_policy_insight_panel()` (after line ~546):

```python
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

    See: spec Section 5.
    """
    row = selected["row"]
    col = selected["col"]
    file_num = 9 - col
    rank = row + 1
    square_label = f"{file_num}{rank}"

    st.caption(f"Square {square_label} actions")

    if not insight_available:
        st.text("Policy insight not available. Enable in config to see action breakdown.")
        return

    # Piece context from board_state
    board = board_state.get("board", [])
    if 0 <= row < len(board) and 0 <= col < len(board[row]):
        piece = board[row][col]
        if piece:
            st.text(f"Contains: {piece['color'].capitalize()} {piece['type'].replace('_', ' ').title()}")

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
                f'height:12px;background:#ff8c00;border-radius:2px;'
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
            announce = f"Selected {square_label}. Top action: {actions[0]['action']} {actions[0]['prob']*100:.1f}%."
        else:
            announce = f"Selected {square_label}. No actions target this square."

        st.markdown(
            f'<div role="status" aria-live="polite" '
            f'style="position:absolute;width:1px;height:1px;'
            f'overflow:hidden;clip:rect(0,0,0,0);">'
            f"{announce}</div>",
            unsafe_allow_html=True,
        )
```

- [ ] **Step 2: Commit**

```bash
git add keisei/webui/streamlit_app.py
git commit -m "feat(webui): add selected square detail panel renderer"
```

---

## Task 8: Integration — Wire Component into Game Tab

**Files:**
- Modify: `keisei/webui/streamlit_app.py` (render_game_tab, fragment lifecycle)

This task replaces `render_board()` with `shogi_board()` and adds the invalidation, focus-pause, and selected square panel wiring.

- [ ] **Step 1: Add board_component import**

At the top of `streamlit_app.py`, add:

```python
from keisei.webui.board_component import shogi_board
```

- [ ] **Step 2: Update render_game_tab() to use the component**

Replace the current `render_game_tab()` function. Key changes:

1. Call `shogi_board()` instead of `render_board()`
2. Process the component's return value for selection and focus events
3. Insert `render_selected_square_panel()` when a square is selected
4. Pass `_PIECE_SVG_CACHE` as `piece_images`

```python
def render_game_tab(env: EnvelopeParser) -> None:
    """Render the Game tab: board, policy insight, game status, moves."""
    board_state = env.board_state
    step_info = env.step_info
    metrics = env.metrics
    insight = env.policy_insight

    if not board_state:
        st.info("Waiting for first episode...")
        return

    # Selected square invalidation (spec Section 6)
    prev_move = st.session_state.get("last_move_count")
    curr_move = board_state.get("move_count", 0)
    if prev_move is not None and prev_move != curr_move:
        st.session_state.selected_square = None
    st.session_state.last_move_count = curr_move

    # Skip-to-board link
    st.markdown(
        '<a href="#shogi-board" style="font-size:0;position:absolute;">'
        "Skip to board</a>",
        unsafe_allow_html=True,
    )

    # Heatmap overlay
    heatmap = None
    if st.session_state.get("show_heatmap", False) and insight:
        heatmap = insight.get("action_heatmap")

    # Square actions for the component and panel
    square_actions = insight.get("square_actions", {}) if insight else {}

    board_col, insight_col = st.columns([2, 3])
    with board_col:
        # Custom bidirectional board component
        board_result = shogi_board(
            board_state=board_state,
            heatmap=heatmap,
            square_actions=square_actions,
            piece_images=_PIECE_SVG_CACHE,
            selected_square=st.session_state.get("selected_square"),
            key="main_board",
        )

        # Process component interaction events
        if board_result:
            event_type = board_result.get("type")
            if event_type == "select":
                st.session_state.selected_square = {
                    "row": board_result["row"],
                    "col": board_result["col"],
                }
            elif event_type == "deselect":
                st.session_state.selected_square = None
                st.session_state.last_announced_square = None
            elif event_type == "focus":
                st.session_state.board_focused = True
            elif event_type == "blur":
                st.session_state.board_focused = False

        render_hands(board_state)

    with insight_col:
        render_policy_insight_panel(insight)

        # Selected square detail panel (spec Section 5)
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
                st.caption(
                    "Hot Squares: "
                    + ", ".join(str(s) for s in hot_squares)
                )
```

- [ ] **Step 3: Update the fragment for focus-pause and board_state None invalidation**

In `_live_data_section()`, update the pause check to include board focus:

```python
@st.fragment(run_every=timedelta(seconds=2))
def _live_data_section() -> None:
    paused_by_toggle = not st.session_state.get("auto_refresh", True)
    paused_by_focus = st.session_state.get("board_focused", False)

    if paused_by_toggle or paused_by_focus:
        cached = st.session_state.get("last_state")
        if cached is None:
            st.info("Paused. No cached state available.")
            return
        env = EnvelopeParser(cached)
        # Show pause reason in header
        if paused_by_focus:
            st.info("Paused — inspecting board")
        render_stale_warning(env)
        _render_dashboard_content(env)
        return

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
```

- [ ] **Step 4: Clear board_focused on non-Game-tab render paths**

In `render_metrics_tab()` and `render_lineage_tab()`, add at the top:

```python
def render_metrics_tab(env: EnvelopeParser) -> None:
    """Render the Metrics tab: training curves, win rates, buffer."""
    # Clear board focus when not on Game tab (prevents stuck focus-pause)
    st.session_state.board_focused = False
    # ... existing code ...

def render_lineage_tab(env: EnvelopeParser) -> None:
    """Render the Lineage tab: generation tree, Elo, events."""
    st.session_state.board_focused = False
    render_lineage_panel(env)
```

Additionally, add a **timeout-based fallback** in the fragment's pause check to handle edge cases where the blur event is missed (e.g., component unmount, browser iframe issues, user clicks outside the board without switching tabs):

```python
import time

# Timeout fallback: if board_focused has been True for > 30s without
# a fresh focus event, auto-clear it. This catches cases where blur
# is never received (component unmount, iframe issues).
if st.session_state.get("board_focused", False):
    last_focus_time = st.session_state.get("board_focus_timestamp", 0)
    if time.time() - last_focus_time > 30:
        st.session_state.board_focused = False
```

Set `board_focus_timestamp` whenever a focus event is received:

```python
if board_result and board_result.get("type") == "focus":
    st.session_state.board_focused = True
    st.session_state.board_focus_timestamp = time.time()
```

- [ ] **Step 5: Remove old render_board() function**

**Do NOT delete `render_board()`.** Keep it as a fallback renderer in case the custom component fails to load (JS error, iframe blocked, etc.). Wrap the `shogi_board()` call in the Game tab with a try/except:

```python
try:
    board_result = shogi_board(...)
except Exception:
    # Fallback to non-interactive board rendering
    render_board(board_state, heatmap=heatmap)
    board_result = None
```

This ensures the Game tab is never blank. The fallback loses interactivity but preserves the board display. Keep `_compute_heatmap_overlay()` and `_piece_aria_label()` for the fallback path.

Keep `_load_piece_svgs()`, `_PIECE_SVG_CACHE`, and `_piece_image_key()` — these produce the data URIs passed to the component AND used by the fallback.

- [ ] **Step 6: Test manually in demo mode**

Run: `streamlit run keisei/webui/streamlit_app.py`

Verify:
- Board renders correctly with pieces and heatmap
- Click a square → blue selection highlight appears, detail panel shows in right column
- Click same square again → deselection, panel disappears
- Arrow keys navigate, Enter selects
- Tab exits the board
- Focus on board pauses auto-refresh, "Paused — inspecting board" appears
- Switch to Metrics tab → auto-refresh resumes
- Move counter change clears selection

- [ ] **Step 7: Commit**

```bash
git add keisei/webui/streamlit_app.py
git commit -m "feat(webui): integrate board component with focus-pause and selection"
```

---

## Task 9: Integration Test — End-to-End Policy Insight with square_actions

**Files:**
- Modify: `tests/integration/test_webui_state_snapshot.py`

- [ ] **Step 1: Write integration test**

Add to `tests/integration/test_webui_state_snapshot.py`:

```python
class TestPolicyInsightSquareActions:
    """End-to-end: StepManager obs → build_snapshot → square_actions in envelope."""

    def test_square_actions_in_snapshot(
        self, shogi_game, ppo_agent, session_policy_mapper
    ):
        """When policy insight is enabled and obs is available, square_actions is populated.

        Uses existing fixtures from conftest.py:
        - shogi_game: fresh ShogiGame
        - ppo_agent: PPOAgent(model=ActorCritic(46, 13527), config=integration_config, device=cpu)
        - session_policy_mapper: PolicyOutputMapper (cached per session)
        """
        from keisei.webui.state_snapshot import extract_policy_insight

        # Get observation from game
        obs = shogi_game.reset()

        # Get a real legal mask from the current position
        legal_moves = shogi_game.get_legal_moves()
        legal_mask = session_policy_mapper.get_legal_mask(
            legal_moves, device=torch.device("cpu")
        )

        result = extract_policy_insight(
            ppo_agent, obs, session_policy_mapper, top_k=5, legal_mask=legal_mask
        )
        assert result is not None
        assert "square_actions" in result
        assert isinstance(result["square_actions"], dict)

        # At least some squares should have actions
        assert len(result["square_actions"]) > 0

        # Each entry should have correct format
        for key, actions in result["square_actions"].items():
            assert "," in key
            for act in actions:
                assert "action" in act
                assert "prob" in act
                assert act["prob"] > 0
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/integration/test_webui_state_snapshot.py::TestPolicyInsightSquareActions -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_webui_state_snapshot.py
git commit -m "test(webui): add integration test for square_actions in policy insight"
```

---

## Task 10: Final Cleanup and Validation

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/unit/ tests/integration/ -v --tb=short`
Expected: All PASS, no regressions

- [ ] **Step 2: Run linting**

Run: `black keisei/webui/ keisei/training/step_manager.py tests/unit/test_square_actions.py tests/unit/test_board_component.py`
Run: `flake8 keisei/webui/ keisei/training/step_manager.py`

- [ ] **Step 3: Run type checking**

Run: `mypy keisei/webui/view_contracts.py keisei/webui/state_snapshot.py keisei/webui/board_component/__init__.py`

- [ ] **Step 4: Manual smoke test in demo mode**

Run: `streamlit run keisei/webui/streamlit_app.py`

Full checklist:
- [ ] Board renders with correct pieces, promotion zones, coordinates
- [ ] Heatmap toggle works
- [ ] Click selects square, blue highlight (solid `#2a52b0` + fill) appears
- [ ] Detail panel shows below Action Distribution with top-3 actions
- [ ] Click same square deselects, panel disappears
- [ ] Arrow keys navigate with dashed focus indicator (`2px dashed #888`)
- [ ] **Focus/selection distinction:** Select a cell (Enter), then arrow away. Selected cell keeps solid blue outline + fill; navigated cell shows dashed focus ring only.
- [ ] Enter/Space toggles selection (select → deselect on same cell)
- [ ] Escape clears selection
- [ ] Tab exits the board immediately (does NOT cycle through 81 cells)
- [ ] Shift-Tab also exits the board
- [ ] "Paused — inspecting board" appears when board is focused
- [ ] Switching to Metrics tab resumes auto-refresh (focus-pause clears)
- [ ] Move counter change clears selection
- [ ] Episode boundary (board disappears between episodes) clears selection
- [ ] Detail panel shows "Policy insight not available..." when insight is None
- [ ] **DOM verification:** Open DevTools → inspect board table → verify `role="grid"` on table, `role="gridcell"` on cells (NOT `role="table"`)
- [ ] **Dark theme:** Toggle prefers-color-scheme → verify column headers are readable (`#e0e0e0`)
- [ ] **Screen reader:** Open browser Accessibility panel. Select a square → verify announcement fires exactly once ("Selected 76. Top action: ..."). Wait 4+ seconds → verify it does NOT repeat on re-render. Select a different square → verify new announcement fires.
- [ ] Blur debounce: rapidly arrow-navigate between cells → verify no pause/unpause flicker

- [ ] **Step 5: Final commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore(webui): final cleanup for v2.1 board interactivity"
```
