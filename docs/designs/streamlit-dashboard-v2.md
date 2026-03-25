# Keisei Streamlit Dashboard v2 — Detailed Design

**Status**: Reviewed (incorporating UX specialist + dev feedback)
**Date**: 2026-03-26
**Reviewers**: UX Critic (lyra), Primary Dev

---

## 1. Design Goals

### Primary
- Surface **domain-specific ML insight** (policy decisions, value estimates) that generic dashboards like W&B cannot provide
- Make the dashboard **inspectable** with pause/freeze controls for examining specific states
- Fix **critical UX blockers** (page-level rerun, piece distinguishability)

### Secondary
- Scale the layout to accommodate upcoming views (league, skill_differential, model_profile)
- Meet baseline WCAG AA accessibility
- Support both live training monitoring and post-hoc state inspection

### Non-Goals
- Mobile-first design (this is a desktop monitoring tool)
- Replacing W&B for experiment tracking (complementary, not competitive)
- Interactive game play (this is observation/analysis only)
- Board click-to-inspect in v2 initial release (deferred to v2.1 — see Section 6.3)

---

## 2. Architecture Changes

### 2.1 Fragment-Based Refresh (Critical)

**Current**: `time.sleep(2) + st.rerun()` at page bottom — reruns entire page every 2s, destroying all user interaction.

**Proposed**: Use Streamlit's `@st.fragment(run_every=timedelta(seconds=2))` decorator on data-dependent sections only.

**Streamlit version**: Requires `>=1.33` for fragment support. The project floor is already `>=1.55.0` in `pyproject.toml`, so this is satisfied.

#### Fragment/Tab Boundary Architecture

The `st.tabs()` widget **must live inside the fragment**, because only code inside the fragment re-executes on each refresh cycle. Tab content rendered outside the fragment would go stale.

The active tab is tracked via `st.session_state.active_tab_index` (an integer). On each fragment rerun, `st.tabs()` re-renders but Streamlit preserves the selected tab via its internal widget key. Only the active tab's content is evaluated, so performance cost is low.

```
Page Structure:
  ├── st.set_page_config()                    (runs once)
  ├── Sidebar controls                        (runs once)
  │     ├── st.toggle("Auto-refresh")
  │     ├── st.toggle("Show heatmap")
  │     └── st.button("Export state")
  └── @st.fragment(run_every=...):            (reruns on schedule)
        ├── load_state()
        ├── Header metrics row
        ├── st.tabs(["Metrics", "Game", "Lineage"])
        │     ├── Metrics tab content
        │     ├── Game tab content
        │     └── Lineage tab content
        └── Status indicators
```

**Pause/Resume**: A `st.toggle("Auto-refresh", value=True, key="auto_refresh")` in the sidebar controls whether the fragment auto-refreshes.

**Important**: The `@st.fragment(run_every=...)` decorator argument is evaluated once at function definition time, not on each Streamlit rerun. Toggling `auto_refresh` in the sidebar won't change the fragment's `run_every` until the page is fully re-executed. Two viable approaches:

**Approach A (Recommended)**: The sidebar toggle calls `st.rerun()` on change. This is acceptable because it's user-initiated (not the automatic 2s rerun we're eliminating). The fragment is always defined with `run_every=timedelta(seconds=2)`, but internally checks `auto_refresh` and short-circuits when paused:

```python
@st.fragment(run_every=timedelta(seconds=2))
def live_data_section():
    if not st.session_state.get("auto_refresh", True):
        # Paused — render from cached state, don't reload
        state = st.session_state.get("last_state")
        if state is None:
            st.info("Paused. No cached state available.")
            return
        env = EnvelopeParser(state)
    else:
        state = load_state(args.state_file)
        if state is None:
            st.error("Waiting for training data...")
            return
        env = EnvelopeParser(state)
        st.session_state.last_state = state

    # ... render from env ...
```

**Approach B**: Define the fragment with a conditional `run_every`, and trigger `st.rerun()` from the toggle callback to force re-evaluation of the decorator. Less clean but simpler code.

**Verified**: `st.fragment` accepts `run_every=None` (its default) to disable auto-refresh. Confirmed via inspection of the Streamlit >= 1.55 API signature.

### 2.2 State Contract Extensions

New fields added to `TrainingViewState` (backward-compatible, all optional):

```python
# In view_contracts.py — new TypedDict
class PolicyInsight(TypedDict, total=False):
    """Per-square action probability summary for the current board state."""
    # 9x9 grid, each cell = sum of action probabilities targeting that square
    # Values are log-scaled for display (see Section 3.3)
    action_heatmap: list[list[float]]  # [row][col], raw probability sums
    # Top-K actions with human-readable labels
    top_actions: list[dict]  # [{action: "P76", prob: 0.23, type: "move"}, ...]
    # Critic's value estimate for current position
    value_estimate: float  # V(s), typically in [-1, 1]
    # Confidence: entropy of the action distribution (lower = more confident)
    action_entropy: float
```

Added as an optional field in `TrainingViewState`:
```python
policy_insight: PolicyInsight | None  # None when unavailable
```

#### Producer Data Flow (Implementation Note)

**Problem**: Currently `state_snapshot.py` only receives serialized manager objects — it does not have access to the live `PPOAgent` instance or the current observation tensor. The `build_snapshot()` function signature is:

```python
def build_snapshot(trainer, speed, pending_updates) -> dict
```

The `trainer` object provides access to managers, but the PPO agent and current observation need to be threaded through.

**Proposed interface**:

```python
def extract_policy_insight(
    ppo_agent: PPOAgent,
    observation: torch.Tensor,
    policy_mapper: PolicyOutputMapper,
    top_k: int = 10,
) -> PolicyInsight | None:
    """Extract policy insight from the agent's last decision.

    Called from build_snapshot() when webui.policy_insight is enabled.
    Uses torch.no_grad() — no gradient computation.

    Returns None if:
    - observation is None (between episodes)
    - PPO update is in progress (weights being modified)
    - Any extraction error occurs (non-fatal)
    """
```

**Data flow — push model, not pull**: Rather than having `state_snapshot.py` reach back into `StepManager` via a new accessor (which breaks manager encapsulation), `StepManager` pushes the current observation into the snapshot data during its normal `collect_experience()` call. Concretely:

1. `StepManager.collect_experience()` already produces observation tensors. After each step, it stashes a **detached clone** of the current observation: `self._latest_obs_for_snapshot = obs.detach().clone()`. The clone is essential — a bare reference would alias a tensor that may be modified in-place by environment reset or the next step, causing correctness bugs in the snapshot.
2. `build_snapshot()` receives this via an explicit parameter, not by reaching into the manager:

```python
def build_snapshot(
    trainer,
    speed: float,
    pending_updates: dict,
    latest_observation: torch.Tensor | None = None,  # NEW
) -> dict:
```

3. `StreamlitManager._write_if_due()` passes `trainer.step_manager._latest_obs_for_snapshot` to `build_snapshot()`. This is a single attribute read, not a new public API — the underscore prefix signals it's an internal detail used only by the snapshot pathway.

4. The `PPOAgent` is accessed via `trainer.ppo_agent` (already a public attribute). The `PolicyOutputMapper` is accessed via `trainer.env_manager.policy_mapper` (also public).

**Threading safety**: Policy insight extraction is **gated on training step boundaries**, not wall-clock time. One extra `torch.no_grad()` forward pass per PPO update cycle is negligible. The extraction is skipped when `metrics.processing == True` (PPO update in progress) to avoid reading partially-updated weights.

**Performance guard**: Gated behind `webui.policy_insight: bool` config flag. Disabled entirely when the Streamlit subprocess is not alive.

---

## 3. Layout & Navigation

### 3.1 Tab-Based Navigation

Replace the single scrolling page with `st.tabs()`:

```
┌─────────────────────────────────────────────────────────┐
│  Keisei Training Dashboard           [Auto-refresh: ON] │
│  Timestep: 12,500  Episodes: 85  Speed: 245 steps/s    │
├─────────────────────────────────────────────────────────┤
│  [Metrics]  [Game]  [Lineage]                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│              (Active tab content here)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Header metrics row** stays above the tabs (always visible).

**Tab definitions**:

| Tab | Content | Visible When |
|-----|---------|--------------|
| Metrics | Training curves, experience buffer, gradient norm, processing indicator | Always |
| Game | Board + hands, action heatmap overlay, move log, game status, move statistics, value estimate | Always (shows "Waiting for first episode..." when `board_state` is None) |
| Lineage | Generation tree, Elo rating, recent events, ancestor chain | `env.has_view("lineage")` is True |

**No Evaluation tab in v2**. The tab is added dynamically when `env.has_view("league")` or similar evaluation views become available. Showing a placeholder tab that can never be populated is a broken affordance. The `render_evaluation_tab()` function exists in code, ready for the data.

### 3.2 Metrics Tab (Training Metrics)

```
┌───────────────────────────────────────────────────────┐
│  Metrics                                               │
├───────────────┬───────────────┬───────────────────────┤
│ Policy Loss   │ Value Loss    │  Win Rate Trends      │
│ ┌───────────┐ │ ┌───────────┐ │  ┌─────────────────┐  │
│ │  ~~~~~~~~ │ │ │  ~~~~~~~~ │ │  │  ~~~ ~~~ ~~~    │  │
│ │  ~~~~~~~  │ │ │  ~~~~~~~  │ │  │  Black/White/Draw│  │
│ │  current: │ │ │  current: │ │  └─────────────────┘  │
│ │  0.67     │ │ │  0.78     │ │                       │
│ └───────────┘ │ └───────────┘ │  Gradient Norm: 0.35  │
├───────────────┼───────────────┤  Buffer: [████░░] 50% │
│ Entropy       │ KL Divergence │                       │
│ ┌───────────┐ │ ┌───────────┐ │  Black WR: 41.2%     │
│ │  ~~~~~~~~ │ │ │  ~~~~~~~~ │ │  White WR: 35.3%     │
│ │           │ │ │           │ │  Draw:     23.5%     │
│ └───────────┘ │ └───────────┘ │                       │
├───────────────┼───────────────┤  LR: 3.0e-4          │
│ Clip Fraction │ Ep. Reward    │                       │
│ ┌───────────┐ │ ┌───────────┐ │                       │
│ │  ~~~~~~~~ │ │ │  ~~~~~~~~ │ │                       │
│ └───────────┘ │ └───────────┘ │                       │
└───────────────┴───────────────┴───────────────────────┘
```

**Changes from v1**:
- Chart height increased from 150px to 220px
- Each chart shows the **current value** as a metric annotation below the chart
- Learning rate shown as single inline metric (only rendered as chart if variance detected across the 50-sample window)
- Right column: win rate trends + buffer + gradient norm + summary stats
- 3-column layout: `st.columns([2, 2, 3])` — 2 chart columns (left), summary column (right)
- **Minimum viewport note**: The 3-column layout targets 1920px width. At 1440px, the right column may crowd. Fallback: move buffer bar below the chart columns at narrower viewports.

### 3.3 Game Tab (Board & Policy Insight)

This is the **key differentiating view** — domain-specific ML insight.

```
┌─────────────────────────────────────────────────────────────┐
│  Game                                                        │
├─────────────────────────────┬───────────────────────────────┤
│                             │  Position Assessment           │
│   White (Gote) hand:        │  ┌───────────────────────────┐│
│   Pawn: 1                   │  │ V(s): +0.23 (Black +)     ││
│                             │  │ Confidence: 72% (Focused)  ││
│   9  8  7  6  5  4  3  2  1│  └───────────────────────────┘│
│  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐│                               │
│ 1│香│桂│銀│金│王│金│銀│桂│香││  Action Distribution          │
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┤│  ┌───────────────────────────┐│
│ 2│  │飛│  │  │  │  │  │角│  ││  │ P76        23.1%  ███████ ││
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┤│  │ S67         8.4%  ███    ││
│ 3│歩│歩│歩│  │歩│歩│  │歩│歩││  │ G48         6.2%  ██     ││
│  ╞══╪══╪══╪══╪══╪══╪══╪══╪══╡│  │ P25         5.8%  ██     ││
│ 4│  │  │  │歩│  │  │歩│  │  ││  │ B27+        4.1%  █      ││
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┤│  │ ...5 more                ││
│ 5│  │  │  │  │  │  │  │  │  ││  └───────────────────────────┘│
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┤│                               │
│ 6│  │  │歩│  │  │  │  │  │  ││                               │
│  ╞══╪══╪══╪══╪══╪══╪══╪══╪══╡│                               │
│ 7│歩│歩│  │歩│歩│歩│歩│歩│歩││                               │
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┤│                               │
│ 8│  │角│  │  │  │  │  │飛│  ││                               │
│  ├──┼──┼──┼──┼──┼──┼──┼──┼──┤│                               │
│ 9│香│桂│銀│金│王│金│銀│桂│香││                               │
│  └──┴──┴──┴──┴──┴──┴──┴──┴──┘│                               │
│   Black (Sente) hand:       │                               │
│   (empty)                   │                               │
├─────────────────────────────┼───────────────────────────────┤
│  Move 4 — Black to play     │  Move Statistics              │
│                             │  Captures: B 1 / W 1         │
│  Move Log:                  │  Drops:    B 0 / W 0         │
│  ┌─────────────────────┐    │  Promos:   B 1 / W 0         │
│  │ 1. P76    2. P34    │    │                               │
│  │ 3. P26    4. P84    │    │  Hot Squares (heatmap):       │
│  │ ...                 │    │  55 (center), 76, 34          │
│  │ ← auto-scrolls ↓   │    │                               │
│  └─────────────────────┘    │                               │
└─────────────────────────────┴───────────────────────────────┘
```

**Key changes from v1**:

#### Board Coordinates
- **Rows use numeric labels 1-9** (standard Shogi notation), not alphabetic a-i. This matches the codebase in `shogi_game.py` which uses numeric coordinates throughout. Row 1 = top (White/Gote side), Row 9 = bottom (Black/Sente side).
- Columns labeled 9-1 right-to-left (unchanged, correct).
- Cell references use standard Shogi notation: column then row, e.g., "55" = center square, "76" = file 7 rank 6. In display text and aria-labels, use dash-separated format for screen reader legibility: "5-5", "7-6".

#### Promotion Zone Indicators
- Faint background tint on rows 1-3 (`rgba(100,149,237,0.08)` — blue tint, White's promotion zone) and rows 7-9 (`rgba(220,80,80,0.08)` — red tint, Black's promotion zone).
- Rank 3/6 boundaries rendered with thicker 2px borders (shown as `╞══╡` in the wireframe above).

#### Action Heatmap Overlay
- When `policy_insight.action_heatmap` is available and the "Show heatmap" sidebar toggle is on, each square gets a colored overlay.
- **Log scale** (decided). PPO action distributions over 13,527 actions are extremely peaked in Shogi — often >90% probability on one move. A linear scale makes all but 1-2 squares invisible. Log scale makes the full distribution visible.
- Color scale: transparent (low probability) → `rgba(255,165,0, 0.5)` (high probability), with `log(prob + epsilon)` normalized to [0, 1].
- A small **legend** is shown below the board: a gradient bar from transparent to orange, with "Low" and "High" labels, and the text "Log scale".
- **Hot squares** are consolidated into the heatmap overlay as distinct border highlights (2px dashed orange). The separate hot squares text list is removed when heatmap is active; it appears as a fallback when heatmap data is unavailable.

#### Right Panel — Grouped by Question

The right panel is organized into two visual groups with explicit headings:

**"Position Assessment"** (global state evaluation):
- **Value estimate** `V(s)`: Color-coded + directional text label. Green `+0.23 (Black +)` or red `-0.15 (White +)` or grey `0.00 (Even)`. The text label removes ambiguity for users unfamiliar with the sign convention.
- **Confidence meter**: Derived from action entropy. Progress bar from "Uncertain" (high entropy) to "Focused" (low entropy), with percentage.

**"Action Distribution"** (move-level analysis):
- **Top actions**: Horizontal bar chart of top-K actions with probabilities and human-readable Shogi notation.
- **Move notation format**: `PolicyOutputMapper.action_idx_to_usi_move()` produces USI coordinate notation (e.g., `7g7f`, `2b7g+`). The existing move log uses SFEN notation (e.g., `P7g-7f`). The top-actions display should use USI format directly (it's compact and unambiguous). A formatter/adapter is needed if we want SFEN-style labels instead — this is a Phase 3 implementation decision, not a design blocker. The wireframe examples in this document use simplified notation for readability; the implementation should use whatever `PolicyOutputMapper` produces.
- No click-to-inspect in v2 initial (see Section 6.3).

#### Move Log
- **Chronological order** (earliest first, most recent at bottom) — matches standard kifu convention.
- Scrollable container with fixed height (max 300px).
- **Auto-scrolls to bottom** on each data refresh using JavaScript: `element.scrollTop = element.scrollHeight` in the `st.components.v1.html()` call.
- Full game history displayed. The state contract's 20-move window is expanded to include all moves from the current game. The contract field `move_log` capacity is increased to hold up to 300 moves (sufficient for any Shogi game, which typically run 80-120 moves, max ~300 with repetition).
- Move numbers displayed: "1. P76  2. P34  3. P26  4. P84 ..."

### 3.4 Lineage Tab

Mostly unchanged from current implementation, with these additions:

- **Visual timeline**: Replace text-based ancestor chain with a vertical timeline using Streamlit columns and custom HTML. Each node shows: model ID (truncated), Elo delta, event count.
- **Elo trend chart**: Small line chart of Elo rating over generations (if multiple data points available).
- Metrics row stays as-is: Generation, Elo Rating, Events.

---

## 4. Piece Visual Design

### 4.1 Side Differentiation

**Current**: Both sides use identical `#f5deb3` fill. Only 180-degree rotation distinguishes them.

**Proposed — two strong cues with reinforcing detail**:

| Property | Black (Sente) | White (Gote) |
|----------|---------------|--------------|
| Fill | `#f0d9a0` (warm wheat) | `#d4cbb8` (cool grey-tan) |
| Stroke | `#000000` 1.5px | `#000000` 1.5px |
| Kanji color | `#1a1a1a` (normal), `#cc0000` (promoted) | `#1a1a1a` (normal), `#cc0000` (promoted) |
| Rotation | Upright (0 deg) | Inverted (180 deg) |

**Two strong independent cues**: fill shade and rotation. The fill difference (`#f0d9a0` vs `#d4cbb8`) is wider than the original proposal to reliably meet the 3:1 non-text contrast ratio for adjacent pieces on the same board. The stroke is kept identical to avoid relying on a 1.5px distinction at 40px size, which is unreliable on low-DPI displays.

**Contrast verification required**: Before finalizing, measure the contrast ratio between `#f0d9a0` and `#d4cbb8` using a WCAG contrast calculator. Target: >= 3:1 for non-text elements (WCAG 1.4.11). If the ratio is insufficient, increase the separation further — e.g., `#f0d9a0` vs `#c8bfad`.

### 4.2 SVG Generator Changes

Update `piece_svg_generator.py` to accept side-specific colors:

```python
_FILLS = {"black": "#f0d9a0", "white": "#d4cbb8"}
_STROKES = {"black": "#000000", "white": "#000000"}
_KANJI_COLORS = {"black": "#1a1a1a", "white": "#1a1a1a"}
```

### 4.3 Board Cell Styling

```
Standard cell:      #f5deb3 / #deb887 alternating (unchanged)
Promotion zone:     +rgba overlay (subtle tint, see Section 3.3)
Rank 3/6 borders:   2px solid #8b7355 (thicker grid lines)
Heatmap overlay:    rgba(255,165,0, log_scaled_probability * 0.5)
Hot square border:  2px dashed #ff8c00 (when heatmap active)
Focus indicator:    3px solid #4169e1 (royal blue, keyboard nav)
```

---

## 5. Accessibility Improvements

### 5.1 Board Table Markup

Use `role="table"` (not `role="grid"`) to avoid the full ARIA grid keyboard interaction contract, which requires complete arrow-key navigation JavaScript. Individual action buttons can be added inside cells in v2.1 when click-to-inspect is implemented.

```html
<table role="table" aria-label="Shogi board, move 4, Black to play">
  <caption class="sr-only">
    Shogi board position. Black (Sente) plays from bottom, White (Gote) from top.
  </caption>
  <thead>
    <tr>
      <th scope="col" aria-hidden="true"></th>
      <th scope="col">9</th>
      <th scope="col">8</th>
      <!-- ... -->
      <th scope="col">1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">1</th>
      <td aria-label="9-1: white lance">
        <img src="..." alt="white lance" width="40" height="40">
      </td>
      <td aria-label="8-1: white knight">
        <img src="..." alt="white knight" width="40" height="40">
      </td>
      <!-- ... -->
    </tr>
  </tbody>
</table>
```

### 5.2 Color Contrast Fixes

| Element | Current | Proposed | Ratio |
|---------|---------|----------|-------|
| Fallback text (white pieces) | `#c00` on `#deb887` | `#8b0000` on `#deb887` | 5.2:1 (AA pass) |
| Fallback text (black pieces) | `#000` on `#f5deb3` | unchanged | 10.4:1 (AAA pass) |
| Value estimate text | N/A (new) | Green `#006400` / Red `#8b0000` on white bg | Verify >= 4.5:1 |
| Heatmap legend | N/A (new) | `#000` on white bg | 21:1 (pass) |

### 5.3 Screen Reader Support

- Board table gets `aria-label` with game state summary
- Each cell gets `aria-label` with full description: "9-1: white lance" or "5-5: empty square"
- **Scoped `aria-live` region**: A small dedicated `<div role="status" aria-live="polite">` element containing only a one-line summary (e.g., "PPO update in progress" or "Game over — Black wins"). This element is separate from the main content areas — it does NOT wrap the entire fragment or any large section.
- Move log entries are an ordered list `<ol>` for proper enumeration

### 5.4 Keyboard Navigation

Deferred to Phase 4 (items 29-30). Not expected in Phase 1 or 2. When implemented:

- Board cells will contain focusable button elements (not `tabindex` on `<td>`)
- A **"Skip to board"** link will be the first focusable element on the Game tab (Phase 4, item 30)
- **Auto-refresh/focus conflict resolution**: When any board cell has focus, auto-refresh is implicitly paused for the board component. The fragment continues refreshing header metrics and non-board elements. Focus state is preserved via JavaScript `document.activeElement` save/restore around the board HTML re-render. When focus leaves the board, auto-refresh resumes for the full fragment.

---

## 6. State Management

### 6.1 Session State Keys

```python
# In st.session_state
auto_refresh: bool = True          # Pause/resume toggle
show_heatmap: bool = False         # Action probability overlay toggle
last_state: dict | None = None     # Cache for pause mode
```

(Removed `selected_square` and `active_tab` — selected square is deferred to v2.1, and Streamlit manages active tab internally via widget keys.)

### 6.2 Data Loading Pattern (Approach A)

The fragment always runs on a 2-second timer. Pause/resume is handled internally by checking `auto_refresh` and short-circuiting to cached state when paused. This avoids the decorator-evaluation-time pitfall described in Section 2.1.

```python
@st.fragment(run_every=timedelta(seconds=2))
def live_data_section():
    """Fragment that auto-refreshes. Serves cached state when paused."""
    if not st.session_state.get("auto_refresh", True):
        # Paused — render from cached state, don't reload from disk
        state = st.session_state.get("last_state")
        if state is None:
            st.info("Paused. No cached state available.")
            return
    else:
        state = load_state(args.state_file)
        if state is None:
            st.error("Waiting for training data...")
            return
        st.session_state.last_state = state

    env = EnvelopeParser(state)

    render_stale_warning(env)
    render_header_metrics(env)

    # Tabs live inside the fragment so their content refreshes
    available_tabs = ["Metrics", "Game"]
    if env.has_view("lineage"):
        available_tabs.append("Lineage")
    # Future: add "Evaluation" when league/skill views are available

    tabs = st.tabs(available_tabs)
    with tabs[0]:
        render_metrics_tab(env)
    with tabs[1]:
        render_game_tab(env)
    if len(tabs) > 2:
        with tabs[2]:
            render_lineage_tab(env)
```

The page-level code (outside the fragment) handles:
- `st.set_page_config()`
- Sidebar controls: auto-refresh toggle, heatmap toggle, export button

### 6.3 Board Click-to-Inspect (Deferred to v2.1)

**Decision**: Skip click-to-inspect for the v2 initial release. Both reviewers converged on this recommendation.

**Rationale**: The heatmap overlay alone provides most of the insight value (which squares the policy is considering) without needing a click interaction mechanism. The available click handling options all carry risk:
- **Community component** (`st_click_detector`): No maintenance guarantee, dependency risk for a "show it to others" tool.
- **Custom component** (`declare_component`): Significant implementation complexity.
- **Button grid**: Visually compromises the board.

**v2.1 plan**: Implement a custom Streamlit component using `declare_component` with a bidirectional iframe. This gives full control over the board interaction model, supports both click and keyboard navigation, and has no external dependencies. This is scoped as a separate design task.

### 6.4 Selected Square Invalidation (v2.1 Note)

When click-to-inspect is added, the `selected_square` session state must be invalidated when the board position changes (new episode, move played). Compare `board_state.move_count` between refreshes; if it differs, clear `selected_square`.

---

## 7. Configuration

### 7.1 New Config Fields

```yaml
# In default_config.yaml under webui:
webui:
  enabled: true
  port: 8501
  host: localhost
  update_rate_hz: 2.0
  # New in v2:
  policy_insight: true      # Extract action probabilities for heatmap
  policy_insight_top_k: 10  # Number of top actions to include
```

```python
# In config_schema.py
class WebUIConfig(BaseModel):
    enabled: bool = Field(False, description="Enable Streamlit training dashboard")
    port: int = Field(8501, description="Streamlit server port")
    host: str = Field("localhost", description="Server host")
    update_rate_hz: float = Field(2.0, description="State file update frequency in Hz")
    policy_insight: bool = Field(
        True, description="Include policy action probabilities in dashboard state"
    )
    policy_insight_top_k: int = Field(
        10, description="Number of top actions to surface in policy insight panel"
    )
```

### 7.2 Backward Compatibility

- All new envelope fields are optional (`total=False` in TypedDict)
- Schema version stays `v1.0.0` — optional field additions are patch-level per the versioning policy in `view_contracts.py:9` ("patches add optional fields only")
- Old state files without `policy_insight` render normally — the Game tab hides the policy insight panel sections
- `EnvelopeParser` returns `None` for missing fields (existing pattern)

---

## 8. Implementation Phases

### Phase 1A: Critical Fixes (Unblock Interaction)
1. Replace `time.sleep(2) + st.rerun()` with `@st.fragment(run_every=...)` (Approach A — internal pause check)
2. Add pause/resume toggle in sidebar with `st.rerun()` on toggle change

### Phase 1B: Visual Fixes (Parallelizable with Phase 2)
3. Update piece SVGs for side differentiation (verify contrast ratios)
4. Add board semantic markup (`role="table"`, `<th>`, alt text, dash-separated coordinates)
5. Fix board row labels: numeric 1-9 instead of alphabetic a-i

### Phase 2A: Tab Structure (Verify layout before investing in per-tab polish)
6. Refactor `main()` into tab-based layout with `st.tabs()` inside fragment
7. Rename "Dashboard" to "Metrics" tab; conditionally show "Lineage" tab
8. Remove static Evaluation tab placeholder
9. Split rendering into `render_metrics_tab()`, `render_game_tab()`, `render_lineage_tab()`

### Phase 2B: Metrics & Game Tab Polish
10. Add gradient norm display to Metrics tab
11. Display hot squares (text fallback when heatmap unavailable, border overlay when active)
12. Improve chart sizing to 220px, add current-value annotations
13. Move log: chronological ordering with auto-scroll-to-bottom
14. Increase move log contract capacity to 300 moves

### Phase 3: Policy Insight (The Differentiator)

**Pre-Phase 3 task**: Validate the push-model data flow (Section 2.2) by confirming that `StepManager.collect_experience()` can stash the observation tensor without affecting training performance or memory. A lightweight spike — add `_latest_obs_for_snapshot` to StepManager, verify it works end-to-end with a dummy `build_snapshot()` call.

15. Extend state contract with `PolicyInsight` TypedDict
16. Add `_latest_obs_for_snapshot` cache to `StepManager.collect_experience()`
17. Implement `extract_policy_insight()` in `state_snapshot.py` with `torch.no_grad()`
18. Gate extraction on training step boundaries (skip when `processing == True`)
19. Build action heatmap overlay renderer (log scale, with legend)
20. Build top-actions horizontal bar display
21. Build value estimate display with directional text labels
22. Build confidence meter (entropy-derived)
23. Group right panel into "Position Assessment" and "Action Distribution" sections
24. Add `aria-live` status region for game state changes
25. Update `sample_state.json` with sample policy insight data

### Phase 4: Polish & Accessibility
26. Promotion zone visual indicators (rank 3/6 thicker borders + tint)
27. Dark theme detection and CSS variable adaptation
28. Export state button (download current envelope as JSON)
29. Board keyboard navigation (deferred from Phase 1 — requires focus management)
30. "Skip to board" link on Game tab

### Phase v2.1: Board Interactivity (Separate Design)
- Custom Streamlit component for board click handling
- Selected square detail panel
- Per-square action probability breakdown
- Keyboard arrow-key navigation within board
- Auto-refresh pause on board focus

---

## 9. File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `streamlit_app.py` | Major rewrite | Tab layout, fragment refresh, new renderers |
| `piece_svg_generator.py` | Modify | Side-specific fill colors |
| `static/images/*.svg` | Regenerate | New SVGs with differentiated fills |
| `view_contracts.py` | Extend | Add `PolicyInsight` TypedDict, expand `move_log` capacity to 300 |
| `state_snapshot.py` | Extend | Add `extract_policy_insight()`, agent data flow |
| `envelope_parser.py` | Extend | Add `policy_insight` property (file already exists at `keisei/webui/envelope_parser.py`) |
| `config_schema.py` | Extend | Add `policy_insight` config fields |
| `default_config.yaml` | Extend | Add `policy_insight` defaults |
| `sample_state.json` | Update | Add sample policy insight data |
| `training/step_manager.py` | Minor | Stash `_latest_obs_for_snapshot` during `collect_experience()` |
| `streamlit_manager.py` | Minor | Pass `latest_observation` to `build_snapshot()` |

---

## 10. Resolved Decisions

Decisions made during review, with rationale:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Click handling approach | Deferred to v2.1 | Both reviewers recommend shipping heatmap read-only first. Custom component designed separately. |
| 2 | Policy insight performance | Gate on training step boundaries, not wall-clock | One extra `no_grad()` forward pass per PPO update is negligible. Skip during weight updates. |
| 3 | Streamlit version floor | Resolved — `>=1.55.0` | Already set in `pyproject.toml`. Fragment support (`>=1.33`) is satisfied. |
| 4 | Heatmap color scale | Log scale | Action distributions are extremely peaked in Shogi (often >90% on one move). Linear makes all but top action invisible. |
| 5 | Move log history depth | Full game history, scrollable | 20 moves is too few for Shogi (80-120+ moves typical). Contract expanded to 300 moves. Auto-scroll to bottom. |
| 6 | Tab naming | "Metrics" (not "Dashboard") | "Dashboard" is a container concept; "Metrics" matches ML researcher mental model. |
| 7 | Evaluation tab | Conditional (not placeholder) | Tabs are affordances; a perpetually empty tab is a broken promise. Show only when data exists. |
| 8 | Board row labels | Numeric 1-9 (not alphabetic a-i) | Standard Shogi notation uses numeric ranks. Codebase uses numeric coordinates. |
| 9 | ARIA role for board | `role="table"` (not `role="grid"`) | Grid role requires full keyboard interaction contract. Table is appropriate until click-to-inspect adds interactive elements. |
| 10 | Hot squares display | Heatmap border overlay (primary), text fallback (secondary) | Single visual system when heatmap active; text list only when heatmap unavailable. |
| 11 | Piece stroke differentiation | Same stroke both sides | 1.5px stroke difference unreliable at 40px on low-DPI. Rely on fill + rotation instead. |
| 12 | Fragment/tab boundary | Tabs inside fragment | Only fragment-internal code re-executes on refresh. Tabs must be inside to get fresh data. |
| 13 | Pause/resume mechanism | Approach A (always-on fragment, internal check) | Decorator `run_every` is evaluated once at definition time. Internal short-circuit avoids the gotcha. |
| 14 | Observation tensor stash | `obs.detach().clone()` | Bare reference aliases a tensor modified in-place by env reset/next step. Clone required for correctness. |
| 15 | Move notation in top-actions | USI format from `PolicyOutputMapper` | Direct output, no adapter needed. Compact and unambiguous. SFEN adapter optional. |

---

## 11. Open Items (Pre-Implementation)

1. **Contrast verification**: Measure `#f0d9a0` vs `#d4cbb8` with a WCAG contrast calculator. If < 3:1, widen the gap (e.g., `#c8bfad` for white pieces).

2. **Dark theme**: Board colors (`#f5deb3`, `#deb887`) and piece fills chosen for light mode will look different on dark backgrounds. Document specific dark-theme color overrides needed, even if implementation is deferred to Phase 4.
