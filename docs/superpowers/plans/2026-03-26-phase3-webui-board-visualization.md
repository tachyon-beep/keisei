# Phase 3: WebUI Board Visualization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Streamlit training dashboard visually useful — add Shogi piece SVGs for board readability, wire the lineage panel to the Phase 1 read model, fix the outdated demo data, and verify everything works.

**Architecture:** The WebUI is a read-only Streamlit dashboard consuming a JSON state file written by the training loop. State flows: `Trainer` → `state_snapshot.build_snapshot()` → atomic JSON file → `streamlit_app.py` via `EnvelopeParser`. SVG pieces are loaded once at import time as base64 data URIs. The lineage view adds a new extraction path: `LineageRegistry` → `LineageGraph` → `extract_lineage_summary()` → envelope `lineage` key → `EnvelopeParser.lineage` → `render_lineage_panel()`.

**Tech Stack:** Python, Streamlit, SVG (inline generation), Pydantic config, JSONL lineage events

**Spec:** `docs/superpowers/specs/2026-03-25-phase3-webui-board-visualization-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `keisei/webui/static/images/*.svg` (28 files) | Shogi piece graphics |
| Create | `keisei/webui/piece_svg_generator.py` | Standalone SVG generation script |
| Modify | `keisei/webui/view_contracts.py:224-233` | Expand `LineageViewState` fields |
| Modify | `keisei/webui/state_snapshot.py:180-202` | Add lineage extraction to `build_snapshot()` |
| Modify | `keisei/webui/envelope_parser.py` (after line 91) | Add `lineage` accessor property |
| Modify | `keisei/webui/streamlit_app.py:286-295` | Replace lineage placeholder with real panel |
| Modify | `keisei/webui/sample_state.json` | Update to v1 BroadcastStateEnvelope format |
| Create | `tests/webui/test_piece_svgs.py` | SVG existence and loading tests |
| Create | `tests/webui/test_lineage_panel.py` | Lineage extraction and parser tests |
| Modify | `tests/webui/conftest.py` | Add lineage-enabled envelope factory |

---

## Task 1: Generate SVG Piece Assets

28 SVG files using the traditional Shogi pentagonal piece shape with kanji characters.

**Files:**
- Create: `keisei/webui/piece_svg_generator.py`
- Create: `keisei/webui/static/images/*.svg` (28 files)
- Create: `tests/webui/test_piece_svgs.py`

### Piece Catalog

| Base Type | Kanji (unpromoted) | Promoted Type | Kanji (promoted, red) |
|-----------|-------------------|---------------|----------------------|
| king | 王 (Ōshō) / 玉 (Gyokushō) | — | — |
| rook | 飛 (Hisha) | promoted_rook | 龍 (Ryū) |
| bishop | 角 (Kaku) | promoted_bishop | 馬 (Uma) |
| gold | 金 (Kin) | — | — |
| silver | 銀 (Gin) | promoted_silver | 全 (Zen) |
| knight | 桂 (Kei) | promoted_knight | 圭 (Kei) |
| lance | 香 (Kyō) | promoted_lance | 杏 (Kyō) |
| pawn | 歩 (Fu) | promoted_pawn | と (To) |

Colors: black (sente, points up), white (gote, rotated 180°).
Naming convention: `{type}_{color}.svg` matching `_piece_image_key()` in `streamlit_app.py:57-63`.

### SVG Design

Each piece is a 40×44px pentagonal shape:
- Fill: `#f5deb3` (wheat/tan)
- Stroke: `#000000`, 1.5px
- Kanji: centered, 18px font, `#000000` for unpromoted, `#cc0000` for promoted
- White pieces: entire group rotated 180° around center
- Pentagon path: flat top, angled sides narrowing to a point at bottom

- [ ] **Step 1: Write SVG validation tests**

Create `tests/webui/test_piece_svgs.py`:

```python
import pytest
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent.parent.parent / "keisei" / "webui" / "static" / "images"

EXPECTED_PIECES = [
    "king", "rook", "bishop", "gold", "silver", "knight", "lance", "pawn",
    "promoted_rook", "promoted_bishop", "promoted_silver",
    "promoted_knight", "promoted_lance", "promoted_pawn",
]
COLORS = ["black", "white"]


@pytest.mark.unit
class TestPieceSVGs:
    def test_images_directory_exists(self):
        assert IMAGES_DIR.is_dir(), f"Missing directory: {IMAGES_DIR}"

    @pytest.mark.parametrize("piece", EXPECTED_PIECES)
    @pytest.mark.parametrize("color", COLORS)
    def test_svg_file_exists(self, piece, color):
        path = IMAGES_DIR / f"{piece}_{color}.svg"
        assert path.exists(), f"Missing SVG: {path.name}"

    @pytest.mark.parametrize("piece", EXPECTED_PIECES)
    @pytest.mark.parametrize("color", COLORS)
    def test_svg_is_valid_xml(self, piece, color):
        import xml.etree.ElementTree as ET
        path = IMAGES_DIR / f"{piece}_{color}.svg"
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag.endswith("svg"), f"{path.name} root is not <svg>"

    @pytest.mark.parametrize("piece", EXPECTED_PIECES)
    @pytest.mark.parametrize("color", COLORS)
    def test_svg_has_viewbox(self, piece, color):
        import xml.etree.ElementTree as ET
        path = IMAGES_DIR / f"{piece}_{color}.svg"
        tree = ET.parse(path)
        root = tree.getroot()
        assert "viewBox" in root.attrib, f"{path.name} missing viewBox"

    def test_total_file_count(self):
        svgs = list(IMAGES_DIR.glob("*.svg"))
        assert len(svgs) == 28, f"Expected 28 SVGs, found {len(svgs)}"

    def test_svg_cache_loads(self):
        """Verify the app's SVG loading function populates the cache."""
        from keisei.webui.streamlit_app import _PIECE_SVG_CACHE, _load_piece_svgs
        _PIECE_SVG_CACHE.clear()
        _load_piece_svgs()
        assert len(_PIECE_SVG_CACHE) == 28, f"Cache has {len(_PIECE_SVG_CACHE)} entries"
        for piece in EXPECTED_PIECES:
            for color in COLORS:
                key = f"{piece}_{color}"
                assert key in _PIECE_SVG_CACHE, f"Missing cache key: {key}"
                assert _PIECE_SVG_CACHE[key].startswith("data:image/svg+xml;base64,")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/webui/test_piece_svgs.py -v --tb=short 2>&1 | head -40`
Expected: FAIL — directory doesn't exist, no SVG files.

- [ ] **Step 3: Create the SVG generator script**

Create `keisei/webui/piece_svg_generator.py` — a standalone script that generates all 28 SVGs.

The generator should:
1. Define the piece catalog (type → kanji mapping)
2. Generate a pentagonal piece shape SVG for each (type, color) combination
3. Apply 180° rotation for white pieces
4. Use red kanji for promoted pieces
5. Write files to `keisei/webui/static/images/`
6. Be runnable via `python -m keisei.webui.piece_svg_generator` (include `if __name__ == "__main__"` block)

Key SVG template structure:
```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 44" width="40" height="44">
  <g transform="...">  <!-- rotate(180, 20, 22) for white -->
    <path d="M20 2 L36 14 L32 42 L8 42 L4 14 Z"
          fill="#f5deb3" stroke="#000" stroke-width="1.5"/>
    <text x="20" y="30" text-anchor="middle" font-size="18"
          font-family="serif" fill="#000">歩</text>
  </g>
</svg>
```

The pentagonal path approximates a Shogi piece: pointed top, widening shoulders, flat bottom.

- [ ] **Step 4: Run the generator**

Run: `python -m keisei.webui.piece_svg_generator`
Expected: 28 `.svg` files created in `keisei/webui/static/images/`

- [ ] **Step 5: Run SVG tests to verify they pass**

Run: `pytest tests/webui/test_piece_svgs.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/webui/piece_svg_generator.py keisei/webui/static/images/ tests/webui/test_piece_svgs.py
git commit -m "feat(webui): add 28 Shogi piece SVG assets with generator script"
```

---

## Task 2: Fix sample_state.json to v1 Envelope Format

The current `sample_state.json` uses the pre-envelope flat format. Demo mode is broken because `EnvelopeParser` expects envelope keys.

**Files:**
- Modify: `keisei/webui/sample_state.json`

- [ ] **Step 1: Write a test that validates sample_state.json**

Add to `tests/webui/test_piece_svgs.py` (or a new test if preferred):

```python
class TestSampleState:
    def test_sample_state_passes_envelope_validation(self):
        import json
        from keisei.webui.view_contracts import validate_envelope
        sample_path = Path(__file__).resolve().parent.parent.parent / "keisei" / "webui" / "sample_state.json"
        with open(sample_path) as f:
            data = json.load(f)
        errors = validate_envelope(data)
        assert errors == [], f"Validation errors: {errors}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/webui/test_piece_svgs.py::TestSampleState -v`
Expected: FAIL — missing required keys (schema_version, mode, active_views, health, training, pending_updates).

- [ ] **Step 3: Update sample_state.json to v1 format**

Wrap the existing board/metrics/step/buffer/model data under a `training` key and add the required envelope fields. Keep all existing sample data intact — just restructure it.

The updated file should have:
```json
{
  "schema_version": "v1.0.0",
  "timestamp": 1707400000.0,
  "speed": 245.3,
  "mode": "single_opponent",
  "active_views": ["training"],
  "health": {
    "training": "ok",
    "league": "missing",
    "lineage": "missing",
    "skill_differential": "missing",
    "model_profile": "missing"
  },
  "training": {
    "board_state": { ... existing ... },
    "metrics": { ... existing ... },
    "step_info": { ... existing ... },
    "buffer_info": { ... existing ... },
    "model_info": { ... existing ... }
  },
  "pending_updates": {}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/webui/test_piece_svgs.py::TestSampleState -v`
Expected: PASS

- [ ] **Step 5: Run full webui test suite to check nothing broke**

Run: `pytest tests/webui/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/webui/sample_state.json tests/webui/test_piece_svgs.py
git commit -m "fix(webui): update sample_state.json to v1 BroadcastStateEnvelope format"
```

---

## Task 3: Expand LineageViewState and Add Extraction

Wire the Phase 1 lineage read model into the snapshot pipeline.

**Files:**
- Modify: `keisei/webui/view_contracts.py:224-233` — expand `LineageViewState`
- Modify: `keisei/webui/state_snapshot.py:180-202` — add `extract_lineage_summary()` and update `build_snapshot()`
- Modify: `keisei/webui/envelope_parser.py` — add `lineage` property
- Create: `tests/webui/test_lineage_panel.py`
- Modify: `tests/webui/conftest.py` — add lineage-enabled envelope factory

### 3a. Expand LineageViewState

- [ ] **Step 1: Write test for expanded LineageViewState**

Create `tests/webui/test_lineage_panel.py`:

```python
import pytest
from keisei.webui.view_contracts import validate_envelope, make_health_map, SCHEMA_VERSION


def _make_lineage_view():
    """Minimal valid LineageViewState with expanded fields."""
    return {
        "event_count": 5,
        "latest_checkpoint_id": "run-1::checkpoint_ts5000",
        "parent_id": "run-1::checkpoint_ts2500",
        "model_id": "run-1::checkpoint_ts5000",
        "run_name": "run-1",
        "generation": 3,
        "latest_rating": 1050.0,
        "recent_events": [
            {"event_type": "checkpoint_created", "model_id": "run-1::checkpoint_ts5000", "emitted_at": "2026-03-26T10:00:00Z"},
        ],
        "ancestor_chain": ["run-1::checkpoint_ts2500", "run-1::init"],
    }


def _make_envelope_with_lineage():
    """Build a valid envelope with lineage view active."""
    from tests.webui.conftest import _make_valid_training_view
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp": 1707400000.0,
        "speed": 42.5,
        "mode": "single_opponent",
        "active_views": ["training", "lineage"],
        "health": make_health_map(training="ok", lineage="ok"),
        "training": _make_valid_training_view(),
        "lineage": _make_lineage_view(),
        "pending_updates": {},
    }


@pytest.mark.unit
class TestLineageViewState:
    def test_envelope_with_lineage_validates(self):
        env = _make_envelope_with_lineage()
        errors = validate_envelope(env)
        assert errors == [], f"Validation errors: {errors}"

    def test_envelope_without_lineage_validates(self):
        env = _make_envelope_with_lineage()
        del env["lineage"]
        env["active_views"] = ["training"]
        env["health"]["lineage"] = "missing"
        errors = validate_envelope(env)
        assert errors == [], f"Validation errors: {errors}"
```

- [ ] **Step 2: Run test to verify baseline**

Run: `pytest tests/webui/test_lineage_panel.py::TestLineageViewState -v`
Expected: PASS (validation doesn't deep-check optional views)

- [ ] **Step 3: Expand LineageViewState in view_contracts.py**

Update `keisei/webui/view_contracts.py:224-233`. Add fields that the lineage read model can populate:

```python
class LineageViewState(TypedDict):
    """Lineage / provenance view — populated when lineage is enabled.

    Backed by append-only JSONL events and the LineageGraph read model.
    """

    event_count: int
    latest_checkpoint_id: Optional[str]
    parent_id: Optional[str]
    model_id: Optional[str]
    run_name: Optional[str]
    generation: int  # length of ancestor chain + 1
    latest_rating: Optional[float]
    recent_events: List[Dict[str, Any]]  # last 10 raw events (summary)
    ancestor_chain: List[str]  # model_ids from parent to root
```

- [ ] **Step 4: Commit**

```bash
git add keisei/webui/view_contracts.py tests/webui/test_lineage_panel.py
git commit -m "feat(webui): expand LineageViewState with model_id, generation, ancestor_chain"
```

### 3b. Add Lineage Extraction to state_snapshot.py

- [ ] **Step 5: Write test for lineage extraction**

Add to `tests/webui/test_lineage_panel.py`:

```python
@pytest.mark.unit
class TestLineageExtraction:
    def test_extract_lineage_summary_with_events(self):
        from keisei.webui.state_snapshot import extract_lineage_summary
        from keisei.lineage.registry import LineageRegistry
        from keisei.lineage.graph import LineageGraph
        import tempfile, json
        from pathlib import Path

        # Create a minimal JSONL registry with known events
        events = [
            {"event_id": "evt-001", "event_type": "training_started", "model_id": "run::init",
             "run_name": "run", "emitted_at": "2026-01-01T00:00:00Z", "sequence_number": 0,
             "payload": {}},
            {"event_id": "evt-002", "event_type": "checkpoint_created", "model_id": "run::cp1",
             "run_name": "run", "emitted_at": "2026-01-01T01:00:00Z", "sequence_number": 1,
             "payload": {"parent_model_id": "run::init", "checkpoint_path": "/tmp/cp1.pt",
                         "global_timestep": 1000}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
            tmp_path = Path(f.name)

        try:
            registry = LineageRegistry(tmp_path)
            graph = LineageGraph.from_events(registry.load_all())
            result = extract_lineage_summary(registry, graph, current_model_id="run::cp1")

            assert result["event_count"] == 2
            assert result["model_id"] == "run::cp1"
            assert result["parent_id"] == "run::init"
            assert result["generation"] == 2  # init -> cp1
            assert "run::init" in result["ancestor_chain"]
        finally:
            tmp_path.unlink()

    def test_extract_lineage_summary_empty_registry(self):
        from keisei.webui.state_snapshot import extract_lineage_summary
        from keisei.lineage.registry import LineageRegistry
        from keisei.lineage.graph import LineageGraph
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            registry = LineageRegistry(tmp_path)
            graph = LineageGraph.from_events(registry.load_all())
            result = extract_lineage_summary(registry, graph, current_model_id=None)

            assert result["event_count"] == 0
            assert result["model_id"] is None
            assert result["generation"] == 0
            assert result["ancestor_chain"] == []
        finally:
            tmp_path.unlink()
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/webui/test_lineage_panel.py::TestLineageExtraction -v`
Expected: FAIL — `extract_lineage_summary` doesn't exist yet.

- [ ] **Step 7: Implement extract_lineage_summary in state_snapshot.py**

Add to `keisei/webui/state_snapshot.py`, before `build_snapshot()`:

```python
def extract_lineage_summary(
    registry: Any,
    graph: Any,
    current_model_id: Optional[str],
) -> Dict[str, Any]:
    """Extract lineage summary for the broadcast envelope.

    Parameters
    ----------
    registry : LineageRegistry
        The event registry (for event_count and recent events).
    graph : LineageGraph
        The read model built from registry events.
    current_model_id : str or None
        The model_id of the currently active checkpoint.
    """
    event_count = registry.event_count

    if current_model_id is None or graph.get_node(current_model_id) is None:
        return {
            "event_count": event_count,
            "latest_checkpoint_id": None,
            "parent_id": None,
            "model_id": None,
            "run_name": None,
            "generation": 0,
            "latest_rating": None,
            "recent_events": [],
            "ancestor_chain": [],
        }

    node = graph.get_node(current_model_id)
    ancestors = graph.ancestors(current_model_id)

    # Recent events: last 10 from registry, summarised
    all_events = registry.load_all()
    recent = [
        {
            "event_type": e["event_type"],
            "model_id": e["model_id"],
            "emitted_at": e.get("emitted_at", ""),
        }
        for e in all_events[-10:]
    ]

    return {
        "event_count": event_count,
        "latest_checkpoint_id": current_model_id,
        "parent_id": node.parent_model_id,
        "model_id": current_model_id,
        "run_name": node.run_name,
        "generation": len(ancestors) + 1,
        "latest_rating": node.latest_rating,
        "recent_events": recent,
        "ancestor_chain": [a.model_id for a in ancestors],
    }
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/webui/test_lineage_panel.py::TestLineageExtraction -v`
Expected: PASS

- [ ] **Step 9: Update build_snapshot() to include lineage**

Modify `build_snapshot()` in `state_snapshot.py` to accept optional lineage parameters:

```python
def build_snapshot(
    trainer: Any,
    speed: float = 0.0,
    pending_updates: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    training = _build_training_view(trainer)

    active_views = ["training"]
    health_overrides = {"training": "ok"}

    # Lineage view — only when registry is available
    lineage_data = None
    registry = getattr(trainer, "lineage_registry", None)
    if registry is not None:
        from keisei.lineage.graph import LineageGraph
        graph = LineageGraph.from_events(registry.load_all())
        current_model_id = getattr(trainer.model_manager, "current_model_id", None)
        lineage_data = extract_lineage_summary(registry, graph, current_model_id)
        active_views.append("lineage")
        health_overrides["lineage"] = "ok"

    envelope = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.time(),
        "speed": speed,
        "mode": _resolve_mode(trainer),
        "active_views": active_views,
        "health": make_health_map(**health_overrides),
        "training": training,
        "pending_updates": sanitize_pending_updates(pending_updates),
    }

    if lineage_data is not None:
        envelope["lineage"] = lineage_data

    return envelope
```

Note: We need to check whether `model_manager` exposes `current_model_id`. If not, derive it from the latest checkpoint event in the registry. The implementation should handle the case where no checkpoint has been created yet.

- [ ] **Step 10: Commit**

```bash
git add keisei/webui/state_snapshot.py tests/webui/test_lineage_panel.py
git commit -m "feat(webui): add lineage extraction to build_snapshot pipeline"
```

### 3c. Add Lineage Accessor to EnvelopeParser

- [ ] **Step 11: Write test for EnvelopeParser lineage property**

Add to `tests/webui/test_lineage_panel.py`:

```python
from keisei.webui.envelope_parser import EnvelopeParser


@pytest.mark.unit
class TestEnvelopeParserLineage:
    def test_lineage_present(self):
        env = _make_envelope_with_lineage()
        parser = EnvelopeParser(env)
        assert parser.lineage is not None
        assert parser.lineage["model_id"] == "run-1::checkpoint_ts5000"
        assert parser.lineage["generation"] == 3

    def test_lineage_absent(self):
        env = _make_envelope_with_lineage()
        del env["lineage"]
        parser = EnvelopeParser(env)
        assert parser.lineage is None

    def test_lineage_in_available_optional_views(self):
        env = _make_envelope_with_lineage()
        parser = EnvelopeParser(env)
        assert "lineage" in parser.available_optional_views()

    def test_lineage_not_in_missing_views_when_active(self):
        env = _make_envelope_with_lineage()
        parser = EnvelopeParser(env)
        assert "lineage" not in parser.missing_optional_views()
```

- [ ] **Step 12: Run test to verify it fails**

Run: `pytest tests/webui/test_lineage_panel.py::TestEnvelopeParserLineage -v`
Expected: FAIL — `EnvelopeParser` has no `lineage` property.

- [ ] **Step 13: Add lineage property to EnvelopeParser**

Add to `keisei/webui/envelope_parser.py`, after the `model_info` property (around line 91):

```python
    # -- optional views ----------------------------------------------------

    @property
    def lineage(self) -> Optional[Dict[str, Any]]:
        """Lineage view payload, or ``None`` when lineage is not active."""
        return self._raw.get("lineage")
```

- [ ] **Step 14: Run test to verify it passes**

Run: `pytest tests/webui/test_lineage_panel.py::TestEnvelopeParserLineage -v`
Expected: PASS

- [ ] **Step 15: Commit**

```bash
git add keisei/webui/envelope_parser.py tests/webui/test_lineage_panel.py
git commit -m "feat(webui): add lineage accessor to EnvelopeParser"
```

---

## Task 4: Add Lineage Panel to Streamlit App

Replace the placeholder in `streamlit_app.py` with a real lineage panel.

**Files:**
- Modify: `keisei/webui/streamlit_app.py:286-295, 387-388`

- [ ] **Step 1: Add render_lineage_panel function**

Add to `streamlit_app.py`, before `render_optional_view_placeholders()`:

```python
def render_lineage_panel(env: EnvelopeParser) -> None:
    """Render the model lineage panel when lineage data is available."""
    lineage = env.lineage
    if lineage is None:
        return

    st.subheader("Model Lineage")

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Generation", lineage.get("generation", 0))
    with c2:
        rating = lineage.get("latest_rating")
        st.metric("Elo Rating", f"{rating:.0f}" if rating is not None else "—")
    with c3:
        st.metric("Events", lineage.get("event_count", 0))

    # Current model info
    model_id = lineage.get("model_id", "—")
    parent_id = lineage.get("parent_id", "—")
    run_name = lineage.get("run_name", "—")
    st.text(f"Model:  {model_id}")
    st.text(f"Parent: {parent_id}")
    st.text(f"Run:    {run_name}")

    # Ancestor chain
    ancestors = lineage.get("ancestor_chain", [])
    if ancestors:
        with st.expander(f"Ancestor chain ({len(ancestors)} models)", expanded=False):
            for i, ancestor_id in enumerate(ancestors):
                prefix = "└── " if i == len(ancestors) - 1 else "├── "
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
```

- [ ] **Step 2: Wire lineage panel into main layout**

In `streamlit_app.py`, in the `main()` function, replace the call to `render_optional_view_placeholders(env)` at line 388 with logic that renders the lineage panel if available, then shows placeholders for remaining missing views:

```python
    # --- Optional views ---
    if env.has_view("lineage"):
        render_lineage_panel(env)

    # Show placeholders for views that are still missing
    render_optional_view_placeholders(env)
```

This is minimally invasive — `render_optional_view_placeholders` already skips active views, so the lineage placeholder will disappear automatically when lineage is active.

- [ ] **Step 3: Run full webui test suite**

Run: `pytest tests/webui/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add keisei/webui/streamlit_app.py
git commit -m "feat(webui): add lineage panel with generation, ancestry, and recent events"
```

---

## Task 5: Add Lineage-Enabled Envelope to Test Conftest

**Files:**
- Modify: `tests/webui/conftest.py`

- [ ] **Step 1: Add lineage factory and fixture to conftest.py**

Add at the end of `tests/webui/conftest.py`:

```python
def _make_lineage_view():
    """Minimal valid LineageViewState."""
    return {
        "event_count": 5,
        "latest_checkpoint_id": "run-1::checkpoint_ts5000",
        "parent_id": "run-1::checkpoint_ts2500",
        "model_id": "run-1::checkpoint_ts5000",
        "run_name": "run-1",
        "generation": 3,
        "latest_rating": 1050.0,
        "recent_events": [
            {"event_type": "checkpoint_created", "model_id": "run-1::checkpoint_ts5000",
             "emitted_at": "2026-03-26T10:00:00Z"},
        ],
        "ancestor_chain": ["run-1::checkpoint_ts2500", "run-1::init"],
    }


def _make_valid_envelope_with_lineage(ts=None):
    """Valid envelope with lineage view active."""
    env = _make_valid_envelope(ts)
    env["active_views"].append("lineage")
    env["health"]["lineage"] = "ok"
    env["lineage"] = _make_lineage_view()
    return env


@pytest.fixture
def valid_envelope_with_lineage():
    """Canonical valid v1 envelope with lineage view populated."""
    return _make_valid_envelope_with_lineage()
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/webui/ -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/webui/conftest.py
git commit -m "test(webui): add lineage-enabled envelope factory to conftest"
```

---

## Task 6: Add model_manager.current_model_id Property

The `build_snapshot()` lineage integration needs to know the current model ID. Check if `ModelManager` already exposes this. If not, add a simple property.

**Files:**
- Modify: `keisei/training/model_manager.py` (if needed)
- Create: test in `tests/unit/test_model_manager.py` (if needed)

- [ ] **Step 1: Check if current_model_id exists**

Run: `grep -n "current_model_id" keisei/training/model_manager.py`

If it exists, skip to Task 7. If not:

- [ ] **Step 2: Add current_model_id property**

The model manager tracks the `_run_name` and emits checkpoint events with model IDs following the pattern `{run_name}::checkpoint_ts{timestep}`. Add a property that returns the most recently saved checkpoint model ID, or construct it from the last checkpoint path.

Implementation depends on what's tracked — if `_lineage_registry` is set, the most recent `checkpoint_created` event gives us the model_id. Otherwise, return `None`.

```python
@property
def current_model_id(self) -> Optional[str]:
    """The model_id of the most recently saved checkpoint, or None."""
    if self._lineage_registry is None:
        return None
    events = self._lineage_registry.load_all()
    for event in reversed(events):
        if event["event_type"] == "checkpoint_created":
            return event["model_id"]
    return None
```

- [ ] **Step 3: Commit (if changes were made)**

```bash
git add keisei/training/model_manager.py
git commit -m "feat(model_manager): add current_model_id property for lineage integration"
```

---

## Task 7: End-to-End Verification

Verify the full dashboard works in demo mode and (optionally) live mode.

**Files:** No code changes — verification only.

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short 2>&1 | tail -30`
Expected: All PASS, no regressions.

- [ ] **Step 2: Run type checker**

Run: `mypy keisei/webui/ --ignore-missing-imports`
Expected: No errors (or only pre-existing ones).

- [ ] **Step 3: Run linter**

Run: `flake8 keisei/webui/`
Expected: Clean.

- [ ] **Step 4: Run formatter**

Run: `black --check keisei/webui/`
Expected: All files formatted.

- [ ] **Step 5: Verify demo mode (manual)**

Run: `streamlit run keisei/webui/streamlit_app.py`
Expected: Dashboard loads with sample data, board shows SVG pieces (not text fallback), all charts render, no lineage panel (not in sample data).

- [ ] **Step 6: Add lineage to sample_state.json for demo verification (optional)**

If we want the demo to show the lineage panel, add a `lineage` key to `sample_state.json` with sample data:

```json
{
  "active_views": ["training", "lineage"],
  "health": { ..., "lineage": "ok" },
  "lineage": {
    "event_count": 12,
    "latest_checkpoint_id": "demo-run::checkpoint_ts12500",
    "parent_id": "demo-run::checkpoint_ts10000",
    "model_id": "demo-run::checkpoint_ts12500",
    "run_name": "demo-run",
    "generation": 6,
    "latest_rating": 1120.5,
    "recent_events": [
      {"event_type": "checkpoint_created", "model_id": "demo-run::checkpoint_ts12500", "emitted_at": "2026-03-26T12:00:00Z"},
      {"event_type": "model_promoted", "model_id": "demo-run::checkpoint_ts10000", "emitted_at": "2026-03-26T11:30:00Z"}
    ],
    "ancestor_chain": ["demo-run::checkpoint_ts10000", "demo-run::checkpoint_ts7500", "demo-run::checkpoint_ts5000", "demo-run::checkpoint_ts2500", "demo-run::init"]
  }
}
```

- [ ] **Step 7: Commit any verification fixes**

```bash
git add -u
git commit -m "fix(webui): verification fixes from Phase 3 E2E testing"
```

---

## Summary

| Task | What | Files Changed | Tests |
|------|------|--------------|-------|
| 1 | SVG piece assets + generator | 30 new files | `test_piece_svgs.py` |
| 2 | Fix sample_state.json format | 1 modified | `test_piece_svgs.py::TestSampleState` |
| 3 | Lineage extraction pipeline | 3 modified, 1 new test | `test_lineage_panel.py` |
| 4 | Lineage panel in Streamlit | 1 modified | Manual + existing |
| 5 | Test conftest lineage factory | 1 modified | Fixture only |
| 6 | current_model_id property | 1 modified (maybe) | Existing |
| 7 | End-to-end verification | 0-1 modified | Full suite |

**Estimated commits:** 7-8
**Risk areas:** SVG rendering in Streamlit (base64 data URIs tested by cache test), `model_manager.current_model_id` availability (Task 6 handles this)
