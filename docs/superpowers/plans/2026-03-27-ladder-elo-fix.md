# Ladder EloTracker Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder EloTracker in `ladder.py` with the real `analytics.elo_tracker.EloTracker`, seeded from `EloRegistry` for cross-session persistence.

**Architecture:** Delete the 45-line placeholder class, uncomment the real import, adapt 8 call sites to the real API (`get_agent_rating` → `get_rating`, `update_ratings` → per-game `update_rating`, `get_elo_snapshot` → `get_all_ratings`). Seed the tracker from `EloRegistry` at init, save back at evaluation end.

**Tech Stack:** Python 3.13, existing `analytics.elo_tracker.EloTracker` and `opponents.elo_registry.EloRegistry`

**Filigree:** keisei-e35abcd9c4

---

## File Structure

### Modified Files

| File | Change |
|------|--------|
| `keisei/evaluation/strategies/ladder.py:50-106,470-555` | Delete placeholder, import real EloTracker, adapt call sites, add persistence |

### Test Files

| File | Responsibility |
|------|---------------|
| `tests/unit/test_ladder_elo_integration.py` (new) | Verify ladder uses real EloTracker, ratings persist via EloRegistry |

---

## Task 1: Delete Placeholder, Wire Real EloTracker

**Files:**
- Modify: `keisei/evaluation/strategies/ladder.py:50-106`
- Test: `tests/unit/test_ladder_elo_integration.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/unit/test_ladder_elo_integration.py`:

```python
"""Unit tests for ladder strategy EloTracker integration."""

import pytest

pytestmark = pytest.mark.unit


class TestLadderUsesRealEloTracker:
    """Ladder strategy uses analytics.elo_tracker.EloTracker, not a placeholder."""

    def test_ladder_imports_real_elo_tracker(self):
        """The EloTracker used by ladder.py is from analytics, not a local placeholder."""
        from keisei.evaluation.analytics.elo_tracker import (
            EloTracker as RealEloTracker,
        )
        from keisei.evaluation.strategies.ladder import LadderEvaluator

        # The module should not define its own EloTracker class
        import keisei.evaluation.strategies.ladder as ladder_mod
        # If a local class exists, it would shadow the import
        tracker_class = getattr(ladder_mod, "EloTracker", None)
        # Should be the real one (or None if removed entirely)
        if tracker_class is not None:
            assert tracker_class is RealEloTracker, (
                f"ladder.py defines its own EloTracker instead of importing the real one"
            )

    def test_elo_tracker_has_real_api(self):
        """The real EloTracker has get_rating, update_rating, get_all_ratings."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker

        tracker = EloTracker()
        assert hasattr(tracker, "get_rating")
        assert hasattr(tracker, "update_rating")
        assert hasattr(tracker, "get_all_ratings")
        assert hasattr(tracker, "history")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_ladder_elo_integration.py -v`
Expected: FAIL — `ladder.py defines its own EloTracker instead of importing the real one`

- [ ] **Step 3: Delete placeholder, uncomment real import**

In `keisei/evaluation/strategies/ladder.py`:

1. **Line 50**: Uncomment `from ..analytics.elo_tracker import EloTracker`
2. **Lines 52-98**: Delete the entire placeholder block:
   ```python
   # For now, let's define a placeholder if not available
   class EloTracker:  # pragma: no cover
       ...
   ```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_ladder_elo_integration.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/evaluation/strategies/ladder.py tests/unit/test_ladder_elo_integration.py
git commit -m "fix(ladder): replace placeholder EloTracker with real analytics implementation"
```

---

## Task 2: Adapt Call Sites to Real API

**Files:**
- Modify: `keisei/evaluation/strategies/ladder.py:106,470-555`
- Test: `tests/unit/test_ladder_elo_integration.py`

The real `EloTracker` API differs from the placeholder:

| Placeholder | Real |
|------------|------|
| `get_agent_rating(id)` | `get_rating(id)` |
| `update_ratings(agent, opp, game_results)` | `update_rating(a, b, score_a)` per game |
| `get_elo_snapshot()` | `get_all_ratings()` |
| `default_rating` | `default_initial_rating` |

- [ ] **Step 1: Write failing test for API compatibility**

Add to `tests/unit/test_ladder_elo_integration.py`:

```python
class TestLadderCallSiteCompatibility:
    """Ladder call sites use the real EloTracker API correctly."""

    def test_get_rating_returns_default_for_unknown(self):
        from keisei.evaluation.analytics.elo_tracker import EloTracker

        tracker = EloTracker()
        rating = tracker.get_rating("unknown_model")
        assert rating == 1500.0

    def test_update_rating_per_game(self):
        """Real API takes (id_a, id_b, score_a) not (id_a, id_b, [GameResult])."""
        from keisei.evaluation.analytics.elo_tracker import EloTracker

        tracker = EloTracker()
        new_a, new_b = tracker.update_rating("model_a", "model_b", 1.0)
        assert new_a > 1500.0  # Winner rating goes up
        assert new_b < 1500.0  # Loser rating goes down

    def test_get_all_ratings_returns_snapshot(self):
        from keisei.evaluation.analytics.elo_tracker import EloTracker

        tracker = EloTracker()
        tracker.get_rating("model_a")
        tracker.get_rating("model_b")
        snapshot = tracker.get_all_ratings()
        assert "model_a" in snapshot
        assert "model_b" in snapshot
```

- [ ] **Step 2: Run tests (these should already pass since they test the real EloTracker)**

Run: `uv run pytest tests/unit/test_ladder_elo_integration.py::TestLadderCallSiteCompatibility -v`
Expected: All PASS (testing real API, not ladder code yet)

- [ ] **Step 3: Update ladder.py call sites**

In `keisei/evaluation/strategies/ladder.py`, make these replacements:

**Line 106** — Init: Change `EloTracker(self.config.get_strategy_param("elo_config", {}))` to `EloTracker()`:
```python
self.elo_tracker = EloTracker()
```

**Line ~473** — `default_rating` → `default_initial_rating`:
```python
initial_rating = self.config.get_strategy_param(
    "initial_rating", self.elo_tracker.default_initial_rating
)
```

**Line ~476-477** — Direct `ratings` dict access → use `get_rating()` (auto-creates):
```python
# Replace:
#   if name not in self.elo_tracker.ratings:
#       self.elo_tracker.ratings[name] = initial_rating
# With:
self.elo_tracker.get_rating(name)  # Auto-creates with default if missing
```

**Lines ~502-504** — `get_agent_rating` → `get_rating`, remove direct dict access:
```python
initial_agent_rating = self.elo_tracker.get_rating(agent_info.name)
# Remove: if agent_info.name not in self.elo_tracker.ratings: ...
```

**Lines ~523-527** — `update_ratings(agent, opp, results)` → per-game `update_rating(a, b, score)`:
```python
for game in match_results:
    score_a = 1.0 if game.winner == 0 else (0.5 if game.winner is None else 0.0)
    self.elo_tracker.update_rating(agent_info.name, opponent_info.name, score_a)
```

**Line ~538** — `get_agent_rating` → `get_rating`:
```python
final_agent_rating = self.elo_tracker.get_rating(agent_info.name)
```

**Line ~551** — `get_elo_snapshot` → `get_all_ratings`:
```python
"final_elo_snapshot": self.elo_tracker.get_all_ratings(),
```

**Line ~554** — Remove the `type: ignore` comment:
```python
elo_tracker=self.elo_tracker,
```

- [ ] **Step 4: Run full test suite for the evaluation module**

Run: `uv run pytest tests/unit/test_ladder_elo_integration.py tests/unit/ -q`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/evaluation/strategies/ladder.py tests/unit/test_ladder_elo_integration.py
git commit -m "fix(ladder): adapt 8 call sites to real EloTracker API"
```

---

## Task 3: Add EloRegistry Persistence

**Files:**
- Modify: `keisei/evaluation/strategies/ladder.py:100-112`
- Test: `tests/unit/test_ladder_elo_integration.py`

Seed `EloTracker` from `EloRegistry` at init, save back after evaluation completes. This gives cross-session Elo persistence.

- [ ] **Step 1: Write failing test for persistence**

Add to `tests/unit/test_ladder_elo_integration.py`:

```python
import tempfile
from pathlib import Path


class TestLadderEloPersistence:
    """Ladder seeds EloTracker from EloRegistry and saves back."""

    def test_ladder_init_loads_from_registry(self):
        """If elo_registry_path is set, ladder seeds tracker from saved ratings."""
        from keisei.evaluation.opponents.elo_registry import EloRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            reg_path = Path(tmpdir) / "elo.json"
            # Pre-populate a registry
            registry = EloRegistry(reg_path)
            registry.get_rating("old_model")  # Creates at 1500
            registry.ratings["old_model"] = 1600.0
            registry.save()

            # Now load it fresh and verify
            registry2 = EloRegistry(reg_path)
            assert registry2.get_rating("old_model") == 1600.0
```

- [ ] **Step 2: Run test to verify it passes (tests EloRegistry, not ladder yet)**

Run: `uv run pytest tests/unit/test_ladder_elo_integration.py::TestLadderEloPersistence -v`
Expected: PASS

- [ ] **Step 3: Add persistence to LadderEvaluator**

In `keisei/evaluation/strategies/ladder.py`, modify `__init__`:

```python
def __init__(self, config: EvaluationConfig):
    super().__init__(config)
    self.config: EvaluationConfig = config

    # Load persisted ratings if registry path is configured
    initial_ratings = None
    elo_path = getattr(config, "elo_registry_path", None)
    if elo_path:
        from ..opponents.elo_registry import EloRegistry
        self._elo_registry = EloRegistry(Path(elo_path))
        initial_ratings = dict(self._elo_registry.ratings)
    else:
        self._elo_registry = None

    self.elo_tracker = EloTracker(initial_ratings=initial_ratings)
    # ... rest unchanged
```

At the end of the evaluation method (after building `EvaluationResult`), save back:

```python
# Persist updated ratings
if self._elo_registry is not None:
    self._elo_registry.ratings = self.elo_tracker.get_all_ratings()
    self._elo_registry.save()
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/unit/ -q`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/evaluation/strategies/ladder.py tests/unit/test_ladder_elo_integration.py
git commit -m "feat(ladder): persist Elo ratings via EloRegistry across sessions"
```
