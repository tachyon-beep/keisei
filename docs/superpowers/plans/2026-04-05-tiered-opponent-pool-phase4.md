# Tiered Opponent Pool Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add concurrent match execution, priority-based scheduling, and role-specific Elo for eviction/promotion to the league tournament system.

**Architecture:** PriorityScorer (scoring formula for matchup informativeness) + ConcurrentMatchPool (interleaved game loop with partitioned VecEnv) + MatchScheduler/TierManager/FrontierPromoter modifications (priority ordering, elo_dynamic eviction, elo_frontier promotion).

**Tech Stack:** Python 3.13, PyTorch, SQLite (WAL), frozen dataclasses, threading, `uv run pytest` for all tests.

**Spec:** `docs/superpowers/specs/2026-04-05-tiered-opponent-pool-phase4-design.md`

---

## Critical Implementation Notes

### Existing Patterns (Read Before Implementing)

1. **Config structure:** All configs are frozen dataclasses with `__post_init__` validation. New configs nest inside `LeagueConfig`. TOML loading in `load_config()` pops nested sections and passes to constructors. See `keisei/config.py`.

2. **OpponentEntry fields:** `elo_dynamic`, `elo_frontier`, `lineage_group`, `parent_entry_id` already exist (Phase 2-3). No schema changes needed for Phase 4.

3. **Test entry creation:** Each test file has its own `_make_entry()` helper. Include all required fields. Use `OpponentEntry(id=..., display_name=..., architecture="resnet", model_params={}, checkpoint_path="/tmp/test.pt", elo_rating=..., created_epoch=0, games_played=0, created_at="2026-01-01", flavour_facts=[], role=..., status=EntryStatus.ACTIVE)`.

4. **TinyModel:** Import from `tests._helpers` for tests needing a model. Returns `SimpleNamespace(policy_logits=..., value_logits=...)` matching KataGo output contract. `policy_logits` shape is `(batch, 1, 11259)` — must be reshaped to `(batch, 11259)` before masking.

5. **VecEnv API:** `reset()` → `SimpleNamespace(observations, legal_masks)`. `step(actions)` → `SimpleNamespace(observations, legal_masks, current_players, rewards, terminated, truncated)`. Observations shape `(num_envs, 50, 9, 9)`, legal_masks `(num_envs, 11259)`. Auto-resets terminated envs.

6. **Model forward in matches:** `play_batch` in `keisei/training/match_utils.py` splits envs by `current_players`, runs separate forward passes per model with legal mask filtering and categorical sampling. The ConcurrentMatchPool replicates this pattern per partition.

7. **Always use `uv run`** for pytest/python.

### File Layout

| File | Action | Responsibility |
|---|---|---|
| `keisei/config.py` | Modify | Add `PriorityScorerConfig`, `ConcurrencyConfig`, wire into `LeagueConfig` |
| `keisei/training/priority_scorer.py` | Create | `PriorityScorer` — matchup priority scoring |
| `keisei/training/match_scheduler.py` | Modify | Accept `PriorityScorer`, return priority-sorted pairings |
| `keisei/training/tier_managers.py` | Modify | `DynamicManager.evict_weakest` uses `elo_dynamic` |
| `keisei/training/frontier_promoter.py` | Modify | `FrontierPromoter.should_promote` uses `elo_frontier` |
| `keisei/training/concurrent_matches.py` | Create | `ConcurrentMatchPool`, `MatchResult`, `_MatchSlot` |
| `keisei/training/tournament.py` | Modify | Use `ConcurrentMatchPool` in `_run_loop` |
| `tests/test_priority_scorer.py` | Create | Unit tests for PriorityScorer |
| `tests/test_concurrent_matches.py` | Create | Unit tests for ConcurrentMatchPool |

---

## Task 1: Config Additions

**Files:**
- Modify: `keisei/config.py`
- Test: `tests/test_league_config.py`

- [ ] **Step 1: Write failing test for PriorityScorerConfig defaults**

Add to `tests/test_league_config.py`:

```python
from keisei.config import PriorityScorerConfig, ConcurrencyConfig


def test_priority_scorer_config_defaults():
    c = PriorityScorerConfig()
    assert c.under_sample_weight == 1.0
    assert c.uncertainty_weight == 0.5
    assert c.recent_fixed_bonus == 0.3
    assert c.diversity_weight == 0.3
    assert c.repeat_penalty == -0.5
    assert c.lineage_penalty == -0.3
    assert c.repeat_window_rounds == 5


def test_concurrency_config_defaults():
    c = ConcurrencyConfig()
    assert c.parallel_matches == 4
    assert c.envs_per_match == 8
    assert c.total_envs == 32
    assert c.max_resident_models == 10


def test_concurrency_config_validation_env_budget():
    import pytest
    with pytest.raises(ValueError, match="total_envs"):
        ConcurrencyConfig(parallel_matches=4, envs_per_match=8, total_envs=16)


def test_concurrency_config_validation_model_budget():
    import pytest
    with pytest.raises(ValueError, match="max_resident_models"):
        ConcurrencyConfig(parallel_matches=4, max_resident_models=4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_league_config.py::test_priority_scorer_config_defaults -v`
Expected: FAIL with `ImportError: cannot import name 'PriorityScorerConfig'`

- [ ] **Step 3: Implement PriorityScorerConfig and ConcurrencyConfig**

Add `import math` at the top of `keisei/config.py` if not already present.

Add to `keisei/config.py` after `RoleEloConfig` (after line 130):

```python
@dataclass(frozen=True)
class PriorityScorerConfig:
    under_sample_weight: float = 1.0
    uncertainty_weight: float = 0.5
    recent_fixed_bonus: float = 0.3
    diversity_weight: float = 0.3
    repeat_penalty: float = -0.5
    lineage_penalty: float = -0.3
    repeat_window_rounds: int = 5

    def __post_init__(self) -> None:
        for field_name in (
            "under_sample_weight", "uncertainty_weight", "recent_fixed_bonus",
            "diversity_weight", "repeat_penalty", "lineage_penalty",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{field_name} must be finite, got {val}")
        if self.repeat_window_rounds < 1:
            raise ValueError(
                f"repeat_window_rounds must be >= 1, got {self.repeat_window_rounds}"
            )


@dataclass(frozen=True)
class ConcurrencyConfig:
    parallel_matches: int = 4
    envs_per_match: int = 8
    total_envs: int = 32
    max_resident_models: int = 10

    def __post_init__(self) -> None:
        needed_envs = self.parallel_matches * self.envs_per_match
        if needed_envs > self.total_envs:
            raise ValueError(
                f"parallel_matches * envs_per_match ({needed_envs}) "
                f"exceeds total_envs ({self.total_envs})"
            )
        min_models = self.parallel_matches * 2
        if self.max_resident_models < min_models:
            raise ValueError(
                f"max_resident_models ({self.max_resident_models}) must be >= "
                f"parallel_matches * 2 ({min_models})"
            )
```

Add to `LeagueConfig` fields (after `role_elo`):

```python
    priority: PriorityScorerConfig = PriorityScorerConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
```

- [ ] **Step 4: Update TOML loading in load_config()**

In `load_config()`, after the line `role_elo_raw = lg.pop("role_elo", {})`, add:

```python
        priority_raw = lg.pop("priority", {})
        concurrency_raw = lg.pop("concurrency", {})
```

And update the `LeagueConfig(...)` constructor call to include:

```python
            priority=PriorityScorerConfig(**priority_raw),
            concurrency=ConcurrencyConfig(**concurrency_raw),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_league_config.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/config.py tests/test_league_config.py
git commit -m "feat(config): add PriorityScorerConfig and ConcurrencyConfig for Phase 4"
```

---

## Task 2: PriorityScorer

**Files:**
- Create: `keisei/training/priority_scorer.py`
- Create: `tests/test_priority_scorer.py`

- [ ] **Step 1: Write failing tests for PriorityScorer**

Create `tests/test_priority_scorer.py`:

```python
"""Tests for PriorityScorer — matchup informativeness ranking."""

from __future__ import annotations

import pytest

from keisei.config import PriorityScorerConfig
from keisei.training.opponent_store import EntryStatus, OpponentEntry, Role
from keisei.training.priority_scorer import PriorityScorer


def _make_entry(
    id: int,
    role: Role = Role.DYNAMIC,
    elo: float = 1000.0,
    lineage: str | None = None,
    parent_id: int | None = None,
) -> OpponentEntry:
    return OpponentEntry(
        id=id,
        display_name=f"e{id}",
        architecture="resnet",
        model_params={},
        checkpoint_path=f"/tmp/{id}.pt",
        elo_rating=elo,
        created_epoch=0,
        games_played=10,
        created_at="2026-01-01",
        flavour_facts=[],
        role=role,
        status=EntryStatus.ACTIVE,
        lineage_group=lineage,
        parent_entry_id=parent_id,
    )


class TestScore:
    def test_under_sampled_pair_scores_higher(self):
        """Pairs with fewer past games get higher priority."""
        scorer = PriorityScorer(PriorityScorerConfig())
        a, b, c = _make_entry(1), _make_entry(2), _make_entry(3)
        scorer.record_result(a.id, b.id)
        scorer.record_result(a.id, b.id)
        scorer.record_result(a.id, b.id)
        score_ab = scorer.score(a, b)
        score_ac = scorer.score(a, c)
        assert score_ac > score_ab

    def test_uncertainty_bonus_for_close_elo(self):
        """Pairs within 100 Elo get uncertainty bonus."""
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, elo=1000.0)
        b_close = _make_entry(2, elo=1050.0)
        b_far = _make_entry(3, elo=1200.0)
        score_close = scorer.score(a, b_close)
        score_far = scorer.score(a, b_far)
        assert score_close > score_far

    def test_recent_fixed_bonus(self):
        """Pairings involving RECENT_FIXED entries get a bonus."""
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, role=Role.DYNAMIC)
        b_rf = _make_entry(2, role=Role.RECENT_FIXED)
        b_dyn = _make_entry(3, role=Role.DYNAMIC)
        score_rf = scorer.score(a, b_rf)
        score_dyn = scorer.score(a, b_dyn)
        assert score_rf > score_dyn

    def test_repeat_penalty(self):
        """Pairs that played recently get penalized."""
        scorer = PriorityScorer(PriorityScorerConfig())
        a, b = _make_entry(1), _make_entry(2)
        score_before = scorer.score(a, b)
        scorer.record_round_result(a.id, b.id)
        scorer.record_round_result(a.id, b.id)
        score_after = scorer.score(a, b)
        assert score_after < score_before

    def test_lineage_penalty_parent_child(self):
        """Direct parent/child pairings are penalized."""
        scorer = PriorityScorer(PriorityScorerConfig())
        parent = _make_entry(1, lineage="lin-1")
        child = _make_entry(2, lineage="lin-1", parent_id=1)
        unrelated = _make_entry(3, lineage="lin-2")
        score_related = scorer.score(parent, child)
        score_unrelated = scorer.score(parent, unrelated)
        assert score_unrelated > score_related

    def test_lineage_penalty_same_group(self):
        """Same lineage group (siblings) gets partial penalty."""
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, lineage="lin-1")
        sibling = _make_entry(2, lineage="lin-1")
        unrelated = _make_entry(3, lineage="lin-2")
        score_sibling = scorer.score(a, sibling)
        score_unrelated = scorer.score(a, unrelated)
        assert score_unrelated > score_sibling

    def test_diversity_bonus_cross_lineage(self):
        """Cross-lineage pairings get a diversity bonus."""
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, lineage="lin-1")
        same = _make_entry(2, lineage="lin-1")
        diff = _make_entry(3, lineage="lin-2")
        score_same = scorer.score(a, same)
        score_diff = scorer.score(a, diff)
        assert score_diff > score_same

    def test_all_weights_zero_produces_zero(self):
        """With all weights zeroed, every pair scores 0."""
        cfg = PriorityScorerConfig(
            under_sample_weight=0.0,
            uncertainty_weight=0.0,
            recent_fixed_bonus=0.0,
            diversity_weight=0.0,
            repeat_penalty=0.0,
            lineage_penalty=0.0,
        )
        scorer = PriorityScorer(cfg)
        a, b = _make_entry(1), _make_entry(2)
        assert scorer.score(a, b) == 0.0

    def test_repeat_window_slides(self):
        """Repeat counts outside the window are forgotten."""
        cfg = PriorityScorerConfig(repeat_window_rounds=2)
        scorer = PriorityScorer(cfg)
        a, b = _make_entry(1), _make_entry(2)
        scorer.record_round_result(a.id, b.id)
        scorer.advance_round()
        scorer.record_round_result(a.id, b.id)
        scorer.advance_round()
        # Two rounds recorded, now advance past window
        scorer.advance_round()
        # First round fell off; only 1 repeat in window
        one_repeat_score = scorer.score(a, b)
        # Fresh scorer with 0 repeats
        fresh_score = PriorityScorer(cfg).score(a, b)
        assert fresh_score > one_repeat_score  # 0 repeats > 1 repeat


class TestScoreRound:
    def test_returns_sorted_by_priority_descending(self):
        """score_round returns pairings ordered highest-priority first."""
        scorer = PriorityScorer(PriorityScorerConfig())
        a = _make_entry(1, elo=1000.0)
        b = _make_entry(2, elo=1050.0)  # close Elo → uncertainty bonus
        c = _make_entry(3, elo=1500.0)  # far Elo → no uncertainty bonus
        pairings = [(a, b), (a, c), (b, c)]
        sorted_pairings = scorer.score_round(pairings)
        scores = [scorer.score(p[0], p[1]) for p in sorted_pairings]
        assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_priority_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'keisei.training.priority_scorer'`

- [ ] **Step 3: Implement PriorityScorer**

Create `keisei/training/priority_scorer.py`:

```python
"""PriorityScorer — ranks matchups by informativeness for scheduling."""

from __future__ import annotations

from collections import Counter, deque

from keisei.config import PriorityScorerConfig
from keisei.training.opponent_store import OpponentEntry, Role


class PriorityScorer:
    """Computes priority scores for candidate pairings.

    Higher scores = more informative matchups that should be played first.
    Maintains in-memory state for pair game counts and round-level repeat tracking.
    """

    def __init__(self, config: PriorityScorerConfig) -> None:
        self.config = config
        self._pair_games: Counter[tuple[int, int]] = Counter()
        self._round_history: deque[set[tuple[int, int]]] = deque(
            maxlen=config.repeat_window_rounds,
        )
        self._current_round_pairs: set[tuple[int, int]] = set()

    def _pair_key(self, id_a: int, id_b: int) -> tuple[int, int]:
        """Canonical (smaller, larger) key for a pair."""
        return (min(id_a, id_b), max(id_a, id_b))

    def record_result(self, id_a: int, id_b: int) -> None:
        """Record that a game was played between these entries."""
        self._pair_games[self._pair_key(id_a, id_b)] += 1

    def record_round_result(self, id_a: int, id_b: int) -> None:
        """Record that this pair was matched in the current round."""
        self._current_round_pairs.add(self._pair_key(id_a, id_b))

    def advance_round(self) -> None:
        """Advance the sliding window: push current round, drop oldest if full."""
        self._round_history.append(self._current_round_pairs)
        self._current_round_pairs = set()

    def _under_sample_bonus(self, id_a: int, id_b: int) -> float:
        count = self._pair_games[self._pair_key(id_a, id_b)]
        return 1.0 / max(1, count)

    def _uncertainty_bonus(self, a: OpponentEntry, b: OpponentEntry) -> float:
        return 1.0 if abs(a.elo_rating - b.elo_rating) < 100 else 0.0

    def _has_recent_fixed(self, a: OpponentEntry, b: OpponentEntry) -> float:
        return 1.0 if a.role == Role.RECENT_FIXED or b.role == Role.RECENT_FIXED else 0.0

    def _lineage_diversity(self, a: OpponentEntry, b: OpponentEntry) -> float:
        if a.lineage_group is None or b.lineage_group is None:
            return 1.0
        return 0.0 if a.lineage_group == b.lineage_group else 1.0

    def _repeat_count(self, id_a: int, id_b: int) -> float:
        key = self._pair_key(id_a, id_b)
        return sum(1 for round_pairs in self._round_history if key in round_pairs)

    def _lineage_closeness(self, a: OpponentEntry, b: OpponentEntry) -> float:
        if a.parent_entry_id == b.id or b.parent_entry_id == a.id:
            return 1.0
        if (
            a.lineage_group is not None
            and b.lineage_group is not None
            and a.lineage_group == b.lineage_group
        ):
            return 0.5
        return 0.0

    def score(self, a: OpponentEntry, b: OpponentEntry) -> float:
        """Compute priority score for a pairing. Higher = more informative."""
        c = self.config
        return (
            c.under_sample_weight * self._under_sample_bonus(a.id, b.id)
            + c.uncertainty_weight * self._uncertainty_bonus(a, b)
            + c.recent_fixed_bonus * self._has_recent_fixed(a, b)
            + c.diversity_weight * self._lineage_diversity(a, b)
            + c.repeat_penalty * self._repeat_count(a.id, b.id)
            + c.lineage_penalty * self._lineage_closeness(a, b)
        )

    def score_round(
        self, pairings: list[tuple[OpponentEntry, OpponentEntry]],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Score all pairings and return sorted by priority (highest first)."""
        scored = [(self.score(a, b), a, b) for a, b in pairings]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(a, b) for _, a, b in scored]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_priority_scorer.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/priority_scorer.py tests/test_priority_scorer.py
git commit -m "feat: add PriorityScorer for matchup informativeness ranking"
```

---

## Task 3: MatchScheduler Priority Integration

**Files:**
- Modify: `keisei/training/match_scheduler.py`
- Modify: `tests/test_match_scheduler.py`

- [ ] **Step 1: Write failing test for priority-ordered generate_round**

The existing `_make_scheduler` in `tests/test_match_scheduler.py` (around line 10) needs to be updated. Also add new tests. First, update the helper:

```python
def _make_scheduler(priority_scorer=None, **overrides):
    defaults = dict(
        learner_dynamic_ratio=0.50,
        learner_frontier_ratio=0.30,
        learner_recent_ratio=0.20,
    )
    defaults.update(overrides)
    config = MatchSchedulerConfig(**defaults)
    return MatchScheduler(config, priority_scorer=priority_scorer)
```

Then add a new test class:

```python
from keisei.config import PriorityScorerConfig
from keisei.training.priority_scorer import PriorityScorer


class TestPriorityRound:
    def test_generate_round_returns_priority_sorted(self):
        """With a scorer, generate_round returns pairings sorted by priority."""
        scorer = PriorityScorer(PriorityScorerConfig())
        scheduler = _make_scheduler(priority_scorer=scorer)
        e1 = _make_entry(1, Role.DYNAMIC, elo=1000.0)
        e2 = _make_entry(2, Role.DYNAMIC, elo=1050.0)  # close to e1
        e3 = _make_entry(3, Role.DYNAMIC, elo=1500.0)  # far from both
        entries = [e1, e2, e3]
        pairings = scheduler.generate_round(entries)
        scores = [scorer.score(a, b) for a, b in pairings]
        assert scores == sorted(scores, reverse=True)

    def test_generate_round_without_scorer_still_works(self):
        """Without a scorer, generate_round returns all pairings (unordered)."""
        scheduler = _make_scheduler()
        entries = [_make_entry(i, Role.DYNAMIC) for i in range(1, 5)]
        pairings = scheduler.generate_round(entries)
        assert len(pairings) == 6  # 4 choose 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_match_scheduler.py::TestPriorityRound -v`
Expected: FAIL with `TypeError` (MatchScheduler doesn't accept priority_scorer)

- [ ] **Step 3: Implement MatchScheduler changes**

Replace the full contents of `keisei/training/match_scheduler.py`:

```python
"""MatchScheduler -- role-weighted opponent selection for the tiered pool."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from keisei.config import MatchSchedulerConfig
from keisei.training.opponent_store import OpponentEntry, Role

if TYPE_CHECKING:
    from keisei.training.priority_scorer import PriorityScorer


class MatchScheduler:
    def __init__(
        self,
        config: MatchSchedulerConfig,
        priority_scorer: PriorityScorer | None = None,
    ) -> None:
        self.config = config
        self._priority_scorer = priority_scorer

    def sample_for_learner(
        self, entries_by_role: dict[Role, list[OpponentEntry]],
    ) -> OpponentEntry:
        ratios = self.effective_ratios(entries_by_role)
        non_empty = {r: w for r, w in ratios.items() if w > 0}
        if not non_empty:
            raise ValueError("No entries available in any tier")
        roles = list(non_empty.keys())
        weights = [non_empty[r] for r in roles]
        chosen_role = random.choices(roles, weights=weights, k=1)[0]
        return random.choice(entries_by_role[chosen_role])

    def generate_round(
        self, entries: list[OpponentEntry],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Generate all N*(N-1)/2 pairings for a full round-robin round.

        If a PriorityScorer is configured, returns pairings sorted by priority
        (highest first). Otherwise, returns shuffled pairings.
        """
        pairings: list[tuple[OpponentEntry, OpponentEntry]] = []
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                pairings.append((entries[i], entries[j]))
        if self._priority_scorer is not None:
            return self._priority_scorer.score_round(pairings)
        random.shuffle(pairings)
        return pairings

    def effective_ratios(
        self, entries_by_role: dict[Role, list[OpponentEntry]],
    ) -> dict[Role, float]:
        raw = {
            Role.DYNAMIC: self.config.learner_dynamic_ratio,
            Role.FRONTIER_STATIC: self.config.learner_frontier_ratio,
            Role.RECENT_FIXED: self.config.learner_recent_ratio,
        }
        non_empty = {r: w for r, w in raw.items() if entries_by_role.get(r)}
        if not non_empty:
            return {r: 0.0 for r in raw}
        total = sum(non_empty.values())
        result = {}
        for role in raw:
            if role in non_empty:
                result[role] = non_empty[role] / total
            else:
                result[role] = 0.0
        return result
```

- [ ] **Step 4: Run full match_scheduler tests**

Run: `uv run pytest tests/test_match_scheduler.py -v`
Expected: all PASS (existing tests + new priority tests)

- [ ] **Step 5: Commit**

```bash
git add keisei/training/match_scheduler.py tests/test_match_scheduler.py
git commit -m "feat(scheduler): integrate PriorityScorer for priority-ordered rounds"
```

---

## Task 4: DynamicManager Uses elo_dynamic for Eviction

**Files:**
- Modify: `keisei/training/tier_managers.py`
- Modify: `tests/test_tier_managers.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_tier_managers.py`. Note: look at the existing test file to find where DynamicManager tests live. Add this new class after the existing Dynamic tests:

```python
class TestDynamicEloDynamic:
    """DynamicManager.evict_weakest uses elo_dynamic, not elo_rating."""

    def test_evict_weakest_uses_elo_dynamic(self, store):
        config = DynamicConfig(slots=10, min_games_before_eviction=0, protection_matches=0)
        mgr = DynamicManager(store, config)
        e1 = _add_entry(store, epoch=1, role=Role.DYNAMIC, elo=1500.0)
        e2 = _add_entry(store, epoch=2, role=Role.DYNAMIC, elo=1000.0)
        # e1: HIGH elo_rating (1500) but LOW elo_dynamic (800)
        # e2: LOW elo_rating (1000) but HIGH elo_dynamic (1200)
        with store.transaction():
            store.update_role_elo(e1.id, "elo_dynamic", 800.0)
            store.update_role_elo(e2.id, "elo_dynamic", 1200.0)
        evicted = mgr.evict_weakest()
        assert evicted is not None
        assert evicted.id == e1.id  # evicted despite higher elo_rating

    def test_weakest_dynamic_elo_returns_elo_dynamic(self, store):
        config = DynamicConfig(slots=10, min_games_before_eviction=0, protection_matches=0)
        mgr = DynamicManager(store, config)
        e1 = _add_entry(store, epoch=1, role=Role.DYNAMIC, elo=1500.0)
        e2 = _add_entry(store, epoch=2, role=Role.DYNAMIC, elo=1000.0)
        with store.transaction():
            store.update_role_elo(e1.id, "elo_dynamic", 800.0)
            store.update_role_elo(e2.id, "elo_dynamic", 1200.0)
        assert mgr.weakest_dynamic_elo() == 800.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tier_managers.py::TestDynamicEloDynamic -v`
Expected: FAIL — `evicted.id == e2.id` (wrong entry) or `AttributeError` for `weakest_dynamic_elo`

- [ ] **Step 3: Modify DynamicManager**

In `keisei/training/tier_managers.py`:

Change line 345 in `evict_weakest()`:
```python
# Before:
        weakest = min(eligible, key=lambda e: e.elo_rating)
# After:
        weakest = min(eligible, key=lambda e: e.elo_dynamic)
```

Change the log message at line 348-350:
```python
# Before:
        logger.info(
            "Dynamic evict_weakest: retired id=%d (elo=%.1f)",
            weakest.id,
            weakest.elo_rating,
        )
# After:
        logger.info(
            "Dynamic evict_weakest: retired id=%d (elo_dynamic=%.1f)",
            weakest.id,
            weakest.elo_dynamic,
        )
```

Add `weakest_dynamic_elo()` method after `weakest_elo()` (after line 373):

```python
    def weakest_dynamic_elo(self) -> float | None:
        """Return the elo_dynamic of the weakest eligible entry, or None."""
        entries = self._store.list_by_role(Role.DYNAMIC)
        eligible = [
            e
            for e in entries
            if e.protection_remaining == 0
            and e.games_played >= self._config.min_games_before_eviction
        ]
        if not eligible:
            return None
        return min(e.elo_dynamic for e in eligible)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tier_managers.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/tier_managers.py tests/test_tier_managers.py
git commit -m "feat(dynamic): evict_weakest uses elo_dynamic instead of composite Elo"
```

---

## Task 5: FrontierPromoter Uses elo_frontier for Promotion

**Files:**
- Modify: `keisei/training/frontier_promoter.py`
- Modify: `tests/test_frontier_promoter.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_frontier_promoter.py`. Check existing imports — you may need to add `OpponentEntry` to the import list if not present:

```python
class TestEloFrontierPromotion:
    """should_promote uses elo_frontier, not elo_rating."""

    def test_promotes_on_high_elo_frontier_despite_low_elo_rating(self):
        config = _make_frontier_config(
            promotion_margin_elo=10.0,
            min_games_for_promotion=0,
            streak_epochs=0,
            max_lineage_overlap=10,
        )
        promoter = FrontierPromoter(config)
        # Frontier entries with LOW elo_frontier (900)
        frontier = [
            OpponentEntry(**{**_make_entry(1, elo=1500.0, role=Role.FRONTIER_STATIC).__dict__, "elo_frontier": 900.0}),
            OpponentEntry(**{**_make_entry(2, elo=1500.0, role=Role.FRONTIER_STATIC).__dict__, "elo_frontier": 900.0}),
        ]
        # Candidate: LOW elo_rating (800) but HIGH elo_frontier (950)
        candidate = OpponentEntry(**{**_make_entry(10, elo=800.0, games=100).__dict__, "elo_frontier": 950.0})
        promoter._topk_streaks[candidate.id] = 0
        result = promoter.should_promote(candidate, frontier, epoch=100)
        assert result is True  # 950 >= 900 + 10

    def test_rejects_low_elo_frontier_despite_high_elo_rating(self):
        config = _make_frontier_config(
            promotion_margin_elo=10.0,
            min_games_for_promotion=0,
            streak_epochs=0,
            max_lineage_overlap=10,
        )
        promoter = FrontierPromoter(config)
        frontier = [
            OpponentEntry(**{**_make_entry(1, elo=500.0, role=Role.FRONTIER_STATIC).__dict__, "elo_frontier": 1000.0}),
        ]
        # HIGH elo_rating (2000) but LOW elo_frontier (900)
        candidate = OpponentEntry(**{**_make_entry(10, elo=2000.0, games=100).__dict__, "elo_frontier": 900.0})
        promoter._topk_streaks[candidate.id] = 0
        result = promoter.should_promote(candidate, frontier, epoch=100)
        assert result is False  # 900 < 1000 + 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_frontier_promoter.py::TestEloFrontierPromotion -v`
Expected: FAIL — first test fails because `should_promote` uses `elo_rating`

- [ ] **Step 3: Modify FrontierPromoter.should_promote**

In `keisei/training/frontier_promoter.py`, change lines 94-95:

```python
        # Before:
        weakest_frontier_elo = min(e.elo_rating for e in frontier_entries)
        if candidate.elo_rating < weakest_frontier_elo + self.config.promotion_margin_elo:
        # After:
        weakest_frontier_elo = min(e.elo_frontier for e in frontier_entries)
        if candidate.elo_frontier < weakest_frontier_elo + self.config.promotion_margin_elo:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_frontier_promoter.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/frontier_promoter.py tests/test_frontier_promoter.py
git commit -m "feat(promoter): should_promote uses elo_frontier instead of composite Elo"
```

---

## Task 6: ConcurrentMatchPool

**Files:**
- Create: `keisei/training/concurrent_matches.py`
- Create: `tests/test_concurrent_matches.py`

### Sub-task 6a: Data Structures

- [ ] **Step 1: Write failing test for MatchResult**

Create `tests/test_concurrent_matches.py`:

```python
"""Tests for ConcurrentMatchPool — interleaved match execution."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from keisei.config import ConcurrencyConfig
from keisei.training.concurrent_matches import ConcurrentMatchPool, MatchResult
from keisei.training.opponent_store import EntryStatus, OpponentEntry, OpponentStore, Role


def _make_entry(
    id: int,
    role: Role = Role.DYNAMIC,
    elo: float = 1000.0,
) -> OpponentEntry:
    return OpponentEntry(
        id=id,
        display_name=f"e{id}",
        architecture="resnet",
        model_params={},
        checkpoint_path=f"/tmp/{id}.pt",
        elo_rating=elo,
        created_epoch=0,
        games_played=10,
        created_at="2026-01-01",
        flavour_facts=[],
        role=role,
        status=EntryStatus.ACTIVE,
    )


class TestMatchResult:
    def test_fields(self):
        a, b = _make_entry(1), _make_entry(2)
        r = MatchResult(entry_a=a, entry_b=b, a_wins=3, b_wins=1, draws=0, rollout=None)
        assert r.a_wins == 3
        assert r.b_wins == 1
        assert r.rollout is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_concurrent_matches.py::TestMatchResult -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create concurrent_matches.py with data structures**

Create `keisei/training/concurrent_matches.py`:

```python
"""ConcurrentMatchPool — interleaved match execution on a single GPU.

Manages parallel match execution by partitioning a VecEnv across multiple
concurrent pairings. Models stay resident to avoid load/unload overhead.
The VecEnv step is batched (one call for all environments), while inference
runs per-partition with each partition's own model pair.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from keisei.config import ConcurrencyConfig
from keisei.training.match_utils import release_models
from keisei.training.opponent_store import OpponentEntry, OpponentStore, Role

if TYPE_CHECKING:
    from keisei.training.dynamic_trainer import MatchRollout

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single completed match."""

    entry_a: OpponentEntry
    entry_b: OpponentEntry
    a_wins: int
    b_wins: int
    draws: int
    rollout: MatchRollout | None


@dataclass
class _MatchSlot:
    """Internal state for one concurrent match partition."""

    index: int
    env_start: int
    env_end: int
    entry_a: OpponentEntry | None = None
    entry_b: OpponentEntry | None = None
    model_a: torch.nn.Module | None = None
    model_b: torch.nn.Module | None = None
    a_wins: int = 0
    b_wins: int = 0
    draws: int = 0
    games_target: int = 0
    active: bool = False
    collect_rollout: bool = False
    _obs: list[torch.Tensor] = field(default_factory=list)
    _actions: list[torch.Tensor] = field(default_factory=list)
    _rewards: list[torch.Tensor] = field(default_factory=list)
    _dones: list[torch.Tensor] = field(default_factory=list)
    _masks: list[torch.Tensor] = field(default_factory=list)
    _perspective: list[torch.Tensor] = field(default_factory=list)

    @property
    def games_completed(self) -> int:
        return self.a_wins + self.b_wins + self.draws

    def reset_for_pairing(
        self,
        entry_a: OpponentEntry,
        entry_b: OpponentEntry,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        games_target: int,
        collect_rollout: bool = False,
    ) -> None:
        """Reset slot state for a new pairing."""
        self.entry_a = entry_a
        self.entry_b = entry_b
        self.model_a = model_a
        self.model_b = model_b
        self.a_wins = 0
        self.b_wins = 0
        self.draws = 0
        self.games_target = games_target
        self.active = True
        self.collect_rollout = collect_rollout
        self._obs.clear()
        self._actions.clear()
        self._rewards.clear()
        self._dones.clear()
        self._masks.clear()
        self._perspective.clear()

    def to_result(self) -> MatchResult:
        """Package slot state into a MatchResult."""
        rollout = None
        if self.collect_rollout and self._obs:
            from keisei.training.dynamic_trainer import MatchRollout

            rollout = MatchRollout(
                observations=torch.stack(self._obs),
                actions=torch.stack(self._actions),
                rewards=torch.stack(self._rewards),
                dones=torch.stack(self._dones),
                legal_masks=torch.stack(self._masks),
                perspective=torch.stack(self._perspective),
            )
        assert self.entry_a is not None and self.entry_b is not None
        return MatchResult(
            entry_a=self.entry_a,
            entry_b=self.entry_b,
            a_wins=self.a_wins,
            b_wins=self.b_wins,
            draws=self.draws,
            rollout=rollout,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_concurrent_matches.py::TestMatchResult -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/concurrent_matches.py tests/test_concurrent_matches.py
git commit -m "feat: add MatchResult and _MatchSlot data structures for ConcurrentMatchPool"
```

### Sub-task 6b: ConcurrentMatchPool.run_round

- [ ] **Step 6: Write MockVecEnv and failing tests for run_round**

Add to `tests/test_concurrent_matches.py`:

```python
from tests._helpers import TinyModel


class MockVecEnv:
    """Deterministic VecEnv mock for testing ConcurrentMatchPool.

    Terminates each env after ``terminate_after`` steps with reward +1.0 (A wins).
    After termination, auto-resets (ply counter resets to 0).
    """

    def __init__(self, num_envs: int, terminate_after: int = 3) -> None:
        self.num_envs = num_envs
        self.terminate_after = terminate_after
        self._ply = np.zeros(num_envs, dtype=int)

    def reset(self) -> SimpleNamespace:
        self._ply = np.zeros(self.num_envs, dtype=int)
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, 50, 9, 9).astype(np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
        )

    def step(self, actions: np.ndarray) -> SimpleNamespace:
        self._ply += 1
        terminated = self._ply >= self.terminate_after
        rewards = np.where(terminated, 1.0, 0.0).astype(np.float32)
        self._ply[terminated] = 0  # auto-reset
        return SimpleNamespace(
            observations=np.random.randn(self.num_envs, 50, 9, 9).astype(np.float32),
            legal_masks=np.ones((self.num_envs, 11259), dtype=bool),
            current_players=np.zeros(self.num_envs, dtype=np.uint8),
            rewards=rewards,
            terminated=terminated,
            truncated=np.zeros(self.num_envs, dtype=bool),
        )


def _make_mock_store(entries: list[OpponentEntry]) -> MagicMock:
    """Create a mock OpponentStore that returns TinyModel for any load."""
    store = MagicMock(spec=OpponentStore)
    store.load_opponent = MagicMock(side_effect=lambda entry, device="cpu": TinyModel())
    return store


class TestRunRound:
    def test_processes_all_pairings(self):
        config = ConcurrencyConfig(
            parallel_matches=2, envs_per_match=2, total_envs=4, max_resident_models=4,
        )
        pool = ConcurrentMatchPool(config)
        entries = [_make_entry(i) for i in range(1, 5)]
        pairings = [(entries[0], entries[1]), (entries[2], entries[3])]
        vecenv = MockVecEnv(num_envs=4, terminate_after=3)
        store = _make_mock_store(entries)
        results = pool.run_round(
            pairings, vecenv, store=store, device=torch.device("cpu"),
            games_per_match=4,
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, MatchResult)
            assert r.a_wins + r.b_wins + r.draws >= 4

    def test_more_pairings_than_parallel_slots(self):
        config = ConcurrencyConfig(
            parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2,
        )
        pool = ConcurrentMatchPool(config)
        entries = [_make_entry(i) for i in range(1, 7)]
        pairings = [
            (entries[0], entries[1]),
            (entries[2], entries[3]),
            (entries[4], entries[5]),
        ]
        vecenv = MockVecEnv(num_envs=2, terminate_after=2)
        store = _make_mock_store(entries)
        results = pool.run_round(
            pairings, vecenv, store=store, device=torch.device("cpu"),
            games_per_match=2,
        )
        assert len(results) == 3
        assert results[0].entry_a.id == 1
        assert results[1].entry_a.id == 3
        assert results[2].entry_a.id == 5

    def test_empty_pairings(self):
        config = ConcurrencyConfig(
            parallel_matches=2, envs_per_match=2, total_envs=4, max_resident_models=4,
        )
        pool = ConcurrentMatchPool(config)
        vecenv = MockVecEnv(num_envs=4)
        store = _make_mock_store([])
        results = pool.run_round(
            [], vecenv, store=store, device=torch.device("cpu"),
            games_per_match=4,
        )
        assert results == []

    def test_stop_event_interrupts(self):
        import threading

        config = ConcurrencyConfig(
            parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2,
        )
        pool = ConcurrentMatchPool(config)
        entries = [_make_entry(i) for i in range(1, 5)]
        pairings = [(entries[0], entries[1]), (entries[2], entries[3])]
        vecenv = MockVecEnv(num_envs=2, terminate_after=100)
        store = _make_mock_store(entries)
        stop = threading.Event()
        stop.set()  # immediate stop
        results = pool.run_round(
            pairings, vecenv, store=store, device=torch.device("cpu"),
            games_per_match=1000, stop_event=stop,
        )
        assert isinstance(results, list)

    def test_rollout_collection_for_trainable(self):
        config = ConcurrencyConfig(
            parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2,
        )
        pool = ConcurrentMatchPool(config)
        a = _make_entry(1, role=Role.DYNAMIC)
        b = _make_entry(2, role=Role.DYNAMIC)
        pairings = [(a, b)]
        vecenv = MockVecEnv(num_envs=2, terminate_after=3)
        store = _make_mock_store([a, b])
        results = pool.run_round(
            pairings, vecenv, store=store, device=torch.device("cpu"),
            games_per_match=2,
            trainable_fn=lambda ea, eb: True,
        )
        assert len(results) == 1
        assert results[0].rollout is not None
        assert results[0].rollout.observations.ndim == 4

    def test_no_rollout_when_trainable_fn_false(self):
        config = ConcurrencyConfig(
            parallel_matches=1, envs_per_match=2, total_envs=2, max_resident_models=2,
        )
        pool = ConcurrentMatchPool(config)
        a, b = _make_entry(1), _make_entry(2)
        pairings = [(a, b)]
        vecenv = MockVecEnv(num_envs=2, terminate_after=2)
        store = _make_mock_store([a, b])
        results = pool.run_round(
            pairings, vecenv, store=store, device=torch.device("cpu"),
            games_per_match=2,
            trainable_fn=lambda ea, eb: False,
        )
        assert len(results) == 1
        assert results[0].rollout is None
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `uv run pytest tests/test_concurrent_matches.py::TestRunRound -v`
Expected: FAIL with `ImportError` (ConcurrentMatchPool not defined)

- [ ] **Step 8: Implement ConcurrentMatchPool.run_round**

Add to `keisei/training/concurrent_matches.py`:

```python
class ConcurrentMatchPool:
    """Manages parallel match execution on a single GPU.

    Partitions a VecEnv across concurrent pairings. Each partition runs its
    own model pair. The VecEnv step is batched (one call for all envs),
    while inference runs per-partition.
    """

    def __init__(self, config: ConcurrencyConfig) -> None:
        self.config = config

    def run_round(
        self,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
        vecenv: Any,
        *,
        store: OpponentStore,
        device: torch.device,
        games_per_match: int,
        stop_event: Any | None = None,
        trainable_fn: Callable[[OpponentEntry, OpponentEntry], bool] | None = None,
    ) -> list[MatchResult]:
        """Execute pairings with up to parallel_matches concurrent.

        Args:
            pairings: Ordered list of (entry_a, entry_b). High-priority first.
            vecenv: VecEnv instance with total_envs environments.
            store: OpponentStore for loading models.
            device: Torch device for inference.
            games_per_match: Number of games per pairing.
            stop_event: Optional threading.Event for early termination.
            trainable_fn: Called per pairing to decide rollout collection.

        Returns:
            List of MatchResult in completion order.
        """
        if not pairings:
            return []

        results: list[MatchResult] = []
        remaining = collections.deque(pairings)
        cfg = self.config

        slots = [
            _MatchSlot(
                index=i,
                env_start=i * cfg.envs_per_match,
                env_end=(i + 1) * cfg.envs_per_match,
            )
            for i in range(cfg.parallel_matches)
        ]

        for slot in slots:
            if remaining:
                self._assign_pairing(
                    slot, remaining.popleft(), store, device,
                    games_per_match, trainable_fn,
                )

        reset = vecenv.reset()
        obs = torch.from_numpy(np.asarray(reset.observations)).to(device)
        legal_masks = torch.from_numpy(np.asarray(reset.legal_masks)).to(device)
        current_players = np.zeros(cfg.total_envs, dtype=np.uint8)

        max_steps = games_per_match * 512 * 3

        for _ in range(max_steps):
            if stop_event is not None and stop_event.is_set():
                break
            if not any(s.active for s in slots):
                break

            actions = torch.zeros(cfg.total_envs, dtype=torch.long, device=device)

            for slot in slots:
                if not slot.active:
                    for idx in range(slot.env_start, slot.env_end):
                        legal = legal_masks[idx].nonzero(as_tuple=True)[0]
                        actions[idx] = legal[0] if legal.numel() > 0 else 0
                    continue

                s, e = slot.env_start, slot.env_end
                slot_players = current_players[s:e]

                if slot.collect_rollout:
                    slot._obs.append(obs[s:e].cpu())
                    slot._masks.append(legal_masks[s:e].cpu())
                    slot._perspective.append(
                        torch.from_numpy(slot_players.copy()),
                    )

                a_local = np.where(slot_players == 0)[0]
                if len(a_local) > 0:
                    a_global = torch.tensor(
                        a_local + s, dtype=torch.long, device=device,
                    )
                    with torch.no_grad():
                        a_out = slot.model_a(obs[a_global])
                        a_logits = a_out.policy_logits.reshape(len(a_local), -1)
                        a_masked = a_logits.masked_fill(
                            ~legal_masks[a_global], float("-inf"),
                        )
                        a_probs = F.softmax(a_masked, dim=-1)
                        actions[a_global] = torch.distributions.Categorical(
                            a_probs,
                        ).sample()

                b_local = np.where(slot_players == 1)[0]
                if len(b_local) > 0:
                    b_global = torch.tensor(
                        b_local + s, dtype=torch.long, device=device,
                    )
                    with torch.no_grad():
                        b_out = slot.model_b(obs[b_global])
                        b_logits = b_out.policy_logits.reshape(len(b_local), -1)
                        b_masked = b_logits.masked_fill(
                            ~legal_masks[b_global], float("-inf"),
                        )
                        b_probs = F.softmax(b_masked, dim=-1)
                        actions[b_global] = torch.distributions.Categorical(
                            b_probs,
                        ).sample()

                if slot.collect_rollout:
                    slot._actions.append(actions[s:e].cpu())

            step_result = vecenv.step(actions.cpu().numpy())
            obs = torch.from_numpy(np.asarray(step_result.observations)).to(device)
            legal_masks = torch.from_numpy(
                np.asarray(step_result.legal_masks),
            ).to(device)
            current_players = np.asarray(
                step_result.current_players, dtype=np.uint8,
            )
            rewards = np.asarray(step_result.rewards)
            terminated = np.asarray(step_result.terminated)
            truncated = np.asarray(step_result.truncated)
            done = terminated | truncated

            for slot in slots:
                if not slot.active:
                    continue
                s, e = slot.env_start, slot.env_end
                slot_rewards = rewards[s:e]
                slot_done = done[s:e]

                if slot.collect_rollout:
                    slot._rewards.append(
                        torch.from_numpy(slot_rewards.copy().astype(np.float32)),
                    )
                    slot._dones.append(
                        torch.from_numpy(slot_done.copy().astype(np.float32)),
                    )

                slot.a_wins += int(((slot_rewards > 0) & slot_done).sum())
                slot.b_wins += int(((slot_rewards < 0) & slot_done).sum())
                slot.draws += int(((slot_rewards == 0) & slot_done).sum())

                if slot.games_completed >= slot.games_target:
                    results.append(slot.to_result())
                    release_models(
                        slot.model_a, slot.model_b,
                        device_type=device.type,
                    )
                    slot.model_a = None
                    slot.model_b = None
                    if remaining:
                        self._assign_pairing(
                            slot, remaining.popleft(), store, device,
                            games_per_match, trainable_fn,
                        )
                    else:
                        slot.active = False

        for slot in slots:
            if slot.active:
                results.append(slot.to_result())
                if slot.model_a is not None:
                    release_models(
                        slot.model_a, slot.model_b,
                        device_type=device.type,
                    )

        return results

    def _assign_pairing(
        self,
        slot: _MatchSlot,
        pairing: tuple[OpponentEntry, OpponentEntry],
        store: OpponentStore,
        device: torch.device,
        games_per_match: int,
        trainable_fn: Callable[[OpponentEntry, OpponentEntry], bool] | None,
    ) -> None:
        """Load models and configure a slot for a new pairing."""
        entry_a, entry_b = pairing
        model_a = store.load_opponent(entry_a, device=str(device))
        model_b = store.load_opponent(entry_b, device=str(device))
        collect = trainable_fn(entry_a, entry_b) if trainable_fn else False
        slot.reset_for_pairing(
            entry_a, entry_b, model_a, model_b,
            games_target=games_per_match,
            collect_rollout=collect,
        )
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `uv run pytest tests/test_concurrent_matches.py -v`
Expected: all PASS

- [ ] **Step 10: Commit**

```bash
git add keisei/training/concurrent_matches.py tests/test_concurrent_matches.py
git commit -m "feat: add ConcurrentMatchPool with partitioned VecEnv game loop"
```

---

## Task 7: Tournament Integration

**Files:**
- Modify: `keisei/training/tournament.py`
- Modify: `tests/test_tournament.py`

- [ ] **Step 1: Write failing test for concurrent tournament**

Add to `tests/test_tournament.py`. First check existing imports and add what's needed:

```python
from keisei.training.concurrent_matches import ConcurrentMatchPool
from keisei.config import ConcurrencyConfig
```

Then add test class:

```python
class TestConcurrentTournament:
    def test_tournament_accepts_concurrent_pool(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        scheduler = _make_scheduler()
        config = ConcurrencyConfig(
            parallel_matches=2, envs_per_match=2, total_envs=4, max_resident_models=4,
        )
        pool = ConcurrentMatchPool(config)
        t = _make_tournament(store, scheduler, concurrent_pool=pool)
        assert t.concurrent_pool is pool

    def test_tournament_without_pool_still_works(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        league_dir = tmp_path / "league"
        league_dir.mkdir()
        store = OpponentStore(db_path, str(league_dir))
        t = _make_tournament(store)
        assert t.concurrent_pool is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tournament.py::TestConcurrentTournament -v`
Expected: FAIL with `TypeError` (unexpected keyword argument `concurrent_pool`)

- [ ] **Step 3: Add concurrent_pool parameter to LeagueTournament.__init__**

In `keisei/training/tournament.py`:

Add import at top:
```python
from keisei.training.concurrent_matches import ConcurrentMatchPool
```

Add parameter to `__init__` after `dynamic_trainer`:
```python
        concurrent_pool: ConcurrentMatchPool | None = None,
```

Add in body:
```python
        self.concurrent_pool = concurrent_pool
```

- [ ] **Step 4: Add _run_concurrent_round method**

Add this method to `LeagueTournament`:

```python
    def _run_concurrent_round(
        self,
        vecenv: object,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
        epoch: int,
    ) -> None:
        """Run a round using the ConcurrentMatchPool."""
        assert self.concurrent_pool is not None
        results = self.concurrent_pool.run_round(
            pairings,
            vecenv,
            store=self.store,
            device=self.device,
            games_per_match=self.games_per_match,
            stop_event=self._stop_event,
            trainable_fn=self._is_trainable_match if self.dynamic_trainer else None,
        )
        for result in results:
            total = result.a_wins + result.b_wins + result.draws
            if total == 0:
                continue
            current_a = self.store.get_entry(result.entry_a.id)
            current_b = self.store.get_entry(result.entry_b.id)
            if current_a is None or current_b is None:
                continue
            result_score = (result.a_wins + 0.5 * result.draws) / total
            new_a_elo, new_b_elo = compute_elo_update(
                current_a.elo_rating, current_b.elo_rating,
                result=result_score, k=self.k_factor,
            )
            self.store.record_result(
                epoch=epoch,
                learner_id=result.entry_a.id,
                opponent_id=result.entry_b.id,
                wins=result.a_wins,
                losses=result.b_wins,
                draws=result.draws,
                elo_delta_a=round(new_a_elo - current_a.elo_rating, 1),
                elo_delta_b=round(new_b_elo - current_b.elo_rating, 1),
            )
            self.store.update_elo(result.entry_a.id, new_a_elo, epoch=epoch)
            self.store.update_elo(result.entry_b.id, new_b_elo, epoch=epoch)
            logger.info(
                "  %s vs %s — %dW %dL %dD",
                result.entry_a.display_name, result.entry_b.display_name,
                result.a_wins, result.b_wins, result.draws,
            )
            if self.dynamic_trainer and result.rollout is not None:
                for i, entry in enumerate([result.entry_a, result.entry_b]):
                    if entry.role == Role.DYNAMIC:
                        self.dynamic_trainer.record_match(
                            entry.id, result.rollout, side=i,
                        )
                        if (
                            self.dynamic_trainer.should_update(entry.id)
                            and not self.dynamic_trainer.is_rate_limited()
                        ):
                            self.dynamic_trainer.update(
                                entry, device=str(self.device),
                            )
```

- [ ] **Step 5: Modify _run_loop to dispatch to concurrent or sequential path**

In `_run_loop`, replace the sequential match loop (the `for entry_a, entry_b in pairings:` block, approximately lines 171-183) with:

```python
                if self.concurrent_pool is not None:
                    self._run_concurrent_round(vecenv, pairings, epoch)
                else:
                    for entry_a, entry_b in pairings:
                        if self._stop_event.is_set():
                            break
                        try:
                            self._run_one_match(vecenv, entry_a, entry_b, epoch=epoch)
                        except Exception:
                            logger.exception(
                                "Match failed: %s vs %s",
                                entry_a.display_name, entry_b.display_name,
                            )
                            continue

                        self._stop_event.wait(self.pause_seconds)
```

Also update VecEnv creation to use `total_envs` when concurrent pool is active. Change the VecEnv construction (around line 142):

```python
        num_envs = (
            self.concurrent_pool.config.total_envs
            if self.concurrent_pool is not None
            else self.num_envs
        )
        vecenv = VecEnv(
            num_envs=num_envs,
            max_ply=self.max_ply,
            observation_mode="katago",
            action_mode="spatial",
        )
```

- [ ] **Step 6: Run tournament tests**

Run: `uv run pytest tests/test_tournament.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add keisei/training/tournament.py tests/test_tournament.py
git commit -m "feat(tournament): integrate ConcurrentMatchPool for parallel match execution"
```

---

## Task 8: Wire PriorityScorer and ConcurrentMatchPool into Training Pipeline

**Files:**
- Modify: `keisei/training/tiered_pool.py` and/or `keisei/training/katago_loop.py`

This task requires reading the actual wiring code to find where `MatchScheduler` and `LeagueTournament` are constructed. The implementer must:

- [ ] **Step 1: Find MatchScheduler and LeagueTournament construction sites**

Read `keisei/training/tiered_pool.py` and `keisei/training/katago_loop.py` to find where these are instantiated. Search for `MatchScheduler(` and `LeagueTournament(`.

- [ ] **Step 2: Wire PriorityScorer into MatchScheduler**

At the MatchScheduler construction site, create a PriorityScorer and pass it:

```python
from keisei.training.priority_scorer import PriorityScorer

priority_scorer = PriorityScorer(league_config.priority)
scheduler = MatchScheduler(league_config.scheduler, priority_scorer=priority_scorer)
```

- [ ] **Step 3: Wire ConcurrentMatchPool into LeagueTournament**

At the LeagueTournament construction site, create a ConcurrentMatchPool and pass it:

```python
from keisei.training.concurrent_matches import ConcurrentMatchPool

concurrent_pool = ConcurrentMatchPool(league_config.concurrency)
```

Pass `concurrent_pool=concurrent_pool` to the `LeagueTournament(...)` constructor.

- [ ] **Step 4: Add scorer state updates to tournament**

In `keisei/training/tournament.py`, update scorer state after matches.

In `_run_concurrent_round`, after the results loop:
```python
        if self.scheduler._priority_scorer is not None:
            for result in results:
                total = result.a_wins + result.b_wins + result.draws
                for _ in range(total):
                    self.scheduler._priority_scorer.record_result(
                        result.entry_a.id, result.entry_b.id,
                    )
                self.scheduler._priority_scorer.record_round_result(
                    result.entry_a.id, result.entry_b.id,
                )
            self.scheduler._priority_scorer.advance_round()
```

In the sequential path (`_run_loop`), after each match completes successfully:
```python
                    if self.scheduler._priority_scorer is not None:
                        self.scheduler._priority_scorer.record_round_result(
                            entry_a.id, entry_b.id,
                        )
```

And after the sequential `for` loop completes:
```python
                if self.scheduler._priority_scorer is not None:
                    self.scheduler._priority_scorer.advance_round()
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add keisei/training/tiered_pool.py keisei/training/katago_loop.py keisei/training/tournament.py
git commit -m "feat: wire PriorityScorer and ConcurrentMatchPool into training pipeline"
```

---

## Task 9: Final Validation

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: all PASS

- [ ] **Step 2: Verify imports are clean**

Run: `uv run python -c "from keisei.training.priority_scorer import PriorityScorer; from keisei.training.concurrent_matches import ConcurrentMatchPool, MatchResult; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: Phase 4 final validation and cleanup"
```
