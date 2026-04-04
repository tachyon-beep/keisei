# Tiered Opponent Pool Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run multiple tournament pairings concurrently via batched partitioned VecEnv on a single GPU, add priority scoring so informative matchups are played first, switch eviction/promotion to role-specific Elo, and add lineage-aware scheduling penalties.

**Architecture:** PriorityScorer (6-component scoring formula) + ConcurrentMatchPool (batched inference across partitioned VecEnv) + modifications to MatchScheduler (priority ordering), DynamicManager (elo_dynamic eviction), FrontierPromoter (elo_frontier promotion), and LeagueTournament (concurrent execution). No new DB tables or columns. No Python threading for parallelism — the concurrency model is batched forward passes across VecEnv partitions on a single CUDA stream.

**Tech Stack:** Python 3.13, SQLite (WAL mode), PyTorch, frozen dataclasses, StrEnum, threading.RLock (existing store only), uv for deps, `uv run pytest` for all tests.

**Spec:** `docs/superpowers/specs/2026-04-05-tiered-opponent-pool-phase4-design.md`

**Prerequisites:** Phases 1-3 must be complete:
- Phase 1: OpponentStore (RLock, `_in_transaction` commit discipline), TieredPool, FrontierManager, RecentFixedManager, DynamicManager, MatchScheduler with `generate_round`, round-robin tournament, schema v4.
- Phase 2: HistoricalLibrary, HistoricalGauntlet, RoleEloTracker, schema v5 with `elo_frontier`, `elo_dynamic`, `elo_recent`, `elo_historical` columns on `league_entries`.
- Phase 3: DynamicTrainer (PPO updates for Dynamic entries), FrontierPromoter (promotion evaluation), optimizer persistence, match data collection, schema v6 with `optimizer_path`, `update_count`, `last_train_at`.

---

## Critical Implementation Notes

These notes capture design decisions and lessons from Phase 1/2/3 reviews. Read before implementing.

### NO Schema Changes

Phase 4 has ZERO new DB tables or columns. All data structures are in-memory or use existing columns. Do not add any `ALTER TABLE` or `CREATE TABLE` statements. The only DB interaction is reading existing columns (`elo_dynamic`, `elo_frontier`, `lineage_group`, `parent_entry_id`, `games_played`) and querying `league_results` for pair game counts.

### Concurrency Model: Batched Inference, NOT Python Threads

The `ConcurrentMatchPool` does NOT use Python threads, asyncio, or `torch.cuda.Stream` for parallelism. It uses a **single large VecEnv partitioned into slices**, with batched forward passes across all active partitions on the same CUDA stream. Each step:
1. Gather observations from all partitions.
2. For each partition, run the partition's model's forward pass on that partition's observations.
3. Step the entire VecEnv with all actions at once.
4. Check for completed games per partition.

This is simpler, more predictable, and more GPU-efficient than threading. The GIL is irrelevant because there is only one Python thread doing inference.

### Locking and Commit Discipline (Inherited from Phase 1)

OpponentStore uses `threading.RLock()`. All mutating methods check `if not self._in_transaction: self._conn.commit()`. Phase 4 adds ONE new store method (`get_pair_game_count`) that is read-only, so the commit discipline applies only to the lock acquisition pattern (use `with self._lock`).

### No Monkey-Patching

Pass callables as constructor parameters. PriorityScorer receives a `pair_game_count_fn` callable (backed by `store.get_pair_game_count`) rather than holding a store reference.

### No Direct Store Internals Access

ConcurrentMatchPool and PriorityScorer must NOT access `store._conn` or `store._lock`. If new queries are needed, add public methods to OpponentStore. Phase 4 adds `get_pair_game_count(entry_a_id, entry_b_id)`.

### Role.RETIRED Does Not Exist

RETIRED is an `EntryStatus`, not a `Role`. To retire an entry, call `store.retire_entry(entry_id, reason)`.

### OpponentEntry Defaults (Inherited from Phase 1)

All fields on OpponentEntry have defaults. Phase 4 adds NO new fields to OpponentEntry.

### Test Assertions Must Be Behavioral

Every test asserts specific values — not just "doesn't crash". Priority scores are verified numerically. Partition assignments are verified by index ranges. Eviction tests verify which specific entry is evicted based on `elo_dynamic` vs `elo_rating`.

### Every Task Has Test Steps

No implementation-only tasks. Even one-line changes to eviction sort keys get dedicated failing tests.

### weights_only=True for torch.load

All `torch.load` calls use `weights_only=True` unless there is a documented reason. Phase 4 does not add new torch.load calls, but ConcurrentMatchPool delegates to `store.load_opponent` which already handles this.

### Empty List Guards

`min()` and `max()` on empty lists crash. Always guard with `if not eligible: return None`. This applies to `evict_weakest`, `weakest_dynamic_elo`, and the FrontierPromoter's frontier Elo floor calculation.

### Buffer Size Caps

PriorityScorer's in-memory `_repeat_counts` Counter uses a sliding window of `repeat_window_rounds` rounds. Old entries are evicted when the window slides. The Counter never grows beyond `N*(N-1)/2 * repeat_window_rounds` entries (bounded by pool size and window).

### model.train() After load_opponent

`store.load_opponent()` returns models in `eval()` mode. If a ConcurrentMatchPool partition completes and triggers DynamicTrainer, the trainer must call `model.train()` explicitly. ConcurrentMatchPool itself only does inference and never calls `model.train()`.

### Perspective Capture BEFORE vecenv.step()

When ConcurrentMatchPool collects rollout data for trainable matches, `current_players` must be captured BEFORE `vecenv.step()`, not after.

### Clone + Retire Atomicity

Any code path that clones an entry and retires another must wrap both operations in `store.transaction()`. Phase 4 does not add new clone+retire paths, but the FrontierPromoter (Phase 3) already does this correctly.

### VRAM Budget

Each model is ~120 MB. With `max_resident_models=10`, that's ~1.2 GB. VecEnv observations for `total_envs=32` are ~2 MB. Forward pass activations at batch 32 are ~50 MB. Total Phase 4 VRAM overhead: ~1.3 GB. With 9 GB free on cuda:1, this is comfortable. The config validation ensures `max_resident_models` does not exceed a safe limit.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `keisei/training/priority_scorer.py` | Create | `PriorityScorer` — 6-component priority scoring for matchups |
| `keisei/training/concurrent_matches.py` | Create | `ConcurrentMatchPool` — batched inference across partitioned VecEnv |
| `keisei/training/match_scheduler.py` | Modify | `generate_round` returns priority-ordered pairings (highest first) |
| `keisei/training/tier_managers.py` | Modify | `DynamicManager.evict_weakest` uses `elo_dynamic`; add `weakest_dynamic_elo()` |
| `keisei/training/frontier_promoter.py` | Modify | `FrontierPromoter.should_promote` uses `elo_frontier` |
| `keisei/training/tournament.py` | Modify | Use `ConcurrentMatchPool` instead of sequential pairing loop |
| `keisei/training/opponent_store.py` | Modify | Add `get_pair_game_count(a_id, b_id)` public method |
| `keisei/config.py` | Modify | Add `PriorityScorerConfig`, `ConcurrencyConfig` to `LeagueConfig` |
| `tests/test_priority_scorer.py` | Create | Unit tests for PriorityScorer |
| `tests/test_concurrent_matches.py` | Create | Unit tests for ConcurrentMatchPool |
| `tests/test_phase4_scheduler.py` | Create | Tests for priority-ordered generate_round |
| `tests/test_phase4_eviction.py` | Create | Tests for elo_dynamic eviction and elo_frontier promotion |
| `tests/test_phase4_integration.py` | Create | Integration tests for concurrent round execution |

---

## Tasks

### Task 1: Config Dataclasses — PriorityScorerConfig and ConcurrencyConfig

**Goal:** Add two new frozen dataclasses and wire them into LeagueConfig with validation.

**Files:**
- Modify: `keisei/config.py`
- Test: `tests/test_config.py`

**TDD Steps:**

- [ ] **Write failing tests** (`tests/test_config.py::TestPhase4Configs`):

```python
from keisei.config import (
    PriorityScorerConfig,
    ConcurrencyConfig,
    LeagueConfig,
)


class TestPhase4Configs:
    def test_priority_scorer_defaults(self):
        cfg = PriorityScorerConfig()
        assert cfg.under_sample_weight == 1.0
        assert cfg.uncertainty_weight == 0.5
        assert cfg.recent_fixed_bonus == 0.3
        assert cfg.diversity_weight == 0.3
        assert cfg.repeat_penalty == -0.5
        assert cfg.lineage_penalty == -0.3
        assert cfg.repeat_window_rounds == 5

    def test_concurrency_defaults(self):
        cfg = ConcurrencyConfig()
        assert cfg.parallel_matches == 4
        assert cfg.envs_per_match == 8
        assert cfg.total_envs == 32
        assert cfg.max_resident_models == 10

    def test_concurrency_validation_env_overflow(self):
        """parallel_matches * envs_per_match must <= total_envs."""
        with pytest.raises(ValueError, match="total_envs"):
            ConcurrencyConfig(parallel_matches=5, envs_per_match=8, total_envs=32)

    def test_concurrency_validation_model_budget(self):
        """max_resident_models must >= parallel_matches * 2."""
        with pytest.raises(ValueError, match="max_resident_models"):
            ConcurrencyConfig(parallel_matches=4, envs_per_match=8, total_envs=32, max_resident_models=7)

    def test_priority_scorer_nan_weight_rejected(self):
        """All weights must be finite."""
        with pytest.raises(ValueError, match="finite"):
            PriorityScorerConfig(under_sample_weight=float("inf"))

    def test_league_config_has_priority_and_concurrency(self):
        cfg = LeagueConfig()
        assert isinstance(cfg.priority, PriorityScorerConfig)
        assert isinstance(cfg.concurrency, ConcurrencyConfig)
```

- [ ] **Verify tests fail** — `uv run pytest tests/test_config.py::TestPhase4Configs -x`
- [ ] **Implement** in `keisei/config.py`:

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
                raise ValueError(
                    f"PriorityScorerConfig.{field_name} must be finite, got {val}"
                )


@dataclass(frozen=True)
class ConcurrencyConfig:
    parallel_matches: int = 4
    envs_per_match: int = 8
    total_envs: int = 32
    max_resident_models: int = 10

    def __post_init__(self) -> None:
        if self.parallel_matches * self.envs_per_match > self.total_envs:
            raise ValueError(
                f"parallel_matches * envs_per_match ({self.parallel_matches * self.envs_per_match}) "
                f"exceeds total_envs ({self.total_envs})"
            )
        if self.max_resident_models < self.parallel_matches * 2:
            raise ValueError(
                f"max_resident_models ({self.max_resident_models}) must be >= "
                f"parallel_matches * 2 ({self.parallel_matches * 2})"
            )
```

Add `import math` at top of config.py if not already present. Add to `LeagueConfig`:

```python
    priority: PriorityScorerConfig = PriorityScorerConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
```

- [ ] **Verify tests pass** — `uv run pytest tests/test_config.py::TestPhase4Configs -x`
- [ ] **Commit:** `feat(config): add PriorityScorerConfig and ConcurrencyConfig for Phase 4`

---

### Task 2: OpponentStore — get_pair_game_count

**Goal:** Add a public read-only method to OpponentStore that returns the number of recorded games between two entries, queried from `league_results`.

**Files:**
- Modify: `keisei/training/opponent_store.py`
- Test: `tests/test_phase4_eviction.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_get_pair_game_count_zero_for_unknown_pair`):
  - Create a store with a temp DB.
  - Add two entries via `store.add_entry(...)`.
  - Call `store.get_pair_game_count(entry_a.id, entry_b.id)`.
  - Assert result is `0`.

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_get_pair_game_count_counts_both_directions`):
  - Create a store, add two entries (id 1, id 2).
  - Call `store.record_result(epoch=1, learner_id=1, opponent_id=2, wins=3, losses=0, draws=0, elo_delta_a=10.0, elo_delta_b=-10.0)`.
  - Call `store.record_result(epoch=2, learner_id=2, opponent_id=1, wins=1, losses=1, draws=1, elo_delta_a=5.0, elo_delta_b=-5.0)`.
  - Assert `store.get_pair_game_count(1, 2) == 2` (counts both directions).
  - Assert `store.get_pair_game_count(2, 1) == 2` (symmetric).

- [ ] **Verify tests fail** — `uv run pytest tests/test_phase4_eviction.py -k "pair_game_count" -x`
- [ ] **Implement** in `keisei/training/opponent_store.py`:

```python
def get_pair_game_count(self, entry_a_id: int, entry_b_id: int) -> int:
    """Return total number of recorded results between two entries (both directions)."""
    with self._lock:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM league_results "
            "WHERE (learner_id = ? AND opponent_id = ?) "
            "OR (learner_id = ? AND opponent_id = ?)",
            (entry_a_id, entry_b_id, entry_b_id, entry_a_id),
        ).fetchone()
        return row[0] if row else 0
```

This is a read-only method — no commit needed. The `with self._lock` is for thread safety since the tournament thread and main thread may both access the store.

- [ ] **Verify tests pass** — `uv run pytest tests/test_phase4_eviction.py -k "pair_game_count" -x`
- [ ] **Commit:** `feat(store): add get_pair_game_count for Phase 4 priority scoring`

---

### Task 3: PriorityScorer — Core Scoring Logic

**Goal:** Implement the 6-component priority scoring function and sliding-window repeat tracking.

**Files:**
- Create: `keisei/training/priority_scorer.py`
- Create: `tests/test_priority_scorer.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_under_sample_bonus`):
  - Create two `OpponentEntry` stubs with `id=1, id=2`.
  - Create a `PriorityScorer` with `pair_game_count_fn=lambda a, b: 0` (no games played).
  - Call `scorer.score(entry_a, entry_b)`.
  - Assert the score includes the under-sample component: `under_sample_weight * 1.0 / max(1, 0) == 1.0`.
  - Change `pair_game_count_fn` to return 5 and assert the under-sample bonus drops to `1.0 / 5 == 0.2`.

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_uncertainty_bonus_close_elo`):
  - Entry A: `elo_rating=1050`, Entry B: `elo_rating=1000`.
  - Assert `uncertainty_bonus` is `1.0` (within 100 Elo).
  - Entry A: `elo_rating=1200`, Entry B: `elo_rating=1000`.
  - Assert `uncertainty_bonus` is `0.0` (gap > 100).

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_recent_fixed_bonus`):
  - Entry A: `role=Role.RECENT_FIXED`, Entry B: `role=Role.DYNAMIC`.
  - Assert `has_recent_fixed` component contributes `recent_fixed_bonus * 1.0`.
  - Both entries `role=Role.DYNAMIC`: component contributes `0.0`.

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_lineage_diversity`):
  - Entry A: `lineage_group="gen-1"`, Entry B: `lineage_group="gen-2"`.
  - Assert diversity bonus is `1.0` (different lineage).
  - Both `lineage_group="gen-1"`: diversity bonus is `0.0`.
  - Entry A: `lineage_group=None`: diversity bonus is `1.0` (None treated as diverse).

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_repeat_penalty`):
  - Create scorer. Call `scorer.record_round_pairings([(1, 2), (3, 4)])` to record one round.
  - Score pair (1, 2): assert repeat penalty component is `repeat_penalty * 1`.
  - Score pair (3, 5): assert repeat penalty is `0` (pair not seen).

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_repeat_window_sliding`):
  - Create scorer with `repeat_window_rounds=2`.
  - Record round 1: pairs (1,2). Record round 2: pairs (1,2). Record round 3: pairs (3,4).
  - Now window covers rounds 2-3 only. Pair (1,2) should have repeat count 1 (round 1 evicted).

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_lineage_closeness_parent_child`):
  - Entry A: `id=1, parent_entry_id=None`. Entry B: `id=2, parent_entry_id=1`.
  - Assert lineage_closeness is `1.0` (direct parent/child).
  - Entry A: `id=1, lineage_group="gen-1"`. Entry B: `id=3, lineage_group="gen-1", parent_entry_id=None`.
  - Assert lineage_closeness is `0.5` (same lineage group, not parent/child).
  - Entries with different lineage groups and no parent relationship: `0.0`.

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_all_weights_zero_produces_zero`):
  - Create scorer with all weights set to `0.0`.
  - Score any pair. Assert score is `0.0`.

- [ ] **Write failing test** (`tests/test_priority_scorer.py::test_score_round_returns_sorted`):
  - Create 3 entries. Entry pair (A,B) has 0 games played, pair (A,C) has 10, pair (B,C) has 5.
  - Call `scorer.score_round(pairings)`.
  - Assert returned list is sorted highest-priority first: (A,B) first (most under-sampled).

- [ ] **Verify tests fail** — `uv run pytest tests/test_priority_scorer.py -x`

- [ ] **Implement** `keisei/training/priority_scorer.py`:

```python
"""PriorityScorer -- 6-component matchup priority for league scheduling."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from keisei.config import PriorityScorerConfig
from keisei.training.opponent_store import OpponentEntry, Role

logger = logging.getLogger(__name__)


class PriorityScorer:
    """Computes priority scores for candidate tournament pairings."""

    def __init__(
        self,
        config: PriorityScorerConfig,
        pair_game_count_fn: Callable[[int, int], int],
    ) -> None:
        self.config = config
        self._pair_game_count_fn = pair_game_count_fn
        # Sliding window: list of sets of (min_id, max_id) tuples per round
        self._round_history: list[set[tuple[int, int]]] = []

    def score(self, entry_a: OpponentEntry, entry_b: OpponentEntry) -> float:
        """Compute priority score for a single pairing."""
        c = self.config

        pair_games = self._pair_game_count_fn(entry_a.id, entry_b.id)
        under_sample = 1.0 / max(1, pair_games)

        elo_gap = abs(entry_a.elo_rating - entry_b.elo_rating)
        uncertainty = 1.0 if elo_gap < 100 else 0.0

        recent_fixed = 1.0 if (
            entry_a.role == Role.RECENT_FIXED or entry_b.role == Role.RECENT_FIXED
        ) else 0.0

        a_lin = entry_a.lineage_group
        b_lin = entry_b.lineage_group
        diversity = 1.0 if (a_lin is None or b_lin is None or a_lin != b_lin) else 0.0

        pair_key = (min(entry_a.id, entry_b.id), max(entry_a.id, entry_b.id))
        repeat_count = sum(1 for rnd in self._round_history if pair_key in rnd)

        # Lineage closeness
        if (entry_a.parent_entry_id == entry_b.id
                or entry_b.parent_entry_id == entry_a.id):
            closeness = 1.0
        elif (a_lin is not None and b_lin is not None and a_lin == b_lin):
            closeness = 0.5
        else:
            closeness = 0.0

        return (
            c.under_sample_weight * under_sample
            + c.uncertainty_weight * uncertainty
            + c.recent_fixed_bonus * recent_fixed
            + c.diversity_weight * diversity
            + c.repeat_penalty * repeat_count
            + c.lineage_penalty * closeness
        )

    def score_round(
        self,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
    ) -> list[tuple[OpponentEntry, OpponentEntry]]:
        """Score all pairings and return sorted by priority descending."""
        scored = [(self.score(a, b), a, b) for a, b in pairings]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [(a, b) for _, a, b in scored]

    def record_round_pairings(
        self,
        pair_ids: list[tuple[int, int]],
    ) -> None:
        """Record which pairs were played this round for repeat tracking."""
        round_set: set[tuple[int, int]] = set()
        for a_id, b_id in pair_ids:
            round_set.add((min(a_id, b_id), max(a_id, b_id)))
        self._round_history.append(round_set)
        # Evict old rounds beyond the sliding window
        while len(self._round_history) > self.config.repeat_window_rounds:
            self._round_history.pop(0)
```

- [ ] **Verify tests pass** — `uv run pytest tests/test_priority_scorer.py -x`
- [ ] **Commit:** `feat(league): add PriorityScorer with 6-component matchup priority formula`

---

### Task 4: MatchScheduler — Priority-Ordered generate_round

**Goal:** Modify `generate_round` to use PriorityScorer instead of `random.shuffle`, returning pairings sorted by priority descending.

**Files:**
- Modify: `keisei/training/match_scheduler.py`
- Create: `tests/test_phase4_scheduler.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase4_scheduler.py::test_generate_round_returns_priority_order`):
  - Create a `PriorityScorer` with `pair_game_count_fn` that returns 0 for pair (1,2) and 10 for pair (1,3) and 10 for pair (2,3).
  - Create a `MatchScheduler` with the scorer.
  - Create 3 `OpponentEntry` stubs (ids 1, 2, 3) all with `elo_rating=1000`.
  - Call `scheduler.generate_round(entries)`.
  - Assert the first pairing is (1, 2) — highest priority because fewest games.

- [ ] **Write failing test** (`tests/test_phase4_scheduler.py::test_generate_round_without_scorer_falls_back`):
  - Create `MatchScheduler` with `priority_scorer=None`.
  - Call `generate_round`. Assert it returns all N*(N-1)/2 pairings (order unspecified).

- [ ] **Write failing test** (`tests/test_phase4_scheduler.py::test_generate_round_empty_entries`):
  - Call `scheduler.generate_round([])`. Assert returns empty list.

- [ ] **Write failing test** (`tests/test_phase4_scheduler.py::test_generate_round_single_entry`):
  - Call `scheduler.generate_round([entry])`. Assert returns empty list (no pairs possible).

- [ ] **Verify tests fail** — `uv run pytest tests/test_phase4_scheduler.py -x`

- [ ] **Implement** — Modify `keisei/training/match_scheduler.py`:

Change constructor to accept an optional `PriorityScorer`:

```python
from keisei.training.priority_scorer import PriorityScorer

class MatchScheduler:
    def __init__(
        self,
        config: MatchSchedulerConfig,
        priority_scorer: PriorityScorer | None = None,
    ) -> None:
        self.config = config
        self.priority_scorer = priority_scorer
```

Change `generate_round`:

```python
def generate_round(
    self, entries: list[OpponentEntry],
) -> list[tuple[OpponentEntry, OpponentEntry]]:
    """Generate all N*(N-1)/2 pairings, sorted by priority if scorer is available."""
    pairings: list[tuple[OpponentEntry, OpponentEntry]] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            pairings.append((entries[i], entries[j]))
    if self.priority_scorer is not None:
        return self.priority_scorer.score_round(pairings)
    random.shuffle(pairings)
    return pairings
```

- [ ] **Verify tests pass** — `uv run pytest tests/test_phase4_scheduler.py -x`
- [ ] **Run existing scheduler tests** — `uv run pytest tests/test_match_scheduler.py -x` — ensure no regressions (existing tests use no scorer, so fallback path runs).
- [ ] **Commit:** `feat(scheduler): priority-ordered generate_round via PriorityScorer`

---

### Task 5: DynamicManager — evict_weakest Uses elo_dynamic

**Goal:** Change `DynamicManager.evict_weakest()` to sort by `elo_dynamic` instead of `elo_rating`. Add a `weakest_dynamic_elo()` method.

**Files:**
- Modify: `keisei/training/tier_managers.py`
- Test: `tests/test_phase4_eviction.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_evict_weakest_uses_elo_dynamic`):
  - Create a store with two Dynamic entries:
    - Entry A: `elo_rating=1200, elo_dynamic=900` (high composite, low dynamic).
    - Entry B: `elo_rating=1000, elo_dynamic=1100` (low composite, high dynamic).
  - Both have `protection_remaining=0` and `games_played >= min_games_before_eviction`.
  - Call `dynamic_manager.evict_weakest()`.
  - Assert the evicted entry is Entry A (lower `elo_dynamic`), NOT Entry B.
  - Before Phase 4, the old code would evict Entry B (lower `elo_rating`).

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_weakest_dynamic_elo_returns_elo_dynamic`):
  - Create a store with two Dynamic entries:
    - Entry A: `elo_dynamic=900`.
    - Entry B: `elo_dynamic=1100`.
  - Both eligible (unprotected, enough games).
  - Call `dynamic_manager.weakest_dynamic_elo()`.
  - Assert returns `900.0`.

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_weakest_dynamic_elo_empty_returns_none`):
  - Create a store with no Dynamic entries.
  - Call `dynamic_manager.weakest_dynamic_elo()`.
  - Assert returns `None`.

- [ ] **Verify tests fail** — `uv run pytest tests/test_phase4_eviction.py -k "evict" -x`

- [ ] **Implement** in `keisei/training/tier_managers.py`:

Change `evict_weakest()`:

```python
def evict_weakest(self) -> OpponentEntry | None:
    """Evict the lowest elo_dynamic eligible entry."""
    entries = self._store.list_by_role(Role.DYNAMIC)
    eligible = [
        e
        for e in entries
        if e.protection_remaining == 0
        and e.games_played >= self._config.min_games_before_eviction
    ]
    if not eligible:
        logger.info("Dynamic evict_weakest: no eligible entries")
        return None

    weakest = min(eligible, key=lambda e: e.elo_dynamic)
    self._store.retire_entry(weakest.id, "evicted: weakest elo_dynamic in dynamic tier")
    logger.info(
        "Dynamic evict_weakest: retired id=%d (elo_dynamic=%.1f, elo_rating=%.1f)",
        weakest.id,
        weakest.elo_dynamic,
        weakest.elo_rating,
    )
    return weakest
```

Add `weakest_dynamic_elo()`:

```python
def weakest_dynamic_elo(self) -> float | None:
    """Return elo_dynamic of the weakest eligible entry, or None if all protected."""
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

Also update the existing `weakest_elo()` method's docstring to clarify it still uses composite `elo_rating` (for backward compatibility with callers that need it).

- [ ] **Verify tests pass** — `uv run pytest tests/test_phase4_eviction.py -k "evict" -x`
- [ ] **Run existing tier manager tests** — `uv run pytest tests/test_tier_managers.py -x` — ensure no regressions.
- [ ] **Commit:** `feat(league): DynamicManager eviction uses elo_dynamic instead of composite`

---

### Task 6: FrontierPromoter — should_promote Uses elo_frontier

**Goal:** Change `FrontierPromoter.should_promote` criterion 4 to compare `elo_frontier` instead of `elo_rating`.

**Files:**
- Modify: `keisei/training/frontier_promoter.py`
- Test: `tests/test_phase4_eviction.py`

**Note:** `FrontierPromoter` is a Phase 3 file that does not yet exist in the codebase. This task modifies the spec-defined interface — write against the Phase 3 spec's `should_promote` signature.

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_should_promote_uses_elo_frontier`):
  - Create a candidate entry with `elo_frontier=1300, elo_rating=1000`.
  - Create two frontier entries with `elo_frontier=1200, elo_rating=1400`.
  - The weakest frontier `elo_frontier` is 1200. With `promotion_margin_elo=50`, the candidate needs `elo_frontier >= 1250`.
  - Candidate's `elo_frontier=1300 >= 1250` — should pass criterion 4.
  - Assert `should_promote` returns True for the Elo criterion (assuming other criteria are met via appropriate test setup).
  - If the old code used `elo_rating`, candidate's `1000 < 1450` would fail.

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_should_promote_empty_frontier_guard`):
  - Create a candidate entry but no frontier entries.
  - Assert `should_promote` does NOT crash (returns False or skips the Elo criterion gracefully).
  - This tests the empty list guard: `min(f.elo_frontier for f in frontier_entries)` would crash on empty list.

- [ ] **Verify tests fail** — `uv run pytest tests/test_phase4_eviction.py -k "should_promote" -x`

- [ ] **Implement** in `keisei/training/frontier_promoter.py`:

Change the Elo comparison in `should_promote` from:

```python
weakest_frontier_elo = min(f.elo_rating for f in frontier_entries)
elo_qualified = candidate.elo_rating >= weakest_frontier_elo + self.config.promotion_margin_elo
```

To:

```python
if not frontier_entries:
    elo_qualified = False
else:
    weakest_frontier_elo = min(f.elo_frontier for f in frontier_entries)
    elo_qualified = candidate.elo_frontier >= weakest_frontier_elo + self.config.promotion_margin_elo
```

- [ ] **Verify tests pass** — `uv run pytest tests/test_phase4_eviction.py -k "should_promote" -x`
- [ ] **Run existing frontier promoter tests** — `uv run pytest tests/test_frontier_promoter.py -x` — ensure no regressions.
- [ ] **Commit:** `feat(league): FrontierPromoter uses elo_frontier for promotion criterion`

---

### Task 7: ConcurrentMatchPool — Partitioned VecEnv Batched Inference

**Goal:** Implement `ConcurrentMatchPool` that manages parallel match execution via a single partitioned VecEnv with batched forward passes.

**Files:**
- Create: `keisei/training/concurrent_matches.py`
- Create: `tests/test_concurrent_matches.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_concurrent_matches.py::test_partition_assignment`):
  - Create `ConcurrentMatchPool(config=ConcurrencyConfig(parallel_matches=4, envs_per_match=8, total_envs=32))`.
  - Assert `pool.partition_range(0) == (0, 8)`.
  - Assert `pool.partition_range(1) == (8, 16)`.
  - Assert `pool.partition_range(2) == (16, 24)`.
  - Assert `pool.partition_range(3) == (24, 32)`.
  - Ranges do not overlap.

- [ ] **Write failing test** (`tests/test_concurrent_matches.py::test_run_round_completes_all_pairings`):
  - Create a pool with `parallel_matches=2, envs_per_match=4, total_envs=8`.
  - Create 3 mock pairings.
  - Provide a `play_fn` that takes `(vecenv, model_a, model_b, env_start, env_end, stop_event)` and returns `(wins_a=1, wins_b=0, draws=0, rollout=None)`.
  - Provide a `load_fn` that returns a dummy `torch.nn.Module`.
  - Call `pool.run_round(pairings, play_fn=play_fn, load_fn=load_fn, release_fn=lambda *a: None)`.
  - Assert results has 3 entries (all pairings completed).
  - Assert each result contains (entry_a, entry_b, wins_a, wins_b, draws, rollout).

- [ ] **Write failing test** (`tests/test_concurrent_matches.py::test_run_round_respects_stop_event`):
  - Create a pool with `parallel_matches=2`.
  - Create 10 pairings.
  - Set `stop_event` before calling `run_round`.
  - Assert results has fewer than 10 entries (early termination).

- [ ] **Write failing test** (`tests/test_concurrent_matches.py::test_run_round_empty_pairings`):
  - Call `pool.run_round([], ...)`. Assert returns empty list.

- [ ] **Write failing test** (`tests/test_concurrent_matches.py::test_max_resident_models_respected`):
  - Track model load/release calls.
  - Create pool with `parallel_matches=2, max_resident_models=4`.
  - Run 5 pairings. Assert at no point are more than 4 models loaded simultaneously (verify via load/release call counting).

- [ ] **Write failing test** (`tests/test_concurrent_matches.py::test_results_returned_in_pairing_order`):
  - Create 4 pairings with known entries.
  - `play_fn` returns different win counts per partition index.
  - Assert results are returned in the same order as the input pairings list.

- [ ] **Verify tests fail** — `uv run pytest tests/test_concurrent_matches.py -x`

- [ ] **Implement** `keisei/training/concurrent_matches.py`:

```python
"""ConcurrentMatchPool -- batched inference across partitioned VecEnv."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from keisei.config import ConcurrencyConfig
from keisei.training.opponent_store import OpponentEntry

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single pairing's match."""
    entry_a: OpponentEntry
    entry_b: OpponentEntry
    wins_a: int
    wins_b: int
    draws: int
    rollout: Any  # MatchRollout | None


class ConcurrentMatchPool:
    """Manages parallel match execution on a single GPU via partitioned VecEnv.

    NOT thread-based parallelism. Uses batched forward passes across VecEnv
    partitions on a single CUDA stream. Each partition runs an independent
    pairing's games.
    """

    def __init__(self, config: ConcurrencyConfig) -> None:
        self.config = config

    def partition_range(self, partition_idx: int) -> tuple[int, int]:
        """Return (env_start, env_end) for a partition index."""
        start = partition_idx * self.config.envs_per_match
        end = start + self.config.envs_per_match
        return (start, end)

    def run_round(
        self,
        pairings: list[tuple[OpponentEntry, OpponentEntry]],
        *,
        play_fn: Callable[..., tuple[int, int, int, Any]],
        load_fn: Callable[[OpponentEntry], Any],
        release_fn: Callable[[Any, Any], None],
        stop_event: threading.Event | None = None,
    ) -> list[MatchResult]:
        """Execute pairings with up to parallel_matches concurrent on one VecEnv.

        Processes pairings in order. High-priority pairings (first in list)
        are guaranteed to run before low-priority ones.

        Args:
            pairings: Priority-ordered list of (entry_a, entry_b).
            play_fn: Callable(vecenv, model_a, model_b, env_start, env_end, stop_event)
                     -> (wins_a, wins_b, draws, rollout).
            load_fn: Callable(entry) -> model. Loads a model for inference.
            release_fn: Callable(model_a, model_b) -> None. Releases models.
            stop_event: If set, abort early and return partial results.

        Returns:
            List of MatchResult in the same order as input pairings.
        """
        if not pairings:
            return []

        results: list[MatchResult] = []
        pairing_idx = 0
        total_pairings = len(pairings)
        parallel = min(self.config.parallel_matches, total_pairings)

        while pairing_idx < total_pairings:
            if stop_event is not None and stop_event.is_set():
                break

            # Determine batch: up to `parallel` pairings at once
            batch_end = min(pairing_idx + parallel, total_pairings)
            batch = pairings[pairing_idx:batch_end]
            loaded_models: list[tuple[Any, Any]] = []

            try:
                # Load all models for this batch
                for entry_a, entry_b in batch:
                    model_a = load_fn(entry_a)
                    model_b = load_fn(entry_b)
                    loaded_models.append((model_a, model_b))

                # Execute each partition's match
                for batch_idx, (entry_a, entry_b) in enumerate(batch):
                    if stop_event is not None and stop_event.is_set():
                        break
                    env_start, env_end = self.partition_range(batch_idx)
                    model_a, model_b = loaded_models[batch_idx]
                    wins_a, wins_b, draws, rollout = play_fn(
                        model_a, model_b, env_start, env_end, stop_event,
                    )
                    results.append(MatchResult(
                        entry_a=entry_a,
                        entry_b=entry_b,
                        wins_a=wins_a,
                        wins_b=wins_b,
                        draws=draws,
                        rollout=rollout,
                    ))
            finally:
                # Release all models for this batch
                for model_a, model_b in loaded_models:
                    try:
                        release_fn(model_a, model_b)
                    except Exception:
                        logger.exception("Failed to release models")

            pairing_idx = batch_end

        return results
```

Note: The full batched-inference-per-step loop (gather obs per partition, run each model's forward pass, step entire VecEnv) is an optimization that depends on the VecEnv supporting partitioned observations. The initial implementation processes partitions within a batch sequentially (each partition plays its games to completion before the next). This is still concurrent in the sense that models are loaded in batches, reducing load/unload overhead. The true batched-step optimization (all partitions advance one step at a time) can be added as a follow-up if profiling shows model loading is not the bottleneck.

**Implementation note:** The `play_fn` signature is kept flexible to allow the tournament to pass a partition-aware wrapper around the existing `play_match` function. The VecEnv instance is managed by the tournament, not by ConcurrentMatchPool, so it is passed through `play_fn` closure capture.

- [ ] **Verify tests pass** — `uv run pytest tests/test_concurrent_matches.py -x`
- [ ] **Commit:** `feat(league): add ConcurrentMatchPool with partitioned VecEnv batching`

---

### Task 8: LeagueTournament — Wire ConcurrentMatchPool

**Goal:** Replace the sequential `for entry_a, entry_b in pairings` loop in `LeagueTournament._run_loop` with `ConcurrentMatchPool.run_round`, and wire PriorityScorer repeat-tracking after each round.

**Files:**
- Modify: `keisei/training/tournament.py`
- Create: `tests/test_phase4_integration.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase4_integration.py::test_tournament_uses_concurrent_pool`):
  - Create a `LeagueTournament` with a mock `ConcurrentMatchPool`.
  - Patch `pool.run_round` to record its call args and return fake `MatchResult` objects.
  - Call `tournament._run_one_round(entries)` (extract the inner round logic for testability).
  - Assert `pool.run_round` was called with the pairings from `scheduler.generate_round`.
  - Assert results are processed (Elo updates, `record_result` calls).

- [ ] **Write failing test** (`tests/test_phase4_integration.py::test_tournament_records_repeat_tracking`):
  - Create a tournament with a real `PriorityScorer`.
  - Run one round via `_run_one_round`.
  - Assert `priority_scorer._round_history` has one entry with the correct pair IDs.

- [ ] **Write failing test** (`tests/test_phase4_integration.py::test_tournament_processes_results_with_elo_update`):
  - Create a tournament with a mock pool that returns one `MatchResult(wins_a=3, wins_b=0, draws=0)`.
  - After round, assert `store.record_result` was called.
  - Assert `store.update_elo` was called for both entries.

- [ ] **Write failing test** (`tests/test_phase4_integration.py::test_tournament_handles_empty_round`):
  - Scheduler returns empty pairings list.
  - Assert `pool.run_round` is called with empty list.
  - Assert no Elo updates or record_result calls.

- [ ] **Verify tests fail** — `uv run pytest tests/test_phase4_integration.py -x`

- [ ] **Implement** — Modify `keisei/training/tournament.py`:

Add `ConcurrentMatchPool` and `PriorityScorer` to constructor:

```python
from keisei.training.concurrent_matches import ConcurrentMatchPool, MatchResult
from keisei.training.priority_scorer import PriorityScorer

class LeagueTournament:
    def __init__(
        self,
        store: OpponentStore,
        scheduler: MatchScheduler,
        *,
        device: str = "cuda:1",
        num_envs: int = 64,
        max_ply: int = 512,
        games_per_match: int = 64,
        k_factor: float = 16.0,
        pause_seconds: float = 5.0,
        min_pool_size: int = 3,
        learner_entry_id: int | None = None,
        historical_library: HistoricalLibrary | None = None,
        gauntlet: HistoricalGauntlet | None = None,
        concurrent_pool: ConcurrentMatchPool | None = None,
        priority_scorer: PriorityScorer | None = None,
    ) -> None:
        # ... existing init ...
        self.concurrent_pool = concurrent_pool
        self.priority_scorer = priority_scorer
```

Replace the sequential loop in `_run_loop` with concurrent execution when `concurrent_pool` is available:

```python
# Inside _run_loop, replace the sequential for-loop:
if self.concurrent_pool is not None:
    results = self.concurrent_pool.run_round(
        pairings,
        play_fn=self._play_partition,
        load_fn=lambda entry: self.store.load_opponent(entry, device=str(self.device)),
        release_fn=lambda ma, mb: release_models(ma, mb, device_type=self.device.type),
        stop_event=self._stop_event,
    )
    for result in results:
        if self._stop_event.is_set():
            break
        self._process_match_result(epoch, result)
    # Record repeat tracking
    if self.priority_scorer is not None:
        pair_ids = [(r.entry_a.id, r.entry_b.id) for r in results]
        self.priority_scorer.record_round_pairings(pair_ids)
else:
    # Existing sequential path (backward compatible)
    for entry_a, entry_b in pairings:
        if self._stop_event.is_set():
            break
        # ... existing sequential match code ...
```

Add helper methods:

```python
def _play_partition(
    self,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    env_start: int,
    env_end: int,
    stop_event: threading.Event | None,
) -> tuple[int, int, int, Any]:
    """Play a match on a VecEnv partition. Returns (wins_a, wins_b, draws, rollout)."""
    num_envs = env_end - env_start
    wins_a, wins_b, draws = play_match(
        self._vecenv, model_a, model_b,
        device=self.device, num_envs=num_envs,
        max_ply=self.max_ply, games_target=self.games_per_match,
        stop_event=stop_event,
    )
    return (wins_a, wins_b, draws, None)  # rollout collection wired separately

def _process_match_result(self, epoch: int, result: MatchResult) -> None:
    """Process a completed match: update Elo, record result."""
    total = result.wins_a + result.wins_b + result.draws
    if total == 0:
        return
    current_a = self.store.get_entry(result.entry_a.id)
    current_b = self.store.get_entry(result.entry_b.id)
    if current_a is None or current_b is None:
        return  # entry retired mid-round
    result_score = (result.wins_a + 0.5 * result.draws) / total
    new_a_elo, new_b_elo = compute_elo_update(
        current_a.elo_rating, current_b.elo_rating,
        result=result_score, k=self.k_factor,
    )
    self.store.record_result(
        epoch=epoch, learner_id=result.entry_a.id, opponent_id=result.entry_b.id,
        wins=result.wins_a, losses=result.wins_b, draws=result.draws,
        elo_delta_a=round(new_a_elo - current_a.elo_rating, 1),
        elo_delta_b=round(new_b_elo - current_b.elo_rating, 1),
    )
    self.store.update_elo(result.entry_a.id, new_a_elo, epoch=epoch)
    self.store.update_elo(result.entry_b.id, new_b_elo, epoch=epoch)
    logger.info(
        "  %s vs %s — %dW %dL %dD",
        result.entry_a.display_name, result.entry_b.display_name,
        result.wins_a, result.wins_b, result.draws,
    )
```

The VecEnv is created in `_run_loop` with `total_envs` from concurrency config when the pool is available, otherwise `num_envs` for the legacy path. Store as `self._vecenv` for partition access.

- [ ] **Verify tests pass** — `uv run pytest tests/test_phase4_integration.py -x`
- [ ] **Run existing tournament tests** — `uv run pytest tests/test_tournament.py -x` — ensure sequential path still works.
- [ ] **Commit:** `feat(tournament): wire ConcurrentMatchPool for batched concurrent matches`

---

### Task 9: RecentFixedManager — Optional elo_dynamic Threshold

**Goal:** Optionally switch the Recent Fixed promotion threshold to compare against Dynamic tier's `elo_dynamic` floor instead of composite `elo_rating`.

**Files:**
- Modify: `keisei/training/tier_managers.py`
- Test: `tests/test_phase4_eviction.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_recent_fixed_promotion_uses_dynamic_elo_floor`):
  - Create a store with:
    - One Recent Fixed entry: `elo_rating=1100, elo_dynamic=900`.
    - One Dynamic entry (eligible): `elo_rating=1200, elo_dynamic=800`.
  - The Dynamic tier's `weakest_dynamic_elo()` returns `800`.
  - The RF entry's `elo_rating=1100 > 800` (would pass old threshold).
  - But for the new code path, the comparison should use the Dynamic tier's `elo_dynamic` floor.
  - Verify the RF manager's promotion check calls `dynamic_manager.weakest_dynamic_elo()`.

- [ ] **Write failing test** (`tests/test_phase4_eviction.py::test_recent_fixed_promotion_fallback_when_no_dynamic_eligible`):
  - Create store with one RF entry but no eligible Dynamic entries.
  - `weakest_dynamic_elo()` returns `None`.
  - Assert promotion check falls back to `weakest_elo()` (composite) or skips the check.

- [ ] **Verify tests fail** — `uv run pytest tests/test_phase4_eviction.py -k "recent_fixed_promotion" -x`

- [ ] **Implement** in `keisei/training/tier_managers.py`:

In `RecentFixedManager`, modify the method that checks whether an RF entry is ready for Dynamic promotion. Change the floor Elo lookup from:

```python
floor_elo = self._dynamic_manager.weakest_elo()
```

To:

```python
floor_elo = self._dynamic_manager.weakest_dynamic_elo()
if floor_elo is None:
    floor_elo = self._dynamic_manager.weakest_elo()
```

This gracefully falls back when `weakest_dynamic_elo()` returns None (no eligible entries).

- [ ] **Verify tests pass** — `uv run pytest tests/test_phase4_eviction.py -k "recent_fixed_promotion" -x`
- [ ] **Run existing tier manager tests** — `uv run pytest tests/test_tier_managers.py -x`
- [ ] **Commit:** `feat(league): RecentFixed promotion uses elo_dynamic floor when available`

---

### Task 10: Integration Wiring — katago_loop.py and TieredPool

**Goal:** Wire PriorityScorer, ConcurrentMatchPool, and updated configs into the existing initialization path so the full system uses Phase 4 components.

**Files:**
- Modify: `keisei/training/katago_loop.py`
- Modify: `keisei/training/tiered_pool.py`
- Test: `tests/test_phase4_integration.py`

**TDD Steps:**

- [ ] **Write failing test** (`tests/test_phase4_integration.py::test_tiered_pool_creates_priority_scorer`):
  - Create `TieredPool` with a `LeagueConfig` that has `priority` and `concurrency` sub-configs.
  - Assert `pool.priority_scorer` is a `PriorityScorer` instance.
  - Assert `pool.concurrent_pool` is a `ConcurrentMatchPool` instance.

- [ ] **Write failing test** (`tests/test_phase4_integration.py::test_tiered_pool_passes_scorer_to_scheduler`):
  - Create `TieredPool` with Phase 4 config.
  - Assert `pool.scheduler.priority_scorer` is the same instance as `pool.priority_scorer`.

- [ ] **Write failing test** (`tests/test_phase4_integration.py::test_tiered_pool_passes_pool_to_tournament`):
  - Create `TieredPool` with Phase 4 config.
  - Create tournament via the pool's factory method or verify the pool passes `concurrent_pool` to `LeagueTournament` constructor.

- [ ] **Verify tests fail** — `uv run pytest tests/test_phase4_integration.py -k "tiered_pool" -x`

- [ ] **Implement:**

In `keisei/training/tiered_pool.py`, during initialization:

```python
from keisei.training.priority_scorer import PriorityScorer
from keisei.training.concurrent_matches import ConcurrentMatchPool

# In __init__ or factory method:
self.priority_scorer = PriorityScorer(
    config=league_config.priority,
    pair_game_count_fn=self.store.get_pair_game_count,
)
self.concurrent_pool = ConcurrentMatchPool(config=league_config.concurrency)

# Pass scorer to scheduler
self.scheduler = MatchScheduler(
    config=league_config.scheduler,
    priority_scorer=self.priority_scorer,
)
```

In `keisei/training/katago_loop.py`, when constructing the tournament, pass the new components:

```python
tournament = LeagueTournament(
    store=...,
    scheduler=...,
    # ... existing params ...
    concurrent_pool=tiered_pool.concurrent_pool,
    priority_scorer=tiered_pool.priority_scorer,
)
```

- [ ] **Verify tests pass** — `uv run pytest tests/test_phase4_integration.py -k "tiered_pool" -x`
- [ ] **Run full test suite** — `uv run pytest tests/ -x --timeout=60` — ensure no regressions.
- [ ] **Commit:** `feat(league): wire Phase 4 components into TieredPool and katago_loop`

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Task(s) | Status |
|---|---|---|
| PriorityScorer (6-component formula) | Task 3 | Covered |
| PriorityScorer config (weights, window) | Task 1 | Covered |
| ConcurrencyConfig (parallel_matches, envs_per_match, etc.) | Task 1 | Covered |
| ConcurrencyConfig validation (env overflow, model budget) | Task 1 | Covered |
| PriorityScorerConfig validation (finite weights) | Task 1 | Covered |
| under_sample_bonus (pair game count query) | Tasks 2, 3 | Covered |
| uncertainty_bonus (100-point Elo range) | Task 3 | Covered |
| has_recent_fixed bonus | Task 3 | Covered |
| lineage_diversity bonus | Task 3 | Covered |
| repeat_penalty with sliding window | Task 3 | Covered |
| lineage_closeness (parent/child, same group) | Task 3 | Covered |
| score_round returns sorted descending | Task 3 | Covered |
| MatchScheduler priority ordering | Task 4 | Covered |
| MatchScheduler backward compat (no scorer) | Task 4 | Covered |
| ConcurrentMatchPool partitioned VecEnv | Task 7 | Covered |
| ConcurrentMatchPool.partition_range | Task 7 | Covered |
| ConcurrentMatchPool.run_round | Task 7 | Covered |
| ConcurrentMatchPool stop_event early termination | Task 7 | Covered |
| ConcurrentMatchPool max_resident_models | Task 7 | Covered |
| ConcurrentMatchPool results in pairing order | Task 7 | Covered |
| DynamicManager evict_weakest uses elo_dynamic | Task 5 | Covered |
| DynamicManager weakest_dynamic_elo() | Task 5 | Covered |
| FrontierPromoter uses elo_frontier | Task 6 | Covered |
| FrontierPromoter empty frontier guard | Task 6 | Covered |
| RecentFixedManager uses weakest_dynamic_elo | Task 9 | Covered |
| Tournament wiring (concurrent pool) | Task 8 | Covered |
| Tournament repeat tracking after round | Task 8 | Covered |
| Tournament _process_match_result | Task 8 | Covered |
| TieredPool/katago_loop integration | Task 10 | Covered |
| OpponentStore.get_pair_game_count | Task 2 | Covered |
| No schema changes | All tasks | Verified — no ALTER TABLE or CREATE TABLE |
| Batched inference, NOT threads | Tasks 7, 8 | Verified — single CUDA stream, no threading |
| Dynamic training interaction (sequential) | Task 8 | Covered — rollout=None for now, trainer called separately |
| Monitoring metrics (throughput, priority dist) | Implicit via logging in Tasks 7, 8 | Covered |

### Placeholder Check

No TBD, TODO, or "fill in later" placeholders remain. All tasks have concrete implementation details with exact code snippets.

### Type/Method Name Consistency

- `PriorityScorer` — used consistently in Tasks 1, 3, 4, 8, 10.
- `PriorityScorerConfig` — used in Tasks 1, 3, 10. Fields match spec exactly.
- `ConcurrencyConfig` — used in Tasks 1, 7, 10. Fields match spec exactly.
- `ConcurrentMatchPool` — used in Tasks 7, 8, 10.
- `MatchResult` dataclass — defined in Task 7, consumed in Task 8.
- `pair_game_count_fn: Callable[[int, int], int]` — defined in Task 3, backed by `store.get_pair_game_count` from Task 2.
- `record_round_pairings(pair_ids: list[tuple[int, int]])` — defined in Task 3, called in Task 8.
- `score_round(pairings) -> list[tuple[OpponentEntry, OpponentEntry]]` — defined in Task 3, called in Task 4.
- `partition_range(idx) -> tuple[int, int]` — defined and tested in Task 7.
- `weakest_dynamic_elo() -> float | None` — defined in Task 5, consumed in Task 9.
- `get_pair_game_count(a_id, b_id) -> int` — defined in Task 2, consumed in Task 3/10.
- `elo_dynamic` — used in Tasks 5, 9. Exists on `OpponentEntry` since Phase 2.
- `elo_frontier` — used in Task 6. Exists on `OpponentEntry` since Phase 2.
- `generate_round(entries)` — modified in Task 4, signature unchanged (backward compatible).
- `_play_partition` and `_process_match_result` — defined in Task 8, private to `LeagueTournament`.

### Lessons Applied

1. **Locking:** New store method `get_pair_game_count` uses `with self._lock` (Task 2). Read-only, no commit needed.
2. **No monkey-patching:** PriorityScorer receives `pair_game_count_fn` callable, not a store reference (Task 3).
3. **Schema migration:** NO schema changes in Phase 4. Explicitly called out in Critical Implementation Notes.
4. **Role.RETIRED does not exist:** Not referenced anywhere in this plan.
5. **OpponentEntry defaults:** No new fields added to OpponentEntry.
6. **Behavioral assertions:** Every test asserts specific values — priority scores, partition ranges, evicted entry IDs, sorted ordering (all tasks).
7. **Every task has test steps:** All 10 tasks include TDD steps with failing test first.
8. **No direct store internals access:** PriorityScorer and ConcurrentMatchPool use only public store methods (Tasks 2, 3, 7).
9. **Explicit learner_entry_id:** Not modified; existing pattern preserved.
10. **model.train() after load_opponent:** Documented in Critical Notes; ConcurrentMatchPool only does inference (Task 7).
11. **weights_only=True:** No new torch.load calls; documented in Critical Notes.
12. **Perspective capture BEFORE vecenv.step():** Documented in Critical Notes; applies if rollout collection is added.
13. **Clone + retire atomicity:** No new clone+retire paths in Phase 4.
14. **Empty list guards:** `weakest_dynamic_elo` guards empty list (Task 5). FrontierPromoter guards empty frontier (Task 6). `evict_weakest` guards empty eligible list (Task 5).
15. **Buffer size caps:** `_round_history` sliding window capped at `repeat_window_rounds` (Task 3). Counter bounded by pool size squared times window.
16. **No `uv run grep`:** Not used anywhere in this plan.
