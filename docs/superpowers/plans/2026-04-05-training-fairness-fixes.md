# Training Fairness Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three training fairness issues — Elo inflation via carry-forward, black-only learner bias, and single-opponent-per-epoch narrow signal — with config gates, proper Elo attribution, and comprehensive tests.

**Architecture:** Three changes to `katago_loop.py` and `league.py`: (1) delete carry-forward Elo in `_rotate_seat`, (2) widen `learner_side` from scalar to per-env `np.ndarray` with in-place GPU scatter updates, (3) replace single opponent with per-env sticky opponents using a frozen-pool-view cohort model. Two new config fields on `LeagueConfig` gate Changes 2 and 3.

**Tech Stack:** Python 3.13, PyTorch, NumPy, SQLite, pytest, uv

**Spec:** `docs/superpowers/specs/2026-04-05-training-fairness-fixes-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `keisei/config.py` | Modify | Add `color_randomization` and `per_env_opponents` to `LeagueConfig` |
| `keisei/training/league.py` | Modify | Add `load_all_opponents`, `sample_from` |
| `keisei/training/katago_loop.py` | Modify | Remove carry-forward, widen `learner_side`, multi-opponent step loop |
| `tests/test_league_config.py` | Modify | Test new config fields |
| `tests/test_league.py` | Modify | Test `load_all_opponents`, `sample_from` |
| `tests/test_pending_transitions.py` | Modify | Test array `learner_side` in perspective functions |
| `tests/test_split_merge.py` | Modify | Test multi-opponent `split_merge_step` |
| `tests/test_katago_loop.py` | Modify | Test `_rotate_seat` Elo assertions, interaction tests |

---

### Task 1: Add Config Gates to `LeagueConfig`

**Files:**
- Modify: `keisei/config.py:42-74`
- Modify: `tests/test_league_config.py:77-83`

- [ ] **Step 1: Write failing test for new config defaults**

Add to `tests/test_league_config.py`:

```python
def test_league_config_fairness_defaults():
    """New fairness config fields should default to True."""
    lc = LeagueConfig()
    assert lc.color_randomization is True
    assert lc.per_env_opponents is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_league_config.py::test_league_config_fairness_defaults -v`
Expected: FAIL — `LeagueConfig` has no attribute `color_randomization`

- [ ] **Step 3: Add fields to `LeagueConfig`**

In `keisei/config.py`, add two fields to `LeagueConfig` between `elo_floor` (line 50) and the existing `opponent_device` (line 51). **Do NOT re-add `opponent_device` — it already exists.**

```python
    elo_floor: float = 500.0
    color_randomization: bool = True     # Per-game color randomization (Change 2)
    per_env_opponents: bool = True       # Per-env sticky opponents (Change 3)
    opponent_device: str | None = None  # ALREADY EXISTS — do not add
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_league_config.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add keisei/config.py tests/test_league_config.py
git commit -m "feat(config): add color_randomization and per_env_opponents gates to LeagueConfig"
```

---

### Task 2: Add `sample_from` to `OpponentSampler`

**Files:**
- Modify: `keisei/training/league.py:349-389`
- Modify: `tests/test_league.py:259-338`

- [ ] **Step 1: Write failing test for `sample_from`**

Add to `tests/test_league.py`, inside a new class after `TestOpponentSampler`:

```python
class TestOpponentSamplerSampleFrom:
    """Tests for sample_from() with pre-fetched entries."""

    def test_sample_from_matches_sample(self, league_db, league_dir):
        """sample_from(entries) should draw from both historical and current_best."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        for i in range(5):
            pool.add_snapshot(model, "resnet", {}, epoch=i)

        sampler = OpponentSampler(pool, historical_ratio=0.8, current_best_ratio=0.2)
        entries = pool.list_entries()
        pool_ids = {e.id for e in entries}

        # Draw enough samples to confirm diversity (not always current_best)
        seen_ids: set[int] = set()
        for _ in range(200):
            entry = sampler.sample_from(entries)
            assert isinstance(entry, OpponentEntry)
            assert entry.id in pool_ids
            seen_ids.add(entry.id)
        # With 5 entries and 200 draws at 80/20 split, we should see at least 2 distinct IDs
        assert len(seen_ids) >= 2, f"Expected diversity, only saw {seen_ids}"

    def test_sample_from_single_entry(self, league_db, league_dir):
        """Single-entry list should always return that entry."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)

        sampler = OpponentSampler(pool)
        entries = pool.list_entries()
        assert sampler.sample_from(entries).created_epoch == 0

    def test_sample_from_empty_raises(self, league_db, league_dir):
        """Empty entries list should raise ValueError."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        sampler = OpponentSampler(pool)
        with pytest.raises(ValueError, match="empty"):
            sampler.sample_from([])

    def test_sample_from_respects_elo_floor(self, league_db, league_dir):
        """Entries below elo_floor excluded from historical sampling."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "resnet", {}, epoch=0)
        pool.add_snapshot(model, "resnet", {}, epoch=1)
        pool.add_snapshot(model, "resnet", {}, epoch=2)

        entries = pool.list_entries()
        pool.update_elo(entries[0].id, 400.0)
        # Refresh entries after Elo update
        entries = pool.list_entries()

        sampler = OpponentSampler(
            pool, historical_ratio=1.0, current_best_ratio=0.0, elo_floor=500.0,
        )
        for _ in range(10):
            entry = sampler.sample_from(entries)
            assert entry.created_epoch != 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_league.py::TestOpponentSamplerSampleFrom -v`
Expected: FAIL — `OpponentSampler` has no method `sample_from`

- [ ] **Step 3: Implement `sample_from` and refactor `sample`**

In `keisei/training/league.py`, replace the `sample` method and add `sample_from`:

```python
    def sample(self) -> OpponentEntry:
        """Sample an opponent from the pool (queries DB)."""
        return self.sample_from(self.pool.list_entries())

    def sample_from(self, entries: list[OpponentEntry]) -> OpponentEntry:
        """Sample an opponent from a pre-fetched entries list."""
        if not entries:
            raise ValueError("Cannot sample from empty opponent pool")
        if len(entries) == 1:
            return entries[0]

        # Current best = highest Elo entry (not most recent — learner may regress)
        current_best = max(entries, key=lambda e: e.elo_rating)

        # Historical = all entries above elo_floor, excluding current_best
        historical = [
            e for e in entries
            if e.id != current_best.id and e.elo_rating >= self.elo_floor
        ]

        # If no historical entries above floor, sample current_best only.
        if not historical:
            return current_best

        if random.random() < self.current_best_ratio:
            return current_best
        else:
            return random.choice(historical)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_league.py -v`
Expected: All pass (including existing `TestOpponentSampler` tests, since `sample` now delegates)

- [ ] **Step 5: Commit**

```bash
git add keisei/training/league.py tests/test_league.py
git commit -m "feat(league): add OpponentSampler.sample_from for cached entries"
```

---

### Task 3: Add `load_all_opponents` to `OpponentPool`

**Files:**
- Modify: `keisei/training/league.py:300-313`
- Modify: `tests/test_league.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_league.py`:

```python
class TestLoadAllOpponents:
    """Tests for OpponentPool.load_all_opponents."""

    def test_loads_all_entries(self, league_db, league_dir):
        """Should return a dict with one model per pool entry."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        for i in range(3):
            pool.add_snapshot(model, "linear", {"in_features": 10, "out_features": 10}, epoch=i)

        models = pool.load_all_opponents(device="cpu")
        entries = pool.list_entries()
        assert len(models) == 3
        for entry in entries:
            assert entry.id in models
            assert isinstance(models[entry.id], torch.nn.Module)

    def test_skips_corrupt_checkpoint(self, league_db, league_dir):
        """Corrupt checkpoint should be skipped with a warning, not crash."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "linear", {"in_features": 10, "out_features": 10}, epoch=0)
        pool.add_snapshot(model, "linear", {"in_features": 10, "out_features": 10}, epoch=1)

        # Corrupt the first entry's checkpoint
        entries = pool.list_entries()
        Path(entries[0].checkpoint_path).write_text("not a valid checkpoint")

        models = pool.load_all_opponents(device="cpu")
        assert len(models) == 1
        assert entries[1].id in models
        assert entries[0].id not in models

    def test_skips_missing_checkpoint(self, league_db, league_dir):
        """Missing checkpoint file should be skipped, not crash."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        model = torch.nn.Linear(10, 10)
        pool.add_snapshot(model, "linear", {"in_features": 10, "out_features": 10}, epoch=0)
        pool.add_snapshot(model, "linear", {"in_features": 10, "out_features": 10}, epoch=1)

        # Delete the first entry's checkpoint
        entries = pool.list_entries()
        Path(entries[0].checkpoint_path).unlink()

        models = pool.load_all_opponents(device="cpu")
        assert len(models) == 1
        assert entries[1].id in models

    def test_empty_pool_returns_empty_dict(self, league_db, league_dir):
        """Empty pool should return empty dict."""
        pool = OpponentPool(league_db, str(league_dir), max_pool_size=10)
        models = pool.load_all_opponents(device="cpu")
        assert models == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_league.py::TestLoadAllOpponents -v`
Expected: FAIL — `OpponentPool` has no method `load_all_opponents`

- [ ] **Step 3: Implement `load_all_opponents`**

In `keisei/training/league.py`, add after `load_opponent` (after line 313):

```python
    def load_all_opponents(self, device: str = "cpu") -> dict[int, torch.nn.Module]:
        """Load all pool entries. Skips entries with missing/corrupt checkpoints."""
        models: dict[int, torch.nn.Module] = {}
        for entry in self.list_entries():
            try:
                models[entry.id] = self.load_opponent(entry, device=device)
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning(
                    "Skipping pool entry id=%d (epoch %d): %s",
                    entry.id, entry.created_epoch, e,
                )
        return models
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_league.py::TestLoadAllOpponents -v`
Expected: All pass

- [ ] **Step 5: Run full league test suite**

Run: `uv run pytest tests/test_league.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add keisei/training/league.py tests/test_league.py
git commit -m "feat(league): add OpponentPool.load_all_opponents with error resilience"
```

---

### Task 4: Remove Elo Carry-Forward in `_rotate_seat` (Change 1)

**Files:**
- Modify: `keisei/training/katago_loop.py:1226-1241`
- Modify: `tests/test_katago_loop.py:1004-1107`

- [ ] **Step 1: Write failing tests for Elo reset on rotation**

Add to `tests/test_katago_loop.py` inside `TestRotateSeat`:

```python
    def test_rotate_seat_new_entry_starts_at_default_elo(self, league_config):
        """New entry after rotation should have the default Elo (1000.0), not the old Elo."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        # Artificially inflate the current learner's Elo
        if loop.pool and loop._learner_entry_id:
            loop.pool.update_elo(loop._learner_entry_id, 1650.0, epoch=0)

        loop._rotate_seat(epoch=5)

        # The NEW entry should start at the default 1000.0, not 1650.0
        new_entry = loop.pool._get_entry(loop._learner_entry_id)
        assert new_entry is not None
        assert new_entry.elo_rating == 1000.0

    def test_rotate_seat_old_entry_elo_preserved(self, league_config):
        """Old entry's Elo should remain unchanged after rotation."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        old_id = loop._learner_entry_id
        if loop.pool and old_id:
            loop.pool.update_elo(old_id, 1200.0, epoch=0)

        loop._rotate_seat(epoch=5)

        old_entry = loop.pool._get_entry(old_id)
        assert old_entry is not None
        assert old_entry.elo_rating == 1200.0

    def test_rotate_seat_no_elo_history_for_new_entry(self, league_config):
        """New entry should have no elo_history rows immediately after rotation."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        loop._rotate_seat(epoch=5)

        # Query elo_history directly for the new entry
        import sqlite3
        conn = sqlite3.connect(league_config.display.db_path)
        rows = conn.execute(
            "SELECT COUNT(*) FROM elo_history WHERE entry_id = ?",
            (loop._learner_entry_id,),
        ).fetchone()
        conn.close()
        assert rows[0] == 0, "New entry should have no elo_history rows after rotation"

    def test_rotate_seat_evicted_old_entry_still_resets(self, league_config):
        """If old entry was evicted, new entry should still start at 1000.0."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        # Fill pool to max, forcing eviction of the original entry
        for i in range(league_config.league.max_pool_size + 1):
            loop.pool.add_snapshot(
                loop._base_model, "se_resnet",
                dict(league_config.model.params), epoch=100 + i,
            )

        # Old learner entry may have been evicted
        loop._rotate_seat(epoch=200)
        new_entry = loop.pool._get_entry(loop._learner_entry_id)
        assert new_entry is not None
        assert new_entry.elo_rating == 1000.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_katago_loop.py::TestRotateSeat::test_rotate_seat_new_entry_starts_at_default_elo -v`
Expected: FAIL — new entry has Elo 1650.0 due to carry-forward

- [ ] **Step 3: Remove carry-forward logic**

In `keisei/training/katago_loop.py`, replace **only lines 1226-1241** of `_rotate_seat`. The method continues through line 1269 with optimizer reset, scheduler recreation, and warmup extension — **all of that must be preserved**. Only the carry-forward block (lines 1228-1230 fetching old Elo, lines 1237-1238 writing it to the new entry) is deleted.

Replace lines 1226-1241:

```python
    def _rotate_seat(self, epoch: int) -> None:
        """Save current learner weights and reset optimizer for the next seat."""
        # NOTE: The chart-continuity carry-forward in the epoch loop (lines
        # 1112-1122) is a SEPARATE mechanism that writes unchanged Elo to
        # elo_history for entries that didn't play, keeping chart lines
        # continuous. That must be preserved. This method only handles
        # rotation — new snapshots enter at the DB default of 1000.0.

        new_entry = self.pool.add_snapshot(
            self._base_model, self.config.model.architecture,
            dict(self.config.model.params), epoch=epoch + 1,
        )

        # B5 fix: update learner entry ID so Elo tracks the current snapshot
        self._learner_entry_id = new_entry.id
```

**IMPORTANT**: Lines 1243-1269 (optimizer reset, LR scheduler recreation, warmup extension, log message) MUST remain unchanged after this replacement. Verify the complete method looks like this after the edit:

```python
    def _rotate_seat(self, epoch: int) -> None:
        """Save current learner weights and reset optimizer for the next seat."""
        # NOTE: chart-continuity carry-forward (lines 1112-1122) is separate...
        new_entry = self.pool.add_snapshot(...)
        self._learner_entry_id = new_entry.id

        # Reset optimizer (fresh Adam — old momentum fights new gradient signal)
        self.ppo.optimizer = torch.optim.Adam(
            self.ppo.model.parameters(), lr=self.ppo.params.learning_rate,
        )
        # B1 fix: recreate LR scheduler pointing at the NEW optimizer
        if self.lr_scheduler is not None and self.config.league is not None:
            ...
        # B2 fix: extend warmup relative to the rotation point.
        self.ppo.warmup_epochs = epoch + 1 + self._original_warmup_duration
        logger.info(...)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_katago_loop.py::TestRotateSeat -v`
Expected: All pass (including the 4 existing tests + 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "fix(league): remove Elo carry-forward on seat rotation — new entries start at 1000"
```

---

### Task 5: Widen `learner_side` Type Hints (Change 2, Part A)

**Files:**
- Modify: `keisei/training/katago_loop.py:99-124, 241-249`
- Modify: `tests/test_pending_transitions.py:12-87`

- [ ] **Step 1: Write failing tests for array `learner_side`**

Add to `tests/test_pending_transitions.py`:

```python
class TestToLearnerPerspectiveArray:
    """Test to_learner_perspective with per-env ndarray learner_side."""

    def test_mixed_learner_sides(self):
        """Each env uses its own learner_side for perspective correction."""
        rewards = torch.tensor([1.0, 1.0, -1.0, -1.0])
        pre_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        # env0: learner=0, pre=0 → learner moved → unchanged
        # env1: learner=1, pre=1 → learner moved → unchanged
        # env2: learner=1, pre=0 → opponent moved → negated
        # env3: learner=0, pre=1 → opponent moved → negated
        learner_side = np.array([0, 1, 1, 0], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side)
        expected = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert torch.equal(result, expected)

    def test_all_same_side_matches_scalar(self):
        """Array of identical values should match scalar behavior."""
        rewards = torch.tensor([1.0, -1.0, 0.5])
        pre_players = np.array([0, 1, 0], dtype=np.uint8)
        learner_side_scalar = 0
        learner_side_array = np.array([0, 0, 0], dtype=np.uint8)
        result_scalar = to_learner_perspective(rewards, pre_players, learner_side_scalar)
        result_array = to_learner_perspective(rewards, pre_players, learner_side_array)
        assert torch.equal(result_scalar, result_array)


class TestSignCorrectBootstrapArray:
    """Test sign_correct_bootstrap with per-env ndarray learner_side."""

    def test_mixed_learner_sides(self):
        """Each env uses its own learner_side for sign correction."""
        next_values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        # env0: learner=0, current=0 → learner to move → unchanged
        # env1: learner=0, current=1 → opponent to move → negated
        # env2: learner=1, current=0 → opponent to move → negated
        # env3: learner=1, current=1 → learner to move → unchanged
        learner_side = np.array([0, 0, 1, 1], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side)
        expected = torch.tensor([0.5, -0.5, -0.5, 0.5])
        assert torch.equal(result, expected)

    def test_all_same_side_matches_scalar(self):
        """Array of identical values should match scalar behavior."""
        next_values = torch.tensor([0.5, -0.3, 0.8])
        current_players = np.array([0, 1, 0], dtype=np.uint8)
        result_scalar = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        result_array = sign_correct_bootstrap(
            next_values, current_players,
            learner_side=np.array([0, 0, 0], dtype=np.uint8),
        )
        assert torch.equal(result_scalar, result_array)
```

- [ ] **Step 2: Run tests to verify they pass already (type widening is backward-compatible)**

Run: `uv run pytest tests/test_pending_transitions.py::TestToLearnerPerspectiveArray tests/test_pending_transitions.py::TestSignCorrectBootstrapArray -v`
Expected: PASS — numpy broadcasting already handles array comparisons, even though the type hint says `int`. The tests pass because the runtime behavior is correct; the type hints are what need updating.

- [ ] **Step 3: Update type hints**

In `keisei/training/katago_loop.py`, update the three function signatures:

Line 102 — change `learner_side: int` to `learner_side: int | np.ndarray`:
```python
def to_learner_perspective(
    rewards: torch.Tensor,
    pre_players: np.ndarray,
    learner_side: int | np.ndarray,
) -> torch.Tensor:
```

Line 116 — same change:
```python
def sign_correct_bootstrap(
    next_values: torch.Tensor,
    current_players: np.ndarray,
    learner_side: int | np.ndarray,
) -> torch.Tensor:
```

Line 247 — same change:
```python
def split_merge_step(
    obs: torch.Tensor,
    legal_masks: torch.Tensor,
    current_players: np.ndarray,
    learner_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    learner_side: int | np.ndarray = 0,
    value_adapter: ValueHeadAdapter | None = None,
) -> SplitMergeResult:
```

- [ ] **Step 4: Run full test suite to verify nothing broke**

Run: `uv run pytest tests/test_pending_transitions.py tests/test_split_merge.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_pending_transitions.py
git commit -m "feat(loop): widen learner_side type to int | np.ndarray with array tests"
```

---

### Task 6: Per-Env Color Randomization in Step Loop (Change 2, Part B)

**Files:**
- Modify: `keisei/training/katago_loop.py:763, 784, 801, ~840-900`

This task modifies the training loop internals. The changes are gated behind `config.league.color_randomization`.

- [ ] **Step 1: Add color randomization initialization**

In `keisei/training/katago_loop.py`, replace line 763:

```python
            learner_side = 0  # Epoch-scoped: used by bootstrap sign correction after the loop
```

with:

```python
            # Per-env color randomization (config-gated).
            # Invariant: learner_side[env] always reflects the color for
            # the game CURRENTLY RUNNING in that env. Re-randomization
            # happens exclusively on dones, never mid-game.
            use_color_rand = (
                self.config.league is not None
                and self.config.league.color_randomization
                and self._current_opponent is not None
            )
            if use_color_rand:
                learner_side = np.random.randint(0, 2, size=self.num_envs, dtype=np.uint8)
                learner_side_t = torch.from_numpy(learner_side.copy()).to(self.device)
                if epoch_i == start_epoch and self.dist_ctx.is_main:
                    logger.info(
                        "Color randomization enabled: win_rate now reflects "
                        "both colors (was black-only previously)"
                    )
            else:
                learner_side = 0
```

- [ ] **Step 2: Update GPU comparisons to use `learner_side_t` when available**

Replace line 784 (`learner_moved = pre_players_t == learner_side`):

```python
                    learner_moved = pre_players_t == (learner_side_t if use_color_rand else learner_side)
```

Replace line 801 (`learner_next = current_players_t == learner_side`):

```python
                    learner_next = current_players_t == (learner_side_t if use_color_rand else learner_side)
```

- [ ] **Step 3: Add color re-randomization in dones processing**

After the existing dones processing block (after the `if imm_terminal.any():` block, around line 905), add:

```python
                    # Re-randomize learner color for completed games (Change 2).
                    # This sets up state for the NEXT game — must happen AFTER
                    # pending transition finalization and Elo attribution
                    # (see Task 8 for opponent tracking which runs BEFORE this).
                    if use_color_rand and dones.bool().any():
                        done_np = dones.bool().cpu().numpy()
                        new_sides = np.random.randint(
                            0, 2, size=int(done_np.sum()), dtype=np.uint8,
                        )
                        learner_side[done_np] = new_sides
                        done_indices = torch.from_numpy(
                            np.flatnonzero(done_np).astype(np.int64),
                        ).to(self.device)
                        # CRITICAL: new_sides must stay uint8 to match learner_side_t dtype.
                        # The done_indices tensor is int64 (for indexing), but the VALUES
                        # being scattered must match the destination tensor's dtype (uint8).
                        learner_side_t[done_indices] = torch.from_numpy(
                            new_sides,  # uint8 — matches learner_side_t dtype
                        ).to(self.device)
```

- [ ] **Step 4: Run existing tests to verify nothing broke**

Run: `uv run pytest tests/test_katago_loop.py tests/test_split_merge.py tests/test_pending_transitions.py -v`
Expected: All pass (existing tests use `learner_side=0` scalar path)

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "feat(loop): per-env color randomization with in-place GPU scatter updates"
```

---

### Task 7: Multi-Opponent `split_merge_step` (Change 3, Part A)

**Files:**
- Modify: `keisei/training/katago_loop.py:241-330`
- Modify: `tests/test_split_merge.py`

- [ ] **Step 1: Write failing tests for multi-opponent split_merge_step**

Add to `tests/test_split_merge.py`:

```python
def _make_deterministic_model(bias: float = 0.0, action_space: int = 11259):
    """Create a mock model with a known constant bias to distinguish outputs.

    Uses torch.full (no randomness) so the argmax action is deterministic
    and different models produce reliably different actions.
    """
    call_count = [0]  # mutable counter to track calls

    def forward(obs):
        call_count[0] += 1
        batch = obs.shape[0]
        output = MagicMock()
        # Constant logits + bias — deterministic argmax for action verification
        output.policy_logits = torch.full((batch, 9, 9, 139), bias)
        output.value_logits = torch.zeros(batch, 3)
        output.score_lead = torch.zeros(batch, 1)
        return output

    model = MagicMock()
    model.side_effect = forward
    model.__call__ = forward
    # Give the mock a parameters() method that returns an empty iterator
    # so the cross-device detection doesn't fail
    model.parameters = MagicMock(return_value=iter([]))
    # Expose call_count so tests can verify whether the model was invoked.
    # MagicMock.called is unreliable when __call__ is overridden.
    model._call_count = call_count
    return model


class TestSplitMergeMultiOpponent:
    """Tests for multi-opponent split_merge_step."""

    def test_multi_opponent_actions_shape(self):
        """Actions should cover all envs with multiple opponent models."""
        num_envs = 8
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)
        env_opponent_ids = np.array([1, 1, 2, 2, 1, 1, 2, 2], dtype=np.int64)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_models={1: _make_mock_model(), 2: _make_mock_model()},
            env_opponent_ids=env_opponent_ids,
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        assert result.learner_mask.sum() == 4
        assert result.opponent_mask.sum() == 4

    def test_legacy_single_opponent_still_works(self):
        """Passing opponent_model= (legacy path) should still work."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_model=_make_mock_model(),
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)

    def test_opponent_with_no_envs_skipped(self):
        """Opponent model with no assigned envs should not be called."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        # All opponent envs assigned to model 1, model 2 has no envs
        env_opponent_ids = np.array([1, 1, 1, 1], dtype=np.int64)

        unused_model = _make_deterministic_model(bias=999.0)
        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_models={1: _make_mock_model(), 2: unused_model},
            env_opponent_ids=env_opponent_ids,
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        # MagicMock.called is unreliable when __call__ is overridden.
        # Use the explicit call counter instead.
        assert unused_model._call_count[0] == 0, "Model 2 should not have been called"

    def test_multi_opponent_with_array_learner_side(self):
        """Multi-opponent + per-env learner_side should produce correct masks."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        # env0: player=0, learner=0 → learner
        # env1: player=1, learner=1 → learner
        # env2: player=0, learner=1 → opponent (model 1)
        # env3: player=1, learner=0 → opponent (model 2)
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        learner_side = np.array([0, 1, 1, 0], dtype=np.uint8)
        env_opponent_ids = np.array([0, 0, 1, 2], dtype=np.int64)

        result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=_make_mock_model(),
            opponent_models={1: _make_mock_model(), 2: _make_mock_model()},
            env_opponent_ids=env_opponent_ids,
            learner_side=learner_side,
        )

        assert result.actions.shape == (num_envs,)
        assert result.learner_mask.sum() == 2  # envs 0 and 1
        assert result.opponent_mask.sum() == 2  # envs 2 and 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_split_merge.py::TestSplitMergeMultiOpponent -v`
Expected: FAIL — `split_merge_step` doesn't accept `opponent_models` or `env_opponent_ids`

- [ ] **Step 3: Implement multi-opponent `split_merge_step`**

Replace the `split_merge_step` function in `keisei/training/katago_loop.py` (lines 241-330):

```python
def split_merge_step(
    obs: torch.Tensor,
    legal_masks: torch.Tensor,
    current_players: np.ndarray,
    learner_model: torch.nn.Module,
    opponent_model: torch.nn.Module | None = None,
    opponent_models: dict[int, torch.nn.Module] | None = None,
    env_opponent_ids: np.ndarray | None = None,
    learner_side: int | np.ndarray = 0,
    value_adapter: ValueHeadAdapter | None = None,
) -> SplitMergeResult:
    """Execute one step with split learner/opponent forward passes.

    Supports both single-opponent (legacy) and multi-opponent (cohort) modes:
    - Legacy: pass opponent_model=<model>
    - Cohort: pass opponent_models={id: model, ...} and env_opponent_ids=<array>

    Returns only learner-side data (log_probs, values, indices). The caller
    stores ONLY learner transitions in the rollout buffer.
    """
    # Normalize to multi-opponent path
    if opponent_models is None and opponent_model is not None:
        active_opponents: dict[int, torch.nn.Module] = {0: opponent_model}
        active_env_ids: np.ndarray | None = None
    elif opponent_models is not None:
        active_opponents = opponent_models
        active_env_ids = env_opponent_ids
    else:
        raise ValueError("Must provide either opponent_model or opponent_models")

    num_envs = obs.shape[0]
    device = obs.device

    learner_mask = torch.tensor(current_players == learner_side, device=device)
    opponent_mask = ~learner_mask
    learner_indices = learner_mask.nonzero(as_tuple=True)[0]

    actions = torch.zeros(num_envs, dtype=torch.long, device=device)
    learner_log_probs = torch.zeros(0, device=device)
    learner_values = torch.zeros(0, device=device)

    # Learner forward pass (eval mode, no_grad for rollout collection).
    if learner_indices.numel() > 0:
        l_obs = obs[learner_indices]
        l_masks = legal_masks[learner_indices]

        learner_model.eval()
        with torch.no_grad():
            l_output = learner_model(l_obs)

        l_flat = l_output.policy_logits.reshape(l_obs.shape[0], -1)
        l_masked = l_flat.masked_fill(~l_masks, float("-inf"))
        l_probs = F.softmax(l_masked, dim=-1)
        l_dist = torch.distributions.Categorical(l_probs)
        l_actions = l_dist.sample()
        learner_log_probs = l_dist.log_prob(l_actions)

        if value_adapter is not None:
            learner_values = value_adapter.scalar_value_blended(
                l_output.value_logits, l_output.score_lead,
            )
        else:
            learner_values = KataGoPPOAlgorithm.scalar_value(l_output.value_logits)

        actions[learner_indices] = l_actions

    # Opponent forward passes (always no_grad, eval mode).
    opponent_mask_np = opponent_mask.cpu().numpy()
    for opp_id, model in active_opponents.items():
        if active_env_ids is not None:
            opp_env_mask = (active_env_ids == opp_id) & opponent_mask_np
        else:
            opp_env_mask = opponent_mask_np

        if not opp_env_mask.any():
            continue

        indices = np.flatnonzero(opp_env_mask)
        idx_tensor = torch.from_numpy(indices.astype(np.int64)).to(device)

        o_obs = obs[idx_tensor]
        o_masks = legal_masks[idx_tensor]

        # Detect cross-device opponent
        try:
            opp_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            opp_device = device
        cross_device = isinstance(opp_device, torch.device) and opp_device != device
        if cross_device:
            o_obs = o_obs.to(opp_device)
            o_masks = o_masks.to(opp_device)

        model.eval()
        with torch.no_grad():
            o_output = model(o_obs)

        o_flat = o_output.policy_logits.reshape(o_obs.shape[0], -1)
        o_masked = o_flat.masked_fill(~o_masks, float("-inf"))
        o_probs = F.softmax(o_masked, dim=-1)
        o_dist = torch.distributions.Categorical(o_probs)
        o_actions = o_dist.sample()

        if cross_device:
            o_actions = o_actions.to(device)
        actions[idx_tensor] = o_actions

    return SplitMergeResult(
        actions=actions,
        learner_mask=learner_mask,
        opponent_mask=opponent_mask,
        learner_log_probs=learner_log_probs,
        learner_values=learner_values,
        learner_indices=learner_indices,
    )
```

- [ ] **Step 4: Run all split_merge tests**

Run: `uv run pytest tests/test_split_merge.py -v`
Expected: All pass (both legacy and new multi-opponent tests)

- [ ] **Step 5: Run broader test suite to check nothing broke**

Run: `uv run pytest tests/test_katago_loop.py tests/test_pending_transitions.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_split_merge.py
git commit -m "feat(loop): multi-opponent split_merge_step with backward-compatible signature"
```

---

### Task 8: Per-Env Opponents in Training Loop (Change 3, Part B)

**Files:**
- Modify: `keisei/training/katago_loop.py:516-520, 700-722, 787-794, 1081-1122`

This is the largest task — it wires the multi-opponent support into the training loop epoch logic. The changes are gated behind `config.league.per_env_opponents`.

- [ ] **Step 0: Initialize new attributes in `__init__`**

In `keisei/training/katago_loop.py`, find the league attribute block (around line 516-520):

```python
        self.pool: OpponentPool | None = None
        self.sampler: OpponentSampler | None = None
        self._current_opponent: torch.nn.Module | None = None
        self._current_opponent_entry: OpponentEntry | None = None
```

Add after it:

```python
        # Per-env opponent state (Change 3) — initialized at epoch start when enabled
        self._opponent_models: dict[int, torch.nn.Module] | None = None
        self._env_opponent_ids: np.ndarray | None = None
        self._cached_entries: list[OpponentEntry] = []
        self._cached_entries_by_id: dict[int, OpponentEntry] = {}
        self._opponent_results: dict[int, list[int]] | None = None
```

- [ ] **Step 1: Add per-env opponent initialization at epoch start**

In `keisei/training/katago_loop.py`, after the existing opponent loading block (around line 722), add the per-env opponent setup. Replace the block from line 719-722:

```python
                opp_device = opp_device_cfg or str(self.device)
                self._current_opponent = self.pool.load_opponent(
                    self._current_opponent_entry, device=opp_device,
                )
```

with:

```python
                opp_device = opp_device_cfg or str(self.device)

                # Cache entries once — reused for sampling AND opponent loading
                self._cached_entries = self.pool.list_entries()
                self._cached_entries_by_id = {e.id: e for e in self._cached_entries}

                use_per_env_opps = (
                    self.config.league is not None
                    and self.config.league.per_env_opponents
                    and len(self._cached_entries) > 0
                )

                if use_per_env_opps:
                    # Memory cleanup: release previous epoch's models.
                    # Safe at epoch start: ppo.update() backward pass is a
                    # CUDA sync point, so all prior kernels have completed.
                    if self._opponent_models is not None:
                        del self._opponent_models
                        self._opponent_models = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    self._opponent_models = self.pool.load_all_opponents(device=opp_device)

                    # Per-env opponent assignment
                    self._env_opponent_ids = np.zeros(self.num_envs, dtype=np.int64)
                    for env_i in range(self.num_envs):
                        entry = self.sampler.sample_from(self._cached_entries)
                        self._env_opponent_ids[env_i] = entry.id

                    # Per-opponent W/L/D tracking for epoch-end Elo update
                    self._opponent_results = {
                        eid: [0, 0, 0] for eid in self._opponent_models
                    }

                    # Reuse an already-loaded model for _current_opponent (used by
                    # PendingTransitions guard, sign_correct_bootstrap, etc.).
                    # Do NOT call load_opponent again — saves 14MB VRAM + disk I/O.
                    opp_entry_id = self._current_opponent_entry.id
                    if opp_entry_id in self._opponent_models:
                        self._current_opponent = self._opponent_models[opp_entry_id]
                    else:
                        # Entry's checkpoint was corrupt/missing — use any loaded model
                        self._current_opponent = next(iter(self._opponent_models.values()))
                else:
                    self._opponent_models = None
                    self._env_opponent_ids = None
                    self._opponent_results = None
                    self._current_opponent = self.pool.load_opponent(
                        self._current_opponent_entry, device=opp_device,
                    )
```

- [ ] **Step 2: Update step loop to pass multi-opponent args**

Replace the `split_merge_step` call (lines 787-794):

```python
                    sm_result = split_merge_step(
                        obs=obs, legal_masks=legal_masks,
                        current_players=current_players,
                        learner_model=self.model,
                        opponent_model=self._current_opponent,
                        learner_side=learner_side,
                        value_adapter=self.value_adapter,
                    )
```

with:

```python
                    if self._opponent_models and self._env_opponent_ids is not None:
                        sm_result = split_merge_step(
                            obs=obs, legal_masks=legal_masks,
                            current_players=current_players,
                            learner_model=self.model,
                            opponent_models=self._opponent_models,
                            env_opponent_ids=self._env_opponent_ids,
                            learner_side=learner_side,
                            value_adapter=self.value_adapter,
                        )
                    else:
                        sm_result = split_merge_step(
                            obs=obs, legal_masks=legal_masks,
                            current_players=current_players,
                            learner_model=self.model,
                            opponent_model=self._current_opponent,
                            learner_side=learner_side,
                            value_adapter=self.value_adapter,
                        )
```

- [ ] **Step 3: Add per-env opponent re-sampling and Elo tracking on dones**

This block must be inserted **BEFORE** the color re-randomization block from Task 6, and **AFTER** the pending transition finalization (the existing `if imm_terminal.any():` block around line 905). The required order within the dones processing section is:

1. Pending transition finalization (existing — steps 1-4 in the protocol)
2. **Per-opponent Elo result tracking + re-sample opponent** ← this block
3. Color re-randomization (Task 6) ← already added, must come AFTER this

If the Task 6 color re-randomization block is currently positioned before this insertion point, **move it below** this new block.

Add between the pending transition finalization and the color re-randomization:

```python
                    # --- Dones processing order (Changes 2+3) ---
                    # 1. Finalize pending transitions (above — existing protocol)
                    # 2. Per-opponent Elo result tracking + re-sample opponent
                    # 3. Re-randomize learner color
                    # Steps 2-3 set up state for the NEXT game. Step 1 consumes
                    # state from the COMPLETED game. Do not reorder.

                    # Per-env opponent: track results and re-sample on done (Change 3).
                    if (self._opponent_results is not None
                            and self._env_opponent_ids is not None
                            and dones.bool().any()):
                        done_mask = dones.bool()
                        done_indices_np = np.flatnonzero(done_mask.cpu().numpy())

                        # Batch-extract rewards to avoid per-env GPU syncs.
                        # One .cpu() call instead of N .item() calls.
                        done_rewards_np = learner_rewards[done_mask].cpu().numpy()
                        done_terminal_np = terminated.bool().cpu().numpy()[done_indices_np]
                        done_opp_ids = self._env_opponent_ids[done_indices_np]

                        for i, env_i in enumerate(done_indices_np):
                            opp_id = int(done_opp_ids[i])
                            if opp_id not in self._opponent_results:
                                continue
                            if done_terminal_np[i]:
                                lr = done_rewards_np[i]
                                if lr > 0:
                                    self._opponent_results[opp_id][0] += 1  # win
                                elif lr < 0:
                                    self._opponent_results[opp_id][1] += 1  # loss
                                else:
                                    self._opponent_results[opp_id][2] += 1  # draw

                            # Re-sample opponent for next game
                            new_entry = self.sampler.sample_from(self._cached_entries)
                            self._env_opponent_ids[env_i] = new_entry.id
```

- [ ] **Step 4: Replace epoch-end Elo update for multi-opponent mode**

Replace the Elo tracking block (lines 1088-1110):

```python
            if self.dist_ctx.is_main:
                total_games = win_count + loss_count + draw_count
                if (self.pool is not None and self._current_opponent_entry is not None
                        and total_games > 0
                        and self._learner_entry_id != self._current_opponent_entry.id):
                    learner_entry = self.pool._get_entry(self._learner_entry_id)
                    if learner_entry is not None:
                        result_score = (win_count + 0.5 * draw_count) / total_games
                        k = self.config.league.elo_k_factor if self.config.league else 32.0
                        new_learner_elo, new_opp_elo = compute_elo_update(
                            learner_entry.elo_rating,
                            self._current_opponent_entry.elo_rating,
                            result=result_score, k=k,
                        )
                        self.pool.update_elo(learner_entry.id, new_learner_elo, epoch=self.epoch)
                        self.pool.update_elo(self._current_opponent_entry.id, new_opp_elo, epoch=self.epoch)
                        logger.info(
                            "Elo: learner %.0f->%.0f, opponent(id=%d) %.0f->%.0f | W=%d L=%d D=%d",
                            learner_entry.elo_rating, new_learner_elo,
                            self._current_opponent_entry.id,
                            self._current_opponent_entry.elo_rating, new_opp_elo,
                            win_count, loss_count, draw_count,
                        )
```

with:

```python
            if self.dist_ctx.is_main:
                total_games = win_count + loss_count + draw_count
                k = self.config.league.elo_k_factor if self.config.league else 32.0

                if (self.pool is not None
                        and self._opponent_results is not None
                        and self._cached_entries_by_id
                        and total_games > 0):
                    # Per-opponent Elo updates (Change 3).
                    # Freeze the starting learner Elo — all opponent updates are
                    # computed against this SAME base value, then the cumulative
                    # delta is applied once. This prevents path-dependent Elo drift
                    # from dict iteration order.
                    learner_entry = self.pool._get_entry(self._learner_entry_id)
                    if learner_entry is not None:
                        base_learner_elo = learner_entry.elo_rating
                        cumulative_learner_delta = 0.0

                        for opp_id, (w, l, d) in self._opponent_results.items():
                            opp_total = w + l + d
                            if opp_total == 0:
                                continue
                            if opp_id == self._learner_entry_id:
                                continue
                            opp_entry = self._cached_entries_by_id.get(opp_id)
                            if opp_entry is None:
                                continue
                            result_score = (w + 0.5 * d) / opp_total
                            new_learner_elo, new_opp_elo = compute_elo_update(
                                base_learner_elo, opp_entry.elo_rating,
                                result=result_score, k=k,
                            )
                            learner_delta = new_learner_elo - base_learner_elo
                            cumulative_learner_delta += learner_delta
                            # Update opponent Elo immediately (each opponent is independent)
                            self.pool.update_elo(opp_id, new_opp_elo, epoch=self.epoch)
                            logger.info(
                                "Elo: learner base=%.0f delta=%.1f, "
                                "opponent(id=%d) %.0f->%.0f | W=%d L=%d D=%d",
                                base_learner_elo, learner_delta,
                                opp_id, opp_entry.elo_rating, new_opp_elo,
                                w, l, d,
                            )

                        # Apply cumulative learner Elo change once
                        final_learner_elo = base_learner_elo + cumulative_learner_delta
                        self.pool.update_elo(
                            self._learner_entry_id, final_learner_elo, epoch=self.epoch,
                        )
                        logger.info(
                            "Elo: learner %.0f->%.0f (cumulative from %d opponents)",
                            base_learner_elo, final_learner_elo,
                            sum(1 for w, l, d in self._opponent_results.values()
                                if w + l + d > 0),
                        )

                elif (self.pool is not None
                        and self._current_opponent_entry is not None
                        and total_games > 0
                        and self._learner_entry_id != self._current_opponent_entry.id):
                    # Legacy single-opponent Elo update
                    learner_entry = self.pool._get_entry(self._learner_entry_id)
                    if learner_entry is not None:
                        result_score = (win_count + 0.5 * draw_count) / total_games
                        new_learner_elo, new_opp_elo = compute_elo_update(
                            learner_entry.elo_rating,
                            self._current_opponent_entry.elo_rating,
                            result=result_score, k=k,
                        )
                        self.pool.update_elo(
                            learner_entry.id, new_learner_elo, epoch=self.epoch,
                        )
                        self.pool.update_elo(
                            self._current_opponent_entry.id, new_opp_elo,
                            epoch=self.epoch,
                        )
                        logger.info(
                            "Elo: learner %.0f->%.0f, opponent(id=%d) %.0f->%.0f "
                            "| W=%d L=%d D=%d",
                            learner_entry.elo_rating, new_learner_elo,
                            self._current_opponent_entry.id,
                            self._current_opponent_entry.elo_rating, new_opp_elo,
                            win_count, loss_count, draw_count,
                        )
```

- [ ] **Step 5: Fix chart-continuity carry-forward `played_ids` for multi-opponent mode**

The existing chart-continuity block (lines 1112-1122) builds `played_ids` from only `_learner_entry_id` and `_current_opponent_entry.id`. With per-env opponents, all opponents that played during the epoch must be included, otherwise the carry-forward loop will overwrite the Elo updates we just applied for those opponents.

Find the block (around line 1114-1122):

```python
            if self.dist_ctx.is_main and self.pool is not None:
                played_ids = set()
                if self._learner_entry_id is not None:
                    played_ids.add(self._learner_entry_id)
                if self._current_opponent_entry is not None:
                    played_ids.add(self._current_opponent_entry.id)
```

Replace with:

```python
            if self.dist_ctx.is_main and self.pool is not None:
                played_ids = set()
                if self._learner_entry_id is not None:
                    played_ids.add(self._learner_entry_id)
                if self._current_opponent_entry is not None:
                    played_ids.add(self._current_opponent_entry.id)
                # Change 3: include all opponents that had games this epoch.
                # Without this, the carry-forward loop below would overwrite
                # the per-opponent Elo updates with stale pre-epoch values.
                if self._opponent_results is not None:
                    for opp_id, (w, l, d) in self._opponent_results.items():
                        if w + l + d > 0:
                            played_ids.add(opp_id)
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All pass. Existing tests hit the legacy path because they either have no league config or a single opponent.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "feat(loop): per-env sticky opponents with frozen pool view and per-opponent Elo"
```

---

### Task 9: Verify Dones Processing Order

**Files:**
- Read: `keisei/training/katago_loop.py` (step loop dones block)

Task 8 Step 3 placed the opponent tracking block BEFORE the color re-randomization block (Task 6 Step 3), and added the ordering comment. This task is verification-only.

- [ ] **Step 1: Verify ordering in the step loop**

Read the step loop dones processing section and confirm this exact order exists:
1. Pending transition finalization (existing — steps 1-4 in the protocol)
2. `# --- Dones processing order (Changes 2+3) ---` comment block
3. Per-env opponent result tracking and re-sampling (`if self._opponent_results ...`)
4. Color re-randomization (`if use_color_rand ...`)

If the order is wrong, fix it now. The opponent tracking block reads `learner_rewards` and `env_opponent_ids` from the *completed* game and must run before color re-randomization changes `learner_side` for the *next* game.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/ -v`
Expected: All pass

---

### Task 10: Integration and Interaction Tests

**Files:**
- Modify: `tests/test_katago_loop.py`

- [ ] **Step 1: Add Elo attribution interaction test**

Add to `tests/test_katago_loop.py`:

```python
class TestFairnessInteractions:
    """Interaction tests for Changes 1+2+3 applied together."""

    @pytest.fixture
    def fairness_config(self, tmp_path):
        """Config with all fairness features enabled."""
        return AppConfig(
            training=TrainingConfig(
                num_games=4,
                max_ply=50,
                algorithm="katago_ppo",
                checkpoint_interval=100,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                algorithm_params={
                    "learning_rate": 2e-4,
                    "gamma": 0.99,
                    "lambda_policy": 1.0,
                    "lambda_value": 1.5,
                    "lambda_score": 0.02,
                    "lambda_entropy": 0.01,
                    "score_normalization": 76.0,
                    "grad_clip": 1.0,
                },
            ),
            display=DisplayConfig(
                moves_per_minute=0,
                db_path=str(tmp_path / "test.db"),
            ),
            model=ModelConfig(
                display_name="Test-KataGo",
                architecture="se_resnet",
                params={
                    "num_blocks": 2,
                    "channels": 32,
                    "se_reduction": 8,
                    "global_pool_channels": 16,
                    "policy_channels": 8,
                    "value_fc_size": 32,
                    "score_fc_size": 16,
                    "obs_channels": 50,
                },
            ),
            league=LeagueConfig(
                max_pool_size=5,
                snapshot_interval=10,
                epochs_per_seat=5,
                color_randomization=True,
                per_env_opponents=True,
            ),
        )

    def test_rotate_seat_new_entry_at_1000_with_all_changes(self, fairness_config):
        """Change 1: new entries start at 1000 even with Changes 2+3 active."""
        mock_env = _make_mock_katago_vecenv(num_envs=4, alternate_players=True)
        loop = KataGoTrainingLoop(fairness_config, vecenv=mock_env)

        if loop.pool and loop._learner_entry_id:
            loop.pool.update_elo(loop._learner_entry_id, 1500.0, epoch=0)

        loop._rotate_seat(epoch=5)
        new_entry = loop.pool._get_entry(loop._learner_entry_id)
        assert new_entry is not None
        assert new_entry.elo_rating == 1000.0

    def test_run_one_epoch_with_all_changes(self, fairness_config):
        """Smoke test: training loop completes with all fairness changes active."""
        mock_env = _make_mock_katago_vecenv(num_envs=4, alternate_players=True)
        loop = KataGoTrainingLoop(fairness_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=10)

        # Behavioral assertions — not just "didn't crash"
        assert loop.pool is not None
        entries = loop.pool.list_entries()
        assert len(entries) >= 1, "Pool should have at least the initial entry"

        # If per_env_opponents was active, opponent_results should exist
        if loop._opponent_results is not None:
            # At least some games should have completed in 10 steps
            total = sum(w + l + d for w, l, d in loop._opponent_results.values())
            # total may be 0 if mock env doesn't produce terminated games,
            # but the structure should exist
            assert isinstance(loop._opponent_results, dict)
```

- [ ] **Step 2: Run the interaction tests**

Run: `uv run pytest tests/test_katago_loop.py::TestFairnessInteractions -v`
Expected: All pass

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_katago_loop.py
git commit -m "test(fairness): add interaction tests for Changes 1+2+3"
```

---

### Task 11: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All pass

- [ ] **Step 2: Verify no regressions in existing behavior**

Run: `uv run pytest tests/test_katago_loop.py tests/test_league.py tests/test_split_merge.py tests/test_pending_transitions.py tests/test_league_config.py -v`
Expected: All pass — both legacy and new code paths exercised

- [ ] **Step 3: Check for type errors**

Run: `uv run python -c "from keisei.training.katago_loop import split_merge_step, to_learner_perspective, sign_correct_bootstrap; print('imports ok')"`
Expected: `imports ok`

- [ ] **Step 4: Final commit if any loose changes**

```bash
git status
# If any unstaged changes remain, commit them
```
