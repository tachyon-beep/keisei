# Cross-Slot Model Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase league tournament throughput by batching inference across slots that share models, and removing the `effective_parallel` cap that limits concurrent slots to 11.

**Architecture:** Two changes in two files. (1) `ConcurrencyConfig.effective_parallel` returns `parallel_matches` directly; the old `max_resident_models // 2` cap is removed and replaced with a cross-config validation warning in `LeagueConfig.__post_init__`. (2) The per-ply inference loop in `ConcurrentMatchPool.run_round()` is restructured to group env indices by model identity (`id(model)`) and run one batched forward pass per unique model, scattering results back to the global actions tensor.

**Tech Stack:** Python, PyTorch, dataclasses

---

### Task 1: Remove `effective_parallel` cap and add cross-config validation

**Files:**
- Modify: `keisei/config.py:375-388` (ConcurrencyConfig)
- Modify: `keisei/config.py:443-459` (LeagueConfig.__post_init__)
- Modify: `tests/test_league_config.py`

- [ ] **Step 1: Write failing test for new `effective_parallel` behavior**

In `tests/test_league_config.py`, update the existing test and add the new validation test:

```python
def test_effective_parallel_equals_parallel_matches():
    """effective_parallel no longer caps by max_resident_models."""
    c = ConcurrencyConfig(
        parallel_matches=64,
        envs_per_match=4,
        total_envs=256,
        max_resident_models=22,
    )
    assert c.effective_parallel == 64


def test_league_warns_when_cache_too_small(recwarn):
    """Warn when max_resident_models < max_active_entries."""
    LeagueConfig(
        max_active_entries=20,
        concurrency=ConcurrencyConfig(
            parallel_matches=4,
            envs_per_match=8,
            total_envs=32,
            max_resident_models=10,
        ),
    )
    assert len(recwarn) >= 1
    assert "max_resident_models" in str(recwarn[0].message)


def test_league_no_warn_when_cache_sufficient():
    """No warning when max_resident_models >= max_active_entries."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Should NOT raise — cache is large enough
        LeagueConfig(
            max_active_entries=20,
            concurrency=ConcurrencyConfig(
                parallel_matches=64,
                envs_per_match=4,
                total_envs=256,
                max_resident_models=22,
            ),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_league_config.py::test_effective_parallel_equals_parallel_matches tests/test_league_config.py::test_league_warns_when_cache_too_small tests/test_league_config.py::test_league_no_warn_when_cache_sufficient -v`

Expected: `test_effective_parallel_equals_parallel_matches` FAILS (returns 11, not 64). The warn tests fail because the validation doesn't exist yet.

- [ ] **Step 3: Implement the config changes**

In `keisei/config.py`, modify `ConcurrencyConfig`:

```python
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
        if self.max_resident_models < 2:
            raise ValueError(
                f"max_resident_models ({self.max_resident_models}) must be >= 2 "
                f"(at least one model pair)"
            )

    @property
    def effective_parallel(self) -> int:
        """Max concurrent slots — equals parallel_matches.

        Model sharing via the LRU cache means slots don't need 2 unique
        models each.  Cache sizing is validated at LeagueConfig level.
        """
        return self.parallel_matches
```

Remove lines 375-383 (the `min_for_full` warning block).

In `keisei/config.py`, in `LeagueConfig.__post_init__`, add at the end:

```python
        # Cross-config validation: warn if LRU cache can't hold the full pool
        if (
            self.max_active_entries is not None
            and self.concurrency.max_resident_models < self.max_active_entries
        ):
            warnings.warn(
                f"max_resident_models ({self.concurrency.max_resident_models}) < "
                f"max_active_entries ({self.max_active_entries}): LRU model cache "
                f"cannot hold the full opponent pool, which may cause GPU memory "
                f"thrashing during concurrent matches",
                stacklevel=2,
            )
```

- [ ] **Step 4: Update existing test that asserts old cap behavior**

In `tests/test_league_config.py`, find `test_concurrency_config_validation_model_budget` and update:

```python
def test_concurrency_config_validation_model_budget():
    from keisei.config import ConcurrencyConfig

    # max_resident_models < 2 should still fail
    with pytest.raises(ValueError, match="max_resident_models"):
        ConcurrencyConfig(parallel_matches=4, envs_per_match=2, total_envs=8, max_resident_models=1)

    # max_resident < parallel*2 is allowed — effective_parallel is no longer capped
    c = ConcurrencyConfig(parallel_matches=4, envs_per_match=2, total_envs=8, max_resident_models=4)
    assert c.effective_parallel == 4  # was 2, now equals parallel_matches
```

- [ ] **Step 5: Update `test_concurrent_slots_capped_by_max_resident` in test_match_pool.py**

This test (`tests/test_match_pool.py:511-560`) explicitly tests the old capping behavior. With the cap removed, all 4 slots will be active. Update:

```python
class TestMaxResidentModels:
    def test_all_slots_active_with_shared_cache(self) -> None:
        """With LRU cache, all parallel_matches slots can be active
        regardless of max_resident_models (models are shared objects)."""
        config = ConcurrencyConfig(
            parallel_matches=4,
            envs_per_match=2,
            total_envs=8,
            max_resident_models=4,
        )
        pool = ConcurrentMatchPool(config)
        vecenv = MockVecEnv(num_envs=8, terminate_after=2)
        entries = [_make_entry(i) for i in range(8)]
        pairings = [
            (entries[0], entries[1]),
            (entries[2], entries[3]),
            (entries[4], entries[5]),
            (entries[6], entries[7]),
        ]

        results, _stats = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
        )

        assert len(results) == 4  # all pairings completed
```

- [ ] **Step 6: Run all affected tests**

Run: `uv run pytest tests/test_league_config.py tests/test_match_pool.py -v`

Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add keisei/config.py tests/test_league_config.py tests/test_match_pool.py
git commit -m "fix(config): remove effective_parallel cap, add cache-size validation

effective_parallel now returns parallel_matches directly. The old
max_resident_models // 2 cap silently limited concurrent slots to 11
despite parallel_matches=128. Cache sizing is now validated at
LeagueConfig level with an actionable warning."
```

---

### Task 2: Implement cross-slot model batching in `run_round()`

**Files:**
- Modify: `keisei/training/concurrent_matches.py:277-364`
- Test: `tests/test_match_pool.py`

- [ ] **Step 1: Write failing test for batched inference**

Add to `tests/test_match_pool.py`:

```python
class TestCrossSlotBatching:
    """Verify that slots sharing a model use batched forward passes."""

    def test_shared_model_called_once_per_ply(self) -> None:
        """Two slots sharing model_a should produce ONE forward call per ply,
        not two separate calls."""
        pool = _make_pool(parallel_matches=2, envs_per_match=2, total_envs=4)
        vecenv = MockVecEnv(num_envs=4, terminate_after=2)

        forward_calls: list[int] = []  # batch sizes seen
        shared_model = TinyModel()
        original_forward = shared_model.forward

        def tracking_forward(x):
            forward_calls.append(x.shape[0])
            return original_forward(x)

        shared_model.forward = tracking_forward

        other_model_a = TinyModel()
        other_model_b = TinyModel()

        entries = [_make_entry(i) for i in range(4)]
        pairings = [
            (entries[0], entries[1]),  # slot 0: shared_model vs other_model_a
            (entries[2], entries[3]),  # slot 1: shared_model vs other_model_b
        ]

        # load_fn returns shared_model for entries 0 and 2 (model_a in both slots)
        def load_fn(entry):
            if entry.id in (0, 2):
                return shared_model
            elif entry.id == 1:
                return other_model_a
            else:
                return other_model_b

        results, _stats = pool.run_round(
            vecenv,
            pairings,
            load_fn=load_fn,
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
        )

        assert len(results) == 2
        # shared_model should see batched calls (batch > 1) not per-slot calls
        # With 2 slots × ~1 env each for player_a, batch should be ~2
        # The key assertion: we should see FEWER calls than the old per-slot path
        # Old path: 2 calls per ply (one per slot) for shared_model
        # New path: 1 call per ply (batched across slots)
        # Count calls where batch_size > 1 — these are batched
        batched = [b for b in forward_calls if b > 1]
        assert len(batched) > 0, (
            f"Expected batched forward calls (batch > 1) but got sizes: {forward_calls}"
        )

    def test_actions_scattered_correctly_across_slots(self) -> None:
        """After batched inference, each slot's env range must receive
        valid actions (not zeros or crossed wires)."""
        pool = _make_pool(parallel_matches=2, envs_per_match=2, total_envs=4)
        vecenv = MockVecEnv(num_envs=4, terminate_after=3)
        entries = [_make_entry(i) for i in range(4)]
        pairings = [
            (entries[0], entries[1]),
            (entries[2], entries[3]),
        ]

        results, _stats = pool.run_round(
            vecenv,
            pairings,
            load_fn=lambda entry: TinyModel(),
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
        )
        # Both pairings complete with games played — proves actions were valid
        assert len(results) == 2
        for r in results:
            total = r.a_wins + r.b_wins + r.draws
            assert total >= 4

    def test_rollout_integrity_with_batching(self) -> None:
        """Rollout buffers must stay aligned when inference is batched."""
        pool = _make_pool(parallel_matches=2, envs_per_match=2, total_envs=4)
        vecenv = MockVecEnv(num_envs=4, terminate_after=2)
        entries = [_make_entry(i) for i in range(4)]
        pairings = [
            (entries[0], entries[1]),
            (entries[2], entries[3]),
        ]

        shared_model = TinyModel()

        def load_fn(entry):
            if entry.id in (0, 2):
                return shared_model
            return TinyModel()

        results, _stats = pool.run_round(
            vecenv,
            pairings,
            load_fn=load_fn,
            release_fn=lambda ma, mb: None,
            device="cpu",
            games_per_match=4,
            trainable_fn=lambda ea, eb: True,
        )
        assert len(results) == 2
        for r in results:
            assert r.rollout is not None
            T = r.rollout.observations.shape[0]
            assert r.rollout.actions.shape[0] == T
            assert r.rollout.rewards.shape[0] == T
            assert r.rollout.dones.shape[0] == T
            assert r.rollout.legal_masks.shape[0] == T
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_match_pool.py::TestCrossSlotBatching -v`

Expected: `test_shared_model_called_once_per_ply` FAILS (old code makes per-slot calls). The other two should pass since they test correctness that the old code also provides — they establish a regression baseline.

- [ ] **Step 3: Implement batched inference in `run_round()`**

In `keisei/training/concurrent_matches.py`, replace the per-slot inference block (the `for slot in active_slots:` loop at line 293 through `actions[s:e] = slot_actions` and rollout action append at line 364) with the batched version. The replacement spans lines 293-364. Everything before (ply increment, actions tensor init, pre_step_players init) and after (inactive env handling, vecenv.step, post-step processing) stays identical.

Replace lines 293-364 with:

```python
                # --- Phase 1: Collect per-slot data and build model batches ---
                # model_batches: id(model) -> (model, [(global_indices, slot)])
                model_batches: dict[int, tuple[torch.nn.Module, list[tuple[torch.Tensor, _MatchSlot]]]] = {}

                for slot in active_slots:
                    s, e = slot.env_start, slot.env_end
                    partition_obs = obs[s:e]
                    partition_legal = legal_masks[s:e]
                    partition_players = current_players[s:e]

                    # Guard: zero legal actions would produce NaN from softmax(-inf).
                    # Must be checked BEFORE rollout data or pre_step_players are
                    # recorded so the slot is cleanly skipped for this ply.
                    legal_counts = partition_legal.sum(dim=-1)
                    if (legal_counts == 0).any():
                        logger.warning(
                            "Zero legal actions in concurrent pool slot %d — "
                            "ending match early to avoid NaN",
                            slot.index,
                        )
                        slot.games_target = slot.games_completed
                        continue

                    pre_step_players[slot.index] = partition_players.copy()

                    # Collect pre-step rollout data.  Exactly one append per
                    # slot per loop iteration — the loop structure guarantees
                    # the single-pass-per-ply invariant that rollout buffers
                    # depend on.
                    if slot.collect_rollout:
                        slot._obs.append(partition_obs.cpu())
                        slot._masks.append(partition_legal.cpu())
                        slot._perspective.append(
                            torch.from_numpy(partition_players.copy())
                        )

                    assert slot.model_a is not None
                    assert slot.model_b is not None

                    player_a_mask = torch.from_numpy(partition_players == 0).to(device)
                    player_b_mask = ~player_a_mask

                    # Model A envs (player 0 = Black)
                    a_local = player_a_mask.nonzero(as_tuple=True)[0]
                    if a_local.numel() > 0:
                        a_global = a_local + s  # convert to global env indices
                        mid = id(slot.model_a)
                        if mid not in model_batches:
                            model_batches[mid] = (slot.model_a, [])
                        model_batches[mid][1].append((a_global, slot))

                    # Model B envs (player 1 = White)
                    b_local = player_b_mask.nonzero(as_tuple=True)[0]
                    if b_local.numel() > 0:
                        b_global = b_local + s
                        mid = id(slot.model_b)
                        if mid not in model_batches:
                            model_batches[mid] = (slot.model_b, [])
                        model_batches[mid][1].append((b_global, slot))

                # --- Phase 2: Batched forward passes per unique model ---
                for _mid, (model, batch_entries) in model_batches.items():
                    all_indices = torch.cat([idx for idx, _slot in batch_entries])
                    with torch.no_grad():
                        out = model(obs[all_indices])
                        logits = out.policy_logits.reshape(all_indices.numel(), -1)
                        masked = logits.masked_fill(
                            ~legal_masks[all_indices], float("-inf")
                        )
                        probs = F.softmax(masked, dim=-1)
                        sampled = torch.distributions.Categorical(probs).sample()
                    actions[all_indices] = sampled

                # --- Phase 3: Per-slot rollout action collection ---
                for slot in active_slots:
                    if slot.index not in pre_step_players:
                        continue  # skipped by zero-legal guard
                    if slot.collect_rollout:
                        s, e = slot.env_start, slot.env_end
                        slot._actions.append(actions[s:e].cpu())
```

- [ ] **Step 4: Run the new batching tests**

Run: `uv run pytest tests/test_match_pool.py::TestCrossSlotBatching -v`

Expected: ALL PASS

- [ ] **Step 5: Run full test suite for concurrent matches**

Run: `uv run pytest tests/test_match_pool.py tests/test_concurrent_round.py tests/test_reward_attribution.py -v`

Expected: ALL PASS — existing tests verify game counting, rollout integrity, slot lifecycle, stop_event, zero-legal guard, ply ceiling all still work.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/concurrent_matches.py tests/test_match_pool.py
git commit -m "perf(tournament): batch inference across slots by model identity

Replace per-slot serial forward passes with grouped batched inference.
Slots sharing the same model (via LRU cache) now produce one forward
pass per unique model per ply instead of one per slot per player.

With 20-entry pool and 64 active slots: ~20 forward passes at batch
~12-15 instead of ~128 passes at batch ~2."
```

---

### Task 3: Full regression and integration verification

**Files:**
- Test: all match-related test files

- [ ] **Step 1: Run full Python test suite**

Run: `uv run pytest -x -v`

Expected: ALL PASS

- [ ] **Step 2: Verify no regressions in league tournament tests**

Run: `uv run pytest tests/test_league_tournament.py tests/test_match_pool.py tests/test_concurrent_round.py tests/test_reward_attribution.py tests/test_league_config.py -v`

Expected: ALL PASS

- [ ] **Step 3: Commit any test fixes if needed**

Only if Step 1 or 2 revealed failures that need fixing.
