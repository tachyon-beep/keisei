# Value Head Collapse Remediation — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the PPO training collapse caused by truncation bootstrap bug and value head supervision starvation.

**Architecture:** Five remediations (R1-R5) deployed in two phases. Phase 1: R1 (correctness fix: separate terminated from truncated in GAE) + R4 (LR scheduler monitor fix). Phase 2: R2 (score blend into GAE) + R3 (lambda_score increase) + R5 (entropy annealing) via config only. All code changes are in the Python training loop — the Rust engine is untouched.

**Tech Stack:** Python 3.13, PyTorch, uv for package management. Run tests with `uv run pytest`.

**Spec:** `docs/superpowers/specs/2026-04-04-value-head-collapse-remediation-design.md`

**Reviewer feedback incorporated (v2):** Reordered tasks to put Phase 1 critical path first. Fixed broken tests (self-comparison, weak assertions). Added missing fallback GAE path. Added buffer assertion guard. Enumerated all mechanical test update sites. Added pool purge task. Adjusted success criteria for Adam staleness.

---

## File Map

**Production files modified:**
- `keisei/training/gae.py` — R1: rename `dones` -> `terminated` in all 3 GAE functions
- `keisei/training/katago_ppo.py` — R1/R2/R5: buffer schema, params, select_actions, all 3 GAE call paths in update()
- `keisei/training/katago_loop.py` — R1/R2/R4: PendingTransitions, buffer.add sites, value_adapter plumbing, LR monitor, observability
- `keisei/training/value_adapter.py` — R2: `scalar_value_blended` method, `score_blend_alpha` param
- `keisei-500k-league.toml` — R3/R5: production config values

**Test files modified/created:**
- `tests/test_gae.py` — R1: truncation bootstrap tests
- `tests/test_katago_ppo.py` — R1: buffer terminated field tests (+ 27 existing `buf.add()` calls need `terminated` arg)
- `tests/test_value_adapter.py` — R2: blend tests
- `tests/test_entropy_annealing.py` — R5: new file
- `tests/test_pending_transitions.py` — R1: three-population test (+ 17 existing `finalize()` calls need `terminated` arg)
- `tests/test_pytorch_audit_gaps.py` — 5 `buf.add()` calls need `terminated` arg
- `tests/test_compile.py` — 1 `buf.add()` call needs `terminated` arg
- `tests/test_lr_scheduler.py` — 1 `buf.add()` call needs `terminated` arg
- `tests/test_amp.py` — 1 `buf.add()` call needs `terminated` arg
- `tests/test_pytorch_hot_path_gaps.py` — 2 `buf.add()` calls need `terminated` arg

---

## Task Ordering Rationale

Tasks are ordered to **deploy Phase 1 (R1+R4) as fast as possible**, then Phase 2 (R5, R2, R3) afterward. R4 goes before R1 because the spec requires them together — if R1 lands without R4, the LR scheduler misinterprets the increased value_loss as degradation.

```
Task 0:  Pool purge (pre-deployment housekeeping)
Task 1:  R4 — LR scheduler monitor fix (6 lines, independent)
Task 2:  R1-A — GAE functions: rename dones→terminated
Task 3:  R1-B — Buffer schema: add terminated field
Task 4:  R1-C — PendingTransitions.finalize() threading
Task 5:  R1-D — Fix _compute_value_cats + buffer.add sites + terminal_mask
Task 6:  R1-E — GAE call sites in update() with feature flag (all 3 paths)
Task 7:  Observability logging
Task 8:  Config updates (Phase 1)
Task 9:  R5 — Entropy annealing
Task 10: R2 — Score blend in value adapter
Task 11: R2 — Plumb value_adapter through training loop
Task 12: Config updates (Phase 2) + final regression
```

---

### Task 0: Pool Purge (Pre-Deployment Housekeeping)

**Why:** The opponent pool contains snapshots from the collapse window (epochs 350-600) that play degenerate strategies. Training against these opponents teaches the learner to exploit degeneracy rather than learn general strategy.

- [ ] **Step 1: Identify collapse-era snapshots**

```bash
sqlite3 data/keisei-500k-league.db "
SELECT id, display_name, created_epoch, elo_rating, games_played
FROM league_entries
WHERE created_epoch BETWEEN 350 AND 600
ORDER BY created_epoch
"
```

- [ ] **Step 2: Delete collapse-era entries from the pool**

```bash
# For each ID from Step 1, delete from league_entries and associated elo_history:
sqlite3 data/keisei-500k-league.db "
DELETE FROM elo_history WHERE entry_id IN (
  SELECT id FROM league_entries WHERE created_epoch BETWEEN 350 AND 600
);
DELETE FROM league_entries WHERE created_epoch BETWEEN 350 AND 600;
"
```

Verify: `sqlite3 data/keisei-500k-league.db "SELECT COUNT(*) FROM league_entries WHERE created_epoch BETWEEN 350 AND 600"` → should be 0.

- [ ] **Step 3: Document the purge**

```bash
filigree observe "Purged collapse-era snapshots (epochs 350-600) from opponent pool before Phase 1 deploy. See value-head-collapse-remediation spec."
```

---

### Task 1: R4 — Fix LR Scheduler Monitor (do this BEFORE R1)

**Files:**
- Modify: `keisei/training/katago_loop.py:1015-1032`

**Why first:** R1 will increase value_loss (more valid gradient signal). If R4 is not deployed, the scheduler interprets this as degradation and slashes the learning rate. R4 is 6 lines and fully independent of R1.

- [ ] **Step 1: Fix the monitor key**

In `keisei/training/katago_loop.py`, at line 1016:

```python
# Before:
                monitor_value = losses.get("value_loss")
                if monitor_value is not None:
# After:
                monitor_value = losses.get("policy_loss")
                if monitor_value is None:
                    raise RuntimeError(
                        "LR scheduler expects 'policy_loss' in losses dict but it was absent. "
                        "Check that ppo.update() returns 'policy_loss'."
                    )
```

Update the log message at line 1031:
```python
                        logger.info("LR reduced: %.6f -> %.6f (monitor=policy_loss, value=%.4f)",
                                    old_lr, new_lr, monitor_value)
```

- [ ] **Step 2: Run test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "fix(R4): LR scheduler monitors policy_loss instead of value_loss

value_loss trends to 0 during collapse (W/D/L head starvation), which
the scheduler misinterprets as improvement. policy_loss is a more
reliable training health signal. Added defensive assertion if key absent."
```

---

### Task 2: R1 Part A — GAE functions: rename `dones` to `terminated`

**Files:**
- Modify: `keisei/training/gae.py`
- Modify: `tests/test_gae.py`

- [ ] **Step 1: Write tests for GAE truncation bootstrap**

Append to `tests/test_gae.py`:

```python
class TestGAETruncationBootstrap:
    """R1: Verify that truncated episodes bootstrap V(s_next) instead of zeroing."""

    def test_truncated_bootstraps_value(self):
        """With terminated-only signal, truncated positions should bootstrap."""
        # 3 steps, 1 env. Step 1 is truncated (not terminated).
        rewards = torch.tensor([[0.0], [0.0], [1.0]])   # (T=3, N=1)
        values = torch.tensor([[0.5], [0.6], [0.7]])
        next_value = torch.tensor([0.8])

        # Old behavior: dones = terminated | truncated. Step 1 truncated -> dones=True
        old_dones = torch.tensor([[0.0], [1.0], [0.0]])
        adv_old = compute_gae_gpu(rewards, values, old_dones, next_value, gamma=0.99, lam=0.95)

        # New behavior: only truly terminated. Step 1 NOT terminated.
        terminated = torch.tensor([[0.0], [0.0], [0.0]])
        adv_new = compute_gae_gpu(rewards, values, terminated, next_value, gamma=0.99, lam=0.95)

        # Old: delta[1] = 0 + 0.99 * 0.7 * 0 - 0.6 = -0.6 (bootstrap zeroed)
        # New: delta[1] = 0 + 0.99 * 0.7 * 1 - 0.6 = 0.093
        assert abs(adv_old[1, 0].item() - (-0.6)) < 0.15
        assert abs(adv_new[1, 0].item() - 0.093) < 0.1
        # The key: they are substantially different
        assert abs(adv_new[1, 0].item() - adv_old[1, 0].item()) > 0.5

    def test_backward_compat_no_truncation(self):
        """When no truncation occurs, terminated == dones gives identical results."""
        rewards = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
        values = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.7, 0.2]])
        next_value = torch.tensor([0.1, 0.5])

        # Case A: some terminated positions, no truncation
        terminated = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        adv_a = compute_gae_gpu(rewards, values, terminated, next_value,
                                gamma=0.99, lam=0.95)

        # Case B: same signal but named dones (simulating pre-fix: terminated==dones)
        dones_same_as_terminated = terminated.clone()
        adv_b = compute_gae_gpu(rewards, values, dones_same_as_terminated, next_value,
                                gamma=0.99, lam=0.95)

        assert torch.allclose(adv_a, adv_b)

    def test_truncation_vs_terminal_differ(self):
        """Passing dones (merged) vs terminated (split) must give different results."""
        rewards = torch.tensor([[0.0], [0.0]])
        values = torch.tensor([[0.5], [0.5]])
        next_value = torch.tensor([0.8])

        # Terminated only (step 0 is truncated, not terminated)
        terminated = torch.tensor([[0.0], [0.0]])
        adv_terminated = compute_gae_gpu(rewards, values, terminated, next_value,
                                         gamma=0.99, lam=0.95)

        # Merged dones (step 0 is "done" from truncation)
        dones_merged = torch.tensor([[1.0], [0.0]])
        adv_dones = compute_gae_gpu(rewards, values, dones_merged, next_value,
                                     gamma=0.99, lam=0.95)

        # They MUST differ — this is the whole point of R1
        assert not torch.allclose(adv_terminated, adv_dones)


class TestGAEPaddedTruncation:
    """R1: Truncation bootstrap for the padded GAE path (split-merge mode)."""

    def test_padded_truncated_bootstraps(self):
        from keisei.training.gae import compute_gae_padded
        # 2 envs, max_T=3. Env 0 has length 2 (not terminated). Env 1 has length 3.
        rewards = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5]])
        values = torch.tensor([[0.5, 0.3], [0.4, 0.6], [0.0, 0.2]])
        # Env 0: no termination in valid range. Padding at step 2 marked terminated=1.
        # Env 1: no termination.
        terminated_pad = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
        next_values = torch.tensor([0.3, 0.8])
        lengths = torch.tensor([2, 3])

        adv = compute_gae_padded(rewards, values, terminated_pad, next_values, lengths,
                                 gamma=0.99, lam=0.95)
        # Env 0, step 1 (last valid): bootstraps from next_values[0]=0.3
        # delta = 0.0 + 0.99 * 0.3 - 0.4 = -0.103
        assert abs(adv[1, 0].item() - (-0.103)) < 0.05
```

- [ ] **Step 2: Run tests to verify they pass (GAE already accepts positional args)**

Run: `uv run pytest tests/test_gae.py::TestGAETruncationBootstrap tests/test_gae.py::TestGAEPaddedTruncation -v`
Expected: PASS — these tests pass tensors positionally; the rename hasn't happened yet but the math works the same.

- [ ] **Step 3: Rename `dones` to `terminated` in all GAE functions**

In `keisei/training/gae.py`:

For all three functions (`compute_gae`, `compute_gae_padded`, `compute_gae_gpu`):
- Rename parameter `dones` → `terminated`
- Update `not_done = 1.0 - dones[t].float()` → `not_done = 1.0 - terminated[t].float()` (or non-indexed equivalent for `compute_gae_gpu`)
- Update `not_done = 1.0 - dones.float()` → `not_done = 1.0 - terminated.float()` in `compute_gae_gpu`
- Update docstrings: `terminated: True only for genuinely terminal episodes (not truncated). Truncated episodes bootstrap from V(s_next) instead of zeroing it.`

- [ ] **Step 4: Update existing tests that use `dones=` keyword arg**

In `tests/test_gae.py`, search for `dones=` keyword arguments and rename to `terminated=`. Positional calls don't need changes.

- [ ] **Step 5: Run all GAE tests**

Run: `uv run pytest tests/test_gae.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/gae.py tests/test_gae.py
git commit -m "refactor(R1): rename dones->terminated in GAE functions

Semantic rename to clarify that GAE's not_done mask should only reflect
truly terminal episodes, not truncated ones. Truncated episodes now
bootstrap V(s_next) instead of zeroing it. Tests demonstrate old vs new
behavior differs for truncated positions."
```

---

### Task 3: R1 Part B — Buffer schema: add `terminated` field

**Files:**
- Modify: `keisei/training/katago_ppo.py:54-70` (KataGoPPOParams), `82-202` (KataGoRolloutBuffer)
- Modify: 7 test files (37 `buf.add()` call sites total)

- [ ] **Step 1: Write failing test for buffer terminated field**

Append to `tests/test_katago_ppo.py`:

```python
class TestBufferTerminatedField:
    def test_add_stores_terminated(self):
        """Buffer should store terminated separately from dones."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(4, 9, 9), action_space=100)
        obs = torch.zeros(2, 4, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        rewards = torch.zeros(2)
        dones = torch.tensor([1.0, 1.0])
        terminated = torch.tensor([1.0, 0.0])
        legal_masks = torch.zeros(2, 100, dtype=torch.bool)
        legal_masks[:, 0] = True
        value_cats = torch.tensor([-1, -1], dtype=torch.long)
        score_targets = torch.tensor([0.1, -0.1])

        buf.add(obs, actions, log_probs, values, rewards, dones,
                terminated, legal_masks, value_cats, score_targets)

        data = buf.flatten()
        assert "terminated" in data
        assert data["terminated"][0].item() == 1.0
        assert data["terminated"][1].item() == 0.0
        assert "dones" in data
        assert data["dones"][0].item() == 1.0
        assert data["dones"][1].item() == 1.0

    def test_terminated_must_be_subset_of_dones(self):
        """Buffer should reject terminated entries that aren't in dones."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(4, 9, 9), action_space=100)
        obs = torch.zeros(2, 4, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        rewards = torch.zeros(2)
        dones = torch.tensor([0.0, 0.0])       # neither done
        terminated = torch.tensor([1.0, 0.0])   # but env 0 claims terminated!
        legal_masks = torch.zeros(2, 100, dtype=torch.bool)
        legal_masks[:, 0] = True
        value_cats = torch.tensor([-1, -1], dtype=torch.long)
        score_targets = torch.tensor([0.1, -0.1])

        with pytest.raises(AssertionError, match="terminated must be a subset of dones"):
            buf.add(obs, actions, log_probs, values, rewards, dones,
                    terminated, legal_masks, value_cats, score_targets)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_ppo.py::TestBufferTerminatedField -v`
Expected: FAIL — `buffer.add()` does not accept `terminated` parameter.

- [ ] **Step 3: Add params and buffer changes**

In `keisei/training/katago_ppo.py`:

Add to `KataGoPPOParams` (after `compile_dynamic` field, line 70):
```python
    entropy_decay_epochs: int = 0    # R5: 0 = instant transition; >0 = linear decay
    score_blend_alpha: float = 0.0   # R2: 0.0 = pure WDL; >0 blends score_lead into GAE value
    use_terminated_for_gae: bool = False  # R1: default False until buffer has terminated field in all paths
```

**Important:** `use_terminated_for_gae` defaults to `False` here. Production TOML sets it to `True`. This prevents `KeyError` on `data["terminated"]` if any test constructs `KataGoPPOParams()` with defaults before all buffer call sites are updated.

In `clear()` (line 89), add after `self.dones`:
```python
        self.terminated: list[torch.Tensor] = []
```

In `add()` (line 105), add `terminated: torch.Tensor` after `dones` in signature. After `dones_cpu = dones.detach().cpu()`, add:
```python
        terminated_cpu = terminated.detach().cpu()

        # Guard: terminated must be a subset of dones (can't be terminated without being done)
        if not (terminated_cpu <= dones_cpu + 1e-6).all():
            raise AssertionError(
                "terminated must be a subset of dones: every terminated position must also be done. "
                "Got terminated=True where dones=False — likely a call site passing the merged signal."
            )
```

After `self.dones.append(dones_cpu)`, add:
```python
        self.terminated.append(terminated_cpu)
```

In `flatten()` (line 195), add after `"dones"` line:
```python
            "terminated": torch.cat(self.terminated, dim=0).reshape(-1),
```

- [ ] **Step 4: Fix ALL existing `buffer.add()` calls in test files**

This is a mechanical change: every existing `buf.add()` call needs `terminated` inserted after `dones`. For existing tests where `terminated == dones`, pass the same tensor for both.

**Exact call sites to update (37 total):**

| File | Line numbers | Count |
|------|-------------|-------|
| `tests/test_katago_ppo.py` | 52, 59, 78, 92, 188, 207, 224, 239, 261, 284, 348, 378, 391, 403, 422, 484, 510, 529, 552, 572, 590, 612, 620, 633, 719, 740, 808 | 27 |
| `tests/test_pytorch_audit_gaps.py` | 38, 263, 395, 447, 477 | 5 |
| `tests/test_compile.py` | 38 | 1 |
| `tests/test_lr_scheduler.py` | 136 | 1 |
| `tests/test_amp.py` | 106 | 1 |
| `tests/test_pytorch_hot_path_gaps.py` | 38, 346 | 2 |

For each call, find the `dones` argument (positional, after `rewards`) and add the same value as `terminated` after it. Example:
```python
# Before:
buf.add(obs, actions, log_probs, values, rewards, dones, legal_masks, ...)
# After:
buf.add(obs, actions, log_probs, values, rewards, dones, dones, legal_masks, ...)
#                                                        ^^^^^ terminated=dones
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py tests/test_pytorch_audit_gaps.py tests/test_compile.py tests/test_lr_scheduler.py tests/test_amp.py tests/test_pytorch_hot_path_gaps.py -x -q`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add keisei/training/katago_ppo.py tests/
git commit -m "feat(R1): add terminated field to KataGoRolloutBuffer

Buffer stores terminated separately from dones. Includes assertion guard
that terminated <= dones to catch call-site wiring bugs. All 37 existing
test call sites updated with terminated=dones for backward compat.
Added use_terminated_for_gae (default False) and score_blend_alpha to params."
```

---

### Task 4: R1 Part C — PendingTransitions.finalize() threading

**Files:**
- Modify: `keisei/training/katago_loop.py:198-235`
- Modify: `tests/test_pending_transitions.py` (17 existing `finalize()` calls + new test)

- [ ] **Step 1: Write test for finalize with three populations**

Append to `tests/test_pending_transitions.py`:

```python
class TestPendingTransitionsTerminated:
    """R1: finalize() must return terminated separately from dones."""

    def test_finalize_three_populations(self):
        from keisei.training.katago_loop import PendingTransitions
        device = torch.device("cpu")
        pt = PendingTransitions(num_envs=3, obs_shape=(4, 9, 9), action_space=100, device=device)

        env_mask = torch.ones(3, dtype=torch.bool, device=device)
        pt.create(
            env_mask,
            obs=torch.zeros(3, 4, 9, 9, device=device),
            actions=torch.zeros(3, dtype=torch.long, device=device),
            log_probs=torch.zeros(3, device=device),
            values=torch.zeros(3, device=device),
            legal_masks=torch.zeros(3, 100, dtype=torch.bool, device=device),
            rewards=torch.zeros(3, device=device),
            score_targets=torch.zeros(3, device=device),
        )

        # A: terminated (game ended), B: truncated (max_ply), C: epoch flush
        dones = torch.tensor([1.0, 1.0, 0.0], device=device)
        terminated = torch.tensor([1.0, 0.0, 0.0], device=device)

        finalize_mask = torch.ones(3, dtype=torch.bool, device=device)
        result = pt.finalize(finalize_mask, dones, terminated)

        assert result is not None
        assert "terminated" in result
        assert result["terminated"][0].item() == 1.0  # A: terminated
        assert result["terminated"][1].item() == 0.0  # B: truncated
        assert result["terminated"][2].item() == 0.0  # C: flush
        assert result["dones"][0].item() == 1.0
        assert result["dones"][1].item() == 1.0
        assert result["dones"][2].item() == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pending_transitions.py::TestPendingTransitionsTerminated -v`
Expected: FAIL — `finalize()` does not accept `terminated` parameter.

- [ ] **Step 3: Update `finalize()` signature and result dict**

In `keisei/training/katago_loop.py:198`:

```python
    def finalize(
        self,
        finalize_mask: torch.Tensor,
        dones: torch.Tensor,
        terminated: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
```

In the result dict (line 219), add `"terminated"`:
```python
            "terminated": terminated[indices].float(),
```

- [ ] **Step 4: Fix ALL existing `finalize()` calls in tests**

**17 call sites in `tests/test_pending_transitions.py`:**
Lines: 191, 219, 228, 238, 278, 309, 351, 362, 394, 419, 451, 477, 522, 582, 607, 651, 676.

For each, add `terminated=dones` (same tensor) as the third argument:
```python
# Before:
result = pt.finalize(finalize_mask, dones)
# After:
result = pt.finalize(finalize_mask, dones, dones)
```

- [ ] **Step 5: Fix production `finalize()` call sites**

In `keisei/training/katago_loop.py`, update all 3 calls:

Line 833: `finalized = pending.finalize(finalize_mask, dones, terminated)`
Line 872: `imm_finalized = pending.finalize(imm_terminal, dones, terminated)`
Line 951: Add `remaining_terminated = torch.zeros(self.num_envs, device=self.device)` and pass: `remaining = pending.finalize(remaining_mask, remaining_dones, remaining_terminated)`

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_pending_transitions.py -v`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_pending_transitions.py
git commit -m "feat(R1): thread terminated through PendingTransitions.finalize()

finalize() now accepts and returns terminated alongside dones. Three
production call sites and 17 test call sites updated."
```

---

### Task 5: R1 Part D — Fix `_compute_value_cats`, `buffer.add()`, `terminal_mask`

**Files:**
- Modify: `keisei/training/katago_loop.py` (lines 797, 836, 875, 901, 925, 839-845, 878-885, 934-937, 957-964)

- [ ] **Step 1: Write test proving `_compute_value_cats` wrong behavior under dones**

Add to `tests/test_pending_transitions.py` (which already imports `_compute_value_cats`):

```python
class TestComputeValueCatsTerminated:
    """R1: _compute_value_cats must use terminated, not dones."""

    def test_truncated_gets_draw_under_dones_but_ignore_under_terminated(self):
        from keisei.training.katago_loop import _compute_value_cats
        device = torch.device("cpu")
        rewards = torch.tensor([0.0, 1.0, 0.0])

        # With merged dones: truncated game (reward=0) gets labeled as draw
        dones_merged = torch.tensor([True, True, False])
        cats_wrong = _compute_value_cats(rewards, dones_merged, device)
        assert cats_wrong[0].item() == 1  # WRONG: labeled as draw

        # With terminated only: truncated game (not terminated) gets ignored
        terminated = torch.tensor([False, True, False])
        cats_correct = _compute_value_cats(rewards, terminated, device)
        assert cats_correct[0].item() == -1  # CORRECT: ignored
        assert cats_correct[1].item() == 0   # win
        assert cats_correct[2].item() == -1  # still playing
```

- [ ] **Step 2: Run test — it should PASS (proves the bug exists and the fix works)**

Run: `uv run pytest tests/test_pending_transitions.py::TestComputeValueCatsTerminated -v`
Expected: PASS.

- [ ] **Step 3: Fix `_compute_value_cats` call sites**

In `keisei/training/katago_loop.py`:

Line 836-837 (split-merge finalize):
```python
f_value_cats = _compute_value_cats(
    finalized["rewards"], finalized["terminated"].bool(), self.device,
)
```

Line 874-875 (immediate terminal):
```python
imm_value_cats = _compute_value_cats(
    imm_finalized["rewards"], imm_finalized["terminated"].bool(), self.device,
)
```

Line 925 (non-split-merge) — also fix `terminal_mask`:
```python
                    terminal_mask = terminated.bool()
                    ...
                    value_cats = _compute_value_cats(rewards, terminal_mask, self.device)
```

- [ ] **Step 4: Fix `terminal_mask` for win/loss/draw counting**

Line 797 (split-merge): `terminal_mask = terminated.bool()` (was `dones.bool()`)
Line 901 (non-split-merge): `terminal_mask = terminated.bool()` (was `dones.bool()`)

- [ ] **Step 5: Fix all `buffer.add()` call sites to pass `terminated`**

Line 839-846 (split-merge finalize): Insert `finalized["terminated"],` after `finalized["dones"],`
Line 878-885 (immediate terminal): Insert `imm_finalized["terminated"],` after `imm_finalized["dones"],`
Line 934-937 (non-split-merge): Insert `terminated,` after `dones,`
Line 957-964 (epoch-end flush): Insert `remaining["terminated"],` after `remaining["dones"],`

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_pending_transitions.py
git commit -m "feat(R1): fix _compute_value_cats and buffer.add call sites

All _compute_value_cats calls now use terminated (not dones), preventing
truncated games from being mislabeled as draws. All buffer.add() calls
pass terminated. terminal_mask for win/loss/draw counting uses terminated."
```

---

### Task 6: R1 Part E — GAE call sites in `update()` with feature flag (ALL 3 PATHS)

**Files:**
- Modify: `keisei/training/katago_ppo.py:442-529` (all 3 GAE paths in update)

- [ ] **Step 1: Update all 3 GAE call paths with feature flag**

In `keisei/training/katago_ppo.py`, in `update()`:

After `data = buffer.flatten()` (line 433), add:
```python
        gae_dones_key = "terminated" if self.params.use_terminated_for_gae else "dones"
```

**Path 1 — vectorized (line 446):**
```python
            terminated_2d = data[gae_dones_key].reshape(T, N)
```
Pass `terminated_2d` to both `compute_gae_gpu` and `compute_gae`.

**Path 2 — split-merge padded (lines 488-522):**
Rename `env_dones` → `env_terminated`, `dones_pad` → `terminated_pad` throughout:
```python
            env_terminated = []
            ...
                env_terminated.append(data[gae_dones_key][mask])
            ...
            terminated_pad = torch.ones(max_T, N_env)
            ...
                terminated_pad[:length, j] = env_terminated[j]
```
Pass `terminated_pad` to `compute_gae_padded`.

**Path 3 — fallback flat (lines 524-529) — THIS PATH WAS MISSING IN v1:**
```python
            # Before:
            advantages = compute_gae(
                data["rewards"], data["values"], data["dones"],
                bootstrap, gamma=self.params.gamma, lam=self.params.gae_lambda,
            )
            # After:
            advantages = compute_gae(
                data["rewards"], data["values"], data[gae_dones_key],
                bootstrap, gamma=self.params.gamma, lam=self.params.gae_lambda,
            )
```

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add keisei/training/katago_ppo.py
git commit -m "feat(R1): use terminated for GAE in all 3 update() paths

Feature-flagged via use_terminated_for_gae. Vectorized, padded, AND
fallback flat GAE paths all use gae_dones_key. Truncated episodes
correctly bootstrap V(s_next)."
```

---

### Task 7: Observability — Logging additions

**Files:**
- Modify: `keisei/training/katago_loop.py`

- [ ] **Step 1: Add terminated/truncated counting**

Near the top of the epoch loop, initialize:
```python
            terminated_count = 0
            truncated_count = 0
```

In the rollout loop where `terminated` and `truncated` tensors are available (around line 791 for split-merge, 900 for non-split-merge):
```python
                    terminated_count += terminated.bool().sum().item()
                    truncated_count += (truncated.bool() & ~terminated.bool()).sum().item()
```

After the rollout loop, before update:
```python
            logger.info(
                "Epoch %d: %d terminated, %d truncated (bootstrapped)",
                epoch_i, terminated_count, truncated_count,
            )
```

- [ ] **Step 2: Log entropy coefficient every epoch**

After `self.ppo.current_entropy_coeff = ...` (line 986):
```python
            logger.info("Epoch %d: entropy_coeff=%.4f", epoch_i, self.ppo.current_entropy_coeff)
```

- [ ] **Step 3: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "feat: add observability logging for R1 monitoring

Per-epoch terminated/truncated counts and entropy coefficient."
```

---

### Task 8: Config Updates (Phase 1)

**Files:**
- Modify: `keisei-500k-league.toml`

- [ ] **Step 1: Update production config**

In `keisei-500k-league.toml`, in `[training.algorithm_params]`:
```toml
use_terminated_for_gae = true
```

- [ ] **Step 2: Run test suite one more time**

Run: `uv run pytest tests/ -x -q`

- [ ] **Step 3: Commit**

```bash
git add keisei-500k-league.toml
git commit -m "config: enable use_terminated_for_gae for Phase 1 deploy"
```

**Phase 1 is now complete. Before enabling Phase 2, run 50-100 epochs and verify:**
- `value_loss > 0.05` by epoch 30-50 (not immediately — Adam momentum needs ~20 epochs to adapt after ~200 epochs of near-zero gradients)
- `advantages.std() > 0.01`
- Truncation count is logged and non-zero

---

### Task 9: R5 — Entropy Annealing

**Files:**
- Modify: `keisei/training/katago_ppo.py:309-317`
- Create: `tests/test_entropy_annealing.py`

- [ ] **Step 1: Write tests**

Create `tests/test_entropy_annealing.py`:

```python
"""Tests for smooth entropy annealing (R5)."""

from keisei.training.katago_ppo import KataGoPPOAlgorithm, KataGoPPOParams
from unittest.mock import MagicMock


def _make_algo(lambda_entropy=0.01, entropy_decay_epochs=0, warmup_epochs=10, warmup_entropy=0.05):
    params = KataGoPPOParams(lambda_entropy=lambda_entropy, entropy_decay_epochs=entropy_decay_epochs)
    model = MagicMock()
    model.parameters.return_value = iter([])
    return KataGoPPOAlgorithm(params, model, warmup_epochs=warmup_epochs, warmup_entropy=warmup_entropy)


class TestEntropyAnnealing:
    def test_decay_zero_matches_step_behavior(self):
        algo = _make_algo(entropy_decay_epochs=0, warmup_epochs=10)
        assert algo.get_entropy_coeff(9) == 0.05
        assert algo.get_entropy_coeff(10) == 0.01
        assert algo.get_entropy_coeff(100) == 0.01

    def test_linear_decay_at_warmup_boundary(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(10) == 0.05

    def test_linear_decay_midpoint(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10,
                         warmup_entropy=0.05, lambda_entropy=0.01)
        result = algo.get_entropy_coeff(110)
        expected = 0.05 + 0.5 * (0.01 - 0.05)
        assert abs(result - expected) < 1e-9

    def test_linear_decay_one_before_end(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        result = algo.get_entropy_coeff(209)
        assert result > 0.01
        assert result < 0.05

    def test_linear_decay_at_end(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(210) == 0.01

    def test_linear_decay_past_end(self):
        algo = _make_algo(entropy_decay_epochs=200, warmup_epochs=10)
        assert algo.get_entropy_coeff(500) == 0.01
```

- [ ] **Step 2: Implement `get_entropy_coeff()` with linear decay**

Replace `get_entropy_coeff` in `keisei/training/katago_ppo.py` (lines 309-317):

```python
    def get_entropy_coeff(self, epoch: int) -> float:
        """Return the entropy coefficient for the current epoch.

        During warmup: elevated entropy. After warmup: linear decay from
        warmup_entropy to lambda_entropy over entropy_decay_epochs
        (0 = instant transition, matching legacy behavior).
        """
        if epoch < self.warmup_epochs:
            return self.warmup_entropy
        decay_epochs = self.params.entropy_decay_epochs
        if decay_epochs <= 0:
            return self.params.lambda_entropy
        elapsed = epoch - self.warmup_epochs
        if elapsed >= decay_epochs:
            return self.params.lambda_entropy
        t = elapsed / decay_epochs
        return self.warmup_entropy + t * (self.params.lambda_entropy - self.warmup_entropy)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_entropy_annealing.py -v`
Expected: All 6 PASS.

- [ ] **Step 4: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_entropy_annealing.py
git commit -m "feat(R5): smooth entropy annealing with linear decay

entropy_decay_epochs=0 preserves existing step-function behavior.
>0 linearly interpolates from warmup_entropy to lambda_entropy."
```

---

### Task 10: R2 — Score Blend in Value Adapter

**Files:**
- Modify: `keisei/training/value_adapter.py`
- Modify: `tests/test_value_adapter.py`

- [ ] **Step 1: Write tests for `scalar_value_blended`**

Append to `tests/test_value_adapter.py`:

```python
class TestScalarValueBlended:
    def test_alpha_zero_matches_wdl_only(self):
        adapter = MultiHeadValueAdapter(score_blend_alpha=0.0)
        value_logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
        score_lead = torch.tensor([[0.5], [-0.3]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        wdl_only = adapter.scalar_value_from_output(value_logits)
        assert torch.allclose(blended, wdl_only)

    def test_alpha_one_uses_score_only(self):
        adapter = MultiHeadValueAdapter(score_blend_alpha=1.0)
        value_logits = torch.tensor([[2.0, 0.0, 0.0]])
        score_lead = torch.tensor([[0.7]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        assert torch.allclose(blended, torch.tensor([0.7]))

    def test_alpha_half_is_arithmetic_mean(self):
        adapter = MultiHeadValueAdapter(score_blend_alpha=0.5)
        value_logits = torch.tensor([[10.0, 0.0, 0.0]])
        score_lead = torch.tensor([[0.0]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        wdl_value = adapter.scalar_value_from_output(value_logits)
        expected = 0.5 * wdl_value + 0.5 * 0.0
        assert torch.allclose(blended, expected, atol=1e-5)

    def test_extreme_score_clamped(self):
        adapter = MultiHeadValueAdapter(score_blend_alpha=1.0)
        value_logits = torch.tensor([[0.0, 0.0, 0.0]])
        score_lead = torch.tensor([[5.0]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        assert blended.item() == pytest.approx(1.0)

    def test_negative_extreme_clamped(self):
        adapter = MultiHeadValueAdapter(score_blend_alpha=1.0)
        value_logits = torch.tensor([[0.0, 0.0, 0.0]])
        score_lead = torch.tensor([[-5.0]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        assert blended.item() == pytest.approx(-1.0)

    def test_scalar_adapter_inherits_default(self):
        adapter = ScalarValueAdapter()
        value_logits = torch.tensor([[0.5], [-0.3]])
        score_lead = torch.tensor([[0.9], [0.1]])
        blended = adapter.scalar_value_blended(value_logits, score_lead)
        expected = adapter.scalar_value_from_output(value_logits)
        assert torch.allclose(blended, expected)

    def test_get_value_adapter_passes_alpha(self):
        adapter = get_value_adapter("multi_head", score_blend_alpha=0.3)
        assert adapter.score_blend_alpha == 0.3
```

- [ ] **Step 2: Implement `scalar_value_blended`**

In `keisei/training/value_adapter.py`:

Add concrete default on `ValueHeadAdapter` ABC:
```python
    def scalar_value_blended(
        self, value_logits: torch.Tensor, score_lead: torch.Tensor,
    ) -> torch.Tensor:
        """Blend W/D/L value with score for GAE. Default: ignore score_lead."""
        return self.scalar_value_from_output(value_logits)
```

Add `score_blend_alpha` to `MultiHeadValueAdapter.__init__`:
```python
    def __init__(self, lambda_value: float = 1.5, lambda_score: float = 0.02,
                 score_blend_alpha: float = 0.0) -> None:
        self.lambda_value = lambda_value
        self.lambda_score = lambda_score
        self.score_blend_alpha = score_blend_alpha
```

Override `scalar_value_blended` on `MultiHeadValueAdapter`:
```python
    def scalar_value_blended(
        self, value_logits: torch.Tensor, score_lead: torch.Tensor,
    ) -> torch.Tensor:
        """Blend W/D/L value with score prediction for GAE.

        score_lead: (batch, 1) squeezed to (batch,), clamped to [-1, 1].
        """
        wdl_value = self.scalar_value_from_output(value_logits)
        score_value = score_lead.squeeze(-1).clamp(-1, 1)
        alpha = self.score_blend_alpha
        if alpha == 0.0:
            return wdl_value
        return (1 - alpha) * wdl_value + alpha * score_value
```

Add `score_blend_alpha` to `get_value_adapter()`.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_value_adapter.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add keisei/training/value_adapter.py tests/test_value_adapter.py
git commit -m "feat(R2): add scalar_value_blended to value adapters

Concrete default on ABC ignores score_lead. MultiHeadValueAdapter
overrides with configurable alpha blend. score_blend_alpha=0.0 is no-op."
```

---

### Task 11: R2 — Plumb `value_adapter` Through Training Loop

**Files:**
- Modify: `keisei/training/katago_loop.py` (init, split_merge_step call, bootstrap, update call)
- Modify: `keisei/training/katago_ppo.py:345-404` (select_actions)

- [ ] **Step 1: Store `value_adapter` on KataGoTrainingLoop**

In `__init__()`, after `self.ppo = KataGoPPOAlgorithm(...)` (around line 445):

```python
        from keisei.training.value_adapter import get_value_adapter
        # Derive model contract from architecture — se_resnet is multi_head, others are scalar
        _model_contract = "multi_head" if config.model.architecture in _KATAGO_ARCHITECTURES else "scalar"
        self.value_adapter = get_value_adapter(
            model_contract=_model_contract,
            lambda_value=ppo_params.lambda_value,
            lambda_score=ppo_params.lambda_score,
            score_blend_alpha=ppo_params.score_blend_alpha,
        )
```

Note: `_KATAGO_ARCHITECTURES` is already defined at line 381 as `{"se_resnet"}`.

- [ ] **Step 2: Pass to `split_merge_step()`**

Line 773: Add `value_adapter=self.value_adapter` to the call.

Inside `split_merge_step()` line 284: Change to `value_adapter.scalar_value_blended(l_output.value_logits, l_output.score_lead)`.

- [ ] **Step 3: Add `value_adapter` to `select_actions()`**

In `keisei/training/katago_ppo.py:345`, add parameter:
```python
    def select_actions(self, obs, legal_masks, value_adapter=None):
```

At line 401-402:
```python
            if value_adapter is not None:
                scalar_values = value_adapter.scalar_value_blended(output.value_logits, output.score_lead)
            else:
                scalar_values = self.scalar_value(output.value_logits)
```

Update the non-split-merge call at line 888:
```python
                    actions, log_probs, values = self.ppo.select_actions(
                        obs, legal_masks, value_adapter=self.value_adapter,
                    )
```

- [ ] **Step 4: Use blended value for bootstrap**

At line 974:
```python
                next_values = self.value_adapter.scalar_value_blended(
                    output.value_logits, output.score_lead,
                )
```

- [ ] **Step 5: Pass to `ppo.update()`**

At line 995:
```python
            losses = self.ppo.update(
                self.buffer, next_values,
                value_adapter=self.value_adapter,
                heartbeat_fn=self._maybe_update_heartbeat,
            )
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add keisei/training/katago_loop.py keisei/training/katago_ppo.py
git commit -m "feat(R2): plumb value_adapter through training loop

Store adapter on KataGoTrainingLoop, derive model_contract from config.
Pass to split_merge_step, select_actions, bootstrap, update. All value
projections use scalar_value_blended (no-op at alpha=0.0)."
```

---

### Task 12: Config Updates (Phase 2) + Final Regression

**Files:**
- Modify: `keisei-500k-league.toml`

- [ ] **Step 1: Add Phase 2 config values (commented out)**

In `keisei-500k-league.toml`, add a comment block in `[training.algorithm_params]`:
```toml
# Phase 2 values — see docs/superpowers/specs/2026-04-04-value-head-collapse-remediation-design.md
# Enable after Phase 1 validation (50-100 epochs with value_loss > 0.05):
# score_blend_alpha = 0.3     # R2: blend score_lead into GAE value
# lambda_score = 0.1          # R3: 5x increase in score head gradient weight
# entropy_decay_epochs = 200  # R5: linear decay from warmup to base entropy
```

Add warmup config if not present:
```toml
[training.algorithm_params.rl_warmup]
epochs = 50
entropy_bonus = 0.05
```

- [ ] **Step 2: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -40
```
Expected: All PASS.

- [ ] **Step 3: Verify no stale `dones` references in GAE/value_cats paths**

```bash
# Should only appear in the feature-flag fallback and buffer storage:
grep -n 'data\["dones"\]' keisei/training/katago_ppo.py

# Should be zero matches — all call sites use terminated:
grep -n 'finalized\["dones"\]\.bool()' keisei/training/katago_loop.py
```

- [ ] **Step 4: Commit**

```bash
git add keisei-500k-league.toml
git commit -m "config: Phase 2 values documented, warmup config explicit

Phase 2 params commented out pending Phase 1 validation. See spec for
deployment criteria and success metrics."
```
