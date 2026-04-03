# Fix Split-Merge Reward Collection Bugs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two critical bugs in split-merge league mode where (1) opponent-turn terminal rewards are silently dropped from the PPO buffer, and (2) bootstrap values lack sign correction for opponent-to-move states.

**Architecture:** Replace the current "store only learner-turn transitions" approach with a pending-transition pattern: hold each learner transition open until the outcome resolves (possibly on an opponent turn), then finalize with correct perspective-corrected rewards and done flags. Add sign correction to bootstrap values for opponent-to-move states. A private `_negate_where()` helper implements the shared negation-by-mask pattern used by both `to_learner_perspective()` and `sign_correct_bootstrap()`. A `PendingTransitions` class encapsulates per-env pending state with debug assertions.

**Tech Stack:** Python 3.13, PyTorch, numpy, pytest

**Bug report:** `docs/bugs/generated/training/katago_loop.py.md`

**Reviewed by:** Architecture, Systems Thinking, Python Engineering, Quality Engineering, PyTorch Engineering (2026-04-03). All blocking items resolved in this revision.

---

## File Map

- **Modify:** `keisei/training/katago_loop.py` — Add `_negate_where()`, `to_learner_perspective()`, `sign_correct_bootstrap()`, `PendingTransitions` class. Rewrite the split-merge branch of `run()` to use pending transitions. Fix win/loss/draw tracking perspective. Add logging for observability.
- **Create:** `tests/test_pending_transitions.py` — Unit tests for helpers and `PendingTransitions`, protocol-level integration tests with multi-env heterogeneous scenarios, epoch-boundary flush test.

## Reviewer-Driven Design Decisions

These decisions were validated during the review round and should NOT be revisited during implementation:

1. **GPU placement for `PendingTransitions`:** Stays on `self.device` (GPU). The ~90MB overhead is acceptable — the user confirmed memory is not a constraint. This avoids device-mismatch complexity in finalize masks.
2. **`_negate_where` shared helper:** Both `to_learner_perspective` and `sign_correct_bootstrap` use identical clone-mask-negate logic. A private `_negate_where` prevents drift.
3. **`PendingTransitions.create()` asserts no overwrite:** An `assert not (env_mask & self.valid).any()` guard catches protocol violations in development. In Shogi, turns strictly alternate, so this should never fire — but a silent overwrite would be a debugging nightmare.
4. **Finalize return type:** Documented key contract in docstring (not a TypedDict — the class is internal and the dict unpacks directly into `buffer.add()`).
5. **`learner_side` scope:** Defined at epoch level (before the `for step_i` loop), not inside the per-step `if` block. This prevents scope ambiguity at the bootstrap call site.
6. **Logging:** Debug-level log on pending finalizations and epoch-end flush count for production observability.
7. **LR scheduler reset:** Required when applying fix to a checkpoint trained with buggy code, to prevent value-loss spike from triggering premature LR reduction.

---

### Task 1: Add perspective-correction helpers with tests

**Files:**
- Modify: `keisei/training/katago_loop.py` (add functions near line 63, before `split_merge_step`)
- Create: `tests/test_pending_transitions.py`

Three functions: a private `_negate_where()` implementing the shared clone-mask-negate pattern, and two public wrappers `to_learner_perspective()` and `sign_correct_bootstrap()`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pending_transitions.py`:

```python
"""Tests for split-merge pending transition logic."""

import numpy as np
import pytest
import torch

from keisei.training.katago_loop import to_learner_perspective, sign_correct_bootstrap


class TestToLearnerPerspective:
    def test_learner_move_reward_unchanged(self):
        """Reward from learner's own move stays positive."""
        rewards = torch.tensor([1.0, 0.0, -1.0])
        pre_players = np.array([0, 0, 0], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=0)
        assert torch.equal(result, torch.tensor([1.0, 0.0, -1.0]))

    def test_opponent_move_reward_negated(self):
        """Reward from opponent's move is negated for learner perspective."""
        rewards = torch.tensor([1.0, 0.0, -1.0])
        pre_players = np.array([1, 1, 1], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=0)
        assert torch.equal(result, torch.tensor([-1.0, 0.0, 1.0]))

    def test_mixed_turns(self):
        """Mixed learner/opponent turns apply selective negation."""
        rewards = torch.tensor([1.0, 1.0, 0.5, -0.5])
        pre_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=0)
        expected = torch.tensor([1.0, -1.0, 0.5, 0.5])
        assert torch.equal(result, expected)

    def test_does_not_mutate_input(self):
        """Input tensor must not be modified in place."""
        rewards = torch.tensor([1.0, -1.0])
        original = rewards.clone()
        pre_players = np.array([1, 0], dtype=np.uint8)
        to_learner_perspective(rewards, pre_players, learner_side=0)
        assert torch.equal(rewards, original)

    def test_empty_tensor(self):
        """Handles zero-env edge case without error."""
        rewards = torch.tensor([])
        pre_players = np.array([], dtype=np.uint8)
        result = to_learner_perspective(rewards, pre_players, learner_side=0)
        assert result.numel() == 0


class TestSignCorrectBootstrap:
    def test_learner_to_move_unchanged(self):
        """Bootstrap stays positive when learner is to move."""
        next_values = torch.tensor([0.5, -0.3, 0.8])
        current_players = np.array([0, 0, 0], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        assert torch.equal(result, next_values)

    def test_opponent_to_move_negated(self):
        """Bootstrap negated when opponent is to move (value is from opponent POV)."""
        next_values = torch.tensor([0.5, -0.3, 0.8])
        current_players = np.array([1, 1, 1], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        assert torch.equal(result, torch.tensor([-0.5, 0.3, -0.8]))

    def test_mixed_perspective(self):
        """Mixed to-move states: only opponent-to-move envs are negated."""
        next_values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        expected = torch.tensor([0.5, -0.5, 0.5, -0.5])
        assert torch.equal(result, expected)

    def test_does_not_mutate_input(self):
        """Input tensor must not be modified in place."""
        next_values = torch.tensor([0.5, -0.3])
        original = next_values.clone()
        current_players = np.array([1, 0], dtype=np.uint8)
        sign_correct_bootstrap(next_values, current_players, learner_side=0)
        assert torch.equal(next_values, original)

    def test_empty_tensor(self):
        """Handles zero-env edge case without error."""
        next_values = torch.tensor([])
        current_players = np.array([], dtype=np.uint8)
        result = sign_correct_bootstrap(next_values, current_players, learner_side=0)
        assert result.numel() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pending_transitions.py -v`
Expected: FAIL with `ImportError: cannot import name 'to_learner_perspective'`

- [ ] **Step 3: Implement the helpers**

In `keisei/training/katago_loop.py`, add before `split_merge_step` (around line 63):

```python
def _negate_where(
    values: torch.Tensor,
    condition: np.ndarray,
) -> torch.Tensor:
    """Clone a tensor and negate elements where condition is True.

    Shared implementation for perspective correction functions.
    """
    result = values.clone()
    if result.numel() == 0:
        return result
    mask = torch.tensor(condition, device=values.device, dtype=torch.bool)
    result[mask] = -result[mask]
    return result


def to_learner_perspective(
    rewards: torch.Tensor,
    pre_players: np.ndarray,
    learner_side: int,
) -> torch.Tensor:
    """Convert rewards from last-mover perspective to learner perspective.

    In split-merge mode, rewards from vecenv.step() are relative to
    whoever just moved (last_mover = pre_players). When the opponent
    moved, the reward sign must be flipped for the learner.
    """
    return _negate_where(rewards, pre_players != learner_side)


def sign_correct_bootstrap(
    next_values: torch.Tensor,
    current_players: np.ndarray,
    learner_side: int,
) -> torch.Tensor:
    """Correct bootstrap values for learner-centric GAE.

    The value network outputs from current-player perspective. When the
    opponent is to-move, the bootstrap value represents the opponent's
    advantage and must be negated for learner-centric GAE targets.
    """
    return _negate_where(next_values, current_players != learner_side)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pending_transitions.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_pending_transitions.py
git commit -m "feat: add perspective-correction helpers for split-merge reward signs"
```

---

### Task 2: Add `PendingTransitions` class with tests

**Files:**
- Modify: `keisei/training/katago_loop.py` (add class before `split_merge_step`)
- Modify: `tests/test_pending_transitions.py`

This class manages per-env pending transition state. When the learner moves, a transition is opened. When the outcome resolves (terminal or turn returns to learner), the transition is finalized and returned for buffer insertion.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pending_transitions.py`:

```python
from keisei.training.katago_loop import PendingTransitions


class TestPendingTransitions:
    """Test the PendingTransitions state container."""

    def _make_pending(self, num_envs=4, obs_channels=50, action_space=11259):
        return PendingTransitions(
            num_envs=num_envs,
            obs_shape=(obs_channels, 9, 9),
            action_space=action_space,
            device=torch.device("cpu"),
        )

    def test_initially_no_valid_pending(self):
        pt = self._make_pending()
        assert not pt.valid.any()

    def test_create_sets_valid_and_stores_data(self):
        pt = self._make_pending(num_envs=4)
        env_mask = torch.tensor([True, False, True, False], dtype=torch.bool)
        obs = torch.randn(4, 50, 9, 9)
        actions = torch.tensor([10, 20, 30, 40])
        log_probs = torch.tensor([0.0, 0.0, -0.5, 0.0])
        values = torch.tensor([0.0, 0.0, 0.3, 0.0])
        legal_masks = torch.ones(4, 11259, dtype=torch.bool)
        rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])
        score_targets = torch.tensor([0.1, 0.0, 0.2, 0.0])

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)

        assert pt.valid[0] and pt.valid[2]
        assert not pt.valid[1] and not pt.valid[3]
        assert torch.equal(pt.actions[0], torch.tensor(10))
        assert torch.equal(pt.actions[2], torch.tensor(30))
        assert pt.log_probs[2].item() == pytest.approx(-0.5)

    def test_create_rejects_overwrite_of_valid_env(self):
        """create() must not be called on an env with an already-open pending
        transition. This catches protocol violations where finalize was skipped."""
        pt = self._make_pending(num_envs=2)
        env_mask = torch.tensor([True, False], dtype=torch.bool)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        rewards = torch.zeros(2)
        score_targets = torch.zeros(2)

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)

        # Attempting to create again on the same env should fail
        with pytest.raises(AssertionError):
            pt.create(env_mask, obs, actions, log_probs, values,
                      legal_masks, rewards, score_targets)

    def test_accumulate_adds_reward(self):
        """Rewards accumulate across steps. Initial reward=0.5, accumulated=-0.2
        gives final=0.3 for env 0. Initial=-0.5, accumulated=0.3 gives -0.2 for env 1."""
        pt = self._make_pending(num_envs=2)
        env_mask = torch.tensor([True, True], dtype=torch.bool)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        rewards = torch.tensor([0.5, -0.5])
        score_targets = torch.zeros(2)

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)
        # Accumulate opponent-turn reward
        pt.accumulate_reward(torch.tensor([-0.2, 0.3]))
        assert pt.rewards[0].item() == pytest.approx(0.3)   # 0.5 + (-0.2)
        assert pt.rewards[1].item() == pytest.approx(-0.2)  # -0.5 + 0.3

    def test_accumulate_before_create_is_noop(self):
        """accumulate_reward before any create should be a safe no-op."""
        pt = self._make_pending(num_envs=2)
        pt.accumulate_reward(torch.tensor([1.0, -1.0]))
        assert pt.rewards[0].item() == 0.0
        assert pt.rewards[1].item() == 0.0

    def test_finalize_returns_data_and_clears(self):
        pt = self._make_pending(num_envs=3)
        env_mask = torch.tensor([True, True, True], dtype=torch.bool)
        obs = torch.randn(3, 50, 9, 9)
        actions = torch.tensor([5, 10, 15])
        log_probs = torch.tensor([-0.1, -0.2, -0.3])
        values = torch.tensor([0.5, 0.6, 0.7])
        legal_masks = torch.ones(3, 11259, dtype=torch.bool)
        rewards = torch.tensor([0.0, 0.0, 0.0])
        score_targets = torch.tensor([0.1, 0.2, 0.3])

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)

        # Finalize envs 0 and 2 (terminal)
        finalize_mask = torch.tensor([True, False, True], dtype=torch.bool)
        dones = torch.tensor([True, False, True])
        result = pt.finalize(finalize_mask, dones)

        assert result is not None
        assert result["env_ids"].shape == (2,)
        assert torch.equal(result["env_ids"], torch.tensor([0, 2]))
        assert torch.equal(result["actions"], torch.tensor([5, 15]))
        assert torch.equal(result["dones"], torch.tensor([1.0, 1.0]))
        # Finalized envs are cleared
        assert not pt.valid[0] and not pt.valid[2]
        # Non-finalized env stays valid
        assert pt.valid[1]

    def test_finalize_nonterminal_sets_done_false(self):
        pt = self._make_pending(num_envs=2)
        env_mask = torch.tensor([True, True], dtype=torch.bool)
        obs = torch.randn(2, 50, 9, 9)
        actions = torch.zeros(2, dtype=torch.long)
        log_probs = torch.zeros(2)
        values = torch.zeros(2)
        legal_masks = torch.ones(2, 11259, dtype=torch.bool)
        rewards = torch.zeros(2)
        score_targets = torch.zeros(2)

        pt.create(env_mask, obs, actions, log_probs, values,
                  legal_masks, rewards, score_targets)

        finalize_mask = torch.tensor([True, False], dtype=torch.bool)
        dones = torch.tensor([False, False])
        result = pt.finalize(finalize_mask, dones)

        assert result is not None
        assert result["dones"][0].item() == 0.0

    def test_finalize_none_returns_none(self):
        pt = self._make_pending(num_envs=2)
        finalize_mask = torch.tensor([False, False], dtype=torch.bool)
        dones = torch.tensor([False, False])
        result = pt.finalize(finalize_mask, dones)
        assert result is None

    def test_finalize_mask_on_invalid_env_is_safe(self):
        """finalize_mask may include envs where valid=False. These are
        silently skipped via the `to_finalize = finalize_mask & self.valid` guard."""
        pt = self._make_pending(num_envs=2)
        # No pending created — both are invalid
        finalize_mask = torch.tensor([True, True], dtype=torch.bool)
        dones = torch.tensor([True, True])
        result = pt.finalize(finalize_mask, dones)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pending_transitions.py::TestPendingTransitions -v`
Expected: FAIL with `ImportError: cannot import name 'PendingTransitions'`

- [ ] **Step 3: Implement `PendingTransitions`**

In `keisei/training/katago_loop.py`, add before `split_merge_step`:

```python
class PendingTransitions:
    """Per-env state for learner transitions awaiting outcome resolution.

    In split-merge mode, a learner transition is "opened" when the learner
    moves, but its reward and done flag depend on what happens next (which
    may be an opponent move). This class holds the transition data until
    the outcome resolves.

    Memory footprint: For num_envs=512, obs_shape=(50,9,9), action_space=11259,
    the total allocation is ~90 MB on the device. This is a persistent per-epoch
    cost alongside the rollout buffer. Kept on GPU to avoid device mismatches
    in the finalize-mask logic (dones, valid, learner_next are all on GPU).
    """

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple[int, ...],
        action_space: int,
        device: torch.device,
    ) -> None:
        self.num_envs = num_envs
        self.obs = torch.zeros(num_envs, *obs_shape, device=device)
        self.actions = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(num_envs, device=device)
        self.values = torch.zeros(num_envs, device=device)
        self.legal_masks = torch.zeros(num_envs, action_space, dtype=torch.bool, device=device)
        self.rewards = torch.zeros(num_envs, device=device)
        self.score_targets = torch.zeros(num_envs, device=device)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def create(
        self,
        env_mask: torch.Tensor,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        legal_masks: torch.Tensor,
        rewards: torch.Tensor,
        score_targets: torch.Tensor,
    ) -> None:
        """Open pending transitions for envs where the learner just moved.

        Raises AssertionError if any env in env_mask already has a valid
        pending transition (indicates a protocol bug — finalize must be
        called before create for the same env).
        """
        assert not (env_mask & self.valid).any(), (
            "create() called on env(s) with already-valid pending transition. "
            "finalize() must be called first."
        )
        self.obs[env_mask] = obs[env_mask]
        self.actions[env_mask] = actions[env_mask]
        self.log_probs[env_mask] = log_probs[env_mask]
        self.values[env_mask] = values[env_mask]
        self.legal_masks[env_mask] = legal_masks[env_mask]
        self.rewards[env_mask] = rewards[env_mask]
        self.score_targets[env_mask] = score_targets[env_mask]
        self.valid[env_mask] = True

    def accumulate_reward(self, learner_rewards: torch.Tensor) -> None:
        """Add perspective-corrected rewards to all valid pending transitions.

        Correctness assumption: non-mover envs have reward=0.0 from the engine,
        so accumulating learner_rewards across all valid envs is safe. If the
        engine ever emits shaping rewards for non-movers, this method would
        need to accept a per-env mask of which envs actually stepped.
        """
        self.rewards[self.valid] += learner_rewards[self.valid]

    def finalize(
        self,
        finalize_mask: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        """Close pending transitions and return data for buffer insertion.

        Returns None if no transitions need finalizing. Otherwise returns
        a dict with keys: obs, actions, log_probs, values, rewards, dones,
        legal_masks, score_targets, env_ids. All tensors are indexed by
        the finalized subset (not full num_envs).

        The finalize_mask may include envs where valid=False — these are
        safely skipped via the internal `to_finalize = finalize_mask & self.valid`
        guard.
        """
        to_finalize = finalize_mask & self.valid
        if not to_finalize.any():
            return None

        indices = to_finalize.nonzero(as_tuple=True)[0]
        result = {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "log_probs": self.log_probs[indices],
            "values": self.values[indices],
            "rewards": self.rewards[indices],
            "dones": dones[indices].float(),
            "legal_masks": self.legal_masks[indices],
            "score_targets": self.score_targets[indices],
            "env_ids": indices,
        }

        # Clear finalized state
        self.valid[to_finalize] = False
        self.rewards[to_finalize] = 0.0

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pending_transitions.py::TestPendingTransitions -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_pending_transitions.py
git commit -m "feat: add PendingTransitions class for split-merge outcome tracking"
```

---

### Task 3: Add protocol-level integration tests

**Files:**
- Modify: `tests/test_pending_transitions.py`

These tests exercise the pending-transition protocol (the exact sequence of calls the loop will make) with multi-env scenarios, including the three core cases: opponent-turn terminal, learner-turn terminal, non-terminal return, plus a heterogeneous multi-env test and epoch-boundary flush test.

- [ ] **Step 1: Write the integration tests**

Append to `tests/test_pending_transitions.py`:

```python
from keisei.training.katago_loop import PendingTransitions, to_learner_perspective


class TestSplitMergeCollection:
    """Integration test: verify buffer receives correct transitions for
    known game sequences in split-merge mode."""

    def test_opponent_terminal_reaches_buffer(self):
        """When the opponent checkmates, the learner's last transition must
        appear in the buffer with done=True and a negative reward."""
        num_envs = 1
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # --- Step 1: Learner moves, game continues ---
        pre_players_1 = np.array([0], dtype=np.uint8)  # learner to move
        obs_1 = torch.randn(1, *obs_shape)
        actions_1 = torch.tensor([42])
        log_probs_full = torch.tensor([-0.5])
        values_full = torch.tensor([0.3])
        legal_masks_1 = torch.ones(1, action_space, dtype=torch.bool)
        rewards_1 = torch.tensor([0.0])  # non-terminal
        dones_1 = torch.tensor([False])
        score_targets_1 = torch.tensor([0.1])
        current_players_after_1 = np.array([1], dtype=np.uint8)  # opponent next

        learner_rewards_1 = to_learner_perspective(rewards_1, pre_players_1, learner_side)

        # Accumulate (no pending yet, so no-op)
        pt.accumulate_reward(learner_rewards_1)
        # Finalize (nothing to finalize)
        finalize_mask = pt.valid & (
            dones_1.bool()
            | torch.tensor(current_players_after_1 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_1)
        assert result is None

        # Create pending for learner's move
        learner_moved = torch.tensor(pre_players_1 == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs_1, actions_1, log_probs_full, values_full,
                  legal_masks_1, learner_rewards_1, score_targets_1)
        assert pt.valid[0]

        # Check immediate terminal (learner moved + terminal)
        imm_terminal = learner_moved & dones_1.bool()
        assert not imm_terminal.any()

        # --- Step 2: Opponent moves, checkmates learner ---
        pre_players_2 = np.array([1], dtype=np.uint8)  # opponent to move
        rewards_2 = torch.tensor([1.0])  # opponent won, from opponent POV
        dones_2 = torch.tensor([True])   # game over
        current_players_after_2 = np.array([0], dtype=np.uint8)  # reset to start

        learner_rewards_2 = to_learner_perspective(rewards_2, pre_players_2, learner_side)
        assert learner_rewards_2[0].item() == -1.0  # negated for learner

        # Accumulate
        pt.accumulate_reward(learner_rewards_2)
        assert pt.rewards[0].item() == pytest.approx(-1.0)

        # Finalize (terminal)
        finalize_mask = pt.valid & (
            dones_2.bool()
            | torch.tensor(current_players_after_2 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_2)

        assert result is not None
        assert result["env_ids"].tolist() == [0]
        assert result["rewards"][0].item() == pytest.approx(-1.0)
        assert result["dones"][0].item() == 1.0
        assert result["actions"][0].item() == 42
        assert result["values"][0].item() == pytest.approx(0.3)
        assert not pt.valid[0]

    def test_learner_terminal_finalized_immediately(self):
        """When the learner checkmates, the pending transition is created
        and immediately finalized in the same step."""
        num_envs = 1
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        pre_players = np.array([0], dtype=np.uint8)
        obs = torch.randn(1, *obs_shape)
        actions = torch.tensor([99])
        log_probs_full = torch.tensor([-0.1])
        values_full = torch.tensor([0.8])
        legal_masks = torch.ones(1, action_space, dtype=torch.bool)
        rewards = torch.tensor([1.0])  # learner won, from learner POV
        dones = torch.tensor([True])
        score_targets = torch.tensor([0.5])
        current_players_after = np.array([0], dtype=np.uint8)  # reset

        learner_rewards = to_learner_perspective(rewards, pre_players, learner_side)

        # Accumulate (no pending yet)
        pt.accumulate_reward(learner_rewards)

        # Finalize existing (none)
        finalize_mask = pt.valid & (
            dones.bool()
            | torch.tensor(current_players_after == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones)
        assert result is None

        # Create pending
        learner_moved = torch.tensor(pre_players == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs, actions, log_probs_full, values_full,
                  legal_masks, learner_rewards, score_targets)
        assert pt.valid[0]

        # Immediate terminal finalize
        imm_terminal = learner_moved & dones.bool()
        result = pt.finalize(imm_terminal, dones)

        assert result is not None
        assert result["rewards"][0].item() == pytest.approx(1.0)
        assert result["dones"][0].item() == 1.0
        assert not pt.valid[0]

    def test_nonterminal_finalized_when_turn_returns(self):
        """Non-terminal transitions are finalized when the turn returns
        to the learner (opponent moved, game continues)."""
        num_envs = 1
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # Step 1: Learner moves
        pre_players_1 = np.array([0], dtype=np.uint8)
        obs_1 = torch.randn(1, *obs_shape)
        actions_1 = torch.tensor([7])
        log_probs_1 = torch.tensor([-0.3])
        values_1 = torch.tensor([0.2])
        legal_masks_1 = torch.ones(1, action_space, dtype=torch.bool)
        rewards_1 = torch.tensor([0.0])
        dones_1 = torch.tensor([False])
        score_targets_1 = torch.tensor([0.0])
        current_players_after_1 = np.array([1], dtype=np.uint8)

        learner_rewards_1 = to_learner_perspective(rewards_1, pre_players_1, learner_side)
        pt.accumulate_reward(learner_rewards_1)
        pt.finalize(
            pt.valid & (
                dones_1.bool()
                | torch.tensor(current_players_after_1 == learner_side, dtype=torch.bool)
            ),
            dones_1,
        )
        learner_moved = torch.tensor(pre_players_1 == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs_1, actions_1, log_probs_1, values_1,
                  legal_masks_1, learner_rewards_1, score_targets_1)

        # Step 2: Opponent moves, game continues
        pre_players_2 = np.array([1], dtype=np.uint8)
        rewards_2 = torch.tensor([0.0])
        dones_2 = torch.tensor([False])
        current_players_after_2 = np.array([0], dtype=np.uint8)  # back to learner

        learner_rewards_2 = to_learner_perspective(rewards_2, pre_players_2, learner_side)
        pt.accumulate_reward(learner_rewards_2)

        # Finalize: non-terminal, turn returns to learner
        finalize_mask = pt.valid & (
            dones_2.bool()
            | torch.tensor(current_players_after_2 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_2)

        assert result is not None
        assert result["dones"][0].item() == 0.0
        assert result["rewards"][0].item() == pytest.approx(0.0)
        assert not pt.valid[0]

    def test_multi_env_heterogeneous_terminal(self):
        """In the same step, env 0 has an opponent-turn terminal while
        env 1 has a non-terminal opponent move. Both must be handled correctly."""
        num_envs = 2
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # Step 1: Both envs have learner to move
        pre_players_1 = np.array([0, 0], dtype=np.uint8)
        obs_1 = torch.randn(2, *obs_shape)
        actions_1 = torch.tensor([10, 20])
        log_probs_1 = torch.tensor([-0.1, -0.2])
        values_1 = torch.tensor([0.5, 0.6])
        legal_masks_1 = torch.ones(2, action_space, dtype=torch.bool)
        rewards_1 = torch.tensor([0.0, 0.0])
        dones_1 = torch.tensor([False, False])
        score_targets_1 = torch.tensor([0.1, 0.2])
        current_players_after_1 = np.array([1, 1], dtype=np.uint8)

        learner_rewards_1 = to_learner_perspective(rewards_1, pre_players_1, learner_side)
        pt.accumulate_reward(learner_rewards_1)
        pt.finalize(
            pt.valid & (
                dones_1.bool()
                | torch.tensor(current_players_after_1 == learner_side, dtype=torch.bool)
            ),
            dones_1,
        )
        learner_moved = torch.tensor(pre_players_1 == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs_1, actions_1, log_probs_1, values_1,
                  legal_masks_1, learner_rewards_1, score_targets_1)

        # Step 2: Both envs have opponent to move
        # Env 0: opponent checkmates (terminal)
        # Env 1: opponent moves, game continues
        pre_players_2 = np.array([1, 1], dtype=np.uint8)
        rewards_2 = torch.tensor([1.0, 0.0])  # env 0: opponent won; env 1: non-terminal
        dones_2 = torch.tensor([True, False])
        current_players_after_2 = np.array([0, 0], dtype=np.uint8)  # env 0: reset; env 1: back to learner

        learner_rewards_2 = to_learner_perspective(rewards_2, pre_players_2, learner_side)
        pt.accumulate_reward(learner_rewards_2)

        finalize_mask = pt.valid & (
            dones_2.bool()
            | torch.tensor(current_players_after_2 == learner_side, dtype=torch.bool)
        )
        result = pt.finalize(finalize_mask, dones_2)

        assert result is not None
        assert result["env_ids"].tolist() == [0, 1]
        # Env 0: terminal loss — reward = -1.0, done = 1.0
        assert result["rewards"][0].item() == pytest.approx(-1.0)
        assert result["dones"][0].item() == 1.0
        assert result["actions"][0].item() == 10
        # Env 1: non-terminal — reward = 0.0, done = 0.0
        assert result["rewards"][1].item() == pytest.approx(0.0)
        assert result["dones"][1].item() == 0.0
        assert result["actions"][1].item() == 20

        # Both cleared
        assert not pt.valid.any()

    def test_epoch_end_flush(self):
        """Pending transitions remaining at epoch end are finalized with
        done=False and value_cat=-1 (non-terminal bootstrap)."""
        num_envs = 2
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")
        learner_side = 0

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # Step 1: Both envs have learner to move
        pre_players = np.array([0, 0], dtype=np.uint8)
        obs = torch.randn(2, *obs_shape)
        actions = torch.tensor([5, 6])
        log_probs = torch.tensor([-0.1, -0.2])
        values = torch.tensor([0.4, 0.5])
        legal_masks = torch.ones(2, action_space, dtype=torch.bool)
        rewards = torch.tensor([0.0, 0.0])
        score_targets = torch.tensor([0.0, 0.0])

        learner_rewards = to_learner_perspective(rewards, pre_players, learner_side)
        learner_moved = torch.tensor(pre_players == learner_side, dtype=torch.bool)
        pt.create(learner_moved, obs, actions, log_probs, values,
                  legal_masks, learner_rewards, score_targets)

        # Epoch ends — no more steps. Flush remaining pending.
        remaining_mask = pt.valid.clone()
        remaining_dones = torch.zeros(num_envs)
        result = pt.finalize(remaining_mask, remaining_dones)

        assert result is not None
        assert result["env_ids"].tolist() == [0, 1]
        # All flushed as non-terminal
        assert result["dones"][0].item() == 0.0
        assert result["dones"][1].item() == 0.0
        assert result["values"][0].item() == pytest.approx(0.4)
        assert not pt.valid.any()
```

- [ ] **Step 2: Run tests to verify they pass**

These tests use the already-implemented `PendingTransitions` and `to_learner_perspective`, so they should pass immediately — they're testing the *protocol* (the sequence of calls the loop will make).

Run: `uv run pytest tests/test_pending_transitions.py::TestSplitMergeCollection -v`
Expected: PASS (5 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/test_pending_transitions.py
git commit -m "test: add protocol-level integration tests for pending transitions"
```

---

### Task 4: Rewrite split-merge collection loop to use pending transitions

**Files:**
- Modify: `keisei/training/katago_loop.py:465-565` (the split-merge branch of `run()`)

This is the core fix. Replace the "store only learner-turn transitions" logic with the pending-transition pattern.

**Two-pass invariant (IMPORTANT — do not reorder steps 1-2 vs 3-4):**
The pending transition protocol per step has four stages. Steps 1-2 operate on transitions opened in *prior* steps. Steps 3-4 operate on transitions opened in *this* step. These two populations are disjoint because `create` in step 3 only fires for `learner_moved` envs, and step 2's finalize already cleared those envs' valid bit (since `learner_next` covers envs the learner is about to move in). Reordering steps 2 and 3 would cause just-created pending transitions to be spuriously finalized in the same step.

Also fixes win/loss/draw tracking to use learner-perspective rewards.

- [ ] **Step 1: Add the `import logging` and logger if not already present**

Check the top of `keisei/training/katago_loop.py` for an existing `logger`. If present (it should be), no change needed. If absent, add:

```python
import logging
logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Rewrite the split-merge branch in `run()`**

In `keisei/training/katago_loop.py`, apply the following changes to the `run()` method:

**A. Before the `for step_i` loop** (after line 464, before line 465), add pending transition initialization and epoch-scoped `learner_side`:

```python
            learner_side = 0  # Epoch-scoped: used by bootstrap sign correction after the loop
            pending: PendingTransitions | None = None
            if self._current_opponent is not None:
                obs_channels = obs.shape[1]
                action_space = self.buffer.action_space
                pending = PendingTransitions(
                    self.num_envs, (obs_channels, 9, 9), action_space, self.device,
                )
```

**B. Replace the split-merge branch** (lines 470-521, the `if self._current_opponent is not None:` block) with:

```python
                if self._current_opponent is not None:
                    assert pending is not None
                    pre_players = current_players.copy()

                    # Split-merge: learner vs opponent
                    sm_result = split_merge_step(
                        obs=obs, legal_masks=legal_masks,
                        current_players=current_players,
                        learner_model=self.model,
                        opponent_model=self._current_opponent,
                        learner_side=learner_side,
                    )
                    actions = sm_result.actions
                    action_list = actions.tolist()
                    step_result = self.vecenv.step(action_list)

                    current_players = np.asarray(step_result.current_players)

                    rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(self.device)
                    terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(self.device)
                    truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(self.device)
                    dones = terminated | truncated

                    # Convert rewards to learner perspective
                    learner_rewards = to_learner_perspective(rewards, pre_players, learner_side)

                    # Track wins/losses/draws from learner perspective
                    terminal_mask = dones.bool()
                    if terminal_mask.any():
                        t_rewards = learner_rewards[terminal_mask]
                        win_acc += (t_rewards > 0).sum()
                        loss_acc += (t_rewards < 0).sum()
                        draw_acc += (t_rewards == 0).sum()

                    # --- Pending transition protocol ---
                    # Steps 1-2 finalize transitions from PRIOR steps.
                    # Steps 3-4 create and possibly finalize transitions from THIS step.
                    # These populations are disjoint. Do not reorder.

                    # 1. Accumulate rewards into existing pending transitions
                    pending.accumulate_reward(learner_rewards)

                    # 2. Finalize resolved pending transitions:
                    #    - Terminal: game ended (on any player's move)
                    #    - Non-terminal return: opponent moved, turn returns to learner
                    learner_next = torch.tensor(
                        current_players == learner_side, device=self.device, dtype=torch.bool,
                    )
                    finalize_mask = pending.valid & (dones.bool() | learner_next)
                    finalized = pending.finalize(finalize_mask, dones)

                    if finalized is not None:
                        f_rewards = finalized["rewards"]
                        f_dones_bool = finalized["dones"].bool()
                        f_value_cats = torch.full(
                            (finalized["env_ids"].numel(),), -1,
                            dtype=torch.long, device=self.device,
                        )
                        f_value_cats[f_dones_bool & (f_rewards > 0)] = 0
                        f_value_cats[f_dones_bool & (f_rewards == 0)] = 1
                        f_value_cats[f_dones_bool & (f_rewards < 0)] = 2

                        self.buffer.add(
                            finalized["obs"], finalized["actions"],
                            finalized["log_probs"], finalized["values"],
                            finalized["rewards"], finalized["dones"],
                            finalized["legal_masks"], f_value_cats,
                            finalized["score_targets"],
                            env_ids=finalized["env_ids"],
                        )

                    # 3. Create new pending for envs where learner just moved
                    learner_moved = torch.tensor(
                        pre_players == learner_side, device=self.device, dtype=torch.bool,
                    )
                    if learner_moved.any():
                        li = sm_result.learner_indices
                        # Scatter compact learner data to full num_envs tensors
                        full_log_probs = torch.zeros(self.num_envs, device=self.device)
                        full_values = torch.zeros(self.num_envs, device=self.device)
                        if li.numel() > 0:
                            full_log_probs[li] = sm_result.learner_log_probs
                            full_values[li] = sm_result.learner_values

                        material = torch.from_numpy(
                            np.asarray(step_result.step_metadata.material_balance, dtype=np.float32),
                        ).to(self.device)
                        full_score_targets = material / self.score_norm

                        pending.create(
                            learner_moved, obs, actions, full_log_probs, full_values,
                            legal_masks, learner_rewards, full_score_targets,
                        )

                        # 4. Immediately finalize if learner's own move was terminal
                        imm_terminal = learner_moved & dones.bool()
                        if imm_terminal.any():
                            imm_finalized = pending.finalize(imm_terminal, dones)
                            if imm_finalized is not None:
                                imm_rewards = imm_finalized["rewards"]
                                imm_dones_bool = imm_finalized["dones"].bool()
                                imm_value_cats = torch.full(
                                    (imm_finalized["env_ids"].numel(),), -1,
                                    dtype=torch.long, device=self.device,
                                )
                                imm_value_cats[imm_dones_bool & (imm_rewards > 0)] = 0
                                imm_value_cats[imm_dones_bool & (imm_rewards == 0)] = 1
                                imm_value_cats[imm_dones_bool & (imm_rewards < 0)] = 2

                                self.buffer.add(
                                    imm_finalized["obs"], imm_finalized["actions"],
                                    imm_finalized["log_probs"], imm_finalized["values"],
                                    imm_finalized["rewards"], imm_finalized["dones"],
                                    imm_finalized["legal_masks"], imm_value_cats,
                                    imm_finalized["score_targets"],
                                    env_ids=imm_finalized["env_ids"],
                                )
```

**IMPORTANT:** The `obs` used in `pending.create()` is the observation **before** the step (the learner's observation when it chose its action). The `obs` variable is not updated until after this entire block (line 555: `obs = torch.from_numpy(...)`), so this is correct.

**C. After the loop, before bootstrap computation** (between line 557 and 560), add finalization of remaining pending transitions with logging:

```python
            # Finalize any remaining pending transitions at epoch end.
            # These are learner moves whose games did not resolve before the epoch
            # ended. They are stored with done=False so GAE bootstraps from next_values.
            if pending is not None and pending.valid.any():
                flush_count = pending.valid.sum().item()
                remaining_mask = pending.valid.clone()
                remaining_dones = torch.zeros(self.num_envs, device=self.device)
                remaining = pending.finalize(remaining_mask, remaining_dones)
                if remaining is not None:
                    remaining_value_cats = torch.full(
                        (remaining["env_ids"].numel(),), -1,
                        dtype=torch.long, device=self.device,
                    )
                    self.buffer.add(
                        remaining["obs"], remaining["actions"],
                        remaining["log_probs"], remaining["values"],
                        remaining["rewards"], remaining["dones"],
                        remaining["legal_masks"], remaining_value_cats,
                        remaining["score_targets"],
                        env_ids=remaining["env_ids"],
                    )
                    logger.debug(
                        "Epoch %d: flushed %d pending transitions at epoch end",
                        epoch_i, flush_count,
                    )
```

**D. Fix the bootstrap sign correction** (replace the existing bootstrap block, currently at lines 560-565):

```python
            # Bootstrap value for GAE
            self.ppo.forward_model.eval()
            with torch.no_grad():
                output = self.ppo.forward_model(obs)
                next_values = KataGoPPOAlgorithm.scalar_value(output.value_logits)
            self.ppo.forward_model.train()

            # Sign-correct bootstrap for split-merge mode: the value network
            # outputs from current-player perspective. When the opponent is to-move
            # at epoch end, the bootstrap value must be negated for learner-centric GAE.
            if self._current_opponent is not None:
                next_values = sign_correct_bootstrap(
                    next_values, current_players, learner_side,
                )
```

Note: `learner_side` here references the epoch-scoped variable defined in step A, not a hardcoded `0`. This ensures the variable is always in scope even if `steps_per_epoch == 0`.

- [ ] **Step 3: Run existing tests to check for regressions**

Run: `uv run pytest tests/test_split_merge.py tests/test_pending_transitions.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest --timeout=60 -x -q`
Expected: No failures

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "fix: use pending transitions for split-merge reward collection

Fixes two critical bugs in league split-merge mode:
1. Terminal outcomes on opponent turns were silently dropped (P0)
2. Bootstrap values lacked sign correction for opponent-to-move states (P1)

The new pending-transition pattern holds each learner transition open
until the outcome resolves, ensuring all terminal signals (including
opponent-turn checkmates) reach the PPO buffer with correct perspective.

Also fixes win/loss/draw tracking to use learner-perspective rewards
and adds debug logging for pending transition finalization."
```

---

### Task 5: Checkpoint continuity and Elo contamination safeguards

**Files:**
- Modify: `keisei/training/katago_loop.py` (add LR scheduler reset on post-fix resume)

When applying this fix to a checkpoint trained with the buggy code, the value head will suddenly see terminal rewards it never saw before (opponent-turn checkmates). This causes a value-loss spike that can trigger the `ReduceLROnPlateau` scheduler to prematurely reduce LR, potentially locking the model into a low-LR regime.

Additionally, Elo ratings in the DB are contaminated with inverted win/loss data from the old code (opponent wins were counted as learner wins). After the fix, true outcomes are recorded, causing a sudden apparent regression in learner strength.

- [ ] **Step 1: Add LR scheduler reset on first post-fix epoch**

In `keisei/training/katago_loop.py`, find the LR scheduler logic after the PPO update (around line 589-614). Add a one-time reset when league mode is active and the scheduler has accumulated history. The simplest approach: reset the scheduler at the start of the first epoch when `self._current_opponent` is not None, using the existing pattern from the warmup boundary.

Locate the existing warmup reset at approximately line 569-573:
```python
            if epoch_i == self.ppo.warmup_epochs and self.lr_scheduler is not None:
                self.lr_scheduler.best = self.lr_scheduler.mode_worse
                self.lr_scheduler.num_bad_epochs = 0
```

Add a similar block after the pending transition changes, at the top of the epoch loop (inside `for epoch_i`, before the rollout loop), guarded by a one-time flag:

```python
            # One-time LR scheduler reset for league mode to prevent value-loss
            # spike from triggering premature LR reduction after reward-collection
            # fix. Safe to remove once no pre-fix checkpoints are in use.
            if (self._current_opponent is not None
                    and self.lr_scheduler is not None
                    and epoch_i == start_epoch
                    and start_epoch > 0):
                self.lr_scheduler.best = self.lr_scheduler.mode_worse
                self.lr_scheduler.num_bad_epochs = 0
                if self.dist_ctx.is_main:
                    logger.info(
                        "LR scheduler reset at epoch %d for post-fix checkpoint continuity",
                        epoch_i,
                    )
```

- [ ] **Step 2: Add Elo contamination warning on resume**

In the same area, add a log warning about potential Elo contamination from pre-fix data:

```python
            if (self._current_opponent is not None
                    and epoch_i == start_epoch
                    and start_epoch > 0
                    and self.dist_ctx.is_main):
                logger.warning(
                    "Resuming league training with corrected reward collection. "
                    "Elo ratings from epochs before this fix may be inaccurate "
                    "(opponent-turn terminals were previously miscounted). "
                    "Elo will self-correct over subsequent epochs."
                )
```

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest --timeout=60 -x -q`
Expected: No failures

- [ ] **Step 4: Commit**

```bash
git add keisei/training/katago_loop.py
git commit -m "fix: add LR scheduler reset and Elo warning for post-fix checkpoint continuity"
```

---

### Task 6: Final verification

- [ ] **Step 1: Verify the score_targets perspective**

The `material_balance` from the Rust engine is from `last_mover`'s perspective (`vec_env.rs:364-367`). When the learner moves, this is from the learner's perspective — correct for `score_targets` in the pending transition. When the opponent moves and the transition is finalized, the `score_targets` still reflect the learner's move position (saved at creation time), which is correct. No additional sign correction needed for `score_targets`.

**Note:** After this fix, the score head will see score targets for opponent-turn checkmates that were previously dropped. If the opponent captured a large piece on the final move, the score target (from the learner's pre-move material) may be in tension with the value head's terminal loss signal. Monitor `score_loss` for a step-up after deploying this fix — this is expected and benign.

Verify by reading `vec_env.rs:364-367` and confirming `material_balance` uses `last_mover`.

- [ ] **Step 2: Verify the `obs` used in pending.create is pre-step**

In the loop, `obs` is only updated at line 555 (`obs = torch.from_numpy(np.asarray(step_result.observations))`), which comes AFTER the split-merge block. So `obs` in `pending.create()` is the pre-step observation — the state the learner saw when choosing its action. This is correct.

- [ ] **Step 3: Check that non-league path is unchanged**

Read through the `else` branch (lines 523-553) and confirm it has no modifications. The non-league path does not use `pending`, `to_learner_perspective`, or `sign_correct_bootstrap`.

- [ ] **Step 4: Run full test suite one final time**

Run: `uv run pytest --timeout=60 -x -q`
Expected: No failures

---

## Known Limitations (Out of Scope)

### Non-league GAE sign issue

The non-league path has a latent sign issue in its GAE computation: rewards and values alternate perspective each step (mover's POV), but GAE treats them as same-agent transitions. Individual advantage estimates are systematically biased (not just noisy), though this does not fully cancel out even though the same model plays both sides. This is a separate concern documented in the bug report's root cause analysis but NOT addressed by this plan. It does not regress with this fix.

### Truncation vs termination in pending transitions

The pending transition stores `dones = terminated | truncated`, meaning truncated games are treated identically to checkmates (both get `done=True`). In strict PPO, truncated episodes should use bootstrapped values (`done=False`). This is a pre-existing issue (the non-league path has the same behavior) and is not introduced by this fix. Fixing it would require separating `terminated` and `truncated` in the pending finalization logic and applying `done=True` only for `terminated`.

### Elo rating history

Elo ratings computed before this fix used inverted win/loss data for opponent-turn terminals. Historical Elo values in the DB may be inaccurate. Elo will self-correct over subsequent epochs of training with the fixed code. No migration of historical data is attempted.
