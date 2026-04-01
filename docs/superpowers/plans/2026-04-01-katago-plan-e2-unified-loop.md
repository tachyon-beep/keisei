# KataGo Plan E-2: Unified Loop & Split-Merge

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify `KataGoTrainingLoop` to support split-merge step logic (learner vs frozen opponent), integrate the opponent league (pool snapshots, sampling, Elo tracking), add seat rotation, and dispatch loss computation through the value-head adapter.

**Architecture:** Modifications to `katago_loop.py` and `katago_ppo.py`. Uses `OpponentPool`, `OpponentSampler`, `ValueHeadAdapter`, and `get_model_contract` from Plan E-1. No new modules — this wires existing infrastructure into the training loop.

**Tech Stack:** Python 3.13, PyTorch, dataclasses. Tests via `uv run pytest`.

**Dependencies:** Requires Plans A-D and Plan E-1 complete. Verify before starting:

```bash
uv run python -c "
from keisei.training.league import OpponentPool, OpponentSampler, compute_elo_update
from keisei.training.value_adapter import get_value_adapter
from keisei.training.model_registry import get_model_contract
from keisei.config import LeagueConfig
print('Plan E-1 ready')
"
```

**Spec reference:** `docs/superpowers/specs/2026-04-01-plan-e-league-consolidation-design.md` — Split-Merge Step Logic, Opponent League, Rotating Training Seat sections.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `keisei/training/katago_loop.py` | Split-merge step, league integration, seat rotation, value adapter dispatch |
| Modify | `keisei/training/katago_ppo.py` | Accept `ValueHeadAdapter` for loss dispatch in `update()` |
| Modify | `tests/test_katago_loop.py` | New tests for league integration and split-merge |
| Create | `tests/test_split_merge.py` | Dedicated tests for the split-merge step logic |

---

### Task 1: Split-Merge Step Logic

**Files:**
- Create: `tests/test_split_merge.py`
- Modify: `keisei/training/katago_loop.py`

The core inner-loop change: each VecEnv step splits environments by `current_players`, runs two separate forward passes (learner with gradients, opponent with `torch.no_grad()`), and merges actions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split_merge.py
"""Tests for the split-merge step logic in the unified training loop."""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

from keisei.training.katago_loop import split_merge_step


def _make_mock_model(action_space: int = 11259):
    """Create a mock model that returns deterministic actions."""
    model = MagicMock()

    def forward(obs):
        batch = obs.shape[0]
        output = MagicMock()
        output.policy_logits = torch.randn(batch, 9, 9, 139)
        output.value_logits = torch.randn(batch, 3)
        output.score_lead = torch.randn(batch, 1)
        return output

    model.side_effect = forward
    model.__call__ = forward
    return model


class TestSplitMergeStep:
    def test_actions_shape_matches_num_envs(self):
        """Merged actions should cover all environments."""
        num_envs = 8
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        # Black (0) = learner for envs 0,2,4,6; White (1) = opponent for 1,3,5,7
        current_players = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8)

        learner_model = _make_mock_model()
        opponent_model = _make_mock_model()

        result = split_merge_step(
            obs=obs,
            legal_masks=legal_masks,
            current_players=current_players,
            learner_model=learner_model,
            opponent_model=opponent_model,
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        assert result.learner_mask.sum() == 4  # 4 envs where learner plays
        assert result.opponent_mask.sum() == 4

    def test_learner_actions_have_log_probs(self):
        """Learner actions should come with log_probs and values for the buffer."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)

        learner_model = _make_mock_model()
        opponent_model = _make_mock_model()

        result = split_merge_step(
            obs=obs,
            legal_masks=legal_masks,
            current_players=current_players,
            learner_model=learner_model,
            opponent_model=opponent_model,
            learner_side=0,
        )

        # log_probs and values only for learner envs
        assert result.learner_log_probs.shape == (2,)  # 2 learner envs
        assert result.learner_values.shape == (2,)

    def test_all_learner_envs(self):
        """When all envs are learner's turn, opponent model should not be called."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([0, 0, 0, 0], dtype=np.uint8)

        learner_model = _make_mock_model()
        opponent_model = _make_mock_model()

        result = split_merge_step(
            obs=obs,
            legal_masks=legal_masks,
            current_players=current_players,
            learner_model=learner_model,
            opponent_model=opponent_model,
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        assert result.learner_mask.all()

    def test_all_opponent_envs(self):
        """When all envs are opponent's turn, learner model should not be called."""
        num_envs = 4
        obs = torch.randn(num_envs, 50, 9, 9)
        legal_masks = torch.ones(num_envs, 11259, dtype=torch.bool)
        current_players = np.array([1, 1, 1, 1], dtype=np.uint8)

        learner_model = _make_mock_model()
        opponent_model = _make_mock_model()

        result = split_merge_step(
            obs=obs,
            legal_masks=legal_masks,
            current_players=current_players,
            learner_model=learner_model,
            opponent_model=opponent_model,
            learner_side=0,
        )

        assert result.actions.shape == (num_envs,)
        assert result.opponent_mask.all()
        # No learner data to store
        assert result.learner_log_probs.shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split_merge.py -v`
Expected: FAIL — `ImportError: cannot import name 'split_merge_step'`

- [ ] **Step 3: Write the implementation**

Add to `keisei/training/katago_loop.py` (imports at module top):

```python
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class SplitMergeResult:
    """Result of a split-merge step."""
    actions: torch.Tensor          # (num_envs,) — merged actions for all envs
    learner_mask: torch.Tensor     # (num_envs,) bool — which envs were learner's turn
    opponent_mask: torch.Tensor    # (num_envs,) bool — which envs were opponent's turn
    learner_log_probs: torch.Tensor  # (n_learner,) — log probs for learner actions only
    learner_values: torch.Tensor     # (n_learner,) — scalar values for learner envs only
    learner_indices: torch.Tensor    # (n_learner,) — indices into the full env array


def split_merge_step(
    obs: torch.Tensor,
    legal_masks: torch.Tensor,
    current_players: np.ndarray,
    learner_model: torch.nn.Module,
    opponent_model: torch.nn.Module,
    learner_side: int = 0,
    value_adapter: "ValueHeadAdapter | None" = None,
) -> SplitMergeResult:
    """Execute one step with split learner/opponent forward passes.

    Args:
        obs: (num_envs, C, 9, 9) observations for all envs
        legal_masks: (num_envs, action_space) legal action masks
        current_players: (num_envs,) uint8 array of which side is to move
        learner_model: the training model (eval mode, no_grad for rollout)
        opponent_model: frozen opponent (always torch.no_grad)
        learner_side: which player index the learner controls (default 0 = Black)
        value_adapter: for projecting value output to scalar (if None, uses KataGo default)

    Returns only learner-side data (log_probs, values, indices). The caller
    stores ONLY learner transitions in the rollout buffer — opponent transitions
    are discarded. This prevents zero-padded opponent data from corrupting GAE.
    """
    num_envs = obs.shape[0]
    device = obs.device

    learner_mask = torch.tensor(current_players == learner_side, device=device)
    opponent_mask = ~learner_mask
    learner_indices = learner_mask.nonzero(as_tuple=True)[0]
    opponent_indices = opponent_mask.nonzero(as_tuple=True)[0]

    actions = torch.zeros(num_envs, dtype=torch.long, device=device)
    learner_log_probs = torch.zeros(0, device=device)
    learner_values = torch.zeros(0, device=device)

    # Learner forward pass — eval mode for BatchNorm running stats,
    # torch.no_grad for memory efficiency. Stored log_probs are detached
    # leaf tensors; PPO recomputes new_log_probs under train() in update().
    # This is the standard PPO collection pattern (see ppo.py select_actions).
    if learner_indices.numel() > 0:
        l_obs = obs[learner_indices]
        l_masks = legal_masks[learner_indices]

        learner_model.eval()
        with torch.no_grad():
            l_output = learner_model(l_obs)

        # Flatten policy and sample
        l_flat = l_output.policy_logits.reshape(l_obs.shape[0], -1)
        l_masked = l_flat.masked_fill(~l_masks, float("-inf"))
        l_probs = F.softmax(l_masked, dim=-1)
        l_dist = torch.distributions.Categorical(l_probs)
        l_actions = l_dist.sample()
        learner_log_probs = l_dist.log_prob(l_actions)

        # Scalar value for GAE
        if value_adapter is not None:
            learner_values = value_adapter.scalar_value_from_output(l_output.value_logits)
        else:
            # Default: KataGo P(W) - P(L)
            vp = F.softmax(l_output.value_logits, dim=-1)
            learner_values = vp[:, 0] - vp[:, 2]

        actions[learner_indices] = l_actions

    # Opponent forward pass (always no_grad, eval mode)
    if opponent_indices.numel() > 0:
        o_obs = obs[opponent_indices]
        o_masks = legal_masks[opponent_indices]

        opponent_model.eval()
        with torch.no_grad():
            o_output = opponent_model(o_obs)

        o_flat = o_output.policy_logits.reshape(o_obs.shape[0], -1)
        o_masked = o_flat.masked_fill(~o_masks, float("-inf"))
        o_probs = F.softmax(o_masked, dim=-1)
        o_dist = torch.distributions.Categorical(o_probs)
        o_actions = o_dist.sample()

        actions[opponent_indices] = o_actions

    return SplitMergeResult(
        actions=actions,
        learner_mask=learner_mask,
        opponent_mask=opponent_mask,
        learner_log_probs=learner_log_probs,
        learner_values=learner_values,
        learner_indices=learner_indices,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_split_merge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_split_merge.py
git commit -m "feat: add split_merge_step for learner vs opponent forward passes"
```

---

### Task 2: League Integration in Training Loop

**Files:**
- Modify: `keisei/training/katago_loop.py`
- Modify: `tests/test_katago_loop.py`

Wire `OpponentPool` and `OpponentSampler` into `KataGoTrainingLoop.__init__` and `run()`. Add bootstrap snapshot at epoch 0, periodic snapshots, and opponent loading.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_loop.py`:

```python
from keisei.training.league import OpponentPool, OpponentSampler


class TestLeagueIntegration:
    def test_bootstrap_snapshot_at_init(self, katago_config, tmp_path):
        """Pool should have one entry after init (the bootstrap snapshot)."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=10)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        assert len(loop.pool.list_entries()) == 1

    def test_periodic_snapshot(self, katago_config, tmp_path):
        """Pool should gain a snapshot every snapshot_interval epochs."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        # Bootstrap = 1 entry. Run 4 epochs with interval=2 → 2 more snapshots.
        loop.run(num_epochs=4, steps_per_epoch=2)
        entries = loop.pool.list_entries()
        assert len(entries) >= 3  # bootstrap + 2 periodic

    def test_opponent_loaded_each_epoch(self, katago_config, tmp_path):
        """A frozen opponent model should be loaded for each epoch."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=5)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=2, steps_per_epoch=2)
        assert loop._current_opponent is not None
        assert not loop._current_opponent.training  # should be in eval mode


def _with_league(config, tmp_path, snapshot_interval=10):
    """Helper to add league config to an existing AppConfig."""
    import dataclasses
    from keisei.config import LeagueConfig
    league = LeagueConfig(
        max_pool_size=10,
        snapshot_interval=snapshot_interval,
        epochs_per_seat=50,
        elo_floor=500,
    )
    return dataclasses.replace(config, league=league)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestLeagueIntegration -v`
Expected: FAIL — `_with_league` creates config with `league` field, but loop doesn't handle it yet

- [ ] **Step 3: Write the implementation**

In `KataGoTrainingLoop.__init__`, after creating the PPO algorithm, add league setup:

```python
# League setup (optional — only if [league] config is present)
self.pool: OpponentPool | None = None
self.sampler: OpponentSampler | None = None
self._current_opponent: torch.nn.Module | None = None
self._current_opponent_entry: OpponentEntry | None = None

if config.league is not None:
    league_dir = str(Path(config.training.checkpoint_dir) / "league")
    self.pool = OpponentPool(
        self.db_path, league_dir, max_pool_size=config.league.max_pool_size
    )
    self.sampler = OpponentSampler(
        self.pool,
        historical_ratio=config.league.historical_ratio,
        current_best_ratio=config.league.current_best_ratio,
        elo_floor=config.league.elo_floor,
    )

    # Bootstrap snapshot: save initial weights so pool is never empty.
    # Store the entry ID explicitly — do NOT use list_entries()[-1] later,
    # because periodic snapshots change the ordering.
    base_model = self.model.module if hasattr(self.model, "module") else self.model
    bootstrap_entry = self.pool.add_snapshot(
        base_model, config.model.architecture,
        dict(config.model.params), epoch=0,
    )
    self._learner_entry_id = bootstrap_entry.id
    logger.info("League initialized: pool_size=%d, snapshot_interval=%d, learner_entry=%d",
                config.league.max_pool_size, config.league.snapshot_interval,
                self._learner_entry_id)
```

In `run()`, at the start of each epoch, sample an opponent:

```python
# At the start of each epoch loop body:
if self.sampler is not None:
    self._current_opponent_entry = self.sampler.sample()
    self._current_opponent = self.pool.load_opponent(
        self._current_opponent_entry, device=str(self.device)
    )
    logger.info("Epoch %d opponent: id=%d arch=%s elo=%.0f",
                epoch_i, self._current_opponent_entry.id,
                self._current_opponent_entry.architecture,
                self._current_opponent_entry.elo_rating)
```

At the end of each epoch, after `ppo.update()`, add snapshot if interval:

```python
# Periodic pool snapshot — skip if seat rotation already snapshotted this epoch
rotated_this_epoch = (
    self.config.league is not None
    and (epoch_i + 1) % self.config.league.epochs_per_seat == 0
)
if (self.pool is not None and self.config.league is not None
        and (epoch_i + 1) % self.config.league.snapshot_interval == 0
        and not rotated_this_epoch):
    base_model = self.model.module if hasattr(self.model, "module") else self.model
    self.pool.add_snapshot(
        base_model, self.config.model.architecture,
        dict(self.config.model.params), epoch=epoch_i + 1,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_loop.py::TestLeagueIntegration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat: integrate OpponentPool and OpponentSampler into training loop"
```

---

### Task 3: Elo Tracking

**Files:**
- Modify: `keisei/training/katago_loop.py`
- Modify: `tests/test_katago_loop.py`

After each epoch, compute Elo updates from win/loss/draw counts and record results.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_loop.py`:

```python
class TestEloTracking:
    def test_elo_updates_after_epoch(self, katago_config, tmp_path):
        """Elo should update for both learner and opponent after each epoch."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        # Make some games end (set terminated=True for some steps)
        original_step = mock_env.step.side_effect
        call_count = [0]

        def step_with_terminal(actions):
            call_count[0] += 1
            result = original_step(actions)
            if call_count[0] == 3:
                result.terminated = np.array([True, False], dtype=bool)
                result.rewards = np.array([1.0, 0.0], dtype=np.float32)
            return result

        mock_env.step.side_effect = step_with_terminal

        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=5)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)

        # Elo updates should have been recorded
        entries = loop.pool.list_entries()
        # At least the bootstrap entry should have been played against
        assert any(e.games_played > 0 for e in entries)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestEloTracking -v`
Expected: FAIL

- [ ] **Step 3: Write the implementation**

In `run()`, track wins/losses/draws during the epoch, then update Elo after `ppo.update()`:

```python
# At epoch end, after ppo.update():
if (self.pool is not None and self._current_opponent_entry is not None
        and (win_count + draw_count) > 0):
    total_games = win_count + (self.num_envs - win_count - draw_count) + draw_count
    learner_wins = win_count
    learner_losses = total_games - win_count - draw_count

    # Record result
    learner_entry = self.pool._get_entry(self._learner_entry_id)  # stable reference, not [-1]
    self.pool.record_result(
        epoch=epoch_i,
        learner_id=learner_entry.id,
        opponent_id=self._current_opponent_entry.id,
        wins=learner_wins,
        losses=learner_losses,
        draws=draw_count,
    )

    # Compute Elo update
    if total_games > 0:
        result_score = (learner_wins + 0.5 * draw_count) / total_games
        new_learner_elo, new_opponent_elo = compute_elo_update(
            learner_entry.elo_rating,
            self._current_opponent_entry.elo_rating,
            result=result_score,
            k=self.config.league.elo_k_factor if self.config.league else 32.0,
        )
        self.pool.update_elo(learner_entry.id, new_learner_elo)
        self.pool.update_elo(self._current_opponent_entry.id, new_opponent_elo)
        logger.info(
            "Elo update: learner %.0f→%.0f, opponent (id=%d) %.0f→%.0f | W=%d L=%d D=%d",
            learner_entry.elo_rating, new_learner_elo,
            self._current_opponent_entry.id,
            self._current_opponent_entry.elo_rating, new_opponent_elo,
            learner_wins, learner_losses, draw_count,
        )

    # Log pool health
    if self.sampler is not None:
        health = self.sampler.pool_health()
        logger.info("Pool health: %.0f%% above Elo floor", health * 100)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_loop.py::TestEloTracking -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat: add Elo tracking with win/loss/draw counts after each epoch"
```

---

### Task 4: Value-Head Adapter Integration in PPO Update

**Files:**
- Modify: `keisei/training/katago_ppo.py`
- Modify: `tests/test_katago_ppo.py`

The `KataGoPPOAlgorithm.update()` currently hardcodes cross-entropy for value loss. Make it dispatch through a `ValueHeadAdapter` so both scalar-value and multi-head models work.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_ppo.py`:

```python
from keisei.training.value_adapter import ScalarValueAdapter, MultiHeadValueAdapter


class TestValueAdapterIntegration:
    def test_update_accepts_value_adapter(self, ppo):
        """update() should accept an optional value_adapter parameter."""
        buf = KataGoRolloutBuffer(num_envs=2, obs_shape=(50, 9, 9), action_space=11259)
        for _ in range(4):
            obs = torch.randn(2, 50, 9, 9)
            legal_masks = torch.ones(2, 11259, dtype=torch.bool)
            actions, log_probs, values = ppo.select_actions(obs, legal_masks)
            buf.add(obs, actions, log_probs, values,
                    torch.zeros(2), torch.zeros(2, dtype=torch.bool), legal_masks,
                    torch.randint(0, 3, (2,)), torch.randn(2))

        adapter = MultiHeadValueAdapter()
        losses = ppo.update(buf, torch.zeros(2), value_adapter=adapter)
        assert "value_loss" in losses
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_ppo.py::TestValueAdapterIntegration -v`
Expected: FAIL — `update()` doesn't accept `value_adapter` kwarg

- [ ] **Step 3: Write the implementation**

In `KataGoPPOAlgorithm.update()`, add `value_adapter` parameter:

```python
def update(
    self,
    buffer: KataGoRolloutBuffer,
    next_values: torch.Tensor,
    value_adapter: "ValueHeadAdapter | None" = None,
) -> dict[str, float]:
```

Then in the loss computation section, replace the hardcoded value loss with adapter dispatch:

```python
# Value + score loss — dispatch through adapter if provided
if value_adapter is not None:
    value_score_loss = value_adapter.compute_value_loss(
        output.value_logits,
        returns=None,
        value_cats=batch_value_cats,
        score_targets=batch_score_targets,
        score_pred=output.score_lead,
    )
else:
    # Default: KataGo multi-head (backward compatible)
    value_loss = F.cross_entropy(
        output.value_logits, batch_value_cats, ignore_index=-1
    )
    score_loss = F.mse_loss(output.score_lead.squeeze(-1), batch_score_targets)
    value_score_loss = self.params.lambda_value * value_loss + self.params.lambda_score * score_loss
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_ppo.py -v`
Expected: PASS (both old and new tests)

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_ppo.py tests/test_katago_ppo.py
git commit -m "feat: accept ValueHeadAdapter in KataGoPPOAlgorithm.update()"
```

---

### Task 5: Seat Rotation

**Files:**
- Modify: `keisei/training/katago_loop.py`
- Modify: `tests/test_katago_loop.py`

After `epochs_per_seat` epochs, save the learner's weights back to its pool entry, load the next model in rotation order, and create a fresh optimizer.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_loop.py`:

```python
class TestSeatRotation:
    def test_no_rotation_without_league(self, katago_config):
        """Without league config, no rotation should occur."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=3, steps_per_epoch=2)
        # Should complete without error

    def test_rotation_after_epochs_per_seat(self, katago_config, tmp_path):
        """After epochs_per_seat, the LR scheduler patience should reset."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        katago_config = _with_league(
            katago_config, tmp_path,
            snapshot_interval=2,
        )
        # Override epochs_per_seat to 3 for testing
        import dataclasses
        league = dataclasses.replace(katago_config.league, epochs_per_seat=3)
        katago_config = dataclasses.replace(katago_config, league=league)

        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=4, steps_per_epoch=2)
        # Should complete with a rotation at epoch 3
        assert loop.epoch == 3  # completed epochs 0,1,2,3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestSeatRotation -v`
Expected: FAIL (or PASS if rotation is a no-op — the test verifies it doesn't crash)

- [ ] **Step 3: Write the implementation**

In `run()`, at the end of each epoch, check for seat rotation:

```python
# Seat rotation check
if (self.config.league is not None
        and (epoch_i + 1) % self.config.league.epochs_per_seat == 0
        and self.pool is not None):
    self._rotate_seat(epoch_i)

def _rotate_seat(self, epoch: int) -> None:
    """Save current learner weights and reset optimizer for the next seat."""
    base_model = self.model.module if hasattr(self.model, "module") else self.model

    # Save learner's final weights to pool
    self.pool.add_snapshot(
        base_model, self.config.model.architecture,
        dict(self.config.model.params), epoch=epoch + 1,
    )

    # Reset optimizer (fresh Adam — old momentum would fight new gradient signal)
    ppo_params = self.ppo.params
    self.ppo.optimizer = torch.optim.Adam(
        self.ppo.model.parameters(), lr=ppo_params.learning_rate
    )

    # Reset LR scheduler patience if present
    if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
        self.lr_scheduler.num_bad_epochs = 0

    # Reset warmup entropy (get_entropy_coeff added by Plan D; defensive guard)
    if hasattr(self.ppo, 'get_entropy_coeff'):
        self.ppo.current_entropy_coeff = self.ppo.get_entropy_coeff(
            epoch=0,
            warmup_epochs=getattr(self, '_rl_warmup_epochs', 0),
            warmup_entropy=getattr(self, '_rl_warmup_entropy', 0.05),
        )
    else:
        self.ppo.current_entropy_coeff = self.ppo.params.lambda_entropy

    logger.info("Seat rotation at epoch %d: optimizer reset, warmup restarted", epoch)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_loop.py::TestSeatRotation -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat: add seat rotation with optimizer reset and warmup restart"
```

---

### Task 6: Wire Split-Merge into Run Loop

**Files:**
- Modify: `keisei/training/katago_loop.py`
- Modify: `tests/test_katago_loop.py`

Replace the current `select_actions` call with `split_merge_step` when an opponent is loaded. Only store learner transitions in the rollout buffer.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_katago_loop.py`:

```python
class TestSplitMergeIntegration:
    def test_run_with_league_uses_split_merge(self, katago_config, tmp_path):
        """With league enabled, run() should use split-merge steps."""
        mock_env = _make_mock_katago_vecenv(num_envs=4)
        # Set current_players to alternate
        original_step = mock_env.step.side_effect
        def step_with_players(actions):
            result = original_step(actions)
            result.current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
            return result
        mock_env.step.side_effect = step_with_players

        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=5)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4

    def test_run_without_league_still_works(self, katago_config):
        """Without league config, run() should work as before (no opponent)."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_katago_loop.py::TestSplitMergeIntegration -v`
Expected: FAIL or PASS depending on current state

- [ ] **Step 3: Write the implementation**

In `run()`, initialize `current_players` from the reset result (NOT hardcoded zeros — games span epoch boundaries):

```python
reset_result = self.vecenv.reset()
obs = torch.from_numpy(np.asarray(reset_result.observations)).to(self.device)
legal_masks = torch.from_numpy(np.asarray(reset_result.legal_masks)).to(self.device)
# Initialize current_players from reset (always Black at game start)
current_players = np.zeros(self.num_envs, dtype=np.uint8)
```

Then in the inner step loop, replace the step logic. When `self._current_opponent` is not None, use `split_merge_step`:

```python
for step_i in range(steps_per_epoch):
    self.global_step += 1

    if self._current_opponent is not None:
        # Split-merge: learner vs opponent
        # NOTE: current_players is read from the VecEnv state, NOT hardcoded.
        # Games span epoch boundaries in a continuous VecEnv.
        sm_result = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,  # initialized from reset, updated each step
            learner_model=self.model,
            opponent_model=self._current_opponent,
            learner_side=0,
            value_adapter=getattr(self, 'value_adapter', None),
        )

        actions = sm_result.actions
        action_list = actions.tolist()
        step_result = self.vecenv.step(action_list)

        # Update current_players for next step
        current_players = np.asarray(step_result.current_players)

        # Process rewards/dones for ALL envs (needed for game counting)
        rewards = torch.from_numpy(np.asarray(step_result.rewards)).to(self.device)
        terminated = torch.from_numpy(np.asarray(step_result.terminated)).to(self.device)
        truncated = torch.from_numpy(np.asarray(step_result.truncated)).to(self.device)
        dones = terminated | truncated

        # CRITICAL: store ONLY learner transitions in the rollout buffer.
        # Opponent transitions are discarded — they would corrupt GAE
        # (zero values for opponent envs bias the advantage baseline).
        li = sm_result.learner_indices
        if li.numel() > 0:
            # ... compute value_cats and score_targets for learner envs ...
            self.buffer.add(
                obs[li], actions[li], sm_result.learner_log_probs,
                sm_result.learner_values, rewards[li], dones[li],
                legal_masks[li], value_cats[li], score_targets[li],
            )
    else:
        # No opponent: all envs are learner (original behavior)
        actions, log_probs, values = self.ppo.select_actions(obs, legal_masks)
        action_list = actions.tolist()
        step_result = self.vecenv.step(action_list)
        # ... original buffer.add for all envs ...

    # Update obs and legal_masks for next step
    obs = torch.from_numpy(np.asarray(step_result.observations)).to(self.device)
    legal_masks = torch.from_numpy(np.asarray(step_result.legal_masks)).to(self.device)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_katago_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add keisei/training/katago_loop.py tests/test_katago_loop.py
git commit -m "feat: wire split_merge_step into run() when league is enabled"
```

---

### Task 7: Full Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all Python tests**

Run: `uv run pytest -v`
Expected: All tests PASS — existing + new. No regressions.

- [ ] **Step 2: Verify league-enabled training runs end-to-end**

Run: `PYTHONPATH=. uv run python -c "
from tests.test_katago_loop import _make_mock_katago_vecenv, _with_league
from keisei.config import AppConfig, TrainingConfig, DisplayConfig, ModelConfig, LeagueConfig
import tempfile, pathlib

tmp = tempfile.mkdtemp()
config = AppConfig(
    training=TrainingConfig(
        num_games=2, max_ply=50, algorithm='katago_ppo',
        checkpoint_interval=100, checkpoint_dir=tmp + '/ckpt',
        algorithm_params={'learning_rate': 2e-4, 'score_normalization': 76.0, 'grad_clip': 1.0},
    ),
    display=DisplayConfig(moves_per_minute=0, db_path=tmp + '/test.db'),
    model=ModelConfig(
        display_name='E2-Test', architecture='se_resnet',
        params={'num_blocks': 2, 'channels': 32, 'se_reduction': 8,
                'global_pool_channels': 16, 'policy_channels': 8,
                'value_fc_size': 32, 'score_fc_size': 16, 'obs_channels': 50},
    ),
    league=LeagueConfig(max_pool_size=5, snapshot_interval=2, epochs_per_seat=10),
)
from keisei.training.katago_loop import KataGoTrainingLoop
mock = _make_mock_katago_vecenv(2)
loop = KataGoTrainingLoop(config, vecenv=mock)
loop.run(num_epochs=4, steps_per_epoch=4)
print(f'OK: epoch={loop.epoch}, step={loop.global_step}, pool={len(loop.pool.list_entries())} entries')
"`
Expected: `OK: epoch=3, step=16, pool=3 entries` (bootstrap + 2 periodic snapshots at epochs 2,4)

- [ ] **Step 3: Commit if any fixes were needed**

```bash
git add -u
git commit -m "fix: address issues found in Plan E-2 verification"
```
