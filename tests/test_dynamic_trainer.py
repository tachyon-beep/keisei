"""Tests for DynamicTrainer and MatchRollout."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from keisei.config import DynamicConfig
from keisei.db import init_db
from keisei.training.dynamic_trainer import DynamicTrainer, MatchRollout
from keisei.training.opponent_store import OpponentStore, Role

from tests._helpers import TinyModel, make_rollout as _make_rollout

pytestmark = pytest.mark.integration


@pytest.fixture
def store_and_entry(tmp_path):
    """Create an OpponentStore with a TinyModel entry, return (store, entry)."""
    db_path = str(tmp_path / "test.db")
    league_dir = tmp_path / "league"
    league_dir.mkdir()
    init_db(db_path)

    store = OpponentStore(db_path=db_path, league_dir=str(league_dir))
    model = TinyModel()

    with patch(
        "keisei.training.opponent_store.build_model",
        return_value=TinyModel(),
    ):
        entry = store.add_entry(
            model=model,
            architecture="tiny",
            model_params={},
            epoch=1,
            role=Role.DYNAMIC,
        )

    yield store, entry
    store.close()


@pytest.fixture
def default_config() -> DynamicConfig:
    return DynamicConfig()


@pytest.fixture
def trainer(store_and_entry, default_config):
    """Create a DynamicTrainer with default config."""
    store, _entry = store_and_entry
    return DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)


# ---------------------------------------------------------------------------
# MatchRollout tests (pre-existing)
# ---------------------------------------------------------------------------


class TestMatchRolloutConstruction:
    """test_match_rollout_construction — synthetic tensor construction."""

    def test_match_rollout_construction(self) -> None:
        steps, num_envs, obs_channels, action_space = 10, 3, 50, 11259

        observations = torch.zeros(steps, num_envs, obs_channels, 9, 9)
        actions = torch.zeros(steps, num_envs, dtype=torch.long)
        rewards = torch.zeros(steps, num_envs)
        dones = torch.zeros(steps, num_envs)
        legal_masks = torch.zeros(steps, num_envs, action_space)
        perspective = torch.zeros(steps, num_envs, dtype=torch.long)

        rollout = MatchRollout(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            legal_masks=legal_masks,
            perspective=perspective,
        )

        # All fields accessible
        assert rollout.observations is observations
        assert rollout.actions is actions
        assert rollout.rewards is rewards
        assert rollout.dones is dones
        assert rollout.legal_masks is legal_masks
        assert rollout.perspective is perspective

        # Correct shapes
        assert rollout.observations.shape == (steps, num_envs, obs_channels, 9, 9)
        assert rollout.actions.shape == (steps, num_envs)
        assert rollout.rewards.shape == (steps, num_envs)
        assert rollout.dones.shape == (steps, num_envs)
        assert rollout.legal_masks.shape == (steps, num_envs, action_space)
        assert rollout.perspective.shape == (steps, num_envs)

        # All on CPU
        assert rollout.observations.device.type == "cpu"
        assert rollout.actions.device.type == "cpu"
        assert rollout.rewards.device.type == "cpu"
        assert rollout.dones.device.type == "cpu"
        assert rollout.legal_masks.device.type == "cpu"
        assert rollout.perspective.device.type == "cpu"


class TestMatchRolloutFilterByPerspective:
    """test_match_rollout_filter_by_perspective — filter by player perspective."""

    def test_match_rollout_filter_by_perspective(self) -> None:
        steps, num_envs, obs_channels, action_space = 10, 3, 50, 11259

        observations = torch.randn(steps, num_envs, obs_channels, 9, 9)
        actions = torch.randint(0, action_space, (steps, num_envs))
        rewards = torch.randn(steps, num_envs)
        dones = torch.zeros(steps, num_envs)
        legal_masks = torch.ones(steps, num_envs, action_space)

        # Create mixed perspectives: first 6 steps player A (0), last 4 player B (1)
        perspective = torch.zeros(steps, num_envs, dtype=torch.long)
        perspective[6:, :] = 1

        rollout = MatchRollout(
            observations=observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
            legal_masks=legal_masks,
            perspective=perspective,
        )

        # Filter where perspective == 0 (player A)
        mask = rollout.perspective == 0
        filtered_obs = rollout.observations[mask]
        filtered_actions = rollout.actions[mask]
        filtered_rewards = rollout.rewards[mask]

        # 6 steps * 3 envs = 18 entries for player A
        expected_count = 6 * num_envs
        assert filtered_obs.shape[0] == expected_count
        assert filtered_actions.shape[0] == expected_count
        assert filtered_rewards.shape[0] == expected_count

        # Verify player B count
        mask_b = rollout.perspective == 1
        filtered_b = rollout.observations[mask_b]
        expected_b = 4 * num_envs
        assert filtered_b.shape[0] == expected_b


# ---------------------------------------------------------------------------
# DynamicTrainer tests (Task 6 + 7)
# ---------------------------------------------------------------------------


class TestShouldUpdateThreshold:
    """test_should_update_threshold — update_every_matches matches needed."""

    def test_should_update_threshold(self, trainer, store_and_entry) -> None:
        _store, entry = store_and_entry
        threshold = trainer.config.update_every_matches
        # Not enough matches yet
        for i in range(threshold - 1):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)
            assert not trainer.should_update(entry.id), f"should_update at {i + 1} matches"

        # Threshold-th match triggers
        trainer.record_match(entry.id, _make_rollout(side=0), 0)
        assert trainer.should_update(entry.id)


class TestIsRateLimited:
    """test_is_rate_limited — mock time for rate limiting."""

    def test_is_rate_limited(self, trainer) -> None:
        # Fill up rate limit window
        config = trainer.config
        now = time.monotonic()
        for _ in range(config.max_updates_per_minute):
            trainer._update_timestamps.append(now)

        assert trainer.is_rate_limited()

    def test_not_rate_limited_after_expiry(self, trainer) -> None:
        config = trainer.config
        old_time = time.monotonic() - 61.0  # >60 seconds ago
        for _ in range(config.max_updates_per_minute):
            trainer._update_timestamps.append(old_time)

        assert not trainer.is_rate_limited()

    def test_rate_limited_at_exact_boundary(self, trainer) -> None:
        """Timestamps at exactly 60s ago are kept (>= cutoff), so still rate-limited."""
        config = trainer.config
        frozen_now = 1000.0  # arbitrary fixed point
        boundary_time = frozen_now - 60.0  # exactly 60 seconds ago
        for _ in range(config.max_updates_per_minute):
            trainer._update_timestamps.append(boundary_time)

        # Freeze time so cutoff = 1000.0 - 60.0 = 940.0 == boundary_time
        with patch("keisei.training.dynamic_trainer.time") as mock_time:
            mock_time.monotonic.return_value = frozen_now
            # cutoff = now - 60.0; timestamps at cutoff satisfy t >= cutoff
            assert trainer.is_rate_limited()


class TestUpdateModifiesWeights:
    """test_update_modifies_weights — parameter values differ after update."""

    def test_update_modifies_weights(self, store_and_entry, default_config) -> None:
        store, entry = store_and_entry

        # Load parameters before update
        ckpt_path = Path(entry.checkpoint_path)
        before_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        before_params = {k: v.clone() for k, v in before_state.items()}

        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)

        # Record enough matches
        for _ in range(default_config.update_every_matches):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            result = trainer.update(entry, "cpu")

        assert result is True

        # Load parameters after update and compare
        after_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        any_changed = any(
            not torch.equal(before_params[k], after_state[k])
            for k in before_params
        )
        assert any_changed, "Expected at least one parameter tensor to change after update"


class TestUpdateUsesLrScale:
    """test_update_uses_lr_scale — verify optimizer lr = learner_lr * lr_scale."""

    def test_update_uses_lr_scale(self, store_and_entry) -> None:
        store, entry = store_and_entry
        learner_lr = 1e-3
        config = DynamicConfig(lr_scale=0.5)
        trainer = DynamicTrainer(store=store, config=config, learner_lr=learner_lr)

        for _ in range(config.update_every_matches):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            trainer.update(entry, "cpu")

        optimizer = trainer._optimizers[entry.id]
        expected_lr = learner_lr * config.lr_scale
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_lr) < 1e-10


class TestUpdateCallsModelTrain:
    """test_update_calls_model_train — verify model switches to train mode."""

    def test_update_calls_model_train(self, store_and_entry, default_config) -> None:
        store, entry = store_and_entry
        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)

        for _ in range(default_config.update_every_matches):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)

        call_log: list[bool] = []
        original_train = TinyModel.train

        def patched_train(self_model, mode=True):
            call_log.append(mode)
            return original_train(self_model, mode)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ), patch.object(TinyModel, "train", patched_train):
            trainer.update(entry, "cpu")

        assert len(call_log) > 0, "model.train() was never called"
        # load_opponent calls model.eval() (train(False)) first, then
        # _update_inner calls model.train() (train(True)) for training.
        # Verify train(True) was called and is the last mode set before training.
        assert True in call_log, "model.train(True) was never called"
        last_true_idx = max(i for i, m in enumerate(call_log) if m is True)
        modes_after_train = call_log[last_true_idx + 1:]
        assert all(m is True for m in modes_after_train), (
            f"model switched to eval mode after train(True): {call_log}"
        )


class TestUpdateSavesWeightsAfterUpdate:
    """test_update_saves_weights_after_update — checkpoint file modified."""

    def test_update_saves_weights_after_update(self, store_and_entry, default_config) -> None:
        store, entry = store_and_entry
        ckpt_path = Path(entry.checkpoint_path)
        bytes_before = ckpt_path.read_bytes()

        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)

        for _ in range(default_config.update_every_matches):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            trainer.update(entry, "cpu")

        bytes_after = ckpt_path.read_bytes()
        assert bytes_before != bytes_after


class TestUpdateSavesOptimizerAtFlushInterval:
    """test_update_saves_optimizer_at_flush_interval."""

    def test_saves_at_flush_interval(self, store_and_entry) -> None:
        store, entry = store_and_entry
        # checkpoint_flush_every=2 so optimizer is saved on 2nd batch of matches
        config = DynamicConfig(
            update_every_matches=1,
            checkpoint_flush_every=2,
        )
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            # First update: total_matches becomes 1, not divisible by 2
            trainer.record_match(entry.id, _make_rollout(side=0), 0)
            trainer.update(entry, "cpu")
            # Confirm the update actually ran (update_count incremented)
            refreshed = store.get_entry(entry.id)
            assert refreshed is not None and refreshed.update_count == 1, (
                "Update should have run (update_count == 1)"
            )
            # Optimizer not yet saved because flush interval hasn't triggered
            assert store.load_optimizer(entry.id) is None

            # Second update: total_matches becomes 2, divisible by 2 -> save
            # Need to re-fetch entry to get updated state
            entry2 = store.get_entry(entry.id)
            trainer.record_match(entry.id, _make_rollout(side=0), 0)
            trainer.update(entry2, "cpu")
            assert store.load_optimizer(entry.id) is not None


class TestGetUpdateStats:
    """test_get_update_stats — returns (count, timestamp)."""

    def test_get_update_stats(self, store_and_entry, default_config) -> None:
        store, entry = store_and_entry
        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)

        count, ts = trainer.get_update_stats(entry.id)
        assert count == 0
        assert ts is None

        for _ in range(default_config.update_every_matches):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            trainer.update(entry, "cpu")

        count, ts = trainer.get_update_stats(entry.id)
        assert count == 1
        assert ts is not None

    def test_get_update_stats_missing_entry(self, store_and_entry, default_config) -> None:
        store, _entry = store_and_entry
        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)
        count, ts = trainer.get_update_stats(99999)
        assert count == 0
        assert ts is None


class TestRecordMatchCapsBuffer:
    """test_record_match_caps_buffer — max_buffer_depth=3, record 5."""

    def test_caps_buffer(self, store_and_entry) -> None:
        store, entry = store_and_entry
        config = DynamicConfig(max_buffer_depth=3)
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        for _ in range(5):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)

        assert len(trainer._rollout_buffers[entry.id]) == 3


class TestBufferClearedAfterUpdate:
    """test_buffer_cleared_after_update — buffer empty after update."""

    def test_buffer_cleared(self, store_and_entry, default_config) -> None:
        store, entry = store_and_entry
        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)

        for _ in range(default_config.update_every_matches):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)

        assert len(trainer._rollout_buffers[entry.id]) > 0

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            trainer.update(entry, "cpu")

        assert len(trainer._rollout_buffers[entry.id]) == 0


class TestUpdateUsesRewardSignedAdvantages:
    """test_update_uses_reward_signed_advantages — wins get positive, losses negative."""

    def test_reward_signed_advantages(self, store_and_entry) -> None:
        store, entry = store_and_entry
        config = DynamicConfig(update_every_matches=1, update_epochs_per_batch=1)
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        # Create a rollout with a terminal loss (reward=-1)
        rollout_loss = _make_rollout(steps=5, side=0, include_terminal=True)
        rollout_loss.rewards[-1, :] = -1.0  # loss

        trainer.record_match(entry.id, rollout_loss, 0)

        # Spy on ppo_clip_loss to capture the actual advantages used in training
        captured_advantages: list[torch.Tensor] = []
        from keisei.training.katago_ppo import ppo_clip_loss as real_ppo_clip_loss

        def spy_ppo_clip_loss(new_lp, old_lp, advantages, **kwargs):
            captured_advantages.append(advantages.detach().clone())
            return real_ppo_clip_loss(new_lp, old_lp, advantages, **kwargs)

        with patch(
            "keisei.training.dynamic_trainer.ppo_clip_loss",
            side_effect=spy_ppo_clip_loss,
        ), patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            result = trainer.update(entry, "cpu")

        assert result is True
        assert len(captured_advantages) == 1, "Expected 1 epoch of training"
        advs = captured_advantages[0]

        # Loss terminal steps should produce negative advantages
        assert (advs < 0).any(), (
            f"Expected at least one negative advantage for loss, got {advs}"
        )
        # Non-terminal steps should have zero advantage (reward=0 * done=0)
        assert (advs == 0).any(), (
            f"Expected some zero advantages for non-terminal steps, got {advs}"
        )


# ---------------------------------------------------------------------------
# Task 7: Error fallback tests
# ---------------------------------------------------------------------------


class TestErrorFallbackRetriesBeforeDisable:
    """test_error_fallback_retries_before_disable — 3 failures needed."""

    def test_retries_before_disable(self, store_and_entry) -> None:
        store, entry = store_and_entry
        config = DynamicConfig(
            update_every_matches=1,
            max_consecutive_errors=3,
            disable_on_error=True,
        )
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        def bad_load(*args, **kwargs):
            raise RuntimeError("simulated failure")

        with patch.object(store, "load_opponent", side_effect=bad_load):
            for i in range(3):
                trainer.record_match(entry.id, _make_rollout(side=0), 0)
                result = trainer.update(entry, "cpu")
                assert result is False
                if i < 2:
                    # Not yet disabled
                    assert entry.id not in trainer._disabled_entries

            # After 3 consecutive errors, entry should be disabled
            assert entry.id in trainer._disabled_entries


class TestErrorCountResetsOnSuccess:
    """test_error_count_resets_on_success — success resets counter."""

    def test_resets_on_success(self, store_and_entry) -> None:
        store, entry = store_and_entry
        config = DynamicConfig(
            update_every_matches=1,
            max_consecutive_errors=3,
            disable_on_error=True,
        )
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        def bad_load(*args, **kwargs):
            raise RuntimeError("simulated failure")

        # Two failures
        with patch.object(store, "load_opponent", side_effect=bad_load):
            for _ in range(2):
                trainer.record_match(entry.id, _make_rollout(side=0), 0)
                trainer.update(entry, "cpu")

        assert trainer._error_counts[entry.id] == 2

        # One success resets it
        trainer.record_match(entry.id, _make_rollout(side=0), 0)
        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            result = trainer.update(entry, "cpu")
        assert result is True
        assert trainer._error_counts[entry.id] == 0


class TestErrorFallbackDisabledSetting:
    """test_error_fallback_disabled_setting — disable_on_error=False re-raises."""

    def test_reraises_when_disabled(self, store_and_entry) -> None:
        store, entry = store_and_entry
        config = DynamicConfig(
            update_every_matches=1,
            disable_on_error=False,
        )
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        def bad_load(*args, **kwargs):
            raise RuntimeError("simulated failure")

        trainer.record_match(entry.id, _make_rollout(side=0), 0)

        with patch.object(store, "load_opponent", side_effect=bad_load):
            with pytest.raises(RuntimeError, match="simulated failure"):
                trainer.update(entry, "cpu")


# ---------------------------------------------------------------------------
# Disabled-entry path tests
# ---------------------------------------------------------------------------


class TestRecordMatchDisabledEntryIgnored:
    """record_match on a disabled entry does nothing."""

    def test_record_match_disabled_entry_ignored(self, store_and_entry, default_config) -> None:
        store, entry = store_and_entry
        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)

        # Disable the entry
        trainer._disabled_entries.add(entry.id)

        # Record a match — should be silently ignored
        trainer.record_match(entry.id, _make_rollout(side=0), 0)

        # Buffer should not be populated
        assert entry.id not in trainer._rollout_buffers
        # Match count should not be incremented
        assert trainer._match_counts.get(entry.id, 0) == 0


class TestShouldUpdateDisabledEntryFalse:
    """should_update returns False for disabled entries even with enough matches."""

    def test_should_update_disabled_entry_false(self, store_and_entry, default_config) -> None:
        store, entry = store_and_entry
        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)

        # Record enough matches BEFORE disabling
        for _ in range(default_config.update_every_matches):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)

        # Verify it would normally trigger an update
        assert trainer.should_update(entry.id) is True

        # Now disable the entry
        trainer._disabled_entries.add(entry.id)

        # should_update must return False
        assert trainer.should_update(entry.id) is False


class TestUpdateEmptyBatchReturnsFalse:
    """Trainer with no recorded matches calls update -> returns False."""

    def test_update_empty_batch_returns_false(self, store_and_entry, default_config) -> None:
        store, entry = store_and_entry
        trainer = DynamicTrainer(store=store, config=default_config, learner_lr=1e-3)

        # No matches recorded at all — _prepare_batch returns empty tensors
        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ):
            result = trainer.update(entry, "cpu")

        assert result is False


class TestPostCheckpointFailureStillReturnsTrue:
    """Post-checkpoint bookkeeping failures must not count as training failures."""

    def test_save_optimizer_failure_returns_true(self, store_and_entry) -> None:
        """If save_optimizer raises after weights are committed, update() still returns True."""
        store, entry = store_and_entry
        config = DynamicConfig(
            update_every_matches=1,
            checkpoint_flush_every=1,  # force optimizer save on every update
            max_consecutive_errors=1,  # would disable on first real error
            disable_on_error=True,
        )
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)
        trainer.record_match(entry.id, _make_rollout(side=0), 0)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ), patch.object(
            store, "save_optimizer", side_effect=RuntimeError("DB write failed"),
        ):
            result = trainer.update(entry, "cpu")

        # Should succeed — weights were saved, only bookkeeping failed
        assert result is True
        # Should NOT be disabled
        assert entry.id not in trainer._disabled_entries
        # Error count should be reset (successful training)
        assert trainer._error_counts.get(entry.id, 0) == 0


class TestOldLogProbsComputedInEvalMode:
    """old_log_probs must be computed in eval mode to match rollout BN behavior."""

    def test_old_log_probs_use_eval_mode(self, store_and_entry) -> None:
        """Verify model is in eval mode during old_log_probs forward pass,
        then switches to train mode for the gradient update loop."""
        store, entry = store_and_entry
        config = DynamicConfig(update_every_matches=1, update_epochs_per_batch=1)
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)
        trainer.record_match(entry.id, _make_rollout(side=0), 0)

        # Track model.training state at each forward() call
        forward_training_states: list[bool] = []
        original_forward = TinyModel.forward

        def tracking_forward(self_model, x):
            forward_training_states.append(self_model.training)
            return original_forward(self_model, x)

        with patch(
            "keisei.training.opponent_store.build_model",
            return_value=TinyModel(),
        ), patch.object(TinyModel, "forward", tracking_forward):
            trainer.update(entry, "cpu")

        # First forward call = old_log_probs (should be eval mode = False)
        # Subsequent calls = training updates (should be train mode = True)
        assert len(forward_training_states) >= 2, (
            f"Expected at least 2 forward calls, got {len(forward_training_states)}"
        )
        assert forward_training_states[0] is False, (
            "old_log_probs forward pass should be in eval mode (model.training=False)"
        )
        assert all(s is True for s in forward_training_states[1:]), (
            f"Training forward passes should be in train mode, got {forward_training_states[1:]}"
        )


class TestGlobalDisableFallback:
    """Tests for §10.4 global inference-only fallback (_globally_disabled)."""

    def test_global_disable_triggers_on_threshold(self, store_and_entry) -> None:
        """When global error count reaches threshold, training is globally disabled."""
        store, entry = store_and_entry
        config = DynamicConfig(
            update_every_matches=1,
            disable_on_error=True,
            max_consecutive_errors=100,  # high so per-entry disable doesn't trigger first
            global_error_threshold=3,
            global_error_window_seconds=300.0,
        )
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        assert not trainer.is_globally_disabled

        # Simulate 3 global errors by directly adding timestamps
        now = time.monotonic()
        trainer._global_error_timestamps = [now - 10, now - 5, now]
        trainer._check_global_disable()

        assert trainer.is_globally_disabled
        assert trainer.should_update(entry.id) is False

    def test_global_disable_respects_window(self, store_and_entry) -> None:
        """Errors outside the window don't count toward the threshold."""
        store, entry = store_and_entry
        config = DynamicConfig(
            update_every_matches=1,
            disable_on_error=True,
            max_consecutive_errors=100,
            global_error_threshold=3,
            global_error_window_seconds=60.0,
        )
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        # 2 recent errors + 1 expired error = only 2 in window
        now = time.monotonic()
        trainer._global_error_timestamps = [now - 120, now - 5, now]
        trainer._check_global_disable()

        assert not trainer.is_globally_disabled

    def test_global_disable_via_update_errors(self, store_and_entry) -> None:
        """Repeated update() failures trigger global disable through the real code path."""
        store, entry = store_and_entry
        config = DynamicConfig(
            update_every_matches=1,
            disable_on_error=True,
            max_consecutive_errors=100,  # won't hit per-entry disable
            global_error_threshold=2,
            global_error_window_seconds=300.0,
        )
        trainer = DynamicTrainer(store=store, config=config, learner_lr=1e-3)

        # Force update to fail by patching load_opponent to raise
        with patch(
            "keisei.training.opponent_store.build_model",
            side_effect=RuntimeError("model load failed"),
        ):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)
            trainer.update(entry, "cpu")  # error 1

            trainer._match_counts[entry.id] = config.update_every_matches
            trainer.update(entry, "cpu")  # error 2

        assert trainer.is_globally_disabled
        assert trainer.should_update(entry.id) is False
