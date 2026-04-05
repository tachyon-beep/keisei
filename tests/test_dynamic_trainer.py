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
    """test_should_update_threshold — 4 matches needed by default."""

    def test_should_update_threshold(self, trainer, store_and_entry) -> None:
        _store, entry = store_and_entry
        # Not enough matches yet
        for i in range(3):
            trainer.record_match(entry.id, _make_rollout(side=0), 0)
            assert not trainer.should_update(entry.id), f"should_update at {i + 1} matches"

        # Fourth match triggers
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

        assert True in call_log


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

        # Prepare batch and check advantage values
        _obs, _actions, all_rewards, all_dones, _masks = trainer._prepare_batch(
            entry.id, "cpu"
        )
        advantages = all_rewards * all_dones.float()

        # Terminal step should have advantage = -1.0 (loss)
        terminal_advantages = advantages[all_dones.bool()]
        assert (terminal_advantages < 0).all(), (
            f"Expected negative advantages for losses, got {terminal_advantages}"
        )

        # Non-terminal steps should have advantage = 0.0
        non_terminal_advantages = advantages[~all_dones.bool()]
        assert (non_terminal_advantages == 0).all(), (
            f"Expected zero advantages for non-terminal, got {non_terminal_advantages}"
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
