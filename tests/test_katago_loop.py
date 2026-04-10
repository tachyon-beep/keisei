# tests/test_katago_loop.py
"""Unit tests for KataGoTrainingLoop (mocked I/O)."""

import dataclasses
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from keisei.config import AppConfig, DisplayConfig, LeagueConfig, ModelConfig, TrainingConfig
from keisei.db import update_training_progress
from keisei.training.distributed import DistributedContext
from keisei.training.katago_loop import KataGoTrainingLoop


def _make_mock_katago_vecenv(
    num_envs: int = 2, *, terminate_at_step: int | None = None,
    alternate_players: bool = False,
    material_balance: int = 0,
) -> MagicMock:
    """Create a mock VecEnv that returns correct shapes for KataGo mode.

    Args:
        terminate_at_step: If set, env 0 terminates with reward +1.0 at this
            step (1-indexed). This exercises the value categorization and
            score target branches.
    """
    rng = np.random.default_rng(42)
    mock = MagicMock()
    mock.observation_channels = 50
    mock.action_space_size = 11259
    mock.episodes_completed = 0
    mock.mean_episode_length = 0.0
    mock.truncation_rate = 0.0
    mock.draw_rate = 0.0
    step_count = [0]

    def make_reset_result():
        result = MagicMock()
        result.observations = rng.standard_normal((num_envs, 50, 9, 9)).astype(
            np.float32
        )
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        return result

    def make_step_result(actions):
        step_count[0] += 1
        result = MagicMock()
        result.observations = rng.standard_normal((num_envs, 50, 9, 9)).astype(
            np.float32
        )
        result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
        result.rewards = np.zeros(num_envs, dtype=np.float32)
        result.terminated = np.zeros(num_envs, dtype=bool)
        result.truncated = np.zeros(num_envs, dtype=bool)
        if alternate_players:
            # Alternate: even steps = all Black, odd steps = all White
            result.current_players = np.full(
                num_envs, step_count[0] % 2, dtype=np.uint8,
            )
        else:
            result.current_players = np.zeros(num_envs, dtype=np.uint8)

        # step_metadata with material balance (per-step, not terminal-only)
        result.step_metadata = MagicMock()
        result.step_metadata.ply_count = np.zeros(num_envs, dtype=np.uint16)
        result.step_metadata.material_balance = np.full(num_envs, material_balance, dtype=np.int32)

        if terminate_at_step is not None and step_count[0] == terminate_at_step:
            result.terminated[0] = True
            result.rewards[0] = 1.0

        return result

    mock.reset.side_effect = lambda: make_reset_result()
    mock.step.side_effect = make_step_result
    mock.reset_stats = MagicMock()
    return mock


def _make_config(tmp_path: Path | None = None) -> AppConfig:
    """Create a minimal AppConfig for testing.

    Uses a temp directory for checkpoint_dir and db_path. If tmp_path is
    None, uses /tmp with a unique suffix.
    """
    import tempfile

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    return AppConfig(
        training=TrainingConfig(
            num_games=2,
            max_ply=50,
            algorithm="katago_ppo",
            checkpoint_interval=5,
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
    )


class TestDDPInit:
    def test_training_loop_accepts_dist_context(self):
        """KataGoTrainingLoop accepts a DistributedContext."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
        assert loop.dist_ctx is ctx
        assert loop.dist_ctx.is_main is True

    def test_non_distributed_backward_compatible(self):
        """Omitting dist_ctx gives a non-distributed context."""
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        assert loop.dist_ctx.is_distributed is False
        assert loop.dist_ctx.world_size == 1


class TestRankGating:
    def test_non_main_rank_skips_checkpoint(self):
        """Non-main rank should not write checkpoints."""
        ctx = DistributedContext(rank=1, local_rank=1, world_size=2, is_distributed=True)
        config = _make_config()
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, checkpoint_interval=1),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.init_db"), \
             patch("keisei.training.katago_loop.read_training_state", return_value=None), \
             patch("keisei.training.katago_loop.write_training_state"), \
             patch("keisei.training.katago_loop.DDP", side_effect=lambda m, **kw: m), \
             patch("keisei.training.katago_loop.dist.barrier"), \
             patch("keisei.training.katago_loop.dist.broadcast_object_list"):
            loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)

        with patch("keisei.training.katago_loop.save_checkpoint") as mock_save, \
             patch("keisei.training.katago_loop.dist.barrier"), \
             patch("keisei.training.katago_loop.dist.all_reduce"):
            loop.run(num_epochs=1, steps_per_epoch=2)
            mock_save.assert_not_called()

    def test_main_rank_writes_checkpoint(self):
        """Main rank should write checkpoints normally."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        config = _make_config()
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(config.training, checkpoint_interval=1),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)

        with patch("keisei.training.katago_loop.save_checkpoint") as mock_save:
            loop.run(num_epochs=1, steps_per_epoch=2)
            assert mock_save.call_count >= 1

    def test_non_main_rank_skips_metrics(self):
        """Non-main rank should not write metrics to DB."""
        ctx = DistributedContext(rank=1, local_rank=1, world_size=2, is_distributed=True)
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.init_db"), \
             patch("keisei.training.katago_loop.read_training_state", return_value=None), \
             patch("keisei.training.katago_loop.write_training_state"), \
             patch("keisei.training.katago_loop.DDP", side_effect=lambda m, **kw: m), \
             patch("keisei.training.katago_loop.dist.barrier"), \
             patch("keisei.training.katago_loop.dist.broadcast_object_list"):
            loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)

        with patch("keisei.training.katago_loop.write_epoch_summary") as mock_write, \
             patch("keisei.training.katago_loop.dist.barrier"), \
             patch("keisei.training.katago_loop.dist.all_reduce"):
            loop.run(num_epochs=1, steps_per_epoch=2)
            mock_write.assert_not_called()


class TestCreateLrSchedulerUnknownType:
    """M1: create_lr_scheduler() raises ValueError for unknown schedule type."""

    def test_unknown_schedule_type_raises(self):
        """Passing an unknown schedule_type should raise ValueError."""
        import torch

        from keisei.training.katago_loop import create_lr_scheduler

        dummy_model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)

        with pytest.raises(ValueError, match="Unknown schedule type 'cosine'"):
            create_lr_scheduler(optimizer, schedule_type="cosine")


class TestLrSchedulerPrivateInternals:
    """Guard: ReduceLROnPlateau private attrs used by warmup-boundary reset.

    katago_loop.py resets the scheduler at the warmup boundary by writing
    to `best`, `mode_worse`, and `num_bad_epochs`. These are undocumented
    PyTorch internals. If a PyTorch upgrade removes or renames them, this
    test fails in CI before the training loop silently breaks at runtime.
    """

    def test_reduce_lr_on_plateau_has_required_attrs(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            torch.optim.Adam([torch.zeros(1)], lr=1e-3), mode="min",
        )
        assert hasattr(scheduler, "best"), "ReduceLROnPlateau missing 'best'"
        assert hasattr(scheduler, "mode_worse"), "ReduceLROnPlateau missing 'mode_worse'"
        assert hasattr(scheduler, "num_bad_epochs"), "ReduceLROnPlateau missing 'num_bad_epochs'"

    def test_warmup_boundary_reset_works(self):
        """Simulate the warmup-boundary reset and verify it actually resets tracking."""
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            torch.optim.Adam([torch.zeros(1)], lr=1e-3), mode="min", patience=2,
        )
        # Feed a good metric then worse ones to build up state
        scheduler.step(1.0)
        scheduler.step(2.0)
        assert scheduler.num_bad_epochs > 0

        # Apply the same reset that katago_loop.py does at the warmup boundary
        scheduler.best = scheduler.mode_worse
        scheduler.num_bad_epochs = 0

        assert scheduler.num_bad_epochs == 0
        # After reset, the next metric should become the new 'best'
        scheduler.step(5.0)
        assert scheduler.best == 5.0


class TestArchitectureAlgorithmMismatchGuard:
    """H4: Guard that rejects incompatible architecture/algorithm combinations."""

    def test_resnet_rejected_for_katago_ppo(self, tmp_path):
        """algorithm='katago_ppo' with architecture='resnet' must raise ValueError."""
        config = AppConfig(
            training=TrainingConfig(
                num_games=2,
                max_ply=50,
                algorithm="katago_ppo",
                checkpoint_interval=5,
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
                display_name="Test-ResNet",
                architecture="resnet",
                params={
                    "num_blocks": 2,
                    "channels": 32,
                    "obs_channels": 50,
                },
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with pytest.raises(ValueError, match="algorithm='katago_ppo' requires a KataGoBaseModel"):
            KataGoTrainingLoop(config, vecenv=mock_env)

    def test_obs_channel_mismatch_raises(self, tmp_path):
        """VecEnv with wrong observation_channels must raise ValueError."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        mock_env.observation_channels = 42  # config expects 50
        with pytest.raises(ValueError, match="observation"):
            KataGoTrainingLoop(config, vecenv=mock_env)

    def test_action_space_mismatch_raises(self, tmp_path):
        """VecEnv with wrong action_space_size must raise ValueError."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        mock_env.action_space_size = 9999  # expected 11259
        with pytest.raises(ValueError, match="action space"):
            KataGoTrainingLoop(config, vecenv=mock_env)


class TestMaybeUpdateHeartbeat:
    """C2: _maybe_update_heartbeat() time guard."""

    def test_heartbeat_fires_after_10_seconds(self, tmp_path):
        """When >= 10s have elapsed, heartbeat should update the DB."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        with patch("keisei.training.katago_loop.update_training_progress") as mock_update:
            # Simulate 11 seconds elapsed
            loop._last_heartbeat = time.monotonic() - 11.0
            old_heartbeat = loop._last_heartbeat
            loop._maybe_update_heartbeat()

            mock_update.assert_called_once()
            # _last_heartbeat should have been refreshed
            assert loop._last_heartbeat > old_heartbeat

    def test_heartbeat_skipped_within_10_seconds(self, tmp_path):
        """When < 10s have elapsed, heartbeat should NOT fire."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        with patch("keisei.training.katago_loop.update_training_progress") as mock_update:
            # Just set heartbeat to now — well within the 10s window
            loop._last_heartbeat = time.monotonic()
            loop._maybe_update_heartbeat()

            mock_update.assert_not_called()

    def test_heartbeat_db_error_does_not_crash(self, tmp_path):
        """A transient DB error in _maybe_update_heartbeat must not crash the loop."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        with patch(
            "keisei.training.katago_loop.update_training_progress",
            side_effect=OSError("disk full"),
        ):
            loop._last_heartbeat = time.monotonic() - 11.0
            # Should NOT raise — error is caught and logged
            loop._maybe_update_heartbeat()

    def test_snapshot_db_error_does_not_crash(self, tmp_path):
        """A transient DB error in _maybe_write_snapshots must not crash the loop."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        loop.moves_per_minute = 60
        loop._last_snapshot_time = time.monotonic() - 120.0

        # Give the vecenv spectator data so we reach the DB write
        mock_env.get_spectator_data.return_value = [
            {"board": [], "hands": {}, "ply": 1, "is_over": False},
        ]

        with patch(
            "keisei.training.katago_loop.write_game_snapshots",
            side_effect=OSError("disk full"),
        ):
            # Should NOT raise — error is caught and logged
            loop._maybe_write_snapshots()


class TestValueCategoryNoLeague:
    """C1: Value category assignment in the no-league (no opponent) path."""

    def test_value_cats_win_draw_loss(self, tmp_path):
        """Verify value_cat mapping: WIN(>0)=0, DRAW(==0)=1, LOSS(<0)=2."""
        # Create a vecenv that terminates all 3 envs at step 2 with distinct rewards
        num_envs = 3
        rng = np.random.default_rng(99)
        step_count = [0]

        mock_env = MagicMock()
        mock_env.observation_channels = 50
        mock_env.action_space_size = 11259
        mock_env.episodes_completed = 0

        def make_reset():
            result = MagicMock()
            result.observations = rng.standard_normal((num_envs, 50, 9, 9)).astype(np.float32)
            result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
            return result

        def make_step(actions):
            step_count[0] += 1
            result = MagicMock()
            result.observations = rng.standard_normal((num_envs, 50, 9, 9)).astype(np.float32)
            result.legal_masks = np.ones((num_envs, 11259), dtype=bool)
            result.rewards = np.zeros(num_envs, dtype=np.float32)
            result.terminated = np.zeros(num_envs, dtype=bool)
            result.truncated = np.zeros(num_envs, dtype=bool)
            result.current_players = np.zeros(num_envs, dtype=np.uint8)
            result.step_metadata = MagicMock()
            result.step_metadata.material_balance = np.zeros(num_envs, dtype=np.int32)

            # At step 2, terminate all envs with +1, 0, -1 rewards
            if step_count[0] == 2:
                result.terminated[:] = True
                result.rewards[0] = 1.0   # WIN
                result.rewards[1] = 0.0   # DRAW
                result.rewards[2] = -1.0  # LOSS

            return result

        mock_env.reset.side_effect = lambda: make_reset()
        mock_env.step.side_effect = make_step
        mock_env.reset_stats = MagicMock()

        # Override num_games to 3 to match our mock
        config = dataclasses.replace(
            _make_config(tmp_path),
            training=dataclasses.replace(_make_config(tmp_path).training, num_games=num_envs),
        )

        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        # Intercept buffer.add to capture value_cats
        captured_value_cats = []
        original_add = loop.buffer.add

        def spy_add(*args, **kwargs):
            # value_cats is the 9th positional arg (index 8) — index 7 was before terminated was added
            captured_value_cats.append(args[8].clone())
            return original_add(*args, **kwargs)

        loop.buffer.add = spy_add  # type: ignore[method-assign]

        # Run 1 epoch with 4 steps; termination at step 2
        loop.run(num_epochs=1, steps_per_epoch=4)

        # Find the step where termination occurred (step 2 -> second call)
        assert len(captured_value_cats) >= 2, f"Expected >=2 buffer.add calls, got {len(captured_value_cats)}"

        # The second add call (step 2) should have terminal value_cats
        terminal_cats = captured_value_cats[1]
        assert terminal_cats[0].item() == 0, "WIN (reward > 0) should map to value_cat=0"
        assert terminal_cats[1].item() == 1, "DRAW (reward == 0) should map to value_cat=1"
        assert terminal_cats[2].item() == 2, "LOSS (reward < 0) should map to value_cat=2"

        # Non-terminal steps should have value_cat=-1
        nonterminal_cats = captured_value_cats[0]
        assert (nonterminal_cats == -1).all(), "Non-terminal steps should have value_cat=-1"


class TestSwallowedExceptions:
    """CRIT-1: Swallowed exceptions in run() must not crash training."""

    def test_write_epoch_summary_failure_continues_training(self, tmp_path):
        """If write_epoch_summary raises, training should continue to the next epoch."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        with patch("keisei.training.katago_loop.write_epoch_summary",
                   side_effect=RuntimeError("DB write failed")):
            # Should NOT raise — the exception is caught and logged
            loop.run(num_epochs=2, steps_per_epoch=2)

        assert loop.global_step == 4  # both epochs completed

    def test_update_training_progress_failure_continues(self, tmp_path):
        """If update_training_progress raises mid-epoch, training continues."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        call_count = [0]
        original_fn = update_training_progress

        def failing_update(*args, **kwargs):
            call_count[0] += 1
            # Fail on the post-epoch call (not heartbeat calls)
            if len(args) >= 3 or "step" in kwargs:
                raise RuntimeError("progress update failed")
            return original_fn(*args, **kwargs)

        with patch("keisei.training.katago_loop.update_training_progress",
                   side_effect=RuntimeError("progress update failed")):
            loop.run(num_epochs=2, steps_per_epoch=2)

        assert loop.global_step == 4

    def test_checkpoint_save_failure_continues(self, tmp_path):
        """If save_checkpoint raises, training should continue past the checkpoint epoch."""
        config = _make_config(tmp_path)
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        with patch("keisei.training.katago_loop.save_checkpoint",
                   side_effect=OSError("disk full")):
            # Run 6 epochs (checkpoint at epoch 4, then continues to 5)
            loop.run(num_epochs=6, steps_per_epoch=2)

        assert loop.global_step == 12  # all 6 epochs completed
        # No checkpoint file should exist since save was mocked to fail
        ckpt_dir = tmp_path / "checkpoints"
        assert not (ckpt_dir / "epoch_00004.pt").exists()


class TestDDPDBInit:
    def test_non_main_rank_skips_db_init(self):
        """Non-main rank should not call init_db."""
        ctx = DistributedContext(rank=1, local_rank=1, world_size=2, is_distributed=True)
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.init_db") as mock_init, \
             patch("keisei.training.katago_loop.read_training_state", return_value=None), \
             patch("keisei.training.katago_loop.write_training_state"), \
             patch("keisei.training.katago_loop.DDP", side_effect=lambda m, **kw: m), \
             patch("keisei.training.katago_loop.dist.barrier"), \
             patch("keisei.training.katago_loop.dist.broadcast_object_list"):
            _loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
            mock_init.assert_not_called()

    def test_main_rank_calls_db_init(self):
        """Main rank should call init_db normally."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        config = _make_config()
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.init_db") as mock_init, \
             patch("keisei.training.katago_loop.read_training_state", return_value=None), \
             patch("keisei.training.katago_loop.write_training_state"):
            _loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
            mock_init.assert_called_once()


# ---------------------------------------------------------------------------
# PendingTransitions unit tests
# ---------------------------------------------------------------------------


class TestPendingTransitionsCreate:
    """Test PendingTransitions.create() — zero prior tests for PPO rollout accumulation."""

    def test_pending_transitions_create_shapes(self):
        """Construct with known num_envs and obs shapes; verify field dimensions."""
        from keisei.training.katago_loop import PendingTransitions

        num_envs = 4
        obs_shape = (50, 9, 9)
        action_space = 11259
        device = torch.device("cpu")

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        assert pt.obs.shape == (num_envs, *obs_shape)
        assert pt.actions.shape == (num_envs,)
        assert pt.log_probs.shape == (num_envs,)
        assert pt.values.shape == (num_envs,)
        assert pt.legal_masks.shape == (num_envs, action_space)
        assert pt.rewards.shape == (num_envs,)
        assert pt.score_targets.shape == (num_envs,)
        assert pt.valid.shape == (num_envs,)
        assert not pt.valid.any(), "No envs should be valid initially"

    def test_create_sets_valid_and_stores_data(self):
        """After create(), masked envs should have valid=True and correct data."""
        from keisei.training.katago_loop import PendingTransitions

        num_envs = 4
        obs_shape = (2, 3, 3)
        action_space = 10
        device = torch.device("cpu")

        pt = PendingTransitions(num_envs, obs_shape, action_space, device)

        # Create transitions for envs 0 and 2
        env_mask = torch.tensor([True, False, True, False])
        obs = torch.randn(num_envs, *obs_shape)
        actions = torch.arange(num_envs, dtype=torch.long)
        log_probs = torch.randn(num_envs)
        values = torch.randn(num_envs)
        legal_masks = torch.ones(num_envs, action_space, dtype=torch.bool)
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        score_targets = torch.tensor([0.1, 0.2, 0.3, 0.4])

        pt.create(env_mask, obs, actions, log_probs, values, legal_masks, rewards, score_targets)

        assert pt.valid[0] and pt.valid[2]
        assert not pt.valid[1] and not pt.valid[3]
        assert torch.allclose(pt.obs[0], obs[0])
        assert torch.allclose(pt.obs[2], obs[2])
        assert pt.actions[0] == actions[0]
        assert pt.actions[2] == actions[2]

    def test_create_double_open_raises(self):
        """Calling create() on an already-valid env should raise RuntimeError."""
        from keisei.training.katago_loop import PendingTransitions

        num_envs = 2
        pt = PendingTransitions(num_envs, (2,), 5, torch.device("cpu"))

        env_mask = torch.tensor([True, False])
        obs = torch.randn(num_envs, 2)
        actions = torch.zeros(num_envs, dtype=torch.long)
        log_probs = torch.zeros(num_envs)
        values = torch.zeros(num_envs)
        legal_masks = torch.ones(num_envs, 5, dtype=torch.bool)
        rewards = torch.zeros(num_envs)
        score_targets = torch.zeros(num_envs)

        pt.create(env_mask, obs, actions, log_probs, values, legal_masks, rewards, score_targets)

        with pytest.raises(RuntimeError, match="already-valid"):
            pt.create(env_mask, obs, actions, log_probs, values, legal_masks, rewards, score_targets)


class TestPendingTransitionsAccumulateReward:
    """Test PendingTransitions.accumulate_reward()."""

    def test_accumulate_reward_adds_correctly(self):
        """Call accumulate_reward multiple times; verify rewards sum per env."""
        from keisei.training.katago_loop import PendingTransitions

        num_envs = 3
        pt = PendingTransitions(num_envs, (2,), 5, torch.device("cpu"))

        # Open transitions for envs 0 and 1
        env_mask = torch.tensor([True, True, False])
        obs = torch.randn(num_envs, 2)
        actions = torch.zeros(num_envs, dtype=torch.long)
        log_probs = torch.zeros(num_envs)
        values = torch.zeros(num_envs)
        legal_masks = torch.ones(num_envs, 5, dtype=torch.bool)
        rewards = torch.tensor([1.0, 2.0, 0.0])  # initial rewards
        score_targets = torch.zeros(num_envs)

        pt.create(env_mask, obs, actions, log_probs, values, legal_masks, rewards, score_targets)

        # Accumulate additional rewards
        pt.accumulate_reward(torch.tensor([0.5, 0.3, 99.0]))  # env 2 ignored (not valid)
        pt.accumulate_reward(torch.tensor([0.1, 0.2, 88.0]))

        assert torch.isclose(pt.rewards[0], torch.tensor(1.6))  # 1.0 + 0.5 + 0.1
        assert torch.isclose(pt.rewards[1], torch.tensor(2.5))  # 2.0 + 0.3 + 0.2
        assert pt.rewards[2] == 0.0  # env 2 was never valid, reward stays at 0


class TestPendingTransitionsFinalize:
    """Test PendingTransitions.finalize()."""

    def test_finalize_output_shapes(self):
        """Create, accumulate, finalize -> verify output tensor shapes."""
        from keisei.training.katago_loop import PendingTransitions

        num_envs = 4
        obs_shape = (2, 3)
        action_space = 7
        pt = PendingTransitions(num_envs, obs_shape, action_space, torch.device("cpu"))

        # Open transitions for envs 0, 1, 3
        env_mask = torch.tensor([True, True, False, True])
        obs = torch.randn(num_envs, *obs_shape)
        actions = torch.zeros(num_envs, dtype=torch.long)
        log_probs = torch.randn(num_envs)
        values = torch.randn(num_envs)
        legal_masks = torch.ones(num_envs, action_space, dtype=torch.bool)
        rewards = torch.zeros(num_envs)
        score_targets = torch.zeros(num_envs)

        pt.create(env_mask, obs, actions, log_probs, values, legal_masks, rewards, score_targets)
        pt.accumulate_reward(torch.tensor([0.5, -0.5, 0.0, 1.0]))

        # Finalize envs 0 and 1 (env 3 stays open)
        finalize_mask = torch.tensor([True, True, False, False])
        dones = torch.tensor([1.0, 0.0, 0.0, 0.0])
        terminated = torch.tensor([1.0, 0.0, 0.0, 0.0])

        result = pt.finalize(finalize_mask, dones, terminated)
        assert result is not None

        # Should finalize 2 envs (0 and 1)
        assert result["obs"].shape == (2, *obs_shape)
        assert result["actions"].shape == (2,)
        assert result["log_probs"].shape == (2,)
        assert result["values"].shape == (2,)
        assert result["rewards"].shape == (2,)
        assert result["dones"].shape == (2,)
        assert result["terminated"].shape == (2,)
        assert result["legal_masks"].shape == (2, action_space)
        assert result["score_targets"].shape == (2,)
        assert result["env_ids"].shape == (2,)

        # Finalized envs should be cleared
        assert not pt.valid[0]
        assert not pt.valid[1]
        # Env 3 should still be valid
        assert pt.valid[3]

    def test_finalize_returns_none_when_nothing_to_finalize(self):
        """Finalize with no valid envs returns None."""
        from keisei.training.katago_loop import PendingTransitions

        pt = PendingTransitions(2, (2,), 5, torch.device("cpu"))

        finalize_mask = torch.tensor([True, True])
        dones = torch.zeros(2)
        terminated = torch.zeros(2)

        result = pt.finalize(finalize_mask, dones, terminated)
        assert result is None


# ---------------------------------------------------------------------------
# Pure function tests from katago_loop.py
# ---------------------------------------------------------------------------


class TestComputeValueCats:
    """Test _compute_value_cats — value-head category assignment."""

    def test_non_terminal_all_ignore(self):
        """All-False terminal_mask -> all cats == -1 (ignore label)."""
        from keisei.training.katago_loop import _compute_value_cats

        rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
        terminal_mask = torch.zeros(4, dtype=torch.bool)
        device = torch.device("cpu")

        cats = _compute_value_cats(rewards, terminal_mask, device)
        assert (cats == -1).all(), f"Expected all -1 for non-terminal, got {cats}"

    def test_terminal_win_draw_loss(self):
        """Terminal positions: positive=0(win), zero=1(draw), negative=2(loss)."""
        from keisei.training.katago_loop import _compute_value_cats

        rewards = torch.tensor([1.0, 0.0, -1.0])
        terminal_mask = torch.ones(3, dtype=torch.bool)
        device = torch.device("cpu")

        cats = _compute_value_cats(rewards, terminal_mask, device)
        assert cats[0].item() == 0, "Win (reward > 0) should be category 0"
        assert cats[1].item() == 1, "Draw (reward == 0) should be category 1"
        assert cats[2].item() == 2, "Loss (reward < 0) should be category 2"

    def test_mixed_terminal_and_non_terminal(self):
        """Mix of terminal and non-terminal positions."""
        from keisei.training.katago_loop import _compute_value_cats

        rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
        terminal_mask = torch.tensor([True, False, True, False])
        device = torch.device("cpu")

        cats = _compute_value_cats(rewards, terminal_mask, device)
        assert cats[0].item() == 0   # terminal win
        assert cats[1].item() == -1  # non-terminal (ignore)
        assert cats[2].item() == 2   # terminal loss
        assert cats[3].item() == -1  # non-terminal (ignore)


class TestToLearnerPerspective:
    """Test to_learner_perspective — reward sign correction."""

    def test_learner_moved_no_flip(self):
        """When learner moved (pre_players == learner_side), reward unchanged."""
        from keisei.training.katago_loop import to_learner_perspective

        rewards = torch.tensor([1.0, -0.5])
        pre_players = np.array([0, 0], dtype=np.uint8)
        learner_side = 0

        result = to_learner_perspective(rewards, pre_players, learner_side)
        assert torch.allclose(result, rewards)

    def test_opponent_moved_flip(self):
        """When opponent moved (pre_players != learner_side), reward is negated."""
        from keisei.training.katago_loop import to_learner_perspective

        rewards = torch.tensor([1.0, -0.5])
        pre_players = np.array([1, 1], dtype=np.uint8)
        learner_side = 0

        result = to_learner_perspective(rewards, pre_players, learner_side)
        expected = torch.tensor([-1.0, 0.5])
        assert torch.allclose(result, expected)

    def test_mixed_perspective(self):
        """Mixed: learner on side 1, some envs learner-moved, some opponent-moved."""
        from keisei.training.katago_loop import to_learner_perspective

        rewards = torch.tensor([1.0, -1.0, 0.5])
        pre_players = np.array([1, 0, 1], dtype=np.uint8)
        learner_side = 1

        result = to_learner_perspective(rewards, pre_players, learner_side)
        # env 0: pre_player=1 == learner_side=1 -> no flip -> 1.0
        # env 1: pre_player=0 != learner_side=1 -> flip -> 1.0
        # env 2: pre_player=1 == learner_side=1 -> no flip -> 0.5
        expected = torch.tensor([1.0, 1.0, 0.5])
        assert torch.allclose(result, expected)


class TestSignCorrectBootstrap:
    """Test sign_correct_bootstrap — value sign correction for GAE."""

    def test_learner_to_move_no_flip(self):
        """When learner is to-move, bootstrap value is already correct."""
        from keisei.training.katago_loop import sign_correct_bootstrap

        next_values = torch.tensor([0.8, -0.3])
        current_players = np.array([0, 0], dtype=np.uint8)
        learner_side = 0

        result = sign_correct_bootstrap(next_values, current_players, learner_side)
        assert torch.allclose(result, next_values)

    def test_opponent_to_move_negated(self):
        """When opponent is to-move, bootstrap value must be negated."""
        from keisei.training.katago_loop import sign_correct_bootstrap

        next_values = torch.tensor([0.8, -0.3])
        current_players = np.array([1, 1], dtype=np.uint8)
        learner_side = 0

        result = sign_correct_bootstrap(next_values, current_players, learner_side)
        expected = torch.tensor([-0.8, 0.3])
        assert torch.allclose(result, expected)

    def test_mixed_learner_and_opponent(self):
        """Known learner_mask and opponent_mask, verify negation pattern."""
        from keisei.training.katago_loop import sign_correct_bootstrap

        next_values = torch.tensor([1.0, 2.0, -3.0, 4.0])
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        learner_side = 0

        result = sign_correct_bootstrap(next_values, current_players, learner_side)
        # env 0: current=0 == learner -> no flip -> 1.0
        # env 1: current=1 != learner -> flip -> -2.0
        # env 2: current=0 == learner -> no flip -> -3.0
        # env 3: current=1 != learner -> flip -> -4.0
        expected = torch.tensor([1.0, -2.0, -3.0, -4.0])
        assert torch.allclose(result, expected)


class TestMainEntryPoint:
    def test_main_calls_setup_and_cleanup(self):
        """main() should call setup_distributed and cleanup_distributed."""
        with patch("keisei.training.katago_loop.get_distributed_context") as mock_ctx, \
             patch("keisei.training.katago_loop.setup_distributed") as mock_setup, \
             patch("keisei.training.katago_loop.cleanup_distributed") as mock_cleanup, \
             patch("keisei.training.katago_loop.seed_all_ranks") as mock_seed, \
             patch("keisei.training.katago_loop.load_config"), \
             patch("keisei.training.katago_loop.KataGoTrainingLoop") as mock_loop, \
             patch("sys.argv", ["prog", "dummy.toml", "--epochs", "1"]):
            mock_ctx.return_value = DistributedContext(
                rank=0, local_rank=0, world_size=1, is_distributed=False,
            )
            mock_loop.return_value.run = MagicMock()

            from keisei.training.katago_loop import main
            main()

            mock_setup.assert_called_once()
            mock_cleanup.assert_called_once()
            mock_seed.assert_called_once_with(42)  # base_seed(42) + rank(0)


class TestDDPLeagueGuard:
    def test_league_with_ddp_raises(self):
        """League mode is not yet supported with DDP."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=2, is_distributed=True)
        config = _make_config()
        config = dataclasses.replace(config, league=LeagueConfig())
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        with pytest.raises(ValueError, match="League mode.*not.*supported.*DDP"):
            KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)

    def test_league_without_ddp_ok(self):
        """League mode works fine without DDP."""
        ctx = DistributedContext(rank=0, local_rank=0, world_size=1, is_distributed=False)
        config = _make_config()
        config = dataclasses.replace(config, league=LeagueConfig())
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env, dist_ctx=ctx)
        assert loop.store is not None
