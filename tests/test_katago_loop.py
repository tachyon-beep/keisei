# tests/test_katago_loop.py
"""Integration tests for KataGoTrainingLoop."""

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


@pytest.fixture
def katago_config(tmp_path):
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


class TestKataGoTrainingLoopInit:
    def test_initialization(self, katago_config):
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        assert loop.num_envs == 2

    def test_model_is_se_resnet(self, katago_config):
        from keisei.training.models.se_resnet import SEResNetModel

        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        base = loop.model.module if hasattr(loop.model, "module") else loop.model
        assert isinstance(base, SEResNetModel)

    def test_db_training_state_written(self, katago_config):
        """Verify init writes training state to DB."""
        from keisei.db import read_training_state

        mock_env = _make_mock_katago_vecenv(num_envs=2)
        KataGoTrainingLoop(katago_config, vecenv=mock_env)
        state = read_training_state(katago_config.display.db_path)
        assert state is not None

    def test_bad_architecture_algorithm_raises(self, katago_config):
        """katago_ppo with non-KataGo architecture should raise ValueError."""
        bad_config = dataclasses.replace(
            katago_config,
            model=dataclasses.replace(katago_config.model,
                                       architecture="resnet",
                                       params={"hidden_size": 16, "num_layers": 1}),
        )
        with pytest.raises(ValueError, match="requires a KataGoBaseModel"):
            KataGoTrainingLoop(bad_config, vecenv=_make_mock_katago_vecenv())


class TestKataGoTrainingLoopRun:
    def test_run_one_epoch(self, katago_config):
        """Run one epoch of 4 steps — should complete without error."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        # epoch is 0-indexed: after 1 epoch (range 0..0), epoch remains 0
        assert loop.epoch == 0
        assert loop.global_step == 4

    def test_run_with_terminal_episodes(self, katago_config):
        """Verify training completes when episodes terminate mid-epoch.

        This exercises the value categorization (W/D/L) and score target
        branches which are only active for terminal steps.
        """
        mock_env = _make_mock_katago_vecenv(num_envs=2, terminate_at_step=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4

    def test_run_with_nonzero_material(self, katago_config):
        """Exercise the normalization path with non-zero material balance."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, material_balance=38)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4  # 38/76 = 0.5, well within guard

    def test_metrics_written_to_db(self, katago_config):
        """Verify metrics are persisted after each epoch."""
        from keisei.db import read_metrics_since

        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=2, steps_per_epoch=4)
        metrics = read_metrics_since(katago_config.display.db_path, since_id=0)
        assert len(metrics) == 2  # one row per epoch


def _with_league(config, tmp_path, snapshot_interval=10):
    """Helper to add league config to an existing AppConfig."""
    league = LeagueConfig(
        max_pool_size=10,
        snapshot_interval=snapshot_interval,
        epochs_per_seat=50,
        elo_floor=500,
    )
    return dataclasses.replace(config, league=league)


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
        assert not loop._current_opponent.training

    def test_run_with_split_merge_active(self, katago_config, tmp_path):
        """Run 1 epoch with alternate_players=True — exercises split-merge buffer path."""
        config = _with_league(katago_config, tmp_path, snapshot_interval=50)
        mock_env = _make_mock_katago_vecenv(
            num_envs=2, alternate_players=True, terminate_at_step=3,
        )
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step > 0


class TestSplitMergeIntegration:
    def test_run_with_league_completes(self, katago_config, tmp_path):
        """With league enabled, run() should complete without error."""
        mock_env = _make_mock_katago_vecenv(num_envs=4)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=5)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4

    def test_run_with_alternating_players(self, katago_config, tmp_path):
        """W2: exercise the split path with alternating current_players."""
        mock_env = _make_mock_katago_vecenv(num_envs=4, alternate_players=True)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=5)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4

    def test_buffer_stores_learner_only(self, katago_config, tmp_path):
        """W3: with split-merge, buffer should contain fewer samples than total steps * envs."""
        mock_env = _make_mock_katago_vecenv(num_envs=4, alternate_players=True)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=50)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        # Don't run — manually step to inspect buffer
        # With alternating players, ~half the envs are learner each step
        from keisei.training.katago_loop import split_merge_step
        reset_result = loop.vecenv.reset()
        obs = __import__("torch").from_numpy(np.asarray(reset_result.observations)).to(loop.device)
        legal_masks = __import__("torch").from_numpy(np.asarray(reset_result.legal_masks)).to(loop.device)
        current_players = np.array([0, 1, 0, 1], dtype=np.uint8)
        sm = split_merge_step(
            obs=obs, legal_masks=legal_masks,
            current_players=current_players,
            learner_model=loop.model,
            opponent_model=loop.model,  # self-play for test
            learner_side=0,
        )
        # Only 2 of 4 envs are learner
        assert sm.learner_indices.numel() == 2
        assert sm.learner_log_probs.shape == (2,)

    def test_run_without_league_still_works(self, katago_config):
        """Without league config, run() should work as before."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4


def _make_league_config(epochs_per_seat=1, snapshot_interval=1):
    """Create a LeagueConfig with given params (no tmp_path needed)."""
    return LeagueConfig(
        max_pool_size=5,
        snapshot_interval=snapshot_interval,
        epochs_per_seat=epochs_per_seat,
    )


class TestSeatRotation:
    def test_rotation_after_epochs_per_seat(self, katago_config, tmp_path):
        """After epochs_per_seat, optimizer should be reset."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=2)
        league = dataclasses.replace(katago_config.league, epochs_per_seat=3)
        katago_config = dataclasses.replace(katago_config, league=league)

        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=4, steps_per_epoch=2)
        assert loop.epoch == 3  # completed epochs 0,1,2,3

    def test_optimizer_replaced_after_rotation(self, katago_config):
        """Seat rotation must create a new optimizer (old momentum discarded)."""
        config = dataclasses.replace(katago_config, league=_make_league_config(epochs_per_seat=1))
        mock_env = _make_mock_katago_vecenv(num_envs=2, terminate_at_step=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        original_optimizer_id = id(loop.ppo.optimizer)
        loop._rotate_seat(epoch=0)
        assert id(loop.ppo.optimizer) != original_optimizer_id

    def test_new_optimizer_references_current_params(self, katago_config):
        """New optimizer must reference the current model's parameters."""
        config = dataclasses.replace(katago_config, league=_make_league_config(epochs_per_seat=1))
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        loop._rotate_seat(epoch=0)

        opt_params = set()
        for group in loop.ppo.optimizer.param_groups:
            for p in group["params"]:
                opt_params.add(id(p))
        model_params = {id(p) for p in loop.ppo.model.parameters()}
        assert opt_params == model_params

    def test_learner_entry_id_updated(self, katago_config):
        """B5 fix: _learner_entry_id should change after rotation."""
        config = dataclasses.replace(katago_config, league=_make_league_config(epochs_per_seat=1))
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        original_id = loop._learner_entry_id
        loop._rotate_seat(epoch=0)
        assert loop._learner_entry_id != original_id

    def test_warmup_epochs_extended(self, katago_config):
        """B2 fix: warmup_epochs = epoch + 1 + original_warmup_duration."""
        config = dataclasses.replace(katago_config, league=_make_league_config(epochs_per_seat=1))
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(
                config.training,
                algorithm_params={
                    **config.training.algorithm_params,
                    "rl_warmup": {"epochs": 3, "entropy_bonus": 0.05},
                },
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        assert loop._original_warmup_duration == 3
        loop._rotate_seat(epoch=5)
        # warmup_epochs = 5 + 1 + 3 = 9
        assert loop.ppo.warmup_epochs == 9

    def test_lr_scheduler_reconnected(self, katago_config):
        """B1 fix: LR scheduler should point at the new optimizer."""
        config = dataclasses.replace(katago_config, league=_make_league_config(epochs_per_seat=1))
        config = dataclasses.replace(
            config,
            training=dataclasses.replace(
                config.training,
                algorithm_params={
                    **config.training.algorithm_params,
                    "lr_schedule": {"type": "plateau", "factor": 0.5, "patience": 10},
                },
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        assert loop.lr_scheduler is not None
        old_scheduler_id = id(loop.lr_scheduler)
        loop._rotate_seat(epoch=0)
        assert id(loop.lr_scheduler) != old_scheduler_id
        # Scheduler's optimizer should be the new one
        assert loop.lr_scheduler.optimizer is loop.ppo.optimizer


class TestCheckResume:
    """C1: Resume-from-checkpoint branch in _check_resume()."""

    def test_resume_restores_epoch_and_step(self, katago_config, tmp_path):
        """When a checkpoint exists in the DB, _check_resume loads it and
        restores epoch/global_step instead of starting from 0."""
        from keisei.training.checkpoint import save_checkpoint

        mock_env = _make_mock_katago_vecenv(num_envs=2)

        # First: create a loop to initialize the DB and get a model we can save
        loop1 = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        base_model = loop1.model.module if hasattr(loop1.model, "module") else loop1.model

        # Save a checkpoint at epoch=7, step=42
        ckpt_path = tmp_path / "checkpoints" / "resume_test.pt"
        save_checkpoint(
            ckpt_path, base_model, loop1.ppo.optimizer,
            epoch=7, step=42,
            architecture=katago_config.model.architecture,
        )

        # Write checkpoint_path into the DB so _check_resume finds it
        from keisei.db import update_training_progress
        update_training_progress(
            katago_config.display.db_path, epoch=7, step=42,
            checkpoint_path=str(ckpt_path),
        )

        # Now create a second loop — it should resume from the checkpoint
        mock_env2 = _make_mock_katago_vecenv(num_envs=2)
        loop2 = KataGoTrainingLoop(katago_config, vecenv=mock_env2)
        assert loop2.epoch == 7
        assert loop2.global_step == 42

    def test_no_resume_when_checkpoint_missing_on_disk(self, katago_config, tmp_path):
        """If the DB points to a checkpoint file that doesn't exist on disk,
        _check_resume should NOT crash — it falls through to fresh start."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)

        # Create a loop to initialize the DB
        KataGoTrainingLoop(katago_config, vecenv=mock_env)

        # Point DB at a non-existent file
        from keisei.db import update_training_progress
        update_training_progress(
            katago_config.display.db_path, epoch=5, step=20,
            checkpoint_path=str(tmp_path / "nonexistent.pt"),
        )

        # Second loop should start fresh (epoch=0, step=0)
        mock_env2 = _make_mock_katago_vecenv(num_envs=2)
        loop2 = KataGoTrainingLoop(katago_config, vecenv=mock_env2)
        assert loop2.epoch == 0
        assert loop2.global_step == 0

    def test_resume_sl_mode_skips_optimizer(self, katago_config, tmp_path):
        """When resume_mode='sl', _check_resume should call load_checkpoint with skip_optimizer=True."""
        from keisei.training.checkpoint import save_checkpoint

        mock_env = _make_mock_katago_vecenv(num_envs=2)

        # First: create a loop to initialize the DB and get a model we can save
        loop1 = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        base_model = loop1.model.module if hasattr(loop1.model, "module") else loop1.model

        # Save a checkpoint at epoch=7, step=42
        ckpt_path = tmp_path / "checkpoints" / "sl_resume_test.pt"
        save_checkpoint(
            ckpt_path, base_model, loop1.ppo.optimizer,
            epoch=7, step=42,
            architecture=katago_config.model.architecture,
        )

        # Write checkpoint_path into the DB so _check_resume finds it
        from keisei.db import update_training_progress
        update_training_progress(
            katago_config.display.db_path, epoch=7, step=42,
            checkpoint_path=str(ckpt_path),
        )

        # Now create a loop with resume_mode="sl" and verify skip_optimizer is passed
        mock_env2 = _make_mock_katago_vecenv(num_envs=2)
        with patch("keisei.training.katago_loop.load_checkpoint") as mock_load:
            mock_load.return_value = {"epoch": 7, "step": 42}
            loop2 = KataGoTrainingLoop(katago_config, vecenv=mock_env2, resume_mode="sl")

            # Verify load_checkpoint was called with skip_optimizer=True
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("skip_optimizer") is True


class TestRotateSeatIsolation:
    """C2: _rotate_seat() isolation tests."""

    def _make_loop_with_league(self, katago_config, tmp_path):
        """Helper: create a loop with league enabled and LR scheduler."""
        import torch

        # Add lr_schedule to algorithm_params so LR scheduler is created
        algo_params = dict(katago_config.training.algorithm_params)
        algo_params["lr_schedule"] = {
            "type": "plateau",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-6,
        }
        algo_params["rl_warmup"] = {"epochs": 5, "entropy_bonus": 0.05}
        training = dataclasses.replace(katago_config.training, algorithm_params=algo_params)
        config = dataclasses.replace(katago_config, training=training)

        config = _with_league(config, tmp_path, snapshot_interval=10)
        league = dataclasses.replace(config.league, epochs_per_seat=3)
        config = dataclasses.replace(config, league=league)

        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        return loop

    def test_learner_entry_id_updated(self, katago_config, tmp_path):
        """After _rotate_seat, _learner_entry_id points to the new snapshot."""
        loop = self._make_loop_with_league(katago_config, tmp_path)
        old_id = loop._learner_entry_id
        loop._rotate_seat(epoch=2)
        assert loop._learner_entry_id != old_id
        assert loop._learner_entry_id is not None

    def test_optimizer_is_new_object(self, katago_config, tmp_path):
        """After _rotate_seat, the optimizer is a fresh instance."""
        loop = self._make_loop_with_league(katago_config, tmp_path)
        old_optimizer = loop.ppo.optimizer
        loop._rotate_seat(epoch=2)
        assert loop.ppo.optimizer is not old_optimizer

    def test_warmup_extends_correctly(self, katago_config, tmp_path):
        """warmup_epochs = (epoch + 1) + _original_warmup_duration after rotation."""
        loop = self._make_loop_with_league(katago_config, tmp_path)
        original_duration = loop._original_warmup_duration
        assert original_duration == 5  # from rl_warmup config

        loop._rotate_seat(epoch=9)
        # warmup_epochs should be (9 + 1) + 5 = 15
        assert loop.ppo.warmup_epochs == 10 + original_duration

        # Second rotation: still uses _original_ duration, not accumulated
        loop._rotate_seat(epoch=19)
        assert loop.ppo.warmup_epochs == 20 + original_duration

    def test_lr_scheduler_references_new_optimizer(self, katago_config, tmp_path):
        """After rotation, LR scheduler should reference the new optimizer."""
        loop = self._make_loop_with_league(katago_config, tmp_path)
        assert loop.lr_scheduler is not None

        old_scheduler = loop.lr_scheduler
        loop._rotate_seat(epoch=2)

        # The scheduler should be a new object
        assert loop.lr_scheduler is not old_scheduler
        # The new scheduler's optimizer should be the new optimizer
        assert loop.lr_scheduler.optimizer is loop.ppo.optimizer


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

    def test_obs_channel_mismatch_raises(self, katago_config):
        """VecEnv with wrong observation_channels must raise ValueError."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        mock_env.observation_channels = 42  # config expects 50
        with pytest.raises(ValueError, match="observation"):
            KataGoTrainingLoop(katago_config, vecenv=mock_env)

    def test_action_space_mismatch_raises(self, katago_config):
        """VecEnv with wrong action_space_size must raise ValueError."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        mock_env.action_space_size = 9999  # expected 11259
        with pytest.raises(ValueError, match="action space"):
            KataGoTrainingLoop(katago_config, vecenv=mock_env)


class TestMaybeUpdateHeartbeat:
    """C2: _maybe_update_heartbeat() time guard."""

    def test_heartbeat_fires_after_10_seconds(self, katago_config):
        """When >= 10s have elapsed, heartbeat should update the DB."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)

        with patch("keisei.training.katago_loop.update_training_progress") as mock_update:
            # Simulate 11 seconds elapsed
            loop._last_heartbeat = time.monotonic() - 11.0
            old_heartbeat = loop._last_heartbeat
            loop._maybe_update_heartbeat()

            mock_update.assert_called_once()
            # _last_heartbeat should have been refreshed
            assert loop._last_heartbeat > old_heartbeat

    def test_heartbeat_skipped_within_10_seconds(self, katago_config):
        """When < 10s have elapsed, heartbeat should NOT fire."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)

        with patch("keisei.training.katago_loop.update_training_progress") as mock_update:
            # Just set heartbeat to now — well within the 10s window
            loop._last_heartbeat = time.monotonic()
            loop._maybe_update_heartbeat()

            mock_update.assert_not_called()


class TestValueCategoryNoLeague:
    """C1: Value category assignment in the no-league (no opponent) path."""

    def test_value_cats_win_draw_loss(self, katago_config):
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
            katago_config,
            training=dataclasses.replace(katago_config.training, num_games=num_envs),
        )

        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        # Intercept buffer.add to capture value_cats
        captured_value_cats = []
        original_add = loop.buffer.add

        def spy_add(*args, **kwargs):
            # value_cats is the 8th positional arg (index 7)
            captured_value_cats.append(args[7].clone())
            return original_add(*args, **kwargs)

        loop.buffer.add = spy_add

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


class TestCheckpointWrittenToDisk:
    """CRIT-1: run() with checkpoint_interval that triggers a .pt file write."""

    def test_checkpoint_file_created_on_disk(self, katago_config, tmp_path):
        """Run enough epochs to trigger checkpoint_interval, assert .pt file exists."""
        # katago_config has checkpoint_interval=5, so epoch_i=4 triggers
        # (epoch_i + 1) % 5 == 0
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=5, steps_per_epoch=2)

        ckpt_dir = tmp_path / "checkpoints"
        expected = ckpt_dir / "epoch_00004.pt"
        assert expected.exists(), f"Checkpoint file not found at {expected}"

        # Verify the checkpoint is loadable and contains expected keys
        ckpt = torch.load(expected, weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert ckpt["epoch"] == 5
        assert ckpt["step"] == 10  # 5 epochs * 2 steps

    def test_checkpoint_path_recorded_in_db(self, katago_config, tmp_path):
        """After checkpoint write, the path should be stored in training_state."""
        from keisei.db import read_training_state

        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=5, steps_per_epoch=2)

        state = read_training_state(katago_config.display.db_path)
        assert state is not None
        assert state.get("checkpoint_path") is not None
        assert "epoch_00004.pt" in state["checkpoint_path"]


class TestSwallowedExceptions:
    """CRIT-1: Swallowed exceptions in run() must not crash training."""

    def test_write_metrics_failure_continues_training(self, katago_config):
        """If write_metrics raises, training should continue to the next epoch."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)

        with patch("keisei.training.katago_loop.write_metrics",
                   side_effect=RuntimeError("DB write failed")):
            # Should NOT raise — the exception is caught and logged
            loop.run(num_epochs=2, steps_per_epoch=2)

        assert loop.global_step == 4  # both epochs completed

    def test_update_training_progress_failure_continues(self, katago_config):
        """If update_training_progress raises mid-epoch, training continues."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)

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

    def test_checkpoint_save_failure_continues(self, katago_config, tmp_path):
        """If save_checkpoint raises, training should continue past the checkpoint epoch."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)

        with patch("keisei.training.katago_loop.save_checkpoint",
                   side_effect=OSError("disk full")):
            # Run 6 epochs (checkpoint at epoch 4, then continues to 5)
            loop.run(num_epochs=6, steps_per_epoch=2)

        assert loop.global_step == 12  # all 6 epochs completed
        # No checkpoint file should exist since save was mocked to fail
        ckpt_dir = tmp_path / "checkpoints"
        assert not (ckpt_dir / "epoch_00004.pt").exists()


class TestRotateSeat:
    """T6: Verify _rotate_seat() resets optimizer and recreates LR scheduler."""

    @pytest.fixture
    def league_config(self, tmp_path):
        return AppConfig(
            training=TrainingConfig(
                num_games=2,
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
                    "lr_schedule": {
                        "type": "plateau",
                        "factor": 0.5,
                        "patience": 50,
                        "min_lr": 1e-5,
                    },
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
            ),
        )

    def test_rotate_seat_resets_optimizer(self, league_config):
        """After rotation, optimizer should be fresh (no momentum buffers)."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        # Run a few steps to populate optimizer state
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert len(loop.ppo.optimizer.state) > 0, "Optimizer should have state after training"

        # Perform seat rotation
        old_optimizer_id = id(loop.ppo.optimizer)
        loop._rotate_seat(epoch=1)

        # Optimizer should be a new instance with empty state
        assert id(loop.ppo.optimizer) != old_optimizer_id, "Should be a new optimizer instance"
        assert len(loop.ppo.optimizer.state) == 0, "Fresh optimizer should have no state"

    def test_rotate_seat_recreates_lr_scheduler(self, league_config):
        """After rotation, LR scheduler should point at the new optimizer."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        old_scheduler = loop.lr_scheduler
        assert old_scheduler is not None, "League config should create LR scheduler"

        loop._rotate_seat(epoch=1)

        assert loop.lr_scheduler is not old_scheduler, "Should be a new scheduler"
        # Verify the new scheduler is connected to the new optimizer
        assert loop.lr_scheduler.optimizer is loop.ppo.optimizer

    def test_rotate_seat_extends_warmup(self, league_config):
        """Warmup should be extended relative to the rotation epoch."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)
        original_warmup = loop.ppo.warmup_epochs

        loop._rotate_seat(epoch=10)
        # warmup should be epoch+1 + original_duration
        assert loop.ppo.warmup_epochs == 11 + loop._original_warmup_duration

    def test_rotate_seat_updates_learner_entry(self, league_config):
        """Rotation should create a new pool entry and update the learner ID."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        old_id = loop._learner_entry_id
        loop._rotate_seat(epoch=5)
        assert loop._learner_entry_id != old_id, "Learner entry ID should change after rotation"


class TestSLToRLCheckpointHandoff:
    """T7: Verify SL checkpoint loads correctly into RL training."""

    def test_sl_checkpoint_loads_model_weights(self, katago_config, tmp_path):
        """SL-trained weights should survive the handoff to RL."""
        from keisei.training.checkpoint import save_checkpoint
        from keisei.training.models.se_resnet import SEResNetModel, SEResNetParams

        # Create and "train" an SL model
        params = SEResNetParams(
            num_blocks=2, channels=32, se_reduction=8,
            global_pool_channels=16, policy_channels=8,
            value_fc_size=32, score_fc_size=16, obs_channels=50,
        )
        sl_model = SEResNetModel(params)
        sl_optimizer = torch.optim.Adam(sl_model.parameters(), lr=1e-3)

        # Modify weights so they're not default
        obs = torch.randn(2, 50, 9, 9)
        output = sl_model(obs)
        loss = output.policy_logits.sum() + output.value_logits.sum()
        loss.backward()
        sl_optimizer.step()

        sl_model.eval()
        with torch.no_grad():
            sl_output = sl_model(obs)

        # Save SL checkpoint
        ckpt_path = tmp_path / "checkpoints" / "sl_checkpoint.pt"
        save_checkpoint(ckpt_path, sl_model, sl_optimizer, epoch=1, step=0,
                        architecture="se_resnet")

        # Write training state so the RL loop finds the checkpoint
        from keisei.db import init_db, write_training_state
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        write_training_state(db_path, {
            "config_json": "{}",
            "display_name": "SL-Test",
            "model_arch": "se_resnet",
            "algorithm_name": "sl",
            "started_at": "2026-01-01T00:00:00Z",
            "current_epoch": 1,
            "current_step": 0,
            "checkpoint_path": str(ckpt_path),
        })

        # Create RL loop — it should resume from the SL checkpoint
        config = dataclasses.replace(katago_config,
            display=dataclasses.replace(katago_config.display, db_path=db_path))
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)

        # Verify all model parameters match (compare on CPU)
        for name, sl_param in sl_model.named_parameters():
            rl_param = dict(loop._base_model.named_parameters())[name]
            assert torch.allclose(sl_param, rl_param.cpu(), atol=1e-7), \
                f"Parameter mismatch for '{name}' after SL→RL handoff"

        # Also verify buffers (BatchNorm running_mean/var)
        for name, sl_buf in sl_model.named_buffers():
            rl_buf = dict(loop._base_model.named_buffers())[name]
            assert torch.allclose(sl_buf, rl_buf.cpu(), atol=1e-7), \
                f"Buffer mismatch for '{name}' after SL→RL handoff"

        # Epoch and step should be restored
        assert loop.epoch == 1
        assert loop.global_step == 0


class TestCheckpointResumeRoundTrip:
    """Test save/resume round-trip through the training loop orchestrator."""

    def test_resume_restores_epoch_and_step(self, katago_config):
        """Run 2 epochs with checkpoint_interval=1, reconstruct, verify resume."""
        config = dataclasses.replace(
            katago_config,
            training=dataclasses.replace(
                katago_config.training, checkpoint_interval=1,
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        loop.run(num_epochs=2, steps_per_epoch=4)

        saved_step = loop.global_step
        # run() stores self.epoch = epoch_i (0-indexed); the checkpoint and DB
        # record epoch_i + 1.  After 2 epochs epoch_i reaches 1, so the
        # checkpoint carries epoch=2 and that is what _check_resume restores.
        expected_epoch_on_resume = loop.epoch + 1
        assert expected_epoch_on_resume > 0

        # Verify checkpoint file exists
        from pathlib import Path
        ckpt_dir = Path(config.training.checkpoint_dir)
        ckpt_files = list(ckpt_dir.glob("epoch_*.pt"))
        assert len(ckpt_files) >= 1

        # Reconstruct with the same config and DB — should resume
        mock_env2 = _make_mock_katago_vecenv(num_envs=2)
        loop2 = KataGoTrainingLoop(config, vecenv=mock_env2)
        assert loop2.epoch == expected_epoch_on_resume
        assert loop2.global_step == saved_step

    def test_resume_model_weights_match(self, katago_config):
        """Resumed model should have the same weights as the checkpoint."""
        config = dataclasses.replace(
            katago_config,
            training=dataclasses.replace(
                katago_config.training, checkpoint_interval=1,
            ),
        )
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)

        # Snapshot model weights after training
        original_weights = {
            name: p.clone() for name, p in loop._base_model.named_parameters()
        }

        # Reconstruct — should load checkpoint
        mock_env2 = _make_mock_katago_vecenv(num_envs=2)
        loop2 = KataGoTrainingLoop(config, vecenv=mock_env2)

        for name, p in loop2._base_model.named_parameters():
            torch.testing.assert_close(
                p, original_weights[name],
                msg=f"Weight mismatch after resume: {name}",
            )

    def test_resume_architecture_mismatch_raises(self, katago_config):
        """Resuming from a checkpoint whose architecture tag doesn't match
        the config should raise ValueError via load_checkpoint."""
        config = dataclasses.replace(
            katago_config,
            training=dataclasses.replace(
                katago_config.training, checkpoint_interval=1,
            ),
        )

        # Run 1 epoch so a checkpoint is saved (architecture="se_resnet")
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        loop = KataGoTrainingLoop(config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=2)

        # Tamper with the checkpoint file: overwrite its architecture tag
        # so it no longer matches the config's "se_resnet".
        from pathlib import Path
        ckpt_dir = Path(config.training.checkpoint_dir)
        ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
        assert len(ckpt_files) >= 1
        ckpt = torch.load(ckpt_files[-1], weights_only=False)
        ckpt["architecture"] = "tampered_arch"
        torch.save(ckpt, ckpt_files[-1])

        # Constructing a new loop triggers _check_resume → load_checkpoint,
        # which should detect the architecture mismatch and raise ValueError.
        mock_env2 = _make_mock_katago_vecenv(num_envs=2)
        with pytest.raises(ValueError, match="architecture mismatch"):
            KataGoTrainingLoop(config, vecenv=mock_env2)
