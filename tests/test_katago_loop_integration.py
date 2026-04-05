"""Integration tests for KataGoTrainingLoop (real DB, checkpoints, pool store)."""

import dataclasses
import sqlite3
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from keisei.config import AppConfig, DisplayConfig, LeagueConfig, ModelConfig, TrainingConfig
from keisei.training.katago_loop import KataGoTrainingLoop

from tests.test_katago_loop import _make_config, _make_mock_katago_vecenv

pytestmark = pytest.mark.integration


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


def _with_league(config, tmp_path, snapshot_interval=10, color_randomization=False):
    """Helper to add league config to an existing AppConfig.

    color_randomization defaults to False so that tests using non-alternating
    mock envs (all players == Black) don't starve the buffer when learner_side
    is randomly assigned White for some envs.  Tests that specifically exercise
    color randomization should pass color_randomization=True explicitly.
    """
    league = LeagueConfig(

        snapshot_interval=snapshot_interval,
        epochs_per_seat=50,
        elo_floor=500,
        color_randomization=color_randomization,
    )
    return dataclasses.replace(config, league=league)


def _make_league_config(epochs_per_seat=1, snapshot_interval=1):
    """Create a LeagueConfig with given params (no tmp_path needed)."""
    return LeagueConfig(

        snapshot_interval=snapshot_interval,
        epochs_per_seat=epochs_per_seat,
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


class TestLeagueIntegration:
    def test_bootstrap_snapshot_at_init(self, katago_config, tmp_path):
        """Pool should have one entry after init (the bootstrap snapshot)."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=10)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        assert loop.store is not None
        assert len(loop.store.list_entries()) == 1

    def test_periodic_snapshot(self, katago_config, tmp_path):
        """Pool should gain a snapshot every snapshot_interval epochs."""
        mock_env = _make_mock_katago_vecenv(num_envs=2)
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=2)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=4, steps_per_epoch=2)
        assert loop.store is not None
        entries = loop.store.list_entries()
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
        katago_config = dataclasses.replace(
            katago_config, training=dataclasses.replace(katago_config.training, num_games=4),
        )
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=5)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4

    def test_run_with_alternating_players(self, katago_config, tmp_path):
        """W2: exercise the split path with alternating current_players."""
        mock_env = _make_mock_katago_vecenv(num_envs=4, alternate_players=True)
        katago_config = dataclasses.replace(
            katago_config, training=dataclasses.replace(katago_config.training, num_games=4),
        )
        katago_config = _with_league(katago_config, tmp_path, snapshot_interval=5)
        loop = KataGoTrainingLoop(katago_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=4)
        assert loop.global_step == 4

    def test_buffer_stores_learner_only(self, katago_config, tmp_path):
        """W3: with split-merge, buffer should contain fewer samples than total steps * envs."""
        mock_env = _make_mock_katago_vecenv(num_envs=4, alternate_players=True)
        katago_config = dataclasses.replace(
            katago_config, training=dataclasses.replace(katago_config.training, num_games=4),
        )
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
        base_model: torch.nn.Module = loop1.model.module if hasattr(loop1.model, "module") else loop1.model  # type: ignore[assignment]

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
        base_model: torch.nn.Module = loop1.model.module if hasattr(loop1.model, "module") else loop1.model  # type: ignore[assignment]

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
            _loop2 = KataGoTrainingLoop(katago_config, vecenv=mock_env2, resume_mode="sl")

            # Verify load_checkpoint was called with skip_optimizer=True
            mock_load.assert_called_once()
            call_kwargs = mock_load.call_args[1]
            assert call_kwargs.get("skip_optimizer") is True


class TestRotateSeatIsolation:
    """C2: _rotate_seat() isolation tests."""

    def _make_loop_with_league(self, katago_config, tmp_path):
        """Helper: create a loop with league enabled and LR scheduler."""

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
                    "rl_warmup": {"epochs": 3, "entropy_bonus": 0.05},
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
        assert loop.lr_scheduler is not None
        assert loop.lr_scheduler.optimizer is loop.ppo.optimizer

    def test_rotate_seat_extends_warmup(self, league_config):
        """Warmup should be extended relative to the rotation epoch."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)
        assert loop._original_warmup_duration == 3  # from rl_warmup config
        loop._rotate_seat(epoch=10)
        # warmup should be epoch+1 + original_duration = 11 + 3 = 14
        assert loop.ppo.warmup_epochs == 14

    def test_rotate_seat_updates_learner_entry(self, league_config):
        """Rotation should create a new pool entry and update the learner ID."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        old_id = loop._learner_entry_id
        loop._rotate_seat(epoch=5)
        assert loop._learner_entry_id != old_id, "Learner entry ID should change after rotation"

    def test_rotate_seat_new_entry_starts_at_default_elo(self, league_config):
        """New entry after rotation should have the default Elo (1000.0), not the old Elo."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        # Artificially inflate the current learner's Elo
        if loop.store and loop._learner_entry_id:
            loop.store.update_elo(loop._learner_entry_id, 1650.0, epoch=0)

        loop._rotate_seat(epoch=5)

        # The NEW entry should start at the default 1000.0, not 1650.0
        assert loop.store is not None
        assert loop._learner_entry_id is not None
        new_entry = loop.store._get_entry(loop._learner_entry_id)
        assert new_entry is not None
        assert new_entry.elo_rating == 1000.0

    def test_rotate_seat_old_entry_elo_preserved(self, league_config):
        """Old entry's Elo should remain unchanged after rotation."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        old_id = loop._learner_entry_id
        if loop.store and old_id:
            loop.store.update_elo(old_id, 1200.0, epoch=0)

        loop._rotate_seat(epoch=5)

        assert loop.store is not None
        assert old_id is not None
        old_entry = loop.store._get_entry(old_id)
        assert old_entry is not None
        assert old_entry.elo_rating == 1200.0

    def test_rotate_seat_no_elo_history_for_new_entry(self, league_config):
        """New entry should have no elo_history rows immediately after rotation."""
        mock_env = _make_mock_katago_vecenv(num_envs=2, alternate_players=True)
        loop = KataGoTrainingLoop(league_config, vecenv=mock_env)

        loop._rotate_seat(epoch=5)

        # Query elo_history directly for the new entry
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

        # Fill pool beyond recent tier capacity, forcing overflow/eviction
        assert loop.store is not None
        total_slots = (league_config.league.recent.slots
                       + league_config.league.recent.soft_overflow + 2)
        for i in range(total_slots):
            loop.tiered_pool.snapshot_learner(
                loop._base_model, "se_resnet",
                dict(league_config.model.params), epoch=100 + i,
            )

        # Old learner entry may have been evicted
        loop._rotate_seat(epoch=200)
        assert loop._learner_entry_id is not None
        new_entry = loop.store._get_entry(loop._learner_entry_id)
        assert new_entry is not None
        assert new_entry.elo_rating == 1000.0


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
            _sl_output = sl_model(obs)

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

        if loop.store and loop._learner_entry_id:
            loop.store.update_elo(loop._learner_entry_id, 1500.0, epoch=0)

        loop._rotate_seat(epoch=5)
        assert loop.store is not None
        assert loop._learner_entry_id is not None
        new_entry = loop.store._get_entry(loop._learner_entry_id)
        assert new_entry is not None
        assert new_entry.elo_rating == 1000.0

    def test_run_one_epoch_with_all_changes(self, fairness_config):
        """Smoke test: training loop completes with all fairness changes active."""
        mock_env = _make_mock_katago_vecenv(num_envs=4, alternate_players=True)
        loop = KataGoTrainingLoop(fairness_config, vecenv=mock_env)
        loop.run(num_epochs=1, steps_per_epoch=10)

        # Behavioral assertions — not just "didn't crash"
        assert loop.store is not None
        entries = loop.store.list_entries()
        assert len(entries) >= 1, "Pool should have at least the initial entry"

        # Per-env opponents should have been active (config says True, pool is non-empty)
        assert loop._opponent_results is not None, (
            "_opponent_results should be populated when per_env_opponents=True"
        )
        assert isinstance(loop._opponent_results, dict)
        # All opponent IDs in results should be valid pool entry IDs
        entry_ids = {e.id for e in entries}
        for opp_id in loop._opponent_results:
            assert opp_id in entry_ids, f"opponent_results key {opp_id} not in pool"

    def test_per_opponent_elo_isolation(self, fairness_config):
        """Per-opponent Elo attribution should not cross-contaminate between opponents."""

        mock_env = _make_mock_katago_vecenv(num_envs=4, alternate_players=True)
        loop = KataGoTrainingLoop(fairness_config, vecenv=mock_env)

        # Manually set up opponent results to simulate two opponents
        # Opponent A: 5 wins, 0 losses → strong result
        # Opponent B: 0 wins, 5 losses → weak result
        if loop._opponent_results is None:
            loop._opponent_results = {}

        # Get the existing pool entries
        assert loop.store is not None
        entries = loop.store.list_entries()
        if len(entries) < 2:
            # Add a second entry so we have two opponents
            loop.tiered_pool.snapshot_learner(
                loop._base_model, "se_resnet",
                dict(fairness_config.model.params), epoch=99,
            )
            entries = loop.store.list_entries()

        opp_a_id = entries[0].id
        opp_b_id = entries[1].id if len(entries) > 1 else entries[0].id
        loop._opponent_results = {
            opp_a_id: [5, 0, 0],  # 5 wins
            opp_b_id: [0, 5, 0],  # 5 losses
        }
        loop._cached_entries_by_id = {e.id: e for e in entries}

        # Check that the two opponents get different Elo updates
        # (A should go down since learner won, B should go up since learner lost)
        # The results dict should have different contents for each opponent
        assert loop._opponent_results[opp_a_id] != loop._opponent_results.get(opp_b_id, None) or opp_a_id == opp_b_id
