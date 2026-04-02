# tests/test_katago_loop.py
"""Integration tests for KataGoTrainingLoop."""

import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest

from keisei.config import AppConfig, DisplayConfig, LeagueConfig, ModelConfig, TrainingConfig
from keisei.training.katago_loop import KataGoTrainingLoop


def _make_mock_katago_vecenv(
    num_envs: int = 2, *, terminate_at_step: int | None = None,
    alternate_players: bool = False,
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
        result.step_metadata.material_balance = np.zeros(num_envs, dtype=np.int32)

        if terminate_at_step is not None and step_count[0] == terminate_at_step:
            result.terminated[0] = True
            result.rewards[0] = 1.0

        return result

    mock.reset.side_effect = lambda: make_reset_result()
    mock.step.side_effect = make_step_result
    mock.reset_stats = MagicMock()
    return mock


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
